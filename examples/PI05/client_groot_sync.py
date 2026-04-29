#!/usr/bin/env python3
"""ROS2 GR00T 客户端 — 同步执行版 (执行完再推理)
=======================================================

与 ``client_groot.py`` (RTC 异步版) 行为相反, 走最朴素的 chunk 同步执行:

    while True:
        ① 拍下当前 state + image
        ② 调服务端 get_action(obs) 阻塞等推理 (~700ms)
        ③ 把整个 chunk 顺序按 control_hz 执行完
        ④ 回到 ①

推理期间机器人保持上一帧指令 (停顿可见), 但 *绝不会发生 chunk 边界跳变*,
配置一行行直观, 不依赖任何 RTC 调参 (没有 s / blend / d)。

适用场景:
  - 服务端推理延迟 ≥ 500ms, RTC 已经无 blend room (硬切跳变明显)
  - 想要可预测、易复现的执行轨迹 (调试/录制 demo)
  - 任务本身节奏慢, 不需要细粒度反应性

不适用场景:
  - 需要快速反应的 manipulation (推理期间机器人停顿会破坏闭环)
  - 服务端延迟 < 200ms 时 RTC 通常更平滑

本文件 **完全自包含**, 机器人侧只需 ``pyzmq + msgpack + numpy + rclpy``,
不必安装 gr00t 包及其重型依赖 (torch / transformers / flash-attn / 等)。

观测/动作 schema 与 ``client_groot.py`` 完全一致 (见该文件头说明)。
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
from pathlib import Path
import threading
import time
from typing import Any

import msgpack
import numpy as np
from pnd_adam.msg import HandCmd, HandState, LowCmd, LowState, MotorCmd
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
import zmq


# ═══════════════════════════════════════════════════════════════════
#  内联 ZMQ + msgpack 协议  (与 PolicyServer 双向兼容)
# ═══════════════════════════════════════════════════════════════════


def _ndarray_pack(obj: object) -> object:
    """msgpack default 钩子: numpy ndarray → .npy bytes。"""
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    raise TypeError(f"Unsupported type: {type(obj)}")


def _ndarray_unpack(obj: dict) -> object:
    """msgpack object_hook: 还原 ndarray; 其它字段透传。"""
    if not isinstance(obj, dict):
        return obj
    if obj.get("__ndarray_class__"):
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


class GrootZmqClient:
    """轻量级 ZeroMQ REQ 客户端, 仅实现 ping / get_action 两个端点。"""

    def __init__(
        self,
        host: str,
        port: int,
        timeout_ms: int = 30000,
        api_token: str | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self.context = zmq.Context.instance()
        self._init_socket()

    def _init_socket(self) -> None:
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def _call(self, endpoint: str, data: dict | None = None) -> object:
        request: dict = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token
        try:
            self.socket.send(msgpack.packb(request, default=_ndarray_pack))
            message = self.socket.recv()
        except zmq.error.Again:
            self._init_socket()
            raise
        if message == b"ERROR":
            raise RuntimeError("Server error.")
        response = msgpack.unpackb(message, object_hook=_ndarray_unpack)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def ping(self) -> dict:
        return self._call("ping")  # type: ignore[return-value]

    def get_action(self, observation: dict, options: dict | None = None) -> tuple[dict, dict]:
        response = self._call("get_action", {"observation": observation, "options": options})
        return tuple(response)  # type: ignore[return-value]

    def close(self) -> None:
        try:
            self.socket.close(linger=0)
        except Exception:  # pragma: no cover
            pass


# ┌──────────────────────────────────────────────────────────────────┐
# │                        维度常量                                  │
# └──────────────────────────────────────────────────────────────────┘
LOWSTATE_DIM = 19
HANDSTATE_RAW_DIM = 12
HANDSTATE_ENCODED_DIM = 2
STATE_DIM = LOWSTATE_DIM + HANDSTATE_ENCODED_DIM  # 21
ACTION_ARM_DIM = LOWSTATE_DIM
ACTION_HAND_DIM = HANDSTATE_ENCODED_DIM
ACTION_DIM = ACTION_ARM_DIM + ACTION_HAND_DIM  # 21
HAND_CMD_DIM = HANDSTATE_RAW_DIM
LEFT_HAND_SLICE = slice(0, 6)
RIGHT_HAND_SLICE = slice(6, 12)

# ┌──────────────────────────────────────────────────────────────────┐
# │                    默认参数                                      │
# └──────────────────────────────────────────────────────────────────┘
DEFAULT_CAMERA_TOPIC = "/zed/zed_node/rgb/color/rect/image"
DEFAULT_LOWSTATE_TOPIC = "lowstate"
DEFAULT_HANDSTATE_TOPIC = "handstate"
DEFAULT_LOWCMD_TOPIC = "lowcmd"
DEFAULT_HANDCMD_TOPIC = "handcmd"
DEFAULT_HOST = "192.168.31.116"
DEFAULT_PORT = 5555
DEFAULT_CONTROL_HZ = 15.0
DEFAULT_ACTION_HORIZON = 32  # 服务端 PI05 输出 32 步
DEFAULT_RENDER_SIZE = 0  # 0 = 不缩放
DEFAULT_INTERPOLATION_FACTOR = 10
DEFAULT_MAX_ARM_VELOCITY = 3.0  # rad/s, 输出层兜底
DEFAULT_PROMPT = "fold the white T-shirt"

VIDEO_KEY = "zed_rgb"
STATE_KEY = "joints"
ACTION_KEY = "joints"
LANGUAGE_KEY = "annotation.human.task_description"

# ┌──────────────────────────────────────────────────────────────────┐
# │                  19 个关节的 PD 增益配置                         │
# └──────────────────────────────────────────────────────────────────┘
KP = [
    1837.991943359375, 1837.991943359375, 1837.991943359375,
    260.7959899902344, 260.7959899902344,
    294.0791931152344, 294.0791931152344,
    312.9552001953125, 312.9552001953125, 312.9552001953125,
    312.9552001953125, 312.9552001953125,
    294.0791931152344, 294.0791931152344,
    312.9552001953125, 312.9552001953125, 312.9552001953125,
    312.9552001953125, 312.9552001953125,
]  # fmt: skip
KD = [
    30.63319969177246, 30.63319969177246, 30.63319969177246,
    5.2159199714660645, 5.2159199714660645,
    9.802639961242676, 9.802639961242676,
    10.431839942932129, 10.431839942932129, 10.431839942932129,
    10.431839942932129, 10.431839942932129,
    9.802639961242676, 9.802639961242676,
    10.431839942932129, 10.431839942932129, 10.431839942932129,
    10.431839942932129, 10.431839942932129,
]  # fmt: skip


def _default_json_log_path() -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return str(Path.cwd() / "logs" / f"groot_sync_trace_{timestamp}.jsonl")


def _letterbox_resize(image: np.ndarray, size: int) -> np.ndarray:
    """保持宽高比的 letterbox 缩放: 居中填零到 size×size, 纯 numpy 实现。"""
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    scale = min(size / h, size / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    ys = (np.arange(new_h) * h / new_h).astype(np.int64)
    xs = (np.arange(new_w) * w / new_w).astype(np.int64)
    resized = image[ys[:, None], xs[None, :]]
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


# ═══════════════════════════════════════════════════════════════════
#  主 ROS 2 节点
# ═══════════════════════════════════════════════════════════════════


class GrootAdamUSyncClient(Node):
    """ROS2 节点: 同步执行版 (推理 → 执行整个 chunk → 推理)。

    线程架构:
      ┌─────────────┐    ┌──────────────────────────────┐
      │ ROS 回调线程 │───▶│  执行线程 (推理 + 整 chunk 执行) │
      │ (传感器更新) │    │  推理期间机器人保持上一帧指令     │
      └─────────────┘    └──────────────────────────────┘
    """

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("groot_adam_u_sync_client")
        self._args = args
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # 传感器缓冲
        self._latest_low_state: np.ndarray | None = None
        self._latest_hand_state: np.ndarray | None = None
        self._latest_state: np.ndarray | None = None
        self._latest_image: np.ndarray | None = None
        self._latest_state_stamp: float | None = None
        self._latest_image_stamp: float | None = None

        # 服务器连接
        self._policy_client: GrootZmqClient | None = None
        self._server_metadata: dict[str, Any] | None = None

        # 时间参数
        if args.control_hz <= 0.0:
            raise ValueError("control_hz 必须为正数。")
        self._control_dt = 1.0 / args.control_hz

        # 插值
        self._interpolation_factor = max(1, int(args.interpolation_factor))
        self._servo_dt = self._control_dt / self._interpolation_factor

        # JSON 日志
        self._json_log_lock = threading.Lock()
        self._json_logging_enabled = True
        self._json_log_fp = self._open_json_log_file(args.json_log_path)
        self._last_wait_log_time = 0.0

        # Debug 模式
        self._debug = getattr(args, "debug", False)

        # 首次日志标志
        self._has_logged_first_image = False
        self._has_logged_first_state = False
        self._has_logged_first_hand_state = False
        self._has_logged_first_action_apply = False

        # 推理延迟 EMA (诊断用)
        self._infer_latency_ema = 0.0

        # 全局 control tick 计数器
        self._control_tick = 0
        # 全局 chunk 序号 (每次完整推理 +1)
        self._chunk_index = 0

        # ── 输出层 slew-rate limiter (兜底防跳变) ─────────────────
        self._max_arm_velocity = float(args.max_arm_velocity)
        self._max_arm_delta_per_tick = (
            self._max_arm_velocity * self._servo_dt if self._max_arm_velocity > 0.0 else 0.0
        )
        self._last_published_arm: np.ndarray | None = None
        self._slew_clamp_count = 0
        self._slew_total_count = 0
        self._slew_window_clamps = 0
        self._slew_window_total = 0
        self._last_slew_log_time = 0.0
        self._has_logged_first_slew_clamp = False

        # ROS pubsub
        self._lowcmd_pub = self.create_publisher(LowCmd, args.lowcmd_topic, 10)
        self._handcmd_pub = self.create_publisher(HandCmd, args.handcmd_topic, 10)

        self.create_subscription(LowState, args.lowstate_topic, self._on_lowstate, 10)
        self.create_subscription(HandState, args.handstate_topic, self._on_handstate, 10)
        self.create_subscription(Image, args.camera_topic, self._on_image, qos_profile_sensor_data)

        self._write_json_event(
            "session_start",
            mode="sync",
            model="gr00t-n1d7-pi05",
            host=args.host,
            port=args.port,
            prompt=args.prompt,
            control_hz=args.control_hz,
            action_horizon=args.action_horizon,
            interpolation_factor=args.interpolation_factor,
            render_size=args.render_size,
            max_arm_velocity=self._max_arm_velocity,
            max_arm_delta_per_tick=self._max_arm_delta_per_tick,
        )

        self._connect_thread: threading.Thread | None = None
        self._execute_thread: threading.Thread | None = None
        self._start_connect_thread()
        self._start_execute_thread()

        slew_str = (
            f"slew={self._max_arm_velocity:.2f} rad/s "
            f"(±{self._max_arm_delta_per_tick * 1000:.2f} mrad/tick)"
            if self._max_arm_delta_per_tick > 0.0
            else "slew=disabled"
        )
        self.get_logger().info(
            f"GR00T 同步客户端已启动  H={args.action_horizon}  "
            f"control_hz={args.control_hz}  "
            f"interp={self._interpolation_factor}x → "
            f"servo_hz={args.control_hz * self._interpolation_factor:.0f}  "
            f"chunk_duration={args.action_horizon * self._control_dt * 1000:.0f}ms  "
            f"render_size={args.render_size}  {slew_str}"
        )

    # ─────────────────────────────────────────────────────────────
    #  生命周期
    # ─────────────────────────────────────────────────────────────

    def destroy_node(self) -> None:
        self._stop_event.set()
        for t in (self._connect_thread, self._execute_thread):
            if t is not None and t.is_alive() and t is not threading.current_thread():
                t.join(timeout=1.0)
        self._write_json_event("session_end", reason="destroy_node")
        self._close_json_log_file()
        super().destroy_node()

    # ─────────────────────────────────────────────────────────────
    #  JSON 行日志
    # ─────────────────────────────────────────────────────────────

    def _open_json_log_file(self, path_str: str) -> Any:
        path = Path(path_str).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.open("a", encoding="utf-8")

    def _close_json_log_file(self) -> None:
        with self._json_log_lock:
            self._json_logging_enabled = False
            if self._json_log_fp and not self._json_log_fp.closed:
                self._json_log_fp.flush()
                self._json_log_fp.close()

    def _to_jsonable(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): self._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_jsonable(v) for v in value]
        return value

    def _write_json_event(self, event_type: str, **payload: Any) -> None:
        with self._json_log_lock:
            if not self._json_logging_enabled:
                return
            event = {
                "event": event_type,
                "wall_time": time.time(),
                "monotonic_time": time.perf_counter(),
            }
            event.update({k: self._to_jsonable(v) for k, v in payload.items()})
            self._json_log_fp.write(json.dumps(event, ensure_ascii=True) + "\n")
            self._json_log_fp.flush()

    # ─────────────────────────────────────────────────────────────
    #  服务器连接 (后台重连)
    # ─────────────────────────────────────────────────────────────

    def _start_connect_thread(self) -> None:
        if self._connect_thread is not None and self._connect_thread.is_alive():
            return
        self._connect_thread = threading.Thread(
            target=self._connect_to_server, name="groot-sync-connect", daemon=True
        )
        self._connect_thread.start()

    def _connect_to_server(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.get_logger().info(
                    f"正在连接 GR00T 服务器 {self._args.host}:{self._args.port}..."
                )
                client = GrootZmqClient(
                    host=self._args.host,
                    port=self._args.port,
                    timeout_ms=self._args.timeout_ms,
                    api_token=self._args.api_token,
                )
                pong = client.ping()
                with self._lock:
                    self._policy_client = client
                    self._server_metadata = {"ping": pong}
                self.get_logger().info(f"已连接 GR00T 服务器: {pong}")
                return
            except Exception as exc:
                self.get_logger().warning(f"连接失败: {exc}。5 秒后重试。")
                if self._stop_event.wait(5.0):
                    return

    # ─────────────────────────────────────────────────────────────
    #  执行线程启动
    # ─────────────────────────────────────────────────────────────

    def _start_execute_thread(self) -> None:
        if self._execute_thread is None or not self._execute_thread.is_alive():
            self._execute_thread = threading.Thread(
                target=self._execute_loop, name="groot-sync-execute", daemon=True
            )
            self._execute_thread.start()

    # ─────────────────────────────────────────────────────────────
    #  ROS 传感器回调
    # ─────────────────────────────────────────────────────────────

    def _on_lowstate(self, msg: LowState) -> None:
        low_state = np.zeros((LOWSTATE_DIM,), dtype=np.float32)
        motor_states = list(msg.motor_state)
        motor_count = min(len(motor_states), LOWSTATE_DIM)
        for idx in range(motor_count):
            low_state[idx] = float(motor_states[idx].q)

        with self._lock:
            self._latest_low_state = low_state
            self._latest_state = self._build_policy_state()
            self._latest_state_stamp = time.time()
        if not self._has_logged_first_state:
            self._has_logged_first_state = True
            self.get_logger().info(f"首次收到 lowstate: {motor_count} 个电机。")

    def _on_handstate(self, msg: HandState) -> None:
        hand_positions = np.asarray(msg.position, dtype=np.float32)
        if hand_positions.shape[0] != HANDSTATE_RAW_DIM:
            padded = np.zeros((HANDSTATE_RAW_DIM,), dtype=np.float32)
            padded[: min(hand_positions.shape[0], HANDSTATE_RAW_DIM)] = hand_positions[
                :HANDSTATE_RAW_DIM
            ]
            hand_positions = padded

        with self._lock:
            self._latest_hand_state = self._encode_hand_state(hand_positions)
            self._latest_state = self._build_policy_state()
            self._latest_state_stamp = time.time()
        if not self._has_logged_first_hand_state:
            self._has_logged_first_hand_state = True
            self.get_logger().info("首次收到 handstate, 已编码为 2 维。")

    def _on_image(self, msg: Image) -> None:
        try:
            image = self._decode_ros_image(msg)
            image = self._prepare_image(image)
        except Exception as exc:
            self.get_logger().warning(f"图像解码失败: {exc}")
            return

        with self._lock:
            self._latest_image = image
            self._latest_image_stamp = time.time()
        if not self._has_logged_first_image:
            self._has_logged_first_image = True
            self.get_logger().info(
                f"首次收到图像: {msg.encoding} {msg.width}×{msg.height} → {image.shape}"
            )

    # ─────────────────────────────────────────────────────────────
    #  手部编/解码 + 状态拼接
    # ─────────────────────────────────────────────────────────────

    def _encode_hand_state(self, hand_positions: np.ndarray) -> np.ndarray:
        left_avg = np.mean(hand_positions[LEFT_HAND_SLICE][:5]) / 1800.0
        right_avg = np.mean(hand_positions[RIGHT_HAND_SLICE][:5]) / 1800.0
        return np.array(
            [
                1.0 if np.clip(left_avg, 0.0, 1.0) > 0.5 else 0.0,
                1.0 if np.clip(right_avg, 0.0, 1.0) > 0.5 else 0.0,
            ],
            dtype=np.float32,
        )

    def _build_policy_state(self) -> np.ndarray | None:
        if self._latest_low_state is None or self._latest_hand_state is None:
            return None
        return np.concatenate([self._latest_low_state, self._latest_hand_state], axis=0).astype(
            np.float32, copy=False
        )

    def _decode_hand_action(self, hand_action: np.ndarray) -> np.ndarray:
        hand_cmd = np.zeros((HAND_CMD_DIM,), dtype=np.uint32)
        left_cmd = 1000 if float(np.clip(1.0 - hand_action[0], 0.0, 1.0)) > 0.5 else 500
        right_cmd = 1000 if float(np.clip(1.0 - hand_action[1], 0.0, 1.0)) > 0.5 else 500
        hand_cmd[LEFT_HAND_SLICE][:4] = left_cmd
        hand_cmd[RIGHT_HAND_SLICE][:4] = right_cmd
        hand_cmd[LEFT_HAND_SLICE][4] = (
            1000 if float(np.clip(1.0 - hand_action[0], 0.0, 1.0)) > 0.5 else 200
        )
        hand_cmd[RIGHT_HAND_SLICE][4] = (
            1000 if float(np.clip(1.0 - hand_action[1], 0.0, 1.0)) > 0.5 else 200
        )
        hand_cmd[5] = 0
        hand_cmd[11] = 0
        return hand_cmd

    # ─────────────────────────────────────────────────────────────
    #  图像处理
    # ─────────────────────────────────────────────────────────────

    def _decode_ros_image(self, msg: Image) -> np.ndarray:
        encoding = msg.encoding.lower()
        channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}.get(encoding)
        if channels is None:
            raise ValueError(f"不支持的图像编码: {msg.encoding}")
        row_stride = int(msg.step)
        expected_width = int(msg.width) * channels
        flat = np.frombuffer(msg.data, dtype=np.uint8)
        image = flat.reshape((int(msg.height), row_stride))
        image = image[:, :expected_width].reshape((int(msg.height), int(msg.width), channels))
        if encoding == "bgr8":
            image = image[:, :, ::-1]
        elif encoding == "bgra8":
            image = image[:, :, [2, 1, 0, 3]]
        if channels == 1:
            image = np.repeat(image, 3, axis=2)
        elif channels == 4:
            image = image[:, :, :3]
        return np.ascontiguousarray(image)

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        if self._args.render_size and self._args.render_size > 0:
            image = _letterbox_resize(image, int(self._args.render_size))
        image = np.rot90(image, 2)  # 相机倒装
        return np.ascontiguousarray(image, dtype=np.uint8)

    # ─────────────────────────────────────────────────────────────
    #  输出层兜底: 关节滑率限幅
    # ─────────────────────────────────────────────────────────────

    def _slew_limit_arm(self, target_arm: np.ndarray) -> np.ndarray:
        if self._max_arm_delta_per_tick <= 0.0:
            return target_arm

        if self._last_published_arm is None:
            with self._lock:
                anchor = (
                    self._latest_low_state.copy() if self._latest_low_state is not None else None
                )
            self._last_published_arm = (
                anchor.astype(np.float32, copy=False)
                if anchor is not None
                else target_arm.astype(np.float32, copy=True)
            )

        delta = target_arm.astype(np.float32, copy=False) - self._last_published_arm
        max_d = self._max_arm_delta_per_tick
        clamped_delta = np.clip(delta, -max_d, max_d)

        was_clamped = bool(np.any(np.abs(delta) > max_d + 1e-9))
        self._slew_total_count += 1
        self._slew_window_total += 1
        if was_clamped:
            self._slew_clamp_count += 1
            self._slew_window_clamps += 1
            if not self._has_logged_first_slew_clamp:
                self._has_logged_first_slew_clamp = True
                self.get_logger().warning(
                    f"slew limiter 首次触发: max|Δ|={float(np.max(np.abs(delta))):.4f} rad "
                    f"→ clipped to ±{max_d:.4f} rad/tick (tick={self._control_tick})"
                )

        now = time.time()
        if now - self._last_slew_log_time >= 5.0 and self._slew_window_total > 0:
            rate = 100.0 * self._slew_window_clamps / self._slew_window_total
            if rate > 0.5:
                self.get_logger().warning(
                    f"slew limiter active: {rate:.1f}% "
                    f"({self._slew_window_clamps}/{self._slew_window_total} ticks last 5s)"
                )
            self._slew_window_clamps = 0
            self._slew_window_total = 0
            self._last_slew_log_time = now

        new_arm = (self._last_published_arm + clamped_delta).astype(np.float32)
        self._last_published_arm = new_arm
        return new_arm

    # ─────────────────────────────────────────────────────────────
    #  动作执行 (与 RTC 客户端共享)
    # ─────────────────────────────────────────────────────────────

    def _apply_action(self, action: np.ndarray) -> None:
        if action.shape[0] < ACTION_DIM:
            raise ValueError(f"动作维度 {action.shape[0]} < {ACTION_DIM}")

        arm_action = np.asarray(action[:ACTION_ARM_DIM], dtype=np.float32).copy()
        hand_action = action[ACTION_ARM_DIM:ACTION_DIM]

        # 硬安全限位 (先于 slew, 让 slew 跟踪真实发布值)
        if arm_action[1] > 0.5:
            arm_action[1] = 0.5
        arm_action[3] = 0.0
        arm_action[4] = 45.0 / 180.0 * math.pi

        # 输出层 slew 兜底
        arm_action = self._slew_limit_arm(arm_action)

        lowcmd = LowCmd()
        lowcmd.motor_cmd = [MotorCmd() for _ in range(ACTION_ARM_DIM)]
        for idx, target_q in enumerate(arm_action):
            mc = lowcmd.motor_cmd[idx]
            mc.mode = 1
            mc.q = float(target_q)
            mc.dq = 0.0
            mc.tau = 0.0
            mc.kp = KP[idx]
            mc.kd = KD[idx]
            mc.ki = 0.0

        hand_positions = self._decode_hand_action(hand_action)
        handcmd = HandCmd()
        for idx in range(HAND_CMD_DIM):
            handcmd.position[idx] = int(hand_positions[idx])

        if not self._debug:
            self._lowcmd_pub.publish(lowcmd)
            self._handcmd_pub.publish(handcmd)

        if not self._has_logged_first_action_apply:
            self._has_logged_first_action_apply = True
            label = "[DEBUG] 首个 sync 动作 (未发布)" if self._debug else "已执行首个 sync 动作。"
            self.get_logger().info(label)

    # ─────────────────────────────────────────────────────────────
    #  工具函数
    # ─────────────────────────────────────────────────────────────

    def _sleep_until(self, deadline: float) -> None:
        remaining = deadline - time.perf_counter()
        if remaining > 0.0:
            self._stop_event.wait(remaining)

    def _maybe_log_waiting(self, client: Any, state: Any, image: Any) -> None:
        now = time.time()
        if now - self._last_wait_log_time <= 5.0:
            return
        missing = []
        if client is None:
            missing.append("server")
        if state is None:
            missing.append("state")
        if image is None:
            missing.append("image")
        if missing:
            self.get_logger().info(f"等待中: {', '.join(missing)}")
            self._last_wait_log_time = now

    def _build_observation(self, state: np.ndarray, image: np.ndarray) -> dict[str, Any]:
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"image 必须为 (H, W, 3), got {image.shape}")
        if state.ndim != 1 or state.shape[0] < STATE_DIM:
            raise ValueError(f"state 必须 ≥ {STATE_DIM}, got {state.shape}")
        video_arr = np.ascontiguousarray(image, dtype=np.uint8)[None, None]
        state_arr = state[:STATE_DIM].astype(np.float32, copy=False)[None, None]
        return {
            "video": {VIDEO_KEY: video_arr},
            "state": {STATE_KEY: state_arr},
            "language": {LANGUAGE_KEY: [[self._args.prompt]]},
        }

    # ═════════════════════════════════════════════════════════════
    #  执行线程: 推理 → 整 chunk 顺序执行 → 推理
    # ═════════════════════════════════════════════════════════════
    #
    #  推理期间机器人保持上一次发布的指令 (PD 控制下静止), 不会跳变。
    #  整个 chunk 完整执行完才触发下一次推理, 因此每两次推理之间机器人
    #  会切实走完 H × Δt = 32 × 67ms ≈ 2.13s 的轨迹, 然后停顿 ~推理延迟。
    #
    #  由 chunk 之间没有重叠或融合, 切换瞬间靠插值 + slew limit 平滑过渡。

    def _execute_loop(self) -> None:
        prev_action: np.ndarray | None = None  # 上一 chunk 末帧, 用于跨 chunk 插值

        while not self._stop_event.is_set():
            # ── ① 等传感器 + 服务器就绪 ──
            client, state, image = self._snapshot_obs_blocking()
            if client is None or state is None or image is None:
                continue  # 被停止信号唤醒, 下一轮检查 _stop_event 退出

            # ── ② 推理 (阻塞) ──
            try:
                t0 = time.perf_counter()
                obs = self._build_observation(state, image)
                action_dict, _info = client.get_action(obs)
                elapsed = time.perf_counter() - t0
            except Exception as exc:
                self.get_logger().warning(f"推理失败: {exc}")
                with self._lock:
                    self._policy_client = None
                    self._server_metadata = None
                self._start_connect_thread()
                continue

            # ── ③ 解析 chunk ──
            try:
                if not isinstance(action_dict, dict) or ACTION_KEY not in action_dict:
                    raise ValueError(f"服务器返回缺少键 '{ACTION_KEY}'")
                actions = np.asarray(action_dict[ACTION_KEY], dtype=np.float32)
                if actions.ndim == 3:
                    actions = actions[0]
                if actions.ndim != 2 or actions.size == 0:
                    raise ValueError(f"无效 action 张量, shape={actions.shape}")
                if actions.shape[-1] < ACTION_DIM:
                    raise ValueError(f"动作维度 {actions.shape[-1]} < {ACTION_DIM}")
                H = min(actions.shape[0], self._args.action_horizon)
                chunk = actions[:H, :ACTION_DIM].copy()
            except Exception as exc:
                self.get_logger().warning(f"解析推理结果失败: {exc}")
                continue

            alpha = 0.3
            self._infer_latency_ema = alpha * elapsed + (1.0 - alpha) * self._infer_latency_ema
            self._chunk_index += 1

            self._write_json_event(
                "sync_inference",
                chunk_index=self._chunk_index,
                feedback_state=state,
                feedback_state_stamp=self._latest_state_stamp,
                image_stamp=self._latest_image_stamp,
                action_chunk=chunk,
                inference_ms=elapsed * 1000.0,
                latency_ema_ms=self._infer_latency_ema * 1000.0,
                control_tick_at_infer=self._control_tick,
                chunk_horizon=int(H),
            )
            prefix = "[DEBUG] " if self._debug else ""
            self.get_logger().info(
                f"{prefix}sync 推理完成: chunk_idx={self._chunk_index}  shape={chunk.shape}  "
                f"infer_ms={elapsed * 1000:.1f}  latency_ema_ms={self._infer_latency_ema * 1000:.1f}  "
                f"action_range=[{chunk.min():.3f}, {chunk.max():.3f}]"
            )

            # ── ④ 顺序执行整个 chunk ──
            self._execute_chunk(chunk, prev_action)
            # 更新跨 chunk 插值起点 = 本 chunk 最后一帧
            prev_action = chunk[-1].copy()

    def _snapshot_obs_blocking(
        self,
    ) -> tuple[GrootZmqClient | None, np.ndarray | None, np.ndarray | None]:
        """阻塞等待 server + state + image 同时就绪。停止时返回三个 None。"""
        while not self._stop_event.is_set():
            with self._lock:
                client = self._policy_client
                state = None if self._latest_state is None else self._latest_state.copy()
                image = None if self._latest_image is None else self._latest_image.copy()
            if client is not None and state is not None and image is not None:
                return client, state, image
            self._maybe_log_waiting(client, state, image)
            self._stop_event.wait(0.05)
        return None, None, None

    def _execute_chunk(self, chunk: np.ndarray, prev_action: np.ndarray | None) -> None:
        """以 control_dt 节奏顺序发布 chunk 的每一帧 (每帧内部再做 servo 插值)。"""
        factor = self._interpolation_factor
        # 第一帧的"前动作"是上一 chunk 末帧 (跨 chunk 边界做线性插值);
        # 首 chunk 时为 None, 内层会直接发首帧 (slew limiter 会从当前 state 起步)。
        local_prev: np.ndarray | None = prev_action
        next_servo_deadline = time.perf_counter()

        for i in range(chunk.shape[0]):
            if self._stop_event.is_set():
                return
            curr_action = chunk[i]

            self._write_json_event(
                "control_action",
                chunk_index=self._chunk_index,
                step_in_chunk=i,
                control_tick=self._control_tick,
                output_action=curr_action,
            )

            if self._debug:
                arm = curr_action[:ACTION_ARM_DIM]
                hand = curr_action[ACTION_ARM_DIM:ACTION_DIM]
                self.get_logger().info(
                    f"[DEBUG] action suppressed  chunk={self._chunk_index} step={i}/{chunk.shape[0]}  "
                    f"tick={self._control_tick}  "
                    f"arm[0,9]=[{arm[0]:.4f}, {arm[9]:.4f}]  "
                    f"arm_range=[{arm.min():.3f}, {arm.max():.3f}]  "
                    f"hand={hand.tolist()}"
                )

            self._control_tick += 1

            # 内层: factor 个子步, 线性插值并发布
            for sub in range(factor):
                if self._stop_event.is_set():
                    return
                if local_prev is None:
                    interp_action = curr_action
                else:
                    t = (sub + 1) / factor
                    interp_arm = (
                        local_prev[:ACTION_ARM_DIM] * (1.0 - t) + curr_action[:ACTION_ARM_DIM] * t
                    )
                    interp_hand = curr_action[ACTION_ARM_DIM:ACTION_DIM]
                    interp_action = np.concatenate([interp_arm, interp_hand])

                self._apply_action(interp_action)
                next_servo_deadline += self._servo_dt
                self._sleep_until(next_servo_deadline)

            local_prev = curr_action


# ═══════════════════════════════════════════════════════════════════
#  命令行参数
# ═══════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adam U ROS2 GR00T 客户端 (同步执行版, PI05 微调模型)")
    p.add_argument("--host", default=DEFAULT_HOST, help="GR00T 推理服务器地址。")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help="GR00T 服务器 ZMQ 端口。")
    p.add_argument("--api-token", default=None, help="可选 API token。")
    p.add_argument(
        "--api-key", default=None, dest="api_token", help="兼容旧名: 等价于 --api-token。"
    )
    p.add_argument(
        "--timeout-ms",
        type=int,
        default=30000,
        help="ZMQ REQ socket 收发超时 (毫秒), 同步模式下默认 30s 以容忍较高的服务端延迟。",
    )
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="发送给策略的自然语言指令。")

    # ROS 话题
    p.add_argument("--camera-topic", default=DEFAULT_CAMERA_TOPIC)
    p.add_argument("--lowstate-topic", default=DEFAULT_LOWSTATE_TOPIC)
    p.add_argument("--handstate-topic", default=DEFAULT_HANDSTATE_TOPIC)
    p.add_argument("--lowcmd-topic", default=DEFAULT_LOWCMD_TOPIC)
    p.add_argument("--handcmd-topic", default=DEFAULT_HANDCMD_TOPIC)

    # 同步模式核心参数 (无 RTC 调参)
    p.add_argument(
        "--control-hz",
        type=float,
        default=DEFAULT_CONTROL_HZ,
        help="策略动作执行频率 (Hz)。Δt = 1/control_hz。",
    )
    p.add_argument(
        "--action-horizon",
        type=int,
        default=DEFAULT_ACTION_HORIZON,
        help="H: 每次推理执行的 chunk 步数 (默认 32, 与 pi05_config.py 一致)。"
        "可设小一些 (如 16) 减少推理间隔间的停顿但增加推理频率。",
    )
    p.add_argument(
        "--interpolation-factor",
        type=int,
        default=DEFAULT_INTERPOLATION_FACTOR,
        help="插值倍率 N: 伺服频率 = control_hz × N。设为 1 则关闭插值。",
    )
    p.add_argument(
        "--max-arm-velocity",
        type=float,
        default=DEFAULT_MAX_ARM_VELOCITY,
        help="输出层关节滑率限幅 (rad/s)。≤0 则关闭。",
    )

    # 图像
    p.add_argument(
        "--render-size",
        type=int,
        default=DEFAULT_RENDER_SIZE,
        help="客户端预 letterbox 缩放尺寸; 0 = 不缩放 (推荐)。",
    )

    # 杂项
    p.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Debug: 推理照常运行并打印 latency 信息, 但不发布任何 ROS 指令。",
    )
    p.add_argument("--ros-domain-id", type=int, default=1, help="ROS_DOMAIN_ID。")
    p.add_argument(
        "--json-log-path",
        default=_default_json_log_path(),
        help="JSONL 日志文件路径 (用于离线可视化)。",
    )

    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = parse_args()
    os.environ["ROS_DOMAIN_ID"] = str(args.ros_domain_id)

    rclpy.init()
    node = GrootAdamUSyncClient(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        logging.info("正在关闭 GR00T 同步客户端。")
    finally:
        try:
            node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
