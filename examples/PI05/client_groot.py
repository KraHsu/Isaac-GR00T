#!/usr/bin/env python3
"""ROS2 GR00T 客户端 — Real-Time Chunking (RTC) 版本
=======================================================

PI05 微调后的 GR00T N1.7 3B 推理服务的 ROS2 客户端, 复用 Physical
Intelligence 提出的 RTC 执行模式 (frozen prefix + soft-mask blending),
异步推理 + 动作队列融合, 与 ``client_openpi.py`` 行为一致, 仅替换
推理服务端协议与观测/动作格式。

本文件 **完全自包含**, 机器人侧只需 ``pyzmq + msgpack + numpy + rclpy``,
不必安装 gr00t 包及其重型依赖 (torch / transformers / flash-attn / 等)。

─── 与 ``client_openpi.py`` 的核心差异 ───

1. 协议:  WebSocket → ZeroMQ REQ/REP, msgpack 序列化
   连接 ``gr00t/eval/run_gr00t_server.py`` 启动的 ``PolicyServer``;
   ZMQ + msgpack 协议在文件顶部内联实现 (与 PolicyServer 双向兼容)。

2. 观测格式: 必须包含 batch + temporal 维度
     {
       "video":    {"zed_rgb": uint8 (1, 1, H, W, 3)},
       "state":    {"joints":  float32 (1, 1, 21)},
       "language": {"annotation.human.task_description": [["..."]]},
     }

3. 动作格式:  client.get_action(obs) → (action_dict, info)
   PI05 输出 ``action_dict["joints"]`` shape=(1, 32, 21), 取 [0] 喂给
   RTC 队列即可, 其它环节完全不变。

4. 图像预处理: GR00T 服务端的 processor 内部会自动 resize, 客户端
   默认直接发原始相机图像 (HWC uint8); 如需节省带宽可用 ``--render-size``
   做 letterbox 缩放。

─── 术语对照 (与 RTC 论文一致) ───

  H  = action_horizon      模型输出的 chunk 总长度 (PI05 默认 H=32)
  s  = execution_horizon    每次推理之间实际执行的步数 (默认 s=15)
  d  = inference_delay      推理耗时对应的控制步数, 自动估算
  Δt = 1/control_hz         控制周期 (默认 ~67ms, 即 15Hz)
"""

from __future__ import annotations

import argparse
from collections import deque
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
#  内联 ZMQ + msgpack 协议
# ═══════════════════════════════════════════════════════════════════
#
#  与 gr00t/policy/server_client.py 中的 PolicyServer 双向兼容。
#  仅实现客户端真正用到的两个端点 (ping / get_action), 避免拉入
#  gr00t 包及其重型依赖 (torch / transformers / flash-attn / 等)。
#  机器人侧只需: pyzmq + msgpack + numpy (+ ROS distro 自带 rclpy)。


def _ndarray_pack(obj: object) -> object:
    """msgpack default 钩子: 把 numpy ndarray 序列化为 .npy bytes。"""
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    raise TypeError(f"Unsupported type: {type(obj)}")


def _ndarray_unpack(obj: dict) -> object:
    """msgpack object_hook: 还原 ndarray; ModalityConfig 字段不解析直接保留。"""
    if not isinstance(obj, dict):
        return obj
    if obj.get("__ndarray_class__"):
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


class GrootZmqClient:
    """轻量级 ZeroMQ REQ 客户端, 与 gr00t.policy.server_client.PolicyServer 兼容。

    仅实现 ping / get_action 两个端点; 故意不调 get_modality_config 以
    避开服务端返回 ModalityConfig 实例时的反序列化问题 (ModalityConfig
    需要 gr00t.data.types, 我们这里不引入)。
    """

    def __init__(
        self,
        host: str,
        port: int,
        timeout_ms: int = 15000,
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
            # 超时: REQ socket 进入坏状态, 重建后再上抛
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
        # 服务端返回 (action_dict, info_dict); msgpack 解出来是 list
        return tuple(response)  # type: ignore[return-value]

    def close(self) -> None:
        try:
            self.socket.close(linger=0)
        except Exception:  # pragma: no cover
            pass


# ┌──────────────────────────────────────────────────────────────────┐
# │                        维度常量                                  │
# │  Adam U 上半身 19 个关节 + 左右手各 1 个归一化夹爪值 = 21 维     │
# └──────────────────────────────────────────────────────────────────┘
LOWSTATE_DIM = 19  # 上半身关节数 (waist×3, neck×2, left arm×7, right arm×7)
HANDSTATE_RAW_DIM = 12  # 原始手部通道数 (左手6 + 右手6)
HANDSTATE_ENCODED_DIM = 2  # 编码后手部维度 (左手1 + 右手1, 归一化到 0/1)
STATE_DIM = LOWSTATE_DIM + HANDSTATE_ENCODED_DIM  # 策略输入状态维度 = 21
ACTION_ARM_DIM = LOWSTATE_DIM  # 动作中手臂部分维度 = 19
ACTION_HAND_DIM = HANDSTATE_ENCODED_DIM  # 动作中手部部分维度 = 2
ACTION_DIM = ACTION_ARM_DIM + ACTION_HAND_DIM  # 单步动作总维度 = 21
HAND_CMD_DIM = HANDSTATE_RAW_DIM  # 发送给手部控制器的维度 = 12
LEFT_HAND_SLICE = slice(0, 6)  # 左手在 12 维中的索引
RIGHT_HAND_SLICE = slice(6, 12)  # 右手在 12 维中的索引

# ┌──────────────────────────────────────────────────────────────────┐
# │                    默认 ROS 话题与参数                           │
# └──────────────────────────────────────────────────────────────────┘
DEFAULT_CAMERA_TOPIC = "/zed/zed_node/rgb/color/rect/image"
DEFAULT_LOWSTATE_TOPIC = "lowstate"
DEFAULT_HANDSTATE_TOPIC = "handstate"
DEFAULT_LOWCMD_TOPIC = "lowcmd"
DEFAULT_HANDCMD_TOPIC = "handcmd"
DEFAULT_HOST = "192.168.31.116"
DEFAULT_PORT = 5555  # GR00T 服务端默认端口 (ZMQ REQ/REP)
DEFAULT_CONTROL_HZ = 15.0  # 机器人控制频率 (Hz)
DEFAULT_ACTION_HORIZON = 32  # H: 模型输出 chunk 长度 (pi05_config.py 中为 32)
DEFAULT_EXECUTION_HORIZON = 15  # s: 每执行 s 步触发新推理
DEFAULT_RENDER_SIZE = 0  # 0 = 不缩放, 直接发送原始相机图像
DEFAULT_INTERPOLATION_FACTOR = 10  # 插值倍率: 实际伺服频率 = control_hz × factor
DEFAULT_PROMPT = "fold the white T-shirt"

# 观测键名 (与 examples/PI05/pi05_config.py 保持一致)
VIDEO_KEY = "zed_rgb"
STATE_KEY = "joints"
ACTION_KEY = "joints"
LANGUAGE_KEY = "annotation.human.task_description"

# ┌──────────────────────────────────────────────────────────────────┐
# │                  19 个关节的 PD 增益配置                         │
# │  Kp: 比例增益 (位置环)    Kd: 微分增益 (速度环)                 │
# └──────────────────────────────────────────────────────────────────┘
KP = [
    # waist (腰部): roll, pitch, yaw
    1837.991943359375,
    1837.991943359375,
    1837.991943359375,
    # neck (颈部): yaw, pitch
    260.7959899902344,
    260.7959899902344,
    # left arm (左臂): shoulderPitch, shoulderRoll, shoulderYaw, elbow,
    #                   wristYaw, wristPitch, wristRoll
    294.0791931152344,
    294.0791931152344,
    312.9552001953125,
    312.9552001953125,
    312.9552001953125,
    312.9552001953125,
    312.9552001953125,
    # right arm (右臂): 同上
    294.0791931152344,
    294.0791931152344,
    312.9552001953125,
    312.9552001953125,
    312.9552001953125,
    312.9552001953125,
    312.9552001953125,
]
KD = [
    30.63319969177246,
    30.63319969177246,
    30.63319969177246,
    5.2159199714660645,
    5.2159199714660645,
    9.802639961242676,
    9.802639961242676,
    10.431839942932129,
    10.431839942932129,
    10.431839942932129,
    10.431839942932129,
    10.431839942932129,
    9.802639961242676,
    9.802639961242676,
    10.431839942932129,
    10.431839942932129,
    10.431839942932129,
    10.431839942932129,
    10.431839942932129,
]


def _default_json_log_path() -> str:
    """生成默认的 JSONL 日志路径, 带时间戳避免覆盖。"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return str(Path.cwd() / "logs" / f"groot_rtc_trace_{timestamp}.jsonl")


def _letterbox_resize(image: np.ndarray, size: int) -> np.ndarray:
    """保持宽高比的 letterbox 缩放: 居中填零到 size×size。

    输入 HWC uint8, 输出 (size, size, 3) uint8。客户端用纯 numpy 实现,
    避免引入额外依赖 (cv2/Pillow)。
    """
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    scale = min(size / h, size / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    # 最近邻缩放 (足够快; 若需更高质量可用 cv2.resize)
    ys = (np.arange(new_h) * h / new_h).astype(np.int64)
    xs = (np.arange(new_w) * w / new_w).astype(np.int64)
    resized = image[ys[:, None], xs[None, :]]

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


# ═══════════════════════════════════════════════════════════════════
#  RTC 动作队列  (RTCActionQueue)
# ═══════════════════════════════════════════════════════════════════
#
#  与 client_openpi.py 中的实现完全一致: 控制循环每个 tick 从队列头
#  弹出一个动作; 推理线程在新 chunk 就绪后调用 merge() 将其融合进
#  队列尾部 (frozen prefix + soft-mask blending)。
#


class RTCActionQueue:
    """线程安全的 RTC 动作队列, 实现冻结前缀 + 软掩码融合。

    术语 (与 RTC 论文一致):
      H  – action_horizon    模型输出的 chunk 总长度
      s  – execution_horizon  每消耗 s 步就请求新 chunk
      d  – inference_delay    推理延迟对应的控制步数 (冻结前缀长度)
    """

    def __init__(self, action_horizon: int, execution_horizon: int) -> None:
        self._lock = threading.Lock()
        self._H = action_horizon  # chunk 总长 H
        self._s = execution_horizon  # 执行窗口 s
        self._queue: deque[np.ndarray] = deque()  # 动作 FIFO
        self._steps_since_last_merge = 0  # 自上次 merge 以来已消耗的步数
        self._generation = 0  # merge 代数 (每次 merge +1)

    # ── 公共接口 ──────────────────────────────────────────────────

    def get(self) -> np.ndarray | None:
        """弹出队首动作 (控制循环每 tick 调用一次)。

        返回 None 表示队列为空 —— 控制循环应保持上一指令或静止。
        """
        with self._lock:
            if not self._queue:
                return None
            action = self._queue.popleft()
            self._steps_since_last_merge += 1
            return action

    def should_request_new_chunk(self) -> bool:
        """当已消耗 s 步时返回 True, 推理线程据此触发新推理。"""
        with self._lock:
            return self._steps_since_last_merge >= self._s

    def get_left_over(self) -> np.ndarray | None:
        """返回队列中剩余的动作序列 (用于传给服务端做 inpainting)。"""
        with self._lock:
            if not self._queue:
                return None
            return np.array(list(self._queue), dtype=np.float32)

    def merge(
        self,
        new_chunk: np.ndarray,
        inference_delay: int,
        blend_schedule: str = "exp",
        max_guidance_weight: float = 10.0,
    ) -> dict[str, Any]:
        """将新 chunk 融合进队列, 返回融合诊断信息 (用于日志/可视化)。"""
        with self._lock:
            # ① 取出旧队列剩余动作
            prev_left = np.array(list(self._queue), dtype=np.float32) if self._queue else None
            n_prev = len(prev_left) if prev_left is not None else 0

            # ② 计算重叠区域长度
            overlap = min(n_prev, len(new_chunk))

            # ③ 构建融合权重并执行混合
            blend_weights = np.zeros(max(overlap, 1), dtype=np.float32)

            if overlap > 0 and prev_left is not None:
                blended = new_chunk.copy()

                # 冻结前缀: 前 d 步权重 = 1.0 (完全使用旧 chunk)
                frozen = min(inference_delay, overlap)
                blend_weights = np.zeros(overlap, dtype=np.float32)
                blend_weights[:frozen] = 1.0

                # 软掩码区域: [d, overlap) 权重从 1.0 衰减到 ~0
                soft_len = overlap - frozen
                if soft_len > 0:
                    if blend_schedule == "exp":
                        decay_rate = math.log(max_guidance_weight + 1.0) / max(soft_len, 1)
                        for i in range(soft_len):
                            blend_weights[frozen + i] = math.exp(-decay_rate * (i + 1))
                    elif blend_schedule == "linear":
                        for i in range(soft_len):
                            blend_weights[frozen + i] = 1.0 - (i + 1) / (soft_len + 1)
                    elif blend_schedule == "ones":
                        blend_weights[frozen:] = 1.0
                    else:
                        raise ValueError(f"未知的 blend_schedule: {blend_schedule}")

                blend_weights = np.clip(blend_weights, 0.0, 1.0)

                # 执行加权融合: blended = w * prev + (1-w) * new
                for i in range(overlap):
                    blended[i] = (
                        blend_weights[i] * prev_left[i] + (1.0 - blend_weights[i]) * new_chunk[i]
                    )

                # 替换队列
                self._queue.clear()
                for a in blended:
                    self._queue.append(a)
            else:
                # 无重叠 (首次推理或队列已耗尽), 直接填入
                self._queue.clear()
                for a in new_chunk:
                    self._queue.append(a)

            self._steps_since_last_merge = 0
            self._generation += 1

            return {
                "generation": self._generation,
                "n_prev_left": n_prev,
                "overlap": overlap,
                "frozen_steps": min(inference_delay, overlap) if overlap > 0 else 0,
                "blend_weights": blend_weights.copy(),
                "prev_left_over": prev_left.copy() if prev_left is not None else None,
                "raw_new_chunk": new_chunk.copy(),
                "queue_len_after": len(self._queue),
            }

    @property
    def generation(self) -> int:
        """当前 merge 代数。"""
        with self._lock:
            return self._generation

    def __len__(self) -> int:
        with self._lock:
            return len(self._queue)


# ═══════════════════════════════════════════════════════════════════
#  主 ROS 2 节点
# ═══════════════════════════════════════════════════════════════════


class GrootAdamURTCClient(Node):
    """ROS2 节点: 使用 RTC 驱动 Adam U 机器人 (GR00T 推理服务端)。

    线程架构:
      ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
      │ ROS 回调线程 │───▶│  推理线程     │───▶│  控制线程     │
      │ (传感器更新) │    │ (异步生成     │    │ (固定频率弹出 │
      │              │    │  action chunk)│    │  动作并执行)  │
      └─────────────┘    └──────┬───────┘    └──────┬───────┘
                                │                    │
                          RTCActionQueue.merge()  .get()
    """

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("groot_adam_u_rtc_client")
        self._args = args
        self._lock = threading.Lock()  # 保护传感器缓冲区和服务连接
        self._stop_event = threading.Event()  # 用于优雅退出所有线程

        # ── 传感器缓冲区 (最新一帧, 由 ROS 回调写入) ─────────────
        self._latest_low_state: np.ndarray | None = None  # 19 维关节角
        self._latest_hand_state: np.ndarray | None = None  # 2 维编码手部
        self._latest_state: np.ndarray | None = None  # 21 维拼接状态
        self._latest_image: np.ndarray | None = None  # 预处理后图像 (HWC uint8)
        self._latest_state_stamp: float | None = None  # 状态时间戳
        self._latest_image_stamp: float | None = None  # 图像时间戳

        # ── 服务器连接 ───────────────────────────────────────────
        self._policy_client: Any | None = None
        self._server_metadata: dict[str, Any] | None = None

        # ── RTC 动作队列 ─────────────────────────────────────────
        self._action_queue = RTCActionQueue(
            action_horizon=args.action_horizon,
            execution_horizon=args.execution_horizon,
        )

        # ── 时间参数 ─────────────────────────────────────────────
        if args.control_hz <= 0.0:
            raise ValueError("control_hz 必须为正数。")
        self._control_dt = 1.0 / args.control_hz  # 策略动作间隔 Δt (秒)

        # ── 插值参数 ─────────────────────────────────────────────
        self._interpolation_factor = max(1, int(args.interpolation_factor))
        self._servo_dt = self._control_dt / self._interpolation_factor

        # ── JSON 日志 ────────────────────────────────────────────
        self._json_log_lock = threading.Lock()
        self._json_logging_enabled = True
        self._json_log_fp = self._open_json_log_file(args.json_log_path)
        self._last_wait_log_time = 0.0

        # ── Debug 模式 (不发布任何指令, 仅打印信息) ─────────────────
        self._debug = getattr(args, "debug", False)

        # ── "首次收到" 标志, 避免重复打印 ─────────────────────────
        self._has_logged_first_image = False
        self._has_logged_first_state = False
        self._has_logged_first_hand_state = False
        self._has_logged_first_action_apply = False

        # ── 推理延迟 EMA 追踪器 (用于自动估算 d) ─────────────────
        self._infer_latency_ema = 0.0  # 秒, 指数移动平均

        # ── 全局控制 tick 计数器 ────────────────────────────────
        self._control_tick = 0

        # ── ROS 发布器 / 订阅器 ──────────────────────────────────
        self._lowcmd_pub = self.create_publisher(LowCmd, args.lowcmd_topic, 10)
        self._handcmd_pub = self.create_publisher(HandCmd, args.handcmd_topic, 10)

        self.create_subscription(LowState, args.lowstate_topic, self._on_lowstate, 10)
        self.create_subscription(HandState, args.handstate_topic, self._on_handstate, 10)
        self.create_subscription(Image, args.camera_topic, self._on_image, qos_profile_sensor_data)

        # 记录会话启动参数
        self._write_json_event(
            "session_start",
            model="gr00t-n1d7-pi05",
            host=args.host,
            port=args.port,
            prompt=args.prompt,
            control_hz=args.control_hz,
            action_horizon=args.action_horizon,
            execution_horizon=args.execution_horizon,
            blend_schedule=args.blend_schedule,
            max_guidance_weight=args.max_guidance_weight,
            interpolation_factor=args.interpolation_factor,
            render_size=args.render_size,
        )

        # ── 启动工作线程 ─────────────────────────────────────────
        self._connect_thread: threading.Thread | None = None
        self._infer_thread: threading.Thread | None = None
        self._control_thread: threading.Thread | None = None
        self._start_connect_thread()
        self._start_worker_threads()

        self.get_logger().info(
            f"GR00T RTC 客户端已启动  H={args.action_horizon}  s={args.execution_horizon}  "
            f"control_hz={args.control_hz}  blend={args.blend_schedule}  "
            f"β={args.max_guidance_weight}  "
            f"interp={self._interpolation_factor}x → "
            f"servo_hz={args.control_hz * self._interpolation_factor:.0f}  "
            f"render_size={args.render_size}"
        )

    # ─────────────────────────────────────────────────────────────
    #  生命周期
    # ─────────────────────────────────────────────────────────────

    def destroy_node(self) -> None:
        """优雅关闭: 停止所有线程, 刷新日志。"""
        self._stop_event.set()
        for t in (self._connect_thread, self._infer_thread, self._control_thread):
            if t is not None and t.is_alive() and t is not threading.current_thread():
                t.join(timeout=1.0)
        self._write_json_event("session_end", reason="destroy_node")
        self._close_json_log_file()
        super().destroy_node()

    # ─────────────────────────────────────────────────────────────
    #  JSON 行日志  (每一行一个事件, 用于离线可视化)
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
        """递归转换 numpy 等类型为 JSON 可序列化类型。"""
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
        """向 JSONL 文件写入一条事件记录。"""
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
    #  服务器连接  (后台线程, 断线自动重连)
    # ─────────────────────────────────────────────────────────────

    def _start_connect_thread(self) -> None:
        if self._connect_thread is not None and self._connect_thread.is_alive():
            return
        self._connect_thread = threading.Thread(
            target=self._connect_to_server, name="groot-connect", daemon=True
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
    #  工作线程启动
    # ─────────────────────────────────────────────────────────────

    def _start_worker_threads(self) -> None:
        """启动推理线程和控制线程 (各一个, 守护模式)。"""
        if self._infer_thread is None or not self._infer_thread.is_alive():
            self._infer_thread = threading.Thread(
                target=self._infer_loop, name="groot-infer", daemon=True
            )
            self._infer_thread.start()

        if self._control_thread is None or not self._control_thread.is_alive():
            self._control_thread = threading.Thread(
                target=self._control_loop, name="groot-control", daemon=True
            )
            self._control_thread.start()

    # ─────────────────────────────────────────────────────────────
    #  ROS 传感器回调
    # ─────────────────────────────────────────────────────────────

    def _on_lowstate(self, msg: LowState) -> None:
        """接收关节状态: 提取 19 个关节角度, 与手部编码拼接成 21 维策略输入。"""
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
        """接收手部状态: 12 维原始值 → 2 维编码值 (左/右手各 0 或 1)。"""
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
        """接收 ZED 相机图像: 解码 → (可选) 缩放 → 旋转180° → 存入缓冲区。"""
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
    #  手部编码 / 解码
    # ─────────────────────────────────────────────────────────────

    def _encode_hand_state(self, hand_positions: np.ndarray) -> np.ndarray:
        """12 维原始手位 → 2 维归一化夹爪值 (二值化)。"""
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
        """拼接 19 维关节角 + 2 维手部 → 21 维策略输入状态。"""
        if self._latest_low_state is None or self._latest_hand_state is None:
            return None
        return np.concatenate([self._latest_low_state, self._latest_hand_state], axis=0).astype(
            np.float32, copy=False
        )

    def _decode_hand_action(self, hand_action: np.ndarray) -> np.ndarray:
        """2 维归一化夹爪值 → 12 维手部指令。"""
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
        hand_cmd[5] = 0  # 左手拇指旋转固定
        hand_cmd[11] = 0  # 右手拇指旋转固定
        return hand_cmd

    # ─────────────────────────────────────────────────────────────
    #  图像处理
    # ─────────────────────────────────────────────────────────────

    def _decode_ros_image(self, msg: Image) -> np.ndarray:
        """将 ROS Image 消息解码为 HWC uint8 RGB numpy 数组。"""
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
        """旋转 180°; 当 render_size>0 时额外做 letterbox 缩放。

        GR00T 服务端的 processor 内部会再次 resize, 因此默认 (render_size=0)
        直接发送原始相机分辨率的 HWC uint8 图像即可, 不需要额外预处理。
        """
        if self._args.render_size and self._args.render_size > 0:
            image = _letterbox_resize(image, int(self._args.render_size))
        image = np.rot90(image, 2)  # 旋转 180° (相机倒装)
        return np.ascontiguousarray(image, dtype=np.uint8)

    # ─────────────────────────────────────────────────────────────
    #  动作执行
    # ─────────────────────────────────────────────────────────────

    def _apply_action(self, action: np.ndarray) -> None:
        """将 21 维策略动作拆分为 lowcmd (19维) + handcmd (12维) 并发布。"""
        if action.shape[0] < ACTION_DIM:
            raise ValueError(f"动作维度 {action.shape[0]} < {ACTION_DIM}")

        arm_action = action[:ACTION_ARM_DIM]
        hand_action = action[ACTION_ARM_DIM:ACTION_DIM]

        # 构建关节指令
        lowcmd = LowCmd()
        lowcmd.motor_cmd = [MotorCmd() for _ in range(ACTION_ARM_DIM)]
        for idx, target_q in enumerate(arm_action):
            mc = lowcmd.motor_cmd[idx]
            mc.mode = 1  # 位置控制模式
            mc.q = float(target_q)
            mc.dq = 0.0
            mc.tau = 0.0
            mc.kp = KP[idx]
            mc.kd = KD[idx]
            mc.ki = 0.0

        # 构建手部指令
        hand_positions = self._decode_hand_action(hand_action)
        handcmd = HandCmd()
        for idx in range(HAND_CMD_DIM):
            handcmd.position[idx] = int(hand_positions[idx])

        # ── 安全限位 ──
        # 腰部 pitch (索引1) 限制最大前倾角度
        if lowcmd.motor_cmd[1].q > 0.5:
            lowcmd.motor_cmd[1].q = 0.5
        # 颈部 yaw (索引3) 固定为 0 (不转头)
        lowcmd.motor_cmd[3].q = 0.0
        # 颈部 pitch (索引4) 固定低头 45° (看桌面)
        lowcmd.motor_cmd[4].q = 45.0 / 180.0 * math.pi

        if not self._debug:
            self._lowcmd_pub.publish(lowcmd)
            self._handcmd_pub.publish(handcmd)

        if not self._has_logged_first_action_apply:
            self._has_logged_first_action_apply = True
            label = "[DEBUG] 首个 RTC 动作 (未发布)" if self._debug else "已执行首个 RTC 动作。"
            self.get_logger().info(label)

    # ─────────────────────────────────────────────────────────────
    #  工具函数
    # ─────────────────────────────────────────────────────────────

    def _sleep_until(self, deadline: float) -> None:
        """精确等待到目标时刻, 同时响应停止信号。"""
        remaining = deadline - time.perf_counter()
        if remaining > 0.0:
            self._stop_event.wait(remaining)

    def _maybe_log_waiting(self, client: Any, state: Any, image: Any) -> None:
        """每 5 秒打印一次等待信息, 避免刷屏。"""
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

    # ─────────────────────────────────────────────────────────────
    #  观测构造  (GR00T 协议: 必须包含 batch + temporal 维度)
    # ─────────────────────────────────────────────────────────────

    def _build_observation(self, state: np.ndarray, image: np.ndarray) -> dict[str, Any]:
        """构造符合 Gr00tPolicy.check_observation 要求的批量观测字典。

        参考 gr00t/policy/gr00t_policy.py:208-369 的形状契约:
          - video[zed_rgb]:   uint8   (B=1, T=1, H, W, 3)
          - state[joints]:    float32 (B=1, T=1, 21)
          - language[...]:    list[list[str]]  (B=1, T=1)
        """
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"image 必须为 (H, W, 3), got {image.shape}")
        if state.ndim != 1 or state.shape[0] < STATE_DIM:
            raise ValueError(f"state 必须 ≥ {STATE_DIM}, got {state.shape}")

        video_arr = np.ascontiguousarray(image, dtype=np.uint8)[None, None]  # (1,1,H,W,3)
        state_arr = state[:STATE_DIM].astype(np.float32, copy=False)[None, None]  # (1,1,D)
        return {
            "video": {VIDEO_KEY: video_arr},
            "state": {STATE_KEY: state_arr},
            "language": {LANGUAGE_KEY: [[self._args.prompt]]},
        }

    # ═════════════════════════════════════════════════════════════
    #  推理线程  (RTC: 异步, 由 execution_horizon 触发)
    # ═════════════════════════════════════════════════════════════
    #
    #  执行流程:
    #    while not stopped:
    #      ① 检查传感器和服务器是否就绪
    #      ② 等待控制循环消耗 s 步 (或队列为空)
    #      ③ 拍下最新观测快照 (state + image)
    #      ④ 调用 GR00T 服务器推理 (GrootZmqClient.get_action, 阻塞)
    #      ⑤ 计算推理延迟 d = ceil(latency / Δt) 步
    #      ⑥ 将新 chunk 通过 merge() 融合进动作队列
    #
    #  关键: 步骤 ④ 执行期间, 控制线程仍在持续弹出旧动作并执行,
    #        机器人不会暂停。

    def _infer_loop(self) -> None:
        """推理线程主循环。"""
        while not self._stop_event.is_set():
            # ① 检查就绪状态
            with self._lock:
                client = self._policy_client
                state = None if self._latest_state is None else self._latest_state.copy()
                image = None if self._latest_image is None else self._latest_image.copy()

            if client is None or state is None or image is None:
                self._maybe_log_waiting(client, state, image)
                self._stop_event.wait(0.05)
                continue

            # ② 等待队列需要补充
            while not self._stop_event.is_set():
                if len(self._action_queue) == 0 or self._action_queue.should_request_new_chunk():
                    break
                self._stop_event.wait(0.002)  # 2ms 轮询

            if self._stop_event.is_set():
                break

            # ③ 推理前再次拍下最新观测
            with self._lock:
                client = self._policy_client
                state = None if self._latest_state is None else self._latest_state.copy()
                image = None if self._latest_image is None else self._latest_image.copy()

            if client is None or state is None or image is None:
                continue

            # ④⑤⑥ 推理 + 融合
            self._run_inference(client, state, image)

    def _run_inference(self, client: Any, state: np.ndarray, image: np.ndarray) -> None:
        """单次推理: 调用服务器 → 融合新 chunk → 写日志。"""
        try:
            observation = self._build_observation(state, image)

            # ④ 远程推理 (阻塞)
            t0 = time.perf_counter()
            action_dict, _info = client.get_action(observation)
            elapsed = time.perf_counter() - t0

            if not isinstance(action_dict, dict) or ACTION_KEY not in action_dict:
                raise ValueError(
                    f"服务器返回缺少键 '{ACTION_KEY}': {list(action_dict)[:5] if isinstance(action_dict, dict) else type(action_dict)}"
                )

            actions = np.asarray(action_dict[ACTION_KEY], dtype=np.float32)

            # GR00T 输出 shape (B=1, T, D); 移除 batch 维
            if actions.ndim == 3:
                actions = actions[0]
            if actions.ndim != 2 or actions.size == 0:
                raise ValueError(f"服务器返回了无效的动作张量, shape={actions.shape}")
            if actions.shape[-1] < ACTION_DIM:
                raise ValueError(f"动作维度 {actions.shape[-1]} < {ACTION_DIM}")

            # 截取到 (H, ACTION_DIM)
            H = min(actions.shape[0], self._args.action_horizon)
            action_chunk = actions[:H, :ACTION_DIM].copy()

            # ⑤ 计算推理延迟 d (控制步数), EMA 平滑
            alpha = 0.3
            self._infer_latency_ema = alpha * elapsed + (1.0 - alpha) * self._infer_latency_ema
            d = max(1, int(math.ceil(self._infer_latency_ema / self._control_dt)))

            # ⑥ 融合进动作队列
            merge_info = self._action_queue.merge(
                new_chunk=action_chunk,
                inference_delay=d,
                blend_schedule=self._args.blend_schedule,
                max_guidance_weight=self._args.max_guidance_weight,
            )

            self._write_json_event(
                "rtc_inference",
                generation=merge_info["generation"],
                feedback_state=state,
                feedback_state_stamp=self._latest_state_stamp,
                image_stamp=self._latest_image_stamp,
                raw_new_chunk=merge_info["raw_new_chunk"],
                prev_left_over=merge_info["prev_left_over"],
                blend_weights=merge_info["blend_weights"],
                overlap=merge_info["overlap"],
                frozen_steps=merge_info["frozen_steps"],
                inference_delay_steps=d,
                inference_ms=elapsed * 1000.0,
                latency_ema_ms=self._infer_latency_ema * 1000.0,
                queue_len_after=merge_info["queue_len_after"],
                control_tick_at_merge=self._control_tick,
            )
            prefix = "[DEBUG] " if self._debug else ""
            self.get_logger().info(
                f"{prefix}RTC 推理完成: shape={action_chunk.shape}  d={d}  "
                f"overlap={merge_info['overlap']}  frozen={merge_info['frozen_steps']}  "
                f"infer_ms={elapsed * 1000:.1f}  latency_ema_ms={self._infer_latency_ema * 1000:.1f}  "
                f"queue={merge_info['queue_len_after']}  gen={merge_info['generation']}  "
                f"action_range=[{action_chunk.min():.3f}, {action_chunk.max():.3f}]"
            )
        except Exception as exc:
            self.get_logger().warning(f"推理失败: {exc}")
            with self._lock:
                self._policy_client = None
                self._server_metadata = None
            self._start_connect_thread()

    # ═════════════════════════════════════════════════════════════
    #  控制线程  (含插值, 以 servo_hz 频率发送指令)
    # ═════════════════════════════════════════════════════════════
    #
    #  外层按 control_dt 从队列弹出新动作, 内层按 servo_dt 在相邻两帧
    #  之间线性插值并发布。手部动作为二值 (0/1), 不插值, 直接持有新值。

    def _control_loop(self) -> None:
        """控制线程主循环: 以 servo_hz 频率发送插值后的指令。"""
        factor = self._interpolation_factor
        prev_action: np.ndarray | None = None  # 上一个原始动作
        curr_action: np.ndarray | None = None  # 当前原始动作

        next_servo_deadline = time.perf_counter()

        while not self._stop_event.is_set():
            # ── 每 control_dt 从队列弹出一个新动作 ──
            new_action = self._action_queue.get()

            if new_action is not None:
                prev_action = curr_action
                curr_action = new_action

                self._write_json_event(
                    "control_action",
                    generation=self._action_queue.generation,
                    control_tick=self._control_tick,
                    queue_remaining=len(self._action_queue),
                    output_action=curr_action,
                )

                if self._debug:
                    arm = curr_action[:ACTION_ARM_DIM]
                    hand = curr_action[ACTION_ARM_DIM:ACTION_DIM]
                    self.get_logger().info(
                        f"[DEBUG] action suppressed  tick={self._control_tick}  "
                        f"gen={self._action_queue.generation}  "
                        f"queue={len(self._action_queue)}  "
                        f"latency_ema_ms={self._infer_latency_ema * 1000:.1f}  "
                        f"arm[0,9]=[{arm[0]:.4f}, {arm[9]:.4f}]  "
                        f"arm_range=[{arm.min():.3f}, {arm.max():.3f}]  "
                        f"hand={hand.tolist()}"
                    )
            self._control_tick += 1

            if curr_action is None:
                next_servo_deadline += self._control_dt
                self._sleep_until(next_servo_deadline)
                continue

            # ── 内层: factor 个子步, 线性插值并发布 ──
            for sub in range(factor):
                if self._stop_event.is_set():
                    break

                if prev_action is None:
                    interp_action = curr_action
                else:
                    t = (sub + 1) / factor
                    interp_arm = (
                        prev_action[:ACTION_ARM_DIM] * (1.0 - t) + curr_action[:ACTION_ARM_DIM] * t
                    )
                    interp_hand = curr_action[ACTION_ARM_DIM:ACTION_DIM]
                    interp_action = np.concatenate([interp_arm, interp_hand])

                self._apply_action(interp_action)

                next_servo_deadline += self._servo_dt
                self._sleep_until(next_servo_deadline)


# ═══════════════════════════════════════════════════════════════════
#  命令行参数
# ═══════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Adam U ROS2 GR00T 客户端 (RTC 实时分块版, PI05 微调模型)"
    )
    p.add_argument("--host", default=DEFAULT_HOST, help="GR00T 推理服务器地址。")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help="GR00T 推理服务器端口 (ZMQ)。")
    p.add_argument(
        "--api-token",
        default=None,
        help="可选的 API token (与 PolicyServer.api_token 对应)。",
    )
    p.add_argument(
        "--api-key",
        default=None,
        dest="api_token",
        help="兼容旧名: --api-key 等价于 --api-token。",
    )
    p.add_argument(
        "--timeout-ms",
        type=int,
        default=15000,
        help="ZMQ REQ socket 收发超时 (毫秒), 默认 15s。",
    )
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="发送给策略的自然语言指令。")

    # ROS 话题
    p.add_argument("--camera-topic", default=DEFAULT_CAMERA_TOPIC)
    p.add_argument("--lowstate-topic", default=DEFAULT_LOWSTATE_TOPIC)
    p.add_argument("--handstate-topic", default=DEFAULT_HANDSTATE_TOPIC)
    p.add_argument("--lowcmd-topic", default=DEFAULT_LOWCMD_TOPIC)
    p.add_argument("--handcmd-topic", default=DEFAULT_HANDCMD_TOPIC)

    # RTC 核心参数
    p.add_argument(
        "--control-hz",
        type=float,
        default=DEFAULT_CONTROL_HZ,
        help="控制循环频率 (Hz)。Δt = 1/control_hz。",
    )
    p.add_argument(
        "--action-horizon",
        type=int,
        default=DEFAULT_ACTION_HORIZON,
        help="H: 模型输出的 chunk 总长度 (PI05 默认 32)。",
    )
    p.add_argument(
        "--execution-horizon",
        type=int,
        default=DEFAULT_EXECUTION_HORIZON,
        help="s: 每消耗 s 步请求新 chunk (默认 15)。",
    )
    p.add_argument(
        "--blend-schedule",
        choices=("exp", "linear", "ones"),
        default="exp",
        help="RTC 软掩码衰减方式: exp(指数)/linear(线性)/ones(全冻结)。",
    )
    p.add_argument(
        "--max-guidance-weight",
        type=float,
        default=10.0,
        help="β: 融合权重裁剪上限。论文推荐 5~10。",
    )
    p.add_argument(
        "--interpolation-factor",
        type=int,
        default=DEFAULT_INTERPOLATION_FACTOR,
        help="插值倍率 N: 伺服频率 = control_hz × N。设为 1 则关闭插值。",
    )

    # 图像
    p.add_argument(
        "--render-size",
        type=int,
        default=DEFAULT_RENDER_SIZE,
        help="客户端预 letterbox 缩放尺寸 (正方形)。0 = 不缩放, 直接发原图; "
        "GR00T 服务端 processor 会自动 resize, 一般保持默认即可。",
    )

    # 杂项
    p.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Debug 模式: 推理照常运行并打印 latency/queue 信息, 但不向机器人发布任何指令。",
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
    node = GrootAdamURTCClient(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        logging.info("正在关闭 GR00T RTC 客户端。")
    finally:
        try:
            node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
