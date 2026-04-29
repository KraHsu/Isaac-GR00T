#!/usr/bin/env python3
"""ROS2 OpenPI 客户端 — Real-Time Chunking (RTC) 版本
=======================================================

实现 Physical Intelligence 提出的 RTC 执行模式：
  https://www.pi.website/research/real_time_chunking
  论文: https://arxiv.org/abs/2506.07339

─── 与原始同步 / skip-actions 方案的核心区别 ───

1. 真正的异步推理
   推理线程在机器人执行当前 chunk 的 *同时* 就开始生成下一个 chunk，
   不再等 chunk 执行完才调用模型。

2. 执行窗口 (execution_horizon, 记作 s)
   每个 chunk 只执行 s 步就触发新推理（而非执行完全部 H 步）。
   s 远小于 H，因此新旧 chunk 有大量重叠。

3. 冻结前缀 + 软掩码融合 (frozen prefix + soft-mask blending)
   新 chunk 到达时，其前 d 步（推理延迟）被 *冻结* 为旧 chunk 对应值；
   剩余重叠区域用指数衰减权重平滑混合：
     blended[i] = w[i] * prev[i] + (1 - w[i]) * new[i]
   彻底消除 chunk 边界的跳变和暂停。

4. 动作队列 (ActionQueue)
   专用 FIFO 队列解耦推理与执行，控制循环严格按固定频率弹出动作。

─── 术语对照 (与 RTC 论文一致) ───

  H  = action_horizon      模型输出的 chunk 总长度 (本项目中 H=32)
  s  = execution_horizon    每次推理之间实际执行的步数 (默认 s=10)
  d  = inference_delay      推理耗时对应的控制步数，自动估算
  Δt = 1/control_hz         控制周期 (默认 20ms, 即 50Hz)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from pnd_adam.msg import HandCmd, HandState, LowCmd, LowState, MotorCmd
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

try:
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy
except ImportError as exc:  # pragma: no cover
    image_tools = None
    websocket_client_policy = None
    OPENPI_IMPORT_ERROR = exc
else:
    OPENPI_IMPORT_ERROR = None

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
DEFAULT_CONTROL_HZ = 15.0  # 机器人控制频率 (Hz), Δt = 20ms
DEFAULT_ACTION_HORIZON = 32  # H: 模型输出 chunk 长度 (本项目实际为 32)
DEFAULT_EXECUTION_HORIZON = 15  # s: 每执行 s 步触发新推理
DEFAULT_RENDER_SIZE = 224  # 发送给策略的图像尺寸 (正方形)
DEFAULT_INTERPOLATION_FACTOR = 10  # 插值倍率: 实际伺服频率 = control_hz × factor
# 例: 50Hz × 10 = 500Hz, 每对原始 action 之间插 8 个点

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
    """生成默认的 JSONL 日志路径，带时间戳避免覆盖。"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return str(Path.cwd() / "logs" / f"openpi_rtc_trace_{timestamp}.jsonl")


# ═══════════════════════════════════════════════════════════════════
#  RTC 动作队列  (RTCActionQueue)
# ═══════════════════════════════════════════════════════════════════
#
#  核心数据结构。控制循环每个 tick 从队列头部弹出一个动作；推理线程
#  在新 chunk 就绪后调用 merge() 将其融合进队列尾部。
#
#  融合流程 (merge):
#
#    旧队列剩余:  [prev_0, prev_1, ..., prev_{n-1}]
#    新 chunk:     [new_0,  new_1,  ..., new_{H-1}]
#
#    重叠区域长度 overlap = min(n, H)
#
#    对于 i ∈ [0, overlap):
#      若 i < d (冻结前缀):  blended[i] = prev[i]          权重 w=1
#      若 i ∈ [d, overlap):  blended[i] = w·prev[i] + (1-w)·new[i]
#                            w 按指数衰减从 1.0 → ~0
#    对于 i ∈ [overlap, H):
#      blended[i] = new[i]  (无旧数据, 直接使用新 chunk)
#


class RTCActionQueue:
    """线程安全的 RTC 动作队列，实现冻结前缀 + 软掩码融合。

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
        """将新 chunk 融合进队列, 返回融合诊断信息 (用于日志/可视化)。

        参数
        ----
        new_chunk : ndarray, shape (H, action_dim)
            策略服务器返回的原始动作 chunk。
        inference_delay : int
            推理延迟 d (控制步数)。前 d 步被冻结为旧 chunk 对应值。
        blend_schedule : str
            "exp"    — 指数衰减 (推荐, 论文默认)
            "linear" — 线性递减
            "ones"   — 全部冻结 (重叠区域内完全使用旧 chunk)
        max_guidance_weight : float
            β: 融合权重裁剪上限, 控制指数衰减速率。

        返回
        ----
        dict  包含 prev_left_over, blend_weights, raw_new_chunk 等诊断字段。
        """
        with self._lock:
            # ① 取出旧队列剩余动作
            prev_left = (
                np.array(list(self._queue), dtype=np.float32) if self._queue else None
            )
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
                        # 指数衰减: w(i) = exp(-λ·(i+1))
                        # λ 选取使得 w(soft_len) ≈ 1/(β+1)
                        decay_rate = math.log(max_guidance_weight + 1.0) / max(
                            soft_len, 1
                        )
                        for i in range(soft_len):
                            blend_weights[frozen + i] = math.exp(-decay_rate * (i + 1))
                    elif blend_schedule == "linear":
                        # 线性递减: w(i) = 1 - (i+1)/(soft_len+1)
                        for i in range(soft_len):
                            blend_weights[frozen + i] = 1.0 - (i + 1) / (soft_len + 1)
                    elif blend_schedule == "ones":
                        # 全部冻结 (不推荐, 反应性差)
                        blend_weights[frozen:] = 1.0
                    else:
                        raise ValueError(f"未知的 blend_schedule: {blend_schedule}")

                blend_weights = np.clip(blend_weights, 0.0, 1.0)

                # 执行加权融合: blended = w * prev + (1-w) * new
                for i in range(overlap):
                    blended[i] = (
                        blend_weights[i] * prev_left[i]
                        + (1.0 - blend_weights[i]) * new_chunk[i]
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

            # 返回诊断信息, 供日志和可视化使用
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


class OpenPIAdamURTCClient(Node):
    """ROS2 节点: 使用 RTC 驱动 Adam U 机器人。

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
        super().__init__("openpi_adam_u_rtc_client")
        self._args = args
        self._lock = threading.Lock()  # 保护传感器缓冲区和服务连接
        self._stop_event = threading.Event()  # 用于优雅退出所有线程

        # ── 传感器缓冲区 (最新一帧, 由 ROS 回调写入) ─────────────
        self._latest_low_state: np.ndarray | None = None  # 19 维关节角
        self._latest_hand_state: np.ndarray | None = None  # 2 维编码手部
        self._latest_state: np.ndarray | None = None  # 21 维拼接状态
        self._latest_image: np.ndarray | None = None  # 预处理后图像
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
        self._control_dt = 1.0 / args.control_hz  # 策略动作间隔 Δt (秒), 如 20ms

        # ── 插值参数 ─────────────────────────────────────────────
        #  插值倍率 N: 每对原始动作之间插入 N-2 个中间点 + 两端共 N 个子步
        #  实际伺服频率 = control_hz × N, 如 50×10 = 500Hz
        #  伺服周期 servo_dt = control_dt / N, 如 20ms / 10 = 2ms
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

        # ── 全局控制 tick 计数器 (用于日志中标注动作时间线) ────────
        self._control_tick = 0

        if OPENPI_IMPORT_ERROR is not None:
            raise RuntimeError(
                "openpi-client 不可用。请先安装：\n"
                "  cd $OPENPI_ROOT/packages/openpi-client && pip install -e ."
            ) from OPENPI_IMPORT_ERROR

        # ── ROS 发布器 / 订阅器 ──────────────────────────────────
        self._lowcmd_pub = self.create_publisher(LowCmd, args.lowcmd_topic, 10)
        self._handcmd_pub = self.create_publisher(HandCmd, args.handcmd_topic, 10)

        self.create_subscription(LowState, args.lowstate_topic, self._on_lowstate, 10)
        self.create_subscription(
            HandState, args.handstate_topic, self._on_handstate, 10
        )
        self.create_subscription(
            Image, args.camera_topic, self._on_image, qos_profile_sensor_data
        )

        # 记录会话启动参数
        self._write_json_event(
            "session_start",
            host=args.host,
            port=args.port,
            prompt=args.prompt,
            control_hz=args.control_hz,
            action_horizon=args.action_horizon,
            execution_horizon=args.execution_horizon,
            blend_schedule=args.blend_schedule,
            max_guidance_weight=args.max_guidance_weight,
            interpolation_factor=args.interpolation_factor,
        )

        # ── 启动工作线程 ─────────────────────────────────────────
        self._connect_thread: threading.Thread | None = None
        self._infer_thread: threading.Thread | None = None
        self._control_thread: threading.Thread | None = None
        self._start_connect_thread()
        self._start_worker_threads()

        self.get_logger().info(
            f"RTC 客户端已启动  H={args.action_horizon}  s={args.execution_horizon}  "
            f"control_hz={args.control_hz}  blend={args.blend_schedule}  "
            f"β={args.max_guidance_weight}  "
            f"interp={self._interpolation_factor}x → "
            f"servo_hz={args.control_hz * self._interpolation_factor:.0f}"
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
            target=self._connect_to_server, name="openpi-connect", daemon=True
        )
        self._connect_thread.start()

    def _connect_to_server(self) -> None:
        assert websocket_client_policy is not None
        while not self._stop_event.is_set():
            try:
                self.get_logger().info(
                    f"正在连接 OpenPI 服务器 {self._args.host}:{self._args.port}..."
                )
                client = websocket_client_policy.WebsocketClientPolicy(
                    host=self._args.host,
                    port=self._args.port,
                    api_key=self._args.api_key,
                )
                metadata = client.get_server_metadata()
                with self._lock:
                    self._policy_client = client
                    self._server_metadata = metadata
                self.get_logger().info(f"已连接: {metadata}")
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
                target=self._infer_loop, name="openpi-infer", daemon=True
            )
            self._infer_thread.start()

        if self._control_thread is None or not self._control_thread.is_alive():
            self._control_thread = threading.Thread(
                target=self._control_loop, name="openpi-control", daemon=True
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
            # 维度不匹配时补零
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
        """接收 ZED 相机图像: 解码 → 缩放/填充 → 旋转180° → 存入缓冲区。"""
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
                f"首次收到图像: {msg.encoding} {msg.width}×{msg.height}"
            )

    # ─────────────────────────────────────────────────────────────
    #  手部编码 / 解码
    # ─────────────────────────────────────────────────────────────

    def _encode_hand_state(self, hand_positions: np.ndarray) -> np.ndarray:
        """12 维原始手位 → 2 维归一化夹爪值。

        每只手取前 5 个手指通道的均值, 除以 1800 归一化到 [0,1],
        再二值化: >0.5 → 1.0 (张开), ≤0.5 → 0.0 (闭合)。
        第 6 通道 (拇指旋转) 不参与编码, 解码时固定为 0。
        """
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
        return np.concatenate(
            [self._latest_low_state, self._latest_hand_state], axis=0
        ).astype(np.float32, copy=False)

    def _decode_hand_action(self, hand_action: np.ndarray) -> np.ndarray:
        """2 维归一化夹爪值 → 12 维手部指令。

        策略输出 hand_action ∈ {0, 1}, 取反后:
          - 1→闭合 映射为 position=1000
          - 0→张开 映射为 position=0
        前 5 通道为手指, 第 6 通道 (拇指旋转) 固定为 0。
        """
        hand_cmd = np.zeros((HAND_CMD_DIM,), dtype=np.uint32)
        left_cmd = 1000 if float(np.clip(1.0 - hand_action[0], 0.0, 1.0)) > 0.5 else 500
        right_cmd = (
            1000 if float(np.clip(1.0 - hand_action[1], 0.0, 1.0)) > 0.5 else 500
        )
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
        channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}.get(
            encoding
        )
        if channels is None:
            raise ValueError(f"不支持的图像编码: {msg.encoding}")

        row_stride = int(msg.step)
        expected_width = int(msg.width) * channels
        flat = np.frombuffer(msg.data, dtype=np.uint8)
        image = flat.reshape((int(msg.height), row_stride))
        image = image[:, :expected_width].reshape(
            (int(msg.height), int(msg.width), channels)
        )

        # 颜色通道转换: BGR→RGB, BGRA→RGBA→RGB
        if encoding == "bgr8":
            image = image[:, :, ::-1]
        elif encoding == "bgra8":
            image = image[:, :, [2, 1, 0, 3]]
        # 单通道 → 三通道; 四通道 → 丢弃 alpha
        if channels == 1:
            image = np.repeat(image, 3, axis=2)
        elif channels == 4:
            image = image[:, :, :3]
        return np.ascontiguousarray(image)

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """缩放+填充到 render_size × render_size, 旋转180°, 可选转置为 CHW。"""
        assert image_tools is not None
        resized = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(
                image, self._args.render_size, self._args.render_size
            )
        )
        resized = np.rot90(resized, 2)  # 旋转 180° (相机倒装)
        if self._args.image_layout == "chw":
            return np.transpose(resized, (2, 0, 1))
        return resized

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
            mc.q = float(target_q)  # 目标角度 (rad)
            mc.dq = 0.0  # 目标角速度
            mc.tau = 0.0  # 前馈力矩
            mc.kp = KP[idx]  # 比例增益
            mc.kd = KD[idx]  # 微分增益
            mc.ki = 0.0  # 积分增益 (未使用)

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

    # ═════════════════════════════════════════════════════════════
    #  推理线程  (RTC: 异步, 由 execution_horizon 触发)
    # ═════════════════════════════════════════════════════════════
    #
    #  执行流程:
    #
    #    while not stopped:
    #      ① 检查传感器和服务器是否就绪
    #      ② 等待控制循环消耗 s 步 (或队列为空)
    #      ③ 拍下最新观测快照 (state + image)
    #      ④ 调用远程 OpenPI 服务器推理 (阻塞, 耗时 ~100-200ms)
    #      ⑤ 计算推理延迟 d = ceil(latency / Δt) 步
    #      ⑥ 将新 chunk 通过 merge() 融合进动作队列
    #
    #  关键: 步骤 ④ 执行期间, 控制线程仍在持续弹出旧动作并执行,
    #        机器人不会暂停。这就是 RTC 的"异步"核心。

    def _infer_loop(self) -> None:
        """推理线程主循环。"""
        while not self._stop_event.is_set():
            # ① 检查就绪状态
            with self._lock:
                client = self._policy_client
                state = (
                    None if self._latest_state is None else self._latest_state.copy()
                )
                image = (
                    None if self._latest_image is None else self._latest_image.copy()
                )

            if client is None or state is None or image is None:
                self._maybe_log_waiting(client, state, image)
                self._stop_event.wait(0.05)
                continue

            # ② 等待队列需要补充
            #    首次 (队列为空) 或已消耗 s 步时立即触发
            while not self._stop_event.is_set():
                if (
                    len(self._action_queue) == 0
                    or self._action_queue.should_request_new_chunk()
                ):
                    break
                self._stop_event.wait(0.002)  # 2ms 轮询

            if self._stop_event.is_set():
                break

            # ③ 推理前再次拍下最新观测 (尽量新鲜)
            with self._lock:
                client = self._policy_client
                state = (
                    None if self._latest_state is None else self._latest_state.copy()
                )
                image = (
                    None if self._latest_image is None else self._latest_image.copy()
                )

            if client is None or state is None or image is None:
                continue

            # ④⑤⑥ 推理 + 融合
            self._run_inference(client, state, image)

    def _run_inference(self, client: Any, state: np.ndarray, image: np.ndarray) -> None:
        """单次推理: 调用服务器 → 融合新 chunk → 写日志。"""
        try:
            observation = {
                "state": state,
                "images": {"cam_high": image},
                "prompt": self._args.prompt,
            }

            # ④ 远程推理 (阻塞)
            t0 = time.perf_counter()
            result = client.infer(observation)
            elapsed = time.perf_counter() - t0

            actions = np.asarray(result.get("actions"), dtype=np.float32)
            if actions.ndim != 2 or actions.size == 0:
                raise ValueError("服务器返回了空的动作张量。")
            if actions.shape[-1] < ACTION_DIM:
                raise ValueError(f"动作维度 {actions.shape[-1]} < {ACTION_DIM}")

            # 截取到 (H, ACTION_DIM)
            H = min(actions.shape[0], self._args.action_horizon)
            action_chunk = actions[:H, :ACTION_DIM].copy()

            # ⑤ 计算推理延迟 d (控制步数)
            #    使用 EMA 平滑, 避免单次波动导致 d 跳变
            alpha = 0.3
            self._infer_latency_ema = (
                alpha * elapsed + (1.0 - alpha) * self._infer_latency_ema
            )
            d = max(1, int(math.ceil(self._infer_latency_ema / self._control_dt)))

            # ⑥ 融合进动作队列
            merge_info = self._action_queue.merge(
                new_chunk=action_chunk,
                inference_delay=d,
                blend_schedule=self._args.blend_schedule,
                max_guidance_weight=self._args.max_guidance_weight,
            )

            # 写入详细日志 (可视化脚本依赖这些字段)
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
    #  时序示意 (以 control_hz=50, interpolation_factor=10 为例):
    #
    #    策略动作:     a[0]          a[1]          a[2]    ...
    #    时间轴:       |----20ms-----|----20ms-----|
    #    伺服指令:     s0 s1 ... s9  s10 s11...s19
    #                  |--2ms--|     |--2ms--|
    #
    #    s_k = lerp(a[i], a[i+1], t)   其中 t = (k+1) / factor
    #    k=0 → t=0.1 (接近 a[i])
    #    k=9 → t=1.0 (到达 a[i+1])
    #
    #  手部动作为二值 (0/1), 不做线性插值, 直接持有新值。

    def _control_loop(self) -> None:
        """控制线程主循环: 以 servo_hz 频率发送插值后的指令。

        外层按 control_dt (50Hz) 从队列弹出新动作,
        内层按 servo_dt (500Hz) 在相邻两帧之间线性插值并发布。
        """
        factor = self._interpolation_factor
        prev_action: np.ndarray | None = None  # 上一个原始动作
        curr_action: np.ndarray | None = None  # 当前原始动作

        next_servo_deadline = time.perf_counter()

        while not self._stop_event.is_set():
            # ── 每 control_dt 从队列弹出一个新动作 ──
            new_action = self._action_queue.get()

            if new_action is not None:
                prev_action = curr_action  # 保存旧的作为插值起点
                curr_action = new_action

                # 日志: 记录原始 (未插值) 动作 (可视化脚本依赖)
                self._write_json_event(
                    "control_action",
                    generation=self._action_queue.generation,
                    control_tick=self._control_tick,
                    queue_remaining=len(self._action_queue),
                    output_action=curr_action,
                )

                # Debug 模式: 按原始控制频率 (非插值频率) 打印动作信息
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
                # 队列空, 等一个 control_dt 再试
                next_servo_deadline += self._control_dt
                self._sleep_until(next_servo_deadline)
                continue

            # ── 内层: factor 个子步, 线性插值并发布 ──
            for sub in range(factor):
                if self._stop_event.is_set():
                    break

                if prev_action is None:
                    # 首个动作, 无法插值 → 直接发送当前动作
                    interp_action = curr_action
                else:
                    # t ∈ (0, 1]:  t=1/F ... F/F
                    # sub=0 → t=1/F (靠近 prev);  sub=F-1 → t=1.0 (到达 curr)
                    t = (sub + 1) / factor

                    # 手臂关节 (前 19 维): 线性插值
                    interp_arm = (
                        prev_action[:ACTION_ARM_DIM] * (1.0 - t)
                        + curr_action[:ACTION_ARM_DIM] * t
                    )

                    # 手部 (后 2 维): 二值, 不插值, 直接使用新值
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
        description="Adam U ROS2 OpenPI 客户端 (RTC 实时分块版)"
    )
    p.add_argument("--host", default="192.168.31.116", help="OpenPI 策略服务器地址。")
    p.add_argument("--port", type=int, default=8000, help="OpenPI 策略服务器端口。")
    p.add_argument("--api-key", default=None, help="可选的 API 密钥。")
    p.add_argument(
        "--prompt", default="fold the white T-shirt", help="发送给策略的自然语言指令。"
    )

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
        help="H: 模型输出的 chunk 总长度 (本项目 H=32)。",
    )
    p.add_argument(
        "--execution-horizon",
        type=int,
        default=DEFAULT_EXECUTION_HORIZON,
        help="s: 每消耗 s 步请求新 chunk (默认 10)。",
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
        help="插值倍率 N: 伺服频率 = control_hz × N。"
        "每对原始动作之间插入 N-2 个中间点 + 两端 = N 个子步。"
        "例: N=10, 50Hz→500Hz, 每 20ms 内发 10 条指令。"
        "手部二值动作不插值。设为 1 则关闭插值。",
    )

    # 图像
    p.add_argument(
        "--render-size",
        type=int,
        default=DEFAULT_RENDER_SIZE,
        help="发送给策略的图像尺寸 (正方形)。",
    )
    p.add_argument(
        "--image-layout",
        choices=("chw", "hwc"),
        default="chw",
        help="图像布局: chw (PyTorch) 或 hwc (TensorFlow)。",
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
    node = OpenPIAdamURTCClient(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        logging.info("正在关闭 RTC 客户端。")
    finally:
        try:
            node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
