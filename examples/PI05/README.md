# PI05 微调指南

在 `pnd_adam_u` 人形机器人布料操作数据集上微调 GR00T N1.7 3B。

## 数据集概览

- **机器人**：pnd_adam_u，21 DOF（腰3 + 颈2 + 双臂14 + 双手2，最后2维为夹爪 0-1）
- **位置**：`/home/charles/workspace/PND/pi05/results/`
- **格式**：LeRobot v2.1
- **总量**：781 episodes，218,065 frames @ 10 fps
- **视频编码**：AV1（540×960，单路 zed_rgb 相机）

| 任务 | episodes | 描述 |
|------|----------|------|
| flatten | 50 | flatten the shirt |
| fold_and_deliver | 191 | fold the shirt and deliver it to the right |
| takeout | 98 | takeout the shirt |
| takeout_and_flatten | 253 | takeout the shirt and flatten it |
| takeout_flatten_fold | 189 | fold the white T-shirt |

## 文件结构

```
examples/PI05/
├── README.md                   # 本文档
├── modality.json               # 数据列 → 模态键映射
├── pi05_config.py              # ModalityConfig 注册（new_embodiment）
├── launch_finetune_pi05.py     # 自定义启动脚本（支持多任务混合）
└── finetune_pi05.sh            # Shell 包装（单卡/多卡）
```

## 关键设计决策

| 项目 | 选择 | 原因 |
|------|------|------|
| Embodiment tag | `new_embodiment` (idx 10) | 已在 `processing_gr00t_n1d7.py:71` 注册 |
| State | 21-dim joint positions | 仅关节位置（含夹爪） |
| Action | 21-dim ABSOLUTE / NON_EEF | 数据本身是绝对关节位置 |
| Action horizon | 32 步 (`delta_indices=list(range(32))`) | 与 `pi05_config.py` 保持一致 |
| 训练策略 | 5 任务混合 | 781 episodes 共享 backbone |
| Tune 范围 | projector + diffusion head | backbone/visual 冻结 |

## 一次性准备（仅首次需要）

将 `modality.json` 复制到每个任务的 `meta/` 目录（LeRobot loader 在那里查找）：

```bash
for task in flatten fold_and_deliver takeout takeout_and_flatten takeout_flatten_fold; do
    cp examples/PI05/modality.json \
       /home/charles/workspace/PND/pi05/results/$task/meta/modality.json
done
```

## 在训练机上运行

### 单卡

```bash
cd /path/to/Isaac-GR00T
BASE_MODEL_PATH=/home/zch/workspace/GR00T-N1.7-3B \
OUTPUT_DIR=./outputs/pi05_finetune \
bash examples/PI05/finetune_pi05.sh
```

### 多卡（推荐）

```bash
NUM_GPUS=8 \
GLOBAL_BATCH_SIZE=32 \
USE_WANDB=1 \
BASE_MODEL_PATH=/home/zch/workspace/GR00T-N1.7-3B \
OUTPUT_DIR=./outputs/pi05_finetune \
bash examples/PI05/finetune_pi05.sh
```

## 可调环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `BASE_MODEL_PATH` | `/home/zch/workspace/GR00T-N1.7-3B` | 预训练 checkpoint 路径 |
| `OUTPUT_DIR` | `./outputs/pi05_finetune` | 输出目录 |
| `NUM_GPUS` | `1` | GPU 数；>1 时启用 torchrun + DeepSpeed ZeRO-2 |
| `GLOBAL_BATCH_SIZE` | `8` | 全局 batch（按 GPU 数自动切分） |
| `MAX_STEPS` | `10000` | 训练步数上限 |
| `SAVE_STEPS` | `1000` | checkpoint 保存间隔 |
| `NUM_WORKERS` | `4` | DataLoader workers |
| `USE_WANDB` | `0` | 1 启用 wandb 日志 |
| `WANDB_PROJECT` | `pi05_finetune` | wandb 项目名 |
| `VIDEO_BACKEND` | `torchcodec` | 训练机用默认；开发机无 torchcodec 可改 `ffmpeg` |
| `MASTER_PORT` | `29500` | torchrun 端口（多卡） |

## 训练机环境准备

GR00T 默认用 `torchcodec` 解码视频。AV1 编码视频需对应 ffmpeg 支持：

```bash
# 安装依赖（dGPU 平台）
bash scripts/deployment/dgpu/install_deps.sh
source .venv/bin/activate

# 验证 torchcodec 可用
python -c "import torchcodec.decoders; print('ok')"

# 验证 ffmpeg 支持 AV1（应看到 libdav1d 解码器）
ffmpeg -codecs 2>/dev/null | grep -i av1
```

若 `torchcodec` 不可用，设 `VIDEO_BACKEND=ffmpeg` 退回基于 subprocess 的 ffmpeg 解码。

## 输出目录结构

```
outputs/pi05_finetune/
├── processor/                  # 处理器（含 statistics.json、embodiment_id.json）
├── checkpoint-1000/            # 周期性 checkpoint
├── checkpoint-2000/
└── ...
```

`processor/` 必须随模型一起保存——推理时 `Gr00tPolicy` 需要它做归一化与 embodiment 索引解析。

## 推理服务端（训练完成后）

GR00T 推理服务通过 **ZeroMQ REQ/REP** + **msgpack** 提供, 默认监听端口 5555。
非 HTTP/WebSocket。在配有 GPU 的训练机或推理机上启动:

```bash
python gr00t/eval/run_gr00t_server.py \
    --model-path ./outputs/pi05_finetune/checkpoint-10000 \
    --embodiment-tag new_embodiment \
    --port 5555
```

启动成功会看到 `Server is ready and listening on tcp://0.0.0.0:5555`。
首次调试时可加 `--strict false` 放宽输入校验。

客户端用 `gr00t.policy.server_client.PolicyClient` 通过 ZMQ 连接,
详见 `gr00t/policy/server_client.py`。

## ROS2 客户端: 两种执行模式

`examples/PI05/` 提供两个并列的 ROS2 客户端, 取决于服务端推理延迟与任务对反应性的要求:

| 文件 | 模式 | 适用场景 |
|------|------|---------|
| [`client_groot.py`](client_groot.py) | **RTC 异步** (frozen prefix + soft-mask blending) | 服务端延迟 < 300ms; 需要高反应性的 manipulation |
| [`client_groot_sync.py`](client_groot_sync.py) | **同步执行** (推理 → 整 chunk 跑完 → 推理) | 服务端延迟 ≥ 500ms; 任务节奏慢; 想要可预测轨迹 |

两个客户端**完全自包含**, 共用相同的: ZMQ + msgpack 协议、ROS 话题、KP/KD、安全限位、
关节滑率限幅 (`--max-arm-velocity`)、JSONL 日志格式、依赖 (仅 `pyzmq + msgpack + numpy`)。

### 决策树

```
服务端推理延迟 (单次 get_action 耗时, 含网络) 是多少?
├─ < 200ms                → RTC 异步 (client_groot.py)            ★ 最佳反应性
├─ 200~500ms 且能调小 s   → RTC 异步 + --execution-horizon 5      ✓ 折中
├─ ≥ 500ms 或难以下调      → 同步执行 (client_groot_sync.py)      ★ 简单可控
└─ 任务节奏极慢/调试录制  → 同步执行                              ★ 可预测
```

如果不确定, 先用 `--debug` 跑一遍 RTC 客户端看 `latency_ema_ms`, 再决定是否切到同步版。

## ROS2 客户端 (RTC)

`examples/PI05/client_groot.py` 是机器人侧的 ROS2 客户端, 复用 Physical
Intelligence 提出的 **Real-Time Chunking (RTC)** 执行模式:

- 推理线程在机器人执行旧 chunk 的同时异步生成下一个 chunk
- 新 chunk 与队列中剩余动作做 **冻结前缀 + 软掩码融合**, 消除 chunk 边界跳变
- 控制线程按固定频率从动作队列弹出动作, 并以 `interpolation_factor`
  倍频率做线性插值后发布给电机 (例: 15Hz × 10 = 150Hz 伺服)

### 依赖 (机器人侧最小集合)

`client_groot.py` **完全自包含**: 它把 ZMQ + msgpack 协议直接内联在文件顶部
(类 `GrootZmqClient`), **不需要安装 gr00t 包**, 也不需要 torch / transformers /
flash-attn 等任何重型依赖。机器人侧 Python 环境只需:

```bash
pip install pyzmq msgpack numpy
```

外加 ROS2 distro 自带的 `rclpy` / `sensor_msgs`, 以及机器人 ROS2 工作空间提供的
`pnd_adam.msg` (`LowState`/`LowCmd`/`HandState`/`HandCmd`)。

把 `examples/PI05/client_groot.py` 单文件拷贝到机器人就能跑, 不必整体克隆
Isaac-GR00T 仓库。

### 启动命令

```bash
# 基本用法 (机器人会真实运动)
python examples/PI05/client_groot.py \
    --host 192.168.31.116 --port 5555 \
    --prompt "fold the white T-shirt"

# 完整参数示例
python examples/PI05/client_groot.py \
    --host 192.168.31.116 --port 5555 \
    --prompt "fold the white T-shirt" \
    --action-horizon 32 --execution-horizon 15 \
    --blend-schedule exp --max-guidance-weight 10.0 \
    --control-hz 15 --interpolation-factor 10
```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `192.168.31.116` | GR00T 推理服务器地址 |
| `--port` | `5555` | GR00T 服务器 ZMQ 端口 |
| `--api-token` | `None` | 可选, 与服务端 `--api-token` 对应 |
| `--timeout-ms` | `15000` | ZMQ 收发超时 (毫秒) |
| `--prompt` | `fold the white T-shirt` | 任务描述 (映射到 `annotation.human.task_description`) |
| `--control-hz` | `15` | 策略动作执行频率 (Hz), Δt = 1/control_hz |
| `--action-horizon` | `32` | H: 模型输出 chunk 总长度, 与 `pi05_config.py` 一致 |
| `--execution-horizon` | `15` | s: 每消耗 s 步触发一次新推理 |
| `--blend-schedule` | `exp` | RTC 软掩码衰减: `exp` / `linear` / `ones` |
| `--max-guidance-weight` | `10.0` | β: 指数衰减权重上限 (论文推荐 5~10) |
| `--interpolation-factor` | `10` | N: 伺服频率 = `control_hz × N`; `1` 关闭插值 |
| `--max-arm-velocity` | `3.0` | **输出层关节滑率限幅 (rad/s)**, 兜底防跳变。每个 servo tick 单关节 \|Δq\| ≤ `max_arm_velocity / servo_hz`; ≤0 关闭 |
| `--render-size` | `0` | 客户端预 letterbox 缩放尺寸; `0` = 不缩放 (推荐) |
| `--debug` | `false` | 启用后跑完整推理但不发布 `lowcmd`/`handcmd`, 用于联调 |
| `--json-log-path` | `logs/groot_rtc_trace_<ts>.jsonl` | JSONL 事件日志路径 |
| `--ros-domain-id` | `1` | ROS_DOMAIN_ID |

ROS 话题默认值 (覆盖请用对应的 `--*-topic` 参数):

| 话题 | 默认 | 方向 |
|------|------|------|
| `--camera-topic` | `/zed/zed_node/rgb/color/rect/image` | 订阅 |
| `--lowstate-topic` | `lowstate` | 订阅 |
| `--handstate-topic` | `handstate` | 订阅 |
| `--lowcmd-topic` | `lowcmd` | 发布 |
| `--handcmd-topic` | `handcmd` | 发布 |

### 输出层兜底防跳变 (slew-rate limiter)

模型偶发输出大幅跳变时, 仅靠 RTC 融合 / 插值不足以兜住。客户端在 `_apply_action`
里加了一道**关节滑率限幅**, 直接卡住每个 servo tick 上单关节相对上次发布值的最大
位移:

```
|Δq_per_tick| ≤ max_arm_velocity / (control_hz × interpolation_factor)
```

默认 `--max-arm-velocity 3.0` rad/s, 在 `control_hz=15 × interp=10 = 150 Hz` 下:
单关节每 tick 最多动 **0.02 rad ≈ 1.15°**, 单 control 帧最多动 **0.2 rad ≈ 11.5°**。
正常 manipulation 远低于此阈值 (训练数据典型 < 0.5 rad/s), 但能彻底吃掉 chunk
首动作的开机瞬跳与异常值。

特性:

- **首次发布会从当前关节状态 (`lowstate`) 锚定起点**, 因此第一帧不会从默认 0 跳到模型目标
- **限幅是闭环式**: 下一个 tick 的"上次发布值"是已被限幅的指令值, 输入持续偏离时输出会以最大斜率追赶, 不会突变也不会过冲
- **触发时会打日志**: 首次触发 WARN, 之后每 5 秒若窗口限幅率 >0.5% 再 WARN 一次, 提示 policy 输出存在跳变
- **手部二值动作不受影响** (slew 只管 19 维手臂)
- **设 `--max-arm-velocity 0` 关闭**(用于 A/B 对比或调试)

调参建议: 看到 `slew limiter active: X%` 频繁出现且 X 较高(如 >20%), 说明限幅已成
为主导因素而非"兜底", 此时:
1. 先确认是否模型本身输出有问题(看 `RTC 推理完成 ... action_range`)
2. 适度放宽 `--max-arm-velocity`(例如 5.0)
3. 或增大 `--interpolation-factor` 让动作分布到更多 tick(每 tick 步长更小)

### Debug 模式

```bash
python examples/PI05/client_groot.py --host <服务器IP> --debug
```

`--debug` 会照常连接服务端、订阅传感器、跑完整 RTC 推理 / 队列 / 融合 / 日志,
但 **不发布** `lowcmd`/`handcmd`。在真正驱动机器人前用它验证:

- 服务端可达 (`已连接 GR00T 服务器: {'status': 'ok', ...}`)
- 三路传感器都到了 (`首次收到 lowstate / handstate / 图像`)
- 推理频率与延迟正常 (`RTC 推理完成 ... infer_ms=... latency_ema_ms=...`)
- 动作幅度合理 (`action_range=[..., ...]`)

### 与 OpenPI 客户端 (`client_openpi.py`) 的差异

`client_openpi.py` 与 `client_groot.py` 共享同一套 RTC 算法、ROS I/O、
KP/KD、安全限位、控制循环插值与 JSONL 日志格式; 仅推理协议层不同:

| 项 | OpenPI (`client_openpi.py`) | GR00T (`client_groot.py`) |
|----|-----------------------------|----------------------------|
| 协议 | WebSocket | ZeroMQ REQ/REP + msgpack |
| 默认端口 | `8000` | `5555` |
| 客户端类 | `openpi_client.websocket_client_policy.WebsocketClientPolicy` | `gr00t.policy.server_client.PolicyClient` |
| 观测格式 | `{"state": (21,), "images": {"cam_high": img}, "prompt": str}` | `{"video": {"zed_rgb": (1,1,H,W,3) uint8}, "state": {"joints": (1,1,21) f32}, "language": {"annotation.human.task_description": [["..."]]}}` |
| 动作返回 | `result["actions"]` shape `(H, dim)` | `action_dict["joints"]` shape `(1, 32, 21)` (取 `[0]` 后接入 RTC 队列) |
| 图像预处理 | 客户端 resize 到 `render_size×render_size` 并可选 CHW | 默认直接发原始 HWC; 服务端 processor 自动 resize |

### JSONL 日志 / 离线可视化

事件 schema 与 OpenPI 客户端一致 (`session_start`, `rtc_inference`,
`control_action`, `session_end`), 只是默认文件名为
`logs/groot_rtc_trace_<时间戳>.jsonl` 且 `session_start.model="gr00t-n1d7-pi05"`。
现有针对 `openpi_rtc_trace_*.jsonl` 的离线可视化脚本可直接复用。

## ROS2 客户端 (同步执行)

`examples/PI05/client_groot_sync.py` 是不带 RTC 的同步执行版客户端, 走最朴素的循环:

```
while True:
    ① 拍下当前 state + image
    ② 调服务端 get_action(obs) 阻塞等推理 (~700ms)
    ③ 把整个 chunk 顺序按 control_hz 执行完 (~2.13s @ H=32, 15Hz)
    ④ 回到 ①
```

推理期间机器人**保持上一帧指令**(在 PD 控制下基本静止), 整 chunk 执行完才进下一次推理。
配置直观, 完全没有 RTC 调参 (没有 `--execution-horizon` / `--blend-schedule` /
`--max-guidance-weight`); 推理延迟再高也只是**整体节奏变慢**, 不会出现硬切跳变。

### 启动命令

```bash
python examples/PI05/client_groot_sync.py \
    --host 192.168.31.116 --port 5555 \
    --prompt "fold the white T-shirt" \
    --action-horizon 32 --interpolation-factor 10
```

### 关键参数 (相对 RTC 客户端的差集)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--action-horizon` | `32` | 每次推理执行的 chunk 步数 |
| `--timeout-ms` | `30000` | ZMQ 超时 (默认 30s, 同步模式宽松一些以容忍高延迟服务端) |
| `--max-arm-velocity` | `3.0` | 输出层关节滑率限幅, 与 RTC 客户端共用 |
| 其它 (`--host` / `--port` / `--prompt` / `--control-hz` / `--interpolation-factor` / `--render-size` / `--debug`) | 同 RTC 客户端 | |

**没有**这些 RTC-only 参数: `--execution-horizon`, `--blend-schedule`, `--max-guidance-weight`。

### 执行节奏可视化 (默认参数)

```
推理   |--700ms--|                                    |--700ms--|
chunk          [───────── 2133ms execute ─────────][───────── 2133ms execute ─────────]
机器人 静止/PD保持        匀速跟踪                          匀速跟踪
                ↑                                    ↑
              首动作                              chunk 切换
              从 state 起步 + slew limit          跨 chunk 线性插值 + slew limit
```

总循环周期 ≈ `H × Δt + inference_latency` = `2.13s + 0.7s ≈ 2.83s`。
**每次推理之间机器人有约 700ms / 25% 的静止时间**, 这是同步模式的本质代价。

减小 `--action-horizon` (例如 `16`) 可缩短停顿占比, 但会增加推理频率与服务端压力:

| H | execute 时长 | 停顿占比 (700ms 推理) |
|---|------------|----------------------|
| 32 | 2.13s | 25% |
| 16 | 1.07s | 40% |
| 8  | 0.53s | 57% |

### 同步模式 vs RTC 模式行为对比

| 行为 | RTC (`client_groot.py`) | 同步 (`client_groot_sync.py`) |
|------|-------------------------|-------------------------------|
| 推理与执行 | **重叠** 进行 | **串行**, 推理时机器人停顿 |
| chunk 边界 | frozen prefix + soft mask 平滑 | 跨 chunk 线性插值 + slew limit |
| 高延迟下表现 | 边界硬切, 需调参 | 节奏变慢但无跳变, **零调参** |
| 反应性 | 高 (每 s 步重看一次画面) | 低 (每 H 步才重看) |
| 配置参数数量 | 多 (s / blend / β / d) | 少 (无) |
| 适合调试 / 录 demo | 一般 (轨迹依赖时序) | 优 (轨迹完全可重现) |

### JSONL 日志

事件类型: `session_start` / `sync_inference` (替代 `rtc_inference`) / `control_action` /
`session_end`。每条 `control_action` 带 `chunk_index` + `step_in_chunk`, 方便离线还原 chunk
切换边界。`session_start.mode = "sync"`, 与 RTC 模式区分。

## 排错

**1. `Video backend 'torchcodec' is not available`**
训练机未装 torchcodec。运行 `bash scripts/deployment/dgpu/install_deps.sh`，或临时设 `VIDEO_BACKEND=ffmpeg`。

**2. `Original key observation.images.zed_rgb not found in feature config`**
`meta/modality.json` 未复制到数据集目录，回到上面的"一次性准备"步骤。

**3. AV1 解码失败 / "Missing Sequence Header"**
opencv 与多数 decord 构建不支持 AV1。用 `VIDEO_BACKEND=ffmpeg` 或装 `torchcodec`。

**4. OOM**
减小 `GLOBAL_BATCH_SIZE`，或加 `--gradient_accumulation_steps` 至 launcher 的 `extra_args`（需修改 `launch_finetune_pi05.py`）。540×960 单路视觉单卡 24G 显存建议 batch ≤ 4。

**5. `Processor must be set before getting datapoints`**
正常的训练流程会在 pipeline.setup() 中设置；这是直接调用 `dataset.get_shard()` 才会出现的。

## 修改要点（如需调整）

- **更换数据子集**：编辑 `launch_finetune_pi05.py:14` 的 `TASKS` 列表
- **改 action horizon**：编辑 `pi05_config.py` 中 `delta_indices=list(range(16))`
- **打开 backbone 微调**：在 `launch_finetune_pi05.py` 把 `tune_llm/tune_visual` 改为 `True`（需要更多显存）
- **改学习率/warmup**：直接编辑 `launch_finetune_pi05.py` 中的 `config.training.*`

## 相关文件

- `gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py:71` — embodiment → projector 索引映射
- `gr00t/configs/data/embodiment_configs.py:202` — `register_modality_config()`
- `gr00t/experiment/experiment.py` — `run()` 主流程
- `gr00t/data/dataset/sharded_single_step_dataset.py` — 数据 sharding 逻辑
