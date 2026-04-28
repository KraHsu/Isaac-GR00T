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
| Action horizon | 16 步 | manipulation 标准 |
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

## 推理（训练完成后）

```bash
python gr00t/eval/run_gr00t_server.py \
    --model-path ./outputs/pi05_finetune/checkpoint-10000 \
    --embodiment-tag new_embodiment
```

客户端用 `PolicyClient` 通过 ZMQ 连接，详见 `gr00t/policy/server_client.py`。

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
