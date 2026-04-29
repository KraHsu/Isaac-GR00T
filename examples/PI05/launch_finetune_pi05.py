#!/usr/bin/env python3
import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
importlib.import_module("examples.PI05.pi05_config")

from gr00t.configs.base_config import get_default_config
from gr00t.experiment.experiment import run

DATA_ROOT = Path("/home/zch/workspace/results/")
TASKS = ["flatten", "fold_and_deliver", "takeout", "takeout_and_flatten", "takeout_flatten_fold"]

BASE_MODEL = os.environ.get("BASE_MODEL_PATH", "/home/zch/workspace/GR00T-N1.7-3B")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./outputs/pi05_finetune")

config = get_default_config().load_dict({
    "data": {
        "download_cache": False,
        "datasets": [
            {
                "dataset_paths": [str(DATA_ROOT / task) for task in TASKS],
                "mix_ratio": 1.0,
                "embodiment_tag": "new_embodiment",
            }
        ],
    }
})
config.load_config_path = None

config.model.tune_llm = True
config.model.tune_visual = True
config.model.tune_projector = True
config.model.tune_diffusion_model = True
config.model.state_dropout_prob = 0.2
config.model.load_bf16 = False
config.model.reproject_vision = False
config.model.model_name = "nvidia/Cosmos-Reason2-2B"
config.model.backbone_trainable_params_fp32 = True
config.model.use_relative_action = True
config.model.action_horizon = 64
config.model.color_jitter_params = {
    "brightness": 0.3, "contrast": 0.4, "saturation": 0.5, "hue": 0.08
}

config.training.start_from_checkpoint = BASE_MODEL
config.training.output_dir = OUTPUT_DIR
config.training.experiment_name = "pi05_mixed"
config.training.optim = "adamw_torch"
config.training.global_batch_size = int(os.environ.get("GLOBAL_BATCH_SIZE", "8"))
config.training.dataloader_num_workers = int(os.environ.get("NUM_WORKERS", "4"))
config.training.learning_rate = 1e-4
config.training.weight_decay = 1e-5
config.training.warmup_ratio = 0.05
config.training.max_steps = int(os.environ.get("MAX_STEPS", "80000"))
config.training.save_steps = int(os.environ.get("SAVE_STEPS", "10000"))
config.training.save_total_limit = 5
config.training.num_gpus = int(os.environ.get("NUM_GPUS", "1"))
config.training.use_wandb = os.environ.get("USE_WANDB", "0") == "1"
config.training.wandb_project = os.environ.get("WANDB_PROJECT", "pi05_finetune")
config.training.save_only_model = True

config.data.shard_size = 1024
config.data.episode_sampling_rate = 0.1
config.data.num_shards_per_epoch = 100000
config.data.video_backend = os.environ.get("VIDEO_BACKEND", "torchcodec")

run(config)
