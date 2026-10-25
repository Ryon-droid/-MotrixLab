"""使用官方 Trainer 和 override，执行有界的 CPU 作业实验。"""

import json
import os
from dataclasses import asdict
from datetime import datetime
from zoneinfo import ZoneInfo

# 仅影响本次进程；较小并行数的 MLP 在 CPU 上不需要大量工作线程。
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch
from skrl import config

from motrix_rl.skrl.torch.train.ppo import Trainer

torch.set_num_threads(2)
config.torch.backend = "torch"
overrides = {
    "num_envs": 32,
    "play_num_envs": 1,
    "seed": 42,
    "max_env_steps": 32 * 4096,
    "check_point_interval": 128,
}
trainer = Trainer("cartpole", sim_backend="np", cfg_override=overrides, enable_render=False)
print("START", datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), flush=True)
print("DEVICE", "CPU", "TORCH", torch.__version__, flush=True)
print("RESOLVED_CONFIG", json.dumps(asdict(trainer._rlcfg), indent=2), flush=True)
print("BATCH_STEPS", trainer._rlcfg.max_batch_env_steps, flush=True)
trainer.train()
print("TRAINING_COMPLETED", flush=True)
