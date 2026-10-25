"""将此文件内容添加到官方 motrix_rl/src/motrix_rl/cfgs.py 末尾。"""

from dataclasses import dataclass

from motrix_rl.registry import rlcfg
from motrix_rl.skrl.cfg import PPOCfg


@rlcfg("simple_walker")
@dataclass
class SimpleWalkerPPOCfg(PPOCfg):
    seed: int = 42
    num_envs: int = 32
    play_num_envs: int = 1
    max_env_steps: int = 32 * 1024
    check_point_interval: int = 128
    policy_hidden_layer_sizes: tuple[int, ...] = (32, 32)
    value_hidden_layer_sizes: tuple[int, ...] = (32, 32)
    rollouts: int = 32
    learning_epochs: int = 4
    mini_batches: int = 4
    learning_rate: float = 3e-4
