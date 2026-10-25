"""第 6 讲：复用内置 Walker 模型，注册一个独立任务配置。"""

from dataclasses import dataclass

from motrix_envs import registry
from motrix_envs.base import EnvCfg
from motrix_envs.basic.walker.cfg import WalkerEnvCfg


@registry.envcfg("simple_walker")
@dataclass
class SimpleWalkerEnvCfg(EnvCfg):
    # 复用官方模型及其相对引用的资源，不需要另外下载机器人网格。
    model_file: str = WalkerEnvCfg().model_file
    sim_dt: float = 0.0125
    ctrl_dt: float = 0.025
    max_episode_seconds: float = 10.0
    render_spacing: float = 2.0

    # 任务特有字段 1：站立奖励的目标躯干高度，单位 m。
    stand_height: float = 1.2
    # 任务特有字段 2：向前移动奖励的目标速度，单位 m/s。
    move_speed: float = 1.0
    # 任务特有字段 3：重置时关节位置和速度的均匀扰动幅度。
    reset_noise_scale: float = 0.01
