"""第 6 讲：可运行的 NumPy 环境，复用官方 Walker 动力学与奖励。"""

import numpy as np

from motrix_envs import registry
from motrix_envs.basic.walker.walker_np import Walker2DEnv

from .cfg import SimpleWalkerEnvCfg


@registry.env("simple_walker", sim_backend="np")
class SimpleWalkerEnv(Walker2DEnv):
    """Walker2DEnv 继承 NpEnv；三个必需方法在这里明确列出。"""

    _cfg: SimpleWalkerEnvCfg

    def apply_action(self, actions, state):
        # 官方实现将 actions 写入 state.data.actuator_ctrls。
        return super().apply_action(actions, state)

    def update_state(self, state):
        # 官方实现根据 stand_height、move_speed 计算观测、奖励和终止。
        return super().update_state(state)

    def reset(self, data):
        data.reset(self._model)
        n = data.shape[0]
        scale = self._cfg.reset_noise_scale
        qpos = np.tile(self._model.compute_init_dof_pos(), (n, 1))
        qpos += np.random.uniform(-scale, scale, size=qpos.shape)
        qvel = np.random.uniform(-scale, scale, size=(n, self._model.num_dof_vel))
        data.set_dof_pos(qpos, self._model)
        data.set_dof_vel(qvel)
        self._model.forward_kinematic(data)
        rewards = {key: np.zeros(n) for key in ("height", "upright", "stand")}
        if self._move_speed > 0:
            rewards["move"] = np.zeros(n)
        return self._get_obs(data), {"Reward": rewards}
