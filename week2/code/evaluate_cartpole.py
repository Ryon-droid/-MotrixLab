"""独立加载所提交的权重，实际运行 2000 步并统计完成的回合。"""

import argparse
import json

import numpy as np
import torch
from skrl import config
from skrl.utils import set_seed

from motrix_envs import registry
from motrix_rl.skrl.torch import wrap_env
from motrix_rl.skrl.torch.train.ppo import Trainer, _get_cfg

parser = argparse.ArgumentParser()
parser.add_argument("policy")
args = parser.parse_args()
torch.set_num_threads(2)
config.torch.backend = "torch"
set_seed(42)
trainer = Trainer("cartpole", "np", cfg_override={"play_num_envs": 1, "seed": 42})
raw = registry.make("cartpole", sim_backend="np", num_envs=1)
env = wrap_env(raw, enable_render=False)
cfg = trainer._rlcfg
agent = trainer._make_agent(trainer._make_model(env, cfg), env, _get_cfg(cfg, env))
agent.load(args.policy)
agent.set_running_mode("eval")
print("CHECKPOINT_LOADED", args.policy, flush=True)
obs, _ = env.reset()
returns, lengths = [], []
total, length = 0.0, 0
with torch.no_grad():
    for i in range(2000):
        out = agent.act(obs, timestep=i, timesteps=2000)
        actions = out[-1].get("mean_actions", out[0])
        obs, reward, terminated, truncated, _ = env.step(actions)
        assert torch.isfinite(obs).all()
        total += reward.item()
        length += 1
        if (terminated | truncated).item():
            returns.append(total)
            lengths.append(length)
            total, length = 0.0, 0
result = {"steps": 2000, "completed_episodes": len(returns), "episode_lengths": lengths,
          "episode_returns": returns, "mean_completed_return": float(np.mean(returns)) if returns else None,
          "unfinished_episode_steps": length, "all_observations_finite": True}
print(json.dumps(result, indent=2), flush=True)
assert returns, "至少需要一个完整回合作为证据。"
