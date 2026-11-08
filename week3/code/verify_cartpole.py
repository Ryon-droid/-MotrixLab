"""Exercise unmodified MotrixLab v0.3.0 CartPole. No training or checkpoint is used."""
import argparse
import importlib.metadata
import json
import pathlib
import sys

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source_root", type=pathlib.Path, nargs="?", default=None)
args = parser.parse_args()
from runtime import load_cartpole
registry = load_cartpole(args.source_root)

target = pathlib.Path(__file__).resolve().parents[1] / "evidence"
target.mkdir(exist_ok=True)
np.random.seed(42)
report = {"versions": {k: importlib.metadata.version(k) for k in ["motrixsim", "numpy", "gymnasium"]},
          "training_performed": False, "seed": 42, "checks": {}}

def fresh(num=1, noise=0.0):
    env = registry.make("cartpole", sim_backend="np", num_envs=num,
                        env_cfg_override={"reset_noise_scale": noise})
    env.init_state()
    return env

def set_state(env, x=0.0, theta=0.0, dx=0.0, dtheta=0.0):
    data = env.state.data
    data.reset(env.model)
    data.set_dof_pos(np.tile(np.array([x, theta], dtype=np.float32), (env.num_envs, 1)), env.model)
    data.set_dof_vel(np.tile(np.array([dx, dtheta], dtype=np.float32), (env.num_envs, 1)))

env = fresh()
assert env.cfg.max_episode_steps == 1000
assert env.cfg.sim_substeps == 1
assert env.observation_space.shape == (4,)
assert env.action_space.shape == (1,)
report["checks"]["spaces_and_clock"] = {
    "observation_shape": [4], "action_shape": [1], "action_low": env.action_space.low.tolist(),
    "action_high": env.action_space.high.tolist(), "sim_dt": env.cfg.sim_dt,
    "ctrl_dt": env.cfg.ctrl_dt, "max_episode_seconds": env.cfg.max_episode_seconds,
    "max_episode_steps": env.cfg.max_episode_steps, "sim_substeps": env.cfg.sim_substeps}

set_state(env, x=.11, theta=.02, dx=.33, dtheta=.44)
env.update_state(env.state)
np.testing.assert_allclose(env.state.obs[0], [.11, .02, .33, .44], atol=1e-6)
report["checks"]["observation_order"] = env.state.obs[0].tolist()

cases = [(0.0, 0.2, False), (0.0, -.2, False), (.8, 0., False), (-.8, 0., False),
         (0., .201, True), (0., -.201, True), (.801, 0., True), (-.801, 0., True)]
results = []
for x, theta, expected in cases:
    set_state(env, x, theta)
    env.update_state(env.state)
    actual = bool(env.state.terminated[0])
    assert actual == expected, (x, theta, expected, actual)
    results.append({"x": x, "theta": theta, "terminated": actual})
# Test the explicitly implemented NaN angle branch without passing NaN to the physics solver.
from motrix_envs.basic.cartpole.cartpole_np import CartPoleEnv
from types import SimpleNamespace
nan_state = SimpleNamespace(data=SimpleNamespace(dof_pos=np.array([[0., np.nan]]),
                                                dof_vel=np.array([[0., 0.]])))
CartPoleEnv.update_state(env, nan_state)
assert bool(nan_state.terminated[0])
report["checks"]["failure_boundaries"] = results
report["checks"]["nan_angle_terminates"] = True

# A mathematically upright zero-noise state remains balanced with zero input.
# This is a clock test, not evidence of a learned policy.
env = fresh()
rewards, flags = [], []
for step in range(1, 1001):
    state = env.step(np.zeros((1, 1), dtype=np.float32))
    rewards.append(float(state.reward[0]))
    if step < 1000:
        assert not bool(state.done[0]), step
    else:
        flags = [bool(state.terminated[0]), bool(state.truncated[0])]
        assert flags == [False, True], flags
        assert int(state.info["steps"][0]) == 0  # autoreset already occurred
        np.testing.assert_allclose(state.obs, 0.0, atol=1e-7)
assert rewards == [1.0] * 1000
report["checks"]["time_limit"] = {"steps": 1000, "seconds": 10.0, "return": sum(rewards),
    "terminated": flags[0], "truncated": flags[1], "post_reset_step_counter": 0,
    "returned_observation_is_reset_state": True, "zero_noise_equilibrium_clock_test": True}
state = env.step(np.zeros((1, 1), dtype=np.float32))
assert not bool(state.done[0]) and int(state.info["steps"][0]) == 1
report["checks"]["next_episode_first_step"] = {"done": False, "steps": 1}

# Forced failure: step returns the failure flag, but its obs has already been reset.
env = fresh()
set_state(env, theta=.25)
state = env.step(np.zeros((1, 1), dtype=np.float32))
assert bool(state.terminated[0]) and not bool(state.truncated[0])
assert int(state.info["steps"][0]) == 0
np.testing.assert_allclose(state.obs, 0., atol=1e-7)
report["checks"]["failure_autoreset"] = {"terminated": True, "truncated": False,
    "returned_observation": state.obs[0].tolist(), "counter": 0, "failure_step_reward": float(state.reward[0])}

# Sample many resets; only initial states, not masses/friction/gravity, are randomized.
env = fresh(num=256, noise=.01)
samples = [env.init_state().obs.copy() for _ in range(4)]
samples = np.concatenate(samples)
assert np.all(samples >= -.01) and np.all(samples <= .01)
assert np.ptp(samples, axis=0).min() > .015
report["checks"]["reset_samples"] = {"count": len(samples), "min": samples.min(axis=0).tolist(),
                                       "max": samples.max(axis=0).tolist()}

# Physics equivalence confirms XML control saturation, despite the raw stored control.
vels = {}
for action in [-30., -3., 0., 3., 30.]:
    env = fresh()
    state = env.step(np.array([[action]], dtype=np.float32))
    vels[str(action)] = state.data.dof_vel[0].tolist()
np.testing.assert_allclose(vels["3.0"], vels["30.0"], atol=1e-6)
np.testing.assert_allclose(vels["-3.0"], vels["-30.0"], atol=1e-6)
assert vels["3.0"][0] > 0 and vels["-3.0"][0] < 0
report["checks"]["actuator_saturation"] = {"one_step_dof_vel_by_input": vels,
    "plus_3_equals_plus_30": True, "minus_3_equals_minus_30": True,
    "xml_force_rule": "F = 100 * clip(action, -3, 3) N"}
report["passed"] = True
text = json.dumps(report, indent=2, ensure_ascii=True)
(target / "verification.json").write_text(text + "\n", encoding="utf-8")
print(text)
