"""A runnable CartPole viewer: --mode balance (LQR baseline) or --mode random.

This week's assignment does not require RL training. The balance mode is a
deterministic engineering baseline computed from local linear dynamics, NOT an
RL policy/checkpoint. The original task's observation, action, reward and reset
logic are unchanged. Close the render window or press Ctrl+C to stop early.
"""
import argparse
import json
import pathlib
import time

import numpy as np
from runtime import load_cartpole

def make_lqr(env):
    """Finite-difference the actual one-step dynamics, then solve discrete LQR."""
    import motrixsim as mtx
    probe = mtx.SceneData(env.model, batch=[1])
    def dynamics(obs, action):
        probe.reset(env.model)
        probe.set_dof_pos(np.array([obs[:2]], dtype=np.float32), env.model)
        probe.set_dof_vel(np.array([obs[2:]], dtype=np.float32))
        probe.actuator_ctrls = np.array([[action]], dtype=np.float32)
        for _ in range(env.cfg.sim_substeps):
            env.model.step(probe)
        return np.concatenate([probe.dof_pos, probe.dof_vel], axis=-1)[0].astype(np.float64)
    eps = .001
    a = np.column_stack([(dynamics(np.eye(4)[i] * eps, 0.) - dynamics(-np.eye(4)[i] * eps, 0.)) / (2 * eps)
                         for i in range(4)])
    b = ((dynamics(np.zeros(4), eps) - dynamics(np.zeros(4), -eps)) / (2 * eps)).reshape(4, 1)
    q, r = np.diag([3., 60., 1., 4.]), np.array([[.4]])
    p = q.copy()
    for _ in range(10000):
        k = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
        updated = q + a.T @ p @ a - a.T @ p @ b @ k
        if np.max(np.abs(updated - p)) < 1e-10:
            p = updated
            break
        p = updated
    return np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["balance", "random"], default="balance")
    parser.add_argument("--headless", action="store_true", help="Run numeric checks without opening a viewer")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", type=pathlib.Path, help="Optional JSON result path")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    np.random.seed(args.seed)
    registry = load_cartpole()
    env = registry.make("cartpole", sim_backend="np", num_envs=1)
    state = env.init_state()
    gain = make_lqr(env) if args.mode == "balance" else None
    viewer = None
    if not args.headless:
        from motrix_envs.np.renderer import NpRenderer
        viewer = NpRenderer(env)
    result = {"mode": args.mode, "seed": args.seed, "rl_training": False, "controller":
              "local-linear LQR engineering baseline" if gain is not None else "uniform random actions",
              "gain": None if gain is None else gain.tolist(), "episodes": []}
    episode_return, episode_steps, completed_steps = 0., 0, 0
    try:
        for _ in range(args.steps):
            started = time.monotonic()
            obs = state.obs.copy()
            if gain is None:
                action = np.random.uniform(-3., 3., (1, 1)).astype(np.float32)
            else:
                action = (-obs @ gain.T).astype(np.float32)
            state = env.step(action)
            assert np.isfinite(state.obs).all()
            episode_return += float(state.reward[0])
            episode_steps += 1
            completed_steps += 1
            if bool(state.done[0]):
                episode = {"steps": episode_steps, "return": episode_return,
                           "terminated": bool(state.terminated[0]), "truncated": bool(state.truncated[0])}
                result["episodes"].append(episode)
                print(json.dumps(episode), flush=True)
                episode_return, episode_steps = 0., 0
            if viewer is not None:
                if viewer._render.is_closed:
                    break
                viewer.render()
                delay = env.cfg.ctrl_dt - (time.monotonic() - started)
                if delay > 0:
                    time.sleep(delay)
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None and not viewer._render.is_closed:
            viewer._render.__exit__(None, None, None)
    result["total_steps"] = completed_steps
    result["unfinished_episode_steps"] = episode_steps
    result["failure_episodes"] = sum(e["terminated"] for e in result["episodes"])
    result["time_limit_episodes"] = sum(e["truncated"] for e in result["episodes"])
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)

if __name__ == "__main__":
    main()
