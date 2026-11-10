"""Deterministic complete episodes; retain terminal physics state before auto-reset."""
import argparse
import hashlib
import json
import os
import pathlib
import shlex
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch
from runtime import ROOT, bootstrap, make_agent, make_trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True, choices=["cartpole", "pendulum", "acrobot"])
    parser.add_argument("--group", default="A", choices=["A", "B"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--policy", type=pathlib.Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    registry = bootstrap(args.group)
    from skrl.utils import set_seed
    name = args.env + ("_" + args.group if args.env == "pendulum" else "")
    manifest = json.loads((ROOT / "runs" / name / "manifest.json").read_text())
    policy = args.policy.resolve() if args.policy else ROOT / manifest["final_checkpoint"]
    out = ROOT / "evidence" / "replay" / name
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in args.seeds:
        set_seed(seed)
        env = registry.make(args.env, sim_backend="np", num_envs=1)
        trainer = make_trainer(args.env, manifest["num_envs"], manifest["vector_steps"], manifest["seed"])
        wrapped, agent = make_agent(trainer, env)
        agent.init({"timesteps": 1})
        agent.load(str(policy))
        agent.set_running_mode("eval")
        set_seed(seed)
        obs, _ = wrapped.reset()
        qpos = [env.state.data.dof_pos.copy()[0]]
        qvel = [env.state.data.dof_vel.copy()[0]]
        rawobs = [env.state.obs.copy()[0]]
        rewards, actions = [], []
        original_update = env.update_state
        def capture(state):
            state = original_update(state)
            qpos.append(state.data.dof_pos.copy()[0])
            qvel.append(state.data.dof_vel.copy()[0])
            rawobs.append(state.obs.copy()[0])
            rewards.append(float(state.reward[0]))
            return state
        env.update_state = capture
        for step in range(1, env.cfg.max_episode_steps + 2):
            with torch.no_grad():
                outputs = agent.act(obs, timestep=step-1, timesteps=env.cfg.max_episode_steps)
                action = outputs[-1].get("mean_actions", outputs[0])
            actions.append(action.cpu().numpy()[0].copy())
            obs, reward, terminated, truncated, _ = wrapped.step(action)
            if bool(terminated.item()) or bool(truncated.item()):
                break
        else:
            raise RuntimeError("No complete episode at configured time limit")
        assert np.isfinite(qpos).all() and np.isfinite(qvel).all() and np.isfinite(rawobs).all()
        np.savez_compressed(out / f"seed{seed}.npz", qpos=qpos, qvel=qvel, obs=rawobs, rewards=rewards, actions=actions, ctrl_dt=env.cfg.ctrl_dt)
        record = {"task": args.env, "group": args.group, "seed": seed, "policy": str(policy.relative_to(ROOT)),
                  "policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(), "deterministic_mean_actions": True,
                  "steps": step, "ctrl_dt": env.cfg.ctrl_dt, "duration_seconds": step*env.cfg.ctrl_dt,
                  "terminated": bool(terminated.item()), "truncated": bool(truncated.item()),
                  "initial_qpos": qpos[0].tolist(), "terminal_qpos_before_reset": qpos[-1].tolist(),
                  "total_reward": sum(rewards), "terminal_state_captured_before_autoreset": True}
        if args.env == "pendulum":
            tilt = np.rad2deg(np.arccos(np.clip(np.cos(np.asarray(qpos)[:, 0]), -1, 1)))
            hit = np.flatnonzero(tilt <= 15)
            record.update(initial_tilt_degrees=float(tilt[0]), terminal_tilt_degrees=float(tilt[-1]),
                          ever_upright=bool(len(hit)), final_upright=bool(tilt[-1]<=15),
                          first_upright_seconds=float(hit[0]*env.cfg.ctrl_dt) if len(hit) else None,
                          upright_fraction=float(np.mean(tilt[1:]<=15)),
                          minimum_tilt_degrees=float(tilt.min()))
        records.append(record)
        print(json.dumps(record), flush=True)
    (out / "episodes.json").write_text(json.dumps({"command": "python " + shlex.join(["code/evaluate.py", *sys.argv[1:]]), "episodes": records}, indent=2), encoding="utf-8")
    print("EVALUATION_COMPLETED", name, flush=True)


if __name__ == "__main__":
    main()
