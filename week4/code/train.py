"""Finite, auditable training using unmodified v0.3.0 model/agent builders."""
import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import shlex
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch
from runtime import ROOT, bootstrap, make_agent, make_trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=["cartpole", "pendulum", "acrobot"], required=True)
    parser.add_argument("--group", choices=["A", "B"], default="A")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=8192, help="vector steps, per environment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=pathlib.Path, help="New experiment directory for a repeat run")
    args = parser.parse_args()
    if args.group == "B" and args.env != "pendulum":
        parser.error("B applies only to pendulum")
    torch.set_num_threads(1)
    registry = bootstrap(args.group)
    from skrl.trainers.torch import SequentialTrainer
    from skrl.utils import set_seed
    set_seed(args.seed)
    name = args.env + ("_" + args.group if args.env == "pendulum" else "")
    dest = args.output.resolve() if args.output else ROOT / "runs" / name
    if (dest / "manifest.json").exists():
        raise FileExistsError(f"Existing completed experiment: {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    trainer = make_trainer(args.env, args.num_envs, args.steps, args.seed)
    trainer._rlcfg.runner.agent.experiment.experiment_name = f"seed{args.seed}"
    env = registry.make(args.env, sim_backend="np", num_envs=args.num_envs)
    wrapped, agent = make_agent(trainer, env, str(dest))
    info = {
        "command": "python " + shlex.join(["code/train.py", *sys.argv[1:]]),
        "task": args.env, "group": args.group, "seed": args.seed,
        "version": "MotrixLab v0.3.0", "commit": "e1421d10559c825e72bab0f3cbb517c333d67e16",
        "versions": {p: importlib.metadata.version(p) for p in ["torch", "skrl", "motrixsim", "numpy", "gymnasium", "tensorboard"]},
        "python": sys.version, "platform": platform.platform(), "device": str(wrapped.device),
        "vector_steps": args.steps, "num_envs": args.num_envs,
        "environment_transitions": args.steps * args.num_envs,
        "rl_config": dataclasses.asdict(trainer._rlcfg),
        "env_config": dataclasses.asdict(env.cfg),
        "from_scratch": True, "energy_contribution_coefficient": 0 if args.group == "B" else 1,
    }
    (dest / "config.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in info.items() if not k.endswith('config')}, indent=2), flush=True)
    start = time.monotonic()
    runner = SequentialTrainer(cfg={"timesteps": args.steps, "headless": True, "disable_progressbar": True, "close_environment_at_exit": False}, env=wrapped, agents=agent)
    runner.train()
    final = dest / "final_agent.pt"
    agent.save(str(final))
    info.update(elapsed_seconds=time.monotonic()-start, final_checkpoint=os.path.relpath(final, ROOT), final_checkpoint_sha256=hashlib.sha256(final.read_bytes()).hexdigest(), completed=True)
    (dest / "manifest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"TRAINING_COMPLETED task={args.env} group={args.group} seed={args.seed} vector_steps={args.steps} num_envs={args.num_envs} transitions={args.steps*args.num_envs} elapsed_seconds={info['elapsed_seconds']:.2f} checkpoint={os.path.relpath(final, ROOT)}", flush=True)


if __name__ == "__main__":
    main()
