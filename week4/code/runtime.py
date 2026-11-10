"""Load the vendored task subset without importing unrelated robots or JAX."""
import importlib
import importlib.util
import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"


def namespace(name, path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    sys.modules[name] = module


def bootstrap(group="A"):
    if "motrix_envs" in sys.modules:
        raise RuntimeError("Use a fresh Python process for each run")
    ep = SOURCE / "motrix_envs/src/motrix_envs"
    rp = SOURCE / "motrix_rl/src/motrix_rl"
    namespace("motrix_envs", ep)
    namespace("motrix_envs.basic", ep / "basic")
    for task in ["cartpole", "pendulum", "acrobot", "reacher"]:
        package = f"motrix_envs.basic.{task}"
        namespace(package, ep / "basic" / task)
        importlib.import_module(package + ".cfg")
        name = package + f".{task}_np"
        if task == "pendulum" and group == "B":
            spec = importlib.util.spec_from_file_location(name, ROOT / "code/pendulum_no_energy.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        else:
            importlib.import_module(name)
    namespace("motrix_rl", rp)
    namespace("motrix_rl.tasks", rp / "tasks")
    for task in ["cartpole", "pendulum", "acrobot"]:
        importlib.import_module("motrix_rl.tasks." + task)
    return importlib.import_module("motrix_envs.registry")


def make_trainer(task, num_envs, steps, seed):
    from motrix_rl.skrl.torch.train.ppo import Trainer
    # v0.3.0 cfg_override() reconstructs dataclasses, invoking task __post_init__
    # again. Set the runtime fields afterwards so defaults cannot overwrite them.
    trainer = Trainer(task, sim_backend="np")
    cfg = trainer._rlcfg
    cfg.num_envs = num_envs
    cfg.play_num_envs = 1
    cfg.runner.seed = seed
    cfg.runner.trainer.timesteps = steps
    cfg.runner.agent.experiment.write_interval = 128
    cfg.runner.agent.experiment.checkpoint_interval = 2048
    return trainer



def make_agent(trainer, env, log_dir=None):
    from motrix_rl.skrl.torch import wrap_env
    from motrix_rl.skrl.torch.train.ppo import _add_runtime_config
    wrapped = wrap_env(env, enable_render=False)
    cfg = trainer._rlcfg
    models = trainer._make_model(wrapped, cfg)
    ppo_cfg = cfg.runner.agent.to_dict()
    _add_runtime_config(ppo_cfg, wrapped, log_dir=log_dir)
    agent = trainer._make_agent(models, wrapped, ppo_cfg, cfg.runner.memory)
    return wrapped, agent
