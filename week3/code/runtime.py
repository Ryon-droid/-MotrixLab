"""Load only the exact vendored CartPole source; no full MotrixLab installation needed.

The namespace setup avoids importing unrelated robots through upstream __init__.py.
Environment classes, configuration, registry and renderer are unmodified upstream files.
"""
import importlib
import pathlib
import sys
import types

def load_cartpole(source_root=None):
    root = pathlib.Path(source_root) if source_root else pathlib.Path(__file__).resolve().parents[1] / "source"
    package = root.resolve() / "motrix_envs/src/motrix_envs"
    if "motrix_envs" in sys.modules:
        raise RuntimeError("Run this script in a fresh Python process to avoid mixing MotrixLab versions")
    for name, path in [
        ("motrix_envs", package),
        ("motrix_envs.basic", package / "basic"),
        ("motrix_envs.basic.cartpole", package / "basic/cartpole"),
    ]:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        module.__package__ = name
        sys.modules[name] = module
    importlib.import_module("motrix_envs.basic.cartpole.cfg")
    importlib.import_module("motrix_envs.basic.cartpole.cartpole_np")
    return importlib.import_module("motrix_envs.registry")
