"""把作业代码接入官方 v0.2.0 源码；保留原有文件内容，可重复运行。"""

import argparse
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("motrixlab_root", type=Path)
args = parser.parse_args()
root = args.motrixlab_root.resolve()
src = Path(__file__).resolve().parent
env_root = root / "motrix_envs/src/motrix_envs"
rlcfg = root / "motrix_rl/src/motrix_rl/cfgs.py"
basic_init = env_root / "basic/__init__.py"
if not basic_init.is_file() or not rlcfg.is_file():
    raise SystemExit("请提供包含 motrix_envs、motrix_rl 的官方 v0.2.0 源码根目录。")
shutil.copytree(src / "motrix_envs/basic/simple_walker", env_root / "basic/simple_walker", dirs_exist_ok=True)
text = basic_init.read_text(encoding="utf-8")
line = "from . import simple_walker  # noqa: F401; import for registration"
if line not in text:
    basic_init.write_text(text + "\n" + line + "\n", encoding="utf-8")
text = rlcfg.read_text(encoding="utf-8")
if 'class SimpleWalkerPPOCfg(' not in text:
    rlcfg.write_text(text + "\n\n" + (src / "simple_walker_rlcfg.py").read_text(encoding="utf-8"), encoding="utf-8")
print("Installed simple_walker environment, import chain and RL configuration.")
