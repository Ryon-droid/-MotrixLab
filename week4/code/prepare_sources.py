"""Vendor the exact v0.3.0 subset from a downloaded official commit archive."""
import argparse
import hashlib
import json
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[1]
COMMIT = "e1421d10559c825e72bab0f3cbb517c333d67e16"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=pathlib.Path)
    args = parser.parse_args()
    dest = ROOT / "source"
    env = "motrix_envs/src/motrix_envs/"
    rl = "motrix_rl/src/motrix_rl/"
    files = ["LICENSE"]
    files += [env + p for p in ["base.py", "registry.py", "np/env.py", "np/reward.py", "np/renderer.py"]]
    for task in ["cartpole", "pendulum", "acrobot", "reacher"]:
        files += [str(p.relative_to(args.upstream)) for p in (args.upstream / env / "basic" / task).iterdir() if p.suffix in [".py", ".xml"]]
    files += [env + "common/" + p for p in ["visual.xml", "skybox.xml", "materials.xml", "motphys-ground.png"]]
    files += [rl + p for p in ["registry.py", "utils.py", "rslrl/cfg.py", "skrl/__init__.py", "skrl/config.py", "skrl/torch/__init__.py", "skrl/torch/wrap_np.py", "skrl/torch/train/ppo.py", "tasks/cartpole.py", "tasks/pendulum.py", "tasks/acrobot.py"]]
    manifest = {"repository": "https://github.com/Motphys/MotrixLab", "version": "v0.3.0", "commit": COMMIT, "sha256": {}}
    for file in files:
        target = dest / file
        target.parent.mkdir(parents=True, exist_ok=True)
        src = args.upstream / file
        if file.endswith("motphys-ground.png"):
            expected = src.read_text().split("oid sha256:")[1].splitlines()[0]
            texture = ROOT.parent / "week3/source" / file
            assert hashlib.sha256(texture.read_bytes()).hexdigest() == expected
            shutil.copyfile(texture, target)
        else:
            shutil.copyfile(src, target)
        manifest["sha256"][file] = hashlib.sha256(target.read_bytes()).hexdigest()
    (dest / "NOTICE").write_text(f"MotrixLab v0.3.0, commit {COMMIT}.\nCopyright (C) 2020-2025 Motphys Technology Co., Ltd. Apache-2.0.\nUnmodified minimal environment and Torch PPO source snapshot.\nGround texture restored from the identical verified Git LFS object.\nThe B-group single-line modification is outside this snapshot: code/pendulum_no_energy.py.\n", encoding="utf-8")
    (ROOT / "evidence/source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Vendored {len(files)} exact upstream files")


if __name__ == "__main__":
    main()
