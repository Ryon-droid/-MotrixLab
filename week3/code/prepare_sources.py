"""Save the exact upstream Git blobs used in this assignment (not a full checkout)."""
import argparse
import hashlib
import json
import pathlib
import subprocess

PIN = "e1421d10559c825e72bab0f3cbb517c333d67e16"
FILES = [
    "LICENSE", "NOTICE",
    "motrix_envs/src/motrix_envs/base.py",
    "motrix_envs/src/motrix_envs/registry.py",
    "motrix_envs/src/motrix_envs/np/env.py",
    "motrix_envs/src/motrix_envs/np/renderer.py",
    "motrix_envs/src/motrix_envs/basic/cartpole/cartpole_np.py",
    "motrix_envs/src/motrix_envs/basic/cartpole/cfg.py",
    "motrix_envs/src/motrix_envs/basic/cartpole/cartpole.xml",
    "motrix_rl/src/motrix_rl/tasks/cartpole.py",
    "motrix_rl/src/motrix_rl/skrl/torch/wrap_np.py",
    "motrix_rl/src/motrix_rl/skrl/torch/train/ppo.py",
]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=pathlib.Path)
    args = parser.parse_args()
    target = pathlib.Path(__file__).resolve().parents[1]
    manifest = {"repository": "https://github.com/Motphys/MotrixLab", "tag": "v0.3.0",
                "commit": PIN, "files": {}}
    for name in FILES:
        data = subprocess.check_output(["git", "-C", str(args.source_root), "show", f"{PIN}:{name}"])
        path = target / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        manifest["files"][name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    texture_name = "motrix_envs/src/motrix_envs/common/motphys-ground.png"
    texture = (args.source_root / texture_name).read_bytes()
    assert hashlib.sha256(texture).hexdigest() == "c1ed597194fe5c3cf8b46286568a9520be2af547b1f6be219c439956fdbdbfaf"
    texture_path = target / "source" / texture_name
    texture_path.parent.mkdir(parents=True, exist_ok=True)
    texture_path.write_bytes(texture)
    manifest["files"][texture_name] = {"sha256": hashlib.sha256(texture).hexdigest(), "bytes": len(texture)}
    (target / "evidence").mkdir(exist_ok=True)
    (target / "evidence/source_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(FILES)} exact upstream blobs at {PIN}")

if __name__ == "__main__":
    main()
