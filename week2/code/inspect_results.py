"""从原始 TensorBoard events 导出曲线、CSV 和可核验的汇总。"""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root = Path(__file__).resolve().parents[1]
run = next((root / "runs/cartpole").glob("*_PPO"))
events = next(run.glob("events.out.tfevents.*"))
ea = EventAccumulator(str(events), size_guidance={"scalars": 0}).Reload()
data = {tag: ea.Scalars(tag) for tag in ea.Tags()["scalars"]}
with (root / "evidence/scalars.csv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["tag", "step", "wall_time", "value"])
    for tag, points in data.items():
        writer.writerows((tag, p.step, p.wall_time, p.value) for p in points)

tags = ["Reward / Total reward (mean)", "Episode / Total timesteps (mean)",
        "Loss / Policy loss", "Loss / Value loss"]
fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
for ax, tag in zip(axes.flat, tags):
    points = data[tag]
    ax.plot([p.step for p in points], [p.value for p in points], color="#2672c9", lw=1.8)
    ax.set_title(tag, fontsize=11)
    ax.set_xlabel("Vector environment step (32 parallel environments)")
    ax.grid(alpha=0.25)
fig.suptitle("CartPole PPO - original TensorBoard scalar data (no smoothing)", fontsize=15)
fig.savefig(root / "evidence/training_curves.png", dpi=160)
plt.close(fig)

summary = {}
for tag, points in data.items():
    values = [p.value for p in points]
    summary[tag] = {"count": len(points), "first_step": points[0].step,
                    "last_step": points[-1].step, "first": values[0], "last": values[-1],
                    "min": min(values), "max": max(values)}
manifest = {str(p.relative_to(root)).replace("\\", "/"): {
    "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    for p in sorted((root / "runs/cartpole").rglob("*")) if p.is_file()}
(root / "evidence/scalar_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(root / "evidence/run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps({t: summary[t] for t in tags}, indent=2))
