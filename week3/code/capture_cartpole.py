"""Capture official MotrixSim camera images, not desktop/IndWork screenshots.

Only a camera is added to the original XML scene in memory. The physical model,
joint limits, and motor settings are unchanged. The three states are deliberately
set to illustrate thresholds; they are not results of a trained controller.
"""
import argparse
import json
import pathlib
import time

import motrixsim as mtx
import numpy as np
from motrixsim import msd
from motrixsim.render import RenderApp, RenderSettings
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source_root", type=pathlib.Path, nargs="?", default=pathlib.Path(__file__).resolve().parents[1] / "source")
args = parser.parse_args()
out = pathlib.Path(__file__).resolve().parents[1] / "evidence"
xml = args.source_root.resolve() / "motrix_envs/src/motrix_envs/basic/cartpole/cartpole.xml"
scene = msd.from_file(str(xml))
eye = np.array([1.35, -2.8, 1.4])
forward = np.array([0., 0., .25]) - eye
forward /= np.linalg.norm(forward)
right = np.cross(forward, [0., 0., 1.])
right /= np.linalg.norm(right)
up = np.cross(right, forward)
rotation = np.column_stack([right, up, -forward])
# This view has a positive matrix trace; quaternion order is x,y,z,w.
s = np.sqrt(np.trace(rotation) + 1.) * 2.
quat = [(rotation[2, 1] - rotation[1, 2]) / s,
        (rotation[0, 2] - rotation[2, 0]) / s,
        (rotation[1, 0] - rotation[0, 1]) / s, s / 4.]
scene.add_camera(msd.Camera(name="assignment_camera", position=eye.tolist(), orientation=quat,
                           position_track=msd.PositionTrack.FixedWorld,
                           rotation_track=msd.RotationTrack.FixedWorld, fovy=38.))
model = scene.build()
camera = model.cameras["assignment_camera"]
camera.set_render_target("image", 1280, 800)
data = mtx.SceneData(model, batch=[1])
font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 25)
states = [("simulation_upright.png", 0., 0., "Upright / reference state"),
          ("simulation_near_limit.png", .60, .18, "Within limits / manually set state"),
          ("simulation_failed.png", .0, .25, "Angle failure / manually set state")]
manifest = {"type": "MotrixSim 0.7.0 offscreen camera capture", "desktop_screenshot": False,
            "indwork_screenshot": False, "trained_policy": False,
            "physical_model_modified": False, "camera_added_in_memory": True, "states": []}
with RenderApp(headless=True) as render:
    render.launch(model, render_settings=RenderSettings.performance())
    rc = render.get_camera(camera.index)
    for name, x, theta, title in states:
        data.reset(model)
        data.set_dof_pos(np.array([[x, theta]], dtype=np.float32), model)
        data.set_dof_vel(np.zeros((1, 2), dtype=np.float32))
        model.forward_kinematic(data)
        for _ in range(90):
            render.sync(data)
            time.sleep(.025)  # Allow asynchronous textures and meshes to finish loading.
        task = rc.capture()
        render.sync(data, wait=True)
        deadline = time.monotonic() + 20
        while task.state == "pending" and time.monotonic() < deadline:
            render.sync(data, wait=True)
        img = task.take_image()
        if img is None:
            raise RuntimeError(f"No camera image for {name}; state={task.state}")
        raw_name = name.replace(".png", "_raw.png")
        img.save_to_disk(str(out / raw_name))
        raw = Image.open(out / raw_name).convert("RGB")
        canvas = Image.new("RGB", (raw.width, raw.height + 130), "#102135")
        canvas.paste(raw, (0, 130))
        draw = ImageDraw.Draw(canvas)
        draw.text((28, 14), title, font=font, fill="white")
        draw.text((28, 53), f"x={x:.2f} m   theta={theta:.2f} rad   |theta|<=0.20, |x|<=0.80", font=font, fill="#9fe2d1")
        draw.text((28, 92), "Official engine capture | v0.3.0 XML + camera | NOT an IndWork screenshot", font=ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 20), fill="#bacada")
        canvas.save(out / name)
        manifest["states"].append({"file": name, "raw": raw_name, "x_m": x, "theta_rad": theta,
                                   "terminated": abs(theta) > .2 or abs(x) > .8})
(out / "capture_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))
