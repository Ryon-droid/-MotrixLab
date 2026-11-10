"""Render recorded real MotrixSim trajectories with the unchanged official XML.

Adds only a fixed camera; no physics is re-run, and the terminal pose is retained.
All frame indices and camera parameters are written to capture_manifest.json.
"""
import argparse
import json
import time

import imageio.v2 as imageio
import motrixsim as mtx
import numpy as np
from motrixsim import msd
from motrixsim.render import RenderApp, RenderSettings
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
from runtime import ROOT, SOURCE

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
CNFONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def camera_scene(task):
    xml = SOURCE / f"motrix_envs/src/motrix_envs/basic/{task}/{task}.xml"
    scene = msd.from_file(str(xml))
    if task == "pendulum":
        eye, look, fov = [0, -2.5, .65], [0, 0, .65], 38.
    elif task == "acrobot":
        eye, look, fov = [0, -9, 2], [0, 0, 2], 39.
    elif task == "reacher":
        eye, look, fov = [0, 0, .65], [0, .025, 0], 44.
    else:
        eye, look, fov = [1.35, -2.8, 1.4], [0, 0, .25], 38.
    eye, look = np.asarray(eye, float), np.asarray(look, float)
    forward = look - eye
    forward /= np.linalg.norm(forward)
    upref = np.array([0,1,0]) if task == "reacher" else np.array([0,0,1])
    right = np.cross(forward, upref)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    quat = Rotation.from_matrix(np.column_stack([right, up, -forward])).as_quat().tolist()
    scene.add_camera(msd.Camera(name="week4_fixed_camera", position=eye.tolist(), orientation=quat,
                               position_track=msd.PositionTrack.FixedWorld, rotation_track=msd.RotationTrack.FixedWorld, fovy=fov))
    model = scene.build()
    cam = model.cameras["week4_fixed_camera"]
    cam.set_render_target("image", 960, 640)
    return model, cam, dict(eye=eye.tolist(), look=look.tolist(), fovy=fov, quaternion_xyzw=quat, width=960, height=640), (eye,forward,right,up,fov)


def take(render, rc, data):
    render.sync(data, wait=True)
    task = rc.capture()
    render.sync(data, wait=True)
    limit = time.monotonic() + 20
    while task.state == "pending" and time.monotonic() < limit:
        render.sync(data, wait=True)
    result = task.take_image()
    if result is None:
        raise RuntimeError(f"Camera capture failed: {task.state}")
    return Image.fromarray(result.pixels.copy()).convert("RGB")


def header(raw, title, detail):
    out = Image.new("RGB", (960, 736), "#10283e")
    out.paste(raw, (0, 96))
    draw = ImageDraw.Draw(out)
    draw.text((22,10), title, font=ImageFont.truetype(FONT,25), fill="white")
    draw.text((22,47), detail, font=ImageFont.truetype(FONT,20), fill="#afe7dd")
    draw.text((15,714), "MotrixSim 0.7.0 | learned PPO policy | full episode, fixed camera", font=ImageFont.truetype(FONT,15), fill="#ffffff", stroke_width=1, stroke_fill="#10283e")
    return out


def render_task(name):
    task = "pendulum" if name.startswith("pendulum") else name
    replay = ROOT / "evidence/replay" / name
    entries = json.loads((replay / "episodes.json").read_text())["episodes"]
    model, cam, camera, _ = camera_scene(task)
    data = mtx.SceneData(model, batch=[1])
    manifest = {"type": "official engine render from recorded policy trajectories", "camera_added_only": True, "camera": camera, "episodes": []}
    with RenderApp(headless=True) as render:
        render.launch(model, render_settings=RenderSettings.performance())
        rc = render.get_camera(cam.index)
        for _ in range(70):
            render.sync(data)
            time.sleep(.015)
        for entry in entries:
            seed = entry["seed"]
            record = np.load(replay / f"seed{seed}.npz")
            qpos, qvel = record["qpos"], record["qvel"]
            stride = {"pendulum":2, "acrobot":2, "cartpole":4}[task]
            indices = list(range(0, len(qpos), stride))
            if indices[-1] != len(qpos)-1:
                indices.append(len(qpos)-1)
            fps = 1 / (entry["ctrl_dt"]*stride)
            dest = ROOT / "videos" / f"{name}_seed{seed}.mp4"
            dest.parent.mkdir(exist_ok=True)
            with imageio.get_writer(str(dest), fps=fps, codec="libx264", quality=8, macro_block_size=16, ffmpeg_params=["-movflags", "+faststart"]) as writer:
                for index in indices:
                    data.set_dof_pos(qpos[index:index+1], model)
                    data.set_dof_vel(qvel[index:index+1])
                    model.forward_kinematic(data)
                    raw = take(render, rc, data)
                    tilt = float(np.rad2deg(np.arccos(np.clip(np.cos(qpos[index,0]),-1,1)))) if task == "pendulum" else None
                    detail = f"seed={seed}   t={index*entry['ctrl_dt']:.3f}s   step={index}/{entry['steps']}"
                    if tilt is not None:
                        detail += f"   tilt={tilt:.2f} deg   upright <=15 deg"
                    if index == len(qpos)-1:
                        detail += "   END"
                    frame = header(raw, f"Week4 | {name} | full episode replay", detail)
                    writer.append_data(np.asarray(frame))
                    if index in (0,len(qpos)-1):
                        label = "initial" if index == 0 else "terminal"
                        raw.save(replay / f"seed{seed}_{label}_raw.png")
                        frame.save(replay / f"seed{seed}_{label}.png")
            manifest["episodes"].append(dict(seed=seed, file=str(dest.relative_to(ROOT)), fps=fps, frame_indices=indices, steps=entry["steps"], terminal_pose_before_reset=True))
            print("VIDEO_COMPLETED", dest.name, "frames", len(indices), flush=True)
    (replay / "capture_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def reacher_scene():
    # Run upstream reset once and render the same real task state.
    from runtime import bootstrap
    registry = bootstrap()
    np.random.seed(0)
    env = registry.make("dm-reacher", sim_backend="np", num_envs=1)
    env.init_state()
    env.model.forward_kinematic(env.state.data)
    model, cam, camera, projection = camera_scene("reacher")
    data = mtx.SceneData(model,batch=[1])
    data.set_dof_pos(env.state.data.dof_pos,model)
    data.set_dof_vel(env.state.data.dof_vel)
    model.forward_kinematic(data)
    with RenderApp(headless=True) as render:
        render.launch(model,render_settings=RenderSettings.performance())
        rc=render.get_camera(cam.index)
        for _ in range(80):
            render.sync(data);time.sleep(.02)
        raw=take(render,rc,data)
    out=ROOT / "evidence/reacher"
    out.mkdir(parents=True,exist_ok=True)
    raw.save(out / "scene_raw.png")
    annotated=raw.copy()
    draw=ImageDraw.Draw(annotated)
    eye,forward,right,up,fov=projection
    def project(pos):
        v=np.array(pos[:3])-eye
        focal=320/np.tan(np.deg2rad(fov)/2)
        return (480+focal*np.dot(v,right)/np.dot(v,forward),320-focal*np.dot(v,up)/np.dot(v,forward))
    shoulder=model.get_link("arm").get_pose(data)[0,:3]
    wrist=model.get_link("hand").get_pose(data)[0,:3]
    finger=model.get_link("finger").get_pose(data)[0,:3]
    target=env._target_xyz[0,:3]
    labels=[("肩关节 shoulder",shoulder,(40,420)),("第一段连杆 arm",(shoulder+wrist)/2,(40,540)),
            ("腕关节 wrist",wrist,(610,450)),("第二段连杆 hand",(wrist+finger)/2,(610,330)),
            ("目标点 target",target,(40,80)),("末端 finger",finger,(610,130))]
    for label,point,where in labels:
        x,y=project(point)
        draw.ellipse((x-6,y-6,x+6,y+6),outline="#ffcb48",width=3)
        draw.line([(x,y),(where[0]+100,where[1]+20)],fill="#ffcb48",width=3)
        box=draw.textbbox(where,label,font=ImageFont.truetype(CNFONT,22))
        draw.rectangle((box[0]-5,box[1]-3,box[2]+5,box[3]+3),fill="#10283e")
        draw.text(where,label,font=ImageFont.truetype(CNFONT,22),fill="white")
    annotated.save(out / "scene_annotated.png")
    (out / "scene.json").write_text(json.dumps(dict(camera=camera,seed=0,qpos=env.state.data.dof_pos.tolist(),qvel=env.state.data.dof_vel.tolist(),obs=env.state.obs.tolist(),target_xyz=target.tolist()),indent=2),encoding="utf-8")
    print("REACHER_SCENE_COMPLETED",flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task",choices=["cartpole","pendulum_A","pendulum_B","acrobot","reacher"],required=True)
    args=parser.parse_args()
    if args.task == "reacher":
        reacher_scene()
    else:
        render_task(args.task)
