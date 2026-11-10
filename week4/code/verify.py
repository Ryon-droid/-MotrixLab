"""Verify experiment controls, timeout semantics, geometry and source integrity."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from runtime import ROOT, bootstrap

registry = bootstrap()
report = {}
manifest = json.loads((ROOT / "evidence/source_manifest.json").read_text())
for file, expected in manifest["sha256"].items():
    assert hashlib.sha256((ROOT / "source" / file).read_bytes()).hexdigest() == expected
report["upstream_integrity"] = {"verified_files":len(manifest["sha256"]),"commit":manifest["commit"]}
a = (ROOT / "source/motrix_envs/src/motrix_envs/basic/pendulum/pendulum_np.py").read_text()
b = (ROOT / "code/pendulum_no_energy.py").read_text()
assert b == a.replace("+ energy_reward -", "+ 0.0 * energy_reward -")
report["ab_source_change"] = "Exactly one replacement: + energy_reward -> + 0.0 * energy_reward"
ca=json.loads((ROOT / 'runs/pendulum_A/manifest.json').read_text())
cb=json.loads((ROOT / 'runs/pendulum_B/manifest.json').read_text())
assert ca['rl_config'] == cb['rl_config']
assert ca['env_config'] == cb['env_config']
assert ca['num_envs']==ca['rl_config']['num_envs']==128
assert ca['vector_steps']==ca['rl_config']['runner']['trainer']['timesteps']==8192
report['ab_identical_configs']=True
# Inspect both reward implementations at exactly the same physical state.
import importlib.util
spec=importlib.util.spec_from_file_location('motrix_envs.basic.pendulum.check_b',ROOT/'code/pendulum_no_energy.py')
mod=importlib.util.module_from_spec(spec)
# Temporarily free the registration slot solely to inspect the exact B class.
registered = registry._envs['pendulum'].env_cls_dict.pop('np')
spec.loader.exec_module(mod)
registry._envs['pendulum'].env_cls_dict['np'] = registered
from motrix_envs.basic.pendulum.pendulum_np import PendulumEnv
from motrix_envs.basic.pendulum.cfg import PendulumEnvCfg
from motrix_envs.np import reward
fa,fb=PendulumEnv(PendulumEnvCfg(),1),mod.PendulumEnv(PendulumEnvCfg(),1)
sa,sb=fa.init_state(),fb.init_state()
for env,state in [(fa,sa),(fb,sb)]:
 state.data.set_dof_pos(np.array([[0.8]],np.float32),env.model)
 state.data.set_dof_vel(np.array([[0.4]],np.float32))
 state.data.actuator_ctrls=np.array([[0.2]],np.float32)
 state.info['prev_ctrl']=np.zeros(1,np.float32)
 env.model.forward_kinematic(state.data)
ra=float(fa.update_state(sa).reward[0]);rb=float(fb.update_state(sb).reward[0])
energy=0.5*0.4**2+1-np.cos(0.8)
er=float(reward.tolerance(np.array([energy]),bounds=(2,2),margin=2,value_at_margin=.1,sigmoid='gaussian')[0])
assert np.isclose(ra-rb,er,atol=1e-6)
report['ab_reward_difference']={'A':ra,'B':rb,'difference':ra-rb,'computed_energy_reward':er}
# Timeout handling with state held fixed to exclude physical failure.
timeouts={}
for task in ['cartpole','pendulum','acrobot']:
 env=registry.make(task,sim_backend='np',num_envs=1)
 env.init_state()
 # Only bypass physics here; unmodified update_state and episode lifecycle remain.
 env.physics_step=lambda: None
 for i in range(env.cfg.max_episode_steps):
  s=env.step(np.zeros((1,1),np.float32))
 timeouts[task]={'steps':i+1,'terminated':bool(s.terminated[0]),'truncated':bool(s.truncated[0]),'post_reset_step_counter':int(s.info['steps'][0])}
 assert s.truncated[0]
 assert bool(s.terminated[0]) == (task=='acrobot')
 assert int(s.info['steps'][0])==0
report['timeout_flags_physics_held_fixed']=timeouts
# XML zero-angle geometry, using actual link quaternion.
from scipy.spatial.transform import Rotation
up={}
for theta in [0.,np.pi]:
 env=fa;s=env.init_state()
 s.data.set_dof_pos(np.array([[theta]],np.float32),env.model)
 env.model.forward_kinematic(s.data)
 pose=env.model.get_link('pole').get_pose(s.data)[0]
 tip=pose[:3]+Rotation.from_quat(pose[3:]).apply([0,0,.5])
 up[str(theta)]={'pivot_z':float(pose[2]),'tip_z':float(tip[2])}
assert up['0.0']['tip_z']>up['0.0']['pivot_z']
report['pendulum_zero_angle_is_upward']=up
np.random.seed(0)
env=registry.make('dm-reacher',sim_backend='np',num_envs=1024)
s=env.init_state()
assert s.obs.shape==(1024,10)
assert env.model.num_dof_pos==env.model.num_dof_vel==4
assert np.allclose(s.data.dof_pos[:,3],.15)
assert np.allclose(s.data.dof_vel,0)
assert np.all(s.data.dof_pos[:,2]>=-.15) and np.all(s.data.dof_pos[:,2]<=.15)
report['reacher']={'obs_shape':list(s.obs.shape),'num_qpos':4,'num_qvel':4,'joint_names':[j.name for j in env.model.joints],'actuator_names':[a.name for a in env.model.actuators],'sample_count':1024,'target_x_sample_min':float(s.data.dof_pos[:,2].min()),'target_x_sample_max':float(s.data.dof_pos[:,2].max()),'target_y_unique':np.unique(s.data.dof_pos[:,3]).tolist(),'velocity_max_abs':float(abs(s.data.dof_vel).max())}
report['reacher_reward_at_distances']={str(d):float(reward.tolerance(np.array([d]),bounds=(0,.02),margin=.02,value_at_margin=0.,sigmoid='linear')[0]) for d in [0,.01,.02,.03,.04,.05]}
assert report['reacher_reward_at_distances']['0.02']==1.
assert report['reacher_reward_at_distances']['0.04']==0.
report['all_checks_passed']=True
(ROOT/'evidence/verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
