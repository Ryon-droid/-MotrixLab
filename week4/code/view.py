"""Open the real v0.3.0 environment in the MotrixRender viewer."""
import argparse
import time

import numpy as np
from runtime import bootstrap

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--env',default='dm-reacher',choices=['dm-reacher','cartpole','pendulum','acrobot'])
parser.add_argument('--seed',type=int,default=0)
parser.add_argument('--seconds',type=float,default=30)
args=parser.parse_args()
np.random.seed(args.seed)
registry=bootstrap()
from motrix_envs.np.renderer import NpRenderer
env=registry.make(args.env,sim_backend='np',num_envs=1)
env.init_state()
env.model.forward_kinematic(env.state.data)
renderer=NpRenderer(env)
end=time.monotonic()+args.seconds
while time.monotonic()<end:
 renderer.render()
 time.sleep(1/60)
print('VIEW_COMPLETED',args.env)
