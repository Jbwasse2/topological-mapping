import faulthandler
import pdb

import habitat

faulthandler.enable()


# Load embodied AI task (PointNav) and a pre-specified virtual robot
env = habitat.Env(config=habitat.get_config("configs/tasks/pointnav.yaml"))

observations = env.reset()

# Step through environment with random actions
while not env.episode_over:
    pdb.set_trace()
    observations = env.step(env.action_space.sample())
