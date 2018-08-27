import gym
from gym.envs.registration import register

import ray
from ray.tune.registry import register_env
from ray.rllib import ars


def env_creator(env_config):
    register(id='LQREnv-v0', entry_point='gym_lqr.envs:LQREnv',
    		 max_episode_steps=10)
    return gym.make("LQREnv-v0")  # or return your own custom env

register_env("my_env", env_creator)
ray.init(num_cpus=1, redirect_output=True)
trainer = ars.ARSAgent(env="my_env", config={
	"num_workers": 1,
    "env_config": {},  # config to pass to env creator
})

while True:
    print(trainer.train())


######### TESTING #############
# import gym 
# from gym.envs.registration import register

# register(id='LQREnv-v0', entry_point='gym_lqr.envs:LQREnv')
# env = gym.make("LQREnv-v0")
# env.reset()
# print(env.step(env.action_space.sample()))