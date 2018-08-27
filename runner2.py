import ray
from ray.tune.registry import register_env
from gym.envs.registration import register
from ray.rllib import ppo

register(id='LQREnv-v0', entry_point='gym_lqr.envs:LQREnv')

def env_creator(env_config):
    import gym
    return gym.make("LQREnv-v0")  # or return your own custom env

register_env("my_env", env_creator)
ray.init()
trainer = ppo.PPOAgent(env="my_env", config={
    "env_config": {},  # config to pass to env creator
})

while True:
    print(trainer.train())