import gym
from gym.envs.registration import register
import gym_lqr

import ray
import ray.rllib.ppo as ppo
from ray.tune import run_experiments
from ray.tune.registry import register_env

def env_creator(env_config):
    register(id='LQREnv-v0', entry_point='gym_lqr.envs:LQREnv')
    return gym.make("LQREnv-v0")  # or return your own custom env

if __name__ == "__main__":
    ray.init(num_cpus=2, redirect_output=True)

    config = ppo.DEFAULT_CONFIG.copy()
    config["num_workers"] = 1
    config["timesteps_per_batch"] = 1000
    config["gamma"] = 0.999  # discount rate
    config["model"].update({"fcnet_hiddens": [16, 16]})
    config["use_gae"] = True
    config["lambda"] = 0.97
    config["sgd_batchsize"] = min(16 * 1024, config["timesteps_per_batch"])
    config["kl_target"] = 0.02
    config["num_sgd_iter"] = 10
    config["horizon"] = 10

    env_name = 'LQREnv-v0'
    
    # Register as rllib env
    register_env(env_name, env_creator)

    trials = run_experiments({
        'lqr_env_test': {
            "run": "PPO",
            "env": env_name,
            "config": {
                **config
            },
            "checkpoint_freq": 20,
            "max_failures": 999,
            "stop": {
                "training_iteration": 200,
            },
        },
    })
