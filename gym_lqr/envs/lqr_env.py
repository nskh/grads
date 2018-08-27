import gym
from gym import error, utils
from gym.spaces import Box
from gym.utils import seeding

import numpy as np

START_STATE = np.array([1., 1.])

class LQREnv(gym.Env):

    def __init__(self):
        self.A = np.array([[1.001, 0], [0, -0.6]])
        self.B = B = np.array([[1.], [1.]])

        # assert controllability
        cont = np.hstack([self.B, self.A@self.B])
        assert np.linalg.matrix_rank(cont)==cont.shape[0]

        self.Q = np.array([[1.,0.], [0.,1.]])
        self.R = np.array([1.])

        self.state = None
        self.state_bound = np.array([100., 100.])
        self.u_bound = 100.

        self.horizon = 10
        self.time_step = 0

    @property
    def action_space(self):
        return Box(
            low=-self.u_bound,
            high=self.u_bound,
            shape=(1,),
            dtype=np.float32)

    @property
    def observation_space(self):
        return Box(
            low=-self.state_bound,
            high=self.state_bound,
            shape=(2,),
            dtype=np.float32)

    def step(self, action):
        done = self.time_step >= self.horizon
        self.state = self.A @ self.state + self.B @ action  # Ax+Bu
        loss = self.state.T @ self.Q @ self.state + action.T * self.R * action  # xTQx+uTRu
        reward = -loss


        self.time_step += 1 
        return self.state, reward, done, {}

    def reset(self):
        self.state = START_STATE
        self.time_step = 0
        return self.state
