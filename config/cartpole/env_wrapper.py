import logging

import numpy as np
from core.game import Game


class CartPoleWrapper(Game):
    def __init__(self, env, discount: float, **kwargs):
        """CartPole Wrapper
        Parameters
        ----------
        env: Any
            another env wrapper
        discount: float
            discount of env
        """
        super().__init__(env, env.action_space.n, discount)
        if 'cvt_string' in kwargs:
            logging.warning('cvt_string kwarg is not implemented for CartPole!')
            
    def legal_actions(self):
        return [idx for idx in range(self.env.action_space.n)]

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)

        return observation

    def close(self):
        self.env.close()
