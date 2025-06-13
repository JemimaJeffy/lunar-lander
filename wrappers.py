import gym
import numpy as np
from gym.spaces import Box

class NormalizeObservation(gym.ObservationWrapper):
    """This wrapper will normalize observations to the range [-1, 1]."""
    def __init__(self, env):
        super().__init__(env)
        self.obs_low = self.observation_space.low
        self.obs_high = self.observation_space.high
        self.observation_space = Box(low=-1.0, high=1.0,
                                     shape=self.observation_space.shape,
                                     dtype=np.float32)

    def observation(self, obs):
        # Avoid division by zero if low and high are equal
        range = self.obs_high - self.obs_low
        # Normalize obs to [-1, 1]
        normalized_obs = 2.0 * (obs - self.obs_low) / (range + 1e-8) - 1.0
        return normalized_obs