import numpy as np

from mountaincar import MountainCar

"""
This file contains the definition of the environment
in which the agents are run.
"""

ACT_FORWARD = 1
ACT_NONE = 0
ACT_BACKWARD = -1

class Environment:
    def __init__(self, g=10.0, d=100.0, H=10., m=10.0, F=6.0):
        """Instanciate a new environement in its initial state.
        """
        self.mc = MountainCar(g=g, d=d, H=H, m=m, F=F, R=50.0, T=0.0)

    def reset(self):
        self.mc.reset()
        # place the car at a random place near the bottom
        self.mc.x = np.random.uniform(-self.mc.d*1.3, -self.mc.d*0.7)

    def get_range(self):
        return [-1.5*self.mc.d, 0.0]

    def observe(self):
        """Returns the current observation that the agent can make
        of the environment, if applicable.
        """
        return (self.mc.x, self.mc.vx)

    def act(self, action):
        """Perform given action by the agent on the environment,
        and returns a reward.
        """
        reward = self.mc.act(action)
        return (reward, "victory" if reward > 0.0 else None)
