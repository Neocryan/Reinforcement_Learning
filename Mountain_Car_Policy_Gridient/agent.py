import numpy as np

"""
Contains the definition of the agent that will run in an
environment.
"""


def kernel(s, kernel):
    phi = np.exp(-(s[0] - kernel[:, 0]) ** 2) * np.exp(-(s[1] - kernel[:, 1]) ** 2) * 1000
    return phi


class RandomAgent:
    def __init__(self):
        self.p = 10  # x
        self.k = 5  # v
        self._kernel = np.zeros([(self.p + 1) * (self.k + 1), 2])
        self._dmin = 0
        self._w = np.random.rand((self.p + 1) * (self.k + 1))
        self._theta = np.random.rand((self.p + 1) * (self.k + 1))
        self._t = 1
        self._alpha_theta = 0.1
        self._alpha_w = 0.1
        self._gamma = 0.99
        self._I = 1
        self._last_miu = None
        self._lastr = None
        self._thisr = None
        self._last_s = None
        self._this_s = None

        self.sd = 0
        self.lasta = 0
        self.zw = np.zeros((self.p + 1) * (self.k + 1))
        self.ztheta = np.zeros((self.p + 1) * (self.k + 1))
        self.ztheta_sd = np.zeros((self.p + 1) * (self.k + 1))
        self.theta_sd = np.zeros((self.p + 1) * (self.k + 1))

    def reset(self, x_range):
        # self._last_miu = None
        # self._lastr = None
        # self._thisr = None
        # self._last_s = None
        # self._this_s = None
        if self._dmin == -x_range[0]:
            pass
        else:
            self._dmin = -x_range[0]
            for x in range((self.p + 1) * (self.k + 1)):
                i = x // (self.k + 1)
                j = x - (self.k + 1) * i
                self._kernel[x] = [i * (self._dmin / self.p) - self._dmin, j * (40 / self.k) - 20]

    def act(self, observation):
        self._t += 1
        self._this_s = kernel(observation, self._kernel)
        self._this_miu = self._this_s.dot(self._theta)
        # self.sd = np.exp(self._this_s.dot(self.theta_sd)) +0.001
        # print(self._this_s.dot(self.theta_sd))
        # self.sd = self._this_s.dot(self.theta_sd) + 0.001

        if self._t < 6666:
            self.sd = 20 / (np.log(self._t + 1) + 0.001)
        else:
            self.sd = 10 / (np.log(self._t + 1) + 0.001)
        # print(self._this_miu)
        # print(self._this_s.dot(self.theta_sd))

        return np.random.normal(self._this_miu, self.sd)

    def reward(self, observation, action, reward):

        if self._lastr is not None:
            diff = self._lastr + self._gamma * self._this_miu - self._last_s.dot(self._w)
            # self.zw = self._gamma * 0.3 * self.zw + self._I * self._last_s
            # self.ztheta = self._gamma* 0.3 * self.ztheta + self._I * (self.lasta - self._last_s.dot(self._theta)) * self._last_s/self.sd **2
            # # self.ztheta_sd = self._gamma* 0.3 * self.ztheta + self._I * ((self.lasta - self._last_s.dot(self._theta))**2 / (self._last_s.dot(self.theta_sd) **2+0.01) -1) * self._last_s
            # # print(diff)
            # self._w += self._alpha_w * diff * self.zw
            # self._theta += self._alpha_theta *diff *self.ztheta
            # self.theta_sd += self._alpha_theta *diff *self.ztheta_sd

            self._w += self._alpha_w * self._I * diff * self._last_s

            # print(self.sd **2)
            self._theta += self._alpha_theta * self._I * diff * (self.lasta - self._last_s.dot(self._theta)) * self._last_s/self.sd **2
            # self._theta += self._alpha_theta * self._I * diff * (
            #             self.lasta - self._this_miu) * self._last_s / self.sd ** 2

        self._I *= self._gamma
        self._last_s = self._this_s
        self.lasta = action
        self._lastr = reward
        self._last_miu = self._this_miu


class PolicyGradientAgent:

    def __init__(self):
        self.p = 20
        self.k = 20
        self.s = np.zeros((self.p + 1, self.k + 1, 2))
        self.gamma = 0.99
        self.sigma = 10.0


        self.w = np.random.rand(self.p + 1, self.k + 1)
        self.a_w = 0.05
        self.dim = 0
        # Critic parameters: l = lambda, e = trace, a = alpha
        self.theta = np.random.rand(self.p + 1, self.k + 1)
        self.a_theta = 0.05

        self.last_P = None
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.t = 0.0
        self.epochs = 0

    def reset(self, x_range):
        self.epochs+=1
        if self.dim != x_range[0]:
            for i in range(self.p + 1):
                for j in range(self.k + 1):
                    self.s[i, j, 0] = x_range[0] + float(i) * (x_range[1] - x_range[0]) / float(self.p)
                    self.s[i, j, 1] = -10.0 + float(j) * 40.0 / float(self.k)
            self.dim = x_range[0]

        self.last_action = None
        self.last_reward = None
        self.last_state = None
        self.last_P = None
        # self.t = 0
    def phi(self, i, j, x, vx):
        s_1 = self.s[i, j, 0]
        s_2 = self.s[i, j, 1]
        a = np.exp(-((x - s_1) ** 2) / self.p)
        b = np.exp(-((vx - s_2) ** 2) / self.k)
        return a * b

    def Phi(self, obs):
        x, vx = obs
        val = np.zeros(((self.p + 1), (self.k + 1)))
        for i in range(self.p + 1):
            for j in range(self.k + 1):
                val[i, j] = self.phi(i, j, x, vx)
        return val

    def act(self, observation):
        self.t += 1
        P_ = self.Phi(observation)
        mu = sum(sum(self.theta * P_))
        self.sigma = 20.0 / (np.log(self.t+1))
        if self.last_action is not None:
            P = self.last_P
            if self.last_P is None:
                P = self.Phi(self.last_state)
            v_current = sum(sum(self.w * P))
            v_next = sum(sum(self.w * P_))
            delta = self.last_reward + self.gamma * v_next - v_current
            self.w += self.a_w * self.gamma * delta * P
            log_pi_grad = (self.last_action - mu) * P / self.sigma ** 2
            self.theta += self.a_theta * self.gamma * delta * log_pi_grad
            self.last_P = P_
        if self.epochs <=160:
            return np.random.normal(mu, self.sigma)
        else:
            return mu

    def reward(self, observation, action, reward):


        self.last_action = action
        self.last_reward = reward
        self.last_state = observation


Agent = PolicyGradientAgent
