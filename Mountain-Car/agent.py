__author__ = 'Xiaoyu'
import numpy as np

"""
Contains the definition of the agent that will run in an
environment.
"""


class r0:
    def __init__(self):
        self._w = np.random.random([2, 3])
        self._olds = np.random.random([2, 1])
        self._gamma = 0.05
        self._alpha = 0.01
        self._lambda = 0.01
        self._z = np.random.random([2, 1])
        self._s = np.random.random([2, 1])
        self._oldr = -1
        self._range = 150

    def reset(self, x_range):
        # self._w /= 1e10000
        self._range = abs(x_range[1] - x_range[0])
        pass

    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.

        observation = (x, vx)
        """
        self._s = np.array(observation).reshape(2, 1)
        qs = np.dot(self._s.T, self._w)[0]
        return np.argmax(qs) - 1

    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        self._s = np.array(observation).reshape(2, 1)
        self._z = self._gamma * self._lambda * self._z + self._olds
        theta = self._oldr + self._gamma * np.dot(self._s.T, self._w) - np.dot(self._olds.T, self._w)
        self._w += self._alpha * np.dot(self._z, theta)
        self._olds = self._s
        self._oldr = reward
        print(self._w)


def kernel(s, kernel):
    phi = np.exp(-(s[0] - kernel[:, 0]) ** 2) * np.exp(-(s[1] - kernel[:, 1]) ** 2)
    return phi


class SemiGradientTdLambda:
    def __init__(self):
        self.p = 2
        self.k = 2
        self._kernel = np.zeros([(self.p + 1) * (self.k + 1), 2])
        self._phi = None
        self._w = np.zeros([3, (self.p + 1) * (self.k + 1)])
        self._olds = None
        self._olda = 0
        self._oldr = -1
        self._oldphi = None
        self.z = 0
        self._gamma = 0.5
        self._lambda = 5
        self._alpha = 0.5
        self._dmin = 150
        self.e = 0.05
        self._t = 0

    def reset(self, x_range):
        self._dmin = -x_range[0]
        for x in range((self.p + 1) * (self.k + 1)):
            i = x // (self.k + 1)
            j = x - (self.k + 1) * i
            self._kernel[x] = [i * (self._dmin / self.p) - self._dmin, j * (40 / self.k) - 20]

    # def kernel(self,s):
    #     phi = np.exp(-(s[0] - self._kernel[:, 0]) ** 2) * np.exp(-(s[1] - self._kernel[:, 1]) ** 2)
    #     return phi
    def act(self, observation):
        if np.random.uniform() > self.e:
            q = self._w.dot(kernel(observation, self._kernel))
            return np.argmax(q) - 1
        else:
            return np.random.choice([-1, 0, 1])

    def reward(self, observation, action, reward):
        # print(action)
        if self._olds is not None:
            self.z = self._gamma * self._lambda * self.z + kernel(self._olds, self._kernel)
            theta = reward + self._gamma * self._w[self._olda + 1].dot(kernel(observation, self._kernel)) - self._w[
                self._olda + 1].dot(kernel(self._olds, self._kernel))
            # theta = reward + self._gamma * self._w[self._olda+1].dot(kernel(observation,self._kernel)) - self._w[self._olda+1].dot(kernel(self._olds,self._kernel))
            self._w[self._olda] += self._alpha * theta * self.z
        else:
            self._t += 1
            if self._t > 1:
                print(self._t)
        self._olda = action
        self._oldr = reward
        self._olds = observation


class OnlineTdLambda:
    def __init__(self):
        self.p = 2
        self.k = 2
        self._kernel = np.zeros([(self.p + 1) * (self.k + 1), 2])
        self._phi = None
        self._w = np.zeros([3, (self.p + 1) * (self.k + 1)])
        self._olds = None
        self.olda = None
        self._oldr = -1
        self._oldphi = None
        self.z = np.zeros((self.p + 1) * (self.k + 1))
        self._gamma = 0.5
        self._lambda = 4
        self._alpha = 0.5
        self._dmin = 150
        self.e = 0.05
        self._t = 0
        self.oldv = None
        self.oldx = None
    def reset(self, x_range):
        self._dmin = -x_range[0]
        for x in range((self.p + 1) * (self.k + 1)):
            i = x // (self.k + 1)
            j = x - (self.k + 1) * i
            self._kernel[x] = [i * (self._dmin / self.p) - self._dmin, j * (40 / self.k) - 20]

    def act(self, observation):
        if np.random.uniform() > self.e:
            q = self._w.dot(kernel(observation, self._kernel))
            print(q)
            return np.random.choice(np.where(q == q.max())[0]) - 1
        else:
            return np.random.choice([-1, 0, 1])

    def reward(self, observation, action, reward):
        # print(action)
        x_prime = kernel(observation, self._kernel)
        try:
            v_prime = self._w[self.olda + 1].dot(x_prime)
        except:
            v_prime = 0
        if self.olda is not None:

            v = self._w[self.olda+1].dot(self.oldx)

            delta = reward + self._gamma *v_prime - v
            self.z = self._gamma * self._lambda * self.z + (1 - self._alpha * self._gamma *self._lambda *  self.z.dot(self.oldx))*(self.oldx)
            self._w[self.olda+1] += self._alpha * (delta + v -self.oldv) * self.z - self._alpha* (v - self.oldv)* self.oldx


        else:
            self._t+=1
            if self._t >1:
                print(self._t)
                print(self.oldv)

        self.olda = action
        self.oldv = v_prime
        self.oldx = x_prime


class Qlearning:
    def __init__(self):
        self.p = 3
        self.k = 3
        self._kernel = np.zeros([(self.p + 1) * (self.k + 1), 2])
        self._w0 = np.zeros((self.p + 1) * (self.k + 1))
        self._w1 = np.zeros((self.p + 1) * (self.k + 1))
        self._w2 = np.zeros((self.p + 1) * (self.k + 1))
        self._olds = None
        self.olda = None
        self._oldr = -0.1
        self._gamma = 0.5
        self._lambda = 0.5
        self._alpha = 0.001
        self._dmin = 150
        self.e = 0.05
        self._t = 0
        self.oldv = None
        self.thisKernel = None
        self.rbar = 0
        self.s = None
        self.lastKernel = None
    def reset(self, x_range):
        self._dmin = -x_range[0]
        for x in range((self.p + 1) * (self.k + 1)):
            i = x // (self.k + 1)
            j = x - (self.k + 1) * i
            self._kernel[x] = [i * (self._dmin / self.p) - self._dmin, j * (40 / self.k) - 20]

    def act(self, observation):
        if np.random.uniform() > self.e:
            self.thisKernel = kernel(observation,self._kernel)
            q = np.array([self._w0.dot(self.thisKernel),self._w1.dot(self.thisKernel),self._w2.dot(self.thisKernel)])
            # print(q)
            return np.random.choice(np.where(q == q.max())[0]) - 1
        else:
            return np.random.choice([-1, 0, 1])

    def reward(self, observation, action, reward):
        # print(action)
        if action == -1:
            self.s = self.thisKernel.dot(self._w0)
        if action == 0:
            self.s = self.thisKernel.dot(self._w1)
        if action == 1:
            self.s = self.thisKernel.dot(self._w2)

        if self.olda is not None:
            delta = self._oldr - self.rbar + self.s - self._olds
            self.rbar +=self._gamma * delta
            if self.olda == -1:
                self._w0 += self._alpha * delta *self.lastKernel
            if self.olda == 0:
                self._w1 += self._alpha * delta *self.lastKernel
            if self.olda == 1:
                self._w2 += self._alpha * delta *self.lastKernel


        self._olds = self.s
        self.olda = action
        self._oldr = reward
        self.lastKernel = self.thisKernel
        # pass
class soso:
    def __init__(self):
        self.ds = {}
        self.alpha = 0.3
        self.gamma = 0.3
        self.olda = None
        self.olds = None
        self.oldr = None
        self.thiss = None
    def reset(self, x_range):
        self._dmin = -x_range[0]

    def act(self, observation):
        if abs(observation[0])> self._dmin/2:
            x = 1
        else:
            x = 0

        if observation[1]>0:
            v = 2
        elif observation[1]<0:
            v = 1
        else:
            v = 0
        self.thiss = (x,v)
        # self.thiss = v


        if self.thiss in self.ds.keys():
            return np.random.choice(np.where(self.ds[self.thiss] == self.ds[self.thiss].max())[0]) -1
        else:
            return np.random.choice([-1,0,1])

    def reward(self, observation, action, reward):
        if self.thiss not in self.ds.keys():
            self.ds[self.thiss] = np.ones(3)

        if self.olda is not None:
            self.ds[self.olds][self.olda + 1] += self.alpha * (
                        self.oldr + self.gamma * self.ds[self.thiss].max() - self.ds[self.olds][self.olda + 1])
        self.olda = action
        self.oldr = reward
        self.olds = self.thiss
        # print(self.ds)

class TdLambda:
    def __init__(self):
        self.p = 149 #x
        self.k = 49 #v
        self._kernel = np.zeros([(self.p + 1) * (self.k + 1), 2])
        self._w = np.zeros([3,(self.p + 1) * (self.k + 1)])
        self._lastphi = None
        self._thisphi = None
        self._lasta = 0
        self._lastr = -0.1
        self._lastq = np.zeros(3)
        self._thisq = np.zeros(3)
        self._gamma = 0.9
        self._lambda = 0.1
        self._alpha = 0.1
        self._dmin = 150
        self.e = 0.0
        self._t = 0
        self.eg = np.zeros(self._w.shape)
    def reset(self, x_range,a=0.3,b=0.5,c=0.0):
        self._dmin = -x_range[0]
        self._alpha = a
        self._lambda = b
        self._gamma = c
        for x in range((self.p + 1) * (self.k + 1)):
            i = x // (self.k + 1)
            j = x - (self.k + 1) * i
            self._kernel[x] = [i * (self._dmin / self.p) - self._dmin, j * (40 / self.k) - 20]

    def act(self, observation):
        self._thisphi = kernel(observation, self._kernel)
        if np.random.uniform() > 0.01:
            self._thisq = self._w.dot(self._thisphi)
            return np.random.choice(np.where(self._thisq == self._thisq.max())[0]) - 1
        else:
            return np.random.choice([-1, 0, 1])


    def reward(self, observation, action, reward):
        if self._lastphi is not None:
            sigma = self._lastr +self._gamma * np.max(self._thisq) - self._lastq[self._lasta +1]
            eg = np.zeros(self._w.shape)
            eg[self._lasta+1] = self._lastphi
            eg[np.argmax(self._thisq)] -= self._gamma * self._thisphi
            if np.argmax(self._thisq) == self._lasta + 1:
                self.eg = self._gamma * self._lambda * self.eg + eg
            else:
                self.eg = eg
            self._w += self._alpha * sigma *self.eg

        self._lasta = action
        self._lastphi = self._thisphi
        self._lastq = self._thisq
        self._lastr = reward




Agent = TdLambda
