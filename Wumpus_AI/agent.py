import numpy as np
import collections

ACT_UP    = 1
ACT_DOWN  = 2
ACT_LEFT  = 3
ACT_RIGHT = 4
ACT_TORCH_UP    = 5
ACT_TORCH_DOWN  = 6
ACT_TORCH_LEFT  = 7
ACT_TORCH_RIGHT = 8

class QlearningAgent:
    def __init__(self):# e-greedy
        """Init a new agent.
        """
        self.lr = 0.2 # learning_rate
        self.gamma = 0.4   # reward_decay
        self.epsilon = 0.0    # e_greedy
        self.q_table = np.random.rand(784, 8)*30+20
        #
        self.St=((0,0),False,False,3)
        self.At=1
        self.Rtplus1=-10

    def reset(self):
        pass

    def getrow(self,state):
        return state[0][0]*112+state[0][1]*16+state[1]*8+state[2]*4+state[3]

    def act(self, observation):#the observation here is Stplus1
        #updata q_table
        q_predict = self.q_table[self.getrow(self.St), self.At-1]
        q_target = self.Rtplus1 + self.gamma * np.max(self.q_table[self.getrow(observation), :])
        self.q_table[self.getrow(self.St), self.At-1] += self.lr * (q_target - q_predict)
        if np.random.uniform() > self.epsilon:
            state_action = self.q_table[self.getrow(observation), :]
            action = np.argmax(state_action)+1
        else:   # randomly choose action
            action = np.random.choice(np.random.randint(1,9))
        return action

    def reward(self, observation, action, reward):#the observation here is St
        self.St=observation
        self.At=action
        self.Rtplus1=reward


class SarsaAgent:#e-greedy
    def __init__(self):# e-greedy
        """Init a new agent.
        """
        self.lr = 0.2 # learning_rate
        self.gamma = 0.4   # reward_decay
        self.epsilon = 0.0    # e_greedy
        self.q_table = np.random.rand(784, 8)*30+20
        #
        self.St=((0,0),False,False,3)
        self.At=1
        self.Rtplus1=-10

    def reset(self):
        pass

    def getrow(self,state):
        return state[0][0]*112+state[0][1]*16+state[1]*8+state[2]*4+state[3]

    def act(self, observation):#the observation here is Stplus1
        #choose Atplus1 according to q_table
        if np.random.uniform() > self.epsilon:
            state_action = self.q_table[self.getrow(observation), :]
            action = np.argmax(state_action)+1
        else:   # randomly choose action
            action = np.random.choice(np.random.randint(1,9))
        #updata q_table
        q_predict = self.q_table[self.getrow(self.St), self.At-1]
        q_target = self.Rtplus1 + self.gamma * (self.q_table[self.getrow(observation), action-1])
        self.q_table[self.getrow(self.St), self.At-1] += self.lr * (q_target - q_predict)

        return action

    def reward(self, observation, action, reward):#the observation here is St
        self.St=observation
        self.At=action
        self.Rtplus1=reward


Agent = SarsaAgent
