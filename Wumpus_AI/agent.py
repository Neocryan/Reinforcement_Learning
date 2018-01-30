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
        #softmax
        #self.CDFarr = np.array([(i + 1) / 8.0 for i in range(8)])
        #self.temperature = 0.10
        self.lr = 0.4 # learning_rate
        self.gamma = 0.9   # reward_decay
        self.epsilon = 0.1    # e_greedy
        self.q_table = np.random.rand(784, 8)*30+20
        #
        self.St=((0,0),False,False,3)
        self.At=1
        self.Rtplus1=10

    def reset(self):
        self.St=((0,0),False,False,3)
        self.At=1
        self.Rtplus1=100

    def getrow(self,state):
        return state[0][0]*112+state[0][1]*16+state[1]*8+state[2]*4+state[3]
    #
    # def getProb(self,lst):
    #     tmp=np.exp(lst/self.temperature)
    #     ptmp=tmp/(np.sum(tmp)*1.0)
    #     return ptmp

    def act(self, observation):#the observation here is Stplus1
        #updata q_table
        q_predict = self.q_table[self.getrow(self.St), self.At-1]
        q_target = self.Rtplus1 + self.gamma * np.max(self.q_table[self.getrow(observation), :])
        self.q_table[self.getrow(self.St), self.At-1] += self.lr * (q_target - q_predict)
        if np.random.uniform() > self.epsilon:
            state_action = self.q_table[self.getrow(observation), :]
            # for the same state , probably the same q value for different actions
            # state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = np.argmax(state_action)+1
        else:   # randomly choose action
            action = np.random.choice(np.random.randint(1,9))
        return action

    def reward(self, observation, action, reward):#the observation here is St
        self.St=observation
        self.At=action
        self.Rtplus1=reward

class QlearningAgent_1:
    def __init__(self):
        #softmax
        self.CDFarr = np.array([(i + 1) / 8.0 for i in range(8)])
        self.temperature = 0.10
        self.lr = 0.2 # learning_rate
        self.gamma = 0.9   # reward_decay
        self.q_table = np.random.rand(128, 8)*30+20#(A_previuos,B,S,F)
        #
        self.St=(1,False,False,3)
        self.At=1
        self.Rtplus1=5.0

    def reset(self):
        self.St=(1,False,False,3)
        self.At=1
        self.Rtplus1=5

    def getrow(self,state):
        return (state[0]-1)*16+state[1]*8+state[2]*4+state[3]

    def getProb(self,lst):
        tmp=np.exp(lst/self.temperature)
        ptmp=tmp/(np.sum(tmp)*1.0)
        return ptmp

    def act(self, observation):#the observation here is Stplus1
        #updata q_table
        Stplus1=(self.At,observation[1],observation[2],observation[3])
        q_predict = self.q_table[self.getrow(self.St), self.At-1]
        q_target = self.Rtplus1 + self.gamma * np.max(self.q_table[self.getrow(Stplus1), :])
        self.q_table[self.getrow(self.St), self.At-1] += self.lr * (q_target - q_predict)

        q_Stplus1=self.q_table[self.getrow(Stplus1), :]
        Ptmp = self.getProb(q_Stplus1)
        self.CDFarr = np.array([np.sum(Ptmp[0:i]) for i in range(1,9)])
        #choose Atplus1 according to q_table
        action=int(np.where(self.CDFarr >= np.random.uniform())[0][0])+1
        return action

    def reward(self, observation, action, reward):#the observation here is St
        self.St=(self.At,observation[1],observation[2],observation[3])
        self.At=action
        self.Rtplus1=reward

class QlearningAgent_2:# with e greedy
    def __init__(self):
        self.epsilon = 0.1
        self.lr = 0.2 # learning_rate
        self.gamma = 0.9   # reward_decay
        self.q_table = np.random.rand(128, 8)*30+20#(A_previuos,B,S,F)
        #
        self.St=(1,False,False,3)
        self.At=1
        self.Rtplus1=5.0

    def reset(self):
        self.St=(1,False,False,3)
        self.At=1
        self.Rtplus1=5

    def getrow(self,state):
        return (state[0]-1)*16+state[1]*8+state[2]*4+state[3]

    def act(self, observation):#the observation here is Stplus1
        #updata q_table
        Stplus1=(self.At,observation[1],observation[2],observation[3])
        q_predict = self.q_table[self.getrow(self.St), self.At-1]
        q_target = self.Rtplus1 + self.gamma * np.max(self.q_table[self.getrow(Stplus1), :])
        self.q_table[self.getrow(self.St), self.At-1] += self.lr * (q_target - q_predict)
        #choose Atplus1 according to q_table
        if np.random.uniform() > self.epsilon:
            state_action = self.q_table[self.getrow(Stplus1), :]
            # for the same state , probably the same q value for different actions
            #state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = np.argmax(state_action)+1
        else:   # randomly choose action
            action = np.random.choice(np.random.randint(1,9))
        return action

    def reward(self, observation, action, reward):#the observation here is St
        self.St=(self.At,observation[1],observation[2],observation[3])
        self.At=action
        self.Rtplus1=reward

class SarsaAgent:#e-greedy
    def __init__(self):# e-greedy
        """Init a new agent.
        """
        #softmax
        #self.CDFarr = np.array([(i + 1) / 8.0 for i in range(8)])
        #self.temperature = 0.10
        self.lr = 0.4 # learning_rate
        self.gamma = 0.9   # reward_decay
        self.epsilon = 0.2    # e_greedy
        self.q_table = np.random.rand(784, 8)*30+20
        #
        self.St=((0,0),False,False,3)
        self.At=1
        self.Rtplus1=10

    def reset(self):
        self.St=((0,0),False,False,3)
        self.At=1
        self.Rtplus1=100

    def getrow(self,state):
        return state[0][0]*112+state[0][1]*16+state[1]*8+state[2]*4+state[3]
    #
    # def getProb(self,lst):
    #     tmp=np.exp(lst/self.temperature)
    #     ptmp=tmp/(np.sum(tmp)*1.0)
    #     return ptmp

    def act(self, observation):#the observation here is Stplus1
        #choose Atplus1 according to q_table
        if np.random.uniform() > self.epsilon:
            state_action = self.q_table[self.getrow(observation), :]
            action = np.argmax(state_action)+1
        else:   # randomly choose action
            action = np.random.choice(np.random.randint(1,9))
        #updata q_table
        q_predict = self.q_table[self.getrow(self.St), self.At-1]
        q_target = self.Rtplus1 + self.gamma * np.max(self.q_table[self.getrow(observation), action-1])
        self.q_table[self.getrow(self.St), self.At-1] += self.lr * (q_target - q_predict)

        return action

    def reward(self, observation, action, reward):#the observation here is St
        self.St=observation
        self.At=action
        self.Rtplus1=reward

class Exp_SarsaAgent:#e-greedy
    def __init__(self):# e-greedy
        """Init a new agent.
        """
        #softmax
        #self.CDFarr = np.array([(i + 1) / 8.0 for i in range(8)])
        self.temperature = 10
        self.lr = 0.3 # learning_rate
        self.gamma = 0.9   # reward_decay
        self.epsilon = 0.2    # e_greedy
        self.q_table = np.random.rand(784, 8)*30+20
        #
        self.St=((0,0),False,False,3)
        self.At=1
        self.Rtplus1=10

    def reset(self):
        self.St=((0,0),False,False,3)
        self.At=1
        self.Rtplus1=100

    def getrow(self,state):
        return state[0][0]*112+state[0][1]*16+state[1]*8+state[2]*4+state[3]
    #
    def getProb(self,lst):
        tmp=np.exp(lst/self.temperature)
        ptmp=tmp/(np.sum(tmp)*1.0)
        return ptmp

    def act(self, observation):#the observation here is Stplus1
        #choose Atplus1 according to q_table
        if np.random.uniform() > self.epsilon:
            state_action = self.q_table[self.getrow(observation), :]
            action = np.argmax(state_action)+1
        else:   # randomly choose action
            action = np.random.choice(np.random.randint(1,9))
        #updata q_table
        q_predict = self.q_table[self.getrow(self.St), self.At-1]
        q_Stplus1=self.q_table[self.getrow(observation), :]
        prob_Stplus1=self.getProb(q_Stplus1)
        q_target = self.Rtplus1 + self.gamma * (np.inner(prob_Stplus1,q_Stplus1))
        self.q_table[self.getrow(self.St), self.At-1] += self.lr * (q_target - q_predict)

        return action

    def reward(self, observation, action, reward):#the observation here is St
        self.St=observation
        self.At=action
        self.Rtplus1=reward

Agent = Exp_SarsaAgent
