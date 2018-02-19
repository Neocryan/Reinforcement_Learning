"""
This is the machinnery that runs your agent in an environment.

This is not intented to be modified during the practical.
"""

class Runner:
    def __init__(self, environment, agent, verbose=False):
        self.environment = environment
        self.agent = agent
        self.verbose = verbose

    def step(self):
        observation = self.environment.observe()
        action = self.agent.act(observation)
        (reward, stop) = self.environment.act(action)
        self.agent.reward(observation, action, reward)
        return (observation, action, reward, stop)

    def loop(self, games, max_iter):
        cumul_reward = 0.0
        for g in range(1, games+1):
            game_reward = 0.0
            self.environment.reset() 
            self.agent.reset(self.environment.get_range())
            for i in range(1, max_iter+1):
                if self.verbose:
                    print("Simulation step {}:".format(i))
                (obs, act, rew, stop) = self.step()
                game_reward += rew
                if self.verbose:
                    print(" ->       observation: {}".format(obs))
                    print(" ->            action: {}".format(act))
                    print(" ->            reward: {}".format(rew))
                    print(" -> cumulative reward: {}".format(cumul_reward))
                    if stop is not None:
                        print(" ->    Terminal event: {}".format(stop))
                    print("")
                if stop is not None:
                    break
            if True or self.verbose:
                print(" <=> Finished game number: {}, game reward: {} <=>".format(g, game_reward))
                print("")
            cumul_reward += game_reward
        return cumul_reward

def iter_or_loopcall(o, count):
    if callable(o):
        return [ o() for _ in range(count) ]
    else:
        # must be iterable
        return list(iter(o))

class BatchRunner:
    """
    Runs several instances of the same RL problem in parallel
    and aggregates the results.
    """

    def __init__(self, env_maker, agent_maker, count, verbose=False):
        self.environments = iter_or_loopcall(env_maker, count)
        self.agents = iter_or_loopcall(agent_maker, count)
        assert(len(self.agents) == len(self.environments))
        self.verbose = verbose
        self.ended = [ False for _ in self.environments ]

    def game(self, max_iter):
        rewards = []
        for (agent, env) in zip(self.agents, self.environments):
            env.reset()
            agent.reset(env.get_range())
            game_reward = 0
            for i in range(1, max_iter+1):
                observation = env.observe()
                action = agent.act(observation)
                (reward, stop) = env.act(action)
                agent.reward(observation, action, reward)
                game_reward += reward
                if stop is not None:
                    break
            rewards.append(game_reward)
        return sum(rewards)/len(rewards)

    def loop(self, games, max_iter):
        cum_avg_reward = 0.0
        for g in range(1, games+1):
            avg_reward = self.game(max_iter)
            if g >= 0.9*games:
                cum_avg_reward += avg_reward
            if self.verbose:
                print("Simulation game {}:".format(g))
                print(" ->            average reward: {}".format(avg_reward))
                print(" -> cumulative average reward: {}".format(cum_avg_reward))
        return cum_avg_reward
