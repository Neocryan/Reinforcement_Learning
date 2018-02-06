import argparse
import agent
import environment
import runner
import sys
import time
from multiprocessing import Pool
# 2to3 compatibility
try:
    input = raw_input
except NameError:
    pass

parser = argparse.ArgumentParser(description='RL running machine')
parser.add_argument('--environment', metavar='ENV_CLASS', type=str, default='Environment', help='Class to use for the environment. Must be in the \'environment\' module')
parser.add_argument('--agent', metavar='AGENT_CLASS', default='Agent', type=str, help='Class to use for the agent. Must be in the \'agent\' module.')
parser.add_argument('--ngames', type=int, metavar='n', default='200', help='number of games to simulate')
parser.add_argument('--niter', type=int, metavar='n', default='400', help='max number of iterations per game')
parser.add_argument('--batch', type=int, metavar='nagent', default=None, help='batch run several agent at the same time')
parser.add_argument('--verbose', action='store_true', help='Display cumulative results at each step')
parser.add_argument('--interactive', action='store_true',help='After training, play once in interactive mode. Ignored in batch mode.')

def main(a= 0.3, b= 0.5 ,c = 0.0):
    args = parser.parse_args()
    agent_class = eval('agent.{}'.format(args.agent))
    env_class = eval('environment.{}'.format(args.environment))

    if args.batch is not None:
        print("Running a batched simulation with {} agents in parallel...".format(args.batch))
        my_runner = runner.BatchRunner(env_class, agent_class, args.batch, args.verbose)
        final_reward = my_runner.loop(args.ngames, args.niter)
        print("Obtained a final average reward of {}".format(final_reward))
    else:
        print("Running a single instance simulation...")
        print('alpha= {},lambda = {}, e = {}'.format(a, b, c))
        my_runner = runner.Runner(env_class(), agent_class(), args.verbose,a=a,b=b,c=c)
        final_reward = my_runner.loop(args.ngames, args.niter)
        print('alpha= {},lambda = {}, e = {}'.format(a, b, c))
        with open('log.txt', 'a') as log:
            log.write('alpha= {},lambda = {},gamma:{},result:{}\n'.format(a, b, c, final_reward))
        print("Obtained a final reward of {}".format(final_reward))
        if args.interactive:
            # import matplotlib
            # matplotlib.use('TkAgg')
            import pylab as plb
            from mountaincarviewer import MountainCarViewer
            
            # prepare for the visualization
            plb.ion()
            env = my_runner.environment
            a = my_runner.agent
            env.reset()
            a.reset(env.get_range())
            mv = MountainCarViewer(env.mc)
            mv.create_figure(args.niter, args.niter)
            plb.draw()

            
            for _ in range(args.niter):
                print('\rt = {}'.format(env.mc.t))
                print("Enter to continue...")
                input()

                sys.stdout.flush()

                obs = env.observe()
                action = a.act(obs)
                (reward, stop) = env.act(action)
                a.reward(obs, action, reward)

                # update the visualization
                mv.update_figure()
                plb.draw()

                # check for rewards
                if stop is not None:
                    print("\rTop reached at t = {}".format(env.mc.t))
                    break
c = [(0.1,0.1,0.9),(0.1,0.1,0.9),(0.1,0.1,0.7),(0.1,0.1,0.7)]
if __name__ == "__main__":
    p = Pool()
    tic = time.time()
    bb = [p.apply_async(main, args=(x[0], x[1], x[2],)) for x in c]
    [x.get() for x in bb]

    print(time.time()-tic)