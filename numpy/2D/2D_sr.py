#%%
# Code to run the 2D version of the Successor Representation Agent described in the paper

from model import *
from env import *
from utils import *
from sr_utils import *
import numpy as np
from copy import deepcopy
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--episodes', type=int, required=False, help='episodes', default=50000)
parser.add_argument('--obs', type=bool, required=False, help='obs', default=True)
parser.add_argument('--plr', type=float, required=False, help='plr', default=0.01)
parser.add_argument('--clr', type=float, required=False, help='clr', default=0.01)
parser.add_argument('--llr', type=float, required=False, help='llr', default=0.0)
parser.add_argument('--alr', type=float, required=False, help='alr', default=0.0)
parser.add_argument('--slr', type=float, required=False, help='slr', default=0.0)
parser.add_argument('--gamma', type=float, required=False, help='gamma', default=0.9)
parser.add_argument('--npc', type=int, required=False, help='npc', default=21)
parser.add_argument('--seed', type=int, required=False, help='seed', default=0)
parser.add_argument('--pcinit', type=str, required=False, help='pcinit', default='uni')
parser.add_argument('--balpha', type=float, required=False, help='balpha', default=0.0)
parser.add_argument('--noise', type=float, required=False, help='noise', default=0.000)
parser.add_argument('--paramsindex', type=int,nargs='+', required=False, help='paramsindex', default=[])
args, unknown = parser.parse_known_args()



# training params
train_episodes = args.episodes
tmax = 100

# env pararms
envsize = 1
maxspeed = 0.1
goalsize = 0.1
startcoord = [[-0.75,-0.75],[-0.75,0.75], [0.75,0.75]]
# startcoord = [-0.75,-0.75]
goalcoord = [0.75,-0.75]
obs = True
seed = args.seed
initvelocity = 0.0
max_reward = 5
obs= args.obs

#agent params
npc = args.npc**2
sigma = 0.1
alpha = 0.5
nact = 4

savevar = False
savefig = False
savegif = False

llr = args.llr
alr = args.alr
slr = args.slr
pcinit = args.pcinit

# load data
exptname = f'2D_obs_td_multi_0.0ns_012p_{npc}n_0.01plr_0.01clr_0.0llr_0.0alr_0.0slr_uni_4a_0s_50000e_5rmax_0.1rsz'
[logparams, all_rewards, allcoords] = saveload(f"./data/{exptname}",1,"load")


# choose param 
lr = 0.0025
gamma = 0.999

# inner loop training loop
def run_trial(params, env, U):
    state, goal, eucdist, done = env.reset()
    coords = []
    for t in range(tmax):

        pcact = predict_placecell(params, state)

        aprob = predict_action_prob(params, pcact)

        onehotg = get_onehot_action(aprob, nact=nact)

        newstate, reward, done = env.step(onehotg) 

        # learn SR
        nextpcact = predict_placecell(params, newstate)

        pcact = pcact[:,None]
        nextpcact = nextpcact[:,None]

        M = relu(U) @ pcact
        M1 = relu(U) @ nextpcact

        td = pcact.T + gamma * M1 - M 
        delu = td * pcact

        U += lr * delu

        coords.append(state)
        state = newstate.copy()

        if done:
            break

    return U, np.array(coords)


allcoords = []
Us = []
ca1s = []
U = np.eye(npc)
Us.append(deepcopy(U))
env = NDimNav(startcoord=startcoord, goalcoord=goalcoord, goalsize=goalsize, tmax=tmax, 
                    maxspeed=maxspeed,envsize=envsize, nact=nact, max_reward=max_reward, obstacles=obs)

for episode in range(train_episodes):

    params = logparams[episode]

    U, coords = run_trial(params, env,U)

    Us.append(deepcopy(U))
    allcoords.append(coords)

    print(f'Trial {episode+1}, U {np.max(Us[-1])}')

# saveload(f'./data/2D_sr_{lr}_{exptname}',[Us], 'save')

plot_trajectory(allcoords, -1)

xs = get_statespace(41)
ca3 = predict_batch_placecell(logparams[0], xs)
plot_all_sr_pc(Us, ca3, train_episodes)


some_Us = []
for trial in [0, 1000,5000,10000,50000]:
    some_Us.append(Us[trial])
saveload(f'./data/2D_some_sr_{lr}_{exptname}',[some_Us], 'save')







