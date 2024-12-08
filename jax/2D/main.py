#%%
# Copyright (c) 2024 M Ganesh Kumar
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"
import jax.numpy as jnp
import numpy as np
from jax import config
config.update('jax_platform_name', 'cpu')  # need to fix 2D to use GPU
from jax.lib import xla_bridge
device = xla_bridge.get_backend().platform
print(device)


from utils import *
from env import *
from model import *
from copy import deepcopy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--episodes', type=int, required=False, help='episodes', default=5000)
parser.add_argument('--tmax', type=int, required=False, help='tmax', default=600)
parser.add_argument('--obs', type=bool, required=False, help='obs', default=True)
parser.add_argument('--startcoords', type=float,nargs='+', required=False, help='startcoods', default=[[-0.75,-0.75],[0.0,0.75]])
parser.add_argument('--goalcoords', type=float,nargs='+', required=False, help='goalcoords', default=[[0.75,-0.75]])
parser.add_argument('--obscoords', type=float,nargs='+', required=False, help='obscoords', default=[[-0.2,0.2,-1,0.5]])
parser.add_argument('--rsz', type=float, required=False, help='rsz', default=0.1)
parser.add_argument('--rmax', type=int, required=False, help='rmax', default=5)

parser.add_argument('--seed', type=int, required=False, help='seed', default=0)
parser.add_argument('--pcinit', type=str, required=False, help='pcinit', default='homo')
parser.add_argument('--npc', type=int, required=False, help='npc', default=16)
parser.add_argument('--alpha', type=float, required=False, help='alpha', default=1)
parser.add_argument('--sigma', type=float, required=False, help='sigma', default=0.05)

parser.add_argument('--plr', type=float, required=False, help='plr', default=0.01)
parser.add_argument('--clr', type=float, required=False, help='clr', default=0.01)
parser.add_argument('--llr', type=float, required=False, help='llr', default=0.0001) 
parser.add_argument('--alr', type=float, required=False, help='alr', default=0.0001) 
parser.add_argument('--slr', type=float, required=False, help='slr', default=0.0001)
parser.add_argument('--gamma', type=float, required=False, help='gamma', default=0.95)
parser.add_argument('--nact', type=int, required=False, help='nact', default=4)
parser.add_argument('--beta', type=float, required=False, help='beta', default=1)

parser.add_argument('--balpha', type=float, required=False, help='balpha', default=0.0)
parser.add_argument('--paramsindex', type=int,nargs='+', required=False, help='paramsindex', default=[0,1,2])
parser.add_argument('--noise', type=float, required=False, help='noise', default=0.000)

parser.add_argument('--analysis', type=str, required=False, help='analysis', default='na')
parser.add_argument('--datadir', type=str, required=False, help='datadir', default='./data/')
parser.add_argument('--figdir', type=str, required=False, help='figdir', default='./fig/')

args, unknown = parser.parse_known_args()


# training params
train_episodes = args.episodes
tmax = args.tmax
obs = args.obs

# env pararms
envsize = 1
maxspeed = 0.1
goalsize = args.rsz
startcoord = args.startcoords
goalcoords = args.goalcoords
obscoords = args.obscoords
seed = args.seed
max_reward = args.rmax

#agent params
npc = args.npc**2
sigma = args.sigma
alpha = args.alpha
nact = args.nact

# noise params
noise = args.noise
paramsindex = args.paramsindex
piname = ''.join(map(str, paramsindex))
pcinit = args.pcinit

actor_eta = args.plr
critic_eta = args.clr
pc_eta = args.llr
sigma_eta = args.slr
constant_eta = args.alr
etas = [pc_eta, sigma_eta,constant_eta, actor_eta,critic_eta]
gamma = args.gamma
betas = [0.5,args.balpha]

save_figs= False
savevar = True

exptname = f'2D_td_{noise}ns_{piname}p_{npc}n_{actor_eta}plr_{critic_eta}clr_{pc_eta}llr_{constant_eta}alr_{sigma_eta}slr_{pcinit}_{nact}a_{seed}s_{train_episodes}e_{max_reward}rmax_{goalsize}rsz'
figdir = './fig/'
datadir = './data/'

print(exptname)

if pcinit=='homo':
    params = uniform_2D_pc_weights(npc, nact, seed, sigma=sigma, alpha=alpha, envsize=envsize)
elif pcinit == 'hetero':
        params = random_all_pc_weights(npc, nact, seed, sigma=sigma, alpha=alpha, envsize=envsize)

initparams = deepcopy(params)
plot_all_pc([initparams],0)

# inner loop training loop
def run_trial(params, env):
    coords = []
    actions = []
    rewards = []

    state, goal, eucdist, done = env.reset()
    totR = 0
    
    for t in range(tmax):

        pcact = predict_placecell(params, state)

        aprob = predict_action_prob(params, pcact)

        onehotg = get_onehot_action(aprob, nact=nact)

        newstate, reward, done = env.step(onehotg) 

        coords.append(state)
        actions.append(onehotg)
        rewards.append(reward)

        state = newstate.copy()

        totR += reward

        if done:
            coords.append(newstate)  # include new state for value computation
            break

    return jnp.array(coords), jnp.array(rewards).reshape(-1), jnp.array(actions), t


#%%
losses = []
latencys = []
allcoords = []
logparams = []
logparams.append(initparams)
allrewards = []

for goalcoord in goalcoords:

    for obscoord in obscoords:
        env = NDimNav(startcoord=startcoord, goalcoord=goalcoord, goalsize=goalsize, tmax=tmax, 
                        maxspeed=maxspeed,envsize=envsize, nact=nact, max_reward=max_reward, obstacles=obs, obscoord=obscoord)

        for episode in range(train_episodes):

            coords, rewards, actions, latency = run_trial(params, env)
            
            params, grads, loss = update_td_params(params, coords, actions, rewards, etas, gamma, betas)

            # clip large fields
            params[2] = jnp.clip(params[2], 1e-5,2)
            params[1] = correct_covariance_matrices_np(params[1],1e-5, 0.5)

            allcoords.append(coords)
            logparams.append(params)
            latencys.append(latency)
            losses.append(loss)
            allrewards.append(env.total_reward)

            print(f'Start {env.track[1]}, Trial {episode+1}, G {env.total_reward:.3f}, t {latency}')


if args.analysis == 'full':
    saveload(datadir+exptname, [logparams, allrewards, allcoords], 'save')

env.plot_trajectory()
plot_all_pc(logparams,-1)
f,score, drift = plot_analysis(logparams, latencys,allrewards, allcoords, train_episodes//2, exptname=exptname, rsz=goalsize)


if save_figs:
    f.savefig(figdir+exptname+'.svg')

trials = [0,train_episodes//4, train_episodes]
f,ax = plt.subplots(1,len(trials),figsize=(3*len(trials),2*1))

for t,trial in enumerate(trials):
    xy = logparams[trial][0]
    ax[t].scatter(xy[:,0], xy[:,1],s=2,color='k')
    ax[t].set_aspect('equal')
    ax[t].set_title(f'COM $T={trial}$')
f.tight_layout()
