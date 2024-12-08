#%%
# Code to run the 1D version of the Successor Representation Agent described in the paper
from env import *
from model import *
from utils import *
from sr_utils import *
import numpy as np
from copy import deepcopy

# training params
train_episodes = 50000
tmax = 100
envsize = 1
maxspeed = 0.1
goalsize = 0.05
startcoord = [-0.75]
goalcoord = [0.5]
max_reward = 5

# 1) train an agent with fixed place field data first. where clr, alr,llr = 0. this will learn a policy
# 2) load the agent that specifies the policy basd on the fixed fields.

#agent params
npc = 64
sigma = 0.1
alpha = 0.5
nact = 2
lr = 0.0025  # lr for successor fields 
gamma = 0.9 # gamma for successor fields 
seed = 1

exptname = f"full_1D_td_online_both_0.0ns_012p_{npc}n_0.01plr_0.01clr_0.0llr_0.0alr_0.0slr_homo_0.5a_0.1s_2a_{seed}s_50000e_5rmax_0.05rsz"
[logparams, allrewards, allcoords] = saveload(f"./data/{exptname}",1,"load")
print(exptname)

#%%
# inner loop training loop
def run_trial(params, env, U):
    state, goal, eucdist, done = env.reset()

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

        state = newstate.copy()

        if done:
            break

    return U

xs = np.linspace(-1,1,1001)
ca3 = predict_batch_placecell(logparams[0], xs)

Us = []
ca1s = []
U = np.eye(npc)
Us.append(deepcopy(U))
env = OneDimNav(startcoord=startcoord, goalcoord=goalcoord, goalsize=goalsize, tmax=tmax, 
                maxspeed=maxspeed,envsize=envsize, nact=nact, max_reward=max_reward)

for episode in range(train_episodes):

    params = logparams[episode]

    U = run_trial(params, env,U)

    Us.append(deepcopy(U))

    print(f'Trial {episode+1}, U {np.max(Us[-1])}')


saveload(f'./data/sr_data/sr_{lr}_{exptname}',[Us, ca3], 'save')

#%%
# load saved SR variables
[Us, ca3] = saveload(f'./data/sr_data/sr_{lr}_{exptname}', 1, 'load')


#%%
# plot field area, COM shift and examples
gap = 50
f,axs = plt.subplots(2,3,figsize=(12,6))

plot_frequency(allcoords, [gap, train_episodes//10, train_episodes], ax=axs[0,0], gap=gap)
axs[0,0].set_title('Frequency dynamics')

dx_sr = plot_sr_density(Us, ca3, [gap, train_episodes//10, train_episodes],ax=axs[0,1])

plot_density(logparams, [gap, train_episodes//10, train_episodes], ax=axs[0,2])
axs[0,2].set_title('Density learnd using RL')


fxs, dxs, Rs = plot_fxdx_trials(allcoords, logparams,np.linspace(gap, train_episodes,dtype=int, num=51), ax=axs[1,0], gap=gap)
axs[1,0].set_title('f(x):d(x) correlation with learning')
print(Rs)

fxs, dxs, Rs = plot_sr_fxdx_trials_kde(allcoords, Us,logparams, np.linspace(gap, train_episodes,dtype=int, num=51), gap=gap, ax=axs[1,0])
print(Rs)


plot_field_area(logparams, np.linspace(gap, train_episodes, num=51, dtype=int), ax=axs[1,1])
axs[1,1].set_title('Field area increase with learning')
axs[1,1].plot([],[],label='RM', color='tab:blue')
axs[1,1].plot([],[],label='SR',color='tab:orange')
axs[1,1].legend(frameon=False, loc='best')

areas = plot_sr_field_area(Us, ca3, np.linspace(gap, train_episodes, num=51, dtype=int),ax=axs[1,1].twinx())

# change in field location
plot_field_center(logparams, np.linspace(gap, train_episodes, num=51, dtype=int), ax=axs[1,2])
axs[1,2].set_title('Fields shift backward with learning')
axs[1,2].plot([],[],label='RM',color='tab:blue')
axs[1,2].plot([],[],label='SR', color='tab:orange')
axs[1,2].legend(frameon=False, loc='best')

plot_sr_center(Us,ca3, np.linspace(gap, train_episodes, num=51, dtype=int),ax=axs[1,2].twinx())

f.tight_layout()

# f.savefig('./svgs/rl_vs_sr.svg')

f2,axs = plt.subplots(1,3,figsize=(12,3))

plot_sr_pc(Us, ca3, gap, ax=axs[0], title=f'T={gap}')
axs[0].set_ylabel('$\psi(x)$')

plot_sr_pc(Us, ca3,train_episodes//10, ax=axs[1], title=f'T={train_episodes//10}')
axs[1].set_ylabel('$\psi(x)$')

plot_sr_pc(Us, ca3,train_episodes, ax=axs[2], title=f'T={train_episodes}')
axs[2].set_ylabel('$\psi(x)$')

f2.tight_layout()
# f2.savefig('./svgs/sr_pcs.svg')


# %%
