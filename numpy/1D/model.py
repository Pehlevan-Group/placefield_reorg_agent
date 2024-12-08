import numpy as np


# main agent description
def uniform_pc_weights(npc, nact,seed,sigma=0.1, alpha=1,envsize=1):  # homogeneous population
    pc_cent =  np.linspace(-envsize,envsize,npc) 
    pc_sigma = np.ones(npc)*sigma
    pc_constant = np.ones(npc)*alpha 
    np.random.seed(seed)
    return [np.array(pc_cent), np.array(pc_sigma), np.array(pc_constant), 
    1e-5 * np.random.normal(size=(npc,nact)), 1e-5 * np.random.normal(size=(npc,1))]

def random_all_pc_weights(npc, nact,seed,sigma=0.1, alpha=1,envsize=1):  # heterogeneous population
    np.random.seed(seed)
    pc_cent =  np.random.uniform(-envsize,envsize,npc) 
    pc_sigma = np.random.uniform(1e-5, sigma, npc)
    pc_constant = np.random.uniform(0, alpha,size=npc)
    
    return [np.array(pc_cent), np.array(pc_sigma), np.array(pc_constant), 
    1e-5 * np.random.normal(size=(npc,nact)), 1e-5 * np.random.normal(size=(npc,1))]


def predict_placecell(params, x):
    pc_centers, pc_sigmas, pc_constant, actor_weights,critic_weights = params
    exponent = ((x-pc_centers)/pc_sigmas)**2
    pcact = np.exp(-0.5*exponent) * pc_constant**2
    return pcact

def predict_batch_placecell(params, xs):  
    pcacts = []  
    for x in xs:
        pcacts.append(predict_placecell(params, x))
    pcacts = np.array(pcacts)
    return pcacts


def predict_value(params, pcact):
    pc_centers, pc_sigmas, pc_constant, actor_weights,critic_weights = params
    value = np.matmul(pcact, critic_weights)
    return value

def softmax(x):
    x_max = np.max(x, axis=-1, keepdims=True)
    unnormalized = np.exp(x - x_max)
    return unnormalized/np.sum(unnormalized, axis=-1, keepdims=True)

def predict_action_prob(params, pcact, beta=1):
    pc_centers, pc_sigmas, pc_constant, actor_weights,critic_weights = params
    actout = np.matmul(pcact, actor_weights)
    aprob = softmax(beta * actout)
    return aprob

def get_onehot_action(prob, nact=2):
    A = np.random.choice(a=np.arange(nact), p=np.array(prob))
    onehotg = np.zeros(nact)
    onehotg[A] = 1
    return onehotg

def learn(params, reward, newstate,state, onehotg,aprob, gamma, etas,b_sig_alp=[0.0,0.0],clip_sig_alp=[0,0], noise=0.0, paramsindex=[], beta=1, bptype='both'):
    
    pcact = predict_placecell(params, state)
    newpcact = predict_placecell(params, newstate)
    td = (reward + gamma * predict_value(params, newpcact) - predict_value(params, pcact))[0]  # TD error

    # get critic grads
    dcri = pcact[:,None] * td

    # get actor grads
    if bptype == 'actg':
        decay = beta * (onehotg[:,None])  # from Foster et al. 2000, simplified form of the derivative
    else:
        decay = beta * (onehotg[:,None]- aprob[:,None])  # derived from softmax grads
 
    dact = (pcact[:,None] @ decay.T) * td

    # get phi grads: dp = phi' (W^actor @ act + W^critic) * td
    # compute TD error that will be backpropagated through actor/critic: Eq. 48
    if bptype == 'both':
        post_td = (params[3] @ decay + params[4]) * td
    elif bptype ==  'cri':
        post_td = params[4] * td
    elif bptype == 'act':
        post_td = (params[3] @ decay) * td
    elif bptype == 'actg':
        post_td = (params[3] @ decay) * td
    elif bptype == 'none':
        post_td = td

    # compute L2 loss for alpha, sigma. Not necessary, but nice to add to the objective to learn sparse representations. set to 0. 
    l2_grad_alpha =  b_sig_alp[1] * 2*params[2]
    l2_grad_sigma = b_sig_alp[0] * 2*params[1]

    # compute gradients for field parameters using Eq. 49 - 51. 
    dpcc = (post_td * (pcact[:,None]) * ((state - params[0])/params[1]**2)[:,None])[:,0]
    dpcs = (post_td * (pcact[:,None]) * (((state - params[0])**2/params[1]**3) - l2_grad_sigma)[:,None])[:,0]
    dpca = (post_td * (pcact[:,None]) * ((2 / params[2][:,None])) - l2_grad_alpha)[:,0]
    
    grads = [dpcc, dpcs, dpca, dact, dcri]

    #update weights by gradient ascent
    for p in range(len(params)):
        params[p] += etas[p] * grads[p]

    # add Gaussian noise to field parameters or actor-critic weights, define using paramsindex. 
    for p in paramsindex:
        ns = np.random.normal(size=params[p].shape) * noise
        params[p] += ns

    # clip large fields. Not necessary but if you want to keep fields withing some upper bound. 
    # If sigma --> 0, fields will explode. hence lower bound is 1e-5.
    if clip_sig_alp[0] > 0:
        params[1] = np.clip(params[1],1e-5, clip_sig_alp[0])
    if clip_sig_alp[1]>0:
        params[2] = np.clip(params[2], 1e-5,clip_sig_alp[1])

    return params, td



def get_discounted_rewards(rewards, gamma=0.9, norm=False):
    discounted_rewards = []
    cumulative = 0
    for reward in rewards[::-1]:
        cumulative = reward + gamma * cumulative  # discounted reward with gamma
        discounted_rewards.append(cumulative)
    discounted_rewards.reverse()
    if norm:
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-9)
    return np.array(discounted_rewards)[:,0]