import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import os
import csv
import matplotlib.cm as cm

def invert_matrices(tensor):
    """ Compute the inverse for each 2x2 matrix in an N x 2 x 2 tensor efficiently using vectorized operations. """
    # Extract individual matrix elements
    a = tensor[:, 0, 0]
    b = tensor[:, 0, 1]
    c = tensor[:, 1, 0]
    d = tensor[:, 1, 1]
    
    # Compute the determinant for each matrix
    determinant = a * d - b * c
    
    # Compute the inverse of each matrix
    inv_tensor = np.empty((tensor.shape[0], 2, 2))
    inv_tensor[:, 0, 0] = d / determinant
    inv_tensor[:, 0, 1] = -b / determinant
    inv_tensor[:, 1, 0] = -c / determinant
    inv_tensor[:, 1, 1] = a / determinant
    
    return inv_tensor


# main agent description
def uniform_2D_pc_weights(npc, nact,seed=0,sigma=0.1, alpha=1,envsize=1):
    np.random.seed(seed)
    x = np.linspace(-envsize,envsize,int(npc**0.5))
    xx,yy = np.meshgrid(x,x)
    pc_cent = np.concatenate([xx.reshape(-1)[:,None],yy.reshape(-1)[:,None]],axis=1)
    pc_sigma = np.tile(np.eye(2),(npc,1,1))*sigma
    # pc_sigma = np.tile(np.ones([2,2]),(npc,1,1))*sigma
    pc_constant = np.ones(npc) * alpha
    return [np.array(pc_cent), np.array(pc_sigma), np.array(pc_constant), 
    1e-5 * np.random.normal(size=(npc,nact)), 1e-5 * np.random.normal(size=(npc,1))]

def random_all_pc_weights(npc, nact,seed,sigma=0.1, alpha=1,envsize=1):
    np.random.seed(seed)
    x1 = np.random.uniform(-envsize,envsize,int(npc**0.5))
    x2 = np.random.uniform(-envsize,envsize,int(npc**0.5))[::-1]
    xx,yy = np.meshgrid(x1,x2)
    pc_cent = np.concatenate([xx.reshape(-1)[:,None],yy.reshape(-1)[:,None]],axis=1)
    pc_sigma = np.random.uniform(1e-5, sigma, size=(npc, 2,2))
    pc_sigma = correct_covariance_matrices(pc_sigma,1e-5, 0.5)
    pc_constant = np.random.uniform(0, alpha,size=npc)
    return [np.array(pc_cent), np.array(pc_sigma), np.array(pc_constant), 
    1e-5 * np.random.normal(size=(npc,nact)), 1e-5 * np.random.normal(size=(npc,1))]

def predict_placecell(params, x):
    pc_centers, pc_sigmas, pc_constant, actor_weights, critic_weights = params
    inv_sigma = invert_matrices(pc_sigmas)  # Shape: (npc, dim, dim)
    diff = x - pc_centers  # Shape: (npc, dim)
    exponent = np.einsum('ni,nij,nj->n', diff, inv_sigma, diff)
    pcacts = np.exp(-0.5 * exponent) * pc_constant**2
    return pcacts

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

def get_onehot_action(prob, nact=4):
    A = np.random.choice(a=np.arange(nact), p=np.array(prob))
    onehotg = np.zeros(nact)
    onehotg[A] = 1
    return onehotg

def learn(params, reward, newstate, state, onehotg, aprob, gamma, etas, balpha=0.0, noise=0.0, paramsindex=[], beta=1):
    pc_centers, pc_sigmas, pc_constant, actor_weights, critic_weights = params
    
    # Predict place cell activations
    pcact = predict_placecell(params, state)
    newpcact = predict_placecell(params, newstate)
    
    # Predict values
    value = np.dot(pcact, critic_weights)
    newvalue = np.dot(newpcact, critic_weights)
    td = reward + gamma * newvalue - value

    l1_grad = balpha * np.sign(pc_constant)
    
    # Critic grads
    dcri = pcact[:, None] * td
    
    # Actor grads
    decay = beta * (onehotg[:, None] - aprob[:, None])
    dact = np.dot(pcact[:, None], decay.T) * td
    
    # Grads for field parameters
    post_td = (np.dot(actor_weights, decay) + critic_weights) * td

    df = state - pc_centers
    inv_sigma = invert_matrices(pc_sigmas)
    outer = np.einsum('nj,nk->njk',df,df)
    dpcs = 0.5 * (post_td * pcact[:,None])[:,:,None] * np.einsum('njl,njk,nik->nji',inv_sigma, outer, inv_sigma)
    
    dpcc = post_td * pcact[:,None] * np.einsum('nji,nj->ni', inv_sigma, df)
    dpca = (post_td * pcact[:,None] * (2/pc_constant[:,None]) - l1_grad)[:,0]

    grads = [dpcc, dpcs, dpca, dact, dcri]  # dpcc needs to be transposed back
    
    for p in range(len(params)):
        params[p] += etas[p] * grads[p]
    
    for p in paramsindex:
        ns = np.random.normal(size=params[p].shape) * noise
        params[p] += ns

    # clip large fields
    params[2] = np.clip(params[2], 1e-5,2)
    params[1] = correct_covariance_matrices(params[1],1e-5, 0.5)
    
    return params, grads, td


def get_discounted_rewards(rewards, gamma=0.9, norm=False):
    discounted_rewards = []
    cumulative = 0
    for reward in rewards[::-1]:
        cumulative = reward + gamma * cumulative  # discounted reward with gamma
        discounted_rewards.append(cumulative)
    discounted_rewards.reverse()
    if norm:
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-9)
    return discounted_rewards


def correct_covariance_matrices(matrices, min_val=1e-5, max_val=0.5):
    # Check and correct each 2x2 covariance matrix in an N x 2 x 2 array to correctly compute gradients for update. 

    # Ensure each matrix is symmetric
    matrices = (matrices + np.transpose(matrices, axes=(0, 2, 1))) / 2

    # Clip diagonal elements to be within min_val and max_val
    I = np.eye(2, dtype=bool)
    matrices[:, I] = np.clip(matrices[:, I], min_val, max_val)

    # Clip off-diagonal elements to be within -max_val and max_val
    off_diag_mask = ~I
    matrices[:, off_diag_mask] = np.clip(matrices[:, off_diag_mask], -max_val, max_val)
    
    # Ensure positive definiteness by adjusting the off-diagonal elements
    det = matrices[:, 0, 0] * matrices[:, 1, 1] - matrices[:, 0, 1] ** 2
    adjustment_needed = det <= 0
    if np.any(adjustment_needed):
        max_off_diag = np.sqrt(matrices[:, 0, 0] * matrices[:, 1, 1]) - min_val
        max_off_diag = np.clip(max_off_diag, -max_val, max_val)
        adjust_indices = np.where(adjustment_needed)[0]
        for index in adjust_indices:
            matrices[index, 0, 1] = np.sign(matrices[index, 0, 1]) * min(max_off_diag[index], np.abs(matrices[index, 0, 1]))
            matrices[index, 1, 0] = matrices[index, 0, 1]

    return matrices

def learn_diag(params, reward, newstate, state, onehotg, aprob, gamma, etas, balpha=0.0, noise=0.0, paramsindex=[], beta=1):
    # update only diagonal elements
    pc_centers, pc_sigmas, pc_constant, actor_weights, critic_weights = params
    
    # Predict place cell activations
    pcact = predict_placecell(params, state)
    newpcact = predict_placecell(params, newstate)
    
    # Predict values
    value = np.dot(pcact, critic_weights)
    newvalue = np.dot(newpcact, critic_weights)
    td = reward + gamma * newvalue - value

    l1_grad = balpha * np.sign(pc_constant)
    
    # Critic grads
    dcri = pcact[:, None] * td
    
    # Actor grads
    decay = beta * (onehotg[:, None] - aprob[:, None])
    dact = np.dot(pcact[:, None], decay.T) * td
    
    # Grads for field parameters
    post_td = (np.dot(actor_weights, decay) + critic_weights) * td

    df = state - pc_centers
    inv_sigma = invert_matrices(pc_sigmas)
    outer = np.einsum('nj,nk->njk',df,df)
    dpcs = 0.5 * (post_td * pcact[:,None])[:,:,None] * np.einsum('njl,njk,nik->nji',inv_sigma, outer, inv_sigma) * np.eye(2)
    
    dpcc = post_td * pcact[:,None] * np.einsum('nji,nj->ni', inv_sigma, df)
    dpca = (post_td * pcact[:,None] * (2/pc_constant[:,None]) - l1_grad)[:,0]

    grads = [dpcc, dpcs, dpca, dact, dcri]  # dpcc needs to be transposed back
    
    for p in range(len(params)):
        params[p] += etas[p] * grads[p]
    
    for p in paramsindex:
        ns = np.random.normal(size=params[p].shape) * noise
        params[p] += ns

    return params, grads, td