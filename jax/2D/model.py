
import jax.numpy as jnp
from jax import grad, jit, vmap, random, nn, lax,value_and_grad, lax
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def invert_matrices(tensor):
    """ Compute the inverse for each 2x2 matrix in an N x 2 x 2 tensor efficiently using vectorized operations. """
    # Extract individual matrix elements using JAX operations
    a = tensor[:, 0, 0]
    b = tensor[:, 0, 1]
    c = tensor[:, 1, 0]
    d = tensor[:, 1, 1]
    
    # Compute the determinant for each matrix
    determinant = a * d - b * c
    
    # Compute the inverse of each matrix using JAX to avoid implicit conversion issues
    inv_tensor = jnp.empty((tensor.shape[0], 2, 2))
    inv_tensor = inv_tensor.at[:, 0, 0].set(d / determinant)
    inv_tensor = inv_tensor.at[:, 0, 1].set(-b / determinant)
    inv_tensor = inv_tensor.at[:, 1, 0].set(-c / determinant)
    inv_tensor = inv_tensor.at[:, 1, 1].set(a / determinant)
    
    return inv_tensor

def uniform_2D_pc_weights(npc, nact,seed=0,sigma=0.1, alpha=1,envsize=1):
    np.random.seed(seed)
    x = np.linspace(-envsize,envsize,int(npc**0.5))
    xx,yy = np.meshgrid(x,x)
    pc_cent = np.concatenate([xx.reshape(-1)[:,None],yy.reshape(-1)[:,None]],axis=1)
    pc_sigma = np.tile(np.eye(2),(npc,1,1))*sigma
    # pc_sigma = np.tile(np.ones([2,2]),(npc,1,1))*sigma
    pc_constant = np.ones(npc) * alpha
    return [jnp.array(pc_cent), jnp.array(pc_sigma), jnp.array(pc_constant), 
    jnp.array(1e-5 * np.random.normal(size=(npc,nact))), jnp.array(1e-5 * np.random.normal(size=(npc,1)))]

def random_all_pc_weights(npc, nact,seed,sigma=0.1, alpha=1,envsize=1):
    np.random.seed(seed)
    x1 = np.random.uniform(-envsize,envsize,int(npc**0.5))
    x2 = np.random.uniform(-envsize,envsize,int(npc**0.5))[::-1]
    xx,yy = np.meshgrid(x1,x2)
    pc_cent = np.concatenate([xx.reshape(-1)[:,None],yy.reshape(-1)[:,None]],axis=1)
    pc_sigma = np.random.uniform(1e-5, sigma, size=(npc, 2,2))
    pc_sigma = correct_covariance_matrices(pc_sigma,1e-5, 0.5)
    pc_constant = np.random.uniform(0, alpha,size=npc)
    return [jnp.array(pc_cent), jnp.array(pc_sigma), jnp.array(pc_constant), 
    jnp.array(1e-5 * np.random.normal(size=(npc,nact))), jnp.array(1e-5 * np.random.normal(size=(npc,1)))]

def predict_placecell(params, x):
    pc_centers, pc_sigmas, pc_constant, actor_weights, critic_weights = params
    inv_sigma = invert_matrices(pc_sigmas)  # Shape: (npc, dim, dim)
    diff = x - pc_centers  # Shape: (npc, dim)
    exponent = jnp.einsum('ni,nij,nj->n', diff, inv_sigma, diff)
    pcacts = jnp.exp(-0.5 * exponent) * pc_constant**2
    return pcacts

def predict_batch_placecell(params, xs):  
    pcacts = []  
    for x in xs:
        pcacts.append(predict_placecell(params, x))
    pcacts = np.array(pcacts)
    return pcacts

def predict_value(params, pcact):
    pc_centers, pc_sigmas, pc_constant, actor_weights,critic_weights = params
    value = jnp.matmul(pcact, critic_weights)
    return value

def predict_action_prob(params, pcact, beta=1):
    pc_centers, pc_sigmas, pc_constant, actor_weights,critic_weights = params
    actout = jnp.matmul(pcact, actor_weights)
    aprob = nn.softmax(beta * actout)
    return aprob

def get_onehot_action(prob, nact=4):
    A = np.random.choice(a=np.arange(nact), p=np.array(prob))
    onehotg = np.zeros(nact)
    onehotg[A] = 1
    return onehotg

def compute_probas_and_values(params, coord):
    pcact = predict_placecell(params, coord)
    aprob = predict_action_prob(params, pcact)
    value = predict_value(params, pcact)
    return aprob.astype(jnp.float16), value.astype(jnp.float16)

vmap_prob_val = vmap(compute_probas_and_values, in_axes=(None, 0))

def make_correct_format(x):
    return jnp.reshape(jnp.array(x), (-1,1)).astype(jnp.float16)

def td_loss(params, coords, actions, rewards, gamma, betas):
    aprobs, values = vmap_prob_val(params, coords)
    log_likelihood = jnp.sum(jnp.log(aprobs)[:-1] * actions,axis=1)  # log probability of action as policy
    tde = jnp.array(compute_reward_prediction_error(rewards, values.reshape(-1), gamma))

    actor_loss = jnp.sum(log_likelihood * lax.stop_gradient(tde))  # log policy * discounted reward
    critic_loss = -jnp.sum(tde ** 2) # grad decent
    alpha_reg = -jnp.linalg.norm(params[2], ord=1) * (1/len(params[2]))
    tot_loss = actor_loss + 0.5 * critic_loss + betas[1] * alpha_reg
    return tot_loss

@jit
def update_td_params(params, coords, actions, rewards, etas, gamma, betas):
    loss, grads = value_and_grad(td_loss)(params, coords,actions, rewards, gamma, betas)
    pc_centers, pc_sigmas, pc_constant, actor_weights,critic_weights = params
    dpcc, dpcs, dpca, dact, dcri = grads

    # + for gradient ascent
    pc_eta, sigma_eta,constant_eta, actor_eta, critic_eta = etas
    newpc_centers = pc_centers + pc_eta * dpcc
    newpc_sigma = pc_sigmas + sigma_eta * dpcs
    newpc_const = pc_constant + constant_eta * dpca
    newactor_weights = actor_weights + actor_eta * dact
    newcritic_weights = critic_weights + critic_eta * dcri  # gradient descent
    
    # clip large fields
    # newpc_const = jnp.clip(newpc_const, 1e-5,2)
    # newpc_sigma = correct_covariance_matrices(newpc_sigma,1e-5, 0.5)
    return [newpc_centers, newpc_sigma,newpc_const, newactor_weights,newcritic_weights], grads, loss

def compute_reward_prediction_error(rewards, values, gamma=0.9):
    # new_values = jnp.concatenate([values[1:], jnp.array([[0]])])
    td = rewards + gamma * values[1:] - values[:-1]
    assert len(td.shape) == 1  # ensure values and rewards are a vectors, not matrix with T x 1. Else, TD error will be wrong
    return td

def predict_batch_pcs(params):
    x = np.linspace(-1,1,31)
    xx,yy = np.meshgrid(x,x)
    xs = np.concatenate([xx.reshape(-1)[:,None],yy.reshape(-1)[:,None]],axis=1)
    pcacts = []
    for x in xs:
        pcact = predict_placecell(params, x)
        pcacts.append(pcact)
    pcacts = np.array(pcacts)
    return pcacts


def correct_covariance_matrices_(matrices, min_val=1e-5, max_val=0.5):
    """ Correct each 2x2 covariance matrix in an N x 2 x 2 array using JAX operations. """
    # Ensure each matrix is symmetric
    matrices = (matrices + jnp.transpose(matrices, axes=(0, 2, 1))) / 2

    # Clip diagonal elements to be within min_val and max_val
    I = jnp.array([[True, False], [False, True]])  # Mask for diagonal
    matrices = matrices.at[:, I].set(jnp.clip(matrices[:, I], min_val, max_val))

    # Clip off-diagonal elements to be within -max_val and max_val
    off_diag_mask = ~I
    matrices = matrices.at[:, off_diag_mask].set(jnp.clip(matrices[:, off_diag_mask], -max_val, max_val))
    
    # Ensure positive definiteness by adjusting the off-diagonal elements
    det = matrices[:, 0, 0] * matrices[:, 1, 1] - matrices[:, 0, 1] ** 2
    adjustment_needed = det <= 0

    if jnp.any(adjustment_needed):
        max_off_diag = jnp.sqrt(matrices[:, 0, 0] * matrices[:, 1, 1]) - min_val
        max_off_diag = jnp.clip(max_off_diag, -max_val, max_val)

        # Use JAX functions to conditionally adjust matrices
        signs = jnp.sign(matrices[:, 0, 1])
        new_off_diags = signs * jnp.minimum(max_off_diag, jnp.abs(matrices[:, 0, 1]))

        # Apply adjustments where needed
        matrices = lax.map(
            lambda i, m, adj: jnp.where(adj, m.at[i, 0, 1].set(new_off_diags[i]).at[i, 1, 0].set(new_off_diags[i]), m),
            jnp.arange(matrices.shape[0]), matrices, adjustment_needed
        )

    return matrices

def correct_covariance_matrices(matrices, min_val=1e-5, max_val=0.5):
    """ Correct each 2x2 covariance matrix in an N x 2 x 2 array using JAX operations. """
    # Ensure each matrix is symmetric
    matrices = (matrices + jnp.transpose(matrices, axes=(0, 2, 1))) / 2

    # Clip diagonal elements to be within min_val and max_val
    matrices = matrices.at[:, 0, 0].set(jnp.clip(matrices[:, 0, 0], min_val, max_val))
    matrices = matrices.at[:, 1, 1].set(jnp.clip(matrices[:, 1, 1], min_val, max_val))

    # Clip off-diagonal elements to be within -max_val and max_val
    matrices = matrices.at[:, 0, 1].set(jnp.clip(matrices[:, 0, 1], -max_val, max_val))
    matrices = matrices.at[:, 1, 0].set(jnp.clip(matrices[:, 1, 0], -max_val, max_val))
    
    # Ensure positive definiteness by adjusting the off-diagonal elements
    det = matrices[:, 0, 0] * matrices[:, 1, 1] - matrices[:, 0, 1] ** 2
    adjustment_needed = det <= 0

    # Use jax.numpy functionality to adjust the matrices where needed
    matrices = vmap(adjust_matrix)(matrices, adjustment_needed, min_val, max_val)

    return matrices

def adjust_matrix(m, is_adjustment_needed, min_val=1e-5, max_val=0.5):
    if is_adjustment_needed:
        max_off_diag = jnp.sqrt(m[0, 0] * m[1, 1]) - min_val
        max_off_diag = jnp.clip(max_off_diag, -max_val, max_val)
        new_off_diag = jnp.sign(m[0, 1]) * jnp.minimum(max_off_diag, jnp.abs(m[0, 1]))
        return m.at[0, 1].set(new_off_diag).at[1, 0].set(new_off_diag)
    return m



def correct_covariance_matrices_np(matrices, min_val=1e-5, max_val=0.5):
    # Check and correct each 2x2 covariance matrix in an N x 2 x 2 array to correctly compute gradients for update. 
    matrices = np.array(matrices)
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

    return jnp.array(matrices)