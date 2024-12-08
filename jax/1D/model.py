
import jax.numpy as jnp
from jax import grad, jit, vmap, random, nn, lax, value_and_grad
import numpy as np

def random_all_pc_weights(npc, nact,seed,sigma=0.1, alpha=1,envsize=1):
    keys = random.split(random.PRNGKey(seed), num=3)

    pc_cent =  random.uniform(keys[0], shape=(npc,),minval=-envsize,maxval=envsize) 
    pc_sigma = random.uniform(keys[1], shape=(npc,),minval=1e-5,maxval=sigma) 
    pc_constant = random.uniform(keys[2], shape=(npc,),minval=0,maxval=alpha) 
    
    return [jnp.array(pc_cent), jnp.array(pc_sigma), jnp.array(pc_constant), 
    1e-5 * random.normal(keys[3], (npc,nact)), 1e-5 * random.normal(keys[4], (npc,1))]

def uniform_pc_weights_(npc, nact,seed,sigma=0.1, alpha=1,envsize=1):
    pc_cent =  jnp.linspace(-envsize,envsize,npc) 
    pc_sigma = jnp.ones(npc)*sigma
    pc_constant = jnp.ones(npc)*alpha 
    
    actor_key, critic_key = random.split(random.PRNGKey(seed), num=2)
    return [jnp.array(pc_cent), jnp.array(pc_sigma), jnp.array(pc_constant), 
    1e-5 * random.normal(actor_key, (npc,nact)), 1e-5 * random.normal(critic_key, (npc,1))]

def uniform_pc_weights(npc, nact,seed,sigma=0.1, alpha=1,envsize=1):  # homogeneous population
    pc_cent =  np.linspace(-envsize,envsize,npc) 
    pc_sigma = np.ones(npc)*sigma
    pc_constant = np.ones(npc)*alpha 
    np.random.seed(seed)
    actorw = 1e-5 * np.random.normal(size=(npc,nact))
    criticw = 1e-5 * np.random.normal(size=(npc,1))
    return [jnp.array(pc_cent), jnp.array(pc_sigma), jnp.array(pc_constant), 
    jnp.array(actorw), jnp.array(criticw)]

def predict_placecell(params, x):
    pc_centers, pc_sigmas, pc_constant, actor_weights,critic_weights = params
    exponent = ((x-pc_centers)/pc_sigmas)**2
    pcact = jnp.exp(-0.5*exponent) * pc_constant**2 #1/jnp.sqrt(2*jnp.pi*pc_sigmas**2)
    return pcact


def predict_value(params, pcact):
    pc_centers, pc_sigmas, pc_constant, actor_weights,critic_weights = params
    value = jnp.matmul(pcact, critic_weights)
    return value

def predict_action_prob(params, pcact, beta=1):
    pc_centers, pc_sigmas, pc_constant, actor_weights,critic_weights = params
    actout = jnp.matmul(pcact, actor_weights)
    aprob = nn.softmax(beta * actout)
    return aprob


def compute_probas_and_values(params, coord):
    pcact = predict_placecell(params, coord)
    aprob = predict_action_prob(params, pcact)
    value = predict_value(params, pcact)
    return aprob.astype(jnp.float16), value.astype(jnp.float16)

vmap_prob_val = vmap(compute_probas_and_values, in_axes=(None, 0))

def td_loss(params, coords, actions, rewards, gamma, betas):
    aprobs, values = vmap_prob_val(params, coords)
    log_likelihood = jnp.sum(jnp.log(aprobs)[:-1] * actions,axis=1)  # log probability of action as policy
    tde = jnp.array(compute_reward_prediction_error(rewards, values.reshape(-1), gamma))

    actor_loss = jnp.sum(log_likelihood * lax.stop_gradient(tde))  # maximize log policy * discounted reward
    critic_loss = -jnp.sum(tde ** 2) # minimize TD error
    alpha_reg = -jnp.linalg.norm(params[2], ord=1) # minimize L2 norm
    tot_loss = actor_loss + betas[0] * critic_loss + betas[1] * alpha_reg
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
    return [newpc_centers, newpc_sigma,newpc_const, newactor_weights,newcritic_weights], grads, loss

def get_onehot_action(prob, nact=3):
    A = np.random.choice(a=np.arange(nact), p=np.array(prob))
    onehotg = np.zeros(nact)
    onehotg[A] = 1
    return onehotg

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

def compute_reward_prediction_error(rewards, values, gamma=0.9):
    # new_values = jnp.concatenate([values[1:], jnp.array([[0]])])
    td = rewards + gamma * values[1:] - values[:-1]
    assert len(td.shape) == 1  # ensure values and rewards are a vectors, not matrix with T x 1. Else, TD error will be wrong
    return td

def predict_batch_placecell(params, xs):  
    pcacts = []  
    for x in xs:
        pcacts.append(predict_placecell(params, x))
    pcacts = np.array(pcacts)
    return pcacts