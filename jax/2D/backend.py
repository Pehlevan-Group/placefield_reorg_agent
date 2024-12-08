import jax.numpy as jnp
from jax import grad, jit, vmap, random, nn, lax
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

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

def plot_2D_density(params, title):
    pcacts = predict_batch_pcs(params)
    plt.figure(figsize=(3,2))
    plt.title(title)
    plt.imshow(np.sum(pcacts,axis=1).reshape(31,31), origin='lower')
    plt.colorbar()


def uniform_2D_pc_weights_(npc, nact,seed,sigma=0.1, alpha=1,envsize=1, numsigma=2):
    x = np.linspace(-envsize,envsize,int(npc**0.5))
    xx,yy = np.meshgrid(x,x)
    pc_cent = np.concatenate([xx.reshape(-1)[:,None],yy.reshape(-1)[:,None]],axis=1)
    pc_sigma = jnp.ones([npc,numsigma])*sigma
    pc_constant = jnp.ones(npc) * alpha #/ jnp.sqrt((2*jnp.pi * jnp.sum(pc_sigma,axis=1)**2))
    actor_key, critic_key = random.split(random.PRNGKey(seed), num=2)
    return [jnp.array(pc_cent), jnp.array(pc_sigma), jnp.array(pc_constant), 
            1e-5 * random.normal(actor_key, (npc,nact)), 1e-5 * random.normal(critic_key, (npc,1))]


def predict_placecell_(params, x):
    pc_centers, pc_sigmas, pc_constant, actor_weights,critic_weights = params
    exponent = jnp.sum((x - pc_centers)**2 / (2 * pc_sigmas ** 2),axis=1)
    pcact = jnp.exp(-exponent) * pc_constant #/ jnp.sqrt((2*jnp.pi * jnp.sum(pc_sigmas,axis=1)**2))
    return pcact


def uniform_2D_pc_weights(npc, nact,seed=0,sigma=0.1, alpha=1,envsize=1):
    x = np.linspace(-envsize,envsize,int(npc**0.5))
    xx,yy = np.meshgrid(x,x)
    pc_cent = np.concatenate([xx.reshape(-1)[:,None],yy.reshape(-1)[:,None]],axis=1)
    inv_sigma = np.linalg.inv(np.tile(np.eye(2),(npc,1,1))*sigma)
    pc_constant = np.ones(npc) * alpha
    actor_key, critic_key = random.split(random.PRNGKey(seed), num=2)
    return [jnp.array(pc_cent), jnp.array(inv_sigma), jnp.array(pc_constant), 
            1e-5 * random.normal(actor_key, (npc,nact)), 1e-5 * random.normal(critic_key, (npc,1))]


def predict_placecell(params, x):
    pc_centers, inv_sigma, pc_constant, actor_weights, critic_weights = params
    diff = x - pc_centers  # Shape: (npc, dim)
    exponent = jnp.einsum('ni,nij,nj->n', diff, inv_sigma, diff)
    pcacts = jnp.exp(-0.5 * exponent) * pc_constant**2
    return pcacts



def predict_value(params, pcact):
    pc_centers, pc_sigmas, pc_constant, actor_weights,critic_weights = params
    value = jnp.matmul(pcact, critic_weights)
    return value



def predict_action(params, pcact, beta=2):
    pc_centers, pc_sigmas, pc_constant, actor_weights,critic_weights = params
    actout = jnp.matmul(pcact, actor_weights)
    aprob = nn.softmax(beta * actout)
    return aprob

def pg_loss(params, coords, actions, discount_rewards):
    aprobs = []
    for coord in coords:
        pcact = predict_placecell(params, coord)
        aprob = predict_action(params, pcact)
        aprobs.append(aprob)
    aprobs = jnp.array(aprobs)
    neg_log_likelihood = jnp.log(aprobs) * actions  # log probability of action as policy
    weighted_rewards = lax.stop_gradient(jnp.array(discount_rewards)[:,None])
    tot_loss = jnp.sum(jnp.array(neg_log_likelihood * weighted_rewards))  # log policy * discounted reward
    return tot_loss


@jit
def update_params(params, coords, actions, discount_rewards, etas):
    grads = grad(pg_loss)(params, coords,actions, discount_rewards)
    pc_centers, pc_sigmas, pc_constant, actor_weights,critic_weights = params
    dpcc, dpcs, dpca, dact, dcri = grads

    # + for gradient ascent
    pc_eta, sigma_eta,constant_eta, actor_eta, critic_eta = etas
    newpc_centers = pc_centers + pc_eta * dpcc
    newpc_sigma = pc_sigmas + sigma_eta * dpcs
    newpc_const = pc_constant + constant_eta * dpca
    newactor_weights = actor_weights + actor_eta * dact
    newcritic_weights = critic_weights + critic_eta * dcri  # gradient descent
    return [newpc_centers, newpc_sigma,newpc_const, newactor_weights,newcritic_weights], grads


def a2c_loss(params, coords, actions, discount_rewards):
    aprobs = []
    values = []
    for coord in coords:
        pcact = predict_placecell(params, coord)
        aprob = predict_action(params, pcact)
        value = predict_value(params, pcact)
        aprobs.append(aprob)
        values.append(value)
    aprobs = jnp.array(aprobs)
    values = jnp.array(values)

    log_likelihood = jnp.log(aprobs) * actions  # log probability of action as policy
    advantage = jnp.array(discount_rewards)[:,None] - values

    actor_loss = jnp.sum(log_likelihood * lax.stop_gradient(advantage))  # log policy * discounted reward
    critic_loss = -jnp.sum(advantage ** 2) # grad decent
    tot_loss = actor_loss + 0.1 * critic_loss
    return tot_loss

@jit
def update_a2c_params(params, coords, actions, discount_rewards, etas):
    grads = grad(a2c_loss)(params, coords,actions, discount_rewards)
    pc_centers, pc_sigmas, pc_constant, actor_weights,critic_weights = params
    dpcc, dpcs, dpca, dact, dcri = grads

    # + for gradient ascent
    pc_eta, sigma_eta,constant_eta, actor_eta, critic_eta = etas
    newpc_centers = pc_centers + pc_eta * dpcc
    newpc_sigma = pc_sigmas + sigma_eta * dpcs
    newpc_const = pc_constant + constant_eta * dpca
    newactor_weights = actor_weights + actor_eta * dact
    newcritic_weights = critic_weights + critic_eta * dcri  # gradient descent
    return [newpc_centers, newpc_sigma,newpc_const, newactor_weights,newcritic_weights], grads

def get_onehot_action(prob, nact=4):
    A = np.random.choice(a=np.arange(nact), p=np.array(prob))
    onehotg = np.zeros(nact)
    onehotg[A] = 1
    return onehotg

def get_discounted_rewards(rewards, gamma=0.95, norm=False):
    discounted_rewards = []
    cumulative = 0
    for reward in rewards[::-1]:
        cumulative = reward + gamma * cumulative  # discounted reward with gamma
        discounted_rewards.append(cumulative)
    discounted_rewards.reverse()
    if norm:
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-9)
    return discounted_rewards

def compute_reward_prediction_error(rewards, values, gamma=0.95):
    new_values = jnp.concatenate([values[1:], jnp.array([[0]])])
    td = rewards + gamma * new_values - values
    return td

def plot_place_cells(params,title='', num=10, goalcoord=None, goalsize=0.2, obstacles=False):
    pc_centers, pc_sigmas, pc_constant, actor_weights,critic_weights = params
    print(pc_sigmas.shape)
    plt.figure(figsize=(3,2))
    plt.title(title)
    if goalcoord is not None:
        circle = plt.Circle(goalcoord, goalsize, color='r', fill=True, zorder=2)
        plt.gca().add_patch(circle)
    
    if obstacles:
        from matplotlib.patches import Rectangle
        plt.gca().add_patch(Rectangle((-0.6,0.3), 0.2, 0.7, facecolor='grey'))  # top left
        plt.gca().add_patch(Rectangle((0.4,0.3), 0.2, 0.7, facecolor='grey'))  # top right
        plt.gca().add_patch(Rectangle((-0.6,-0.3), 0.2, -0.7, facecolor='grey'))  # bottom left
        plt.gca().add_patch(Rectangle((0.4,-0.3), 0.2, -0.7, facecolor='grey'))  # bottom right
    if pc_sigmas.shape[1]>1:
        from matplotlib.patches import Ellipse
        for pc, pcr in zip(pc_centers[num],pc_sigmas[num]):
            ellipse = Ellipse(xy=pc, width=2*np.sqrt(2*pcr[0]**2),height=2*np.sqrt(2*pcr[1]**2), zorder=1, edgecolor='g', fc='None')
            plt.gca().add_patch(ellipse)
            plt.scatter(pc[0],pc[1],s=5,color='purple',zorder=3)
    else:
        for pc, pcr in zip(pc_centers[num],pc_sigmas[num]):
            circle = plt.Circle(pc, np.sqrt(2*pcr**2), color='g', fill=False, zorder=1)
            plt.gca().add_patch(circle)
            plt.scatter(pc[0],pc[1],s=5,color='purple',zorder=3)
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xticks(np.linspace(-1,1,3))
    plt.yticks(np.linspace(-1,1,3))
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    # plt.xlabel('X')
    # plt.ylabel('Y')
    plt.tight_layout()

def plot_maps(actor_weights,critic_weights, env, npc, title=None):
    npcs = int(npc**0.5)
    plt.figure(figsize=(3,2))
    plt.imshow(critic_weights.reshape([npcs, npcs]), origin='lower')
    plt.colorbar()
    dirction = np.matmul(actor_weights, env.onehot2dirmat)
    xx, yy = np.meshgrid(np.arange(npcs), np.arange(npcs))
    plt.quiver(xx.reshape(-1),yy.reshape(-1), dirction[:,0], dirction[:,1], color='k', scale_units='xy')
    plt.gca().set_aspect('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Value & Policy maps')
    plt.tight_layout()

def saveload(filename, variable, opt):
    import pickle
    if opt == 'save':
        with open(f"{filename}.pickle", "wb") as file:
            pickle.dump(variable, file)
        print('file saved')
    else:
        with open(f"{filename}.pickle", "rb") as file:
            return pickle.load(file)


def moving_average(signal, window_size):
    # Pad the signal to handle edges properly
    padded_signal = np.pad(signal, (window_size//2, window_size//2), mode='edge')
    
    # Apply the moving average filter
    weights = np.ones(window_size) / window_size
    smoothed_signal = np.convolve(padded_signal, weights, mode='valid')
    
    return smoothed_signal

class NDimNav:
    def __init__(self,nact=4,maxspeed=0.1, envsize=1, goalsize=0.1, tmax=300, goalcoord=[0.8,0.8], startcoord=[[-0.8,-0.8]], max_reward=5, obstacles=False) -> None:
        self.tmax = tmax  # maximum steps per trial
        self.minsize = -envsize  # arena size
        self.maxsize = envsize
        self.state = np.zeros(2)
        self.done = False
        self.goalsize = goalsize
        self.goals = np.array(goalcoord)
        self.starts = np.array(startcoord)
        self.statesize = 2
        self.actionsize = nact
        self.maxspeed = maxspeed  # max agent speed per step
        self.tauact = 0.2
        self.total_reward = 0
        self.obstacles = obstacles
        self.max_reward = max_reward
        self.starts = np.array(startcoord)
        self.reward_type = 'gauss'
        self.amp = 1

        # convert agent's onehot vector action to direction in the arena
        self.onehot2dirmat = np.array([
            [0,1],  # up
            [1,0],  # right
            [0,-1],  # down
            [-1,0]  # left
        ])

    def reward_func(self,x, threshold=1e-2):
        rx =  self.amp * np.exp(-0.5*np.linalg.norm(x - self.goal)**2/self.goalsize**2)
        return rx * (rx>threshold)
    
    def action2velocity(self, g):
        # convert onehot action vector from actor to velocity
        return np.matmul(g, self.onehot2dirmat)

    
    def reset(self):
        if len(self.starts) > 2:
            startidx = np.random.choice(np.arange(len(self.starts)),1)
            self.state = self.starts[startidx].copy()[0]
        else:
            self.state = self.starts.copy()

        self.goal = self.goals.copy()
        self.error = self.goal - self.state
        self.eucdist = np.linalg.norm(self.error,ord=2)
        self.done = False
        self.t = 0
        self.reward = 0
        self.total_reward = 0

        self.track = []
        self.track.append(self.goal.copy())
        self.track.append(self.state.copy())

        self.velocity = np.zeros(self.statesize)

        #print(f"State: {self.state}, Goal: {self.goal}")
        return self.state, self.goal, self.reward, self.done

    
    def step(self, g):
        self.t +=1
        newvelocity = self.action2velocity(g) * self.maxspeed  # get velocity from agent's onehot action
        self.velocity += self.tauact * (-self.velocity + newvelocity)
        newstate = self.state.copy() + self.velocity

        self.track.append(self.state.copy())

        # check if new state crosses boundary
        if (newstate > self.maxsize).any() or (newstate < self.minsize).any():
            newstate = self.state.copy()
            self.velocity = np.zeros(self.statesize)

        # check if new state crosses obstacles if initalized
        if self.obstacles:
            if  -0.6 < newstate[0] < -0.3 and  0.25 < newstate[1] < 1:  # top left obs
                newstate = self.state.copy()
                self.velocity = np.zeros(self.statesize)
            
            if  -0.6 < newstate[0] < -0.3 and  -1 < newstate[1] < -0.25:  # bottom left obs
                newstate = self.state.copy()
                self.velocity = np.zeros(self.statesize)
            
            if  0.3 < newstate[0] < 0.6 and  0.25 < newstate[1] < 1:  # top right obs
                newstate = self.state.copy()
                self.velocity = np.zeros(self.statesize)
            
            if  0.3 < newstate[0] < 0.6 and  -1 < newstate[1] < -0.25:  # top right obs
                newstate = self.state.copy()
                self.velocity = np.zeros(self.statesize)
        
        # if new state does not violate boundary or obstacles, update new state
        self.state = newstate.copy()
        self.error = self.goal - self.state
        self.eucdist = np.linalg.norm(self.error,ord=2)

        # check if agent is within radius of goal
        self.reward = 0
        if self.reward_type == 'box':
            if (self.eucdist < self.goalsize).any():
                self.reward = 1
                self.total_reward +=1 

        elif self.reward_type == 'gauss':
            self.reward = self.reward_func(self.state, threshold=0)
            # self.reward *= reward_mod
            if self.reward >1e-2:
                self.total_reward +=1
        
        if self.total_reward == self.max_reward:
            self.done = True
        
        if self.t == self.tmax:
            self.done = True
       
        return self.state, self.reward, self.done

    def plot_trajectory(self, title=None):
        plt.figure(figsize=(3,2))
        plt.title(f'2D {title}')
        plt.axis([self.minsize, self.maxsize, self.minsize, self.maxsize])

        if self.obstacles:
            from matplotlib.patches import Rectangle
            plt.gca().add_patch(Rectangle((-0.6,0.3), 0.2, 0.7, facecolor='grey'))  # top left
            plt.gca().add_patch(Rectangle((0.4,0.3), 0.2, 0.7, facecolor='grey'))  # top right
            plt.gca().add_patch(Rectangle((-0.6,-0.3), 0.2, -0.7, facecolor='grey'))  # bottom left
            plt.gca().add_patch(Rectangle((0.4,-0.3), 0.2, -0.7, facecolor='grey'))  # bottom right

        #plt.scatter(np.array(self.track)[0,0],np.array(self.track)[0,1], color='r', zorder=2, )    
        circle = plt.Circle(xy=self.goal, radius=self.goalsize, color='r')
        plt.gca().add_patch(circle)
        plt.scatter(np.array(self.track)[1,0],np.array(self.track)[1,1], color='g', zorder=2)    
        plt.plot(np.array(self.track)[1:,0],np.array(self.track)[1:,1], marker='o',color='b', zorder=1)

        plt.gca().set_aspect('equal')
   


class TwoDimNav:
    def __init__(self,obstacles=False, maxspeed=0.1, envsize=1, goalsize=0.1, tmax=100, goalcoord=[0,0.8], startcoord='corners') -> None:
        self.tmax = tmax  # maximum steps per trial
        self.minsize = -envsize  # arena size
        self.maxsize = envsize
        self.state = np.zeros(2)
        self.done = False
        self.goalsize = goalsize
        self.tauact = 0.25

        self.statesize = 2 # start + goal information
        self.goal = np.array(goalcoord)

        if startcoord =='corners':  # agent starts from one of 4 corners
            self.starts = np.array([[-0.8,-0.8], [-0.8,0.8], [0.8,0.8], [0.8,-0.8]],dtype=np.float32)
        elif startcoord == 'center':
            self.starts = np.array([0.0,0.0])  # agent starts from the center
        else:
            self.starts = np.array(startcoord)

        self.actionsize = 4
        self.maxspeed = maxspeed  # max agent speed per step

        self.obstacles = obstacles

        # convert agent's onehot vector action to direction in the arena
        self.onehot2dirmat = np.array([
            [0,1],  # up
            [1,0],  # right
            [0,-1],  # down
            [-1,0]  # left
        ])
    
    def action2velocity(self, g):
        # convert onehot action vector from actor to velocity
        return np.matmul(g, self.onehot2dirmat)

    
    def reset(self):

        if len(self.starts) > 2:
            startidx = np.random.choice(np.arange(len(self.starts)),1)
            self.state = self.starts[startidx].copy()[0]
        else:
            self.state = self.starts.copy()

        self.error = self.goal - self.state
        self.eucdist = np.linalg.norm(self.error,ord=2)
        self.done = False
        self.t = 0

        self.track = []
        self.track.append(self.goal.copy())
        self.track.append(self.state.copy())

        self.velocity = np.zeros(self.statesize)

        #print(f"State: {self.state}, Goal: {self.goal}")
        return self.state, self.goal, self.error, self.done

    
    def step(self, g):
        self.t +=1
        newvelocity = self.action2velocity(g) * self.maxspeed  # get velocity from agent's onehot action
        self.velocity += self.tauact * (-self.velocity + newvelocity)
        newstate = self.state.copy() + self.velocity
        
        self.track.append(self.state.copy())

        # check if new state crosses boundary
        if (newstate > self.maxsize).any() or (newstate < self.minsize).any():
            newstate = self.state.copy()
            self.velocity = np.zeros(self.statesize)

        # check if new state crosses obstacles if initalized
        if self.obstacles:
            if  -0.6 < newstate[0] < -0.3 and  0.25 < newstate[1] < 1:  # top left obs
                newstate = self.state.copy()
                self.velocity = np.zeros(self.statesize)
            
            if  -0.6 < newstate[0] < -0.3 and  -1 < newstate[1] < -0.25:  # bottom left obs
                newstate = self.state.copy()
                self.velocity = np.zeros(self.statesize)
            
            if  0.3 < newstate[0] < 0.6 and  0.25 < newstate[1] < 1:  # top right obs
                newstate = self.state.copy()
                self.velocity = np.zeros(self.statesize)
            
            if  0.3 < newstate[0] < 0.6 and  -1 < newstate[1] < -0.25:  # top right obs
                newstate = self.state.copy()
                self.velocity = np.zeros(self.statesize)
        
        # if new state does not violate boundary or obstacles, update new state
        self.state = newstate.copy()
        self.error = self.goal - self.state
        self.eucdist = np.linalg.norm(self.error,ord=2)

        # check if agent is within radius of goal
        if self.eucdist < self.goalsize:
            self.done = True

        return self.state, self.error, self.done

    def random_action(self):
        action = np.random.uniform(low=-1, high=1,size=self.statesize)
        return action 

    def plot_trajectory(self, title=None):
        plt.figure(figsize=(3,2))
        plt.title(f'2D {title}')
        plt.axis([self.minsize, self.maxsize, self.minsize, self.maxsize])

        if self.obstacles:
            from matplotlib.patches import Rectangle
            plt.gca().add_patch(Rectangle((-0.6,0.3), 0.2, 0.7, facecolor='grey'))  # top left
            plt.gca().add_patch(Rectangle((0.4,0.3), 0.2, 0.7, facecolor='grey'))  # top right
            plt.gca().add_patch(Rectangle((-0.6,-0.3), 0.2, -0.7, facecolor='grey'))  # bottom left
            plt.gca().add_patch(Rectangle((0.4,-0.3), 0.2, -0.7, facecolor='grey'))  # bottom right

        #plt.scatter(np.array(self.track)[0,0],np.array(self.track)[0,1], color='r', zorder=2, )    
        circle = plt.Circle(xy=self.goal, radius=self.goalsize, color='r')
        plt.gca().add_patch(circle)
        plt.scatter(np.array(self.track)[1,0],np.array(self.track)[1,1], color='g', zorder=2)    
        plt.plot(np.array(self.track)[1:,0],np.array(self.track)[1:,1], marker='o',color='b', zorder=1)

        plt.gca().set_aspect('equal')

        
class PC_AC_agent:
    def __init__(self, npc=21,pcr=0.25, nact=4, alr=0.0075, clr=0.025):
        self.npc = npc  # number of place cells tiling each dimension
        self.alr = alr  # actor learning rate
        self.clr = clr  # critic learning rate
        self.pcspacing = np.linspace(-1,1,self.npc) # uniformly space place cells
        self.pcr =  pcr # define radius of place cells

        xx, yy = np.meshgrid(self.pcspacing, self.pcspacing)
        self.pcs = np.concatenate([xx.reshape(-1,1), yy.reshape(-1,1)],axis=1)

        self.nact = nact  # number of action units
        self.wC = np.random.normal(loc=0,scale=0.001, size=[len(self.pcs), 1]) #np.zeros([len(self.pcs), 1])  # critic weight matrix
        self.wA = np.random.normal(loc=0,scale=0.001, size=[len(self.pcs), nact]) #np.zeros([len(self.pcs), nact])  # actor weight matrix
        self.gamma = 0.95  # discount factor
        self.beta = 2  # action temperature hyperparameters, higher --> more exploitation
    
    def get_pc(self, x):
        # convert x,y coordinate to place cell activity
        norm = np.sum((x - self.pcs)**2,axis=1)
        pcact = np.exp(-norm / (2 * self.pcr **2))
        return pcact
    
    def softmax(self, prob):
        return np.exp(prob) / np.sum(np.exp(prob))
    
    def get_action(self, x):
        # get place cell activity
        self.h = self.get_pc(x)

        # get critic activity
        self.V = np.matmul(self.h, self.wC)

        # get actor activity
        self.A = np.matmul(self.h, self.wA)

        # choose action using stochastic policy
        self.prob = self.softmax(self.beta* self.A)
        A = np.random.choice(np.arange(self.nact), p=self.prob)

        # convert action to onehot
        self.onehotg = np.zeros(self.nact)
        self.onehotg[A] = 1
        return self.onehotg

    def learn(self, newstate, reward):
        # get value estimate of new state using current critic weights     
        self.V1 = np.matmul(self.get_pc(newstate), self.wC)

        # compute TD error 
        self.td = int(reward) + self.gamma * self.V1 - self.V

        # update weights at each timestep when TD is computed using 2 & 3 factor Hebbian rule
        self.wC += self.clr * self.h[:,None] * self.td
        self.wA += self.alr * np.matmul(self.h[:,None], self.onehotg[:,None].T) * self.td
        
    
    def plot_maps(self, env, title=None):
        plt.figure()
        plt.title(title)
        plt.imshow(self.wC.reshape(self.npc, self.npc), origin='lower')
        plt.colorbar()
        dir = np.matmul(self.wA, env.onehot2dirmat)
        xx, yy = np.meshgrid(np.arange(self.npc), np.arange(self.npc))
        plt.quiver(xx.reshape(-1),yy.reshape(-1), dir[:,0], dir[:,1], color='w', scale_units='xy',scale=None)
        plt.show()

def get_2D_freq_density_corr(allcoords, logparams, end, gap=20, bins=10):
    coord = []
    for t in range(gap):
        for c in allcoords[end-t-1]:
            coord.append(c)
    coord = np.array(coord)
    x = np.linspace(-1,1,bins+1)
    xx,yy = np.meshgrid(x,x)
    x = np.concatenate([xx.reshape(-1)[:,None],yy.reshape(-1)[:,None]],axis=1)
    coord = np.concatenate([coord, x],axis=0)

    hist, x_edges, y_edges = np.histogram2d(coord[:, 0], coord[:, 1], bins=[bins, bins])

    xs = x_edges[:-1] + (x_edges[1] - x_edges[0])/2 
    ys = y_edges[:-1] + (y_edges[1] - y_edges[0])/2 

    xx,yy = np.meshgrid(xs,ys)
    visits = np.concatenate([xx.reshape(-1)[:,None],yy.reshape(-1)[:,None]],axis=1)
    freq = hist.reshape(-1)

    param = logparams[end-1]
    pcacts = []
    for x in visits:
        pcacts.append(predict_placecell(param, x))
    pcacts = np.array(pcacts)
    dxs = np.sum(pcacts,axis=1)

    correlation_coefficient = np.corrcoef(freq, dxs)[0, 1]
    print(correlation_coefficient)
    return xs, visits, freq, dxs, correlation_coefficient

def plot_freq_density_corr(xs, freq, density, bins, title):
    index = np.linspace(0,bins-1,num=4,dtype=int)
    plt.figure(figsize=(6,2))
    plt.suptitle(title)
    plt.subplot(131)
    plt.title('Frequency')
    plt.imshow(freq.reshape(bins,bins), origin='lower')
    plt.xticks(index, np.round(xs[index],1))
    plt.yticks(index, np.round(xs[index],1))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar()
    plt.subplot(132)
    plt.title('Density')
    plt.imshow(density.reshape(bins,bins), origin='lower')
    plt.xticks(index, np.round(xs[index],1))
    plt.yticks(index, np.round(xs[index],1))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar()
    plt.subplot(133)
    plt.scatter(freq, density)
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(freq).reshape(-1), np.array(density).reshape(-1))
    regression_line = slope * np.array(freq).reshape(-1) + intercept
    plt.plot(np.array(freq).reshape(-1), regression_line, color='red', label=f'R:{np.round(r_value,3)}, P:{np.round(p_value,3)}')
    plt.legend(frameon=False, fontsize=8)
    plt.title('Correlation')
    plt.xlabel('Frequency f(x)')
    plt.ylabel('Density d(x)')
    plt.tight_layout()