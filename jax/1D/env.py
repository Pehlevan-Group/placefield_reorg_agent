import numpy as np
import matplotlib.pyplot as plt


class OneDimNav:
    def __init__(self,nact,maxspeed=0.1, envsize=1, goalsize=0.1, tmax=100, goalcoord=[0.8], startcoord=[-0.8], initvelocity=0.0, max_reward=5) -> None:
        self.tmax = tmax  # maximum steps per trial
        self.minsize = -envsize  # arena size
        self.maxsize = envsize
        self.state = 0
        self.done = False
        self.goalsize = goalsize
        self.goals = np.array(goalcoord)
        self.starts = np.array(startcoord)
        self.statesize = 1
        self.actionsize = nact
        self.maxspeed = maxspeed  # max agent speed per step
        self.tauact = 0.2  # smooth action transition
        self.total_reward = 0
        self.initvelocity = np.array(initvelocity)
        self.max_reward = max_reward
        self.reward_type = 'gauss' # gauss or box
        self.amp = 1 # reward magnitude
        self.reward_threshold = 0  # above which reward will follow gaus distribution

        # convert agent's onehot vector action to direction in the arena
        if self.actionsize ==3:
            self.onehot2dirmat = np.array([[-1], [1], [0]])  # actions: move left, right, lick/stay
        else:
            self.onehot2dirmat = np.array([[-1], [1]])  # actions: move left, right

    def reward_func(self,x, threshold=0):
        rx =  self.amp * np.exp(-0.5*((x - self.goal)/self.goalsize)**2)
        return rx * (rx>threshold)
    
    def action2velocity(self, g):
        # convert onehot action vector from actor to velocity
        return np.matmul(g, self.onehot2dirmat)

    
    def reset(self):
        if len(self.starts) > 1:  # choose from multiple start locations
            startidx = np.random.choice(np.arange(len(self.starts)),1)
            self.state = self.starts[startidx].copy()
            if len(self.goals)>1:
                self.goal = self.goals[startidx].copy()
            else:
                self.goal = self.goals.copy()
        else:
            self.state = self.starts.copy()
            self.goal = self.goals.copy()

        self.error = self.goal - self.state
        self.eucdist = abs(self.error)
        self.done = False
        self.t = 0
        self.reward = 0
        self.total_reward = 0

        self.track = []
        self.track.append(self.goal.copy())
        self.track.append(self.state.copy())

        self.velocity = np.zeros(self.statesize)
        self.velocity += self.initvelocity
        return self.state, self.goal, self.reward, self.done

    
    def step(self, g):
        self.t +=1
        acceleration = self.action2velocity(g) * self.maxspeed  # get velocity from agent's onehot action

        # self.velocity += self.tauact * (newvelocity)  # smoothen actions so that agent explores the entire arena. From Foster et al. 2000
        self.velocity += self.tauact * (-self.velocity + acceleration)
        newstate = self.state.copy() + self.velocity

        self.track.append(self.state.copy())

        # check if new state crosses boundary
        if newstate > self.maxsize or newstate < -self.maxsize:
            newstate = self.state.copy()
            self.velocity = np.zeros(self.statesize)
        
        # if new state does not violate boundary or obstacles, update new state
        self.state = newstate.copy()
        self.error = self.goal - self.state
        self.eucdist = abs(self.error)

        # check if agent is within radius of goal
        self.reward = 0
        if self.reward_type == 'box':
            if (self.eucdist < self.goalsize).any():
                # if nact = 2, agent needs to be in the vicinty of goal to get a reward
                if self.actionsize == 2:
                    self.reward = 1
                    self.total_reward +=1 

                # if nact = 3, agent has to lick to get a reward, not merely be in vicinty of goal
                if self.actionsize == 3 and acceleration == 0:
                    self.reward = 1
                    self.total_reward +=1 

        elif self.reward_type == 'gauss':
            self.reward = self.reward_func(self.state, threshold=self.reward_threshold)
            self.total_reward += self.reward

        if self.total_reward >= self.max_reward:
            self.done = True
        
        if self.t == self.tmax:
            self.done = True
       

        return self.state, self.reward, self.done

    def random_action(self):
        action = np.random.uniform(low=-1, high=1,size=self.actionsize)
        return action 

    def plot_trajectory(self, title=None):
        plt.figure(figsize=(4,2))
        plt.title(f'1D {title}')
        plt.hlines(xmin=self.minsize,xmax=self.maxsize, y=1, colors='k')
        plt.eventplot(self.track[1], color='g', zorder=2)
        plt.eventplot(self.track[0], color='orange', zorder=2)
        for i,s in enumerate(self.track):
            if i == 0:
                plt.eventplot(s, color='orange')
            elif i == 1:
                plt.eventplot(s, color='g') 
            else:
                plt.eventplot(s, color='b', zorder=1) 
   

