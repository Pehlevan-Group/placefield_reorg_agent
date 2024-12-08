import numpy as np
import matplotlib.pyplot as plt


class NDimNav:
    def __init__(self,nact=4,maxspeed=0.1, envsize=1, goalsize=0.1, tmax=300, goalcoord=[0.8,0.8], startcoord=[[-0.8,-0.8]], max_reward=5, obstacles=True, obscoord=[-0.2,0.2,-1,0.5],rtype='gauss') -> None:
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
        self.reward_type = rtype
        self.amp = 1
        self.reward_threshold = 1e-2 # cut off for gaussian reward, positive only above this threshold, else reward = 0. Impt for env with obstacle.

        self.obscoords = obscoord

        # convert agent's onehot vector action to direction in the arena
        self.onehot2dirmat = np.array([
            [0,1],  # up
            [1,0],  # right
            [0,-1],  # down
            [-1,0]  # left
        ])

    def reward_func(self,x, threshold=0):
        rx =  self.amp * np.exp(-0.5 * np.sum(((x - self.goal) / self.goalsize) ** 2))
        return rx * (rx>threshold)
    
    def action2velocity(self, g):
        # convert onehot action vector from actor to velocity
        return np.matmul(g, self.onehot2dirmat)

    
    def reset(self):
        if len(self.starts) > 1:
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
            if self.obscoords[0]<newstate[0] <self.obscoords[1] and self.obscoords[2]<newstate[1] <self.obscoords[3]:
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
            self.reward = self.reward_func(self.state, threshold=self.reward_threshold)
            self.total_reward +=self.reward
        
        if self.total_reward >= self.max_reward:
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
            L = self.obscoords[0]
            T = self.obscoords[3]
            dfL = self.obscoords[1]-self.obscoords[0]
            dfT = self.obscoords[2]-self.obscoords[3]
            plt.gca().add_patch(Rectangle((L,T), dfL, dfT, facecolor='grey')) 

        #plt.scatter(np.array(self.track)[0,0],np.array(self.track)[0,1], color='r', zorder=2, )    
        circle = plt.Circle(xy=self.goal, radius=self.goalsize, color='r')
        plt.gca().add_patch(circle)
        plt.scatter(np.array(self.track)[1,0],np.array(self.track)[1,1], color='g', zorder=2)    
        plt.plot(np.array(self.track)[1:,0],np.array(self.track)[1:,1], marker='o',color='b', zorder=1)

        plt.gca().set_aspect('equal')
   