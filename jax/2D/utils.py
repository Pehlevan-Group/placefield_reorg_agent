import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import os
import csv
from io import BytesIO
from model import predict_batch_placecell
from matplotlib.patches import Rectangle
from jax import nn

def plot_analysis(logparams,latencys,allrewards, allcoords, stable_perf, exptname=None , rsz=0.05):
    f, axs = plt.subplots(7,3,figsize=(12,21))
    total_trials = len(latencys)
    gap = 25

    #latency 
    score = plot_latency(latencys,allrewards, ax=axs[0,0])

    plot_pc(logparams, 0,ax=axs[0,1], title='Before Learning', goalsize=rsz)

    plot_pc(logparams, total_trials,ax=axs[0,2], title='After Learning', goalsize=rsz)


    plot_value(logparams, total_trials, ax=axs[3,0], goalsize=rsz)


    plot_velocity(logparams,  total_trials,ax=axs[1,0], goalsize=rsz)



    ## high d at reward
    dx = plot_density(logparams,  total_trials, ax=axs[1,1], goalsize=rsz)

    fx = plot_frequency(allcoords,  total_trials, ax=axs[1,2], gap=gap, goalsize=rsz)

    plot_fx_dx(allcoords, logparams, trial=gap, title='Before',gap=gap,ax=axs[2,0])
    
    plot_fx_dx(allcoords, logparams, trial=total_trials, title='After',gap=gap,ax=axs[2,1])

    plot_fxdx_trials(allcoords, logparams, np.linspace(gap, total_trials,dtype=int, num=21), ax=axs[2,2], gap=gap)

    # change in field area
    plot_field_area(logparams, np.linspace(0, total_trials, num=21, dtype=int), ax=axs[3,1])

    # change in field location
    plot_field_center(logparams, np.linspace(0, total_trials, num=21, dtype=int), ax=axs[3,2])

    ## drift
    trials, pv_corr,rep_corr, startxcor, endxcor = get_pvcorr(logparams, stable_perf, total_trials, num=101)

    plot_rep_sim(startxcor, stable_perf, ax=axs[5,0])

    plot_rep_sim(endxcor, total_trials, ax=axs[5,1])
    
    drift = (np.std(pv_corr))/(np.std(np.array(latencys)[np.linspace(stable_perf, total_trials-1, num=1001, dtype=int)]))

    plot_pv_rep_corr(trials, pv_corr, rep_corr,title=f"D={drift:.3f}",ax=axs[5,2])

    param_delta = get_param_changes(logparams, total_trials)
    plot_param_variance(param_delta, total_trials,axs=axs[4], num=5)

    plot_l1norm(param_delta[2], ax=axs[6,2], stable_perf=0)

    # plot_reward_coding(logparams,[0.75,0.0],total_trials//2-1, ax=axs[6,0])

    # plot_active_frac(logparams, total_trials, num=total_trials//1000, threshold=0.5**2,ax=axs[6,1])

    plot_amplitude_drift(logparams, total_trials, stable_perf, ax=axs[6,0])

    f.text(0.001,0.001, exptname, fontsize=5)
    f.tight_layout()
    return f, score, drift

def get_statespace(num=51):
    x = np.linspace(-1,1,num)
    xx,yy = np.meshgrid(x,x)
    xs = np.concatenate([xx.reshape(-1)[:,None],yy.reshape(-1)[:,None]],axis=1)
    return xs


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


def store_csv(csv_file, args, score, drift):
    # Extract all arguments from args namespace
    arg_dict = vars(args)
    
    # Add score and drift to the dictionary
    arg_dict['score'] = score
    arg_dict['drift'] = drift

    # Create csv_columns from the keys of arg_dict
    csv_columns = list(arg_dict.keys())

    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, 'a' if file_exists else 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header
        writer.writerow(arg_dict)


def evaluate_loss(latencys, threshold=35, stability_window=10000, w1=1,w2=1, w3=1):
    loss_vector = np.array(moving_average(latencys,20))
    # Calculate convergence speed
    try:
        convergence_epoch = next(i for i, v in enumerate(loss_vector) if v < threshold)
    except StopIteration:
        convergence_epoch = len(loss_vector)
    
    # Calculate stability
    stability = np.std(loss_vector[-stability_window:]) if len(loss_vector) >= stability_window else np.std(loss_vector)
    
    # Final loss value
    final_loss = loss_vector[-1]

    score = convergence_epoch*final_loss*stability
    
    return score

# Define the functions to fit
def linear(x, a, b):
    return a * x + b

def exponential(x, a, b, c):
    return a * np.exp(-b * x) + c

def sigmoid(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c)))

def power_law(x, a, b, c):
    return a * np.power(x, -b) + c

# Function to fit the model
def fit_model(x, y, func_type='linear', initial_guess=None):
    if func_type == 'linear':
        func = linear
        if initial_guess is None: initial_guess = [1, 0]
    elif func_type == 'exp':
        func = exponential
        if initial_guess is None: initial_guess = [1, 1, 0]
    elif func_type == 'sigmoid':
        func = sigmoid
        if initial_guess is None: initial_guess = [1, 1, 1]
    elif func_type == 'power':
        func = power_law
        if initial_guess is None: initial_guess = [1, 1, 0]
    else:
        raise ValueError("Unsupported function type. Choose from 'linear', 'exp', 'sigmoid', or 'power'.")

    popt, _ = curve_fit(func, x, y, p0=initial_guess, maxfev=10000)
    return popt


def plot_model_fit(x, y, func_type):
    plt.scatter(x, y)
    popt, func = fit_model(x, y, func_type)
    plt.plot(x, func(x, *popt), label=f'Fitted: {func_type}\nParams: {np.round(popt, 3)}', color='red')
    plt.legend(frameon=False, fontsize=6)
    plt.show()


def plot_fxdx_trials(allcoords, logparams, trials,gap, ax=None):
    if ax is None:
        f,ax = plt.subplots()
    
    Rs = []
    for trial in trials:
        visits, frequency, density, R, pval = get_2D_freq_density_corr(allcoords, logparams, trial, gap=gap)
        Rs.append(R)
    ax.plot(trials, Rs, marker='o')
    # slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(trials).reshape(-1), np.array(Rs).reshape(-1))
    # regression_line = slope * np.array(trials).reshape(-1) + intercept
    # ax.plot(np.array(trials).reshape(-1), regression_line, color='red', label=f'R:{np.round(r_value, 3)}, P:{np.round(p_value, 3)}')
    # ax.legend(frameon=False, fontsize=8)
    ax.set_title('Correlation with learning')
    ax.set_xlabel('Trial')
    ax.set_ylabel('R')


def plot_fx_dx(allcoords, logparams, trial, title,gap,ax=None):
    if ax is None:
        f,ax = plt.subplots()
    
    visits, frequency, density, R, pval = get_2D_freq_density_corr(allcoords, logparams, trial, gap=gap)
    ax.scatter(frequency, density)
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(frequency).reshape(-1), np.array(density).reshape(-1))
    regression_line = slope * np.array(frequency).reshape(-1) + intercept
    ax.plot(np.array(frequency).reshape(-1), regression_line, color='red', label=f'R:{np.round(r_value, 3)}, P:{np.round(p_value, 3)}')
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel('$f(x)$')
    ax.set_ylabel('$d(x)$')


def flatten(xss):
    return np.array([x for xs in xss for x in xs],dtype=np.float32)


def plot_policy(logparams,ax=None):
    if ax is None:
        f,ax = plt.subplots()
    im = ax.imshow(logparams[-1][3],aspect='auto')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('Action')
    ax.set_ylabel('PF')
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels([-1,1])

def plot_active_frac(logparams, train_episodes,num=100, threshold=0.25,ax=None):
    if ax is None:
        f,ax = plt.subplots()
    trials = np.linspace(0,train_episodes-1,num, dtype=int)
    active_frac = []
    for trial in trials:
        amp = logparams[trial][2]**2
        active_frac.append(amp)
    active_frac = np.array(active_frac)

    ax.plot(trials, np.sum(active_frac>=threshold,axis=1)/active_frac.shape[1])
    ax.set_xlabel('Trial')
    ax.set_ylabel(f'Active Fraction > {threshold}')


def get_param_changes(logparams, total_trials, stable_perf=0):

    lambdas = []
    sigmas = []
    alphas = []
    values = []
    policies = []
    episodes = np.arange(stable_perf, total_trials)
    for e in episodes:
        lambdas.append(logparams[e][0])
        sigmas.append(logparams[e][1])
        alphas.append(logparams[e][2])
        values.append(logparams[e][4])
        policies.append(logparams[e][3])
    lambdas = np.array(lambdas)
    sigmas = np.array(sigmas)
    alphas = np.array(alphas)
    policies = np.array(policies)
    values = np.array(values)
    return [lambdas, sigmas, alphas, policies, values]

def get_param_variance(param_change):
    [lambdas, sigmas, alphas, policies, values] = param_change
    variance = []
    for i, param in enumerate([lambdas, sigmas, alphas]):
        if i == 0:
            delta_lambdas = np.linalg.norm(param,ord=2,axis=2)
            variances = delta_lambdas - delta_lambdas[0]
        elif i == 1:
            delta_sigmas = np.sum(np.diagonal(param,axis1=-2, axis2=-1),axis=-1)
            variances = delta_sigmas - delta_sigmas[0]
        elif i == 2:
            variances = param - param[0]
        variance.append(variances)
    return variance

def plot_param_variance(param_change, total_trials,num=3,axs=None):
    if axs is None:
        f,axs = plt.subplots(nrows=1, ncols=3)
    [lambdas, sigmas, alphas, policies, values] = param_change
    episodes = np.arange(0, total_trials)

    labels = [r'$\lambda$', r'$\sigma$',r'$\alpha$']
    for i, param in enumerate([lambdas, sigmas, alphas]):
        if i == 0:
            delta_lambdas = np.linalg.norm(param,ord=2,axis=2)
            variances = delta_lambdas - delta_lambdas[0]
        elif i == 1:
            delta_sigmas = np.sum(np.diagonal(param,axis1=-2, axis2=-1),axis=-1)
            variances = delta_sigmas - delta_sigmas[0]
        elif i == 2:
            variances = param - param[0]
        print(variances.shape)
        top_indices = np.argsort(np.std(variances,axis=0))[-num:][::-1]
        for n in top_indices:
            axs[i].plot(episodes, variances[:,n])

        axs[i].set_xlabel('Trial')
        axs[i].set_ylabel(labels[i])

def plot_pv_rep_corr(trials, pv_corr, rep_corr,title,ax=None):
    if ax is None:
        f,ax = plt.subplots()
    ax.plot(trials, pv_corr,label='$\phi(t)$')
    ax.plot(trials, rep_corr,label=r'$\phi(t)^\top\phi(t)$')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Correlation')
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=6) 

def plot_latency(latencys,allrewards, ax=None, window=20):
    if ax is None:
        f,ax = plt.subplots()

    ax.plot(moving_average(allrewards,window), color='tab:orange')
    ax.set_xlabel('$T$')
    ax.set_ylabel('$G$', color='tab:orange')

    #plt.xscale('log')
    score = evaluate_loss(latencys)

    ax2 = ax.twinx()
    ax2.plot(moving_average(latencys, window), color='tab:blue')
    ax2.set_ylabel('Latency (Steps)', color='tab:blue')
    return score

def plot_l1norm(alpha_delta,stable_perf=0, ax=None):
    if ax is None:
        f,ax = plt.subplots()
    ax.set_ylabel('$|\\alpha|_1$')
    l1norm = np.linalg.norm(alpha_delta,ord=1, axis=1)
    ax.plot(np.arange(len(alpha_delta))[stable_perf:], l1norm[stable_perf:], color='k',linewidth=3)

def plot_amplitude_drift(logparams, total_trials, stable_perf, ax=None):
    if ax is None:
        f,ax = plt.subplots()
    param_delta = get_param_changes(logparams, total_trials, stable_perf)
    mean_amplitude = np.mean(param_delta[2]**2,axis=0)
    param_var = get_param_variance(param_delta)
    deltas = np.sum(np.std(np.array(param_var),axis=1),axis=0)
    ax.scatter(mean_amplitude, deltas)
    if np.std(mean_amplitude) != 0:
        slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(mean_amplitude).reshape(-1), np.array(deltas).reshape(-1))
        regression_line = slope * np.array(mean_amplitude).reshape(-1) + intercept
        ax.plot(np.array(mean_amplitude).reshape(-1), regression_line, color='red', label=f'R:{np.round(r_value, 3)}, P:{np.round(p_value, 3)}')
    ax.legend(frameon=False)
    ax.set_xlabel('Mean Amplitude')
    ax.set_ylabel('$\sum var(\\theta)$')

def plot_rep_sim(xcor,trial, ax=None):
    if ax is None:
        f,ax = plt.subplots()
    im = ax.imshow(xcor,origin='lower')
    plt.colorbar(im,ax=ax,fraction=0.046, pad=0.04)
    ax.set_xlabel('$x_1 x_2$')
    ax.set_ylabel('$x_1 x_2$')
    idx = np.array([0,500,1000])
    ax.set_xticks(np.arange(1001)[idx], np.linspace(-1,1,1001)[idx])
    ax.set_yticks(np.arange(1001)[idx], np.linspace(-1,1,1001)[idx])
    ax.set_title(f'T={trial}')

def plot_value(logparams, trial, goalcoord=[0.75,-0.75], startcoord=[-0.75,-0.75], goalsize=0.1, envsize=1, ax=None):
    if ax is None:
        f,ax = plt.subplots()
    num = 41
    xs = get_statespace(num)
    pcacts = predict_batch_placecell(logparams[trial], xs)
    value = pcacts @ logparams[trial][4] 
    im = ax.imshow(value.reshape(num,num), origin='lower')
    plt.colorbar(im,ax=ax,fraction=0.046, pad=0.04)

    start_circle = plt.Circle(startcoord, 0.05, color='green', fill=True)
    ax.add_artist(start_circle)
    circle = plt.Circle(goalcoord, goalsize, color='r', fill=True)
    ax.add_artist(circle)
    ax.add_patch(Rectangle((-0.2,0.5), 0.4, -1.5, facecolor='grey'))

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xticks([],[])
    ax.set_yticks([],[])
    ax.set_title('Value')



def plot_field_area(logparams, trials,ax=None):
    if ax is None:
        f,ax = plt.subplots()
    num = 41
    xs = get_statespace(num)
    areas = []
    for trial in trials:
        area = np.trapz(predict_batch_placecell(logparams[trial], xs),axis=0)
        areas.append(area)
    areas = np.array(areas)
    norm_area = areas/areas[0]

    ax.errorbar(trials, np.mean(norm_area,axis=1), np.std(norm_area,axis=1)/np.sqrt(len(logparams[0][0])), marker='o')
    ax.set_ylabel('Norm Field Area')
    ax.set_xlabel('Trial')
    return norm_area

def plot_field_center(logparams, trials,ax=None):
    if ax is None:
        f,ax = plt.subplots()
    lambdas = []
    for trial in trials:
        lambdas.append(logparams[trial][0])
    lambdas = np.array(lambdas)

    delta_lambdas = np.linalg.norm(lambdas,ord=2,axis=2)
    norm_lambdas = delta_lambdas - delta_lambdas[0]

    ax.errorbar(trials, np.mean(norm_lambdas,axis=1), np.std(norm_lambdas,axis=1)/np.sqrt(len(logparams[0][0])), marker='o')
    ax.set_ylabel('Centered Field Center')
    ax.set_xlabel('Trial')
    return norm_lambdas



def plot_velocity(logparams, trial, goalcoord=[0.75,-0.75], startcoord=[-0.7,-0.75], goalsize=0.1, envsize=1, ax=None):
    if ax is None:
        f,ax = plt.subplots()
    num=41
    xs = get_statespace(num)

    pcacts = predict_batch_placecell(logparams[trial], xs)
    actout = pcacts @ logparams[trial][3] 
    aprob = nn.softmax(actout)
    onehot2dirmat = np.array([
    [0,1],  # up
    [1,0],  # right
    [0,-1],  # down
    [-1,0]  # left
    ])
    vel = np.matmul(aprob, onehot2dirmat * 0.1)
    xx, yy = np.meshgrid(np.arange(num), np.arange(num))
    ax.quiver(xx.reshape(-1),yy.reshape(-1), vel[:,0], vel[:,1], color='k', scale_units='xy', zorder=2)

    start_circle = plt.Circle(startcoord, 0.05, color='green', fill=True)
    ax.add_artist(start_circle)
    circle = plt.Circle(goalcoord, goalsize, color='r', fill=True)
    ax.add_artist(circle)
    ax.add_patch(Rectangle((-0.2,0.5), 0.4, -1.5, facecolor='grey'))

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xticks([],[])
    ax.set_yticks([],[])
    ax.set_title('Policy')

def plot_all_pc(logparams, trial,goalcoord=[0.75,-0.75], startcoord=[-0.75,-0.75], goalsize=0.1, envsize=1, obs=True):
    start_radius = 0.05
    num = 41
    xs = get_statespace(num)
    pcacts = predict_batch_placecell(logparams[trial], xs)

    num_curves = pcacts.shape[1]
    yidx = xidx = int(num_curves**0.5)
    f,axs = plt.subplots(yidx, xidx, figsize=(12,12))
    pcidx = np.arange(num_curves)
    axs = axs.flatten()
    max_value = np.max(pcacts)
    for i in pcidx:
        ax = axs[i]
        ax.imshow(pcacts[:, i].reshape(num, num), origin='lower', extent=[-envsize, envsize, -envsize, envsize], 
                vmin=0, vmax=max_value)

        start_circle = plt.Circle(startcoord, start_radius, color='green', fill=True)
        ax.add_artist(start_circle)

        reward_circle = plt.Circle(goalcoord, goalsize*2, color='red', fill=True)
        ax.add_artist(reward_circle)

        ax.set_xticks([],[])
        ax.set_yticks([],[])
        ax.text(1.0, 0.0, f'{i}-{max_value:.2f}', transform=ax.transAxes,
                fontsize=6, color='yellow', ha='right')
        if obs:
            ax.add_patch(Rectangle((-0.2,0.5), 0.4, -1.5, facecolor='grey'))


    f.tight_layout()

def find_closest_index(lambdas, target):
    vec = np.linalg.norm(lambdas - target, axis=1)
    dist= np.argmin(vec)
    print(vec.shape)
    return dist

def plot_pc(logparams, trial,pi=None,title='',  ax=None, goalcoord=[0.75,-0.75], startcoord=[-0.75,-0.75], goalsize=0.1, envsize=1, obs=True):
    if ax is None:
        f,ax = plt.subplots()
    
    num = 41
    xs = get_statespace(num)
    pcacts = predict_batch_placecell(logparams[trial], xs)
    
    if pi is None:
        pi = np.argmax(np.mean(pcacts,axis=0))
    
    max_value = np.max(pcacts[:,pi])
    ax.imshow(pcacts[:, pi].reshape(num, num), origin='lower', extent=[-envsize, envsize, -envsize, envsize], 
              vmin=0, vmax=max_value)

    start_radius = 0.05
    start_circle = plt.Circle(startcoord, start_radius, color='green', fill=True)
    ax.add_artist(start_circle)

    num_circles = 1
    for i in range(num_circles):
        radius = goalsize * (num_circles - i) / num_circles
        color = (1, i/num_circles, i/num_circles)  # RGB tuple for gradient from white to red
        circle = plt.Circle(goalcoord, radius*2, color=color, fill=True)
        ax.add_artist(circle)
    if obs:
        ax.add_patch(Rectangle((-0.2,0.5), 0.4, -1.5, facecolor='grey'))

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_xticks([],[])
    ax.set_yticks([],[])
    ax.text(-1.0, 1.0, f'{pi}-{max_value:.2f}', transform=ax.transAxes,
            fontsize=8, color='yellow', ha='right')
    ax.set_title(title)
    

def plot_all_pc(logparams, trial,goalcoord=[0.75,-0.75], startcoord=[-0.75,-0.75], goalsize=0.1, envsize=1, obs=True):
    start_radius = 0.05
    num = 41
    xs = get_statespace(num)
    pcacts = predict_batch_placecell(logparams[trial], xs)

    num_curves = pcacts.shape[1]
    yidx = xidx = int(num_curves**0.5)
    f,axs = plt.subplots(yidx, xidx, figsize=(12,12))
    pcidx = np.arange(num_curves)
    axs = axs.flatten()

    for i in pcidx:
        ax = axs[i]
        max_value = np.max(pcacts[:,i])
        ax.imshow(pcacts[:, i].reshape(num, num), origin='lower', extent=[-envsize, envsize, -envsize, envsize], 
                vmin=0, vmax=max_value)

        start_circle = plt.Circle(startcoord, start_radius, color='green', fill=True)
        ax.add_artist(start_circle)

        reward_circle = plt.Circle(goalcoord, goalsize*2, color='red', fill=True)
        ax.add_artist(reward_circle)

        ax.set_xticks([],[])
        ax.set_yticks([],[])
        ax.text(1.0, 0.0, f'{i}-{max_value:.2f}', transform=ax.transAxes,
                fontsize=6, color='yellow', ha='right')
        if obs:
            ax.add_patch(Rectangle((-0.2,0.5), 0.4, -1.5, facecolor='grey'))

    f.tight_layout()


def plot_reward_coding(logparams,goalcoords,stable_perf, ax=None):
    if ax is None:
        f,ax = plt.subplots()
    x = logparams[stable_perf][0]
    y = logparams[stable_perf*2][0]
    plt.plot(np.linspace(np.min(x),np.max(x),1000),np.linspace(np.min(y),np.max(y),1000), color='k')
    indexes = np.where((x >= goalcoords[0]-0.1) & (x <= goalcoords[0]+0.1))[0]
    values_in_x = x[indexes]
    values_in_y = y[indexes]
    ax.scatter(x, y)
    ax.scatter(values_in_x, values_in_y, color='g')
    ax.axvline(0.5, color='r')
    ax.axhline(0.0, color='r')
    ax.set_xlabel('Before')
    ax.set_ylabel('After')

def plot_trajectory(allcoords,trial, obs=True, goalcoord=[0.75,-0.75],goalsize=0.1, ax=None, obscoord=[-0.2,0.2,-1.0,0.5]):
    if ax is None:
        f,ax = plt.subplots()

    if obs:
        L = obscoord[0]
        T = obscoord[3]
        dfL = obscoord[1]-obscoord[0]
        dfT = obscoord[2]-obscoord[3]
        ax.add_patch(Rectangle((L,T), dfL, -dfT, facecolor='grey'))  # top left

    circle = plt.Circle(xy=goalcoord, radius=goalsize*2, color='r')
    trail = np.array(allcoords[trial-1])
    ax.add_patch(circle)
    ax.scatter(trail[1,0],trail[1,1], color='g', zorder=2)    
    ax.plot(trail[1:,0],trail[1:,1], marker='o',color='b', zorder=1)
    ax.axis([-1, 1,-1, 1])
    ax.set_aspect('equal')
    ax.set_axis_off()

def plot_density(logparams, trial, ax=None, goalcoord=[0.75,-0.75], startcoord=[-0.75,-0.75], goalsize=0.1, envsize=1, obs=True):
    if ax is None:
        f,ax = plt.subplots()

    num = 41
    xs = get_statespace(num)
    pcacts = predict_batch_placecell(logparams[trial], xs)
    dx = np.mean(pcacts,axis=1)

    im = ax.imshow(dx.reshape(num,num), origin='lower', extent=[-envsize, envsize, -envsize, envsize])
    plt.colorbar(im,ax=ax,fraction=0.046, pad=0.04)

    start_circle = plt.Circle(startcoord, 0.05, color='green', fill=True)
    ax.add_artist(start_circle)
    circle = plt.Circle(goalcoord, goalsize, color='r', fill=True)
    ax.add_artist(circle)
    if obs:
        ax.add_patch(Rectangle((-0.2,0.5), 0.4, -1.5, facecolor='grey'))

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xticks([],[])
    ax.set_yticks([],[])
    ax.set_title('Mean Firing Rate')
    return dx.reshape(num,num)

def plot_frequency(allcoords, trial, gap=25, bins=31, goalcoord=[0.75,-0.75], startcoord=[-0.75,-0.75], goalsize=0.1, ax=None):
    if ax is None:
        f,ax = plt.subplots()

    coord = []
    for t in range(gap):
        for c in allcoords[trial-t-1]:
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

    im = ax.imshow(freq.reshape(bins,bins).T,origin='lower')
    plt.colorbar(im,ax=ax,fraction=0.046, pad=0.04)

    start_circle = plt.Circle(startcoord, 0.05, color='green', fill=True)
    ax.add_artist(start_circle)
    circle = plt.Circle(goalcoord, goalsize, color='r', fill=True)
    ax.add_artist(circle)
    ax.add_patch(Rectangle((-0.2,0.5), 0.4, -1.5, facecolor='grey'))

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xticks([],[])
    ax.set_yticks([],[])
    ax.set_title('Frequency')

    return freq.reshape(bins,bins)


def reward_func(xs,goal, rsz, threshold=1e-2):
    rx =  np.exp(-0.5 * np.sum(((xs - goal) / rsz) ** 2, axis=1))
    return rx * (rx>threshold)

def gaussian(xs, center, sigma):
    values = np.exp(-0.5 * np.sum(((xs - center) / sigma) ** 2, axis=1))
    values[values < 1e-1] = np.nan  # Convert values less than 0.01 to NaN
    return values

def moving_average(signal, window_size):
    # Pad the signal to handle edges properly
    padded_signal = np.pad(signal, (window_size//2, window_size//2), mode='edge')
    
    # Apply the moving average filter
    weights = np.ones(window_size) / window_size
    smoothed_signal = np.convolve(padded_signal, weights, mode='valid')
    
    return smoothed_signal[:-1]

def saveload(filename, variable, opt):
    import pickle
    if opt == 'save':
        with open(f"{filename}.pickle", "wb") as file:
            pickle.dump(variable, file)
        print('file saved')
    else:
        with open(f"{filename}.pickle", "rb") as file:
            return pickle.load(file)
    

def get_2D_freq_density_corr(allcoords, logparams, end, gap=25, bins=23):
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
    pcacts = predict_batch_placecell(param, visits)
    dxs = np.sum(pcacts,axis=1).reshape(-1)

    R,pval = stats.pearsonr(freq, dxs)
    return visits, freq, dxs, R, pval

def get_pvcorr(params, start, end, num):
    num = 41
    xs = get_statespace(num)
    startpcs = predict_batch_placecell(params[start], xs)
    startvec = startpcs.flatten()
    trials = np.linspace(start, end-1, num, dtype=int)
    startxcor = startpcs@startpcs.T

    pv_corr = []
    rep_corr = []
    for i in trials:
        endpcs = predict_batch_placecell(params[i], xs)
        endvec = endpcs.flatten()
        R = np.corrcoef(startvec, endvec)[0, 1]
        pv_corr.append(R)

        endxcor = endpcs@endpcs.T
        R_rep = np.corrcoef(startxcor.flatten(), endxcor.flatten())[0, 1]
        rep_corr.append(R_rep)
    return trials, pv_corr,rep_corr, startxcor, endxcor

def get_learning_rate(initial_lr, final_lr, total_steps):
    steps = np.arange(total_steps + 1)
    decay_rate = (final_lr / initial_lr) ** (1 / total_steps)
    learning_rates = initial_lr * (decay_rate ** steps)
    return learning_rates
