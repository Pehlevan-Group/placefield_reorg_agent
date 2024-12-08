import matplotlib.pyplot as plt
import numpy as np


def plot_all_sr_pc(Us, ca3, trial,goalcoord=[0.75,-0.75], startcoord=[-0.75,-0.75], goalsize=0.05, envsize=1, obs=True):
    start_radius = 0.05
    num = 41
    ca1 = get_ca1(ca3, Us[trial])

    num_curves = ca1.shape[1]
    yidx = xidx = int(num_curves**0.5)
    f,axs = plt.subplots(yidx, xidx, figsize=(12,12))
    pcidx = np.arange(num_curves)
    axs = axs.flatten()

    for i in pcidx:
        ax = axs[i]
        max_value = np.max(ca1[:,i])
        ax.imshow(ca1[:, i].reshape(num, num), origin='lower', extent=[-envsize, envsize, -envsize, envsize], 
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

    
def relu(x):
    return np.maximum(x,0)

def get_ca1(ca3, U):
    ca1_sr = []
    for i in range(ca3.shape[0]):
        ca1_sr.append(relu(U) @ ca3[i])
    ca1_sr = np.array(ca1_sr)
    return ca1_sr

def plot_sr_pc(Us, ca3, trial,title='', ax=None, goalcoord=[0.5], startcoord=[-0.75], goalsize=0.05, envsize=1,):
    if ax is None:
        f,ax = plt.subplots()

    # Get a colormap that transitions from purple to yellow
    cmap = cm.viridis
    num_curves = ca3.shape[1]
    xs = np.linspace(-1,1,1001)
    for i in range(num_curves):
        color = cmap(i / num_curves)
        ca1 = get_ca1(ca3, Us[trial])
        ax.plot(xs, ca1[:, i], color=color,zorder=1)

    ax.set_xlabel('x')
    ax.set_ylabel('$\psi(x)$')
    ax.set_title(title)

    # ax.fill_betweenx(np.linspace(0,maxval), goalcoord[0]-goalsize, goalcoord[0]+goalsize, color='r', alpha=0.25, label='Target')
    ax.fill_between(xs, reward_func(xs, goalcoord, goalsize), color='red', alpha=0.25, label='Target',zorder=2)
    ax.axvline(startcoord[0],ymin=0, ymax=1, color='g',linestyle='--',label='Start', linewidth=2,zorder=2)
    ax.hlines(xmin=-envsize,xmax=envsize, y=0, colors='k',zorder=2)
    # plt.legend(frameon=False, fontsize=6)

def plot_sr_field_area(Us, ca3, trials,ax=None):
    if ax is None:
        f,ax = plt.subplots()
    
    areas = []
    for trial in trials:
        ca1 = get_ca1(ca3, Us[trial])
        area = np.trapz(ca1,axis=0)
        areas.append(area)
    areas = np.array(areas)

    norm_area = areas/areas[0]

    mean_deltas = np.mean(norm_area,axis=1)
    sem_deltas = np.std(norm_area,axis=1)/np.sqrt(len(logparams[0][0]))
    ax.plot(trials, mean_deltas, color='tab:orange')
    ax.fill_between(trials, mean_deltas - sem_deltas, mean_deltas + sem_deltas, alpha=0.2, color='tab:orange')
    # ax.plot(trials,mean_deltas, color='tab:orange')
    # ax.fill_between(trials, mean_deltas - sem_deltas, mean_deltas + sem_deltas, color='tab:orange', alpha=0.2)
    ax.set_ylabel('SR Field Area')
    return norm_area


def plot_sr_center(Us,ca3, trials,ax=None):
    if ax is None:
        f,ax = plt.subplots()
    ca1_init = get_ca1(ca3, Us[0])
    xs = np.linspace(-1,1,1001)
    deltas = []
    for trial in trials:
        ca1 = get_ca1(ca3, Us[trial])
        d = []
        for n in range(ca3.shape[1]):
            # ca3_center = xs[np.argmax(ca3[:,n])]
            orig_ca1_center = xs[np.argmax(ca1_init[:,n])]
            ca1_center = xs[np.argmax(ca1[:,n])]
            delta = ca1_center - orig_ca1_center# - ca3_center
            d.append(delta)
        deltas.append(np.array(d))
    deltas = np.array(deltas)

    mean_deltas = np.mean(deltas,axis=1)
    sem_deltas =  np.std(deltas,axis=1)/np.sqrt(ca3.shape[1])
    # ax.errorbar(trials, mean_deltas, yerr=sem_deltas, fmt='s', color='tab:orange')
    ax.plot(trials, mean_deltas, color='tab:orange')
    ax.fill_between(trials, mean_deltas - sem_deltas, mean_deltas + sem_deltas, color='tab:orange', alpha=0.2)
    ax.set_ylabel('SR Centered Fields')
    return deltas

def plot_sr_density(Us,ca3, trials,ax=None, goalcoord=0.5, goalsize=0.05, startcoord=[-0.75], envsize=1):
    if ax is None:
        f,ax=plt.subplots()
    xs = np.linspace(-1,1,1001)
    for trial in trials:
        ca1 = get_ca1(ca3, Us[trial])
        dx = np.sum(ca1,axis=1)/ca1.shape[1]
        ax.plot(xs, dx, label=f'T={trial}')

    ax.set_xlabel('$x$')
    ax.set_ylabel('$d(x)$')
    ax.legend(frameon=False, fontsize=6)
    ax.set_title('Density learnd using SR')

    ax2 = ax.twinx()
    ax2.fill_between(xs, reward_func(xs, goalcoord, goalsize), color='red', alpha=0.25, label='Target')
    ax.axvline(startcoord[0],ymin=0, ymax=1, color='g',linestyle='--',label='Start', linewidth=2)
    ax.hlines(xmin=-envsize,xmax=envsize, y=0, colors='k')

    return dx


def get_sr_1D_kde_density_corr(allcoords, Us, param, trial, gap=25):
    xs = np.linspace(-1,1,1001)
    ca3 = predict_batch_placecell(param, xs)

    dx = []
    fx = []

    for g in range(gap):
        fx.append(allcoords[trial-g-1])

        ca1_sr = []
        for i in range(ca3.shape[0]):
            ca1_sr.append(relu(Us[trial-g-1]) @ ca3[i])
        ca1_sr = np.array(ca1_sr)

        density = np.sum(ca1_sr,axis=1)/ca1_sr.shape[1]
        dx.append(density)
    
    fx = np.array(flatten(fx))
    kde = gaussian_kde(fx.reshape(-1))
    fx_smooth = kde(xs)

    dx = np.array(dx)
    dx = np.mean(dx,axis=0)
    R,pval = stats.pearsonr(fx_smooth, dx)

    return xs, fx_smooth, dx, R, pval



def plot_sr_fxdx_trials_kde(allcoords, Us, logparams, trials,gap, ax=None):
    if ax is None:
        f,ax = plt.subplots()
    
    Rs = []
    fxs = []
    dxs = []
    for trial in trials:
        visits, frequency, density, R, pval = get_sr_1D_kde_density_corr(allcoords, Us,logparams[0], trial, gap=gap)
        Rs.append(R)
        fxs.append(frequency)
        dxs.append(density)
    ax.plot(trials, Rs, marker='s',color='tab:green')
    return np.array(fxs), np.array(dxs), Rs