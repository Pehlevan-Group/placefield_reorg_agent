import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.cm as cm
from io import BytesIO
from model import *
from scipy.stats import gaussian_kde
    

def reward_func(x, xr, rsz, amp=1):
    rx =  amp  * np.exp(-0.5*((x - xr)/rsz)**2)  # (1/np.sqrt(2*np.pi*rsz**2))
    return rx 


def get_1D_freq_density_corr(allcoords, logparams, trial, gap=25):
    fx = []
    dx = []
    xs = np.linspace(-1,1,1001)

    for g in range(gap):
        fx.append(allcoords[trial-g-1])

        param = logparams[trial-g-1]
        pcacts = predict_batch_placecell(param, xs)
        com = xs[np.argmax(pcacts,axis=0)]
        kde = gaussian_kde(com,bw_method=1/11)
        density = kde(xs)
        dx.append(density)
    
    fx = np.array(flatten(fx))
    kde = gaussian_kde(fx.reshape(-1))
    fx_smooth = kde(xs)

    dx = np.array(dx)
    dx = np.mean(dx,axis=0)
    R,pval = stats.pearsonr(fx_smooth, dx)
    return xs, fx_smooth, dx, R, pval

def plot_fxdx_trials(allcoords, logparams, trials,gap,ax=None):
    if ax is None:
        f,ax = plt.subplots()
    
    Rs = []
    pvals = []
    fxs = []
    dxs = []
    for trial in trials:
        visits, frequency, density, R, pval = get_1D_freq_density_corr(allcoords, logparams, trial, gap=gap)
        Rs.append(R)
        pvals.append(pval)
        fxs.append(frequency)
        dxs.append(density)
    pvals =np.array(pvals)
    Rs = np.array(Rs)
    ax.plot(trials, Rs, color='tab:blue')
    significant_mask = pvals < 0.05
    ax.plot(np.array(trials)[significant_mask], Rs[significant_mask], 'o', color='tab:blue')
    ax.set_title('$f(x):d(x)$ Correlation with learning')
    ax.set_xlabel('$T$')
    ax.set_ylabel('R')
    return fxs, dxs, Rs


def plot_fx_dx(allcoords, logparams, trial, title,gap,ax=None):
    # correlation between frequencuy and density
    if ax is None:
        f,ax = plt.subplots()
    
    visits, frequency, density, R, pval = get_1D_freq_density_corr(allcoords, logparams, trial, gap=gap)
    ax.scatter(frequency, density)
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(frequency).reshape(-1), np.array(density).reshape(-1))
    regression_line = slope * np.array(frequency).reshape(-1) + intercept
    ax.plot(np.array(frequency).reshape(-1), regression_line, color='red', label=f'R:{np.round(r_value, 3)}, P:{np.round(p_value, 3)}')
    ax.legend(frameon=False, fontsize=6)
    ax.set_title(title)
    ax.set_xlabel('Frequency $f(x)$')
    ax.set_ylabel('Density $d(x)$')


def flatten(xss):
    return np.array([x for xs in xss for x in xs],dtype=np.float32)


def plot_analysis(logparams,rewards, allcoords, stable_perf, exptname=None , rsz=0.05):
    f, axs = plt.subplots(7,3,figsize=(12,21))
    total_trials = len(logparams)-1
    gap = 25
    
    #latency 
    plot_perf(rewards, ax=axs[0,0])

    plot_pc(logparams, 0,ax=axs[0,1], title='Before Learning', goalsize=rsz)

    plot_pc(logparams, total_trials,ax=axs[0,2], title='After Learning', goalsize=rsz)


    plot_value(logparams, [gap,total_trials//4, total_trials], ax=axs[3,0], goalsize=rsz)


    plot_velocity(logparams,  [gap,total_trials//4,total_trials],ax=axs[1,0], goalsize=rsz)



    ## high d at reward
    plot_density(logparams,  [gap,total_trials//4, total_trials], ax=axs[1,1], goalsize=rsz)

    plot_frequency(allcoords,  [gap,total_trials//4,total_trials], ax=axs[1,2], gap=gap, goalsize=rsz)


    plot_fx_dx(allcoords, logparams, gap,'Before Learning', ax=axs[2,0], gap=gap)

    plot_fx_dx(allcoords, logparams, total_trials,'After Learning', ax=axs[2,1], gap=gap)

    plot_fxdx_trials(allcoords, logparams, np.linspace(gap, total_trials,dtype=int, num=31), ax=axs[2,2], gap=gap)

    # change in field area
    plot_field_area(logparams, np.linspace(0, total_trials, num=51, dtype=int), ax=axs[3,1])

    # change in field location
    plot_field_center(logparams, np.linspace(0, total_trials, num=51, dtype=int), ax=axs[3,2])


    ## drift
    trials, pv_corr,rep_corr, startxcor, endxcor = get_pvcorr(logparams, stable_perf, total_trials, num=101)

    plot_rep_sim(startxcor, stable_perf, ax=axs[4,0])

    plot_rep_sim(endxcor, total_trials, ax=axs[4,1])

    plot_pv_rep_corr(trials, pv_corr, rep_corr,title=f"",ax=axs[4,2])

    param_delta = get_param_changes(logparams, total_trials)
    plot_param_variance(param_delta, total_trials, stable_perf,axs=axs[5])

    plot_policy(logparams,ax=axs[6,2])

    plot_com(logparams,[0.75,-0.2],rsz, total_trials//2-1, ax=axs[6,1])

    plot_amplitude_drift(logparams, total_trials, stable_perf, ax=axs[6,0])

    # plot_active_frac(logparams, np.linspace(0,total_trials,51, dtype=int), total_trials, ax=axs[6,2])x

    f.text(0.001,0.001, exptname, fontsize=5)
    f.tight_layout()
    return f
    

def plot_policy(logparams,ax=None):
    if ax is None:
        f,ax = plt.subplots()
    im = ax.imshow(logparams[-1][3],aspect='auto')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('Action')
    ax.set_ylabel('PF')
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels([-1,1])

def plot_active_frac(logparams,trials, train_episodes,threshold=0.25,ax=None):
    if ax is None:
        f,ax = plt.subplots()
    param_delta = get_param_changes(logparams, train_episodes)

    for threshold in [0.05,0.1,0.25]:
        active_frac = np.mean(param_delta[2]**2>threshold,axis=1)
        ax.plot(active_frac)
    ax.set_xlabel('Trial')
    ax.set_ylabel(f'Active Fraction > {threshold}')
    ax.set_ylim([0.0,1.0])

    ax2 = ax.twinx()
    xs = np.linspace(-1,1,1001)
    for threshold in [0.05,0.1,0.25]:
        af = []
        for trial in trials:
            pcs = predict_batch_placecell(logparams[trial], xs)
            af.append(np.mean(pcs,axis=0)>threshold)
        af = np.array(af)
        ax2.plot(trials, np.mean(af,axis=1), linestyle='--')
    ax2.set_ylabel('Field active fraction')

def normalize_values(x, minusmin=True):
    maxval = np.max(x)
    minval = np.min(x)
    if minusmin:
        return (x-minval)/(maxval-minval)
    else:
        return (x)/(maxval-minval)

def find_closest_indices(vector, target_values):
    vector = np.array(vector)  # Ensure the input is a numpy array
    target_values = np.array(target_values)
    indices = [np.argmin(np.abs(vector - target)) for target in target_values]
    return indices

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


def plot_param_variance(param_change, total_trials, stable_perf,num=10,axs=None):
    if axs is None:
        f,axs = plt.subplots(nrows=1, ncols=3)
    [lambdas, sigmas, alphas, policies, values] = param_change
    # Assuming `lambdas` is your T x N matrix
    variances = np.var(alphas[stable_perf:], axis=0)
    # Get indices of the top 10 variances
    top_indices = np.argsort(variances)[-num:][::-1]
    episodes = np.arange(0, total_trials)

    labels = [r'$\lambda$', r'$\sigma$',r'$\alpha$']
    for i, param in enumerate([lambdas, sigmas, alphas]):
        for n in top_indices:
            axs[i].plot(episodes[stable_perf:], param[stable_perf:,n])
        axs[i].set_xlabel('Trial')
        axs[i].set_ylabel(labels[i])

def plot_pv_rep_corr(trials, pv_corr, rep_corr,title,ax=None):
    if ax is None:
        f,ax = plt.subplots()
    ax.plot(trials, pv_corr,label='PV')  # population vector correlatrion
    ax.plot(trials, rep_corr,label='SM')  # similarlity matrix correlation
    ax.set_xlabel('$T$')
    ax.set_ylabel('$R$')
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=6) 

def plot_perf(rewards, ax=None, window=20):
    if ax is None:
        f,ax = plt.subplots()
    ax.plot(moving_average(rewards, window), color='tab:blue')
    ax.set_xlabel('Trial')
    ax.set_ylabel('$G$')


def plot_l1norm(alpha_delta,stable_perf=0, ax=None):
    if ax is None:
        f,ax = plt.subplots()
    ax.set_ylabel('$|\\alpha|_1$')
    l1norm = np.linalg.norm(alpha_delta,ord=1, axis=1)
    ax.plot(np.arange(len(alpha_delta))[stable_perf:], l1norm[stable_perf:], color='k',linewidth=3)

def plot_amplitude_activefrac(logparams, total_trials, stable_perf, ax=None):
    if ax is None:
        f,ax = plt.subplots()
    param_delta = get_param_changes(logparams, total_trials, stable_perf)
    param_delta = param_delta[:3]
    mean_amplitude = np.mean(param_delta[2]**2,axis=0)

    active_fraction = []
    for threshold in [0.05, 0.1,0.25]:
        active_fraction.append(np.mean(param_delta[2]**2 > threshold,axis=0))
    mean_active_fraction = np.mean(np.array(active_fraction),axis=0)

    ax.scatter(mean_amplitude, mean_active_fraction, color='tab:blue')
    if np.std(mean_amplitude) != 0:
        slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(mean_amplitude).reshape(-1), np.array(mean_active_fraction).reshape(-1))
        regression_line = slope * np.array(mean_amplitude).reshape(-1) + intercept
        ax.plot(np.array(mean_amplitude).reshape(-1), regression_line, color='red', label=f'R:{np.round(r_value, 3)}, P:{np.round(p_value, 3)}')

    ax.legend(frameon=False,loc=2)
    ax.set_xlabel('$<\\alpha>$')
    ax.set_ylabel(f'Fraction of active time')

    trials = np.linspace(stable_perf, total_trials,51,dtype=int)
    xs = np.linspace(-1,1,1001)
    ax2 = ax.twinx()

    active_fraction = []
    for threshold in [0.05,0.1,0.25]:
        af = []
        for trial in trials:
            pcs = predict_batch_placecell(logparams[trial], xs)
            af.append(np.mean(pcs,axis=0)>threshold)
        active_fraction.append(np.mean(np.array(af),axis=0))
    mean_active_fraction = np.mean(np.array(active_fraction),axis=0)
    ax2.scatter(mean_amplitude, mean_active_fraction,color='tab:orange')
    
    if np.std(mean_amplitude) != 0:
        slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(mean_amplitude).reshape(-1), np.array(mean_active_fraction).reshape(-1))
        regression_line = slope * np.array(mean_amplitude).reshape(-1) + intercept
        ax2.plot(np.array(mean_amplitude).reshape(-1), regression_line, color='green', label=f'R:{np.round(r_value, 3)}, P:{np.round(p_value, 3)}')
    ax2.legend(frameon=False,loc=1)
    ax2.set_ylabel(f'Fraction of field active time')


def plot_amplitude_drift(logparams, total_trials, stable_perf, ax=None):
    if ax is None:
        f,ax = plt.subplots()
    param_delta = get_param_changes(logparams, total_trials, stable_perf)
    param_delta = param_delta[:3]
    mean_amplitude = np.mean(param_delta[2]**2,axis=0)

    delta_lambda = np.std(param_delta[0],axis=0)
    delta_alpha = np.std(param_delta[2]**2,axis=0)
    delta_sigma = np.std(param_delta[1]**2,axis=0)
    deltas = normalize_values(delta_lambda) + normalize_values(delta_sigma) + normalize_values(delta_alpha) 

    ax.scatter(mean_amplitude, deltas)
    if np.std(mean_amplitude) != 0:
        slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(mean_amplitude).reshape(-1), np.array(deltas).reshape(-1))
        regression_line = slope * np.array(mean_amplitude).reshape(-1) + intercept
        ax.plot(np.array(mean_amplitude).reshape(-1), regression_line, color='red', label=f'R:{np.round(r_value, 3)}, P:{np.round(p_value, 3)}')
    ax.legend(frameon=False,loc=4)
    ax.set_xlabel('$<\\alpha>$')
    ax.set_ylabel(f'$\sum Var(\\theta)$')



def plot_rep_sim(xcor,trial, ax=None):
    if ax is None:
        f,ax = plt.subplots()
    im = ax.imshow(xcor)
    plt.colorbar(im)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$x$')
    idx = np.array([0,500,1000])
    ax.set_xticks(np.arange(1001)[idx], np.linspace(-1,1,1001)[idx])
    ax.set_yticks(np.arange(1001)[idx], np.linspace(-1,1,1001)[idx])
    ax.set_title(f'T={trial}')

def plot_value(logparams, trials, goalcoord=[0.5], startcoord=[-0.75], goalsize=0.05, envsize=1, ax=None):
    if ax is None:
        f,ax = plt.subplots()
    xs = np.linspace(-1,1,1001)
    maxval  = 0 
    for trial in trials:
        pcacts = predict_batch_placecell(logparams[trial], xs)
        value = pcacts @ logparams[trial][4] 
        ax.plot(xs, value, label=f'T={trial}')
        maxval = max(maxval, np.max(value) * 1.1)
    ax.set_xlabel('$x$')
    ax.set_ylabel('Value v(x)')
    ax.legend(frameon=False, fontsize=6)
    ax.fill_between(xs, reward_func(xs, goalcoord, goalsize), color='red', alpha=0.25, label='Target')
    ax.axvline(startcoord[0],ymin=0, ymax=1, color='g',linestyle='--',label='Start', linewidth=2)
    ax.hlines(xmin=-envsize,xmax=envsize, y=0, colors='k')

def plot_field_area(logparams, trials,ax=None):
    if ax is None:
        f,ax = plt.subplots()
    areas = []
    for trial in trials:
        area = np.trapz(predict_batch_placecell(logparams[trial], np.linspace(-1,1,1001)),axis=0)
        areas.append(area)
    areas = np.array(areas)
    norm_area = areas/areas[0]
    
    mean_deltas = np.mean(norm_area,axis=1)
    sem_deltas =  1.96*np.std(norm_area,axis=1)/np.sqrt(len(logparams[0][0]))
    ax.plot(trials, mean_deltas)
    ax.fill_between(trials, mean_deltas - sem_deltas, mean_deltas + sem_deltas, alpha=0.2)
    ax.set_ylabel('Norm RM Field Area')
    ax.set_xlabel('$T$')
    return norm_area

def plot_field_size(logparams, trials,ax=None):
    if ax is None:
        f,ax = plt.subplots()
    threshold = 1e-3
    xs = np.linspace(-1,1,1001)
    sizes = []
    for trial in trials:
        pcact = predict_batch_placecell(logparams[trial], xs)
        size = np.mean(pcact>threshold,axis=0)

        sizes.append(size)
    sizes = np.array(sizes)
    norm_area = sizes-sizes[0]
    
    mean_deltas = np.mean(norm_area,axis=1)
    sem_deltas =  1.96*np.std(norm_area,axis=1)/np.sqrt(len(logparams[0][0]))
    ax.plot(trials, mean_deltas)
    ax.fill_between(trials, mean_deltas - sem_deltas, mean_deltas + sem_deltas, alpha=0.2)
    ax.set_ylabel('Norm Field Size')
    ax.set_xlabel('$T$')
    return norm_area


def plot_field_center_(logparams, trials,ax=None):
    # compute field center
    if ax is None:
        f,ax = plt.subplots()
    lambdas = []
    for trial in trials:
        lambdas.append(logparams[trial][0])
    lambdas = np.array(lambdas)
    norm_lambdas = lambdas-lambdas[0]
    mean_deltas = np.mean(norm_lambdas,axis=1)
    sem_deltas =  np.std(norm_lambdas,axis=1)/np.sqrt(len(logparams[0][0]))
    ax.plot(trials, mean_deltas)
    ax.fill_between(trials, mean_deltas - sem_deltas, mean_deltas + sem_deltas, alpha=0.2)
    ax.set_ylabel('Centered RM Fields')
    ax.set_xlabel('$T$')
    return norm_lambdas

def plot_field_center(logparams, trials,ax=None):
    # compute COM
    if ax is None:
        f,ax = plt.subplots()

    xs = np.linspace(-1,1,1001)
    ca3_init = predict_batch_placecell(logparams[0], xs)

    deltas = []
    for trial in trials:
        ca3 = predict_batch_placecell(logparams[trial], xs)
        d = []
        for n in range(ca3.shape[1]):
            # ca3_center = xs[np.argmax(ca3[:,n])]
            orig_ca3_center = xs[np.argmax(ca3_init[:,n])]
            ca1_center = xs[np.argmax(ca3[:,n])]
            delta = ca1_center - orig_ca3_center# - ca3_center
            d.append(delta)
        deltas.append(np.array(d))
    deltas = np.array(deltas)

    mean_deltas = np.mean(deltas,axis=1)
    sem_deltas =  np.std(deltas,axis=1)/np.sqrt(ca3.shape[1])
    ax.plot(trials, mean_deltas, color='tab:blue')
    ax.fill_between(trials, mean_deltas - sem_deltas, mean_deltas + sem_deltas, color='tab:blue', alpha=0.2)
    ax.set_ylabel('$\Delta$ RM Field Centers')
    ax.set_xlabel('$T$')
    
    return deltas



def plot_velocity(logparams, trials, goalcoord=[0.5], startcoord=[-0.75], goalsize=0.05, envsize=1, ax=None):
    if ax is None:
        f,ax = plt.subplots()

    xs = np.linspace(-1,1,1001)
    maxval  = 0 
    for trial in trials:
        pcacts = predict_batch_placecell(logparams[trial], xs)
        actout = pcacts @ logparams[trial][3] 
        aprob = softmax(2 * actout)
        if logparams[0][3].shape[1] == 3:
            vel = np.matmul(aprob, np.array([[-1], [1], [0]]))
        else:
            vel = np.matmul(aprob, np.array([[-1], [1]]))
        vel = np.clip(vel, -1,1) * 0.1 

        ax.plot(xs, vel, label=f'T={trial}')
        maxval = max(maxval, np.max(vel) * 1.1)

    ax.set_xlabel('$x$')
    ax.set_ylabel(r'Velocity $\rho(x)$')
    ax.legend(frameon=False, fontsize=6)
    ax.fill_between(xs, reward_func(xs, goalcoord, goalsize), color='red', alpha=0.25, label='Target')
    ax.axvline(startcoord[0],ymin=0, ymax=0.1, color='g',linestyle='--',label='Start', linewidth=2)
    ax.hlines(xmin=-envsize,xmax=envsize, y=0, colors='k')
    

def plot_pc(logparams, trial,title='', ax=None, goalcoord=[0.5], startcoord=[-0.75], goalsize=0.05, envsize=1, ):
    if ax is None:
        f,ax = plt.subplots()

    xs = np.linspace(-1,1,1001)
    pcacts = predict_batch_placecell(logparams[trial], xs)

    # Get a colormap that transitions from purple to yellow
    cmap = cm.viridis
    num_curves = pcacts.shape[1]
    
    for i in range(num_curves):
        color = cmap(i / num_curves)
        ax.plot(xs, pcacts[:, i], color=color,zorder=1)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$\phi(x)$')
    ax.set_title(title)

    ax.fill_between(xs, reward_func(xs, goalcoord, goalsize), color='red', alpha=0.25, label='Target',zorder=2)
    ax.axvline(startcoord[0],ymin=0, ymax=1, color='g',linestyle='--',label='Start', linewidth=2,zorder=2)
    ax.hlines(xmin=-envsize,xmax=envsize, y=0, colors='k',zorder=2)

def plot_com(logparams,goalcoords,radius, change_trial, ax=None):
    if ax is None:
        f,ax = plt.subplots()
    x = logparams[change_trial][0]
    y = logparams[-1][0]

    ax.plot(np.linspace(-1,1,1000),np.linspace(-1,1,1000), color='gray')

    indices = np.where((x >= goalcoords[0]-radius) & (x <= goalcoords[0]+radius))[0]
    values_in_x = x[indices]
    values_in_y = y[indices]
    ax.scatter(x, y)

    slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(x).reshape(-1), np.array(y).reshape(-1))
    regression_line = slope * np.array(x).reshape(-1) + intercept
    ax.plot(np.array(x).reshape(-1), regression_line, color='g', label=f'R:{np.round(r_value, 3)}, P:{np.round(p_value, 3)}')

    ax.scatter(values_in_x, values_in_y, color='g')
    ax.axvline(goalcoords[0], color='r')
    ax.axhline(goalcoords[1], color='r')
    ax.set_xlabel('Before')
    ax.set_ylabel('After')
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])

def plot_mfa(logparams, trials, ax=None, goalcoord=[0.5], startcoord=[-0.75], goalsize=0.05, envsize=1, color=None):
    # mean firing rate
    if ax is None:
        f,ax = plt.subplots()
    xs = np.linspace(-1,1,1001)

    for trial in trials:
        pcacts = predict_batch_placecell(logparams[trial], xs)
        dx = np.mean(pcacts,axis=1)
        ax.plot(xs, dx, label=f'T={trial}',color=color, zorder=2)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$\sum \phi(x)$')
    ax.legend(frameon=False, fontsize=8,loc=9)

    ax2 = ax.twinx()
    ax2.fill_between(xs, reward_func(xs, goalcoord, goalsize), color='red', alpha=0.25, label='Target', zorder=1)
    ax.axvline(startcoord[0],ymin=0, ymax=1, color='g',linestyle='--',label='Start', linewidth=2, zorder=1)
    ax.hlines(xmin=-envsize,xmax=envsize, y=0, colors='k', zorder=1)
    ax2.set_ylabel('$R(x)$')


def plot_density(logparams, trials, ax=None, goalcoord=[0.5], startcoord=[-0.75], goalsize=0.05, envsize=1, color=None):
    # density based on the number of center of mass in a location
    if ax is None:
        f,ax = plt.subplots()
    xs = np.linspace(-1,1,1001)

    for trial in trials:
        pcacts = predict_batch_placecell(logparams[trial], xs)

        com = xs[np.argmax(pcacts,axis=0)]
        kde = gaussian_kde(com,bw_method=1/11)
        dx = kde(xs)
        ax.plot(xs, dx, label=f'T={trial}',color=color, zorder=2)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$d(x)$')
    ax.legend(frameon=False, fontsize=8,loc=9)

    ax2 = ax.twinx()
    ax2.fill_between(xs, reward_func(xs, goalcoord, goalsize), color='red', alpha=0.25, label='Target', zorder=1)
    ax.axvline(startcoord[0],ymin=0, ymax=1, color='g',linestyle='--',label='Start', linewidth=2, zorder=1)
    ax.hlines(xmin=-envsize,xmax=envsize, y=0, colors='k', zorder=1)
    ax2.set_ylabel('$R(x)$')


def plot_frequency(allcoords, trials,ax=None, goalcoord=[0.5], startcoord=[-0.75], goalsize=0.05, envsize=1, gap=25):
    # plot frequency of being in a location. average over past {gap} trials to get a smooth distribution. 
    if ax is None:
        f,ax = plt.subplots()

    xs = np.linspace(-1,1,1001)
    for trial in trials:
        fx = []
        for g in range(gap):
            fx.append(allcoords[trial-g-1])

        fx = np.array(flatten(fx))
        kde = gaussian_kde(fx.reshape(-1))
        fx_smooth = kde(xs)

        ax.plot(xs, fx_smooth, label=f'T={trial-gap}-{trial}')

    ax.set_xlabel('$x$')
    ax.set_ylabel('$f(x)$')
    ax.legend(frameon=False, fontsize=8,loc=9)

    ax2 = ax.twinx()
    ax2.fill_between(xs, reward_func(xs, goalcoord, goalsize), color='red', alpha=0.25, label='Target', zorder=1)
    ax.axvline(startcoord[0],ymin=0, ymax=1, color='g',linestyle='--',label='Start', linewidth=2, zorder=1)
    ax.hlines(xmin=-envsize,xmax=envsize, y=0, colors='k', zorder=1)
    ax2.set_ylabel('$R(x)$')

def get_center_spread(allcoords):
    allcoords = np.array(allcoords)
    center = np.mean(allcoords)
    spread = np.std(allcoords)
    return center, spread


def plot_place_cells(params,startcoord, goalcoord,goalsize, title='', envsize=1):
    xs = np.linspace(-envsize,envsize,1000)
    pcacts = []
    velocity = []
    for x in xs:
        pc = predict_placecell(params, x)
        actout = np.matmul(pc, params[3])
        aprob = softmax(2 * actout)
        if params[3].shape[1] == 3:
            vel = np.matmul(aprob, np.array([[-1], [1], [0]]))
        else:
            vel = np.matmul(aprob, np.array([[-1], [1]]))
        pcacts.append(pc)
        velocity.append(np.tanh(vel)*0.1)
    pcacts = np.array(pcacts)
    velocity = np.array(velocity)

    plt.figure(figsize=(4,4))
    plt.subplot(211)
    plt.title(title)

    cmap = cm.viridis
    num_curves = pcacts.shape[1]
     
    for i in range(num_curves):
        color = cmap(i / num_curves)
        plt.plot(xs, pcacts[:, i], color=color)

    plt.hlines(xmin=-envsize,xmax=envsize, y=0, colors='k')
    plt.axvline(startcoord[0], color='g',linestyle='--',label='Start', linewidth=2)
    plt.fill_between(xs, reward_func(xs, goalcoord, goalsize), color='red', alpha=0.25, label='Target')
    plt.ylabel('Tuning curves $\phi(x)$')
    plt.xlabel('Location (x)')
    plt.tight_layout()

    plt.subplot(212)
    # plt.plot(xs, np.sum(pcacts,axis=1), color='red')
    com = xs[np.argmax(pcacts,axis=0)]
    kde = gaussian_kde(com,bw_method=1/11)
    dx = kde(xs)
    plt.plot(xs, dx, color='b', zorder=2)
    plt.hlines(xmin=-envsize,xmax=envsize, y=0, colors='k')
    plt.axvline(startcoord[0], color='g',linestyle='--',label='Start', linewidth=2)
    # plt.fill_betweenx(np.linspace(0,np.max(np.sum(pcacts,axis=1))), goalcoord[0]-goalsize, goalcoord[0]+goalsize, color='r', alpha=0.25)
    plt.fill_between(xs, reward_func(xs, goalcoord, goalsize), color='red', alpha=0.25, label='Target')
    plt.ylabel('Field density $d(x)$')
    plt.xlabel('Location (x)')
    ax = plt.twinx()
    ax.plot(xs, velocity, color='k')
    ax.set_ylabel('Avg velocity $V(x)$')
    ax.set_ylim(-0.1,0.1)
    #plt.title('Fixed Rewards: Higher place field density at reward location')
    plt.tight_layout()
    return pcacts


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
    

def get_pvcorr(params, start, end, num):
    xs = np.linspace(-1,1,1001)
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


def plot_gif(logparams, startcoord=[-0.75], goalcoord=[0.5], goalsize=0.025, envsize=1, gif_name='place_cells.gif', num_frames=100, duration=5):
    import imageio
    frames = []
    xs = np.linspace(-envsize, envsize, 1001)
    
    # Select indices for frames to use
    frames_to_use = np.linspace(0, len(logparams) - 1, num_frames, dtype=int)
    
    for p in frames_to_use:
        params = logparams[p]
        pcacts = []
        velocity = []
        
        for x in xs:
            pc = predict_placecell(params, x)
            actout = np.matmul(pc, params[3])
            aprob = softmax(2 * actout)
            if params[3].shape[1] == 3:
                vel = np.matmul(aprob, np.array([[-1], [1], [0]]))
            else:
                vel = np.matmul(aprob, np.array([[-1], [1]]))
            pcacts.append(pc)
            velocity.append(np.clip(vel,-1,1) * 0.1)
        
        pcacts = np.array(pcacts)
        velocity = np.array(velocity)

        plt.figure(figsize=(4, 4))
        for i in range(pcacts.shape[1]):
            plt.plot(xs, pcacts[:, i])
        plt.hlines(xmin=-envsize, xmax=envsize, y=0, colors='k')
        plt.axvline(startcoord[0],ymin=0, ymax= 1, color='g', linestyle='--', label='Start', linewidth=2)
        # plt.fill_betweenx(np.linspace(0,  6.01), goalcoord[0] - goalsize, goalcoord[0] + goalsize, color='r', alpha=0.25)
        plt.fill_between(xs, reward_func(xs, goalcoord, goalsize), color='red', alpha=0.25, label='Target')
        plt.ylabel('Tuning curves $\phi(x)$')
        plt.xlabel('Location (x)')
        plt.title(f'T={p}')
        plt.ylim(-0.01, 6.01)
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        frames.append(imageio.imread(buf))
        buf.close()

    # Save GIF
    imageio.mimsave(gif_name, frames, duration=duration)

    # Example usage:
    # plot_gif(logparams, startcoord, goalcoord, goalsize)


def softmax(x):
    x_max = np.max(x, axis=-1, keepdims=True)
    unnormalized = np.exp(x - x_max)
    return unnormalized/np.sum(unnormalized, axis=-1, keepdims=True)