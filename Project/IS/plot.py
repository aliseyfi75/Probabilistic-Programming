import os,torch,itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white")
dirn = os.path.dirname(os.path.abspath(__file__))
labels = {1:{0:"theta0", 1:"theta1", 2:"theta2", 3:"theta3", 4:"theta4", 5:"theta5", 6:"theta6", 7:"theta7", 8:"theta8", 9:"theta9", 10:"theta10", 11:"theta11", 12:"theta12", 13:"theta13", 14:"theta14", 15:"theta15"}}
def get_title(eval,i,dim=False,d=0, fig_type='hist'):
    if dim:
        label = labels[i][d]
        return "{} Sampled {} for Program {} {}".format(eval, fig_type, i, label)
    else:
        return "{} Sampled {} for Program {}".format(eval,fig_type, i)
def get_fname(eval,i,dim=False,d=0, fig_type='hist', alpha=None):
    if dim:
        if alpha != None:
            return dirn + '/figures/{}_plt_{}_program_{}_d_{}_alpha_{}.jpg'.format(eval,fig_type, i,d,alpha)
        else:
            return dirn + '/figures/{}_plt_{}_program_{}_d_{}.jpg'.format(eval,fig_type, i,d)
    else:
        return dirn + '/figures/{}_plt_{}_program_{}.jpg'.format(eval,fig_type, i)
def plot_hists_n(eval, i, hists, weights, alpha=None):
    for d in range(len(hists)):
        samples = np.array(hists[d])
        if samples.ndim < 3:
            # 2D >= random var
            plot_hist_arr(eval,i, samples, weights=weights,dim=True,d=d,alpha=alpha)
        else:
            # ND random var 
            _,k,_ = samples.shape 
            fig, axs = plt.subplots(2,5, figsize=(20,10), dpi=100)
            for j, ax in zip(range(k), axs.flat):
                try:
                    sns.histplot(x=samples[:][j],kde=True,stat='density',cbar=True,multiple='dodge', ax=ax, weights=weights)
                except:
                    pass
                ax.title.set_text('range over {}[{}]'.format(labels[i][d],j))
            fname = get_fname(eval, i,True,d)
            title = get_title(eval, i, True, d)
            fig.suptitle(title, fontsize=24)
            plt.savefig(fname)
            plt.clf()
            plt.close()

def plot_traces_n(eval, i, traces):
    for d in range(len(traces)):
        samples = np.array(traces[d])
        if samples.ndim < 3:
            # 2D >= random var
            plot_trace_arr(eval, i, samples,dim=True,d=d)
        else:
            # ND random var 
            _,k,_ = samples.shape 
            fig, axs = plt.subplots(2,5, figsize=(20,10), dpi=100)
            for j, ax in zip(range(k), axs.flat):
                sns.lineplot(data=samples[:][j],ax=ax)
                ax.title.set_text('range over {}[{}]'.format(labels[i][d],j))
            fname = get_fname(eval, i,True,d, fig_type='trace')
            title = get_title(eval, i, True, d, fig_type='trace')
            fig.suptitle(title, fontsize=24)
            plt.savefig(fname)
            plt.clf()
            plt.close()
def plot_hist_arr(eval, i, samples, weights, dim=False, d=0, alpha=None):
    plt.figure(figsize=(10,7))
    nbins = int(max(samples.max()-samples.min(),30))
    try:
        sns.histplot(x=samples,kde=True,stat='density',cbar=True,multiple='dodge',bins=nbins, weights=weights)
    except:
        pass
    fname = get_fname(eval, i,dim,d, alpha=alpha)
    title = get_title(eval, i, dim, d)
    plt.title(title)
    plt.savefig(fname)
    plt.clf()
    plt.close()

def plot_trace_arr(eval, i, samples, dim=False, d=0):
    plt.figure(figsize=(10,7))
    plt.plot(samples)
    fname = get_fname(eval, i,dim,d, fig_type='trace')
    title = get_title(eval, i, dim, d, fig_type='trace')
    plt.title(title)
    plt.savefig(fname)
    plt.clf()
    plt.close()

def plot_hist(eval, i, samples, weights, bins):
    plt.figure(figsize=(10,7))
    try:
        sns.histplot(x=samples,bins=bins,kde=True, stat='density', weights=weights)
    except:
        pass
    fname = get_fname(eval, i)
    title = get_title(eval, i)
    plt.title(title)
    plt.savefig(fname)
    plt.clf()
    plt.close()

def plot_trace(eval, i, samples):
    plt.figure(figsize=(10,7))
    plt.plot(samples)
    fname = get_fname(eval, i, fig_type='trace')
    title = get_title(eval, i, fig_type='trace')
    plt.title(title)
    plt.savefig(fname)
    plt.clf()
    plt.close()

def gen_hists(eval, i, samples, weights, alpha=None):
    try:
        # list of c
        min_v, max_v = min(samples), max(samples)
        nbins = int(min(max_v-min_v,30))
        bins = np.linspace(min_v, max_v,nbins)
        plot_hist(eval, i, samples, bins, weights=weights)
    except:
        try:
            # 1 rnd var
            assert(samples.dim <= 1)
            samples = samples.detach().numpy().T
            plot_hist_arr(eval, i, samples, weights=weights, alpha=alpha)
        except:
            # n rnd vars
            hists = []
            samples = samples.detach().numpy()
            for d in range(samples.shape[0]):
                hists.append(samples[d,:].T)
            plot_hists_n(eval, i, hists, weights=weights, alpha=alpha)

def gen_traces(eval, i, samples):
    try:
        # list of c
        min_v, max_v = min(samples), max(samples)
        nbins = int(min(max_v-min_v,30))
        bins = np.linspace(min_v, max_v,nbins)
        plot_trace(eval, i, samples, bins)
    except:
        try:
            # 1 rnd var
            assert(samples.dim <= 1)
            samples = samples.detach().numpy().T
            plot_trace_arr(eval, i, samples)
        except:
            # n rnd vars
            traces = []
            samples = samples.detach().numpy()
            for d in range(samples.shape[0]):
                traces.append(samples[d,:].T)
            plot_traces_n(eval, i, traces)

def draw_hists(eval, samples, i, weights=None,alpha=None):
    gen_hists(eval, i, samples, weights=weights, alpha=alpha)

def draw_trace(eval, samples, i):
    gen_traces(eval, i, samples)

def draw_log_joint(eval, i, graph, nodes_values, deterministic_eval, value_subs):
    log_probs = torch.zeros(len(nodes_values))
    for idx, node_value in enumerate(nodes_values):
        for node in node_value:
            if graph[1]['P'][node][0] == 'sample*' or graph[1]['P'][node][0] == 'observe*':
                log_probs[idx] += deterministic_eval(value_subs(graph[1]['P'][node][1], node_value)).log_prob(node_value[node]).detach().numpy()
            
    plt.figure(figsize=(10,7))
    plt.plot(log_probs)
    fname = get_fname(eval, i, fig_type='log_joint')
    title = get_title(eval, i, fig_type='log_joint')
    plt.title(title)
    plt.savefig(fname)
    plt.clf()
    plt.close()

