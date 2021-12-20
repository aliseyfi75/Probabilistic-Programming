import matplotlib.pyplot as plt
import numpy as np
from statistics import variance as var, mean
from numpy import asarray
from numpy import arange
from numpy import meshgrid
import torch
import seaborn as sns
import scipy.stats as stats
import math


def weighted_avg(X, weights):
    return (weights.dot(X)) / weights.sum()

def plots(return_values, prob_sequence, prob_means, q, L, lr):

    num_samples = len(return_values)

    # ELBO trace plot
    plt.figure(figsize=(5,4))
    plt.xlabel("Iterations")
    plt.ylabel("ELBO")
    plt.title("ELBO trace plot")
    plt.plot(prob_means)
    figstr = "save/elbo_plot/elbo"+"_L_"+str(L)+"_lr_"+str(lr)+".jpg"
    plt.savefig(figstr)
    # plt.show()
    print("Last ELBO", prob_sequence[-1])

    for n in range(num_samples):
        return_values[n] = [float(x) for x in return_values[n]]
    
    variables = np.array(return_values,dtype=object).T.tolist()
    # variables = np.array(return_values,dtype=object).tolist()


    for d in range(len(variables)):

        plt.figure(figsize=(5,4))
        v = "sample"+str(d+1)
        xname = "theta["+str(d+1)+"]"
        plt.xlabel(xname)
        plt.ylabel("density")
        plt.title("Weighted posterior probability of " + xname)
        sns.histplot(x=variables[d], weights=np.exp(prob_sequence), kde=True, bins=50, stat="probability")
        figstr = "save/posterior_plot/" + xname + "_L_"+str(L)+"_lr_"+str(lr)+".jpg"
        plt.savefig(figstr)
        # plt.show()

        plt.figure(figsize=(5,4))
        plt.xlabel("mu")
        plt.ylabel("density")
        plt.title("Plot of final q")
        figstr = "save/q_plot/" + xname + "_L_"+str(L)+"_lr_"+str(lr)+".jpg"
        mu = float(q[v].Parameters()[0])
        variance = float(q[v].Parameters()[1])
        s = math.sqrt(abs(variance))
        x = np.linspace(min(variables[d]), max(variables[d]), 100)
        plt.plot(x, stats.norm.pdf(x, mu, s))
        plt.savefig(figstr)
        # plt.show()

        W = np.exp(prob_sequence)
        means = weighted_avg(variables[d], W)
        vars = weighted_avg((variables[d] - means)**2, W)

        print("mean", means)
        print("variance", vars)
        plt.close()
