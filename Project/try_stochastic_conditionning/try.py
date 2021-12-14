import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import average
import seaborn as sns
from scipy import stats
import subprocess
import json
import os


relative = "C:/Users/jlovr/CS532-project/Probabilistic-Programming/Project/CTMCs/hairpin/Fig4_0/discotress/Bonnet4False1/"
logvals=False

# def f(t):
#     # k = 2
#     m = 2.3727131391650352e-05
#     return 1/m * np.exp(-t/m)

def discotress(cwd):
    subprocess.run(['discotress'],capture_output=True, cwd=cwd)

def F(t):
    # uses m (mean first passage time)
    # m = 2.3727131391650352e-05
    # return 1 - np.exp(-t/m)  
    
    # same thing, uses k (log_10 of the rate constant) instead of m
    k = 4.553860314632207
    return 1 - np.exp(-t*10**(k)) 


def test_seed_independence(relative):
    scores = []
    for seed in range(0,50):
        os.remove(relative+"input.kmc")
        with open(relative+"input.kmc", "a") as f:
            text = "NNODES " + str(16)+"\n"
            text += "NEDGES " + str(25)+"\n"
            text += "WRAPPER BTOA" + "\n"
            text += "TRAJ BKL" + "\n"
            text += "BRANCHPROBS" + "\n"
            text += "NABPATHS 1000" + "\n"
            text += "COMMSFILE communities.dat " + str(16)+"\n"
            text += "NODESAFILE nodes.A 1"+"\n"
            text += "NODESBFILE nodes.B 1"+"\n"
            text += "NTHREADS 4"+"\n"
            text += "SEED "+str(seed)+"\n"
            f.write(text)
        scores.append(get_score(relative))
    print(scores)
    print("mean: ", np.average(scores))
    print("variance: ", np.var(scores))
    plt.figure(figsize=(10.,7.)) # size in inches
    sns.histplot(x=scores, stat="probability", bins=20)
    plt.xlim([0.0835,0.85])
    plt.xlabel("p(KS_statistic | random seed)")
    plt.title("50 simulator runs, 1000 samples each")
    plt.savefig("peaked_KStest")
    plt.show()


def get_score(relative):
    discotress(relative)

    vals = []
    with open(relative+"fpp_properties.dat","r") as pathprops_f:
        for line in pathprops_f.readlines():
            val=float(line.split()[1])
            vals.append(val)
    
    vals=np.array(vals,dtype=float)
    # plt.figure(figsize=(10.,7.)) # size in inches
    # sns.histplot(x=vals, bins=100, kde=True,stat="probability")
    # sns.histplot(x=vals, bins=100,cumulative=True,stat="density")

    # x1 = np.linspace(0,max(vals),1000)
    # vals2 = F(x1)
    # plt.figure(figsize=(10.,7.)) # size in inches
    # plt.plot(x1, vals2, "red")

    # plt.show()

    # mfpt = average(vals)
    # print("k=",np.log10(1/mfpt))

    print(stats.kstest(vals,cdf=F))
    return stats.kstest(vals,cdf=F)[0]

# plt.figure(figsize=(10.,7.)) # size in inches
# x1 = np.linspace(0,max(vals),1000)
# vals2 = f(x1)
# # plt.figure(figsize=(10.,7.)) # size in inches
# plt.plot(x1, vals2, "red")



# k = 7*1e-5 => KstestResult(statistic=0.010628422809166538, pvalue=3.469857352640312e-10)
# k = 8*1e-4 => KstestResult(statistic=0.7176417217713222, pvalue=0.0)
# k = 8*1e-5 => KstestResult(statistic=0.03974433625725693, pvalue=6.3093989949052205e-137)
# k = 2      => KstestResult(statistic=0.999522480152335, pvalue=0.0)
# GOOD k values give LOW statistic. so 1 minus statistic could be likelihood. Or statistic ~ Normal(0,1)
        

# bonnet hairpin 1 exponential

test_seed_independence(relative)
# get_score(relative)











