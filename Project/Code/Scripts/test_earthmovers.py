import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
import scipy.stats as stats
import seaborn as sns

relative = "C:/Users/jlovr/CS532-project/Probabilistic-Programming/Project/CTMCs/hairpin/Fig4_0/discotress/Bonnet4False1/0.25648893383273186/"

def F(t):
    # uses m (mean first passage time)
    # m = 2.3727131391650352e-05
    # return 1-np.exp(-t/m)
    
    # same thing, uses k (log_10 of the rate constant) instead of m
    # print("real k", real_log_10_rate)
    real_log_10_rate = 4.553860314632207

    return 1 - np.exp(-t*10**(real_log_10_rate))     

def f(t):
    # uses m (mean first passage time)
    # m = 2.3727131391650352e-05
    # return 1-np.exp(-t/m)
    
    # same thing, uses k (log_10 of the rate constant) instead of m
    # print("real k", real_log_10_rate)
    real_log_10_rate = 4.553860314632207

    return 10**real_log_10_rate*np.exp(-t*10**(real_log_10_rate))  

    # 4.978717624582975 4.553860314632207

def get_wasserstein():
    vals = []
    with open(relative+"fpp_properties.dat","r") as pathprops_f:
        for line in pathprops_f.readlines():
            val=float(line.split()[1])
            vals.append(val)
    vals=np.array(vals,dtype=float)

    hist_arr = np.zeros(100,dtype=int)
    width = max(vals)/99
    for val in vals:
        if not (val<0):
            bin = int(np.floor(val/width))
            hist_arr[bin] += 1
    summ = np.sum([width*h for h in hist_arr])
    hist_arr = [x/summ for x in hist_arr]
    X = np.linspace(0, max(vals), num=100)
    return stats.wasserstein_distance(hist_arr[10:90],f(X)[10:90])

get_wasserstein()