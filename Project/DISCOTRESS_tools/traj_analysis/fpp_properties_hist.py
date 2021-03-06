'''
Python script to analyse the probability distributions of dynamical quantities in the first passage path ensemble. These statistics are
printed to the "fpp_properties.dat" output file from DISCOTRESS. Namely: the first passage time (FPT) distribution, and the path length,
path action (i.e. negative of log path probability), and path entropy flow distributions.
The script plots a histogram of the probability distribution and calculates the mean and variance of the distribution (and the associated
standard errors) for the chosen first passage path property.

Daniel J. Sharpe
Jan 2020
'''

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from math import floor
from math import sqrt
import seaborn as sns



# concentration = 10-8
relative = 'C:/Users/jlovr/CS532-project/Probabilistic-Programming/Project/CTMCs/three_waystranddisplacement/Fig3b/discotress/Zhang4/'


# concentration = 1
relative = "C:/Users/jlovr/CS532-project/Probabilistic-Programming/Project/CTMCs/hairpin/Fig4_0/discotress/Bonnet4False1/"
# relative = "C:/Users/jlovr/CS532-project/Probabilistic-Programming/Project/CTMCs/hairpin/Fig4_0/discotress/Bonnet4False2/"
# relative = 'C:/Users/jlovr/CS532-project/Probabilistic-Programming/Project/CTMCs/hairpin1/Fig3_T_0/discotress/GoddardTFalse1/'
# relative = 'C:/Users/jlovr/CS532-project/Probabilistic-Programming/Project/CTMCs/hairpin1/Fig3_T_0/discotress/GoddardTFalse2/'
# relative = 'C:/Users/jlovr/CS532-project/Probabilistic-Programming/Project/CTMCs/hairpin1/Fig3_T_0/discotress/GoddardTFalse3/'
# relative = 'C:/Users/jlovr/CS532-project/Probabilistic-Programming/Project/CTMCs/hairpin1/Fig3_T_0/discotress/GoddardTFalse4/'
# relative = 'C:/Users/jlovr/CS532-project/Probabilistic-Programming/Project/CTMCs/hairpin1/Fig3_T_1/discotress/GoddardTTrue1/'
# relative = 'C:/Users/jlovr/CS532-project/Probabilistic-Programming/Project/CTMCs/hairpin1/Fig3_T_1/discotress/GoddardTTrue2/'
# relative = 'C:/Users/jlovr/CS532-project/Probabilistic-Programming/Project/CTMCs/hairpin1/Fig3_T_1/discotress/GoddardTTrue3/'
# relative = 'C:/Users/jlovr/CS532-project/Probabilistic-Programming/Project/CTMCs/hairpin1/Fig3_T_1/discotress/GoddardTTrue4/'







class Analyse_fpp_properties(object):

    def __init__(self,stat,nbins,binw,bin_min,binall,logvals):
        if stat<1 or stat>4: raise RuntimeError
        self.stat=stat
        self.nbins=nbins
        self.binw=binw
        self.bin_min=bin_min
        self.bin_max=self.bin_min+(self.nbins*self.binw)
        self.binall=binall
        self.logvals=logvals
        self.ntpaths=0
        self.vals=None

    def get_hist_arr(self):
        hist_arr = np.zeros(self.nbins,dtype=int)
        vals=[]
        with open(relative+"fpp_properties.dat","r") as pathprops_f:
            for line in pathprops_f.readlines():
                val=float(line.split()[stat])
                if self.logvals: val=np.log10(val)
                vals.append(val)
                if not (val>=self.bin_max or val<self.bin_min):
                    hist_arr[int(floor((val-self.bin_min)/self.binw))] += 1
                elif self.binall:
                    print("found bad value: ",val,"for path: ",self.ntpaths+1)
                    raise RuntimeError
                self.ntpaths+=1
        self.vals=np.array(vals,dtype=float)
        return hist_arr

    def plot_hist(self,hist_arr,nxticks,nyticks,ymax,fpd_name,figfmt="pdf",color="cornflowerblue",xtick_dp=0,ytick_dp=2,
                  linevals=None,linecolor="deeppink"):
        hist_arr=hist_arr.astype(np.float64)*1./float(self.ntpaths) # normalise
        bins=[self.bin_min+(i*self.binw) for i in range(self.nbins)]
        plt.figure(figsize=(10.,7.)) # size in inches
        sns.histplot(x=self.vals, bins=100, kde=True,stat="probability")
        plt.figure(figsize=(10.,7.)) # size in inches
        sns.histplot(self.vals, bins=100, kde=True,cumulative=True,stat="density")
        if self.logvals:
            plt.xlabel("$\log_{10}("+fpd_name+")$",fontsize=42)
            plt.ylabel("$p ( \log_{10} ("+fpd_name+") )$",fontsize=42)
        else:
            plt.xlabel("$"+fpd_name+"$",fontsize=42)
            plt.ylabel("$p("+fpd_name+")$",fontsize=42)
        if linevals is not None:
            plt.vlines(linevals,0.,ymax,colors=linecolor,linewidths=6.,linestyles="dashed")
        plt.savefig("fp_distribn."+figfmt,format=figfmt,bbox_inches="tight")
        plt.show()

    ''' calculate mean of first passage time (FPT) distribution '''
    def calc_mfpt(self):
        if not self.logvals: self.mfpt = np.sum(self.vals)/float(self.ntpaths)
        else: self.mfpt = np.sum([10**val for val in self.vals])/float(self.ntpaths)
        return self.mfpt

    def calc_rate(self):
        # concentration = 1e-8
        concentration = 1
        print("USING CONCENTRATION, make sure set correctly")
        return np.log10(1/(concentration*self.mfpt))
        # return np.log10(1/self.calc_mfpt())

    ''' calculate variance of first passage time (FPT) distribution '''
    def calc_var_fptd(self):
        if not self.logvals:
            var = (np.sum(np.array([val**2 for val in self.vals]))/float(self.ntpaths))-((np.sum(self.vals)/float(self.ntpaths))**2)
        else:
            var = (np.sum(np.array([(10**val)**2 for val in self.vals]))/float(self.ntpaths))-\
                  ((np.sum(np.array([10**val for val in self.vals]))/float(self.ntpaths))**2)
        return var

    ''' calculate standard error associated with the MFPT. (Alternatively, =sqrt(var)/sqrt(n)) '''
    def calc_stderr_mfpt(self,mfpt):
        stderr=0.
        for val in self.vals:
            if not self.logvals: stderr+=(val-mfpt)**2
            else: stderr+=(10**val-mfpt)**2
        return sqrt((1./float(self.ntpaths-1))*stderr)/sqrt(float(self.ntpaths))

if __name__=="__main__":
    ### CHOOSE PARAMS ###

    # statistic to analyse
    # 1=time, 2=dynamical activity (path length), 3=-ln(path prob) [path action], 4=entropy flow
    # TODO: option 4 doesn't seem to work, yet. 
    stat=1

    # binning params

    nbins=40
    binw=0.1
    bin_min=1.
    binall=False # enforce that all values must be encompassed in the bin range
    logvals=False # take log_10 of values
    # plot params
    nxticks=8 # no. of ticks on x axis
    nyticks=10 # no. of ticks on y axis
    ymax=0.1 # max value for y (prob) axis
    # # can add one or more vertical lines to plot (e.g. to indicate mean value)
    linevals = np.array([3.612046E+03])

    # run
    calc_hist_obj=Analyse_fpp_properties(stat,nbins,binw,bin_min,binall,logvals)
    hist_arr = calc_hist_obj.get_hist_arr()
    # print("\nhistogram bin counts:\n",hist_arr)
    # print("\ntotal number of observed A<-B transition paths:\t",calc_hist_obj.ntpaths)
    # print("total number of binned A<-B transition paths:\t",np.sum(hist_arr))
    mfpt = calc_hist_obj.calc_mfpt()
    rate = calc_hist_obj.calc_rate()
    # var_fptd = calc_hist_obj.calc_var_fptd()
    # std_err = calc_hist_obj.calc_stderr_mfpt(mfpt)
    # std_dev=sqrt(var_fptd)
    print("\nestimate of rate with MFPT:\t\t","{:.6e}".format(rate))
    print("mean of FPT distribution (MFPT):\t","{:.6e}".format(mfpt))
    # print("variance of FPT distribution:\t\t","{:.6e}".format(var_fptd))
    # print("standard error in MFPT:\t\t\t","{:.6e}".format(std_err))
    # print("standard error in var:\t\t\t","{:.6e}".format(var_fptd*sqrt(2./(calc_hist_obj.ntpaths-1.))))
    # plot
    if logvals: linevals = np.log10(linevals)
    fpd_name=None
    if stat==1: fpd_name = "t_\mathrm{FPT}"
    elif stat==2: fpd_name = "\mathcal{L}"
    elif stat==3: fpd_name = "- \ln \mathcal{P}"
    elif stat==4: fpd_name = "\mathcal{S} / k_\mathrm{B}"
    else: quit("error in choice of stat")
    calc_hist_obj.plot_hist(hist_arr,nxticks,nyticks,ymax,fpd_name,figfmt="pdf",xtick_dp=1,ytick_dp=2)
