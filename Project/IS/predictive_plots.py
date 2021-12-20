import numpy as np

def F(self, t):
    # uses m (mean first passage time)
    # m = 2.3727131391650352e-05
    # return 1 - np.exp(-t/m)  
    
    # same thing, uses k (log_10 of the rate constant) instead of m
    # print("real k", self.real_log_10_rate)
    return 1 - np.exp(-t*10**(self.real_log_10_rate)) 


def noise_around_F(k, ks):
    
    vals = 1000 points between 0 and xx
    for each time t:
        d = sample uniformly between -ks and ks
        vals = F(t)+ks

        random.uniform(low=0.0, high=1.0, size=None)
