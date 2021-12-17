from numpy.core.fromnumeric import argmax, argmin
from evaluator import evaluate
import torch
import numpy as np
import json
import sys
import threading
import matplotlib.pyplot as plt
from daphne import daphne

from primitives import log
from plots import plots
import time




def run_until_observe_or_end(res):
    cont, args, sigma = res
    res = cont(sigma, *args)
    while type(res) is tuple:
        if res[2]['type'] == 'observe':
            return res
        cont, args, sigma = res
        res = cont(sigma, *args)

    res = (res, None, {'done' : True}) #wrap it back up in a tuple, that has "done" in the sigma map
    return res

def resample_particles(particles, log_weights, count):
    L = len(log_weights)
    log_ws = torch.FloatTensor(log_weights)
    new_particles = particles

    if count%50==49:
        discrete_dist = torch.distributions.categorical.Categorical(logits=log_ws)
        for i in range(L):
            k = discrete_dist.sample()
            new_particles[i] = particles[k]
        
        # print('intermediate variance: ', np.diag(np.cov(torch.stack(new_particles).float().detach().numpy(),rowvar=False)))  

    logZ = torch.logsumexp(log_ws,0) - torch.log(torch.tensor(log_ws.shape[0],dtype=float))

    return logZ, new_particles


def SMC(n_particles, exp):
    particles = []
    weights = []
    logZs = []
    sigma = {'logW':0}
    output = lambda _, x: x

    for i in range(n_particles):
        cont, args, sigma = evaluate(exp, sigma, env=None)(sigma, 'addr_start', output)
        logW = 0.
        weights.append(logW)
        res = cont, args, {'logW':weights[i]}
        particles.append(res)

    done = False
    smc_cnter = 0
    count=0
    while not done:
        for i in range(n_particles): #Even though this can be parallelized, we run it serially
            res = run_until_observe_or_end(particles[i])
            if 'done' in res[2]: #this checks if the calculation is done
                particles[i] = res[0]
                if i == 0:
                    done = True  #and enforces everything to be the same as the first particle
                    address = ''    
                else:
                    if not done:        # is the /first/ particle i=0 done?
                        raise RuntimeError('Failed SMC, finished one calculation before the other')
            else:  # res[2] == 'observe'
                cont, args, sigma = res
                weights[i] = res[2]['logW'].clone().detach()        # get weights
                particles[i] = cont, args, {'logW':weights[i]}      # get continuation

                if i == 0:
                    address = sigma['alpha']
                try:
                    assert(sigma['alpha'] == address)
                except:
                    raise AssertionError('particle address error')

        if not done:
            count+=1
            logZn, particles = resample_particles(particles, weights, count)
            logZs.append(logZn)
            
        smc_cnter += 1  # number of continuations/observes completed. 

    if logZs == []:
        return 0, particles
    else:
        return logZs[-1], particles


def my_main():

    # exp = daphne(['desugar-hoppl-cps', '-i', 'C:/Users/jlovr/CS532-project/Probabilistic-Programming/Project/smc/programs/{}.daphne'.format(7)])
    # with open('C:/Users/jlovr/CS532-project/Probabilistic-Programming/Project/smc/programs/{}.daphne'.format(7),'w') as f:
    #     json.dump(exp, f)

    with open('C:/Users/jlovr/CS532-project/Probabilistic-Programming/Project/smc/programs/{}.daphne'.format(7),'r') as f:
        exp = json.load(f)

    logZ_list = []

    # for n_particles in [5,50,500]:
    for n_particles in [2]:
        start = time.time()
        
        logZ, particles = SMC(n_particles, exp)

        values = torch.stack(particles)
        
        #### presentation of the results

        print("Number of particles:", n_particles)

        print('posterior mean:', values.float().detach().numpy().mean(axis=0))
        if n_particles > 1:
            print('posterior variance: ', np.diag(np.cov(values.float().detach().numpy(),rowvar=False)))  
            
        print("logZ:", np.array(logZ, dtype=float))
        logZ_list.append(logZ)
        
        plots(particles, n_particles)
        end = time.time()
        print("time: ", end - start)
        print("\n\n\n")
        
    plt.figure(figsize=(8,4))
    plt.xlabel("$\log_{10} (n)$")
    plt.ylabel("logZ")
    plt.title("Marginal log-probability estimate")

    plt.plot(logZ_list)
    figstr = "logZ_estimates/program"
    plt.savefig(figstr)

if __name__ == '__main__':
    sys.setrecursionlimit(100000)
    threading.stack_size(200000000)
    thread = threading.Thread(target=my_main)
    thread.start()     


