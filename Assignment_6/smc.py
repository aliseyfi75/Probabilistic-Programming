from evaluator import evaluate
import torch
import numpy as np
import json
import sys






def run_until_observe_or_end(res):
    cont, args, sigma = res
    res = cont(*args)
    while type(res) is tuple:
        if res[2]['type'] == 'observe':
            return res
        cont, args, sigma = res
        res = cont(*args)

    res = (res, None, {'done' : True}) #wrap it back up in a tuple, that has "done" in the sigma map
    return res

def resample_particles(particles, log_weights):
    paricles_length = len(particles)
    weights = torch.exp(torch.FloatTensor(log_weights)) # convert to weights
    normalized_weights = weights + 1e-10 # add a small number to avoid zero weights
    normalized_weights = normalized_weights / normalized_weights.sum() # normalize weights

    logZ = torch.log(torch.mean(weights)) # calculate logZ

    indices = torch.multinomial(normalized_weights, paricles_length, replacement=True)
    new_particles = [particles[i] for i in indices]

    return logZ, new_particles



def SMC(n_particles, exp):

    particles = []
    weights = []
    logZs = []
    output = lambda x: x

    for i in range(n_particles):
        res = evaluate(exp, env=None)('addr_start', output)
        logW = 0.
        particles.append(res)
        weights.append(logW)

    done = False
    smc_cnter = 0
    while not done:
        new_address = ''
        print('In SMC step {}, Zs: '.format(smc_cnter), logZs)
        for i in range(n_particles): 
            res = run_until_observe_or_end(particles[i])
            if 'done' in res[2]: 
                particles[i] = res[0]
                if i == 0:
                    done = True  
                    address = ''
                else:
                    if not done:
                        raise RuntimeError('Failed SMC, finished one calculation before the other')
            else:
                if i == 0:
                    new_address = res[2]['alpha']
                else:
                    address = res[2]['alpha']
                    if address != new_address:
                        raise RuntimeError('Failed SMC, address changed')
                
                log_prob = res[2]['log_prob']
                weights[i] = weights[i] + log_prob
                particles[i] = res

        if not done:
            logZn, particles = resample_particles(particles, weights)
            logZs.append(logZn)
            weights = [0.] * n_particles
        smc_cnter += 1
    logZ = sum(logZs)
    return logZ, particles


if __name__ == '__main__':

    for i in range(1,5):
        with open('/Users/aliseyfi/Documents/UBC/Semester3/Probabilistic-Programming/HW/Probabilistic-Programming/Assignment_6/programs/{}.json'.format(i),'r') as f:
            exp = json.load(f)
        n_particles = 10**3 #TODO 
        logZ, particles = SMC(n_particles, exp)

        print('logZ: ', logZ)

        values = torch.stack(particles)
        #TODO: some presentation of the results
    print(values)