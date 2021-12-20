import sys
import numpy as np
from torch import distributions
from torch.distributions import distribution
import scipy.stats as stats
import time
import torch
from tqdm import tqdm
import argparse
from joblib import Parallel, delayed

sys.path.append('../../')
sys.path.append('../../Code/Scripts')
sys.path.append('../../Code/')

from Scripts.model import open_csv
from Scripts.model import *
from Scripts.test_theta_on_all import eval_theta_all

parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, default=100)

n_particles = parser.parse_args().N

PATH = '/home/aliseyfi/scratch/Probabilistic-Programming/Project/'
# PATH = '/Users/aliseyfi/Documents/UBC/Probabilistic-Programming/Probabilistic-Programming/Project/'
# PATH = "C:/Users/jlovr/CS532-project/Probabilistic-Programming/Project/"

datasets = {    "hairpin" : ["Fig4_0", "Fig4_1", "Fig6_0", "Fig6_1"],
                "hairpin1" : ["Fig3_T_0", "Fig3_T_1"],
                "hairpin4" : ["Table1_0", "Table1_1"],
                "helix" : ["Fig6_0", "Fig6_1"],
                "helix1" : ["Fig6a"],
                "three_waystranddisplacement" : ["Fig3b"],
                "three_waystranddisplacement1" : ["Fig6b"],
                "bubble": ["Fig4"],
                "four_waystrandexchange": ["Table5.2"]
    }

def from_theta_to_rate_bubble(theta, kinetic_model="ARRHENIUS"):
    reaction_type = 'bubble'
    predicted_log_10_rates, real_log_10_rates = [], []
    for reaction_dataset in datasets[reaction_type]:
        reaction_id = "/" + reaction_type + "/" + reaction_dataset
        document_name = PATH + "/dataset" + reaction_id + ".csv"
        file =  open_csv(document_name)
        row = 1
        while row < len(file) and file[row][0] != '' :
            predicted_log_10_rate, real_log_10_rate, sq_error = estimate_AltanBonnet(row, theta, file, reaction_id, str(row), "Altanbonnet", kinetic_model)
            predicted_log_10_rates.append(predicted_log_10_rate)
            real_log_10_rates.append(real_log_10_rate)
            row+=1
    return predicted_log_10_rates, real_log_10_rates

def from_theta_to_rate_four_waystrandexchange(theta, kinetic_model="ARRHENIUS"):
    reaction_type = 'four_waystrandexchange'
    predicted_log_10_rates, real_log_10_rates = [], []
    for reaction_dataset in datasets[reaction_type]:
        reaction_id = "/" + reaction_type + "/" + reaction_dataset
        document_name = PATH + "/dataset" + reaction_id + ".csv"
        file =  open_csv(document_name)
        row = 1
        while row < len(file) and file[row][0] != '' :
            predicted_log_10_rate, real_log_10_rate, sq_error = estimate_DabbyThesis(row, theta, file, reaction_id, str(row), "Dabby", kinetic_model)
            predicted_log_10_rates.append(predicted_log_10_rate)
            real_log_10_rates.append(real_log_10_rate)
            row+=1
    return predicted_log_10_rates, real_log_10_rates

def from_theta_to_rate_hairpin(theta, kinetic_model="ARRHENIUS"):
    reaction_type = 'hairpin'
    predicted_log_10_rates, real_log_10_rates = [], []
    for reaction_dataset in datasets[reaction_type]:
        _zip = bool(int(reaction_dataset[-1]))
        j = reaction_dataset[-3]
        reaction_id = "/" + reaction_type + "/" + reaction_dataset
        document_name = PATH + "/dataset" + reaction_id + ".csv"
        file =  open_csv(document_name)
        row = 1
        while row < len(file) and file[row][0] != '' :
            predicted_log_10_rate, real_log_10_rate, sq_error = estimate_Bonnet(row, theta, _zip, file, reaction_id, str(row), "Bonnet"+j, kinetic_model)
            predicted_log_10_rates.append(predicted_log_10_rate)
            real_log_10_rates.append(real_log_10_rate)
            row+=1
    return predicted_log_10_rates, real_log_10_rates

def from_theta_to_rate_hairpin1(theta, kinetic_model="ARRHENIUS"):
    reaction_type = 'hairpin1'
    predicted_log_10_rates, real_log_10_rates = [], []
    for reaction_dataset in datasets[reaction_type]:
        _zip = bool(int(reaction_dataset[-1]))
        reaction_id = "/" + reaction_type + "/" + reaction_dataset
        document_name = PATH + "/dataset" + reaction_id + ".csv"
        file =  open_csv(document_name)
        row = 1
        while row < len(file) and file[row][0] != '' :
            predicted_log_10_rate, real_log_10_rate, sq_error = estimate_BonnetThesis(row, theta, _zip, file, reaction_id, str(row), "GoddardT", kinetic_model)
            predicted_log_10_rates.append(predicted_log_10_rate)
            real_log_10_rates.append(real_log_10_rate)
            row+=1
    return predicted_log_10_rates, real_log_10_rates

def from_theta_to_rate_hairpin4(theta, kinetic_model="ARRHENIUS"):
    reaction_type = 'hairpin4'
    predicted_log_10_rates, real_log_10_rates = [], []
    for reaction_dataset in datasets[reaction_type]:
        _zip = bool(int(reaction_dataset[-1]))
        reaction_id = "/" + reaction_type + "/" + reaction_dataset
        document_name = PATH + "/dataset" + reaction_id + ".csv"
        file =  open_csv(document_name)
        row = 1
        while row < len(file) and file[row][0] != '' :
            predicted_log_10_rate, real_log_10_rate, sq_error = estimate_Kim(row, theta, _zip, file, reaction_id, str(row), "Kim", kinetic_model)
            predicted_log_10_rates.append(predicted_log_10_rate)
            real_log_10_rates.append(real_log_10_rate)
            row+=1
    return predicted_log_10_rates, real_log_10_rates

def from_theta_to_rate_helix(theta, kinetic_model="ARRHENIUS"):
    reaction_type = 'helix'
    predicted_log_10_rates, real_log_10_rates = [], []
    for reaction_dataset in datasets[reaction_type]:
        _zip = bool(int(reaction_dataset[-1]))
        reaction_id = "/" + reaction_type + "/" + reaction_dataset
        document_name = PATH + "/dataset" + reaction_id + ".csv"
        file =  open_csv(document_name)
        row = 1
        while row < len(file) and file[row][0] != '' :
            predicted_log_10_rate, real_log_10_rate, sq_error = estimate_Morrison(row, theta, _zip, file, reaction_id, str(row), "Morrison", kinetic_model)
            predicted_log_10_rates.append(predicted_log_10_rate)
            real_log_10_rates.append(real_log_10_rate)
            row+=1
    return predicted_log_10_rates, real_log_10_rates

def from_theta_to_rate_helix1(theta, kinetic_model="ARRHENIUS"):
    reaction_type = 'helix1'
    predicted_log_10_rates, real_log_10_rates = [], []
    for reaction_dataset in datasets[reaction_type]:
        _zip = False
        reaction_id = "/" + reaction_type + "/" + reaction_dataset
        document_name = PATH + "/dataset" + reaction_id + ".csv"
        file =  open_csv(document_name)
        row = 1
        while row < len(file) and file[row][0] != '' :
            predicted_log_10_rate, real_log_10_rate, sq_error = estimate_ReynaldoDissociate(row, theta, _zip, file, reaction_id, str(row), "ReynaldoDissociate", kinetic_model)
            predicted_log_10_rates.append(predicted_log_10_rate)
            real_log_10_rates.append(real_log_10_rate)
            row+=1
    return predicted_log_10_rates, real_log_10_rates

def from_theta_to_rate_three_waystranddisplacement(theta, kinetic_model="ARRHENIUS"):
    reaction_type = 'three_waystranddisplacement'
    predicted_log_10_rates, real_log_10_rates = [], []
    for reaction_dataset in datasets[reaction_type]:
        reaction_id = "/" + reaction_type + "/" + reaction_dataset
        document_name = PATH + "/dataset" + reaction_id + ".csv"
        file =  open_csv(document_name)
        row = 1
        while row < len(file) and file[row][0] != '' :
            predicted_log_10_rate, real_log_10_rate, sq_error = estimate_Zhang(row, theta, file, reaction_id, str(row), "Zhang", kinetic_model)
            predicted_log_10_rates.append(predicted_log_10_rate)
            real_log_10_rates.append(real_log_10_rate)
            row+=1
    return predicted_log_10_rates, real_log_10_rates

def from_theta_to_rate_three_waystranddisplacement1(theta, kinetic_model="ARRHENIUS"):
    reaction_type = 'three_waystranddisplacement1'
    predicted_log_10_rates, real_log_10_rates = [], []
    for reaction_dataset in datasets[reaction_type]:
        reaction_id = "/" + reaction_type + "/" + reaction_dataset
        document_name = PATH + "/dataset" + reaction_id + ".csv"
        file =  open_csv(document_name)
        row = 1
        while row < len(file) and file[row][0] != '' :
            predicted_log_10_rate, real_log_10_rate, sq_error = estimate_ReyanldoSequential(row, theta, file, reaction_id, str(row), "ReynaldoSequential", kinetic_model)
            predicted_log_10_rates.append(predicted_log_10_rate)
            real_log_10_rates.append(real_log_10_rate)
            row+=1
    return predicted_log_10_rates, real_log_10_rates


def log_prob(k, real, sigma):
    return -np.log(sigma*np.sqrt(2*np.pi)) - 0.5*((k-real)/sigma)**2

def list_log_prob(ks, reals, sigma):
    return np.sum([log_prob(k, real, sigma) for k, real in zip(ks, reals)])

def expectation_calculator(results, log_weights, func, *args):
    weights = np.exp(log_weights)
    func_result = func(results, *args)
    return np.sum(weights*func_result, axis=0) / np.sum(weights)

def resample_particles(particles, log_weights):
    L = len(log_weights)
    log_ws = torch.FloatTensor(log_weights)
    new_particles = particles

    discrete_dist = torch.distributions.categorical.Categorical(logits=log_ws)
    for i in range(L):
        k = discrete_dist.sample()
        new_particles[i] = particles[k]
        
    logZ = torch.logsumexp(log_ws,0) - torch.log(torch.tensor(log_ws.shape[0],dtype=float))

    return logZ, new_particles


if __name__ == '__main__':

    start_start_time = time.time()
    prior = [13.0580, 3, 13.0580, 3,  13.0580, 3, 13.0580, 3,  13.0580, 3, 13.0580, 3,  13.0580, 3,   0.0402 ]
    sigma = 1
    thetas = [[np.random.normal(i, 1) for i in prior] for _ in range(n_particles)]
    logZs = []
    print("Initialization time: ", time.time() - start_start_time)
    # implement smc using importance sampling

    # n_particles = 10

    # hairpin
    start_time = time.time()
    kss, realss = zip(*Parallel(n_jobs=8)(delayed(from_theta_to_rate_hairpin)(theta) for theta in tqdm(thetas)))
    ks = list(kss)
    reals = list(realss)
    logprobs = Parallel(n_jobs=8)(delayed(list_log_prob)(k, real, sigma) for k, real in zip(ks, reals))

    logZ, thetas = resample_particles(thetas, logprobs)
    logZs.append(logZ)
    print("hairpin:", time.time() - start_time)

    # hairpin1
    start_time = time.time()
    kss, realss = zip(*Parallel(n_jobs=8)(delayed(from_theta_to_rate_hairpin1)(theta) for theta in tqdm(thetas)))
    ks = list(kss)
    reals = list(realss)
    logprobs = Parallel(n_jobs=8)(delayed(list_log_prob)(k, real, sigma) for k, real in zip(ks, reals))

    logZ, thetas = resample_particles(thetas, logprobs)
    logZs.append(logZ)
    print("hairpin1:", time.time() - start_time)

    # hairpin4
    start_time = time.time()
    kss, realss = zip(*Parallel(n_jobs=8)(delayed(from_theta_to_rate_hairpin4)(theta) for theta in tqdm(thetas)))
    ks = list(kss)
    reals = list(realss)
    logprobs = Parallel(n_jobs=8)(delayed(list_log_prob)(k, real, sigma) for k, real in zip(ks, reals))

    logZ, thetas = resample_particles(thetas, logprobs)
    logZs.append(logZ)
    print("hairpin4:", time.time() - start_time)

    # helix
    start_time = time.time()
    kss, realss = zip(*Parallel(n_jobs=8)(delayed(from_theta_to_rate_helix)(theta) for theta in tqdm(thetas)))
    ks = list(kss)
    reals = list(realss)
    logprobs = Parallel(n_jobs=8)(delayed(list_log_prob)(k, real, sigma) for k, real in zip(ks, reals))

    logZ, thetas = resample_particles(thetas, logprobs)
    logZs.append(logZ)
    print("helix:", time.time() - start_time)

    # helix1
    start_time = time.time()
    kss, realss = zip(*Parallel(n_jobs=8)(delayed(from_theta_to_rate_helix1)(theta) for theta in tqdm(thetas)))
    ks = list(kss)
    reals = list(realss)
    logprobs = Parallel(n_jobs=8)(delayed(list_log_prob)(k, real, sigma) for k, real in zip(ks, reals))

    logZ, thetas = resample_particles(thetas, logprobs)
    logZs.append(logZ)
    print("helix1:", time.time() - start_time)

    # three_waystranddisplacement
    start_time = time.time()
    kss, realss = zip(*Parallel(n_jobs=8)(delayed(from_theta_to_rate_three_waystranddisplacement)(theta) for theta in tqdm(thetas)))
    ks = list(kss)
    reals = list(realss)
    logprobs = Parallel(n_jobs=8)(delayed(list_log_prob)(k, real, sigma) for k, real in zip(ks, reals))

    logZ, thetas = resample_particles(thetas, logprobs)
    logZs.append(logZ)
    print("three_waystranddisplacement:", time.time() - start_time)

    # three_waystranddisplacement
    start_time = time.time()
    kss, realss = zip(*Parallel(n_jobs=8)(delayed(from_theta_to_rate_three_waystranddisplacement1)(theta) for theta in tqdm(thetas)))
    ks = list(kss)
    reals = list(realss)
    logprobs = Parallel(n_jobs=8)(delayed(list_log_prob)(k, real, sigma) for k, real in zip(ks, reals))

    logZ, thetas = resample_particles(thetas, logprobs)
    logZs.append(logZ)
    print("three_waystranddisplacement1:", time.time() - start_time)

    # bubble
    start_time = time.time()
    kss, realss = zip(*Parallel(n_jobs=8)(delayed(from_theta_to_rate_bubble)(theta) for theta in tqdm(thetas)))
    ks = list(kss)
    reals = list(realss)
    logprobs = Parallel(n_jobs=8)(delayed(list_log_prob)(k, real, sigma) for k, real in zip(ks, reals))

    logZ, thetas = resample_particles(thetas, logprobs)
    logZs.append(logZ)
    print("bubble:", time.time() - start_time)

    # four_waystrandexchange
    start_time = time.time()
    kss, realss = zip(*Parallel(n_jobs=8)(delayed(from_theta_to_rate_four_waystrandexchange)(theta) for theta in tqdm(thetas)))
    ks = list(kss)
    reals = list(realss)
    logprobs = Parallel(n_jobs=8)(delayed(list_log_prob)(k, real, sigma) for k, real in zip(ks, reals))

    logZ, thetas = resample_particles(thetas, logprobs)
    logZs.append(logZ)
    print("four_waystrandexchange:", time.time() - start_time)

    thetas = np.array(thetas)
    logprobs = np.array(logprobs).reshape(-1,1)

    samples_mean = expectation_calculator(thetas, logprobs, lambda x:x)
    samples_var = expectation_calculator(thetas, logprobs, lambda x: x**2 - samples_mean**2)

    print('Number of particles: ', n_particles)
    print('Sampling time: ', time.time() - start_start_time)
    print('Mean: ', samples_mean)
    print('Variance: ', samples_var)

    MSE, within_3 = eval_theta_all(samples_mean)

    print('MSE: ', MSE)
    print('Within 3: ', within_3)

    torch.save(torch.tensor(MSE), 'results/MSE_'+str(n_particles)+'.pt')
    torch.save(torch.tensor(within_3), 'results/within_3_'+str(n_particles)+'.pt')

    weights = torch.tensor(np.exp(logprobs))
    torch.save(torch.tensor(thetas).T, 'results/thetas_'+str(n_particles)+'.pt')
    torch.save(weights.T[0], 'results/weights_'+str(n_particles)+'.pt')
    torch.save(torch.tensor(logZs), 'results/logZs_'+str(n_particles)+'.pt')
    # draw_hists("Importance_Sampling", torch.tensor(thetas).T, 1, weights=weights.T[0])