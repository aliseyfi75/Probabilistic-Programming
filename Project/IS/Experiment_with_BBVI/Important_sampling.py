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

N = parser.parse_args().N

def from_theta_to_rate(theta, datasets, kinetic_model="ARRHENIUS"):
    
    # PATH = '/home/aliseyfi/scratch/Probabilistic-Programming/Project/'
    PATH = '/Users/aliseyfi/Documents/UBC/Probabilistic-Programming/Probabilistic-Programming/Project/'
    # PATH = "C:/Users/jlovr/CS532-project/Probabilistic-Programming/Project/"
    predicted_log_10_rates, real_log_10_rates = [], []
    for reaction_type in datasets:
            if reaction_type == "bubble":
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
            if reaction_type == "four_waystrandexchange":
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
            if reaction_type == "hairpin":
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
            if reaction_type == "hairpin1":
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
            if reaction_type == "hairpin4":
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
            if reaction_type == "helix":
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
            if reaction_type == "helix1":
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
            if reaction_type == "three_waystranddisplacement":
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
            if reaction_type == "three_waystranddisplacement1":
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
            else:
                pass

    return predicted_log_10_rates, real_log_10_rates


def log_prob(k, real, sigma):
    return -np.log(sigma*np.sqrt(2*np.pi)) - 0.5*((k-real)/sigma)**2

def list_log_prob_par(ks, reals, sigma):
    return np.sum(Parallel(n_jobs=32)(delayed(log_prob)(k, real, sigma) for k, real in zip(ks, reals)))

def list_log_prob(ks, reals, sigma):
    return np.sum([log_prob(k, real, sigma) for k, real in zip(ks, reals)])

def expectation_calculator(results, log_weights, func, *args):
    weights = np.exp(log_weights)
    func_result = func(results, *args)
    return np.sum(weights*func_result, axis=0) / np.sum(weights)

def par_fun():
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
    prior = [13.0580, 3, 13.0580, 3,  13.0580, 3, 13.0580, 3,  13.0580, 3, 13.0580, 3,  13.0580, 3,   0.0402 ]
    theta = [np.random.normal(i, 1) for i in prior]
    ks, reals = from_theta_to_rate(theta, datasets)
    logprobs = [log_prob(k, real, sigma) for k, real in zip(ks, reals)]
    logprob = sum(logprobs)

    return theta, logprob


if __name__ == '__main__':
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

    mean = [12.543479119992815, 4.226898398415472, 13.337500173890277, 2.8595604197468645, 
             12.248657796490361, 3.0970839005977617, 12.908034847259342, 3.515686400591013,
             12.954033117162176, 3.233111020749515, 12.298319716642903, 2.177464964997234,
             13.542831816491484, 2.839804444835156, 0.13583210648998936]

    variance = [0.028087434304683592, 0.019482164640259283, 0.016245441771574755, 0.009112691618076695, 
                0.038730375626556866, 0.017621165478233853, 0.051216615884872294, 0.02878774592534887,
                0.031570835744080504, 0.04593872444285535, 0.01728817309129082, 0.04589162018443524,
                0.025638869311833987, 0.01855789302744983, 0.007727050475379378]

    sigma_generative = np.sqrt(variance)

    # prior = [13.0580, 3, 13.0580, 3,  13.0580, 3, 13.0580, 3,  13.0580, 3, 13.0580, 3,  13.0580, 3,   0.0402 ]
    sigma = 1

    start_time = time.time()
    thetas = [[np.random.normal(i, j) for i,j in zip(mean, sigma_generative)] for _ in range(N)]
    kss, realss = zip(*Parallel(n_jobs=32)(delayed(from_theta_to_rate)(theta, datasets) for theta in tqdm(thetas)))
    ks = list(kss)
    reals = list(realss)
    logprobs = Parallel(n_jobs=32)(delayed(list_log_prob)(k, real, sigma) for k, real in zip(ks, reals))
    # print("with parallelizing V.2: --- %s seconds ---" % (time.time() - start_time))

    thetas = np.array(thetas)
    logprobs = np.array(logprobs).reshape(-1,1)

    samples_mean = expectation_calculator(thetas, logprobs, lambda x:x)
    samples_var = expectation_calculator(thetas, logprobs, lambda x: x**2 - samples_mean**2)

    print('Number of samples: ', N)
    print('Sampling time: ', time.time() - start_time)
    print('Mean: ', samples_mean)
    print('Variance: ', samples_var)

    weights = torch.tensor(np.exp(logprobs))
    torch.save(torch.tensor(thetas).T, 'results/thetas_'+str(N)+'.pt')
    torch.save(weights.T[0], 'results/weights_'+str(N)+'.pt')

    MSE, within_3 = eval_theta_all(samples_mean)

    print('MSE: ', MSE)
    print('Within 3: ', within_3)

    torch.save(torch.tensor(MSE), 'results/MSE_'+str(N)+'.pt')
    torch.save(torch.tensor(within_3), 'results/within_3_'+str(N)+'.pt')
    