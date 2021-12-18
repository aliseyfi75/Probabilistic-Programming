import sys
import numpy as np
from torch import distributions
from torch.distributions import distribution
import scipy.stats as stats
import time
from plot import draw_hists
import torch
from tqdm import tqdm

from joblib import Parallel, delayed

sys.path.append('../')
sys.path.append('../Code/Scripts')
sys.path.append('../Code/')

from Scripts.model import open_csv
from Scripts.model import *


def from_theta_to_rate(theta, datasets, kinetic_model="ARRHENIUS"):
    
    PATH = '/home/aliseyfi/scratch/Probabilistic-Programming/Project/'
    # PATH = '/Users/aliseyfi/Documents/UBC/Probabilistic-Programming/Probabilistic-Programming/Project/'
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
    return np.sum(Parallel(n_jobs=16)(delayed(log_prob)(k, real, sigma) for k, real in zip(ks, reals)))

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

    prior = [13.0580, 3, 13.0580, 3,  13.0580, 3, 13.0580, 3,  13.0580, 3, 13.0580, 3,  13.0580, 3,   0.0402 ]
    sigma = 1
    # implement importance sampling

    N = 10000
    
    # start_time = time.time()
    # theta, logprob = zip(*Parallel(n_jobs=4)(delayed(par_fun)() for i in range(N)))
    # print("with parallelizing V.1: --- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    thetas = [[np.random.normal(i, 1) for i in prior] for _ in range(N)]
    kss, realss = zip(*Parallel(n_jobs=16)(delayed(from_theta_to_rate)(theta, datasets) for theta in tqdm(thetas)))
    ks = list(kss)
    reals = list(realss)
    logprobs = Parallel(n_jobs=16)(delayed(list_log_prob)(k, real, sigma) for k, real in zip(ks, reals))
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
    draw_hists("Importance_Sampling", torch.tensor(thetas).T, 1, weights=weights.T[0])

    # start_time = time.time()
    # thetas = [[np.random.normal(i, 1) for i in prior] for _ in range(N)]
    # kss, realss = zip(*Parallel(n_jobs=8)(delayed(from_theta_to_rate)(theta, datasets) for theta in thetas))
    # ks = list(kss)
    # reals = list(realss)
    # logprobs = [list_log_prob_par(k, real, sigma) for k, real in zip(ks, reals)]
    # print("with parallelizing V.3: --- %s seconds ---" % (time.time() - start_time))

    # start_time = time.time()
    # thetas = [[np.random.normal(i, 1) for i in prior] for _ in range(N)]
    # kss, realss = zip(*Parallel(n_jobs=8)(delayed(from_theta_to_rate)(theta, datasets) for theta in thetas))
    # ks = list(kss)
    # reals = list(realss)
    # logprobs = [list_log_prob(k, real, sigma) for k, real in zip(ks, reals)]
    # print("with parallelizing V.4: --- %s seconds ---" % (time.time() - start_time))

    # start_time = time.time()
    # for n in range(N):
    #     theta = [np.random.normal(i, 1) for i in prior]
    #     ks, reals = from_theta_to_rate(theta, datasets)
    #     logprobs = [log_prob(k, real, sigma) for k, real in zip(ks, reals)]
    #     logprob = sum(logprobs)
    # print("without parallelizing : --- %s seconds ---" % (time.time() - start_time))

    