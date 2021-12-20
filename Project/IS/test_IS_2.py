import sys
import numpy as np
from torch import distributions
from torch.distributions import distribution
import scipy.stats as stats
import time
from plot import draw_hists
import torch
import argparse
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
# import wandb

# wandb.init(project="Probprog_project", entity="aliseyfi")

from joblib import Parallel, delayed

matplotlib.use('Agg')
sys.path.append('../')
sys.path.append('../Code/Scripts')
sys.path.append('../Code/')

from Scripts.model import open_csv
from Scripts.model import *

from test_theta_on_hairpins import eval_theta as eval_theta_hairpins
from test_theta_on_all import eval_theta_all

# parser = argparse.ArgumentParser()
# parser.add_argument('--alpha', type=float, default=1)

# alpha = parser.parse_args().alpha


def from_theta_to_rate(theta, datasets, kinetic_model="ARRHENIUS", stochastic_conditionning=False):
    
    # PATH = '/Users/aliseyfi/Documents/UBC/Probabilistic-Programming/Probabilistic-Programming/Project/'
    PATH = "C:/Users/jlovr/CS532-project/Probabilistic-Programming/Project/"
    predicted_log_10_rates, real_log_10_rates, errors, used_KS_error = [], [], [], []
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
                        errors.append(sq_error)
                        used_KS_error.append(False)
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
                        errors.append(sq_error)
                        used_KS_error.append(False)
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
                        predicted_log_10_rate, real_log_10_rate, sq_error = estimate_Bonnet(row, theta, _zip, file, reaction_id, str(row), "Bonnet"+j, kinetic_model, stochastic_conditionning)
                        predicted_log_10_rates.append(predicted_log_10_rate)
                        real_log_10_rates.append(real_log_10_rate)
                        errors.append(sq_error)
                        used_KS_error.append(stochastic_conditionning)
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
                        errors.append(sq_error)
                        used_KS_error.append(False)
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
                        errors.append(sq_error)
                        used_KS_error.append(False)
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
                        errors.append(sq_error)
                        used_KS_error.append(False)
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
                        errors.append(sq_error)
                        used_KS_error.append(False)
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
                        errors.append(sq_error)
                        used_KS_error.append(False)
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
                        errors.append(sq_error)
                        used_KS_error.append(False)
                        row+=1
            else:
                pass

    return predicted_log_10_rates, real_log_10_rates, errors, used_KS_error

def log_prob(pred, real, error, used_ks, alpha, squaredKS=False):
    # returns log_prob for a single point
    # return -np.log(sigma*np.sqrt(2*np.pi)) - 0.5*((k-real)/sigma)**2
    # return - 0.5*(ks_stat)**2
    if used_ks:
        ks_stat = error
        if squaredKS:
            # print(-alpha * ks_stat**2)
            return -alpha * ks_stat**2
        else:
            # print("Here")
            # print("alpha")
            # print(-alpha * ks_stat)
            print(-alpha * ks_stat)
            return -alpha * ks_stat
    else:
        squared_error = error
        x =-0.5*np.log(2*np.pi) - 0.5*squared_error
        print(x)
        return x

# def list_log_prob_par(ks, reals, sigma):
#     return np.sum(Parallel(n_jobs=16)(delayed(log_prob)(k, real, sigma) for k, real in zip(ks, reals)))

def list_log_prob(pred_rates, real_rates, errors, used_ks_errors, alpha, squaredKS=False):
    # gets log_prob for one theta sample
    return np.sum([log_prob(pred, real, error, used_ks, alpha, squaredKS) for pred,real,error,used_ks in zip(pred_rates,real_rates,errors,used_ks_errors)])

def expectation_calculator(results, log_weights, func, *args):
    weights = np.exp(log_weights)
    func_result = func(results, *args)
    return np.sum(weights*func_result, axis=0) / np.sum(weights)

# def par_fun():
#     datasets = {    "hairpin" : ["Fig4_0", "Fig4_1", "Fig6_0", "Fig6_1"],
#                 "hairpin1" : ["Fig3_T_0", "Fig3_T_1"],
#                 "hairpin4" : ["Table1_0", "Table1_1"],
#                 "helix" : ["Fig6_0", "Fig6_1"],
#                 "helix1" : ["Fig6a"],
#                 "three_waystranddisplacement" : ["Fig3b"],
#                 "three_waystranddisplacement1" : ["Fig6b"],
#                 "bubble": ["Fig4"],
#                 "four_waystrandexchange": ["Table5.2"]
#     }
#     prior = [13.0580, 3, 13.0580, 3,  13.0580, 3, 13.0580, 3,  13.0580, 3, 13.0580, 3,  13.0580, 3,   0.0402 ]
#     theta = [np.random.normal(i, 1) for i in prior]
#     ks, reals = from_theta_to_rate(theta, datasets)
#     logprobs = [log_prob(k, real, sigma) for k, real in zip(ks, reals)]
#     logprob = sum(logprobs)

#     return theta, logprob


if __name__ == '__main__':
    # datasets = {    "hairpin" : ["Fig4_0", "Fig4_1", "Fig6_0", "Fig6_1"],
    #             "hairpin1" : ["Fig3_T_0", "Fig3_T_1"],
    #             "hairpin4" : ["Table1_0", "Table1_1"],
    #             "helix" : ["Fig6_0", "Fig6_1"],
    #             "helix1" : ["Fig6a"],
    #             "three_waystranddisplacement" : ["Fig3b"],
    #             "three_waystranddisplacement1" : ["Fig6b"],
    #             "bubble": ["Fig4"],
    #             "four_waystrandexchange": ["Table5.2"]
    # }

    datasets = {    "hairpin" : ["Fig4_0"]   }

    prior = [13.0580, 3, 13.0580, 3,  13.0580, 3, 13.0580, 3,  13.0580, 3, 13.0580, 3,  13.0580, 3,   0.0402 ]
    sigma = 1
    # wandb.log({'alpha':alpha})
    # implement importance sampling

    # N = 10
    
    # for alpha=[0.5,1,2,3,4,5]:
    # for N in [1,2,4,8,16,32,64,128,256,512]:
    alpha = 1
    squaredKS = False
    N_set =  [1]

    for SC in [True]:
        mses = []
        within3s = []
        # mses_hairpins = []
        # within3s_hairpins = []
        for N in N_set:
            start_time = time.time()
            thetas = [[np.random.normal(i, 1) for i in prior] for _ in range(N)]

            predicted_log_10_rates, real_log_10_rates, errors, used_KS_error = zip(*Parallel(n_jobs=16)(delayed(from_theta_to_rate)(theta, datasets, stochastic_conditionning=SC) for theta in tqdm(thetas)))
            ks = list(predicted_log_10_rates)
            reals = list(real_log_10_rates)
            # logprobs = Parallel(n_jobs=16)(delayed(list_log_prob)(error) for error in errors)
            logprobs = Parallel(n_jobs=16)(delayed(list_log_prob)(pred,real,error,used_KS, alpha, squaredKS) for pred,real,error,used_KS in zip(predicted_log_10_rates, real_log_10_rates, errors, used_KS_error))

            print("with parallelizing V.2: --- %s seconds ---" % (time.time() - start_time))

            print("n_dataset", len(predicted_log_10_rates[0]))
            thetas = np.array(thetas)
            logprobs = np.array(logprobs).reshape(-1,1)
            print("n_samples x dim-theta", thetas.shape)
            print("n_samples x 1", logprobs.shape)

            samples_mean = expectation_calculator(thetas, logprobs, lambda x:x)
            samples_var = expectation_calculator(thetas, logprobs, lambda x: x**2 - samples_mean**2)

            print('Number of samples: ', N)
            print("SC: ", SC)
            print('Mean: ', samples_mean)
            print('Variance: ', samples_var)

            # mse, within3 = eval_theta_all(samples_mean)
            mse, within3 = eval_theta_hairpins(samples_mean)

            # mse_hairpins, within3_hairpins = eval_theta_hairpins(samples_mean)
            mses.append(mse)
            within3s.append(within3)
            # mses_hairpins.append(mse_hairpins)
            # within3s_hairpins.append(within3_hairpins)

            # wandb.log({'MSE':mse})
            # wandb.log({'Within 3':within3})

            weights = torch.tensor(np.exp(logprobs))

            draw_hists("Importance_Sampling", torch.tensor(thetas).T, 1, weights=weights.T[0], alpha=N)

        plt.figure(1)
        if SC:
            lab = "FPTD observation model"
        else:
            lab = "MFPT observation model"
        plt.plot(N_set, mses, label=lab)
        # if squaredKS:
        #     lab = "only hairpins, FPTD observation model"
        # else:
        #     lab = "only hairpins, MFPT observation model"
        # plt.plot(N_set, mses_hairpins, label=lab)

        plt.figure(2)
        if squaredKS:
            lab = "FPTD observation model"
        else:
            lab = "MFPT observation model"
        plt.plot(N_set, within3s, label=lab)
        # if squaredKS:
        #     lab = "only hairpins, FPTD observation model"
        # else: 
        #     lab = "only hairpins, MFPT observation model"
        # plt.plot(N_set, within3s_hairpins, label=lab)

    plt.figure(1)
    plt.xlabel("N")
    plt.ylabel("MSE")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    figstr = "stochastic_conditionning_results/MSE"
    plt.savefig(figstr, bbox_inches='tight')

    plt.figure(2)
    plt.xlabel("N")
    plt.ylabel("proportion within 3")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    figstr = "stochastic_conditionning_results/within3"
    plt.savefig(figstr, bbox_inches='tight')

    # plt.show()



# pick experiment, t x axis y axis arrival times (cdf or density)       Histogram
# analytical for true measured rate

