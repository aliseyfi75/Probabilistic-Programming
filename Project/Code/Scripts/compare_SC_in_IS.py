import sys
import torch
import pyro
sys.path.append('../')
sys.path.append('../Scripts')
import time

import matplotlib.pyplot as plt
from Scripts.new_sc_model import open_csv
from Scripts.new_sc_model import *


# datasets = { "bubble": ["Fig4"],
#              "four_waystrandexchange": ["Table5.2"],
#              "hairpin" : ["Fig4_0", "Fig4_1", "Fig6_0", "Fig6_1"], 
#              "hairpin1" : ["Fig3_T_0", "Fig3_T_1"],
#              "hairpin4" : ["Table1_0", "Table1_1"],
#              "helix" : ["Fig6_0", "Fig6_1"],
#              "helix1" : ["Fig6a"],
#              "three_waystranddisplacement" : ["Fig3b"], 
#              "three_waystranddisplacement1" : ["Fig6b"]
# }

datasets = { "hairpin" : ["Fig4_0"]}


def from_theta_to_rate(theta, datasets, kinetic_model="ARRHENIUS", stochastic_conditionning=False):
    
    # PATH = '/Users/aliseyfi/Documents/UBC/Probabilistic-Programming/Probabilistic-Programming/Project/'
    PATH = "C:/Users/jlovr/CS532-project/Probabilistic-Programming/Project/"
    predicted_log_10_rates, real_log_10_rates, errors = [], [], []
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
                        if row <=10 :
                            predicted_log_10_rate, real_log_10_rate, sq_error = estimate_Bonnet(row, theta, _zip, file, reaction_id, str(row), "Bonnet"+j, kinetic_model, stochastic_conditionning)
                            predicted_log_10_rates.append(predicted_log_10_rate)
                            real_log_10_rates.append(real_log_10_rate)
                            errors.append(sq_error)
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
                        row+=1
            else:
                pass

    return predicted_log_10_rates, real_log_10_rates, errors


def ESS(W):
    return 1/sum([wi**2 for wi in W])

def weighted_avg(X, weights):
    return (weights.dot(X)) / weights.sum() 

def run_IS_without_SC():
    theta_dim = 15
    theta_mean = [13.0580, 3, 13.0580, 3,  13.0580, 3, 13.0580, 3,  13.0580, 3, 13.0580, 3,  13.0580, 3,   0.0402 ]
    full_samples,full_logWs,full_ESS,full_means,full_vars = [],[],[],[],[]

    normal01 = torch.distributions.Normal(torch.tensor(0), torch.tensor(1))

    for n_samples in [3]:
        samples,logWs = [],[]
        for i in range(n_samples):
            theta = torch.distributions.MultivariateNormal(torch.tensor(theta_mean), torch.eye(theta_dim)).sample()

            preds, reals, errors = from_theta_to_rate(theta, datasets, stochastic_conditionning=False)
            loglik = 0

            for ind, k in enumerate(preds):
                error = errors[ind]
                
                # synthetic likelihood
                loglik += normal01.log_prob(torch.tensor(error))
            
            samples.append(theta)
            logWs.append(loglik)

        W = np.exp([logwi - max(logWs) for logwi in logWs])
        W = W/sum(W)
        ess = ESS(W)

        for n in range(n_samples):
            samples[n] = [float(x) for x in samples[n]]

        means = weighted_avg(samples, W)
        vars = weighted_avg((samples - means)**2, W)

        full_samples.append(samples)
        full_logWs.append(logWs)
        full_ESS.append(ess)
        full_means.append(means)
        full_vars.append(vars)

        print("mean", means)
        print("variance", vars)
        print("ess", ess)


def run_IS_with_SC():
    theta_dim = 15
    theta_mean = [13.0580, 3, 13.0580, 3,  13.0580, 3, 13.0580, 3,  13.0580, 3, 13.0580, 3,  13.0580, 3,   0.0402 ]
    full_samples,full_logWs,full_ESS,full_means,full_vars = [],[],[],[],[]


    for n_samples in [3]:
        samples,logWs = [],[]
        for i in range(n_samples):
            theta = torch.distributions.MultivariateNormal(torch.tensor(theta_mean), torch.eye(theta_dim)).sample()

            preds, reals, ks_stat = from_theta_to_rate(theta, datasets, stochastic_conditionning=True)
            loglik = 0

            for ind, k in enumerate(preds):                
                # synthetic likelihood
                loglik -= ks_stat[ind]
                # loglik += torch.distributions.Normal(torch.tensor(0), torch.tensor(1)).log_prob(torch.tensor(error))
            
            samples.append(theta)
            logWs.append(loglik)

        W = np.exp([logwi - max(logWs) for logwi in logWs])
        W = W/sum(W)
        ess = ESS(W)

        for n in range(n_samples):
            samples[n] = [float(x) for x in samples[n]]

        means = weighted_avg(samples, W)
        vars = weighted_avg((samples - means)**2, W)

        full_samples.append(samples)
        full_logWs.append(logWs)
        full_ESS.append(ess)
        full_means.append(means)
        full_vars.append(vars)

        print("mean", means)
        print("variance", vars)
        print("ess", ess)

def main():
    run_IS_without_SC()

    start = time.time()
    run_IS_with_SC()
    end = time.time()
    print("time:", end - start)





if __name__ == "__main__":
    main()
    
    