import sys
import torch
import pyro
sys.path.append('../')
sys.path.append('../Scripts')
import time

import matplotlib.pyplot as plt
import seaborn as sns
from Scripts.new_sc_model import open_csv
from Scripts.new_sc_model import *
from test_theta_on_hairpins import eval_theta

datasets = { "bubble": ["Fig4"],
             "four_waystrandexchange": ["Table5.2"],
             "hairpin" : ["Fig4_0", "Fig4_1", "Fig6_0", "Fig6_1"], 
             "hairpin1" : ["Fig3_T_0", "Fig3_T_1"],
             "hairpin4" : ["Table1_0", "Table1_1"],
             "helix" : ["Fig6_0", "Fig6_1"],
             "helix1" : ["Fig6a"],
             "three_waystranddisplacement" : ["Fig3b"], 
             "three_waystranddisplacement1" : ["Fig6b"]
}

datasets = { "hairpin" : ["Fig4_0"]}

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

def ESS(W):
    return 1/sum([wi**2 for wi in W])

def weighted_avg(X, weights):
    return (weights.dot(X)) / weights.sum() 

def run_IS(n_samples, stochastic_conditionning, squared_KS=False):
    theta_dim = 15
    theta_mean = [13.0580, 3, 13.0580, 3,  13.0580, 3, 13.0580, 3,  13.0580, 3, 13.0580, 3,  13.0580, 3,   0.0402 ]
    normal01 = torch.distributions.Normal(torch.tensor(0), torch.tensor(1))

    samples,logWs = [],[]
    for i in range(n_samples):
        theta = torch.distributions.MultivariateNormal(torch.tensor(theta_mean), torch.eye(theta_dim)).sample()

        preds, reals, synthetic_errors, used_KS_error = from_theta_to_rate(theta, datasets, stochastic_conditionning=stochastic_conditionning)

        loglik = 0
        for n in range(len(synthetic_errors)):

            if used_KS_error[n]==True:

                if squared_KS==False:
                    # synthetic likelihood
                    ks_stat = synthetic_errors[n]
                    loglik -= ks_stat
                else:
                    ks_stat = synthetic_errors[n]
                    loglik -= ks_stat**2

            else:
                assert(used_KS_error[n]==False)

                # synthetic likelihood
                squared_error = synthetic_errors[n]
                loglik += normal01.log_prob(torch.tensor(squared_error))
        
        samples.append(theta)
        logWs.append(loglik)

    return samples, logWs

def interpret_results(samples, logWs):
    print("\n\n\n")
    n_samples = len(samples)
    W = np.exp([logwi - max(logWs) for logwi in logWs])
    W = W/sum(W)
    ess = ESS(W)

    for n in range(n_samples):
        samples[n] = [float(x) for x in samples[n]]

    means = weighted_avg(samples, W)
    vars = weighted_avg((samples - means)**2, W)

    print("mean", means)
    print("variance", vars)
    print("ess", ess)

    eval_theta(means)

    return samples, W

def main():

    n_samples = 5

    start = time.time()
    samples, logWs = run_IS(n_samples, stochastic_conditionning=False)
    end = time.time()
    samples, W = interpret_results(samples, logWs)

    variables = np.array(samples,dtype=object).T.tolist()
    for d in range(len(variables)):
        plt.figure()
        sns.histplot(x = variables[d], weights=W, kde=True, bins=50)
        plt.ylabel("density")
        plt.savefig("IS_plots/theta"+str(d)+"_n_"+str(n_samples)+"_MFPT")
        plt.close('all')
    print("Without path samples time:", end - start)

    start = time.time()
    samples, logWs = run_IS(n_samples, stochastic_conditionning=True)
    end = time.time()
    samples, W = interpret_results(samples, logWs)
    variables = np.array(samples,dtype=object).T.tolist()
    for d in range(len(variables)):
        plt.figure()
        sns.histplot(x = variables[d], weights=W, kde=True, bins=50)
        plt.ylabel("density")
        plt.savefig("IS_plots/theta"+str(d)+"_n_"+str(n_samples)+"_FPTD")
        plt.close('all')
    print("With path samples time:", end - start)

    start = time.time()
    samples, logWs = run_IS(n_samples, stochastic_conditionning=True, squared_KS=True)
    end = time.time()
    samples, W = interpret_results(samples, logWs)
    variables = np.array(samples,dtype=object).T.tolist()
    for d in range(len(variables)):
        plt.figure()
        sns.histplot(x = variables[d], weights=W, kde=True, bins=50)
        plt.ylabel("density")
        plt.savefig("IS_plots/theta"+str(d)+"_n_"+str(n_samples)+"_FPTD_squaredKS")
        plt.close('all')
    print("With path samples time:", end - start)

if __name__ == "__main__":
    main()