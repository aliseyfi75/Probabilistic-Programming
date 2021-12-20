import sys
import torch
# import pyro
sys.path.append('../')
sys.path.append('../Scripts')
import time

# import matplotlib.pyplot as plt
# import seaborn as sns
from Scripts.new_sc_model import open_csv
from Scripts.new_sc_model import *
from test_theta_on_hairpins import eval_theta as eval_theta_hairpins
from test_theta_on_all import eval_theta_all
# import matplotlib
import json
import argparse

# matplotlib.use('Agg')

from joblib import Parallel, delayed

parser = argparse.ArgumentParser()
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--n_samples', type=int, default=5)
parser.add_argument('--fulldata', type=bool, default=True)
parser.add_argument('--squaredLik', type=bool, default=False)
parser.add_argument('--BBVI_prior', type=bool, default=False)
parser.add_argument('--distance', type=str, default="KS_test")

alpha = parser.parse_args().alpha
n_samples = parser.parse_args().n_samples
fulldata = parser.parse_args().fulldata
squaredLik = parser.parse_args().squaredLik
BBVI_prior = parser.parse_args().BBVI_prior
distance = parser.parse_args().distance

if fulldata:
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
else:
    datasets = { "hairpin" : ["Fig4_0"]}

def from_theta_to_rate(theta, datasets, kinetic_model="ARRHENIUS", stochastic_conditionning=False):
    
    # PATH = '/Users/aliseyfi/Documents/UBC/Probabilistic-Programming/Probabilistic-Programming/Project/'
    PATH = "C:/Users/jlovr/CS532-project/Probabilistic-Programming/Project/"
    # PATH = "/home/jlovrod/projects/def-condon/jlovrod/Probabilistic-Programming/Project/"
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
                        predicted_log_10_rate, real_log_10_rate, sq_error = estimate_Bonnet(row, theta, _zip, file, reaction_id, str(row), "Bonnet"+j, kinetic_model, stochastic_conditionning, distance)
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

def log_prob(pred, real, error, used_ks):
    if used_ks:
        # ks_stat = error
        if squaredLik:
            return -alpha * error**2
        else:
            return -alpha * error
    else:
        # squared_error = error
        x =-0.5*np.log(2*np.pi) - 0.5*error
        return x

def list_log_prob(pred_rates, real_rates, errors, used_ks_errors):
    return np.sum([log_prob(pred, real, error, used_ks) for pred,real,error,used_ks in zip(pred_rates,real_rates,errors,used_ks_errors)])

def run_IS(n_samples, stochastic_conditionning):
    theta_dim = 15

    if BBVI_prior==False:
        theta_mean = [13.0580, 3, 13.0580, 3,  13.0580, 3, 13.0580, 3,  13.0580, 3, 13.0580, 3,  13.0580, 3,   0.0402 ]
        prior = torch.distributions.MultivariateNormal(torch.tensor(theta_mean), torch.eye(theta_dim))
    else:
        theta_mean = [12.543479119992815,4.226898398415472,13.337500173890277,2.8595604197468645,12.248657796490361,3.0970839005977617,12.908034847259342,3.515686400591013, \
            12.954033117162176,3.233111020749515,12.298319716642903,2.177464964997234,13.542831816491484,2.839804444835156,0.13583210648998936]
        theta_var = [0.028087434304683592,0.019482164640259283,0.016245441771574755,0.009112691618076695,0.038730375626556866,0.017621165478233853,0.051216615884872294,0.02878774592534887, \
            0.031570835744080504,0.04593872444285535,0.01728817309129082,0.04589162018443524,0.025638869311833987,0.01855789302744983,0.007727050475379378]
        prior = torch.distributions.MultivariateNormal(torch.tensor(theta_mean), torch.diag(torch.tensor(theta_var)))

    thetas = [prior.sample() for i in range(n_samples)]

    if stochastic_conditionning==False:
        preds, reals, synthetic_errors, used_KS_error = zip(*Parallel(n_jobs=16)(delayed(from_theta_to_rate)(theta, datasets, stochastic_conditionning=stochastic_conditionning) for theta in thetas))
    else:
        preds, reals, synthetic_errors, used_KS_error = zip(*Parallel(n_jobs=16,require='sharedmem')(delayed(from_theta_to_rate)(theta, datasets, stochastic_conditionning=stochastic_conditionning) for theta in thetas))

        # preds, reals, synthetic_errors, used_KS_error = [],[],[],[]
        # for theta in thetas:
        #     pred, real, synthetic_error, used_KS = from_theta_to_rate(theta, datasets, stochastic_conditionning=stochastic_conditionning)
        #     preds.append(pred)
        #     reals.append(real)
        #     synthetic_errors.append(synthetic_error)
        #     used_KS_error.append(used_KS)      

    logprobs = Parallel(n_jobs=16)(delayed(list_log_prob)(pred,real,error,used_KS) for pred,real,error,used_KS in zip(preds, reals, synthetic_errors, used_KS_error))

    return thetas, logprobs

# def run_IS(n_samples, stochastic_conditionning, squared_KS=False):
#     theta_dim = 15
#     theta_mean = [13.0580, 3, 13.0580, 3,  13.0580, 3, 13.0580, 3,  13.0580, 3, 13.0580, 3,  13.0580, 3,   0.0402 ]
#     normal01 = torch.distributions.Normal(torch.tensor(0), torch.tensor(1))

#     samples,logWs = [],[]
#     for i in range(n_samples):
#         theta = torch.distributions.MultivariateNormal(torch.tensor(theta_mean), torch.eye(theta_dim)).sample()

#         preds, reals, synthetic_errors, used_KS_error = from_theta_to_rate(theta, datasets, stochastic_conditionning=stochastic_conditionning)

#         loglik = 0
#         for n in range(len(synthetic_errors)):

#             if used_KS_error[n]==True:

#                 if squared_KS==False:
#                     # synthetic likelihood
#                     ks_stat = synthetic_errors[n]
#                     loglik -= alpha*ks_stat
#                 else:
#                     ks_stat = synthetic_errors[n]
#                     loglik -= alpha*ks_stat**2

#             else:

#                 # ABC synthetic likelihood

#                 # The following are equivalent
#                 # normalp1 = torch.distributions.Normal(torch.tensor(preds[n]), torch.tensor(1))
#                 # z = normal01.log_prob(torch.tensor(preds[n]-reals[n]))
#                 # w = normalp1.log_prob(torch.tensor(reals[n]))
#                 loglik += normal01.log_prob(torch.tensor(preds[n]-reals[n]))
        
#         samples.append(theta)
#         logWs.append(loglik)

#         # if i%100==1:
#         #     # interpret_results(samples,logWs)

#         #     if stochastic_conditionning==True:
#         #         st = "FPTD"
#         #     else:
#         #         st = "MFPT"
#         #     if squared_KS==True:
#         #         sb = "squaredLik"
#         #     else:
#         #         sb = ""

#         #     filestr = "saved_IS_particles/samples_" + str(i+1) + st + sb + "alpha"+str(alpha)+"_hairpins"
#         #     with open(filestr, 'w') as f:
#         #         json.dump([sample.tolist() for sample in samples], f)
#         #     filestr = "saved_IS_particles/weights_" + str(i+1) + st + sb + "alpha"+str(alpha)+"_hairpins"
#         #     with open(filestr, 'w') as f:
#         #         json.dump([logW.item() for logW in logWs], f)
            
#     return samples, logWs


def interpret_results(samples, logWs):

    print("\n\n\n")
    # print(logWs)
    W = np.exp([logwi - max(logWs) for logwi in logWs])
    W = W/sum(W)
    # ess = ESS(W)

    for n in range(n_samples):
        samples[n] = [float(x) for x in samples[n]]

    means = weighted_avg(samples, W)
    vars = weighted_avg((samples - means)**2, W)

    print("mean", means)
    print("variance", vars)
    # print("ess", ess)

    if fulldata:
        mse, within3 = eval_theta_all(means)
    else:
        mse, within3 = eval_theta_hairpins(means)


    if squaredLik==True:
        a = "squaredLik"
    else:
        a = ""

    if fulldata:
        b = "fulldata"
    else:
        b = "hairpins"
    
    if BBVI_prior:
        c = "BBVI"
    else:
        c = "prior"
            
    filestr = "saved_IS_particles/samples_" + str(n_samples) + "alpha"+str(alpha)+a+b+c+distance
    with open(filestr, 'w') as f:
        json.dump([sample for sample in samples], f)
    filestr = "saved_IS_particles/weights_" + str(n_samples) + "alpha"+str(alpha)+a+b+c+distance
    with open(filestr, 'w') as f:
        json.dump([logW for logW in logWs], f)
    filestr = "saved_IS_particles/results_" + str(n_samples) + "alpha"+str(alpha)+a+b+c+distance
    text = "theta mean" + str(means) + "\n" + "MSE " + str(mse) + "\n" + "within3 " + str(within3) + "\n" 
    with open(filestr, 'w') as f:
        f.write(text)
    return samples, W

def main():

    # start = time.time()
    # samples, logWs = run_IS(n_samples, stochastic_conditionning=False)
    # end = time.time()
    # print("time", start-end)
    # samples, W = interpret_results(samples, logWs)

    # variables = np.array(samples,dtype=object).T.tolist()
    # for d in range(len(variables)):
    #     plt.figure()
    #     sns.histplot(x = variables[d], weights=W, kde=True, bins=50)
    #     plt.ylabel("density")
    #     plt.savefig("IS_plots/theta"+str(d)+"_n_"+str(n_samples)+"_MFPT")
    #     plt.close('all')
    # print("Without path samples time:", end - start)

    start = time.time()
    samples, logWs = run_IS(n_samples, stochastic_conditionning=True)
    # samples, logWs = run_IS(n_samples, stochastic_conditionning=True, squared_KS=True)
    end = time.time()
    print("done in", end-start, "s")

    samples, W = interpret_results(samples, logWs)

    # variables = np.array(samples,dtype=object).T.tolist()
    # for d in range(len(variables)):
    #     plt.figure()
    #     try:
    #         sns.histplot(x = variables[d], weights=W, kde=True, bins=50)
    #     except:
    #         sns.histplot(x = variables[d], weights=W, bins=10)
    #     plt.ylabel("density")
    #     plt.savefig("IS_plots/theta"+str(d)+"_n_"+str(n_samples)+"_FPTD")
    #     plt.close('all')
    # print("With path samples time:", end - start)

if __name__ == "__main__":
    main()