import sys
sys.path.append('../Code')
sys.path.append('../Code/Scripts/')
from joblib import Parallel, delayed
import functools
import operator

from Scripts.model import open_csv
from Scripts.model import *

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

def parallel_helper(theta, reaction_type, kinetic_model):
    # PATH = '/home/aliseyfi/scratch/Probabilistic-Programming/Project/'
    PATH = '/Users/aliseyfi/Documents/UBC/Probabilistic-Programming/Probabilistic-Programming/Project/'
    # PATH = "C:/Users/jlovr/CS532-project/Probabilistic-Programming/Project/"
    # PATH = "/home/jlovrod/projects/def-condon/jlovrod/Probabilistic-Programming/Project/"

    predicted_log_10_rates, real_log_10_rates = [], []

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

    return predicted_log_10_rates

def from_theta_to_rate(theta, datasets=datasets, kinetic_model="ARRHENIUS"):
    predicted_log_10_rates = Parallel(n_jobs=8)(delayed(parallel_helper)(theta, reaction_type, kinetic_model) for reaction_type in datasets)
    return functools.reduce(operator.iconcat, predicted_log_10_rates, [])
    

# def from_theta_to_rate(theta, datasets=datasets, kinetic_model="ARRHENIUS"):
    
#     PATH = '/Users/aliseyfi/Documents/UBC/Probabilistic-Programming/Probabilistic-Programming/Project/'
#     # PATH = "C:/Users/jlovr/CS532-project/Probabilistic-Programming/Project/"
#     predicted_log_10_rates, real_log_10_rates = [], []
#     for reaction_type in datasets:
#             if reaction_type == "bubble":
#                 for reaction_dataset in datasets[reaction_type]:
#                     reaction_id = "/" + reaction_type + "/" + reaction_dataset
#                     document_name = PATH + "/dataset" + reaction_id + ".csv"
#                     file =  open_csv(document_name)
#                     row = 1
#                     while row < len(file) and file[row][0] != '' :
#                         predicted_log_10_rate, real_log_10_rate, sq_error = estimate_AltanBonnet(row, theta, file, reaction_id, str(row), "Altanbonnet", kinetic_model)
#                         predicted_log_10_rates.append(predicted_log_10_rate)
#                         real_log_10_rates.append(real_log_10_rate)
#                         row+=1
#             if reaction_type == "four_waystrandexchange":
#                 for reaction_dataset in datasets[reaction_type]:
#                     reaction_id = "/" + reaction_type + "/" + reaction_dataset
#                     document_name = PATH + "/dataset" + reaction_id + ".csv"
#                     file =  open_csv(document_name)
#                     row = 1
#                     while row < len(file) and file[row][0] != '' :
#                         predicted_log_10_rate, real_log_10_rate, sq_error = estimate_DabbyThesis(row, theta, file, reaction_id, str(row), "Dabby", kinetic_model)
#                         predicted_log_10_rates.append(predicted_log_10_rate)
#                         real_log_10_rates.append(real_log_10_rate)
#                         row+=1
#             if reaction_type == "hairpin":
#                 for reaction_dataset in datasets[reaction_type]:
#                     _zip = bool(int(reaction_dataset[-1]))
#                     j = reaction_dataset[-3]
#                     reaction_id = "/" + reaction_type + "/" + reaction_dataset
#                     document_name = PATH + "/dataset" + reaction_id + ".csv"
#                     file =  open_csv(document_name)
#                     row = 1
#                     while row < len(file) and file[row][0] != '' :
#                         predicted_log_10_rate, real_log_10_rate, sq_error = estimate_Bonnet(row, theta, _zip, file, reaction_id, str(row), "Bonnet"+j, kinetic_model)
#                         predicted_log_10_rates.append(predicted_log_10_rate)
#                         real_log_10_rates.append(real_log_10_rate)
#                         row+=1
#             if reaction_type == "hairpin1":
#                 for reaction_dataset in datasets[reaction_type]:
#                     _zip = bool(int(reaction_dataset[-1]))
#                     reaction_id = "/" + reaction_type + "/" + reaction_dataset
#                     document_name = PATH + "/dataset" + reaction_id + ".csv"
#                     file =  open_csv(document_name)
#                     row = 1
#                     while row < len(file) and file[row][0] != '' :
#                         predicted_log_10_rate, real_log_10_rate, sq_error = estimate_BonnetThesis(row, theta, _zip, file, reaction_id, str(row), "GoddardT", kinetic_model)
#                         predicted_log_10_rates.append(predicted_log_10_rate)
#                         real_log_10_rates.append(real_log_10_rate)
#                         row+=1
#             if reaction_type == "hairpin4":
#                 for reaction_dataset in datasets[reaction_type]:
#                     _zip = bool(int(reaction_dataset[-1]))
#                     reaction_id = "/" + reaction_type + "/" + reaction_dataset
#                     document_name = PATH + "/dataset" + reaction_id + ".csv"
#                     file =  open_csv(document_name)
#                     row = 1
#                     while row < len(file) and file[row][0] != '' :
#                         predicted_log_10_rate, real_log_10_rate, sq_error = estimate_Kim(row, theta, _zip, file, reaction_id, str(row), "Kim", kinetic_model)
#                         predicted_log_10_rates.append(predicted_log_10_rate)
#                         real_log_10_rates.append(real_log_10_rate)
#                         row+=1
#             if reaction_type == "helix":
#                 for reaction_dataset in datasets[reaction_type]:
#                     _zip = bool(int(reaction_dataset[-1]))
#                     reaction_id = "/" + reaction_type + "/" + reaction_dataset
#                     document_name = PATH + "/dataset" + reaction_id + ".csv"
#                     file =  open_csv(document_name)
#                     row = 1
#                     while row < len(file) and file[row][0] != '' :
#                         predicted_log_10_rate, real_log_10_rate, sq_error = estimate_Morrison(row, theta, _zip, file, reaction_id, str(row), "Morrison", kinetic_model)
#                         predicted_log_10_rates.append(predicted_log_10_rate)
#                         real_log_10_rates.append(real_log_10_rate)
#                         row+=1
#             if reaction_type == "helix1":
#                 for reaction_dataset in datasets[reaction_type]:
#                     _zip = False
#                     reaction_id = "/" + reaction_type + "/" + reaction_dataset
#                     document_name = PATH + "/dataset" + reaction_id + ".csv"
#                     file =  open_csv(document_name)
#                     row = 1
#                     while row < len(file) and file[row][0] != '' :
#                         predicted_log_10_rate, real_log_10_rate, sq_error = estimate_ReynaldoDissociate(row, theta, _zip, file, reaction_id, str(row), "ReynaldoDissociate", kinetic_model)
#                         predicted_log_10_rates.append(predicted_log_10_rate)
#                         real_log_10_rates.append(real_log_10_rate)
#                         row+=1
#             if reaction_type == "three_waystranddisplacement":
#                 for reaction_dataset in datasets[reaction_type]:
#                     reaction_id = "/" + reaction_type + "/" + reaction_dataset
#                     document_name = PATH + "/dataset" + reaction_id + ".csv"
#                     file =  open_csv(document_name)
#                     row = 1
#                     while row < len(file) and file[row][0] != '' :
#                         predicted_log_10_rate, real_log_10_rate, sq_error = estimate_Zhang(row, theta, file, reaction_id, str(row), "Zhang", kinetic_model)
#                         predicted_log_10_rates.append(predicted_log_10_rate)
#                         real_log_10_rates.append(real_log_10_rate)
#                         row+=1
#             if reaction_type == "three_waystranddisplacement1":
#                 for reaction_dataset in datasets[reaction_type]:
#                     reaction_id = "/" + reaction_type + "/" + reaction_dataset
#                     document_name = PATH + "/dataset" + reaction_id + ".csv"
#                     file =  open_csv(document_name)
#                     row = 1
#                     while row < len(file) and file[row][0] != '' :
#                         predicted_log_10_rate, real_log_10_rate, sq_error = estimate_ReyanldoSequential(row, theta, file, reaction_id, str(row), "ReynaldoSequential", kinetic_model)
#                         predicted_log_10_rates.append(predicted_log_10_rate)
#                         real_log_10_rates.append(real_log_10_rate)
#                         row+=1
#             else:
#                 pass

#     return predicted_log_10_rates

# from_theta_to_rate([13.0580, 3, 13.0580, 3,  13.0580, 3, 13.0580, 3,  13.0580, 3, 13.0580, 3,  13.0580, 3,   0.0402 ], datasets)