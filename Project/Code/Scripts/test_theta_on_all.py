import numpy as np
from math import pow, log10, sqrt
import csv

from numpy.lib.function_base import average

import bubble
import hairpin
import helix
import three_waystranddisplacement
import four_waystrandexchange

from Scripts.model import open_csv
from Scripts.model import *


PATH = 'C:/Users/jlovr/CS532-project/Probabilistic-Programming/Project/'

    
def eval_theta_all(theta):

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

    kinetic_model = "ARRHENIUS"

    total_sq_error = []
    for reaction_type in datasets:
        if reaction_type == "bubble":
            for reaction_dataset in datasets[reaction_type]:
                reaction_id = "/" + reaction_type + "/" + reaction_dataset
                document_name = PATH + "/dataset" + reaction_id + ".csv"
                file =  open_csv(document_name)
                row = 1
                while row < len(file) and file[row][0] != '' :
                    pred, real, sq_error = estimate_AltanBonnet(row, theta, file, reaction_id, str(row), "Altanbonnet", kinetic_model)
                    total_sq_error.append(sq_error)
                    row+=1
        if reaction_type == "four_waystrandexchange":
            for reaction_dataset in datasets[reaction_type]:
                reaction_id = "/" + reaction_type + "/" + reaction_dataset
                document_name = PATH + "/dataset" + reaction_id + ".csv"
                file =  open_csv(document_name)
                row = 1
                while row < len(file) and file[row][0] != '' :
                    pred, real, sq_error = estimate_DabbyThesis(row, theta, file, reaction_id, str(row), "Dabby", kinetic_model)
                    total_sq_error.append(sq_error)
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
                    pred, real, sq_error = estimate_Bonnet(row, theta, _zip, file, reaction_id, str(row), "Bonnet"+j, kinetic_model)
                    total_sq_error.append(sq_error)
                    row+=1
        if reaction_type == "hairpin1":
            for reaction_dataset in datasets[reaction_type]:
                _zip = bool(int(reaction_dataset[-1]))
                reaction_id = "/" + reaction_type + "/" + reaction_dataset
                document_name = PATH + "/dataset" + reaction_id + ".csv"
                file =  open_csv(document_name)
                row = 1
                while row < len(file) and file[row][0] != '' :
                    pred, real, sq_error = estimate_BonnetThesis(row, theta, _zip, file, reaction_id, str(row), "GoddardT", kinetic_model)
                    total_sq_error.append(sq_error)
                    row+=1
        if reaction_type == "hairpin4":
            for reaction_dataset in datasets[reaction_type]:
                _zip = bool(int(reaction_dataset[-1]))
                reaction_id = "/" + reaction_type + "/" + reaction_dataset
                document_name = PATH + "/dataset" + reaction_id + ".csv"
                file =  open_csv(document_name)
                row = 1
                while row < len(file) and file[row][0] != '' :
                    pred, real, sq_error = estimate_Kim(row, theta, _zip, file, reaction_id, str(row), "Kim", kinetic_model)
                    total_sq_error.append(sq_error)
                    row+=1
        if reaction_type == "helix":
            for reaction_dataset in datasets[reaction_type]:
                _zip = bool(int(reaction_dataset[-1]))
                reaction_id = "/" + reaction_type + "/" + reaction_dataset
                document_name = PATH + "/dataset" + reaction_id + ".csv"
                file =  open_csv(document_name)
                row = 1
                while row < len(file) and file[row][0] != '' :
                    pred, real, sq_error = estimate_Morrison(row, theta, _zip, file, reaction_id, str(row), "Morrison", kinetic_model)
                    total_sq_error.append(sq_error)
                    row+=1
        if reaction_type == "helix1":
            for reaction_dataset in datasets[reaction_type]:
                _zip = False
                reaction_id = "/" + reaction_type + "/" + reaction_dataset
                document_name = PATH + "/dataset" + reaction_id + ".csv"
                file =  open_csv(document_name)
                row = 1
                while row < len(file) and file[row][0] != '' :
                    pred, real, sq_error = estimate_ReynaldoDissociate(row, theta, _zip, file, reaction_id, str(row), "ReynaldoDissociate", kinetic_model)
                    total_sq_error.append(sq_error)
                    row+=1
        if reaction_type == "three_waystranddisplacement":
            for reaction_dataset in datasets[reaction_type]:
                reaction_id = "/" + reaction_type + "/" + reaction_dataset
                document_name = PATH + "/dataset" + reaction_id + ".csv"
                file =  open_csv(document_name)
                row = 1
                while row < len(file) and file[row][0] != '' :
                    pred, real, sq_error = estimate_Zhang(row, theta, file, reaction_id, str(row), "Zhang", kinetic_model)
                    total_sq_error.append(sq_error)
                    row+=1
        if reaction_type == "three_waystranddisplacement1":
            for reaction_dataset in datasets[reaction_type]:
                reaction_id = "/" + reaction_type + "/" + reaction_dataset
                document_name = PATH + "/dataset" + reaction_id + ".csv"
                file =  open_csv(document_name)
                row = 1
                while row < len(file) and file[row][0] != '' :
                    pred, real, sq_error = estimate_ReyanldoSequential(row, theta, file, reaction_id, str(row), "ReynaldoSequential", kinetic_model)
                    total_sq_error.append(sq_error)
                    row+=1

    mse = np.average(total_sq_error)
    count = 0
    for e in total_sq_error:
        if e <= log10(3):
            count+=1
    print("MSE", mse)
    print("Within factor of three (Nasim)", count/len(total_sq_error))

    count = 0
    for e in total_sq_error:
        if sqrt(e) <= log10(3):
            count+=1
    within3 = count/len(total_sq_error)
    print("Within factor of three (correct)", within3)

    return mse, within3