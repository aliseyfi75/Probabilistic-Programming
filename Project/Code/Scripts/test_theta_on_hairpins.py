import numpy as np
from math import pow, log10, sqrt
import csv

from numpy.lib.function_base import average

import bubble
import hairpin
import helix
import three_waystranddisplacement
import four_waystrandexchange


PATH = 'C:/Users/jlovr/CS532-project/Probabilistic-Programming/Project/'

def open_csv(document) :
    """open a csv file"""
    with open(document, 'rt') as f:
        my_CSV = list(csv.reader(f))
    return my_CSV

def estimate_Bonnet(row, theta, _zip, file, dataset_name, docID, name, kinetic_model, stochastic_conditionning=False):
    docID = name + str(_zip) + docID
    magnesium = 0
    # [sq_error, predicted_log_10_rate, real_log_10_rate, stuctureCounterUniLocal, half_context_biLocal] \
    [sq_error, predicted_log_10_rate, real_log_10_rate, stuctureCounterUniLocal, half_context_biLocal] \
        = hairpin.main( float (file[row][5]), theta, file[row][1].rstrip(),file[row][2].rstrip(), _zip, 1000/  float( file[row][3] )- 273.15, float ( file[row][7] ), float ( file[row][8] ), magnesium, dataset_name, docID, kinetic_model, stochastic_conditionning)

    return predicted_log_10_rate, real_log_10_rate, sq_error
    
def eval_theta(theta):

    datasets = { "hairpin" : ["Fig4_0"]}
    kinetic_model = "ARRHENIUS"

    total_sq_error = []
    for reaction_type in datasets:
        if reaction_type == "hairpin":
            for reaction_dataset in datasets[reaction_type]:
                _zip = bool(int(reaction_dataset[-1]))
                j = reaction_dataset[-3]
                reaction_id = "/" + reaction_type + "/" + reaction_dataset
                document_name = PATH + "/dataset" + reaction_id + ".csv"
                file =  open_csv(document_name)
                row = 1
                while row < len(file) and file[row][0] != '' :
                    pred, real, score = estimate_Bonnet(row, theta, _zip, file, reaction_id, str(row), "Bonnet"+j, kinetic_model)
                    total_sq_error.append(score)
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
    print("Within factor of three (correct)", count/len(total_sq_error))