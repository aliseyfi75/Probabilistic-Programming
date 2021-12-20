import numpy as np
from math import pow, log10, sqrt
import csv
import sys

sys.path.append('../')
sys.path.append('../Code/Scripts')
sys.path.append('../Code/')

import hairpin


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

    kinetic_model = "ARRHENIUS"
    reaction_type = "hairpin"
    reaction_dataset = "Fig4_0"

    _zip = bool(int(reaction_dataset[-1]))
    j = reaction_dataset[-3]
    reaction_id = "/" + reaction_type + "/" + reaction_dataset
    document_name = PATH + "/dataset" + reaction_id + ".csv"
    file =  open_csv(document_name)
    row = 1
    pred, real, score = estimate_Bonnet(row, theta, _zip, file, reaction_id, str(row), "Bonnet"+j, kinetic_model)

    return pred, real


# # # alpha = 2, samples = 250
# theta = [13.10052837,  3.06820081, 13.08189501,  3.06649163, 12.96714323,  2.94806634, \
#     12.97405858,  2.94277147, 13.0963243,   2.97091484, 13.11001231,  3.02589739, \
#         13.14727626,  2.8982616,   0.021218  ]
        
# eval_theta(theta)