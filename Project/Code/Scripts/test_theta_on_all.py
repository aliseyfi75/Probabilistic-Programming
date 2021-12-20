import numpy as np
from math import pow, log10, sqrt
import csv
import sys

sys.path.append('../')
sys.path.append('../Code/Scripts')
sys.path.append('../Code/')

from numpy.lib.function_base import average

import bubble
import hairpin
import helix
import three_waystranddisplacement
import four_waystrandexchange

from Scripts.model import open_csv
from Scripts.model import *



# PATH = '/home/aliseyfi/scratch/Probabilistic-Programming/Project/'
# PATH = '/Users/aliseyfi/Documents/UBC/Probabilistic-Programming/Probabilistic-Programming/Project/'
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


# # # alpha = 1, samples=50
# # theta = [13.06118825,  2.96903267, 13.08879566 , 3.08324999, 13.1165558,   3.09489443, \
# #      12.97678333 , 2.88934503, 12.96237856,  3.01360373, 12.89525469 , 2.9064522  ,\
# #           12.84361076,  3.15635367, -0.02622133]
# # MSE 5.924722419549694
# # Within factor of three (Nasim) 0.584375
# # Within factor of three (correct) 0.446875

# # # alpha=1, theta = 100
# # theta = [13.02079458,  3.05853647, 13.1689483  , 2.95179187, 12.83109197,  3.00781622, \
# #     13.04630884 , 3.06074229 ,13.14258137 , 3.26062992, 13.16966668,  2.8629887,\
# #         12.83335831 , 2.75356964 , 0.02324164]
# # MSE 0.6225347436713771
# # Within factor of three (Nasim) 0.665625
# # Within factor of three (correct) 0.440625

# #  # alpha = 1, theta = 250
# # theta = [13.08422124,  2.91626845, 13.15549004,  2.94207538, 13.06600446,  2.96746481,  \
# #     13.10760293,  2.90387457, 13.04850659,  2.98913925, 13.12353384,  3.03318451,   \
# #         12.98680545,  2.93091869,  0.03826118]
# # MSE 0.769348909555317
# # Within factor of three (Nasim) 0.5875  
# # Within factor of three (correct) 0.3625

#  # alpha = 2, theta = 50
# # theta = [13.47270692,  2.96462669, 13.07673163 , 3.03380658, 13.43435709,  3.00044792, \
# #       12.76918284 , 2.86601009, 13.31278815  ,2.97103586 ,13.2342286  , 3.18714006, 
# #        12.97825425 , 3.01830007, -0.10731259]
# # MSE 5.410478619047245
# # Within factor of three (Nasim) 0.515625
# # Within factor of three (correct) 0.340625

# # # alpha = 2, theta = 100
# # theta = [ 1.30945015e+01,  2.88553283e+00,  1.28909934e+01,  3.21180607e+00, \
# #   1.31362605e+01,  3.02593002e+00,  1.31859112e+01,  2.96292555e+00, \
# #   1.29623183e+01,  2.94133526e+00,  1.30061344e+01,  3.00568477e+00, \
# #   1.29894063e+01,  2.97815887e+00, -5.89998716e-03]
# # MSE 5.451022063013895
# # Within factor of three (Nasim) 0.65
# # Within factor of three (correct) 0.590625

# # # alpha = 2, samples = 250
# # theta = [13.10052837,  3.06820081, 13.08189501,  3.06649163, 12.96714323,  2.94806634, \
# #     12.97405858,  2.94277147, 13.0963243,   2.97091484, 13.11001231,  3.02589739, \
# #         13.14727626,  2.8982616,   0.021218  ]
# # MSE 0.5656632439519658
# # Within factor of three (Nasim) 0.715625  
# # Within factor of three (correct) 0.540625

# #  # alpha = 4, samples = 50
# # theta = [13.04616153,  2.99346818, 13.02384133,  2.89847984, 12.98567399,  3.14762197, \
# #     13.21290992,  3.1381616 , 13.23129644 , 3.02215763, 13.16989318 , 3.01209141, \
# #         12.9081389 ,  2.97200462, -0.07885271]
# # MSE 5.575008006289387
# # Within factor of three (Nasim) 0.51875
# # Within factor of three (correct) 0.321875

# #  # alpha = 4, samples = 100
# # theta = [12.91871982,  2.99136128, 13.12235062,  2.86346823, 13.1008569,   2.95072911, \
# #      13.15901815,  2.90054416, 12.86449901,  2.913547,   12.97036645 , 2.88958644, \
# #          13.03420697,  3.09293625, -0.16487232]
# # MSE 4.701011531513089
# # Within factor of three (Nasim) 0.45625   
# # Within factor of three (correct) 0.246875

#  # alpha = 4, samples = 250 
# theta = [13.11927312,  2.94534727, 13.16256703,  2.94424204, 13.07477452,  \
#     2.86149381, 13.07832981,  2.99417638, 13.11264839,  3.06783864, 13.12858785,  \
#         2.94152868, 13.09096029,  3.03766899,  0.07344786]
# # MSE 0.7801398683272824
# # Within factor of three (Nasim) 0.578125  
# # Within factor of three (correct) 0.346875

# theta = [12.48210154,  3.90275578, 13.09800791,  2.84315634, 12.93531016,  3.03627209,
#  12.73331783,  3.37982778, 13.59546396,  3.64466079, 12.79966671,  3.41207022,
#  13.22211924,  3.11840773,  0.11566866]

# eval_theta_all(theta)

# theta= [12.2302,  3.8982, 13.2600,  2.8296, 12.7198,  3.1725, 12.4778,  3.3283,
#         13.1929,  2.8059, 12.9790,  3.0551, 13.0931,  2.7412,  0.1115]

# theta= [12.08498719,  3.9264619,  12.78452361,  2.4836623,  12.70107978,  3.31345123,
#  13.06708679,  3.36649785, 12.21150548,  2.96242796, 12.51392362,  2.8573691,
#  12.83886797, 2.99156085,  0.13127697]

# eval_theta_all(theta)

# BBVI
# theta = [12.543479119992815,
#             4.226898398415472,
#             13.337500173890277,
#             2.8595604197468645,
#             12.248657796490361,
#             3.0970839005977617,
#             12.908034847259342,
#             3.515686400591013,
#             12.954033117162176,
#             3.233111020749515,
#             12.298319716642903,
#             2.177464964997234,
#             13.542831816491484,
#             2.839804444835156,
#             0.13583210648998936]

# eval_theta_all(theta)

