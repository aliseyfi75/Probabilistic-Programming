import numpy as np
from math import pow, log10, sqrt
import csv

from numpy.lib.function_base import average

import bubble
import hairpin
import helix
import three_waystranddisplacement
import four_waystrandexchange

PATH = '/Users/aliseyfi/Documents/UBC/Probabilistic-Programming/Probabilistic-Programming/Project/'
PATH = 'C:/Users/jlovr/CS532-project/Probabilistic-Programming/Project/'


def open_csv(document) :
    """open a csv file"""
    with open(document, 'rt') as f:
        my_CSV = list(csv.reader(f))
    return my_CSV


# bubble
def estimate_AltanBonnet(row, theta, file, dataset_name, docID, name, kinetic_model) :
    docID = name + docID
    flurPosition = 17
    real_log_10_rate = 1 / float( file[row][5])

    [sq_error, predicted_log_10_rate, real_log_10_rate, stuctureCounterUniLocal, half_context_biLocal] \
        = bubble.main(real_log_10_rate, theta, file[row][1].rstrip(),file[row][2].rstrip(),file[row][3].rstrip(), (1000/ float (file[row][4] ))-273.15, float (file[row][8] ), float (file[row][9] ), 0, flurPosition, dataset_name, docID, kinetic_model) 

    return predicted_log_10_rate, real_log_10_rate, sq_error

# four_waystrandexchange
def estimate_DabbyThesis(row, theta , file, dataset_name , docID, name, kinetic_model) :
    docID = name +docID
    [  sq_error ,  predicted_log_10_rate, real_log_10_rate ,  stuctureCounterUniLocal, half_context_biLocal] \
        = four_waystrandexchange.main(float( file[row][8]) ,   float( file[row][13])  ,  int(file[row][1]) , int(file[row][2]) ,  file[row][3]  ,  file[row][4]  , file[row][5] ,  6,6 , theta,  1000/  float (file[row][6] )- 273.15 , np.max ( ( float (file[row][16] ), float (file[row][17]  ) ) ) ,  float (file[row][11]) ,float (file[row][12])  ,  dataset_name, docID , name, kinetic_model)

    return predicted_log_10_rate, real_log_10_rate, sq_error


# hairpin
def estimate_Bonnet(row, theta, _zip, file, dataset_name, docID, name, kinetic_model):
    docID = name + str(_zip) + docID
    magnesium = 0
    [sq_error, predicted_log_10_rate, real_log_10_rate, stuctureCounterUniLocal, half_context_biLocal] \
        = hairpin.main( float (file[row][5]), theta, file[row][1].rstrip(),file[row][2].rstrip(), _zip, 1000/  float( file[row][3] )- 273.15, float ( file[row][7] ), float ( file[row][8] ), magnesium, dataset_name, docID, kinetic_model)

    return predicted_log_10_rate, real_log_10_rate, sq_error


# hairpin1
def estimate_BonnetThesis( row, theta, _zip, file, dataset_name, docID, name, kinetic_model):
    docID = name + str(_zip) +docID
    real_log_10_rate = 1 / float( file[row][4])
    [sq_error, predicted_log_10_rate, real_log_10_rate, stuctureCounterUniLocal, half_context_biLocal] \
        =   hairpin.main(  real_log_10_rate, theta, file[row][1].rstrip(),file[row][2].rstrip(), _zip, 1000/  float (file[row][3] ) - 273.15,  float (file[row][7] ), float (file[row][8] ), 0, dataset_name, docID , kinetic_model)
    return predicted_log_10_rate, real_log_10_rate, sq_error


# hairpin4
def estimate_Kim( row, theta, _zip, file, dataset_name, docID, name, kinetic_model):
    docID = name + str(_zip) +docID
    magnesium = 0
    [sq_error, predicted_log_10_rate, real_log_10_rate, stuctureCounterUniLocal, half_context_biLocal] \
        =   hairpin.main( float (file[row][5]), theta, file[row][1].rstrip(),file[row][2].rstrip(), _zip, 1000/  float( file[row][3] )- 273.15, float ( file[row][7] ), float ( file[row][8] ), magnesium, dataset_name, docID, kinetic_model)

    return predicted_log_10_rate, real_log_10_rate, sq_error


# helix
def estimate_Morrison( row, theta, _zip, file, dataset_name, docID, name, kinetic_model) :
    docID = name + str(_zip) +docID
    [sq_error, predicted_log_10_rate, real_log_10_rate, stuctureCounterUniLocal, half_context_biLocal] \
        = helix.main(  pow(10, float (file[row][5] )), theta, file[row][1].rstrip(), _zip, 1000/  float (file[row][3] ) - 273.15, np.max ( ( float (file[row][16] ), float (file[row][17]  ) ) ), float (file[row][8] ), 0, "", dataset_name, docID, name, kinetic_model)

    return predicted_log_10_rate, real_log_10_rate, sq_error


# helix1
def estimate_ReynaldoDissociate( row, theta, _zip, file, dataset_name, docID, name, kinetic_model) :
    docID = name + str(_zip) +docID
    [sq_error, predicted_log_10_rate, real_log_10_rate,stuctureCounterUniLocal, half_context_biLocal] \
        = helix.main(  float( file[row][5] ), theta, file[row][2].rstrip(), _zip, float (file[row][3] ), np.max ( ( float (file[row][16] ), float (file[row][17]  ) ) ), float (file[row][7] ), float (file[row][8] ),file[row][9], dataset_name, docID, name , kinetic_model)

    return predicted_log_10_rate, real_log_10_rate, sq_error


# three_waystranddisplacement
def estimate_Zhang( row, theta, file,dataset_name, docID, name, kinetic_model) :
    docID = name + docID
    real_log_10_rate = pow(10, float( file[row][7])  )
    [sq_error, predicted_log_10_rate, real_log_10_rate, stuctureCounterUniLocal, half_context_biLocal] \
        = three_waystranddisplacement.main( True, file[row][2], real_log_10_rate, int ( file[row][1]   )  ,file[row][3], file[row][4], theta, 1000/ float (file[row][5]) - 273.15, np.max ( ( float (file[row][16] ), float (file[row][17]  ) ) ), float (file[row][9]), float (file[row][10]), "", dataset_name, docID, name, "", "", "", "", kinetic_model )

    return predicted_log_10_rate, real_log_10_rate, sq_error


# three_waystranddisplacement1
def estimate_ReyanldoSequential( row, theta, file, dataset_name, docID, name, kinetic_model) :
    docID = name +docID
    real_log_10_rate =float( file[row][5])
    [sq_error, predicted_log_10_rate, real_log_10_rate,  stuctureCounterUniLocal, half_context_biLocal] \
        = three_waystranddisplacement.main( False ,"",  real_log_10_rate, 0 ,file[row][2] ,"", theta, float( file[row][3]), np.max ( ( float (file[row][16] ), float (file[row][17]  ) ) ), float (file[row][7]), float( file[row][8]), file[row][9], dataset_name, docID, name,  "", "", "", "" , kinetic_model)

    return predicted_log_10_rate, real_log_10_rate, sq_error


    ## Alpha = kbi/kuni = 0.0402


def main(): 

    # kinetic_model = "METROPOLIS"
    kinetic_model = "ARRHENIUS"

    if kinetic_model == "ARRHENIUS":
        # Initial parameter set for the Arrhenius model
        # theta = [13.0580, 3, 13.0580, 3,  13.0580, 3, 13.0580, 3,  13.0580, 3, 13.0580, 3,  13.0580, 3,   0.0402 ]

        # near optimal parameter set for the Arrhenius model (roughly fig 6 from DNA23) : Total error 131
        # theta = [13.0580, 5, 17.0580, 5,  10.0580, 1, 1.0580, -2,  13.0580, 1, 5.0580, 0,  4.0580, -2,   0.0402 ]

        # result from importance sampling : Total error 87
        theta = [11.8214,  2.4291, 13.5964,  3.7051, 14.5229,  2.7318, 11.2671,  3.9952, 12.7210,  3.8409, 12.5655,  4.5394, 11.9421,  2.3308,  0.1189]


    elif kinetic_model == "METROPOLIS":
        # Initial parameter set for the Metropolis model
        theta = [8.2 *  (10 **6), 3.3  * (10**5) ]
        
        # decent parameter set for the Metropolis model (from pathway elaboration fig 8)
        # theta = [3.61 *  (10 **6), 1.12  * (10**5) ]


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
                    print(pred, real, sq_error)
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
                    print(pred, real, sq_error)
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
                    print(pred, real, sq_error)
                    total_sq_error.append(sq_error)
                    row+=1
        if reaction_type == "hairpin1":
            # for reaction_dataset in datasets[reaction_type]:
            for reaction_dataset in datasets[reaction_type][1:]:
                # TODO: set back to normal
                _zip = bool(int(reaction_dataset[-1]))
                reaction_id = "/" + reaction_type + "/" + reaction_dataset
                document_name = PATH + "/dataset" + reaction_id + ".csv"
                file =  open_csv(document_name)
                row = 1
                # TODO: set back to normal
                # while row < len(file) and file[row][0] != '' :
                while row==1:
                    pred, real, sq_error = estimate_BonnetThesis(row, theta, _zip, file, reaction_id, str(row), "GoddardT", kinetic_model)
                    print(pred, real, sq_error)
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
                    print(pred, real, sq_error)
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
                    print(pred, real, sq_error)
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
                    print(pred, real, sq_error)
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
                    print(pred, real, sq_error)
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
                    print(pred, real, sq_error)
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
    print("Within factor of three (correct)", count/len(total_sq_error))


if __name__ == "__main__":
    main()