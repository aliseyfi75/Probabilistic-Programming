import numpy as np
from math import pow, log10, sqrt
import csv

from numpy.lib.function_base import average

import bubble
import hairpin
import helix
import three_waystranddisplacement
import four_waystrandexchange

# PATH = '/home/aliseyfi/scratch/Probabilistic-Programming/Project/'
# PATH = '/Users/aliseyfi/Documents/UBC/Probabilistic-Programming/Probabilistic-Programming/Project/'
PATH = 'C:/Users/jlovr/CS532-project/Probabilistic-Programming/Project/'
# PATH = "/home/jlovrod/projects/def-condon/jlovrod/Probabilistic-Programming/Project/"


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
def estimate_Bonnet(row, theta, _zip, file, dataset_name, docID, name, kinetic_model, stochastic_conditionning=False, distance="KS_test"):
    docID = name + str(_zip) + docID
    magnesium = 0
    # [sq_error, predicted_log_10_rate, real_log_10_rate, stuctureCounterUniLocal, half_context_biLocal] \
    [error_list, predicted_log_10_rate, real_log_10_rate, stuctureCounterUniLocal, half_context_biLocal] \
        = hairpin.main( float (file[row][5]), theta, file[row][1].rstrip(),file[row][2].rstrip(), _zip, 1000/  float( file[row][3] )- 273.15, float ( file[row][7] ), float ( file[row][8] ), magnesium, dataset_name, docID, kinetic_model, stochastic_conditionning, distance)

    return predicted_log_10_rate, real_log_10_rate, error_list


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
        # theta = [11.8214,  2.4291, 13.5964,  3.7051, 14.5229,  2.7318, 11.2671,  3.9952, 12.7210,  3.8409, 12.5655,  4.5394, 11.9421,  2.3308,  0.1189]

        # result from importance sampling: 10 without stochastic conditionning
        # theta = [12.8374,  2.5197, 13.6469,  3.4312, 12.8005,  3.0433, 12.4569,  3.1836, 13.0407,  2.4607, 13.2525,  3.2284, 12.4343,  2.7114, -1.0636]
        # theta = [13.0709,  2.8145, 13.9252,  3.7936, 13.1373,  3.5791, 13.0436,  3.4662, 13.3587,  4.0021, 12.7691,  2.8062, 12.1392,  3.3764, -0.1215]
        # theta = [13.0401,  3.1595, 13.0070,  3.0607, 13.1856,  3.0344, 13.1405,  3.0649, 13.2826,  2.9854, 13.0824,  2.9047, 12.9854,  2.9556,  0.0321]
        # theta = [12.5230,  3.8405, 12.2272,  2.0046, 13.4973,  2.2954, 12.7737,  3.8148, 14.0437,  3.7377, 12.7112,  2.7404, 13.0197,  0.9761,  0.4915]

        # new IS results (no SC) 3 samples

        # new IS results (no SC) 100 samples        
        theta = [12.91512395419514, 2.9431004111756014, 13.1271482285833, 3.20040827852152, 13.100286228824169, 3.078559286894262, 12.804722134945063, 3.185934701538036, 13.186370889155395, 3.222938390271056, 13.144532504573368, 3.285365265119654, 12.624075834843783, 2.97537564936361, -0.16676269332788818]

        # new IS results (with SC)
        # theta = [13.1463739, 4.60096502, 13.68513883,  2.73753184, 13.97862454,  2.11238813, 13.76575808,  3.03109213, 14.67367892,  1.89077002, 11.6302467,   2.40079966, 13.11094522,  2.40366625,  1.08333854]

        # new IS result, 3 samples IS, no SC
        theta = [13.53770944,  2.3781851,  13.56991038,  3.07536884, 13.00340611,  4.45302736, 11.85610727,  1.55858067, 12.10633774,  3.58755744, 13.60933626,  1.86953814, 13.69105729,  2.4883468, -0.31591701]

        # new IS results, 3 samples IS, WITH SC
        theta = [13.54281141,  3.6989281,  13.07090143,  3.05406672, 14.29042946,  3.24569291, 14.91177878,  2.94148881, 11.80375963,  2.40034719, 12.36229114,  2.02421194, 13.34261972,  3.91425952, -1.81351612]
    
        # theta = [12.89711972,  3.1447547,  12.85693048,  3.03952734, 12.95408299,  2.77311255, 13.06968852,  2.90305143, 13.22004211,  2.95483632, 12.97743843,  2.9226678, 12.91125904,  2.96000395,  0.27776098]
        # FAST IS results, with SC:
        theta = [12.81201438,  2.93021193, 12.84488501,  3.11085196, 13.03820814,  2.88789423, 13.18246348,  2.90877865, 12.80971955,  3.37062998, 13.11346644,  3.05289144, 13.07358522,  2.8079438,   0.03255077]
    

        # FAST IS results, with SC, many hairpins
        theta = [12.67698866,  3.54344835, 13.65335947,  3.46840521, 12.40396588,  3.28032403, 12.99585543,  3.35862022, 12.63212201,  3.17113805, 13.86436576,  2.75644894, 12.93273246,  3.18377645,  0.41221341]
        theta = [13.19807623,  3.4027508,  13.38693288,  3.38801766, 13.24209065,  2.66902877, 12.84123496,  3.12923239, 12.81115959,  3.30187187, 12.97879413,  2.97998387, 13.16985691,  2.88507918, -0.04079903]
    

        # ks_test_squared, 1500 Importance samples. 
        theta = [12.98738276,  3.18292788, 12.91227223,  3.08759203, 13.0251467,   3.05538864 , 13.07837567,  2.99802686, 13.01760334,  3.02412123, 13.09092281,  3.06756969, 13.03141334,  3.06337368,  0.10149955]
        # MSE 0.08761631856722736
        # Within factor of three (Nasim) 1.0
        # Within factor of three (correct) 0.8648648648648649

        # ks_test, 1500 Importance samples. 
        theta = [ 12.9561932,  3.19018079,  13.1208651,  3.21702807,  13.1039766,  3.09602045,  13.1361887,  2.98588854,  13.0154658,  3.08898115,  12.8827507,  3.03187897,  13.1090835,  3.05818951, -1.88602100e-03]
        # MSE 0.08148109302197314
        # Within factor of three (Nasim) 1.0
        # Within factor of three (correct) 0.8918918918918919

        # MFPT, 1500 importance samples. 
        theta = [12.98125694,  3.16567847, 12.95882798,  3.11836914, 13.04789623,  3.03831106, 13.09115818,  2.99322587, 13.07014458,  3.04926228, 13.08575512,  2.91565609,  13.12455029,  2.98065487,  0.03550729]
        # MSE 0.08841835974604582
        # Within factor of three (Nasim) 1.0
        # Within factor of three (correct) 0.8648648648648649

        # MFPT, 10 importance samples. 
        theta = [12.74245407,  3.20298699, 12.36996956,  2.56962269, 12.85924211,  3.02210123, 12.97066044,  2.51351766, 12.89394806,  2.65528498, 12.61308151,  2.67130969, 13.80901682,  3.22884826, -0.69744547]
        # MSE 0.17762209830228734
        # Within factor of three (Nasim) 0.8918918918918919
        # Within factor of three (correct) 0.8108108108108109

        # ks_test, 10 Importance samples. 
        # theta = [12.92851843,  2.76352728, 13.4035104,   3.08737629, 13.12772025,  3.29456687, 12.5313715,   3.46128908, 13.37084083,  3.1282274,  12.76577248,  3.08709212, 13.38435881,  2.68516058,  0.31703005]
        # MSE 0.4835026851311203
        # Within factor of three (Nasim) 0.5135135135135135
        # Within factor of three (correct) 0.16216216216216217

        # ks_test squared, 10 Importance samples. 
        # theta = [12.55547824,  3.02131136, 13.32774127,  3.0014533,  13.36095061,  3.0932119, 13.14714057,  3.37075929, 12.82789794,  1.95641196, 13.09309228,  2.72663031, 13.25575739,  2.83015251,  0.04902172]
        # MSE 0.3434587963823442
        # Within factor of three (Nasim) 0.8378378378378378
        # Within factor of three (correct) 0.35135135135135137

    elif kinetic_model == "METROPOLIS":
        # Initial parameter set for the Metropolis model
        theta = [8.2 *  (10 **6), 3.3  * (10**5) ]
        
        # decent parameter set for the Metropolis model (from pathway elaboration fig 8)
        # theta = [3.61 *  (10 **6), 1.12  * (10**5) ]


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
                    print(pred, real, score)
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


if __name__ == "__main__":
    main()

#  - reasonable to do BBVI to optimize proposal for IS and SMC? 
#  - BBVI hyperparameters
#  - synthetic likelihood for emperical distribution
#  - different observation model for part of the data