from __future__ import division
import warnings
import json
import subprocess
import os
import numpy as np
import  random
import math
import pickle as pickle
from numpy.lib.function_base import average
from  scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, coo_matrix
from scipy.special import logsumexp as logsumexp 
from scipy import stats
from filelock import Timeout, FileLock
from shutil import rmtree

R = 0.001987 # R is the molar gas constant in kcal/(mol K).
MOLARITY_OF_WATER = 55.14 # mol/L at 37C
NUCLEOTIDES = "ACTG"
TRANSLATION_TABLE = str.maketrans(NUCLEOTIDES, "TGAC")
RETURN_MINUS_INF = 1e-10

# PATH = '/home/aliseyfi/scratch/Probabilistic-Programming/Project/'
# PATH = '/Users/aliseyfi/Documents/UBC/Probabilistic-Programming/Probabilistic-Programming/Project/'
PATH = 'C:/Users/jlovr/CS532-project/Probabilistic-Programming/Project/'
# PATH = "/home/jlovrod/projects/def-condon/jlovrod/Probabilistic-Programming/Project/"

class MyStrand(object):   
    def __init__(self, sequence, complement=None):
        """Create a strand by providing a sequence from 5' to 3' ends"""
        self.sequence = sequence
        if complement:
            self.complement = complement
        else:
            if type(self.sequence) == bytes:
                self.sequence = self.sequence.decode("utf-8") 
            seq = ''.join(list(reversed(self.sequence))).translate(
                    TRANSLATION_TABLE)
            self.complement = MyStrand(seq, self)
    
    def __len__(self):
        return len(self.sequence)
        
    def __eq__(self, other):
        return self.sequence == other.sequence

class ParentComplex(object):
    """Contains function and variables that different type of reaction have in common"""
    def __init__(self, theta,  T, concentration, sodium, magnesium, dataset_name, docID,rate_method):
  
        if rate_method == "ARRHENIUS":
            self.kinetic_parameters = { "stack": (theta[0] , theta[1]) ,
                               "loop": (theta[2] , theta[3]),
                               "end": (theta[4] , theta[5]),
                               "stack+loop": (theta[6] , theta[7]),
                               "stack+end": (theta[8] , theta[9]),
                               "loop+end": (theta[10] , theta[11]),
                               "stack+stack": (theta[12] , theta[13]), 
                               "alpha" : (theta[14]) }
        elif rate_method == "METROPOLIS":
            self.kinetic_parameters ={ "k_uni" : theta[0] , "k_bi" : theta[1] }
        else:
            raise ValueError('Error: Please specify rate_method to be Arrhenius or Metropolis in the configuration file!')
        
        # self.create_discotress = False
        # self.discotress_samples=False
        self.dataset_name = dataset_name
        self.docID = docID 
        self.T = T
        self.concentration = concentration
        self.sodium = sodium
        self.magnesium = magnesium

        with open(PATH + "CTMCs/" +self.dataset_name+ "/" + "statespace" + "/" + "statespace" + str(self.docID), "rb") as state_file:
            self.statespace = pickle.load(state_file, encoding='latin1')

        with open(PATH + "CTMCs/" +self.dataset_name+ "/" + "energy" + "/" + "energy" + str(self.docID), "rb") as energy_file:
            self.energies = pickle.load(energy_file, encoding='latin1')

        with open(PATH + "CTMCs/" +self.dataset_name+ "/" + "transition_structure" + "/" + "transition_structure" + str(self.docID), "rb") as context_file:
            self.transition_structure= pickle.load(context_file, encoding='latin1')

        with open(PATH + "CTMCs/" +self.dataset_name+ "/" + "fast_access" + "/" + "fast_access" + str(self.docID), "rb") as fast_file:
            self.fast_access = pickle.load(fast_file, encoding='latin1')


        # self.discotress_path = PATH + "CTMCs" +self.dataset_name+ "/" + "discotress" + "/" + str(self.docID) + "/"
        self.discotress_path = PATH + "CTMCs" +self.dataset_name+ "/" + "discotress" + "/" + str(self.docID) + "/" + str(random.uniform(0,1)) + "/"
        while os.path.exists(self.discotress_path):
            self.discotress_path = PATH + "CTMCs" +self.dataset_name+ "/" + "discotress" + "/" + str(self.docID) + "/" + str(random.uniform(0,1)) + "/"
        os.makedirs(self.discotress_path)
        
        # self.lock_path = self.discotress_path+"edge_weights.dat.lock"

        # if self.create_discotress:
        #     try:
        #         self.discotress_path = PATH + "CTMCs" +self.dataset_name+ "/" + "discotress" + "/" + str(self.docID) + "/"
        #         try:
        #             os.mkdir(PATH + "CTMCs" +self.dataset_name+"/" + "discotress")
        #         except:
        #             pass
        #         os.mkdir(PATH + "CTMCs" +self.dataset_name+ "/" + "discotress" + "/" + str(self.docID))
        #     except:
        #         print("Exception creating directories")
        #         raise


        self.neighbours_dictionary = {s:set([]) for s in self.statespace}
        for s0, s1 in list(self.transition_structure.keys()):
            self.neighbours_dictionary[s1].add(s0)
            self.neighbours_dictionary[s0].add(s1)


        self.rates ={}
        self.local_context_bi = dict()     
        self.local_context_uni = dict()    

        for i in self.kinetic_parameters: 
            for j in self.kinetic_parameters:
                self.local_context_bi [i , j]  = 0
                self.local_context_uni [i  , j]  = 0


    def possible_states(self, state):
        return self.neighbours_dictionary[state] 
    
        
    def initial_final_state(self):
        initialStateConfig, finalStateConfig = self.initial_final_state_config()
        initialState = self.statespace.index(initialStateConfig) 
        finalState = self.statespace.index(finalStateConfig) 

        return [initialState, finalState]

    def Metropolis_rate(self, state1, state2 ):
        """ Uses the Metropolis kinetic model to calculate the transition  rate from state1 to state and the transition rate from state2 to state1. Only returns the transition rate from state1 to state2 """
        
        transition1 = (state1,state2 )
        transition2 = (state2,state1)
        rate1 = self.rates.get(transition1)
        
        if rate1:
            return rate1
       
        k_uni  = self.kinetic_parameters["k_uni"]
        k_bi  = self.kinetic_parameters["k_bi"]
        DeltaG = (self.energies[state2] -  self.energies[state1])
        DeltaG2   = -DeltaG
        RT = R * (self.T + 273.15)
    
        if   ( self.n_complex[state1] - self.n_complex[state2] ) == 1    :
            rate1 = k_bi * self.concentration
            rate2 = k_bi * np.e ** ( - DeltaG2 / RT)
        
        elif (self.n_complex[state1]  - self.n_complex[state2] ) ==  -1   :
            rate1 = k_bi * np.e ** ( - DeltaG / RT)
            rate2 = k_bi * self.concentration
         
        elif  self.n_complex[state1] == self.n_complex[state2]   :
            
            if DeltaG > 0.0:
                rate1 = k_uni  * np.e **(-DeltaG  / RT)
                rate2 = k_uni
            else:
                rate1 = k_uni
                rate2 = k_uni  * np.e **(-DeltaG2  / RT)
        else :
            raise ValueError('Exception, fix this in Metropolis_rate function.  Check transition rate calculations!')
       
        self.rates[transition1] = rate1
        self.rates[transition2] = rate2
        
        return rate1

       
    def Arrhenius_rate(self, state1, state2   ):
        """Uses the Arrhenius kinetic model to calculate the transition  rate from state1 to state and the transition rate from state2 to state1. Only returns the transition rate from state1 to state2 """

        transition1 = (state1,state2 )
        transition2 = (state2,state1)
        rate1 = self.rates.get(transition1)
       
        if rate1:
            return rate1
        try :
            
            left, right  = self.transition_structure[state1, state2  ]
        except:
          
            left, right = self.local_context(state1, state2)
            self.transition_structure[state1 , state2 ]  =  [left, right ]
            
        lnA_left, E_left = self.kinetic_parameters[left]
        lnA_right, E_right = self.kinetic_parameters[right]
        lnA = lnA_left + lnA_right
        E = E_left + E_right
        DeltaG = (self.energies[state2] -  self.energies[state1])
        DeltaG2   = -DeltaG
        RT = R * (self.T + 273.15)
       
        n_complex1 = self.n_complex[state1]
        n_complex2 = self.n_complex[state2]
        n_complexdiff = n_complex1 - n_complex2
    
        if   n_complexdiff ==0  :
            self.local_context_uni[left, right] += 2
            """Using plus 2 instead of plus 1 since were calculating we're calculating the transition rate from state1 to state2 and from state2 to state1 simultaneously. """
            if left != right :
                self.local_context_uni[right , left ] += 2
            if DeltaG > 0.0:
                rate1 = np.e **(lnA - (DeltaG + E) / RT)
                rate2 = np.e ** (lnA - E / RT)
            else:
                rate1 = np.e ** (lnA - E / RT)
                rate2 = np.e **(lnA - (DeltaG2 + E) / RT)
        elif   n_complexdiff == 1    :
            rate1 = (self.kinetic_parameters["alpha"] * self.concentration) * np.e  ** (lnA - E / RT)
            rate2 = self.kinetic_parameters["alpha"] * np.e ** (lnA - (DeltaG2 + E) / RT)
            self.local_context_bi[left, right] += 2
            if left != right:
                self.local_context_bi[right, left ] += 2
        elif n_complexdiff ==  -1   :
          
            self.local_context_bi[left, right] += 2
            if left != right:
                self.local_context_bi[right, left] += 2
            rate1 = self.kinetic_parameters["alpha"] * np.e ** (lnA - (DeltaG + E) / RT)
            rate2 = (self.kinetic_parameters["alpha"] * self.concentration) * np.e  ** (lnA - E / RT)
        else :
            raise ValueError('Exception, fix this in Arrhenius_rate function.  Check transition rate calculations!')
      
        self.rates[transition1] = rate1
        self.rates[transition2] = rate2
        
        return rate1

    def sample_path(self, s, final, rate_method):
        ''' samples a path from s->final, returns the time'''
        time = 0
        while s != final:
            rates = []
            ps = list(self.possible_states(s))  # possible sstates
            for si in ps: 
                if rate_method == "ARRHENIUS":
                    rate = self.Arrhenius_rate(s, si)
                elif rate_method == "METROPOLIS":
                    rate = self.Metropolis_rate(s, si)
                rates.append(rate)
            unnormalized_probs = [1/r for r in rates]
            summ = sum(unnormalized_probs)
            probs = [p/summ for p in unnormalized_probs]
            # sample next state 
            s = ps[list(np.random.multinomial(1, probs)).index(1)]
            # sample holding time from exponential
            time += np.random.exponential(1/sum(rates))
        return time

    def MFPT_paths(self, rate_method):
        """finds the MFPT from the initial state to the final state simulating paths (SSA) """
        [initialState,finalState] = self.initial_final_state()
        
        samples = []
        n_paths = 1000
        for _ in range(n_paths):
            samples.append(self.sample_path(self.statespace[initialState], self.statespace[finalState], rate_method))

        return average(samples)

    def discotress(self):
        subprocess.run(['discotress'],capture_output=True, cwd=self.discotress_path)
        
    def MFPT_matrix(self, rate_method):
        """finds the MFPT from the initial state to the final state by solving a system of linear equations """
        
        [initialState,finalState] = self.initial_final_state()
        self.num_complex()


        vals = []
        rows = []
        cols = []
        d = dict()
        d1 =dict()
        diags = [0 for i in range (len(self.statespace) ) ]

        lens = len(self.statespace)

        for s1 in  range(lens)  :

            state = self.statespace[s1]
            ps = self.possible_states(state)

            for state2 in ps:
                s2 = self.fast_access[state2]
                if rate_method =="ARRHENIUS":
                    myRate = self.Arrhenius_rate(state, state2 )
                elif rate_method == "METROPOLIS":
                    myRate = self.Metropolis_rate(state,state2)
                else:
                    raise ValueError('Error: Please specify rate_method to be Arrhenius or Metropolis!')
          
                sss = (s1, s2, myRate ) 
        
                if (sss[0], sss[1] ) not in d1 : 
                   
                    diags[sss[0] ] = diags[sss[0]]  - sss[2]
                    d1 [sss[0] , sss[1] ] = 1 
                if sss[0] == finalState or sss[1] == finalState: 
                    continue 
                    
                row = sss[0]- (sss[0] > finalState)
                col = sss[1] - (sss[1] > finalState)

                if (row, col ) in d : 
                    continue 
               
                rows.append(row) 
                cols.append(col )
                vals.append(sss[2])
                d [ row, col  ] = 1 

        b= -1 * np.ones(len(self.statespace)-1)
        diags=  np.delete(diags, finalState, 0)
        vals+= list(diags)
        rowtemp = [i for i in range( len(self.statespace) -1 )] 
        rows += rowtemp  
        cols += rowtemp
        warnings.filterwarnings('error')
        try: 
            rate_matrix_coo = coo_matrix((vals, (rows,cols)), shape=(len(self.statespace) -1, len(self.statespace) -1 ) , dtype=np.float64)
            rate_matrix_csr = csr_matrix(rate_matrix_coo)
            firstpassagetimes = spsolve(rate_matrix_csr  , b   )
        
        except RuntimeWarning as  w: 
            s = str(w) 
            if 'overflow' in s : 
                print( "Overflow warning :( ")
                return RETURN_MINUS_INF
            if 'underflow' in s : 
                print( "Underflow warning :( ")
                return 2.2250738585072014e-308
            return RETURN_MINUS_INF
        
        except Exception as w :
            print( "Singular matrix exception :( ")
            return RETURN_MINUS_INF
        except : 
            print(  "Exception - Don't know what happend  :( ")
            return RETURN_MINUS_INF
        
        if initialState > finalState : 
            firstpassagetime = firstpassagetimes[initialState-1]
        else : 
            firstpassagetime = firstpassagetimes[initialState]

        # if self.create_discotress:
        #     self.create_discotress_files()

        # if self.discotress_samples:
        #     self.discotress()

        return firstpassagetime

    
    def create_discotress_files(self):
        "making input for DISCOTRESS"
        
        # self.create_discotress = False

        # if os.path.isfile(self.discotress_path+"fpp_properties.dat"):
        #     os.remove(self.discotress_path+"fpp_properties.dat")

        # if os.path.isfile(self.discotress_path+"tp_stats.dat"):
        #     os.remove(self.discotress_path+"tp_stats.dat")

        if not os.path.isfile(self.discotress_path+"stat_prob.dat"):
            self.stat_prob = dict()
            kb = 1.380649E10-23
            C = logsumexp(-np.array(list(self.energies.values())) / (kb*self.T))

            with open(self.discotress_path+"stat_prob.dat", "w") as f:
                for si in self.statespace:
                    self.stat_prob[si] = - C - self.energies[si] / (kb*self.T)
                    line = str(self.stat_prob[si])+"\n"
                    f.write(line)

        if not os.path.isfile(self.discotress_path+"input.kmc"):
            with open(self.discotress_path+"input.kmc", "w") as f:
                text =  "NNODES " + str(len(self.statespace))+"\n"
                text += "NEDGES " + str(len(self.transition_structure))+"\n"
                text += "WRAPPER BTOA" + "\n"
                text += "TRAJ KPS" + "\n"
                text += "BRANCHPROBS" + "\n"
                text += "NELIM 5000" + "\n"
                text += "NABPATHS 10000" + "\n"
                text += "COMMSFILE communities.dat 2" +"\n"
                text += "NODESAFILE nodes.A 1"+"\n"
                text += "NODESBFILE nodes.B 1"+"\n"
                text += "NTHREADS 4"+"\n"
                f.write(text)

        initial, final = self.initial_final_state()
        if not os.path.isfile(self.discotress_path+"nodes.A"):
            with open(self.discotress_path+"nodes.A", "w") as f:
                f.write(str(final+1)+"\n")

        if not os.path.isfile(self.discotress_path+"nodes.B"):
            with open(self.discotress_path+"nodes.B", "w") as f:
                f.write(str(initial+1)+"\n")  

        if not os.path.isfile(self.discotress_path+"communities.dat"):
            with open(self.discotress_path+"communities.dat", "w") as f:
                for i in range(len(self.statespace)):
                    if i==final:
                        f.write("1\n")     
                    else:
                        f.write("0\n")  

        if not os.path.isfile(self.discotress_path+"edge_conns.dat"):
            with open(self.discotress_path+"edge_conns.dat", "w") as f:
                for s1, s2 in self.transition_structure.keys():
                    line = str(self.statespace.index(s1)+1)+"\t"+str(self.statespace.index(s2)+1)+"\n"
                    f.write(line)

        # if os.path.isfile(self.discotress_path+"edge_conns.dat"):
        #     os.remove(self.discotress_path+"edge_weights.dat")

        with open(self.discotress_path+"edge_weights.dat", "w") as f:
            for s1, s2 in self.transition_structure.keys():
                try:
                    line = str(np.log(self.rates[(s1,s2)]).item())+"\t"+str(np.log(self.rates[(s2,s1)].item())) + "\n"
                except:
                    line = str(np.log(self.rates[(s1,s2)]))+"\t"+str(np.log(self.rates[(s2,s1)])) + "\n"
                f.write(line)

        # if os.path.isfile(self.discotress_path+"edge_conns.dat"):
        #     os.remove(self.discotress_path+"edge_weights.dat")
        # with open(self.discotress_path+"edge_weights.dat", "a") as f:

    def f(self, t):

        return 10**self.real_log_10_rate*np.exp(-t*10**(self.real_log_10_rate))  

    def F(self, t):
        # uses m (mean first passage time)
        # m = 2.3727131391650352e-05
        # return 1 - np.exp(-t/m)  
        
        # same thing, uses k (log_10 of the rate constant) instead of m
        # print("real k", self.real_log_10_rate)
        return 1 - np.exp(-t*10**(self.real_log_10_rate)) 


    def get_wasserstein(self):

        self.create_discotress_files()

        self.discotress()

        vals = []
        with open(self.discotress_path+"fpp_properties.dat","r") as pathprops_f:
            for line in pathprops_f.readlines():
                val=float(line.split()[1])
                vals.append(val)
        vals=np.array(vals,dtype=float)

        hist_arr = np.zeros(100,dtype=int)
        width = max(vals)/99
        for val in vals:
            if not (val<0):
                bin = int(np.floor(val/width))
                hist_arr[bin] += 1
        summ = np.sum([width*h for h in hist_arr])
        hist_arr = [x/summ for x in hist_arr]
        X = np.linspace(0, max(vals), num=100)

        return stats.wasserstein_distance(hist_arr[10:90],self.f(X)[10:90])


    def get_score(self):
        self.create_discotress_files()

        self.discotress()

        vals = []
        with open(self.discotress_path+"fpp_properties.dat","r") as pathprops_f:
            for line in pathprops_f.readlines():
                val=float(line.split()[1])
                vals.append(val)
        
        vals=np.array(vals,dtype=float)

        # print(stats.kstest(vals,cdf=self.F))
        return stats.kstest(vals,cdf=self.F)[0]
    
        
    def rate_constant(self, concentration, real_rate, bimolTransition, kinetic_model, stochastic_conditionning=False, distance="KS_test"):
        """ Computes the estimated rate constant, and the error """

        real_log_10_rate = np.log(real_rate)/np.log(10)
        self.real_log_10_rate = real_log_10_rate

        # TODO: don't actually need to get the mfpt if stochastic_conditionning=True, just need to make sure the matrix is constructed. 
        mfpt = self.MFPT_matrix(kinetic_model) 

        # This estimates MFPT using SSA. Only works for very quick reactions such as hairpins
        # mfpt = self.MFPT_paths(kinetic_model)

        # Estimating reaction rate constant from first passage time.
        if bimolTransition == True :
            predicted_rate= 1.0 / (mfpt * concentration)
        else : 
            predicted_rate= 1.0 / mfpt
        warnings.filterwarnings('error')
        
        try:
            predicted_log_10_rate = np.log(predicted_rate)/np.log(10)
        except:
            predicted_log_10_rate = RETURN_MINUS_INF
            

        if stochastic_conditionning==True:

            if distance == "KS_test":
                error = self.get_score()
            elif distance == "wasserstein":
                error = self.get_wasserstein()
            rmtree(self.discotress_path)
            return  [   error ,  predicted_log_10_rate , real_log_10_rate , self.local_context_uni, self.local_context_bi]
        else:
            error  = math.pow( real_log_10_rate - predicted_log_10_rate, 2)
            rmtree(self.discotress_path)
            return  [   error ,  predicted_log_10_rate , real_log_10_rate , self.local_context_uni, self.local_context_bi]