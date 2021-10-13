import torch
import torch.distributions as dist

from daphne import daphne

from primitives import baseprimitives, distlist
from evaluation_based_sampling import evaluate_program

from tests import is_tol, run_prob_test,load_truth

# Put all function mappings from the deterministic language environment to your
# Python evaluation context here:
env = {**baseprimitives, **distlist}

def deterministic_eval(exp):
    "Evaluation function for the deterministic target language of the graph based representation."
    return evaluate_program(exp)


def sample_from_joint(graph):
    "This function does ancestral sampling starting from the prior."
    # TODO insert your code here
    return torch.tensor([0.0, 0.0, 0.0])


def get_stream(graph):
    """Return a stream of prior samples
    Args: 
        graph: json graph as loaded by daphne wrapper
    Returns: a python iterator with an infinite stream of samples
        """
    while True:
        yield sample_from_joint(graph)




#Testing:

def run_deterministic_tests():
    
    for i in range(1,13):
        #note: this path should be with respect to the daphne path!
        graph = daphne(['graph','-i','/Users/aliseyfi/Documents/UBC/Semester3/Probabilistic-Programming/HW/Probabilistic-Programming/Assignment_2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('/Users/aliseyfi/Documents/UBC/Semester3/Probabilistic-Programming/HW/Probabilistic-Programming/Assignment_2/programs/tests/deterministic/test_{}.truth'.format(i))
        ret = deterministic_eval(graph[-1])
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for graph {}'.format(ret,truth,graph))
        
        print('Test passed')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    #TODO: 
    num_samples=1e4
    max_p_value = 1e-4
    
    for i in range(1,7):
        #note: this path should be with respect to the daphne path!        
        graph = daphne(['graph', '-i', '/Users/aliseyfi/Documents/UBC/Semester3/Probabilistic-Programming/HW/Probabilistic-Programming/Assignment_2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('/Users/aliseyfi/Documents/UBC/Semester3/Probabilistic-Programming/HW/Probabilistic-Programming/Assignment_2/programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(graph)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')    


        
        
if __name__ == '__main__':
    

    run_deterministic_tests()
    # run_probabilistic_tests()




    # for i in range(1,5):
    #     graph = daphne(['graph','-i','../CS532-HW2/programs/{}.daphne'.format(i)])
    #     print('\n\n\nSample of prior of program {}:'.format(i))
    #     print(sample_from_joint(graph))    

    