import torch
import torch.distributions as dist

from daphne import daphne

from primitives import baseprimitives, distlist
from evaluation_based_sampling import evaluate_program

from tests import is_tol, run_prob_test,load_truth

def topological_sort(graph):
    nodes = graph[1]['V']
    edges = graph[1]['A']
    is_visited = dict.fromkeys(nodes, False)
    node_stack = []
    node_order_reverse = []
    for node in nodes:
        if not is_visited[node]:
            node_stack.append((node, False))
        while len(node_stack) > 0:
            node, flag = node_stack.pop()
            if flag:
                node_order_reverse.append(node)
                continue
            is_visited[node] = True
            node_stack.append((node, True))
            if node not in edges:
                continue
            children = edges[node]
            for child in children:
                if not is_visited[child]:
                    node_stack.append((child, False))
    return node_order_reverse[::-1]

# Put all function mappings from the deterministic language environment to your
# Python evaluation context here:
env = {**baseprimitives, **distlist}

def deterministic_eval(exp):
    "Evaluation function for the deterministic target language of the graph based representation."
    if isinstance(exp, list):
        if exp[0] == 'hash-map':
            exp = ['hash-map'] + [value for expression in exp[1:] for value in expression]
    return evaluate_program(exp)

def value_subs(expressions, variables):
    if isinstance(expressions, list):
        result = []
        for expression in expressions:
            result.append(value_subs(expression, variables))
    else:
        if expressions in variables:
            result = variables[expressions]
        else:
            result = expressions
    return result

def sample_from_joint(graph):
    "This function does ancestral sampling starting from the prior."
    # TODO insert your code here
    node_order = topological_sort(graph)
    results = {}
    for node in node_order:
        first_statement, *other_statements = graph[1]['P'].get(node)
        if first_statement == 'sample*':
            dist = deterministic_eval(value_subs(other_statements, results))
            result = dist.sample()
        if first_statement == 'observe*':
            result = graph[1]['Y'].get(node)
        results[node] = result
    return deterministic_eval(value_subs(graph[2], results))


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
    
    for i in range(1,14):
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
    run_probabilistic_tests()


    for i in range(1,5):
        graph = daphne(['graph','-i','/Users/aliseyfi/Documents/UBC/Semester3/Probabilistic-Programming/HW/Probabilistic-Programming/Assignment_2//programs/{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(sample_from_joint(graph))    

    