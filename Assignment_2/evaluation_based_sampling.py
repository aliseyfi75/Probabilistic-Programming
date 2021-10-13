from daphne import daphne
from tests import is_tol, run_prob_test,load_truth

import torch

from primitives import baseprimitives, distlist

        
def evaluate_program(ast):
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """
    funcs = {}
    final_ast = ast
    if isinstance(ast, list):
        if isinstance(ast[0], list):
            if ast[0][0] == 'defn':
                for statement in ast:
                    if statement[0]== 'defn':
                        funcs[statement[1]] = (statement[1], statement[2], statement[3])
                        final_ast = final_ast[1:]
                    else:
                        result = eval(statement, {}, {}, funcs)
            if final_ast[0][0] != 'defn':
                result = eval(final_ast[0], {}, {}, funcs)
        else:
            result = eval(ast, {}, {}, funcs)
    else:
        result = eval(ast, {}, {}, funcs)
    return result[0]

def eval(x, sigma, l, funcs):
    "Evaluate an expression in an environment."
    if isinstance(x, list) and len(x) == 1:
        x = x[0]
    if not isinstance(x, list):
        if isinstance(x, int) or isinstance(x, float):
            result = torch.tensor(x, dtype=float)
        elif x in baseprimitives or torch.is_tensor(x) or x in funcs or x in distlist:
            result = x
        else:
            result = l[x]
    elif x[0] == 'if':
        cond_result, sigma = eval(x[1], sigma, l, funcs)
        if cond_result:
            result = x[2]
        else:
            result = x[3]
    elif x[0] == 'let':
        result_temp, sigma = eval(x[1][1], sigma, l, funcs)
        for e in x[2:-1]:
            _, sigma = eval(e, sigma, {**l, **{x[1][0]: result_temp}}, funcs)
        result, sigma = eval(x[-1], sigma, {**l, **{x[1][0]: result_temp}}, funcs)
    elif x[0] == 'sample':
        dist, sigma = eval(x[1], sigma, l, funcs)
        result = dist.sample()
    elif x[0] == 'observe':
        result = None
    else:
        statements = []
        for expression in x:
            statement, sigma = eval(expression, sigma, l, funcs)
            statements.append(statement)
        
        first_statemnt, other_statements = statements[0], statements[1:]
        if first_statemnt in baseprimitives:
            result = baseprimitives[first_statemnt](other_statements)

        elif first_statemnt in distlist:
            result = distlist[first_statemnt](other_statements)

        if first_statemnt in funcs:
            _, variables, process = funcs[first_statemnt]
            assignment = {key:value for key, value in zip(variables, other_statements)}
            result, sigma = eval(process, sigma, {**l, **assignment}, funcs)
        else:
            pass
    return result, sigma
           
    


def get_stream(ast):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(ast)
    


def run_deterministic_tests():
    
    for i in range(28,35):
        #note: this path should be with respect to the daphne path!
        ast = daphne(['desugar', '-i', '/Users/aliseyfi/Documents/UBC/Semester3/Probabilistic-Programming/HW/Probabilistic-Programming/Assignment_2/programs/tests/deterministic_tests/test_{}.daphne'.format(i)])
        truth = load_truth('/Users/aliseyfi/Documents/UBC/Semester3/Probabilistic-Programming/HW/Probabilistic-Programming/Assignment_2/programs/tests/deterministic_tests/test_{}.truth'.format(i))
        ret = evaluate_program(ast)
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,ast))
        
        print('Test passed')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    num_samples=1e4
    max_p_value = 1e-4
    
    for i in range(1,7):
        #note: this path should be with respect to the daphne path!        
        ast = daphne(['desugar', '-i', '/Users/aliseyfi/Documents/UBC/Semester3/Probabilistic-Programming/HW/Probabilistic-Programming/Assignment_2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('/Users/aliseyfi/Documents/UBC/Semester3/Probabilistic-Programming/HW/Probabilistic-Programming/Assignment_2/programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(ast)
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        try:
            assert(p_val > max_p_value)
        except AssertionError:
            raise AssertionError('wrong answer')
        print(f'Test {i} passed')

    print('All probabilistic tests passed')    

        
if __name__ == '__main__':

    run_deterministic_tests()
    run_probabilistic_tests()


    for i in range(1,5):
        ast = daphne(['desugar', '-i', '/Users/aliseyfi/Documents/UBC/Semester3/Probabilistic-Programming/HW/Probabilistic-Programming/Assignment_2/programs/{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(evaluate_program(ast))