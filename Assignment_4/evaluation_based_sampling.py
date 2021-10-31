from daphne import daphne
from plot import draw_hists
from tests import is_tol, run_prob_test,load_truth

import torch

import time

from primitives import baseprimitives, distlist

        
def evaluate_program(ast, sigma={}):
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
                        result, sigma = eval(statement, sigma, {}, funcs)
            if final_ast[0][0] != 'defn':
                result, sigma = eval(final_ast[0], sigma, {}, funcs)
        else:
            result, sigma = eval(ast, sigma, {}, funcs)
    else:
        result, sigma = eval(ast, sigma, {}, funcs)
    if sigma == {}:
        results = result
    else:
        results = [result, sigma]
    return results

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
            result, sigma = eval(x[2], sigma, l, funcs)
        else:
            result, sigma = eval(x[3], sigma, l, funcs)
    elif x[0] == 'let':
        name, exp = x[1]
        result, sigma = eval(exp, sigma, l, funcs)
        l[name]= result
        return eval(x[2], sigma, l, funcs)
    elif x[0] == 'sample':
        dist, sigma = eval(x[1], sigma, l, funcs)
        result = dist.sample()
    elif x[0] == 'observe':
        dist, sigma = eval(x[1], sigma, l, funcs)
        while isinstance(dist, list):
            dist, sigma = eval(dist, sigma, l, funcs)
        result, sigma = eval(x[2], sigma, l, funcs)
        try:
            sigma['logW'] = sigma['logW'] + dist.log_prob(result)
        except:
            pass
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

        elif first_statemnt in funcs:
            _, variables, process = funcs[first_statemnt]
            assignment = {key:value for key, value in zip(variables, other_statements)}
            result, sigma = eval(process, sigma, {**l, **assignment}, funcs)
        else:
            result = torch.tensor(statements)
    return result, sigma
           
def likelihood_weighting(L, exp):
    sigma = {'logW':0}
    results_temp , sigma_temp = evaluate_program(exp, sigma)
    n_params = 1
    if results_temp.dim() != 0:
        n_params = len(results_temp)
    results = torch.zeros((n_params, L))
    weights = []
    for l in range(L):
        sigma = {'logW':0}
        results_temp , sigma_temp = evaluate_program(exp, sigma)
        results[:,l] = results_temp
        weights.append(sigma_temp['logW'])
    return results, torch.tensor(weights)

def expectation_calculator(results, log_weights, func, *args):
    weights = torch.exp(log_weights)
    func_result = func(results, *args)
    return torch.sum(weights*func_result, dim=1) / torch.sum(weights)

def get_stream(ast):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(ast)
    


def run_deterministic_tests():
    
    for i in range(1,14):
        #note: this path should be with respect to the daphne path!
        ast = daphne(['desugar', '-i', '/Users/aliseyfi/Documents/UBC/Semester3/Probabilistic-Programming/HW/Probabilistic-Programming/Assignment_2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('/Users/aliseyfi/Documents/UBC/Semester3/Probabilistic-Programming/HW/Probabilistic-Programming/Assignment_2/programs/tests/deterministic/test_{}.truth'.format(i))
        ret = evaluate_program(ast)
        print(ret)
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

    # run_deterministic_tests()
    # run_probabilistic_tests()

    for i in range(1,6):
        ast = daphne(['desugar', '-i', '/Users/aliseyfi/Documents/UBC/Semester3/Probabilistic-Programming/HW/Probabilistic-Programming/Assignment_3/programs/{}.daphne'.format(i)])
        print('\n\n\nSample of posterior of program {}:'.format(i))
        
        # Importance sampling
        n_samples = int(1e5)
        print('\nImportance sampling:')
        start_time = time.time()
        samples, log_weights = likelihood_weighting(n_samples, ast)
        print('Time taken:', time.time() - start_time)
        samples_mean = expectation_calculator(samples, log_weights, lambda x:x)
        samples_var = expectation_calculator(samples, log_weights, lambda x: x**2 - samples_mean.view(samples.shape[0],1)**2)
        print('Mean:', samples_mean)
        weighted_samples = samples * torch.exp(log_weights) / torch.sum(torch.exp(log_weights))
        weighted_samples_remove_mean = weighted_samples - samples_mean.view(samples.shape[0],1)
        cov_matrix = torch.matmul(weighted_samples_remove_mean, torch.t(weighted_samples_remove_mean)) / (weighted_samples_remove_mean.shape[1] - 1)

        print('Number of samples: ', n_samples)
        print('Mean: ', samples_mean)
        print('Variance: ', samples_var)
        print('Covariance matrix: ', cov_matrix)
        draw_hists("Importance_Sampling", samples, i, weights=torch.exp(log_weights))