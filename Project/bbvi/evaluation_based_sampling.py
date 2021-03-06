from daphne import daphne
from torch.functional import Tensor
from tests import is_tol, run_prob_test,load_truth
from primitives import standard_env, Symbol, Env
import torch
from plots import plots
import numpy as np

def evaluate_program(ast, sigma = {}):
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """
    for i in range(len(ast)):
        ei, sigma = eval(ast[i],sigma)
        if ei != None:
            res = ei
    return res, sigma

class Procedure(object):
    "A user-defined Scheme procedure."
    def __init__(self, parms, body, sigma, env):
        self.parms, self.body, self.sigma, self.env = parms, body, sigma, env
    def __call__(self, *args): 
        return eval(self.body, self.sigma, Env(self.parms, args, self.env))


def eval(x, sigma, env=standard_env()):
    "Evaluate an expression in an environment."
    if isinstance(x, Symbol):    # variable reference
        return env.find(x)[x], sigma
    elif not isinstance(x, list) and type(x)==Tensor: # constant 
        return x, sigma
    elif not isinstance(x, list):
        return torch.tensor(x), sigma
    op, *args = x       
    if op == 'if':             # conditional
        (test, conseq, alt) = args
        res, sigma = eval(test, sigma, env)
        exp = (conseq if res else alt)
        return eval(exp, sigma, env)
    elif op == 'defn':         # definition
        (string, parms, body) = args
        env[string] = Procedure(parms, body, sigma, env)
        return None, sigma
    elif op == 'let':
        (symbol, exp) = args[0]
        res, sigma = eval(exp, sigma, env)
        env[symbol] = res
        return eval(args[1], sigma, env)
    elif op == 'sample':
        dist, sigma = eval(args[0], sigma, env)
        return dist.sample(), sigma
    elif op == 'observe':
        dist, sigma = eval(args[0], sigma, env)
        c, sigma = eval(args[1], sigma, env)
        sigma['logW'] = sigma['logW'] + dist.log_prob(c)
        return c, sigma
    else:                        # procedure call
        proc, sigma = eval(op, sigma, env)
        # TODO: is this the real sigma we want? Idk. 
        #   maybe we should ignore instead of store. Not sure. 
        vals = [x[0] for x in (eval(arg, sigma, env) for arg in args)]
        if type(proc)==Procedure:   # user defined
            r, _ = proc(*vals)
        else:                       # primitive
            r = proc(*vals)
        # print("done", op, args, r)
        return r, sigma

def get_stream(ast):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(ast)
    
def run_deterministic_tests():
    
    for i in range(1,14):
        #note: this path should be with respect to the daphne path!
        ast = daphne(['desugar', '-i', 'C:/Users/jlovr/CS532-HW3/Inference-algorithms/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('C:/Users/jlovr/CS532-HW3/Inference-algorithms/programs/tests/deterministic/test_{}.truth'.format(i))
        ret, sig = evaluate_program(ast)
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,ast))
        
        print(i, 'Test passed, returned', ret)
        
    print('All deterministic tests passed')
    

def run_probabilistic_tests():
    
    num_samples= 1e4
    max_p_value = 1e-4
    
    for i in range(1,7):
        #note: this path should be with respect to the daphne path!        
        ast = daphne(['desugar', '-i', 'C:/Users/jlovr/CS532-HW3/Inference-algorithms/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('C:/Users/jlovr/CS532-HW3/Inference-algorithms/programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(ast)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        assert(p_val > max_p_value)

        print('Test {} passed,'.format(i), 'p value', p_val)

    print('All probabilistic tests passed')    

        
if __name__ == '__main__':

    # print("deterministic tests \n")
    # run_deterministic_tests()
    # print("\n\n\n")
    # print("probabilistic tests \n")
    # run_probabilistic_tests()
    # print("\n\n\n")

    for i in range(1,6):
        ast = daphne(['desugar', '-i', 'C:/Users/jlovr/CS532-HW3/Inference-algorithms/programs/{}.daphne'.format(i)])

        # samples = []
        # for _ in range(n):
        #     samples.append(evaluate_program(ast)[0])
        # plots(i, samples, "_standard.pdf")

        print('\n\n\nSample of prior of program {}:'.format(i))
        print(evaluate_program(ast)[0], "\n\n\n")  



