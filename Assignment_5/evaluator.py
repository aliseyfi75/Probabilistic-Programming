from torch.distributions import distribution
from torchvision.models.detection import retinanet
from primitives import env as penv, Env, Procedure
from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
from pyrsistent import pmap,plist
import torch

Symbol = str              # A Scheme Symbol is implemented as a Python str
Number = (int, float)     # A Scheme Number is implemented as a Python int or float
Atom   = (Symbol, Number) # A Scheme Atom is a Symbol or Number
List   = list             # A Scheme List is implemented as a Python list
Exp    = (Atom, List)     # A Scheme expression is an Atom or List


def standard_env():
    "An environment with some Scheme standard procedures."
    env = Env()
    env.update(penv)
    env.update({'alpha' : ''}) 
    return env



def evaluate(exp, env=None): #TODO: add sigma, or something

    if env is None or len(env) == 0:
        env = standard_env()
    result = eval(exp[2], env=env)

    if type(result) == type(pmap()):
        result = dict(result)
    return result

def eval(exp, env=None):
    "Evaluate an expression in an environment."
    if isinstance(exp, Symbol):    
        if exp.startswith('"') and exp.endswith('"'):
            result = exp
        else:
            result = env.find(exp)[exp]
    
    elif not isinstance(exp, List):  
        if isinstance(exp, int) or isinstance(exp, float):
            result = torch.tensor(float(exp))
        else:
            result = exp
    else:
        operation, *args = exp
        if operation == 'if':
            (condition, true_exp, false_exp) = args
            if eval(condition, env):
                result = eval(true_exp, env)
            else:
                result = eval(false_exp, env)
        
        elif operation == 'define':
            (name, value) = args
            env[name] = eval(value, env)
            result = None
        
        elif operation == 'fn':
            (params, body) = args
            result = Procedure(params[1:], body, env, eval)

        elif operation == 'set!':
            (name, value) = args
            env.find(name)[name] = eval(value, env)
            result = None

        elif operation == 'sample':
            alpha = eval(args[0], env)
            dist = eval(args[1], env)
            result = dist.sample()
        
        elif operation == 'observe':
            alpha = eval(args[0], env)
            dist = eval(args[1], env)
            observation = eval(args[2], env)
            result = observation

        elif operation == 'push-address':
            result = None
        
        else:
            proc = eval(operation, env)
            alpha = eval(args[0], env)
            vars = [eval(arg, env) for arg in args[1:]]
            result = proc(vars)

    return result

    
def get_stream(exp):
    while True:
        yield evaluate(exp)


def run_deterministic_tests():
    
    for i in range(1,14):

        exp = daphne(['desugar-hoppl', '-i', '/Users/aliseyfi/Documents/UBC/Semester3/Probabilistic-Programming/HW/Probabilistic-Programming/Assignment_5/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('/Users/aliseyfi/Documents/UBC/Semester3/Probabilistic-Programming/HW/Probabilistic-Programming/Assignment_5/programs/tests/deterministic/test_{}.truth'.format(i))
        ret = evaluate(exp)
        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))
        
        print('FOPPL Tests passed')
        
    for i in range(1,13):

        exp = daphne(['desugar-hoppl', '-i', '/Users/aliseyfi/Documents/UBC/Semester3/Probabilistic-Programming/HW/Probabilistic-Programming/Assignment_5/programs/tests/hoppl-deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('/Users/aliseyfi/Documents/UBC/Semester3/Probabilistic-Programming/HW/Probabilistic-Programming/Assignment_5/programs/tests/hoppl-deterministic/test_{}.truth'.format(i))
        ret = evaluate(exp)
        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))
        
        print('Test passed')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    num_samples=1e4
    max_p_value = 1e-2
    
    for i in range(1,7):
        exp = daphne(['desugar-hoppl', '-i', '/Users/aliseyfi/Documents/UBC/Semester3/Probabilistic-Programming/HW/Probabilistic-Programming/Assignment_5/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('/Users/aliseyfi/Documents/UBC/Semester3/Probabilistic-Programming/HW/Probabilistic-Programming/Assignment_5/programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(exp)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')    



if __name__ == '__main__':
    
    # run_deterministic_tests()
    # run_probabilistic_tests()
    

    for i in range(1,4):
        print(i)
        exp = daphne(['desugar-hoppl', '-i', '/Users/aliseyfi/Documents/UBC/Semester3/Probabilistic-Programming/HW/Probabilistic-Programming/Assignment_5/programs/{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(evaluate(exp))        
