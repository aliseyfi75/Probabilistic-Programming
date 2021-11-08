import copy
from pyrsistent import pmap
import torch
from torch import distributions as dist
import distributions

class Env(dict):
    "An environment: a dict of {'var': val} pairs, with an outer Env."
    def __init__(self, parms=(), args=(), outer=None):
        try:
            if type(args[0]) == type([]):
                self.update(pmap(zip(parms, args[0])))
            else:
                self.update(pmap(zip(parms, args)))
        except:
            self.update(pmap(zip(parms, args)))
        self.outer = outer
    def find(self, var):
        "Find the innermost Env where var appears."
        return self if (var in self) else self.outer.find(var)

class Procedure(object):
    "A user-defined Scheme procedure."
    def __init__(self, parms, body, env, eval_func):
        self.parms, self.body, self.env, self.eval = parms, body, env, eval_func
    def __call__(self, *args): 
        return self.eval(self.body, Env(self.parms, args, outer=self.env))

class Dist:
    def __init__(self, name, distribution, num_par, *par):
        self.name = name
        self.distribution = distribution
        self.num_par = num_par
        self.pars = []
        for i in range(num_par):
            self.pars.append(par[i])
    
    def sample(self):
        return self.distribution.sample()

    def log_prob(self, x):
        return self.distribution.log_prob(x)

    def parameters(self):
        return self.distribution.Parameters()

    def make_copy_with_grads(self):
        temp_dist = self.distribution
        self.distribution = None
        dist_copy = copy.deepcopy(self)
        self.distribution = temp_dist
        dist_copy.distribution = temp_dist.make_copy_with_grads()
        return dist_copy

class normal(Dist):
    def __init__(self, pars):
        mean = pars[0]
        var = pars[1]
        super().__init__('normal', distributions.Normal(mean, var), 2, mean, var)

class beta(Dist):
    def __init__(self, pars):
        alpha = pars[0]
        betta = pars[1]
        super().__init__('beta', dist.Beta(alpha, betta), 2, alpha, betta)

class exponential(Dist):
    def __init__(self, par):
        lamda = par[0]
        super().__init__('exponential', dist.Exponential(lamda), 1, lamda)

class uniform(Dist):
    def __init__(self, pars):
        a = pars[0]
        b = pars[1]
        super().__init__('uniform', distributions.Uniform(a, b), 2, a, b)

class discrete(Dist):
    def __init__(self, pars):
        prob = pars[0]
        super().__init__('discrete', distributions.Categorical(prob), 0)

class bernoulli(Dist):
    def __init__(self, pars):
        p = pars[0]
        super().__init__('bernoulli', distributions.Bernoulli(p), 1, p)

def push_addr(alpha, value):
    return alpha + value

def vector(x):
    try:
        vector = torch.stack(x)
    except:
        vector = x
    return vector

def list(x):
    try:
        list = torch.stack(x)
    except:
        list = x
    return list

def get(x):
    if type(x[0]) == type(pmap()):
        if torch.is_tensor(x[1]):
            item = x[1].item()
        else:
            item = x[1]
        value = x[0].get(item)
    else:
        value = x[0][x[1].long()]
    return value

def put(x):
    if type(x[0]) == type(pmap()):
        if torch.is_tensor(x[1]):
            item = x[1].item()
        else:
            item = x[1]
        
        x[0] = x[0].set(item, x[2])
    else:
        x[0][x[1].long()] = x[2]
    return x[0]

def hash_map(x):
    keys = x[::2]
    value = x[1::2]
    new_keys = []
    for key in keys:
        if torch.is_tensor(key):
            new_keys.append(key.item())
        else:
            new_keys.append(key)
    result = pmap(zip(new_keys, value))
    return result

def append(x):
    first = x[0]
    second = x[1]

    if type(first) == type([]):
        first = torch.tensor(first)
    elif first.dim() == 0:
        first = first.unsqueeze(0)
    if type(second) == type([]):
        second = torch.tensor(second)
    if second.dim() == 0:
        second = second.unsqueeze(0)
    return torch.cat((first, second))

def conj(x):
    first = x[0]
    second = x[1]

    if type(first) == type([]):
        first = torch.tensor(first)
    elif first.dim() == 0:
        first = first.unsqueeze(0)
    if type(second) == type([]):
        second = torch.tensor(second)
    if second.dim() == 0:
        second = second.unsqueeze(0)
    return torch.cat((first, second))


env = {
        '+': lambda x: x[0] + x[1],
        '-': lambda x: x[0] - x[1],
        '*': lambda x: x[0] * x[1],
        '/': lambda x: x[0] / x[1],
        '>': lambda x: x[0] > x[1],
        '>=': lambda x: x[0] >= x[1],
        '<': lambda x: x[0] < x[1],
        '<=': lambda x: x[0] <= x[1],
        '==': lambda x: x[0] == x[1],
        'sqrt': lambda x: torch.sqrt(x[0]),
        'exp': lambda x: torch.exp(x[0]),
        'log': lambda x: torch.log(x[0]),
        'or': lambda x: x[0] or x[1],
        'and': lambda x: x[0] and x[1],
        'empty?': lambda x: len(x[0]) == 0,
        'vector': vector,
        'list': list,
        'get': get,
        'put': put,
        'hash-map': hash_map,
        'push-address' : push_addr,
        'first': lambda x: x[0][0],
        'last': lambda x: x[0][-1],
        'nth': lambda x: x[0][int(x[1].item())],
        'second': lambda x: x[0][1],
        'rest': lambda x: x[0][1:],
        'peek': lambda x: x[0][-1],
        'append': append,
        # 'cons': lambda x: append([x[1],x[0]]),
        'conj': conj,
        'mat-add': lambda x: x[0] + x[1],
        'mat-mul': lambda x: torch.matmul(x[0], x[1]),
        'mat-transpose': lambda x: x[0].T,
        'mat-tanh': lambda x: x[0].tanh(),
        'mat-repmat': lambda x: x[0].repeat((int(x[1].item()), int(x[2].item()))),
        'normal' : normal,
        'beta' : beta,
        'exponential' : exponential,
        'uniform' : uniform,
        'discrete' : discrete,
        'bernoulli' : bernoulli,
        'uniform-continuous' : uniform,
        'flip' : bernoulli
       }