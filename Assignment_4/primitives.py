import torch
from torch import distributions
from torch.distributions.bernoulli import Bernoulli
import copy
import distributions as dist

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

    def log_prob(self, c):
        return self.distribution.log_prob(c)

    def parameters(self):
        return self.dist.Parameters()

    def make_copy_with_grads(self):
        temp_dist = self.dist
        self.dist = None
        dist_copy = copy.deepcopy(self)
        self.dist = temp_dist
        dist_copy.dist = temp_dist.make_copy_with_grads()
        return dist_copy

class normal(Dist):
    def __init__(self, pars):
        mean = pars[0]
        var = pars[1]
        normal_dist = dist.Normal(mean, var)
        super().__init__('normal', normal_dist, 2, mean, var)

class beta(Dist):
    def __init__(self, pars):
        alpha = pars[0]
        betta = pars[1]
        super().__init__('beta', distributions.Beta(alpha, betta), 2, alpha, betta)

class exponential(Dist):
    def __init__(self, par):
        lamda = par[0]
        super().__init__('exponential', distributions.Exponential(lamda), 1, lamda)

class uniform(Dist):
    def __init__(self, pars):
        a, b = pars[0], pars[1]
        super().__init__('uniform', distributions.Uniform(a, b), 2, a, b)

class discrete(Dist):
    def __init__(self, pars):
        prob = pars[0]
        discrete_dist = dist.Categorical(prob)
        super().__init__('discrete', discrete_dist, 0)

class bernoulli(Dist):
    def __init__(self, pars):
        p = pars[0]
        bernoulli_dist = dist.Bernoulli(p)
        super().__init__('bernoulli', bernoulli_dist, 1, p)

class gamma(Dist):
    def __init__(self, pars):
        alpha, betta = pars[0], pars[1]
        gamma_dist = dist.Gamma(alpha, betta)
        super().__init__('gamma', gamma_dist, 2, alpha, betta)

class dirichlet(Dist):
    def __init__(self, pars):
        dirichlet_dist = dist.Dirichlet(pars)
        super().__init__('dirichlet', dirichlet_dist, len(pars), *pars)

class dirac(Dist):
    def __init__(self, value):
        mean = value[0]
        mean = torch.clip(mean, -1e5, 1e5)
        var = torch.tensor(1e-5)
        super().__init__('normal', distributions.Normal(mean, var), 2, mean, var)


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
    if type(x[0]) == dict:
        value = x[0][x[1].item()]
    else:
        value = x[0][x[1].long()]
    return value

def put(x):
    if type(x[0]) == dict:
        x[0][x[1].item()] = x[2]
    else:
        x[0][x[1].long()] = x[2]
    return x[0]

def hash_map(x):
    keys = x[::2]
    value = x[1::2]
    new_keys = []
    for key in keys:
        try:
            new_keys.append(key.item())
        except:
            new_keys.append(key)
    result = dict(zip(new_keys, value))
    return result

def append(x):
    first = x[0]
    second = x[1]

    if first == 'vector':
        first = torch.tensor([])
    elif first.dim() == 0:
        first = first.unsqueeze(0)
    if second == 'vector':
        second = torch.tensor([])
    if second.dim() == 0:
        second = second.unsqueeze(0)
    return torch.cat((first, second))


baseprimitives = {
    '+': lambda x: x[0] + x[1],
    '-': lambda x: x[0] - x[1],
    '*': lambda x: x[0] * x[1],
    '/': lambda x: x[0] / x[1],
    '>': lambda x: x[0] > x[1],
    '>=': lambda x: x[0] >= x[1],
    '<': lambda x: x[0] < x[1],
    '<=': lambda x: x[0] <= x[1],
    '==': lambda x: x[0] == x[1],
    '=': lambda x: torch.tensor([1]) if x[0] == x[1] else torch.tensor([0]),
    'and': lambda x: x[0] and x[1],
    'or': lambda x: x[0] or x[1],
    'sqrt': lambda x: torch.sqrt(x[0]),
    'exp': lambda x: torch.exp(x[0]),
    'log': lambda x: torch.log(x[0]),
    'vector': vector,
    'list': list,
    'get': get,
    'put': put,
    'hash-map': hash_map,
    'first': lambda x: x[0][0],
    'last': lambda x: x[0][-1],
    'nth': lambda x: x[0][int(x[1].item())],
    'second': lambda x: x[0][1],
    'rest': lambda x: x[0][1:],
    'append': append,
    'cons': lambda x: append([x[1],x[0]]),
    'conj': append,
    'mat-add': lambda x: x[0] + x[1],
    'mat-mul': lambda x: torch.matmul(x[0], x[1]),
    'mat-transpose': lambda x: x[0].T,
    'mat-tanh': lambda x: x[0].tanh(),
    'mat-repmat': lambda x: x[0].repeat((int(x[1].item()), int(x[2].item())))
}

distlist = {
    'normal' : normal,
    'beta' : beta,
    'exponential' : exponential,
    'uniform' : uniform,
    'discrete' : discrete,
    'bernoulli': bernoulli,
    'gamma': gamma,
    'dirichlet': dirichlet,
    'flip': bernoulli,
    'dirac': dirac
}