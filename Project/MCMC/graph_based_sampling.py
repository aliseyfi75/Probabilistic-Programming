import torch

import json
import time

from primitives import baseprimitives, distlist
from evaluation_based_sampling import evaluate_program, expectation_calculator

from plot import draw_hists, draw_trace, draw_log_joint
from tqdm import trange, tqdm

PATH = '/home/aliseyfi/scratch/Probabilistic-Programming/Project/'
# PATH = '/Users/aliseyfi/Documents/UBC/Probabilistic-Programming/Probabilistic-Programming/Project/'

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

def sample_from_joint(graph, var=False):
    "This function does ancestral sampling starting from the prior."
    node_order = topological_sort(graph)
    results = {}
    for node in node_order:
        first_statement, *other_statements = graph[1]['P'].get(node)
        if first_statement == 'sample*':
            dist = deterministic_eval(value_subs(other_statements, results))
            result = dist.sample()
        if first_statement == 'observe*':
            result = deterministic_eval(graph[1]['Y'].get(node))
        results[node] = result
    
    if var:
        return results
    else:
        return deterministic_eval(value_subs(graph[2], results))

# MH with Gibbs sampling

def mh_within_gibbs_sampling(graph, num_samples):
    _, unobserved_variables = extract_variables(graph)
    _, free_variables_inverse = extract_free_variables(graph)

    values = [sample_from_joint(graph, var=True)]

    for _ in trange(num_samples):
        values.append(gibbs_step(graph[1]['P'], unobserved_variables, values[-1], free_variables_inverse))

    sample_temp = deterministic_eval(value_subs(graph[2], values[0]))

    n_params = 1
    if sample_temp.dim() != 0:
        n_params = len(sample_temp)
    samples = torch.zeros(n_params, num_samples+1)

    for idx, value in enumerate(tqdm(values)):
        sample = deterministic_eval(value_subs(graph[2], value))
        samples[:, idx] = sample
    return samples, values


def extract_variables(graph):
    observed_variables = []
    for node in graph[1]['V']:
        if graph[1]['P'].get(node)[0] == 'observe*':
            observed_variables.append(node)
    unobserved_variables = [v for v in graph[1]['V'] if v not in observed_variables]
    return observed_variables, unobserved_variables


def extender(l):
    if isinstance(l, list):
        return sum([extender(e) for e in l], [])
    else:
        return [l]

def extract_free_variables(graph):
    free_variables = {}
    for node in graph[1]['V']:
        expressions = extender(graph[1]['P'].get(node)[1])
        for expression in expressions:
            if expression != node:
                if expression in graph[1]['V']:
                    if node in free_variables:
                        free_variables[node].append(expression)
                    else:
                        free_variables[node] = [expression]
    free_var_inverse = {}   
    for node in graph[1]['V']:
        for variable in free_variables:
            if node in free_variables[variable]:
                if node not in free_var_inverse:
                    free_var_inverse[node] = []
                free_var_inverse[node].append(variable)
    return free_variables, free_var_inverse


def gibbs_step(p, unobserved_variables, value, free_var_inverse):
    for selected_variable in tqdm(unobserved_variables):
        q = deterministic_eval(value_subs(p[selected_variable][1], value))
        value_new = value.copy()
        value_new[selected_variable] = q.sample()
        alpha = mh_accept(p, selected_variable, value_new, value, free_var_inverse)
        if alpha > torch.rand(1):
            value = value_new
    return value


def mh_accept(p, selected_variable, value_new, value_old, free_var_inverse):
    q_new = deterministic_eval(value_subs(p[selected_variable][1], value_new))
    q_old = deterministic_eval(value_subs(p[selected_variable][1], value_old))

    log_q_new = q_new.log_prob(value_old[selected_variable])
    log_q_old = q_old.log_prob(value_new[selected_variable])

    log_alpha = log_q_new - log_q_old

    Vx = free_var_inverse[selected_variable] + [selected_variable]
    for v in Vx:
        log_alpha += deterministic_eval(value_subs(p[v][1], value_new)).log_prob(value_new[v])
        log_alpha -= deterministic_eval(value_subs(p[v][1], value_old)).log_prob(value_old[v])
    log_alpha = torch.clip(log_alpha, max=0)
    return torch.exp(log_alpha)


def get_stream(graph):
    """Return a stream of prior samples
    Args: 
        graph: json graph as loaded by daphne wrapper
    Returns: a python iterator with an infinite stream of samples
        """
    while True:
        yield sample_from_joint(graph)
        
        
if __name__ == '__main__':

    with open(PATH+'bbvi/programs/_with_loop.daphne','r') as f:
        graph = json.load(f)
    
    # print('\n\n\nSample of posterior') 
    n_samples = int(1e0)
    # print('\n\n\nMH within Gibbs:')
    start_time = time.time()
    samples, nodes_values = mh_within_gibbs_sampling(graph, num_samples=n_samples)
    print('Time taken: {}'.format(time.time() - start_time))

    samples_mean = expectation_calculator(samples, torch.zeros(samples.shape[1]), lambda x:x)
    samples_var = expectation_calculator(samples, torch.zeros(samples.shape[1]), lambda x: x**2 - samples_mean.view(samples.shape[0],1)**2)

    print('number of samples: ', n_samples)
    print('Mean:', samples_mean)
    print('Var:', samples_var)

    draw_hists("MH_within_Gibbs", samples, 1)

    draw_trace("MH_within_Gibbs", samples, 1)
    draw_log_joint("MH_within_Gibbs", 1, graph, nodes_values, deterministic_eval, value_subs)
        