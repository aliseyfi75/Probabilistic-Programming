import torch
import torch.distributions as dist

import copy

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
    for _ in range(num_samples):
        values.append(gibbs_step(graph[1]['P'], unobserved_variables, values[-1], free_variables_inverse))

    sample_temp = deterministic_eval(value_subs(graph[2], values[0]))
    n_params = 1
    if sample_temp.dim() != 0:
        n_params = len(sample_temp)
    samples = torch.zeros(n_params, num_samples+1)

    for idx, value in enumerate(values):
        sample = deterministic_eval(value_subs(graph[2], value))
        samples[:, idx] = sample
    return samples


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
    for selected_variable in unobserved_variables:
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

# Hamiltonian Monte Carlo

def hmc(graph, num_samples=1000, num_leapfrog_steps=10, epsilon=0.1, M=None):
    list_observed_variables, list_unobserved_variables = extract_variables(graph)
    initial_variable_values = sample_from_joint(graph, var=True)

    observed_variables = {}
    unobserved_variables = {}
    for variable in initial_variable_values:
        if variable in list_observed_variables:
            observed_variables[variable] = initial_variable_values[variable]
        else:
            unobserved_variables[variable] = initial_variable_values[variable]
            if not torch.is_tensor(unobserved_variables[variable]):
                unobserved_variables[variable] = torch.tensor(unobserved_variables[variable], dtype=torch.float64)
            else:
                unobserved_variables[variable] = unobserved_variables[variable].type(torch.float64)
            unobserved_variables[variable].requires_grad = True

    if M is None:
        M = torch.eye(len(list_unobserved_variables))
    
    M_inverse = torch.inverse(M)
    P = graph[1]['P']
    samples = []

    normal_generator = torch.distributions.MultivariateNormal(torch.zeros(len(M)), M)
    for _ in range(num_samples):
        r = normal_generator.sample()
        new_unobserved_variables, new_r = leapfrog(P, num_leapfrog_steps, epsilon, copy.deepcopy(unobserved_variables), observed_variables, r)
        u = torch.rand(1)
        current_energy = energy(P, M_inverse, unobserved_variables, observed_variables, r)
        new_energy = energy(P, M_inverse, new_unobserved_variables, observed_variables, new_r)
        if u < torch.exp(current_energy - new_energy):
            unobserved_variables = new_unobserved_variables

        samples.append(unobserved_variables)


    sample_temp = deterministic_eval(value_subs(graph[2], samples[0]))
    n_params = 1
    if sample_temp.dim() != 0:
        n_params = len(sample_temp)
    final_samples = torch.zeros(n_params, num_samples)

    for idx, sample in enumerate(samples):
        final_sample = deterministic_eval(value_subs(graph[2], sample))
        final_samples[:, idx] = final_sample
    
    return final_samples

def energy(P, M_inverse, unobserved_variables, observed_variables, r):
    K = torch.matmul(r, torch.matmul(M_inverse, r)) * 0.5

    U = 0
    for variable in observed_variables:
        U -= deterministic_eval(value_subs(P[variable][1], {**unobserved_variables, **observed_variables})).log_prob(observed_variables[variable])

    return K + U

def leapfrog(P, num_leapfrog_steps, epsilon, unobserved_variables, observed_variables, r):
    r_half = r - 0.5*epsilon*grad_energy(P, unobserved_variables, observed_variables)
    new_unobserved_variables = unobserved_variables
    for _ in range(num_leapfrog_steps):
        new_unobserved_variables = detach_and_add_dict_vector(new_unobserved_variables, epsilon*r_half)
        r_half = r_half - epsilon*grad_energy(P, new_unobserved_variables, observed_variables)
    final_unobserved_variables = detach_and_add_dict_vector(new_unobserved_variables, epsilon*r_half)
    final_r = r_half - 0.5*epsilon*grad_energy(P, final_unobserved_variables, observed_variables)
    return final_unobserved_variables, final_r

def detach_and_add_dict_vector(dictionary, vector):
    new_dictionary = {}
    for i, key in enumerate(list(dictionary.keys())):
        new_dictionary[key] = dictionary[key].detach() + vector[i]
        new_dictionary[key].requires_grad = True
    return new_dictionary

def grad_energy(P, unobserved_variables, observed_variables):
    U = 0
    for variable in observed_variables:
        U -= deterministic_eval(value_subs(P[variable][1], {**unobserved_variables, **observed_variables})).log_prob(observed_variables[variable])
    U.backward()

    U_gradients = torch.zeros(len(unobserved_variables))
    for i, key in enumerate(list(unobserved_variables.keys())):
        U_gradients[i] = unobserved_variables[key].grad
    return U_gradients

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
        graph = daphne(['graph','-i','/Users/aliseyfi/Documents/UBC/Semester3/Probabilistic-Programming/HW/Probabilistic-Programming/Assignment_3/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('/Users/aliseyfi/Documents/UBC/Semester3/Probabilistic-Programming/HW/Probabilistic-Programming/Assignment_3/programs/tests/deterministic/test_{}.truth'.format(i))
        ret = deterministic_eval(graph[-1])
        print(ret)
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
        graph = daphne(['graph', '-i', '/Users/aliseyfi/Documents/UBC/Semester3/Probabilistic-Programming/HW/Probabilistic-Programming/Assignment_3/programs/tests/probabilistic/test_{}.daphne'.format(i)])

        truth = load_truth('/Users/aliseyfi/Documents/UBC/Semester3/Probabilistic-Programming/HW/Probabilistic-Programming/Assignment_3/programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(graph)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')    


        
        
if __name__ == '__main__':
    

    # run_deterministic_tests()
    # run_probabilistic_tests()


    for i in range(1,6):
        graph = daphne(['graph','-i','/Users/aliseyfi/Documents/UBC/Semester3/Probabilistic-Programming/HW/Probabilistic-Programming/Assignment_3//programs/{}.daphne'.format(i)])
        print('\n\n\nSample of posterior of program {}:'.format(i)) 
        # MH within gibbs
        print('\n\n\nMH within Gibbs:')
        samples = mh_within_gibbs_sampling(graph, num_samples=10000)
        samples_mean = torch.mean(samples, dim=1)
        samples_var = torch.var(samples, dim=1)

        print('Mean:', samples_mean)
        print('Var:', samples_var)

        # HMC
        if i<3:
            print('\n\n\nHMC:')
            samples = hmc(graph, num_samples=10000)
            samples_mean = torch.mean(samples, dim=1)
            samples_var = torch.var(samples, dim=1)

            print('Mean:', samples_mean)
            print('Var:', samples_var)

    