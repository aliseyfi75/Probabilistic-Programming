import torch

import copy
import time
import numpy as np

from daphne import daphne

from primitives import baseprimitives, distlist
from evaluation_based_sampling import evaluate_program, expectation_calculator

from tests import is_tol, run_prob_test,load_truth

from plot import draw_hists, draw_trace, draw_log_joint, draw_hitmap
import matplotlib.pyplot as plt

# import wandb

# wandb.init(project="HW4", entity="aliseyfi")

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

        energy_diff = current_energy - new_energy
        energy_diff_clip = torch.clip(energy_diff, max=0)
        if u < torch.exp(energy_diff_clip):
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

    return final_samples, samples

def energy(P, M_inverse, unobserved_variables, observed_variables, r):
    K = torch.matmul(r, torch.matmul(M_inverse, r)) * 0.5

    U = 0

    all_variables = {**observed_variables, **unobserved_variables}
    for variable in all_variables:
        U = U - deterministic_eval(value_subs(P[variable][1], {**unobserved_variables, **observed_variables})).log_prob(all_variables[variable])

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


def BBVI_evaluator(order_node, graph, sigma):
    P = graph[1]['P']
    Y = graph[1]['Y']
    Q = sigma['Q']
    G = sigma['G']
    optimizer = sigma['optimizer']
    results = {}

    for node in order_node:
        link_function = P[node][0]

        if link_function == 'sample*':
            d = deterministic_eval(value_subs(P[node][1], results))
            if node not in Q:
                Q[node] = d.make_copy_with_grads()
                optimizer[node] = torch.optim.Adam(Q[node].parameters(), lr=0.01)
            result = Q[node].sample()
            G[node] = grad_log_prob(Q[node], result)
            try:
                sigma_temp = d.log_prob(result) - Q[node].log_prob(result)
                sigma['logW'] += sigma_temp
            except:
                sigma['logW'] += 0
        
        elif link_function == 'observe*':
            result = torch.tensor(Y[node])
            d = deterministic_eval(value_subs(P[node][1], results))
            sigma_temp = d.log_prob(result)
            sigma['logW'] += sigma_temp
        
        results[node] = result
    
    return results, sigma

def grad_log_prob(dist, value):
    log_prob = dist.log_prob(value)
    log_prob.backward()
    #.clone().detach()
    grad = [param.grad for param in dist.parameters()]
    return grad

def BBVI(graph, T, L):
    sigma = {'Q':{}, 'optimizer':{}}
    order_node = topological_sort(graph)
    
    results = []
    log_weights = []
    posteriers = []

    for t in range(T):
        sigma['G'] = {}
        gradients = []
        log_ws = []

        for l in range(L):
            sigma['logW'] = 0
            result, sigma = BBVI_evaluator(order_node, graph, sigma)
            gradients.append(copy.deepcopy(sigma['G']))
            log_ws.append(sigma['logW'])

        if t==0:
            posteriers.append(copy.deepcopy(sigma['Q']['sample2'].parameters()))
        
        ELBO_gradients(gradients, log_ws, sigma['Q'])

        for optimizer in sigma['optimizer'].values():
            optimizer.step()
            optimizer.zero_grad()
        
        post_temp = {}
        for q in sigma['Q']:
            post_temp[q] = sigma['Q'][q].parameters().copy()

        posteriers.append(post_temp)
        result_temp = deterministic_eval(value_subs(graph[2], result))
        results.append(result_temp)
        log_weights.append(log_ws[-1])

        # print_weights = torch.exp(torch.stack(log_weights)).detach().numpy()
        # print_results = torch.stack(results).detach().numpy()

        # print_mean = (print_results * print_weights.reshape(-1,1)).sum(axis=0) / print_weights.sum()
        
        # wandb.log({'ELBO': torch.mean(torch.stack(log_weights)).detach().numpy()})

        # for i in range(len(print_mean)):
        #     wandb.log({'mean_'+str(i): print_mean[i]})


    return results, log_weights, posteriers

def inf_skipper(gradients, log_ws):
    temp_gradients = []
    temp_log_ws = []

    for i in range(len(log_ws)):
        if log_ws[i] == float('-inf'):
            continue
        temp_gradients.append(gradients[i])
        temp_log_ws.append(log_ws[i])
    
    return temp_gradients, temp_log_ws

def ELBO_gradients(gradients, log_ws, posteriors):
    
    gradients, log_ws = inf_skipper(gradients, log_ws)
    len_grads = len(gradients)

    var_union = list(set([var for grad in gradients for var in grad]))
    
    Fs = []
    Gs = []
    stack = {}

    for var in var_union:
        gradient_var = gradients[0][var]
        if len(gradient_var[0].shape) > 0 and len(gradient_var[0]) > 1:
            gradient_var = [grad.clone().detach().requires_grad_(True) for grad in gradient_var[0]]
            stack[var] = len(gradient_var)

        len_vars = len(gradient_var)

        G_var = torch.zeros((len_grads, len_vars))
        F_var = torch.zeros((len_grads, len_vars))

        for lg in range(len_grads):
            G_var[lg, :] = torch.stack(gradients[lg][var])
            F_var[lg, :] = G_var[lg, :] * log_ws[lg]
        Gs.append(G_var.detach().numpy())
        Fs.append(F_var.detach().numpy())
    
    Gs = np.column_stack(Gs)
    Fs = np.column_stack(Fs)
    
    num = np.sum([np.cov(Fs[:, v], Gs[:, v])[0, 1] for v in range(Gs.shape[1])])
    denum = np.sum([np.var(Gs[:, v]) for v in range(Gs.shape[1])])
    b_hat = 0.
    if not denum == 0. and not np.isnan(num):
        b_hat = num/denum

    counter_1 = 0
    for var in var_union:
        gradient_var = gradients[0][var]
        counter_2 = len(gradient_var)
        if var in stack:
            counter_2 = stack[var]
        g_hat = np.array([np.sum(Fs[:, v] - b_hat * Gs[:, v]) / len_grads for v in range(counter_1, counter_1+counter_2)])
        # g_hat = np.array([np.sum(Fs[:, v]) / len_grads for v in range(counter_1, counter_1 + counter_2)])
        if var in stack:
            g_hat = [g_hat]
        for i, parameter in enumerate(posteriors[var].parameters()):
            parameter.grad = torch.tensor(-g_hat[i], dtype=parameter.grad.dtype)
        counter_1 += counter_2
    return
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

    # Task 1
    # graph_1 = daphne(['graph','-i','/Users/aliseyfi/Documents/UBC/Semester3/Probabilistic-Programming/HW/Probabilistic-Programming/Assignment_4/programs/{}.daphne'.format(1)])
    # print('\n\n\nSample of posterior of program {}:'.format(1)) 
    # T_1 = int(5 * 1e3)
    # L_1 = 25
    # start_time_1 = time.time()
    # samples_1, log_weights_1, posteriors_1 = BBVI(graph_1, T_1, L_1)
    # print('Time taken:', time.time() - start_time_1)
    # samples_1 = torch.stack(samples_1).numpy()
    # weights_1 = np.exp(torch.stack(log_weights_1).detach().numpy())
    
    # samples_mean = (samples_1 * weights_1).sum() / weights_1.sum()
    # samples_var =  ((samples_1 **2 - samples_mean ** 2) * weights_1).sum() / weights_1.sum()
    
    # print(samples_mean)
    # print(samples_var)

    # new_samples = torch.distributions.Normal(*posteriors_1[-1]['sample2']).sample((10000,)).view(1,-1)
    # draw_hists('BBVI', new_samples, 1)


    # Task 2
    # graph_2 = daphne(['graph','-i','/Users/aliseyfi/Documents/UBC/Semester3/Probabilistic-Programming/HW/Probabilistic-Programming/Assignment_4/programs/{}.daphne'.format(2)])
    # print('\n\n\nSample of posterior of program {}:'.format(2)) 
    # T_2 = int(5 * 1e3)
    # L_2 = 25
    # start_time_2 = time.time()
    # samples_2, log_weights_2, posteriors_2 = BBVI(graph_2, T_2, L_2)
    # print('Time taken:', time.time() - start_time_2)
    # samples_2 = torch.stack(samples_2).numpy()
    # weights_2 = np.exp(torch.stack(log_weights_2).detach().numpy())
    
    # samples_mean_slope = (samples_2[:,0] * weights_2).sum() / weights_2.sum()

    # samples_mean_bias = (samples_2[:,1] * weights_2).sum() / weights_2.sum()
    
    # print(samples_mean_slope)
    # print(samples_mean_bias)

    # Task 3
    # graph_3 = daphne(['graph','-i','/Users/aliseyfi/Documents/UBC/Semester3/Probabilistic-Programming/HW/Probabilistic-Programming/Assignment_4/programs/{}.daphne'.format(3)])
    # print('\n\n\nSample of posterior of program {}:'.format(3)) 
    # T_3 = int(5 * 1e1)
    # L_3 = 25
    # start_time_3 = time.time()
    # samples_3, log_weights_3, posteriors_3 = BBVI(graph_3, T_3, L_3)
    # print('Time taken:', time.time() - start_time_3)
    # samples_3 = torch.stack(samples_3).numpy()
    # weights_3 = np.exp(torch.stack(log_weights_3).detach().numpy())
    
    # samples_mean = (samples_3 * weights_3.reshape(-1,1)).sum(axis=0) / weights_3.sum()
    
    # print(samples_mean)

    # Task 4
    # graph_4 = daphne(['graph','-i','/Users/aliseyfi/Documents/UBC/Semester3/Probabilistic-Programming/HW/Probabilistic-Programming/Assignment_4/programs/{}.daphne'.format(4)])
    # print('\n\n\nSample of posterior of program {}:'.format(4)) 
    # T_4 = int(5 * 1e0)
    # L_4 = 25
    # start_time_4 = time.time()
    # samples_4, log_weights_4, posteriors_4 = BBVI(graph_4, T_4, L_4)
    # print('Time taken:', time.time() - start_time_4)
    # w_0 = np.zeros((10,1))
    # b_0 = np.zeros((10,1))
    # w_1 = np.zeros((10,10))
    # b_1 = np.zeros((10,1))

    # weights = np.exp(torch.stack(log_weights_4).detach().numpy())
    # for i, sample in enumerate(samples_4):
    #     w_0 += sample[0].numpy() * weights[i]
    #     b_0 += sample[1].numpy() * weights[i]
    #     w_1 += sample[2].numpy() * weights[i]
    #     b_1 += sample[3].numpy() * weights[i]

    # sum_weights = weights.sum()

    # w_0 /= sum_weights
    # b_0 /= sum_weights
    # w_1 /= sum_weights
    # b_1 /= sum_weights

    # draw_hitmap('BBVI', w_0, '4_w0')
    # draw_hitmap('BBVI', b_0, '4_b0')
    # draw_hitmap('BBVI', w_1, '4_w1')
    # draw_hitmap('BBVI', b_1, '4_b1')


    # Task 5
    graph_5 = daphne(['graph','-i','/Users/aliseyfi/Documents/UBC/Semester3/Probabilistic-Programming/HW/Probabilistic-Programming/Assignment_4/programs/{}.daphne'.format(5)])
    print('\n\n\nSample of posterior of program {}:'.format(5))
    T_5 = int(5 * 1e3)
    L_5 = 25
    start_time_5 = time.time()
    samples_5, log_weights_5, posteriors_5 = BBVI(graph_5, T_5, L_5)
    print('Time taken:', time.time() - start_time_5)

    new_samples = torch.distributions.Uniform(*posteriors_5[-1]['sample2']).sample((10000,)).view(1,-1)
    draw_hists('BBVI', new_samples, 5)