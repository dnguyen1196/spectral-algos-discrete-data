import numpy as np
from scipy.stats import norm
import scipy.stats as stats
import collections

# Mapping functions for Normal-RUMs

def phi_normal(p, sigma=1):
    return norm.ppf(p, 0, np.sqrt(2) * sigma)

def F_normal(x, sigma=1):
    return norm.cdf(x, 0, np.sqrt(2) * sigma)

def F_prime_normal(x, sigma=1):
    return norm.pdf(x, 0, np.sqrt(2) * sigma)


# Mapping functions for Gumbel-RUMs

def phi_gumbel(p, sigma=1):
    return np.log(p/(1-p)) * sigma

def F_gumbel(x, sigma=1):
    return 1./(1+np.exp(-x/sigma))

def F_prime_gumbel(x, sigma=1):
    return np.exp(-x/sigma) / (sigma * (1 + np.exp(-x/sigma))**2)


# Mapping function for generalized normal
def phi_gennorm(p, sigma=1, beta=1):
    return stats.gennorm.ppf(p, beta, 0, sigma)

def F_gennorm(x, sigma=1, beta=1):
    return stats.gennorm.cdf(x, beta, 0, sigma)

def F_prime_gennorm(x, sigma=1, beta=1):
    return stats.gennorm.pdf(x, beta, 0, sigma)
    

# Mapping function for generalized extreme value distribution
def phi_genextreme(q, sigma=1, c=1):
    return stats.genextreme.ppf(q, c, 0, sigma)

def F_genextreme(x, sigma=1, c=1):
    return stats.genextreme.cdf(x, c, 0, sigma)

def F_prime_genextreme(x, sigma=1, c=1):
    return stats.genextreme.pdf(x, c, 0, sigma)




# Function to check for connectedness of the pairwise matrix

def topological_sort(G: dict, pi_est: list) -> list:
    """
    :param pi_est:
    :param G: dict
    :return:
    """

    def dfs(node: int, visited_nodes: set, unvisited_nodes: list,
            top_order: list):
        visited_nodes.add(node)
        unvisited_nodes.remove(node)
        neighbors = G[node]
        for neighbor in neighbors:
            if neighbor not in visited_nodes:
                dfs(neighbor, visited_nodes, unvisited_nodes, top_order)
        top_order.append(node)

    n = len(pi_est)
    visited = set()
    unvisited = [i for i in pi_est]
    order = []

    while len(visited) < n:
        candidate = list(unvisited)[0]
        dfs(candidate, visited, unvisited, order)

    return list(reversed(order))


def construct_preference_graph(P):
    n = P.shape[0]
    G = dict([(i, list()) for i in range(n)])
    for i in range(n-1):
        for j in range(i+1, n):
            if P[i, j] < 1./2:
                G[j].append(i)
            else:
                G[i].append(j)
    return G


def construct_comparison_graph(P):
    n = P.shape[0]
    G = dict([(i, list()) for i in range(n)])
    for i in range(n-1):
        for j in range(i+1, n):
            if P[i, j] != 0 and P[j, i] != 0:
                G[j].append(i)
                G[i].append(j)
    return G


def check_connectedness_undirected(G):
    def dfs(node: int, visited_nodes: set, unvisited_nodes: list,
            top_order: list):
        visited_nodes.add(node)
        if node in unvisited_nodes:
            unvisited_nodes.remove(node)
        neighbors = G[node]
        for neighbor in neighbors:
            if neighbor not in visited_nodes:
                dfs(neighbor, visited_nodes, unvisited_nodes, top_order)
        top_order.append(node)

    n = len(G.keys())
    visited = set()
    unvisited = [i for i in range(n)]
    order = []
    candidate = list(unvisited)[0]
    dfs(candidate, visited, unvisited, order) # Run DFS

    return len(order) == n

def construct_comparison_graph_from_menus(menus):
    G = collections.defaultdict(list)
    for menu in menus:
        for i in list(menu):
            for j in list(menu):
                if i != j:
                    G[i].append(j)

    return G


def find_connected_components_from_menus(menus):    
    # Construct a graph from the menus
    # Then run multiple DFS to recover all the connected components
    
    vertices = set()
    G = construct_comparison_graph_from_menus(menus)
    visited_nodes = set()
    unvisited_nodes = set(G.keys())
    
    def dfs(node: int, visited_nodes: set, unvisited_nodes: list,
            connected_components: list):
        
        visited_nodes.add(node)
        if node in unvisited_nodes:
            unvisited_nodes.remove(node)
        neighbors = G[node]
        for neighbor in neighbors:
            if neighbor not in visited_nodes:
                dfs(neighbor, visited_nodes, unvisited_nodes, connected_components)
        connected_components.append(node)
    
    connected_components = [] # This should return a list of items that belong in the same connected components, sorted in increasing index order

    while len(unvisited_nodes) > 0:
        connected_comp = []
        dfs(list(unvisited_nodes)[0], visited_nodes, unvisited_nodes, connected_comp)
        connected_components.append(sorted(connected_comp))
    
    return connected_components


