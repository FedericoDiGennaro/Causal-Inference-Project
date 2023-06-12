# This python file contains useful functions implementing the main algorithms to use in order to solve both tasks.
# For a better understaing of each function, please refer to the main notebooks, where the functions are used 
# and the results are plotted.

# Importing useful libraries

from ci_test import ci_test
from scipy.io import loadmat
import networkx as nx
from itertools import chain, combinations, permutations
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
import random
from pulp import *


########### TASK 1 HELPER FUNCTIONS #################################

def alpha_tuning(data_matrix, G): 
    
    """ 
    The function takes as input a graph G, a data matrix
    and plots how the performance of the CI test changes with respect to 
    different values of alpha. This function is used to find the best value
    for such hyperparameter.
    """
    # The function receives a graph G, which is the true graph. This is useful in order
    # to have the true labels (conditional independence or dependence) for every pair of nodes.
    
    # Defining a set of candidate values 
    alphas = np.arange(0.001,0.8,0.1) 
    indices = list(range(data_matrix.shape[1]))
    
    outputs, true_labels, f1_scores = [], [], []
    
    for alpha in alphas:
        
        # Looping over (x,y) pairs to run the CI test and compare the results with the true label    
        for x_index in indices[:-1]:
            
            for y_index in indices[x_index+1:]:
                
                Z = [elem for elem in indices if (elem != x_index and elem != y_index)]
                
                # Defining the set of all potential separating sets
                power_set = list(chain.from_iterable(combinations(Z, r) for r in range(len(Z)+1)))
                
                for set_ in power_set:
                
                    output = int(ci_test(data_matrix, x_index, y_index, list(set_), alpha))
                    outputs.append(output)
                    
                    true_label = int(nx.d_separated(G, set([x_index]), set([y_index]), set(set_)))
                    true_labels.append(true_label)
                    
        # We then use the F1 score as a metric to evaluate the performance of the chosen alpha.
        # The F1 score is preferred over the accuracy since the labels are really unbalanced (many False)
        f1_scores.append(f1_score(true_labels, outputs))
        
    plt.plot(alphas,f1_scores)
    plt.ylabel('F1 score')
    plt.xlabel('Alpha')
    plt.title('Choice of alpha')
    
def Markov_Boundary(data_matrix, x_index, alpha): 
    """
    This function returns the Markov Boundary for a given node x_index. Data_matrix and alpha
    are used in order to compute CI tests which are essentials to build such a boundary.
    """
    
    # GROW PHASE: we iteratively add nodes to the boundary
    
    # We initialize the markov boundary to be an empty set
    M = [] 
    
    M_start = []
    
    list_of_indices = list(range(data_matrix.shape[1]))
    
    random.shuffle(list_of_indices)
    
    while True:       
        for y_index in list_of_indices:           
            if (y_index != x_index) and (y_index not in M):      
                if not ci_test(data_matrix,x_index,y_index, M, alpha):  # if Dependent, add y_index to the MB
                    M.append(y_index)
        if M == M_start:
            break    
        else:
            M.sort()
            M_start = M.copy()
                    
    # SHRINK PHASE: we iteratively remove nodes from the boundaries
    
    cont = 0
    
    M.sort()
    
    M_start = M.copy()
    
    while True:
        cont = 0
        for y in M:
            M_y = [elem for elem in M if (elem != y)]
            if ci_test(data_matrix,x_index,y, M_y, alpha):  # if independent
                M.remove(y)
        if M_start == M:
            break
        else:
            M.sort()
            M_start = M.copy()
                
    return M


def build_moralized_graph(data_matrix, alpha): 
    """
    This function builds an undirected graph (nx.Graph()) in which every node is connected to 
    all the nodes in its Markov Boundary.
    """
    
    # Initializing the graph
    G = nx.Graph()
    
    # Inizializing a dictionary contaitning, for every node, its boundary (which will later become the set
    # of its neigbors)
    MBs_dict = {}
    
    list_of_node_ids = list(range(data_matrix.shape[1]))
    
    # Adding entries to the dictionary and nodes to the graph
    for node_id in list_of_node_ids:
        G.add_node(node_id)
        MBs_dict[node_id] = Markov_Boundary(data_matrix, node_id, alpha)
    
    # Before building the graph and adding edges, we must ensure that the Markov Boundaries are symmetric.
    # This result holds theoretically, but it is not granted in practice because of the results of the CI test function.
    # Therefore, whenever y is in the MB of x but x is not in the MB of y, we remove y from the MB of x. Otherwise,
    # we build the edge
    for key in MBs_dict.keys():
        for node in MBs_dict[key]:
            if key in MBs_dict[node]:
                G.add_edge(node, key)
            else:
                MBs_dict[key].remove(node)
    
    return G, MBs_dict

def direct_neighbor(data_matrix, x_index, y_index, alpha, MBs_dict):
    """
    Function to check whether y_index is a neighbor of x_index. Notice that, when the function is called,
    y_index should belong to the Markov Boundary of x_index (the condition is not checked here but must
    be checked before calling the function). Please refer to the next function, where the condition is 
    verified before calling direct_neighbor().
    """
    
    # Defining Markov boundaries for both nodes
    M_B_x = MBs_dict[x_index]
    M_B_y = MBs_dict[y_index]

    
    # Computing the cardinalities in order to later define T
    card_mb_x, card_mb_y = len(M_B_x) - int(y_index in M_B_x), len(M_B_y) - int(x_index in M_B_y)
    
    # Defining T
    if card_mb_x < card_mb_y:
        T = [elem for elem in M_B_x if (elem != y_index)]
    else:
        T = [elem for elem in M_B_y if (elem != x_index)]
        
    # Defining all subsets S in T using the power set implementation provided by python
    power_set_of_T = list(chain.from_iterable(combinations(T, r) for r in range(len(T)+1)))
    
    # Check whether y_index is independent of x_index given every S in the power set defined above
    count = 0
    
    separating_set = None
    
    # As soon as we find a separating set, we save and later return it
    for S in power_set_of_T:
        if ci_test(data_matrix,x_index, y_index, list(S) ,alpha):
            count+= 1
            separating_set = list(S)
            break
        
    
    if count == 0: # dependence is satisfied for every conditioning set S
        # y_index is a true neighbor of x_index
        return True, separating_set
    else:
        return False, separating_set
    
    
def build_neighbors_dictionary(G,data_matrix,alpha,MBs_dict):
    """
    This helper function is useful to initialize two dictionaries containing the neighborhood for each node
    (according to the definition and implementation illustrated in the previous function) and a separating
    set for each pair of nodes (if found).
    """
    
    list_of_node_ids = list(G.nodes)
    
    # Initializing empty dictionaries
    neigh_dict = {}
    sep_set_dict = {}
    
    # Filling the dictionaries
    for node in list_of_node_ids:
        neigh_dict[node] = []
        for second_node in list_of_node_ids:
            if node != second_node:
                sep_set_dict[(node,second_node)] = None
        
    
    for x_node in list_of_node_ids:
        for y_node in MBs_dict[x_node]: # in list(G.neighbors(x_node))
            flag, sep_set = direct_neighbor(data_matrix, x_node, y_node, alpha, MBs_dict)
            if flag:
                neigh_dict[x_node].append(y_node)
            
            sep_set_dict[(x_node,y_node)] = sep_set # None if neighbors
            
    
    return neigh_dict, sep_set_dict
                    
    
def second_step_GS(G, data_matrix, alpha, MBs_dict):
    """
    This function implements the second step of the GS algorithms. We therefore identify the v structures
    in the graph and then we remove some edges.
    """
    
    # Defining useful dictionaries
    neigh_dict, sep_set_dict = build_neighbors_dictionary(G, data_matrix, alpha, MBs_dict)
    
    list_of_nodes = list(G.nodes())
    
    # We now create the new undirected graph (edges directed in both directions)
    new_G = nx.DiGraph() 
    
    for node in list_of_nodes:
        new_G.add_node(node)
    
    for key in neigh_dict.keys():
        for neigh in neigh_dict[key]:
            new_G.add_edge(key, neigh)
            new_G.add_edge(neigh,key)
                       
    # Identifying all the triplets in the graph, also considering the permutations of the nodes.
    triplets = [list(elem) for elem in combinations(list_of_nodes, 3)]
    triplets_with_permutations = [list(tripl) for elem in triplets for tripl in permutations(elem)]
    
    
    v_structures = []
    
    # Identifying v structures in the graph (according to the definition given in the pdf and implemented
    # in the if clause)
    for triplet in triplets_with_permutations:
        
        a,b,c = triplet[0], triplet[1], triplet[2]
        
        if ((b,a) in new_G.edges()) and ((b,c) in new_G.edges()) and ((a,c) not in new_G.edges()):
            
            S = sep_set_dict[(a,c)]

            if (S != None) and (b not in S):
                
                # We avoid saving the same triplets twice
                if [c,b,a] not in v_structures:
                    
                    v_structures.append([a,b,c])

    # We deal with the v structures. Before removing and edge (let's say (a,b)), we verify whether the edge
    # in the opposite direction ((b,a)) is still present. If not, it means we have already intervened on this pair
    # of nodes and we avoid removing and edge, because we would otherwise remove every connection between the chosen nodes.
    for triplet in v_structures:
        
        a,b,c = triplet[0], triplet[1], triplet[2]
        
        flag = ((a,b) in new_G.edges()) and ((c,b) in new_G.edges()) # if True, no problem due to previous removal
                
        if ((b,a) in new_G.edges()) and flag:  
            # if the second edge is not in the graph another triplet has been handled before
            new_G.remove_edge(b,a)
            
        if ((b,c) in new_G.edges) and flag:
            new_G.remove_edge(b,c)
           
    return new_G # v_structures


def meek_rule1(G):
    """
    This function implements the first Meek's Rule.
    """
    
    list_of_nodes = list(G.nodes())
    triplets_comb = [list(elem) for elem in combinations(list_of_nodes, 3)]
    triplets = [list(tripl) for elem in triplets_comb for tripl in permutations(elem)]

    
    for triplet in triplets:
        a, b, c = triplet[0], triplet[1], triplet[2]
        
        if  ( ((a,b) in G.edges()) and ((b,a) not in G.edges()) and ((c,b) in G.edges()) and ((b,c) in G.edges()) ):
            
            #if (c,b) in G.edges():
            G.remove_edge(c,b)
            
    return G


def meek_rule2(G):
    """
    This function implements the second Meek's Rule.
    """
        
    list_of_nodes = list(G.nodes())
    
    triplets_comb = [list(elem) for elem in combinations(list_of_nodes, 3)]
    triplets = [list(tripl) for elem in triplets_comb for tripl in permutations(elem)]
    
    for triplet in triplets:
        
        a, b, c = triplet[0], triplet[1], triplet[2]
        
        edges = list(nx.edges(G))
        
        if  ( ((a,b) in edges) and ((b,a) not in edges) and ((b,c) in edges) and 
             ((c,b) not in edges) and ((a,c) in edges) and ((c,a) in edges) ):
            
            #if (c,a) in list(G.edges):
            G.remove_edge(c,a) 
            
    return G


def meek_rule3(G):
    """
    This function implements the third Meek's Rule.
    """
        
    
    list_of_nodes = list(G.nodes())
    
    quadruplets_comb = [list(elem) for elem in combinations(list_of_nodes, 4)]
    quadruplets = [list(tripl) for elem in quadruplets_comb for tripl in permutations(elem)]
    
    for quad in quadruplets:
        
        a, b, c, d = quad[0], quad[1], quad[2], quad[3]
        
        edges = list(nx.edges(G))
        
        if  ( ((a,b) in edges) and ((b,a) in edges) 
             and ((a,c) in edges) and ((c,a) in edges) and ((a,d) in edges) and ((d,a) in edges) 
             and ((b,c) not in edges) and ((c,b) not in edges) and ((b,d) in edges)
             and ((d,b) not in edges) and ((c,d) in edges) and ((d,c) not in edges) ):
            
            #if (d,a) in list(G.edges):
            G.remove_edge(d,a) 
            
    return G


def meek_rule4(G):
    """
    This function implements the fourth Meek's Rule.
    """
    
    list_of_nodes = list(G.nodes())
    
    quadruplets_comb = [list(elem) for elem in combinations(list_of_nodes, 4)]
    quadruplets = [list(tripl) for elem in quadruplets_comb for tripl in permutations(elem)]
  
    
    for quad in quadruplets:
        
        a, b, c, d = quad[0], quad[1], quad[2], quad[3]
          
        #create a list of edges of the v_structured graph
        edges = list(nx.edges(G))
        
        if  ( ((a,b) in edges) and ((b,a) in edges) and ((a,c) in edges) and ((c,a) in edges)
            and ((a,d) in edges) and ((d,a) in edges) and ((b,c) not in edges) and ((c,b) not in edges)
            and ((b,d) not in edges) and ((d,b) in edges) and ((c,d) in edges) and ((d,c) not in edges) ):
            
            #if (b,a) in list(G.edges):
            G.remove_edge(b,a) 
            
    return G


def meek_orientation(G, data_matrix, alpha):
    """
    This function orients the graph obtained after dealing with potential v structures.
    In order to perform the orientation, the rules implemented above are used. Notice that more than one rule
    might be applied on the graph at the same moment. To better deal with these extreme cases, we
    decided to fix the order in which the rules are checked and potentially applied to 
    modify the graph.
    """
    
    # We build the moralize graph
    G, MBs_dict = build_moralized_graph(data_matrix, alpha)
    
    # We build the graph keeping trace of the v structures
    G_start = second_step_GS(G,data_matrix, alpha, MBs_dict)
    
    # While it is possible, apply the rules. The order in which the rules
    # are first checked and applied is fixed by choice a priori (R1, R2, R3, R4)
    while True:
        
        G_final = meek_rule1(G_start)
        G_final = meek_rule2(G_final)
        G_final = meek_rule3(G_final)
        G_final = meek_rule4(G_final)
        
        if nx.is_isomorphic(G_start, G_final):
            print('Final graph is ready!!')
            break
            
        else:
            G_start = G_final
    
    return G_final

# -------------------------------------------------------------------------------------------------------------------------------


########### TASK 2 HELPER FUNCTIONS #################################

def ancestor(G_directed, S):
    """
    Given a directed graph G and a set of nodes S, this function computes and returns the
    ancestor set of S (defined as the union of the ancestor sets of every node in S).
    """
    
    # Ensuring S is a set
    if not isinstance(S,set):
        raise TypeError
    
    # Initializing the result
    result = set()
    
    # Computing the ancestor set for every node in S and add its elements to results
    for node in S:
        
        ancestor_of_node = nx.ancestors(G_directed, node)
        result = result.union(ancestor_of_node)
    
    return result.union(S) # since by definition we also need to include the nodes of S


def HHull(G_directed, G_bidirected, S):
    """
    This function implements the HHull algorithm. It receives a directed graph, a bidirected graph and 
    a subset of vertices. This function implicitely assumes that S is a c-component (no check is computed inside the function).
    If this assumption is not satisfied, please refer to GeneralMinCostIntervention, which calls the HHull() 
    subroutines on a subset of S that has been proved to be a c-component.
    """
    
    # We convert S to be set
    set_S = set(S)
    
    # We initialize F
    F = set(G_directed.nodes())

    while True:
        
        # Finding F1
        G_F_bidirected = G_bidirected.subgraph(F)       
        connected_components_G_F = [elem for elem in nx.connected_components(G_F_bidirected)]
        
        # Since we are assuming that all the nodes in S are included in the same connected components, we expect
        # indices_and_len to have length equal to one
        
        # Here we exploit the assumption that S is a c-component. Therefore it is enough to keep
        # the maximal connected components which contains all the nodes in S.
        indices_and_len = [(index,len(elem)) for index,elem in enumerate(connected_components_G_F) if set_S.issubset(elem)]
        
        try:
            max_index = max(indices_and_len, key= lambda x: x[1])[0]
        except:
            print('Entering except condition')
            return set_S
        
        # We find the largest component (maximal component) containing S
        F1 = connected_components_G_F[max_index] # set
        
        # Finding F2
        G_F1_directed = G_directed.subgraph(F1)       
        F2 = ancestor(G_F1_directed, set_S)
        
        # Checking the condition to decide whether to continue or return the final result
        if F2 != F:           
            F = F2
        else:
            return F  
        

def hitting(S, set_of_sets):
    """
    This simple function checks whether the input argument S hits the set of sets, i.e. whether it 
    has a non empty intersection with all the sets in the set of sets.
    """
    
    # Counting the total number of elements (sets) contained in set of sets
    total_sets = len(set_of_sets)
    cont = 0
    
    # Looping over the set of sets to count the number of non empty intersections
    for set_ in set_of_sets:
        if len(S.intersection(set_)) != 0:
            cont+=1
            
    # Returning the result of the checking procedure (true if S hits, false otherwise)
    if cont == total_sets:
        return True
    return False


def WMHS(set_of_sets, costs):
    """ 
    This function returns the weighted minimum hitting set. Given a set of sets and a dictionary containing
    the cost of intervening on each node, it returns the minimum cost solution which has non empty intersection
    with every set in set of sets.
    """
    
    # Set of sets is a list of sets
    # Costs is a dictionary having the nodes as keys and 
    
    # Initializing the solution and the min_cost variable (which initially assumes the highest possible value by default)
    solution = set()
    min_cost = sum(costs.values()) # worst case
    
    universal_set = set()
    
    # We perform the union over all sets in set of sets, in order to obtain a unique set containing all the elements
    for set_ in set_of_sets:
        universal_set = universal_set.union(set_)
     
    # We generate the power set of the universal set
    power_set = list(chain.from_iterable(combinations(universal_set, r) for r in range(len(universal_set)+1)))
    
    # We loop over evey set in the power set. Each one of them might be a candidate solution
    for set_ in power_set: 
        candidate_solution = set(set_)
        
        # We start by checking if the current candidate solution hits the set of sets
        flag = hitting(candidate_solution, set_of_sets) 
        
        # If the current candidate hits the sets, then we compute its cost (sum of costs of its nodes) and we
        # compare it to the best solution found so far
        if flag:
            current_cost = 0
            
            # We compute the cost by adding the cost of each node contained in the current candidate
            for node in candidate_solution:
                current_cost+= costs[node]
            
            # We compare the cost of the current solution to the best one found so far. If it's better, we save it
            if current_cost < min_cost:
                solution = candidate_solution
                min_cost = current_cost
            
    return solution


def MinCostIntervention(S, G_directed, G_bidirected, costs):
    """ 
    This function implement the Min Cost intervention algorithm as described in the project description
    and report. The function takes as inputs a set of nodes S (assuming the graph restricted to S is a c-component), 
    an ADMG and a dictionary containing the cost of intervening on each node.
    """
    
    # We initialize useful variables
    F = []
    V = set(G_directed.nodes())
    
    # We run HHull on S. Please notice that we are assuming that G_[S] is a c-component
    H = HHull(G_directed, G_bidirected,S) 
    if H == S: 
        # If we enter this condition, there is no hedge formed for Q[S] and therefore the query
        # is already identifiable
        return set()
    
    while True:
        while True:
            
            # Solving argmin of costs problem              
            H_minus_S = H - S
            argmin = None
            min_cost = sum(costs.values()) + 1000
            for node in H_minus_S:
                if costs[node] <  min_cost:
                    argmin = node
                    min_cost = costs[node]
            argmin = set([argmin])
            
            # Defining a new subset of nodes to restrict our graph using the previous solution
            H_minus_argmin = H - argmin
            
            # Defining new subgraphs
            G_directed_H_minus_argmin = G_directed.subgraph(H_minus_argmin)
            G_bidirected_H_minus_argmin = G_bidirected.subgraph(H_minus_argmin) 
            
            # Running HHull on the new subgraph. Since we run HHull, we are again assuming
            # the nodes of S are contained in the same connected component.
            new_hull = HHull(G_directed_H_minus_argmin, G_bidirected_H_minus_argmin, S)       
            if new_hull == S:
                F.append(H)
                break
            else:
                H = new_hull
        # We now loop over the sets contained in F and we subtract the set S given as input. This is essential
        # in order to define the set of sets taken as input by WMHS
        set_of_sets = []
        for elem in F:
            new_elem = set(elem) - S
            if new_elem not in set_of_sets and new_elem != set():
                set_of_sets.append(new_elem)
                
        # Finding the minimum cost hitting set
        A = WMHS(set_of_sets, costs)
        
        # Defining a new subgraph
        V_minus_A = V - A
        G_directed_V_minus_A = G_directed.subgraph(V_minus_A)
        G_bidirected_V_minus_A = G_bidirected.subgraph(V_minus_A) 
        new_hull = HHull(G_directed_V_minus_A, G_bidirected_V_minus_A, S)
        
        if new_hull == S:
            # We not only return the minimum cost hitting solution, but also F (collection of discovered hedges)
            return A, F
        
        H = new_hull
        

def min_nodes_cut(G,source_set,S,weights):
    """
    This function allows to move from the min weight vertex cut to a max-flow / min-cut problem
    according to the dual theory. Starting from the original undirected graph G, we build a new undirected
    graph in which, for every node z in G, we create a pair of nodes (z_in, z_out) connected through an edge. If
    z belongs to S, then the capacity of such edge is set to infinity (weights[z] otherwise).
    We then recreate all the original connections contained in G, setting the weights to infinity.
    This choice of assigning the weights is essential in order to ensure that, when running the min cut algorithm,
    only edges of the form (z_in, z_out) for z not in S are considered. Therefore, if the final result contains 
    an edge of the form (z_in, z_out), it will imply we have to intervene on z by analogy.
    """
    
    # We initialize an empty undirected graph.
    # The goal is that of building a graph in which removing an edge corresponds to cutting a vertex
    # in the original undirected graph G given as input. This is achieved by the way
    # we assign the weight (capacity) to each edge.
    weighted_edges_graph = nx.DiGraph()

    # We proceed by connecting the arbitrary node x to the nodes in source_set. Since the
    # extremes of each edge are distinct nodes, we set the capacity to infinity
    for node in source_set:
        weighted_edges_graph.add_edge('x',str(node)+'_in',capacity = np.inf)
    
    # For every node in S, we create a pair of nodes (node_in, node_out) connected through an edge.
    # Since we cannot intervene (cut) the nodes in S, we make sure the min cut algorithm will
    # ignore these edges by setting their capacity to infinity.
    for node in S:
        weighted_edges_graph.add_edge(str(node)+'_in',str(node)+'_out',capacity = np.inf)
        weighted_edges_graph.add_edge(str(node)+'_out','y',capacity = np.inf)

    # For every one of the remaining nodes, we create a pair of nodes (node_in, node_out) connected through an edge.
    # Since these remaining nodes are the ones on which we can intervene, we assign them a weight which is
    # equal to the cost of intervening on the vertex.
    for node in G.nodes - S:
        weighted_edges_graph.add_edge(str(node)+'_in',str(node)+'_out',capacity=weights[node])
        # We now reconstruct the connections contained in G, setting the capacity to infinity
        for neighbor in G.adj[node]:
            weighted_edges_graph.add_edge(str(node)+'_out',str(neighbor)+'_in',capacity = np.inf)

    # We solve the min-cut problem using networkx
    cost, cut_sets = nx.minimum_cut(weighted_edges_graph,'x','y')

    # We save the two partitions into two different variables
    setA , setB = cut_sets

    # Exploting the analogy between nodes in G and edges in weighted_edges_graph, we retrieve
    # the nodes on which we should intervene (cut)
    cut_set = set()
    for node in G.nodes - S:
        if (str(node)+'_in' in setA and str(node)+'_out' in setB) or (str(node)+'_in' in setB and str(node)+'_out' in setA):
            cut_set = cut_set.union([node])
            
    return (cost,cut_set)

def heuristic_algorithm(G_directed, G_bidirected, S, costs):    
    """
    This function implements how to solve the min weight vertex cut problem in order to later intervene on 
    the returned solution and identify Q[S].
    """
    
    H = HHull(G_directed, G_bidirected, S)
    
    # Redefining the costs for all the nodes in the graph except for the ones in S
    new_costs = {key:value for key,value in costs.items() if key not in S}
   
    # We now compute pa(S) intersection S, as stated in the project description
    pa_S = set()
    for node in S:
        pa_S = pa_S.union(set(G_directed.predecessors(node)))
    
    pa_inter_H = H & pa_S
    
    # We solve the problem using the previous function
    cost,cut_set = min_nodes_cut(G_bidirected.subgraph(H), pa_inter_H, S, new_costs)
    
    return cut_set


def GeneralMinCostIntervention(S, G_directed, G_bidirected, costs):
    """
    This function implements the Min Cost Intervention algorithm in the general case, i.e. 
    when the nodes of S are split across different connected components of the bidirected graph
    given as input to the function. The idea behind this function is based on Theorem 1 and Lemma 1 provided
    in the project description: a query Q[S] is identifiable if and only if there is no edge
    formed for every partition S_1, S_2, ..., S_k of S. Once all the hedges formed for such partitions are discovered   
    ({Fi,1,...,Fi,mi} for every i = 1,2,...,k), the solution must correspond to the minimum cost hitting solution 
    for the sets {(F1,1\S1),...,(F1,m1 \S1),(F2,1\S2),...,(Fk,mk \Sk)}.
    Therefore, it is enough to identify how to partition S, run the HHull and MinCostIntervention algorithm on each one of these     c-components in order to retrieve the sets {Fi,1,...,Fi,mi} for every i and then find the minimum cost hitting solution.
    """
    
    # We start by computing the connected components of the graph and we check whether all the nodes
    # of S are included in the same components or not. The partitions among the nodes of S will be
    # contained in the structure subsets_of_S (S_1, S_2, ..., S_k)
    connected_components = nx.connected_components(G_bidirected)
    valid_components = [elem for elem in connected_components if len(S.intersection(elem)) != 0 ]
    subsets_of_S = []  
    for component in valid_components:
        temp = []
        for element in component:
            if element in S:
                temp.append(element)
        subsets_of_S.append(set(temp)) # this list contains S_1, S_2, ..., S_k
    
    # Subsets_of_S now contains a partition of the nodes of S which reflects the fact that each of these partitions
    # might be contained in different connected components of the graph given as input
        
    # After computing the partitions among the nodes of S, we need to run the code for every subset as we did
    # before. Our idea is that of running the MinCostIntervention algorithm for every subset, in order to identify 
    # the discovered hedges for each c-component and eventually identify the minimum cost hitting solution when considering
    # all of them.
    sets_of_all_discovered_hedges = [] # This list will contain {(F1,1\S1),...,(F1,m1 \S1),(F2,1\S2),...,(Fk,mk \Sk)
    
    for subset_of_S in subsets_of_S:
        # We run the MinCostIntervention Algorithm only to recover the discovered hedges for the current partition of S
        _, F_of_partition = MinCostIntervention(subset_of_S, G_directed, G_bidirected, costs)
        for hedge in F_of_partition:
            hedge_minus_partition_of_S = hedge - subset_of_S
            if (hedge_minus_partition_of_S not in sets_of_all_discovered_hedges) and (hedge_minus_partition_of_S != set()):
                sets_of_all_discovered_hedges.append(hedge_minus_partition_of_S)
        
    final_result = WMHS(sets_of_all_discovered_hedges, costs)
    
    return final_result
    

def LP_hitting_set(list_of_sets, costs):
    """
    This function returns an approximation of optimal hitting set: 
    by relaxing the integer constraint, we include in the hitting set all decision variables with 
    x(i) > 1/k, where k is the maximum set size.
    """

    # Create single set from union of all sets
    L = len(list_of_sets)
    union_set = set().union(*list_of_sets)
    
    if len(list_of_sets)>1:
        
        # Initialization of the problem
        problem = LpProblem("hitting_set",LpMinimize)
        
        # We now define the decision variable x and we impose the constraint for which it must be positive
        x = LpVariable.dicts("x", union_set, lowBound=0)
        
        # We add to the problem the objective function, which corresponds to the total cost that we want to minimize
        problem += lpSum(costs[elem] * x[elem] for elem in union_set)
        
        # Add constraint of at least a total weight of 1 for each set (in order to have non trivial solutions)
        for set_ in list_of_sets:
            problem += lpSum(x[elem] for elem in set_) >= 1
        
        # Solve the minimization problem calling the method
        problem.solve()
        
        # Add to the hitting set only the decision variables with value greater than 1/k
        k = max(len(s) for s in list_of_sets)

        hitting_set = {elem for elem in union_set if value(x[elem]) >= 1/k}
        return hitting_set

    # If we only have one set, return the element with minimum cost
    elif len(list_of_sets) == 1: 
        return {min(list_of_sets[0],key=lambda x: costs[x])}


def LP_MinCostIntervention(S, G_directed, G_bidirected, costs):
    """
    This function runs an algorithm which is similar to MinCostIntervention
    but using the LP approach implemented above to compute the hitting set.
    """
    # The structure needed for this function is extremely similar to the previously implemented
    # function MinCostIntervention
    # We initialize useful variables
    F = []
    V = set(G_directed.nodes())
    
    # We run HHull on S. Please notice that we are assuming that G_[S] is a c-component
    H = HHull(G_directed, G_bidirected,S) 
    if H == S: 
        # If we enter this condition, there is no hedge formed for Q[S] and therefore the query
        # is already identifiable
        return set()
    
    while True:
        while True:
            
            # Solving argmin of costs problem              
            H_minus_S = H - S
            argmin = None
            min_cost = sum(costs.values()) + 1000
            for node in H_minus_S:
                if costs[node] <  min_cost:
                    argmin = node
                    min_cost = costs[node]
            argmin = set([argmin])
            
            # Defining a new subset of nodes to restrict our graph using the previous solution
            H_minus_argmin = H - argmin
            
            # Defining new subgraphs
            G_directed_H_minus_argmin = G_directed.subgraph(H_minus_argmin)
            G_bidirected_H_minus_argmin = G_bidirected.subgraph(H_minus_argmin) 
            
            # Running HHull on the new subgraph. Since we run HHull, we are again assuming
            # the nodes of S are contained in the same connected component.
            new_hull = HHull(G_directed_H_minus_argmin, G_bidirected_H_minus_argmin, S)       
            if new_hull == S:
                F.append(H)
                break
            else:
                H = new_hull
        # We now loop over the sets contained in F and we subtract the set S given as input. This is essential
        # in order to define the set of sets taken as input by WMHS
        set_of_sets = []
        for elem in F:
            new_elem = set(elem) - S
            if new_elem not in set_of_sets and new_elem != set():
                set_of_sets.append(new_elem)
                
        # Finding the minimum cost hitting set using the Linear Programming approach
        A = LP_hitting_set(set_of_sets, costs)
        
        # Defining a new subgraph
        V_minus_A = V - A
        G_directed_V_minus_A = G_directed.subgraph(V_minus_A)
        G_bidirected_V_minus_A = G_bidirected.subgraph(V_minus_A) 
        new_hull = HHull(G_directed_V_minus_A, G_bidirected_V_minus_A, S)
        
        if new_hull == S:
            # We not only return the minimum cost hitting solution, but also F (collection of discovered hedges)
            return A
        
        H = new_hull
        
    
    
    
    


    

    
    
            
            
    
   
        
      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
                    