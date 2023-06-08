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


########### TASK 1 HELPER FUNCTIONS #################################

def alpha_tuning(data_matrix, G): 
    
    """ 
    The function takes as input a graph G, two vertices x and y
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
    are used in order to compute CI tests which are essentials to build sucha a boundary.
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
    all the node in its Markov Boundary.
    """
    
    # Initializing the graph
    G = nx.Graph()
    
    # Inizializing a dictionary contaitning, for every node, its boundary (which will later become the set
    # of its neigbors)
    MBs_dict = {}
    
    list_of_node_ids = list(range(data_matrix.shape[1]))
    
    # AAdding entries to the dictionary and nodes to the graph
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
    be checked before calling the function).
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
    set for each pair of nodes.
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
    This function implements the second step of the GS algorithms. We therefore identify the v structure
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
    # in the opposite directino ((b,a)) is still present. If not, it means we have already intervened on this pair
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
    In order to perform the orientation, the rules implemented above are used.
    """
    
    # We build the moralize graph
    G, MBs_dict = build_moralized_graph(data_matrix, alpha)
    
    # We build the graph keeping trace of the v structures
    G_start = second_step_GS(G,data_matrix, alpha, MBs_dict)
    
    # While it is possible, apply the rules. The order is fixed by choice
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

def ancestor(G_bidirected, S):
    """
    Given a graph G and a set of nodes S, this function computes and returns the
    ancestor set of S (defined as the union of the ancestor sets of every node in S).
    """
    
    # Ensuring S is a set
    if not isinstance(S,set):
        raise TypeError
    
    # Initializing the result
    result = set()
    
    # Computing the ancestor set for every node in S and add its elements to results
    for node in S:
        
        ancestor_of_node = nx.ancestors(G_bidirected, node)
        result = result.union(ancestor_of_node)
    
    return result.union(S)


def HHull(G_directed, G_bidirected, S):
    """
    This function implements the HHull algorithm. It receive a directed graph, a bidirected graph and 
    a subset of vertices. This function implicitely assumes that all the nodes of S are contained in 
    the same connected components (connected components are specific partitions of the graph we obtain
    when running nx.connected_component on the graph). If this assumption is not satisfied, please refer
    to GeneralMinCostIntervention.
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
        # indices_and_len to have length equal to one. If not, we raise an error
        indices_and_len = [(index,len(elem)) for index,elem in enumerate(connected_components_G_F) if set_S.issubset(elem)]
        
        # We verify and check the assumption. If it is not satisfied, we raise an error. If the nodes of S are split
        # across different partitions, then indices_and_len will be empty and have length equal to 0.
        assert(len(indices_and_len) == 1)
        
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
    This simple function checks whether S hits the set of sets, i.e. whether it 
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
    This function return the weighted minimum hitting set. Given a set of sets and a dictionary containing
    the cost of intervening on each node, it returns the minimum cost solution which has non empty intersection
    with every set in set of sets.
    """
    
    # Set of sets is a list of sets
    # Costs is a dictionary having the nodes as keys and 
    
    # Initializing the solution and the min_cost variable (which is set to assume the highes possible value by default)
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
    and report. The function takes as inputs a set of nodes S (assuming all its nodes are contained in the same
    connected components), an ADMG and a dictionary containing the cost of intervening on each node.
    """
    
    # We initialize useful variables
    F = []
    V = set(G_directed.nodes())
    
    # We run HHull on S. Please notice that we are assuming that all the nodes in S
    H = HHull(G_directed, G_bidirected,S) 
    if H == S: 
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
            return A
        
        H = new_hull
        
        
def GeneralMinCostIntervention(S, G_directed, G_bidirected, costs):
    """
    This function implements the Min Cost Intervention algorithm in the general case, i.e. 
    when the nodes of S are split across different connected components of the bidirected graph
    given as input to the function.
    """
    
    # We start by computing the connected components of the graph and we check whether all the nodes
    # of S are included in the same components or not. The partitions among the nodes of S will be
    # contained in the structure subsets_of_S
    connected_components = nx.connected_components(G_bidirected)
    valid_components = [elem for elem in connected_components if len(S.intersection(elem)) != 0 ]
    subsets_of_S = []  
    for component in valid_components:
        temp = []
        for element in component:
            if element in S:
                temp.append(element)
        subsets_of_S.append(set(temp))
    
    # Subsets_of_S now contains a partition of the nodes of S which reflects the fact that each of these partitions
    # are contained in different connected components of the graph given as input
        
    # After computing the partitions among the nodes of S, we need to run the code for every subset as we did
    # before. Our idea is that of running the algorithm for every subset, in order to later compute the union 
    # of the results 
    final_result = set()
    for subset_of_S in subsets_of_S:
        temp_sol = MinCostIntervention(subset_of_S, G_directed, G_bidirected, costs)
        final_result = final_result.union(temp_sol)
    
    return final_result
    


def min_nodes_cut(G,source_set,S,weights):
    """
    This function allows to move from the min weight vertex cut to a max-flow / min-cut problem
    according to the dual theory.
    """
    
    # We initialize an empty undirected graph
    weighted_edges_graph = nx.DiGraph()

    # We proceed by adding nodes and edges to the graph. The set of node is given as input.
    # Moreover, we add the capacity to each edge
    for node in source_set:
        weighted_edges_graph.add_edge('x',str(node)+'_in',capacity = np.inf)
    
    for node in S:
        weighted_edges_graph.add_edge(str(node)+'_in',str(node)+'_out',capacity = np.inf)
        weighted_edges_graph.add_edge(str(node)+'_out','y',capacity = np.inf)

    for node in G.nodes - S:
        weighted_edges_graph.add_edge(str(node)+'_in',str(node)+'_out',capacity=weights[node])
        for neighbor in G.adj[node]:
            weighted_edges_graph.add_edge(str(node)+'_out',str(neighbor)+'_in',capacity = np.inf)

    # We solve the min-cut problem using networkx
    cost, cut_sets = nx.minimum_cut(weighted_edges_graph,'x','y')

    # We save the two partitions into two different variables
    setA , setB = cut_sets

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
    cost, cut_set = min_nodes_cut(G_bidirected.subgraph(H), pa_inter_H, S, new_costs)
    
    return cut_set
    
    
    
    
    
    


    

    
    
            
            
    
   
        
      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
                    