# Some helpers functions

from ci_test import ci_test
from scipy.io import loadmat
import networkx as nx
from itertools import chain, combinations, permutations
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt


########### TASK 1 HELPER FUNCTIONS #################################

def alpha_tuning(data_matrix, G): #correct
    
    """ 
    The function takes as input a graph G, two vertices x and y
    and outputs whehter x and y are independent given Z
    """
    
    # We decide to pass G in order to use the already implemented function in networkx to
    # check whether the nodes are truly d separated or not
    
    
    alphas = np.arange(0.001,0.8,0.1) 
    indices = list(range(data_matrix.shape[1]))
    
    outputs, true_labels, f1_scores = [], [], []
    
    for alpha in alphas:
    
        for x_index in indices[:-1]:
            
            for y_index in indices[x_index+1:]:
                
                Z = [elem for elem in indices if (elem != x_index and elem != y_index)]
                
                power_set = list(chain.from_iterable(combinations(Z, r) for r in range(len(Z)+1)))
                
                for set_ in power_set:
                
                    output = int(ci_test(data_matrix, x_index, y_index, list(set_), alpha))
                    outputs.append(output)
                    
                    true_label = int(nx.d_separated(G, set([x_index]), set([y_index]), set(set_)))
                    true_labels.append(true_label)
                    
    # We then use the F1 score as a metric to evaluate the performance of the chosen alpha
        f1_scores.append(f1_score(true_labels, outputs))
        
    plt.plot(alphas,f1_scores)
    plt.ylabel('F1 score')
    plt.xlabel('Alpha')
    plt.title('Choice of alpha')
    
def Markov_Boundary(data_matrix, x_index, alpha): #correct (we checked the result theoretically)
    
    # We start with the grow phase
    
    # We initialize the markov boundary to be an empty set
    
    M = [] 
    
    M_start = []
    
    list_of_indices = list(range(data_matrix.shape[1]))
    
    while True:
        
        for y_index in list_of_indices:
            
            if (y_index != x_index) and (y_index not in M):
                
                if not ci_test(data_matrix,x_index,y_index, M, alpha):  # if dependent
                    M.append(y_index)
        
        if M == M_start:
            break
            
        else:
            M.sort()
            M_start = M.copy()
                    
    # It now starts the shrink phase
    
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


def build_moralized_graph(data_matrix, alpha): #correct
    
    # Initializing the graph
    G = nx.Graph()
    
    MBs_dict = {}
    
    list_of_node_ids = list(range(data_matrix.shape[1]))
    
    # Adding nodes to the graph
    for node_id in list_of_node_ids:
        G.add_node(node_id)
        MBs_dict[node_id] = Markov_Boundary(data_matrix, node_id, alpha)
        
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
    y_index should belong to the Markov Boundary of x_index
    """
    
    # Defining Markov boundaries for both nodes
    M_B_x = MBs_dict[x_index]
    M_B_y = MBs_dict[y_index]

    
    # Computing the cardinalities
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
    
    list_of_node_ids = list(G.nodes)
    
    neigh_dict = {}
    sep_set_dict = {}
    
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
    
    neigh_dict, sep_set_dict = build_neighbors_dictionary(G, data_matrix, alpha, MBs_dict)
    
    list_of_nodes = list(G.nodes())
    
    # We now create the new graph (edges directed in both directions)
    new_G = nx.DiGraph() 
    
    for node in list_of_nodes:
        new_G.add_node(node)
    
    for key in neigh_dict.keys():
        for neigh in neigh_dict[key]:
            new_G.add_edge(key, neigh)
            new_G.add_edge(neigh,key)
                       
    
    triplets = [list(elem) for elem in combinations(list_of_nodes, 3)]
    triplets_with_permutations = [list(tripl) for elem in triplets for tripl in permutations(elem)]
    
    
    v_structures = []
    
    for triplet in triplets_with_permutations:
        
        a,b,c = triplet[0], triplet[1], triplet[2]
        
        if ((b,a) in new_G.edges()) and ((b,c) in new_G.edges()) and ((a,c) not in new_G.edges()):
            
            S = sep_set_dict[(a,c)]

            if (S != None) and (b not in S):
                
                if [c,b,a] not in v_structures:
                    
                    v_structures.append([a,b,c])
                    

    for triplet in v_structures:
        
        a,b,c = triplet[0], triplet[1], triplet[2]
        
        if (b,a) in new_G.edges():
            new_G.remove_edge(b,a)
        
        if (b,c) in new_G.edges:
            new_G.remove_edge(b,c)
           
    return new_G


def meek_rule1(G):
    
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
    
    # We build the moralize graph
    G, MBs_dict = build_moralized_graph(data_matrix, alpha)
    
    # We build the graph keeping trace of the v structures
    G_start = second_step_GS(G,data_matrix, alpha, MBs_dict)
    
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
    
    if not isinstance(S,set):
        raise TypeError
    
    result = set()
    
    for node in S:
        
        ancestor_of_node = nx.ancestors(G_bidirected, node)
        result = result.union(ancestor_of_node)
    
    return result.union(S)


def HHull(G_directed, G_bidirected, S):
    
    set_S = set(S)
    F = set(G_directed.nodes())
    
    # directed graph, bidirected graph, subset of vertices
    
    while True:
        
        # Finding F1
        G_F_bidirected = G_bidirected.subgraph(F)       
        connected_components_G_F = [elem for elem in nx.connected_components(G_F_bidirected)]
        indices_and_len = [(index,len(elem)) for index,elem in enumerate(connected_components_G_F) if set_S.issubset(elem)]
        try:
            max_index = max(indices_and_len, key= lambda x: x[1])[0]
        except:
            print('Entering except condition')
            return set_S
        F1 = connected_components_G_F[max_index] # set
        
        # Finding F2
        G_F1_directed = G_directed.subgraph(F1)       
        F2 = ancestor(G_F1_directed, set_S)
        
        if F2 != F:
            
            F = F2
        
        else:
            return F  
        

def hitting(S, set_of_sets):
    
    total_sets = len(set_of_sets)
    cont = 0
    for set_ in set_of_sets:
        if len(S.intersection(set_)) != 0:
            cont+=1
    if cont == total_sets:
        return True
    return False


def WMHS(set_of_sets, costs):
    
    # set of sets is a list of sets
    # costs is a dictionary
    
    solution = set()
    min_cost = sum(costs.values()) # worst case
    
    universal_set = set()
    
    for set_ in set_of_sets:
        universal_set = universal_set.union(set_)
        
    power_set = list(chain.from_iterable(combinations(universal_set, r) for r in range(len(universal_set)+1)))
    
    for set_ in power_set:
        
        candidate_solution = set(set_)
        
        flag = hitting(candidate_solution, set_of_sets)
        
        if flag:
            
            current_cost = 0
            
            for node in candidate_solution:
                current_cost+= costs[node]
            
            if current_cost < min_cost:
                solution = candidate_solution
                min_cost = current_cost
            
    return solution


def MinCostIntervention(S, G_directed, G_bidirected, costs):
    
    F = set()
    
    V = set(G_directed.nodes())
    
    H = HHull(G_directed, G_bidirected,S)
    
    if H == S:
        
        return set()
    
    while True:
        
        while True:
            
            H_minus_S = H - S
            
            # Solving argmin of costs problem           
            argmin = None
            min_cost = sum(costs.values())
            for node in H_minus_S:
                if costs[node] <  min_cost:
                    argmin = node
                    min_cost = costs[node]
            argmin = set(argmin)
            
            H_minus_argmin = H - argmin
            G_directed_H_minus_argmin = G_directed.subgraph(H_minus_argmin)
            G_bidirected_H_minus_argmin = G_bidirected.subgraph(H_minus_argmin) 
            
            new_hull = HHull(G_directed_H_minus_argmin, G_bidirected_H_minus_argmin, S)
            
            if new_hull == S:
                
                F = F.union(H)
            
            else:
                
                H = new_hull
                
        power_set_of_F = list(chain.from_iterable(combinations(F, r) for r in range(len(F)+1)))
        set_of_sets = []
        for elem in power_set_of_F:
            new_elem = set(elem) - S
            if new_elem not in set_of_sets:
                set_of_sets.append(new_elem)
                
        A = WMHS(set_of_sets, costs)
        
        V_minus_A = V - A
        G_directed_V_minus_A = G_directed.subgraph(V_minus_A)
        G_bidirected_V_minus_A = G_bidirected.subgraph(V_minus_A) 
        new_hull = HHull(G_directed_V_minus_A, G_bidirected_V_minus_A, S)
        
        if new_hull == S:
            
            return A
        
        H = new_hull
    
def initialize_cal_H(G_directed, G_bidirected, S):
    pa_S = set()
    S_conn = set()
    
    for node in list(S):
        pa_S = pa_S.union(set(G_directed.predecessors(node)))
        S_conn = S_conn.union(set(G_bidirected.neighbors(node)))
        
    pa_double = pa_S.intersection(S_conn)
    
    V = set(G_directed.nodes())
    
    # initialization
    V_minus_pa_double = V - pa_double
    G_directed_V_minus_pa_double = G_directed.subgraph(V_minus_pa_double)
    G_bidirected_V_minus_pa_double = G_bidirected.subgraph(V_minus_pa_double) 
    
    # Hhull
    H = HHull(G_directed_V_minus_pa_double, G_bidirected_V_minus_pa_double, S)
    
    list_of_nodes = list(H)
    
    pairs_comb = [tuple(elem) for elem in combinations(list_of_nodes, 2)]
    print(pairs_comb)
    
    cal_H = nx.Graph()
    
    for node in list(H.union({'x','y'})):
        cal_H.add_node(node)
        
    for pair in pairs_comb:
        if pair in G_bidirected.edges():
            cal_H.add_edge(pair[0], pair[1])
            
    for node in list(S):
        cal_H.add_edge(node,'y')
        
    for node in list(pa_S.intersection(H)):
        cal_H.add_edge('x',node)
        
    return cal_H




    

    
    
            
            
    
   
        
      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
                    