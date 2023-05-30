# Some helpers functions

from ci_test import ci_test
from scipy.io import loadmat
import networkx as nx
from itertools import chain, combinations, permutations
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt

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
    list_of_indices = list(range(data_matrix.shape[1]))
    cont = 0
    
    while cont + 1 + len(M) < len(list_of_indices):
        cont = 0
        for y_index in list_of_indices:
            if y_index != x_index and y_index not in M:
                if not ci_test(data_matrix,x_index,y_index, M, alpha):  # if dependent
                    M.append(y_index)
                    break      
                else:
                    cont+=1
                    
    # It now starts the shrink phase
    
    cont = 0
    while cont < len(M) :
        cont = 0
        for y_index in M:
            M_y = [elem for elem in M if elem != y_index]           
            if ci_test(data_matrix,x_index,y_index, M_y, alpha):  # if independent
                M.remove(y_index)
                break
            else:
                cont+=1                    
    return M


def build_moralized_graph(data_matrix, alpha): #correct
    
    # Initializing the graph
    G = nx.Graph()
    
    # Adding nodes to the graph
    for node_id in range(data_matrix.shape[1]):
        
        G.add_node(node_id)
    
    list_of_node_ids = np.arange(data_matrix.shape[1])
    
    for node_id in list_of_node_ids:
        
        M_B = Markov_Boundary(data_matrix, node_id, alpha)
        
        for neighbor in M_B:
            
            G.add_edge(node_id, neighbor)
    
    return G

def direct_neighbor(data_matrix, x_index, y_index, alpha):
    """
    Function to check whether y_index is a neighbor of x_index. Notice that, when the function is called,
    y_index should belong to the Markov Boundary of x_index
    """
    
    # Defining Markov boundaries for both nodes
    M_B_x = Markov_Boundary(data_matrix, x_index, alpha)
    M_B_y = Markov_Boundary(data_matrix, y_index, alpha)

    
    # Computing the cardinalities
    card_mb_x, card_mb_y = len(M_B_x) - int(y_index in M_B_x), len(M_B_y) - int(x_index in M_B_y)
    
    # Defining T
    if card_mb_x < card_mb_y:
        T = [elem for elem in M_B_x if elem != y_index]
    else:
        T = [elem for elem in M_B_y if elem != x_index]
        
    # Defining all subsets S in T using the power set implementation provided by python
    power_set_of_T = list(chain.from_iterable(combinations(T, r) for r in range(len(T)+1)))
    
    # Check whether y_index is independent of x_index given every S in the power set defined above
    count = 0
    
    separating_set = []
    
    for S in power_set_of_T:
        if ci_test(data_matrix,x_index, y_index, list(S) ,alpha):
            count+= 1
            break
        
    
    if count == 0: # dependence is satisfied for every conditioning set S
        # y_index is a true neighbor of x_index
        return True
    else:
        return False
    
    
def build_neighbors_dictionary(G,data_matrix,alpha):
    
    list_of_node_ids = list(G.nodes)
    
    neigh_dict = {}
    
    for node in list_of_node_ids:
        neigh_dict[node] = []
        
    
    for x_node in list_of_node_ids:
        M_B_x = Markov_Boundary(data_matrix, x_node, alpha)
        for y_node in M_B_x:
            if direct_neighbor(data_matrix, x_node, y_node, alpha):
                neigh_dict[x_node].append(y_node)
    
    return neigh_dict
                    
    
def second_step_GS(G,data_matrix, alpha):
    
    neigh_dict = build_neighbors_dictionary(G,data_matrix,alpha)
    
    list_of_nodes = list(G.nodes())
    
    # We now create the new graph
    new_G = nx.DiGraph() 
    for key in neigh_dict.keys():
        for neigh in neigh_dict[key]:
            new_G.add_edge(key, neigh)
            new_G.add_edge(neigh,key)
            
    
    triplets = [list(elem) for elem in combinations(list_of_nodes, 3)]
    
    triplets_with_permutations = [list(tripl) for elem in triplets for tripl in permutations(elem)]
    
    v_structures = []
    
    for triplet in triplets_with_permutations:
        
        a,b,c = triplet[0], triplet[1], triplet[2]
        
        if (b in neigh_dict[a]) and (b in neigh_dict[c]) and ((a not in neigh_dict[c]) and (c not in neigh_dict[a])):
            
            remaining_indices = [elem for elem in list_of_nodes if (elem != a) and (elem != c) and (elem!=b)]
            
            power_set = list(chain.from_iterable(combinations(remaining_indices, r) for r in range(len(remaining_indices)+1)))
            
            for S in power_set:
                
                if ci_test(data_matrix, a,c,list(S),alpha) and ((c,b,a) not in v_structures):
                    v_structures.append((a,b,c))
                    break

    for triplet in v_structures:
        
        a,b,c = triplet[0], triplet[1], triplet[2]
        
        if (b,a) in new_G.edges():
            
            new_G.remove_edge(b,a)
        
        if (c,a) in new_G.edges:
            
            new_G.remove_edge(c,a)
           
        
    return new_G


def meek_orientation(data_matrix, alpha):
    
    # We build the moralize graph
    G = build_moralized_graph(data_matrix, alpha)
    # We build the graph keeping trace of the v structures
    v_structure_G = second_step_GS(G,data_matrix, alpha)
            
                    
    
    
    
            
            
    
   
        
      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
                    