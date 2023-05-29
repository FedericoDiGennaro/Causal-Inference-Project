# Some helpers functions

from ci_test import ci_test
from scipy.io import loadmat
import networkx as nx
from itertools import chain, combinations
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt

def alpha_tuning(data_matrix, G):
    
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
    
def Markov_Boundary(data_matrix, x_index, alpha):
    
    # We start with the grow phase
    
    # We initialize the markov boundary to be an empty set
    
    M = []
    
    list_of_indices = [0,1,2,3,4]
    
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


def build_graph(data_matrix, alpha):
    
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
    
                    