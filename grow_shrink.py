from ci_test import ci_test
from scipy.io import loadmat
import networkx as nx


"""
    Import data from data1.mat and data2.mat
"""
D1 = loadmat('data1.mat')['D']
data2 = loadmat('data2.mat')
D2 = data2['D2']
adj2 = data2['A2']  # adjacency matrix of D2 for sanity check
nx.draw(nx.DiGraph(adj2), with_labels=True)

""" CI Testing (D1) 
    Let's see whether x is independent of s given {w,z}:
"""
x = 0
y = 1
w = 2
z = 3
s = 4
alpha = 0.01
print(f'testing with \\alpha = {alpha}')
ci = ci_test(D1, x, s, [w, z], alpha)
if ci:
    print('x and s are independent given {w,z}.')
else:
    print('x and s are not independent given {w,z}.')

# Try another value of alpha:
alpha = 0.2
print(f'\ntesting with \\alpha = {alpha}')
ci = ci_test(D1, x, s, [w, z], alpha)
if ci:
    print('x and s are independent given {w,z}.')
else:
    print('x and s are not independent given {w,z}.')

"""
    D2:
"""
alpha = 0.5
ci = ci_test(D2, 1, 2, [3, 4, 5], alpha)
