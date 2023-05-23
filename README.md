# Description of the files.

Project description.pdf:
The detailed description of the project and the tasks.

data1.mat:
This is a data matrix with 5 columns and 500 rows corresponding to the vertices of the graph and the samples, respectively.
The columns 1 through 5 of this matrix correspond to x, y, w, z, and s, respectively (see Figure 2.a of the project description.)

data2.mat:
This Matlab struct object contains two matrices:
D2, which is a 15 by 1000 data matrix,
and A2, which is a 15 by 15 adjacency matrix corresponding to the ground truth DAG from which D2 was generated.

CItest.m:
Matlab function for conditional independence tests. See GrowShrink.m for example usage.

ci_test.py:
Python function for conditional independence tests. See grow_shrink.py for example usage.

GrowShrink.m:
Matlab script showing how to import the datasets and examples of CI test function usage.

grow_shrink.py:
Python script showing how to import the datasets and examples of CI test function usage.
