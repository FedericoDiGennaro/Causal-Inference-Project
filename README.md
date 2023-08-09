# Causal Inference project
In this repository you can find our code for the mini-project of the course [Causal Inference](https://edu.epfl.ch/coursebook/en/causal-inference-MGT-416).  
In this project we approached two different problems:  
* Implementation of a causal discovery algorithm.  
* Implementation of theoretical results on experimental design for causal effect identification.  

## Team  
Our team is composed by:  
* Brioschi Riccardo: [@RiccardoBrioschi](https://github.com/RiccardoBrioschi)    
* Di Gennaro Federico: [@FedericoDiGennaro](https://github.com/FedericoDiGennaro)    
* Francesco Tripoli  

## Causal Discovery algorithm  
The first part of the project aims at implementing a Causal Discovery Algorithm, i.e. a procedure to recover the causal graphical structure through data. To this end, we implemented the Grow-Shrink (GS) algorithm [1].  
The Grow-Shrink algorithm is a constraint-based methods, thus relying on conditional independence (CI) tests to recover the causal structure of the graph. We used the provided $ci$_$test()$ function in order to perform such tests but we first had to determine the significance level $\alpha$, that is the threshold under which we no longer accept the null hypothesis. To do so, after pairing nodes and defining potential separating set in the DAG provided in the assignment sheet, we evaluated the performance of a set of candidate values for $\alpha$ by comparing the output of the tests to the "ground truth" obtained by checking the d-separations in the true graph. To conduct such performance evaluation we used the F1 score as a metric and then chose the parameter that maximized this quantity, i.e. $\alpha = 0.01$.  

## Experimental Design for Causal Effect Identification  
The second part of the project aims at implementing a procedure that, given a non-identifiable query Q[S], returns the minimum cost set intervening upon which makes the causal query Q[S] identifiable [2].  
The first task of this part was implementing the subroutine $HHull()$ to discover the maximal hedge formed for Q[S] in $\mathcal{G}$. This routine was then called by the main function of this section, namely 
$MinCostIntervention(S, \mathcal{G}, C(\cdot))$. For the whole part 2, we represented every ADMG (\textit{acyclic directed mixed graph}) with two graphs, one directed and one undirected.  
We then implemented the function $WMHS(set$_$of$_$sets, costs)$ that, given a set of sets and a dictionary containing the cost of intervening on each node, returns the minimum cost solution which has non empty intersection (hitting solution) with every set in set\_of\_ sets. For this purpose, after creating a universal set containing all the singletons which are in the sets given as input argument, the solution is found by looping over the power set of such universal set.  
As a last step, by combining this functions together, we were able to implement the $MinCostIntervention(S, \mathcal{G}, C(\cdot))$ function, that outputs the minimum cost intervention that eliminates every hedge formed for Q[S]. If there is no such hedge, it outputs the empty set. Please notice that, in order to make the code robust and general, we implemented two versions of the proposed algorithm, in order to deal with set S whose nodes might or might not be split into different maximal connected components that partition the graph. \\
Finally, we modeled the problem as a linear program problem and we solved it using this approach supported by the bound on the optimality of the solution of this method with the true solution.

## Description of folders and files  
Here you can find a detailed description of what each file in this repository contains.  
|--- `Task1_first_dataset.ipynb`: Main applying causal discovery algorithm to the first dataset of observed data provided.     
|--- `Task1_second_dataset.ipynb`: Main applying causal discovery algorithm to the second dataset of observed data provided.      
|--- `Task2.ipynb`: Main containing our solved second task, i.e. implementation of theoretical results of [2] on Experimental Design for Causal Effect Identification.   
|--- `ci_test.py`: python file with the function responsible to output result of statistical test on independence.  
|--- `data1.mat`: first dataset.  
|--- `data2.mat`: second dataset.  
|--- `helpers.py`: python file for ausiliary functions and sub-procedures needed for both tasks.  
|--- `grow_shrink.py`: debugging file to test grow shrink algorithm.  

## References

#### Main theoretical references: 
[1] [Bayesian Network Induction via Local Neighborhoods, *Dimitris Margaritis, Sebastian Thrun*. NIPS, 1999.](https://papers.nips.cc/paper_files/paper/1999/hash/5d79099fcdf499f12b79770834c0164a-Abstract.html)  
[2] [Minimum Cost Intervention Design for Causal Effect Identification, *Sina Akbari, Jalal Etesami, Negar Kiyavash*. ICML, 2022.](https://arxiv.org/abs/2205.02232)
