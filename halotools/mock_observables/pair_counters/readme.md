##Pair-counters Submodule

This submodule contains functions to calculate pair counts.

###Available Pair-counters
* pairs.py, a brute force pair counter
* cpairs.pyx, a brute force cython pair counter
* kdpairs.py, uses ckdtree for pair counting operations
* mpipairs.py, MPI implementation which uses ckdtree for pair counting operations, and 
requires openmpi library, and mpi4py package to be installed.

###Guidlines
All of the pair counters included here should have the same form of input and output. 
Namely, the result of passing in two sets of points, and a set of spatial separation 
"bin" edges, bins, should return the number of pairs with separations less than or equal to bins.
To get the true counts in bins, one then simply runs np.diff(result). This will be an 
array with length 1-len(bins).

###Notes
pairs.py stores the distances between all points internally. It will use a lot of 
memory for large data-sets. 

