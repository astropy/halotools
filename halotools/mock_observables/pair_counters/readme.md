##Pair-counters Submodule

This submodule contains functions to calculate pair counts.

###Available Pair-counters
* pairs.py 
* cpairs.pyx
* kdpairs.py
* mpipairs.py

###Guidlines
All of the pair counters included here should have the same form of input and output. 
Namely, the result of passing in two sets of points, and a set of spatial separation 
"bin" edges should return the number of pairs with separations less than or equal to bins.  
To get the true counts in bins, one then simply runs np.diff(result).

###Notes
pairs.py stores the distances between all points internally. It will use a lot of 
memory for large data-sets. 

