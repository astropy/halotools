##Pair-counters Submodule

This submodule contains functions to calculate pair counts.  All of the pair counters 
included here should have the same form of input and output.  Namely, the result of 
passing in two sets of points, and a set of spatial separation "bins" should return the 
number of pairs with separations less than or equal to bins[i].  Each entry in bins 
represents the "edge" of a bin.  To get the true counts in bins, one then simply runs 
np.diff(result).

