#Brute Force Pairwise Distance Calculations

author : Duncan Campbell

email : duncan.campbell@yale.edu 

##Summary

This module provides functions for "brute force" pairwise distance calculations.  

"Brute force", in this case, means that the distance between all pairs is calculated.  

The primary purpose of these functions is to be called by higher level functions that partition samples of points, only calculating pairwise distances for necessary pairs; however, this module can also be used as a standalone module. 


##Organization

All the functions are written in cython in order to optimizate for speed and to make it easy to import the modules into python.  Additionally, the functions are written to be as modulular as possible, decreasing the amount of code that must be repeated often.


##Description of Functions

All actual distance caclualtions are done using "cdef" functions defined in "distances.pyx".  All implemented distanances are 3-D cartesian distances, including radial distances (r) and projected/parallel to the line-of-sight (LOS) distances (r_perp, r_para).  Currently, the LOS is defined to be the z-direction (this is sometimes referred to as the distant observer approximation).

Available functions are:

* pair_counting_engines : pair counts in bins
* pairwise_distances : a matrix of distances between points
* per_object_npairs : pair counts in bins per object

There are usually two version of these functions, one assuming non-periodic distances, and another that presumes periodic boundary conditions.  However, the functions that assume non-periodic distances can be used for the periodic case if the points have been shifted to account for the periodic boundaries--this is the current approach of the halotools higher level pair counting modules.

##Details
".pxd" files provide function defintions for the associated functions defined in the ".pyx" file.  The ".pxd" are necessary in order to be able to import the functions into other cython modules.

The "setup_package.py" script compiles the cython modules.  The modules will not be importable unless they have been compiled.





