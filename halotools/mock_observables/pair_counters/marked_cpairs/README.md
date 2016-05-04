#Brute Force Marked Pair Counters

author : Duncan Campbell

email : duncan.campbell@yale.edu 

##Summary

This module provides functions for "brute force" "weighted" pairwise distance calculations.  

"Brute force", in this case, means that the distance between all pairs is calculated.  

The "weighted" aspect of the functions is very general.  For each pair of points, a vector of weights can be passed to a function that returns a float(s).  

The primary purpose of these functions is to be called by higher level functions that partition samples of points, only calculating pairwise distances for necessary pairs; however, this module can also be used as a standalone module. 


##Organization

All the functions are written in cython in order to optimizate for speed and to make it easy to import the modules into python codes.  Additionally, the functions are written to be as modulular as possible, decreasing the amount of code that must be repeated often.


##Description of Functions

All actual distance caclualtions are done using "cdef" functions defined in "distances.pyx".  All implemented distanances are 3-D cartesian distances, including radial distances (r) and projected/parallel to the line-of-sight (LOS) distances (r_perp, r_para).  Currently, the LOS is defined to be the z-direction (this is sometimes referred to as the distant observer approximation).

The weighting functions are defined in "weighting_functions.pyx" and "pairwise_velocity_funcs.pyx".  The Former accepts a vector of floats associated with two points, and returns a float.  The later, accepts a vector of floats (some being velocities) associated with two points and preforms a pairwise velocity caclulation(s), returning three values.  

Available functions are:

* marked_pair_counting_engines : weighted pair counts in bins

There usually two version of these functions, one assuming non-periodic distances, and another that presumes periodic boundary conditions.  However, the functions that assume non-periodic distances can be used for the periodic case if the points have been shifted to account for the periodic boundaries--this is the current approach of the halotools high level pair counters.

##Defining Custom Weighting Functions

A basic Framework has been set up to incorporate user defined weighting functions.  "custom_weighting_func.pyx" can be edited to provide a custom weighting function (within the framework).  The module must be recomiled before this function will be available for calculations.  This is not a very user friendly set-up, so please contact anyone on the halotools team if you have something in mind that you need help setting up. 

##Details
".pxd" files provide function defintions for the associated functions defined in the ".pyx" file.  The ".pxd" are necessary in order to be able to import the functions into other cython modules.

The "setup_package.py" script compiles the cython modules.  The modules will not be importable unless they have been compiled.





