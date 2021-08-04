# Pair Counters

This sub-module contains functions to aid and carry out pairwise counting operations.  This is the fundamental backbone of such calculations at the two point correlation function (TPCF).  


## Types of Pair Counting Operations

Halotools provides functions for a variety of pair counting tasks.  These can be split into catagories based on how distances are calculated:

-  3-D Cartesian real space distances
-  (2+1)-D 'redshift' space distances

,and how pairs tabulated:

-  unweighted
-  weighted

For effecient pair counting, compiled `C` "engines" are used.  These engines are called by python functions which process the inputs and return the results.  A set of helper functions assist in splitting the spatial domain:

-  `rectangular_mesh.py`
-  `rectangular_mesh_2d.py`
-  `mesh_helpers.py`


## Un-weighted Pair Counters

Unweighted pair counters tabulate pairs by counting a pair if they meet a certain seperation condition.

The primary un-weighted pair counting functions are:

-  `npairs_3d.py`
-  `npairs_xy_z.py`
-  `npairs_s_mu.py`
-  `npairs_projected.py`

and related functions used for effeciently calculting jackknife covariance samples:

-  `npairs_jackknife_3d.py`
-  `npairs_jackknife_xy_z.py`

Two closely related functions:

-  `pairwise_distance_3d.py` 
-  `pairwise_distance_xy_z.py`

store the pairwise distances between all pairs.  Finally, 

-  `npairs_per_object_3d.py`

stores the total number of pairs found for each object.  

The `./cpairs` directory contains the `Cython` code needed to compile the `C` engines needed by these functions.


## Weighted Pair Counters

Weighted pair counters evaluate a function (or functions), storing the results, if the pair meets a certain seperation condition. 

The primary weighted pair counting functions are:

-  `marked_npairs_3d.py`
-  `marked_npairs_xy_z.py`
-  `weighted_npairs_s_mu.py`

and a specialized set of functions where the pairwise weights depend on the positions of the pairs: 

-  `positional_marked_npairs_3d.py`
-  `positional_marked_npairs_xy_z.py`

The `./marked_cpairs` directory contains the `Cython` code needed to compile the `C` engines needed by these functions.


## Pure Python Pair Counters

To aid in development, a set of "brute force" pure-Python pair counters are provided in `pairs.py`.  These include:

1. `npairs()`, a 3-D real space pair counting function
2. `wnpairs()`, a 3-D real space weighted pair counting function
3. `xy_z_npairs()`, a (2+1)-D space pair counting function
4. `xy_z_wnpairs()`, a (2+1)-D space weighted pair counting function
5. `s_mu_npairs()`, a (2+1)-D space pair counting function 