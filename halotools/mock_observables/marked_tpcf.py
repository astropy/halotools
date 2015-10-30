# -*- coding: utf-8 -*-

"""
Calculate the marked two point correlation function.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
####import modules########################################################################
import sys
import numpy as np
from math import pi, gamma
from .clustering_helpers import *
from .pair_counters.marked_double_tree_pairs import marked_npairs as obj_wnpairs
from .pair_counters.double_tree_pairs import npairs
##########################################################################################


__all__=['marked_tpcf']
__author__ = ['Duncan Campbell']

np.seterr(divide='ignore', invalid='ignore') #ignore divide by zero in e.g. DD/RR

def marked_tpcf(sample1, rbins, sample2=None, marks1=None, marks2=None,\
                period=None, do_auto=True, do_cross=True, num_threads=1,\
                max_sample_size=int(1e6), aux1=None, aux2=None, wfunc=1):
    """ 
    Calculate the real space marked two-point correlation function, :math:`\\mathcal{M}(r)`.
    
    Parameters 
    ----------
    sample1 : array_like
        Npts x 3 numpy array containing 3-D positions of points.
    
    rbins : array_like
        array of boundaries defining the real space radial bins in which pairs are 
        counted.
    
    sample2 : array_like, optional
        Npts x 3 array containing 3-D positions of points.
    
    marks1: array_like, optional
        length N1 array containing weights used for weighted pair counts.
        deafult is an array one ones.
        
    marks2: array_like, optional
        length N2 array containing weights used for weighted pair counts.
        deafult is an array one ones.
    
    period : array_like, optional
        length 3 array defining axis-aligned periodic boundary conditions. If only
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*3).
        If none, PBCs are set to infinity.
    
    do_auto : boolean, optional
        do auto-correlation?
    
    do_cross : boolean, optional
        do cross-correlation?
    
    num_threads : int, optional
        number of threads to use in calculation. Default is 1. A string 'max' may be used
        to indicate that the pair counters should use all available cores on the machine.
    
    max_sample_size : int, optional
        Defines maximum size of the sample that will be passed to the pair counter. 
        
        If sample size exeeds max_sample_size, the sample will be randomly down-sampled
        such that the subsample is equal to max_sample_size.
    
    aux1: array_like, optional
        length N1 array containing auxiallary weights used for weighted pair counts. 
        deafult is an array one ones.
        
    aux2: array_like, optional
        length N2 array containing auxiallary weights used for weighted pair counts.
        deafult is an array one ones.
    
    wfun: int, optional
        integer indicating which marking function should be used.  See notes for an 
        explanation.  default is 1.
    
    Returns 
    -------
    marked_correlation_function : numpy.array
        len(`rbins`)-1 length array containing the correlation function 
        :math:`\\mathcal{M}(r)` 
        computed in each of the bins defined by input `rbins`.
        
        :math:`1 + \\mathcal{M}(r) \\equiv \\mathrm{WW} / \\mathrm{RR}`, where  
        :math:`\\mathrm{WW}` is the weighted paircounts, and 
        :math:`\\mathrm{RR}` is the randomized weighted pair counts.
        
        If `sample2` is passed as input, three arrays of length len(`rbins`)-1 are 
        returned: :math:`\\mathcal{M}_{11}(r)`, :math:`\\mathcal{M}_{12}(r)`, 
        :math:`\\mathcal{M}_{22}(r)`,
        the autocorrelation of sample1, the cross-correlation between `sample1` and 
        `sample2`, and the autocorrelation of `sample2`.  If `do_auto` or `do_cross` is 
        set to False, the appropriate result(s) is not returned.

    Notes
    -----
    Pairs are counted using the pair_counters.objective_double_pairs module.  This pair 
    counter is optimized to work on points distributed in a rectangular cuboid volume, 
    e.g. a simulation box.  This optimization restricts this function to work on 3-D 
    point distributions.
    
    If the `period` argument is passed, points may not have any component of their 
    coordinates be negative.
    
    The available wfunc functions are:
    func ID 0: custom user-defined and compiled weighting function
    func ID 1: multiplicative weights, return w1*w2
    func ID 2: summed weights, return w1+w2
    func ID 3: equality weights, return r1*r2 if w1==w2
    func ID 4: greater than weights, return r1*r2 if w2>w1
    func ID 5: less than weights, return r1*r2 if w2<w1")
    func ID 6: greater than tolerance weights, return r2 if w2>(w1+r1)
    func ID 7: less than tolerance weights, return r2 if w2<(w1-r1)
    func ID 8: tolerance weights, return r2 if |w1-w2|<r1
    func ID 9: exclusion weights, return r2 if |w1-w2|>r1
    
    where w1, w2 are weights1 and weights2 parameters respectively, and r1, r2 are the 
    aux1 and aux2 paraemeters.
    
    These functions are defined in .pair_counters.objective_cpiars.objective_weights.pyx
    """
    
    #process parameters
    sample1, rbins, sample2, marks1, marks2, period, do_auto, do_cross, num_threads,\
        aux1, aux2, wfunc, _sample1_is_sample2, PBCs = _marked_tpcf_process_args(\
            sample1, rbins, sample2, marks1, marks2, period, do_auto, do_cross,\
            num_threads, max_sample_size, aux1, aux2, wfunc)
    
    
    def marked_pair_counts(sample1, sample2, rbins, period, num_threads,\
                           do_auto, do_cross, marks1, marks2, aux1, aux2,\
                           wfunc, _sample1_is_sample2):
        """
        Count weighted data pairs.
        """
        
        if do_auto==True:
            D1D1 = obj_wnpairs(sample1, sample1, rbins,\
                               weights1=marks1, weights2=marks1,\
                               aux1=aux1, aux2=aux1, wfunc = wfunc,\
                               period=period, num_threads=num_threads)
            D1D1 = np.diff(D1D1)
        else:
            D1D1=None
            D2D2=None
        
        if _sample1_is_sample2:
            D1D2 = D1D1
            D2D2 = D1D1
        else:
            if do_cross==True:
                D1D2 = obj_wnpairs(sample1, sample2, rbins,\
                                   weights1=marks1, weights2=marks2,\
                                   aux1=aux1, aux2=aux2, wfunc = wfunc,\
                                   period=period, num_threads=num_threads)
                D1D2 = np.diff(D1D2)
            else: D1D2=None
            if do_auto==True:
                D2D2 = obj_wnpairs(sample2, sample2, rbins,\
                                   weights1=marks2, weights2=marks2,\
                                   aux1=aux2, aux2=aux2, wfunc = wfunc,\
                                   period=period, num_threads=num_threads)
                D2D2 = np.diff(D2D2)
            else: D2D2=None

        return D1D1, D1D2, D2D2
    
    def random_counts(sample1, sample2, rbins, period, num_threads,\
                      do_auto, do_cross, marks1, marks2, aux1, aux2, wfunc,\
                      _sample1_is_sample2, permutate1, permutate2):
        """
        Count random weighted data pairs.
        """
        
        if do_auto==True:
            R1R1 = obj_wnpairs(sample1, sample1, rbins,\
                               weights1=marks1, weights2=marks1[permutate1],\
                               aux1=aux1, aux2=aux1[permutate1], wfunc = wfunc,\
                               period=period, num_threads=num_threads)
            R1R1 = np.diff(R1R1)
        else:
            R1R1=None
            R2R2=None
        
        if _sample1_is_sample2:
            R1R2 = R1R1
            R2R2 = R1R1
        else:
            if do_cross==True:
                R1R2 = obj_wnpairs(sample1, sample2, rbins,\
                                   weights1=marks1[permutate1], weights2=marks2[permutate2],\
                                   aux1=aux1[permutate1], aux2=aux2[permutate2], wfunc = wfunc,\
                                   period=period, num_threads=num_threads)
                R1R2 = np.diff(R1R2)
            else: R1R2=None
            if do_auto==True:
                R2R2 = obj_wnpairs(sample2, sample2, rbins,\
                                   weights1=marks2, weights2=marks2[permutate2],\
                                   aux1=aux2, aux2=aux2[permutate2], wfunc = wfunc,\
                                   period=period, num_threads=num_threads)
                R2R2 = np.diff(R2R2)
            else: R2R2=None

        return R1R1, R1R2, R2R2
    
    
    #get arrays to randomize marks
    permutate1 = np.random.permutation(np.arange(0,len(sample1)))
    permutate2 = np.random.permutation(np.arange(0,len(sample2)))
    
    #calculate marked pairs
    W1W1,W1W2,W2W2 = marked_pair_counts(sample1, sample2, rbins, period,\
                                        num_threads, do_auto, do_cross,\
                                        marks1, marks2, aux1, aux2, wfunc,\
                                        _sample1_is_sample2)
    
    #calculate randomized marked pairs
    R1R1,R1R2,R2R2 = random_counts(sample1, sample2, rbins, period,\
                                   num_threads, do_auto, do_cross,\
                                   marks1, marks2, aux1, aux2, wfunc,\
                                   _sample1_is_sample2, permutate1, permutate2)
    
    #return results
    if _sample1_is_sample2:
        M_11 = W1W1/R1R1 - 1.0
        return M_11
    else:
        if (do_auto==True) & (do_cross==True): 
            M_11 = W1W1/R1R1 - 1.0
            M_12 = W1W2/R1R2 - 1.0
            M_22 = W2W2/R2R2 - 1.0
            return M_11, M_12, M_22
        elif (do_cross==True):
            M_12 = W1W2/R1R2 - 1.0
            return M_12
        elif (do_auto==True):
            M_11 = W1W1/R1R1 - 1.0
            M_22 = W2W2/R2R2 - 1.0
            return M_11, M_22


