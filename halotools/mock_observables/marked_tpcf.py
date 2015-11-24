# -*- coding: utf-8 -*-

"""
Calculate the marked two point correlation function, MCF.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
####import modules########################################################################
import sys
import numpy as np
from math import pi, gamma
from .clustering_helpers import *
from .pair_counters.marked_double_tree_pairs import marked_npairs
from .pair_counters.double_tree_pairs import npairs
##########################################################################################


__all__=['marked_tpcf']
__author__ = ['Duncan Campbell']

np.seterr(divide='ignore', invalid='ignore') #ignore divide by zero in e.g. DD/RR

def marked_tpcf(sample1, rbins, sample2=None, 
    marks1=None, marks2=None, period=None, do_auto=True, do_cross=True, 
    num_threads=1, max_sample_size=int(1e6), wfunc=1, 
    normalize_by='random_marks', iterations=1, randomize_marks=None):
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
        len(sample1) x N_marks array of marks.  The suplied marks array must have the 
        appropiate shape for the chosen ``wfunc`` (see notes).
        
    marks2: array_like, optional
        len(sample2) x N_marks array of marks.  The suplied marks array must have the 
        appropiate shape for the chosen ``wfunc`` (see notes).
    
    period : array_like, optional
        length 3 array defining axis-aligned periodic boundary conditions. If only
        one number, Lbox, is specified, period is assumed to be [Lbox]*3.

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

    wfunc: int, optional
        Integer ID indicating which marking function should be used.  See notes for a 
        list of available marking functions.
    
    normalize_by: string, optional
        A string indicating how to normailze the weighted pair counts in the marked 
        correlation function calculation.  Options are: 'random_marks' or `number_counts`.
        See Notes for more detail.

    iterations : int, optional
        integer number indicating the number of times to calculate the random weigths, 
        taking the mean of the outcomes.  Only applicable if ``normalize_by`` is set 
        to 'random_marks'.  See notes for further explanation.

    randomize_marks : array_like, optional
        Boolean array of N_marks indicating which weights should be randomized for 
        the random counts.  Default is [True]*N_marks.  Only applicable if 
        ``normalize_by`` is 'random_marks'.

    Returns 
    -------
    marked_correlation_function : numpy.array
        *len(rbins)-1* length array containing the marked correlation function 
        :math:`\\mathcal{M}(r)` computed in each of the bins defined by ``rbins``.
        
        .. math::
            \\mathcal{M}(r) \\equiv \\mathrm{WW}(r) / \\mathrm{XX}(r),
        
        where :math:`\\mathrm{WW}(r)` is the weighted number of pairs with seperations 
        equal to :math:`r`, and :math:`\\mathrm{XX}(r)` is dependent on the choice of the 
        ``normalize_by`` parameter.  If ``normalize_by`` is 'random_marks' 
        :math:`XX \\equiv \\mathcal{RR}`, the weighted pair counts where the marks have 
        been randomized marks.  If ``normalize_by`` is 'number_counts' 
        :math:`XX \\equiv DD`, the unweighted pair counts.  
        See notes for a further discussion.
        
        If ``sample2`` is passed as input, three arrays of length *len(rbins)-1* are 
        returned: 
        
        .. math::
            \\mathcal{M}_{11}(r), \\ \\mathcal{M}_{12}(r), \\ \\mathcal{M}_{22}(r),
        
        the autocorrelation of ``sample1``, the cross-correlation between ``sample1`` and 
        ``sample2``, and the autocorrelation of ``sample2``.  If ``do_auto`` or 
        ``do_cross`` is set to False, the appropriate result(s) is not returned.

    Notes
    -----
    Pairs are counted using 
    `~halotools.mock_observables.pair_counters.marked_npairs`.
    This pair counter is optimized to work on points distributed in a rectangular cuboid 
    volume, e.g. a simulation box.  This optimization restricts this function to work on 
    3-D  point distributions.
    
    ``normalize_by`` indicates how to caclulate :math:`\\mathrm{XX}`.  If ``normalize_by``
    is 'random_marks', then :math:`\\mathrm{XX} \\equiv \\mathcal{RR}`, and 
    :math:`\\mathcal{RR}` is calculated by randomizing the marks among points accorinding 
    to the ``randomize_marks`` mask.  This marked correlation function is then:
    
    .. math::
        \\mathcal{M}(r) \\equiv \\frac{\\sum_{ij}f(m_i,m_j)}{\\sum_{kl}f(m_k,m_l)}
    
    where the sum in the numerator is of pairs :math:`i,j` with seperation :math:`r`, 
    and marks :math:`m_i,m_j`.  :math:`f()` is the marking function, ``wfunc``.  The sum 
    in the denominator is over an equal number of random pairs :math:`k,l`. The 
    calculation of this sum can be done multiple times, by setting the ``iterations`` 
    parameter. The mean of the sum is then taken amongst iterations and used in the 
    calculation.
    
    
    If ``normalize_by`` is 'number_counts', then :math:`\\mathrm{XX} \\equiv \\mathrm{DD}`
    is calculated by counting total number of pairs using 
    `~halotools.mock_observables.pair_counters.marked_double_pairs.npairs`.
    This is:
    
    .. math::
        \\mathcal{M}(r) \\equiv \\frac{1.0}{\\bar{n}(r)}\\sum_{ij}f(m_i,m_j),
    
    where :math:`\\bar{n}(r)` is the mean number density of points as a function of 
    seperation.
    
    
    The available marking functions, ``wfunc``, are:
    
    #. multiplicaitive weights (N_marks = 1)
        .. math::
            f(w_1,w_2) = w_1[0] \\times w_2[0]
    
    #. summed weights (N_marks = 1)
        .. math::
            f(w_1,w_2) = w_1[0] + w_2[0]
    
    #. equality weights (N_marks = 2)
        .. math::
            f(w_1,w_2) = 
                \\left \\{
                \\begin{array}{ll}
                    w_1[0]\\times w_2[0] & : w_1[0] = w_2[0] \\\\
                    0.0 & : w_1[0] \\neq w_2[0] \\\\
                \\end{array}
                \\right.
    
    #. greater than weights (N_marks = 2)
        .. math::
            f(w_1,w_2) = 
                \\left \\{
                \\begin{array}{ll}
                    w_1[0]\\times w_2[0] & : w_2[0] > w_1[0] \\\\
                    0.0 & : w_2[0] \\leq w_1[0] \\\\
                \\end{array}
                \\right.
    
    #. less than weights (N_marks = 2)
        .. math::
            f(w_1,w_2) = 
                \\left \\{
                \\begin{array}{ll}
                    w_1[1]\\times w_2[1] & : w_2[0] < w_1[0] \\\\
                    0.0 & : w_2[0] \\geq w_1[0] \\\\
                \\end{array}
                \\right.
    
    #. greater than tolerance weights (N_marks = 2)
        .. math::
            f(w_1,w_2) = 
                \\left \\{
                \\begin{array}{ll}
                    w_2[1] & : w2[0]>(w1[0]+w1[1]) \\\\
                    0.0 & : w2[0] \\leq (w1[0]+w1[1]) \\\\
                \\end{array}
                \\right.
    
    #. less than tolerance weights (N_marks = 2)
        .. math::
            f(w_1,w_2) = 
                \\left \\{
                \\begin{array}{ll}
                    w_2[1] & : w2[0]<(w1[0]+w1[1]) \\\\
                    0.0 & : w2[0] \\geq (w1[0]+w1[1]) \\\\
                \\end{array}
                \\right.
    
    #. tolerance weights (N_marks = 2)
        .. math::
            f(w_1,w_2) = 
                \\left \\{
                \\begin{array}{ll}
                    w_2[1] & : |w1[0]-w2[0]|<w1[1] \\\\
                    0.0 & : |w1[0]-w2[0]| \\geq w1[1] \\\\
                \\end{array}
                \\right.
    
    #. exclusion weights (N_marks = 2)
        .. math::
            f(w_1,w_2) = 
                \\left \\{
                \\begin{array}{ll}
                    w_2[1] & : |w1[0]-w2[0]|>w1[1] \\\\
                    0.0 & : |w1[0]-w2[0]| \\leq w1[1] \\\\
                \\end{array}
                \\right.
    
    #. radial velocity weights (N_marks = 6)
        .. math::
            \\begin{array}{ll}
                \\mathrm{d}r_x & = w1[0]-w2[0] \\\\
                \\mathrm{d}r_y & = w1[1]-w2[1] \\\\
                \\mathrm{d}r_z & = w1[2]-w2[2] \\\\
                \\mathrm{d}v_x & = w1[3]-w2[3] \\\\
                \\mathrm{d}v_y & = w1[4]-w2[4] \\\\
                \\mathrm{d}v_z & = w1[5]-w2[5] \\\\
            \\end{array}
        .. math::
            f(w_1,w_2) = (\\mathrm{d}r_x \\mathrm{d}v_x+\\mathrm{d}r_y \\mathrm{d}v_y+\\mathrm{d}r_z \\mathrm{d}v_z)/\sqrt{\\mathrm{d}r_x^2+\\mathrm{d}r_y^2+\\mathrm{d}r_z^2}
    
    #. vector dot weights (N_marks = 3)
        .. math::
            f(w_1,w_2) = (w1[0] + w2[0]) + (w1[1] + w2[1]) + (w1[2] + w2[2])
    
    #. vector angle weights (N_marks = 3)
        .. math::
            \\begin{array}{ll}
                {\\rm norm} & = \\sqrt{w1[0]w1[0] + w1[1]w1[1] + w1[2]w1[2]}\\sqrt{w2[0]w2[0] + w2[1]w2[1] + w2[2]w2[2]} \\\\
                f(w_1,w_2) & = (w1[0] + w2[0]) + (w1[1] + w2[1]) + (w1[2] + w2[2])/{\\rm norm} \\\\
            \\end{array}
    
    #. inequality weights (N_marks = 2)
        .. math::
            f(w_1,w_2) = 
                \\left \\{
                \\begin{array}{ll}
                    w_1[0]\\times w_2[0] & : w_1[0] \\neq w_2[0] \\\\
                    0.0 & : w_1[0] = w_2[0] \\\\
                \\end{array}
                \\right.
    
    Examples
    --------
    For demonstration purposes we create a randomly distributed set of points within a 
    periodic unit cube. 
    
    >>> Npts = 1000
    >>> Lbox = 1.0
    >>> period = np.array([Lbox,Lbox,Lbox])
    
    >>> x = np.random.random(Npts)
    >>> y = np.random.random(Npts)
    >>> z = np.random.random(Npts)
    
    We transform our *x, y, z* points into the array shape used by the pair-counter by 
    taking the transpose of the result of `numpy.vstack`. This boilerplate transformation 
    is used throughout the `~halotools.mock_observables` sub-package:
    
    >>> coords = np.vstack((x,y,z)).T
    
    Assign random floats in the range [0,1] to the points to use as the marks:
    
    >>> marks = np.random.random(Npts)
    
    Use the multiplicative marking function:
    
    >>> rbins = np.logspace(-2,-1,10)
    >>> MCF = marked_tpcf(coords, rbins, marks1=marks, period=period, normalize_by='number_counts', wfunc=1)
    
    The result should be consistent with :math:`\\langle {\\rm mark}\\rangle^2` at all *r* 
    within the statistical errors.
    """


    #process parameters
    function_args = [sample1, rbins, sample2, marks1, marks2, period, do_auto, do_cross,\
                     num_threads, max_sample_size, wfunc, normalize_by, iterations, randomize_marks]
    sample1, rbins, sample2, marks1, marks2, period, do_auto, do_cross, num_threads,\
        wfunc, normalize_by, _sample1_is_sample2, PBCs,\
        randomize_marks = _marked_tpcf_process_args(*function_args)
    
    def marked_pair_counts(sample1, sample2, rbins, period, num_threads,\
                           do_auto, do_cross, marks1, marks2,\
                           wfunc, _sample1_is_sample2):
        """
        Count weighted data pairs.
        """
        
        if do_auto==True:
            D1D1 = marked_npairs(sample1, sample1, rbins,\
                                 weights1=marks1, weights2=marks1,\
                                 wfunc = wfunc,\
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
                D1D2 = marked_npairs(sample1, sample2, rbins,\
                                     weights1=marks1, weights2=marks2,\
                                     wfunc = wfunc,\
                                     period=period, num_threads=num_threads)
                D1D2 = np.diff(D1D2)
            else: D1D2=None
            if do_auto==True:
                D2D2 = marked_npairs(sample2, sample2, rbins,\
                                     weights1=marks2, weights2=marks2,\
                                     wfunc = wfunc,\
                                     period=period, num_threads=num_threads)
                D2D2 = np.diff(D2D2)
            else: D2D2=None

        return D1D1, D1D2, D2D2
    
    def random_counts(sample1, sample2, rbins, period, num_threads,\
                      do_auto, do_cross, marks1, marks2, wfunc,\
                      _sample1_is_sample2, permutate1, permutate2, randomize_marks):
        """
        Count random weighted data pairs.
        """
        
        permuted_marks1 = marks1
        permuted_marks2 = marks2
        for i in range(marks1.shape[1]):
            if randomize_marks[i]:
                permuted_marks1[:,i] = marks1[permutate1,i]
        for i in range(marks2.shape[1]):
            if randomize_marks[i]:
                permuted_marks2[:,i] = marks2[permutate2,i]
        
        if do_auto==True:
            R1R1 = marked_npairs(sample1, sample1, rbins,\
                                 weights1=marks1, weights2=permuted_marks1,\
                                 wfunc = wfunc,\
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
                R1R2 = marked_npairs(sample1, sample2, rbins,\
                                     weights1=permuted_marks1,\
                                     weights2=permuted_marks2,\
                                     wfunc = wfunc,\
                                     period=period, num_threads=num_threads)
                R1R2 = np.diff(R1R2)
            else: R1R2=None
            if do_auto==True:
                R2R2 = marked_npairs(sample2, sample2, rbins,\
                                     weights1=marks2,\
                                     weights2=permuted_marks2,\
                                     wfunc = wfunc,\
                                     period=period, num_threads=num_threads)
                R2R2 = np.diff(R2R2)
            else: R2R2=None

        return R1R1, R1R2, R2R2
    
    def pair_counts(sample1, sample2, rbins, period, N_thread, do_auto, do_cross,\
                    _sample1_is_sample2, approx_cell1_size, approx_cell2_size):
        """
        Count data-data pairs.
        """
        if do_auto==True:
            D1D1 = npairs(sample1, sample1, rbins, period=period, num_threads=num_threads,
                          approx_cell1_size=approx_cell1_size,
                          approx_cell2_size=approx_cell1_size)
            D1D1 = np.diff(D1D1)
        else:
            D1D1=None
            D2D2=None
        
        if _sample1_is_sample2:
            D1D2 = D1D1
            D2D2 = D1D1
        else:
            if do_cross==True:
                D1D2 = npairs(sample1, sample2, rbins, period=period,
                              num_threads=num_threads,
                              approx_cell1_size=approx_cell1_size,
                              approx_cell2_size=approx_cell2_size)
                D1D2 = np.diff(D1D2)
            else: D1D2=None
            if do_auto==True:
                D2D2 = npairs(sample2, sample2, rbins, period=period,
                              num_threads=num_threads,
                              approx_cell1_size=approx_cell2_size,
                              approx_cell2_size=approx_cell2_size)
                D2D2 = np.diff(D2D2)
            else: D2D2=None
        
        return D1D1, D1D2, D2D2
    
    #calculate marked pairs
    W1W1,W1W2,W2W2 = marked_pair_counts(sample1, sample2, rbins, period,\
                                        num_threads, do_auto, do_cross,\
                                        marks1, marks2, wfunc,\
                                        _sample1_is_sample2)
    
    if normalize_by=='number_counts':
        R1R1,R1R2,R2R2 = pair_counts(sample1, sample2, rbins, period,
                                     num_threads, do_auto, do_cross, _sample1_is_sample2,
                                     None, None)
    #calculate randomized marked pairs
    elif normalize_by=='random_marks':
        if iterations > 1:
            #create storage arrays of the right shape
            R1R1 = np.zeros((iterations,len(rbins)-1))
            R1R2 = np.zeros((iterations,len(rbins)-1))
            R2R2 = np.zeros((iterations,len(rbins)-1))
            for i in range(iterations):
                print(i)
                #get arrays to randomize marks
                permutate1 = np.random.permutation(np.arange(0,len(sample1)))
                permutate2 = np.random.permutation(np.arange(0,len(sample2)))
                R1R1[i,:],R1R2[i,:],R2R2[i,:] = random_counts(sample1, sample2, rbins, period,\
                                                num_threads, do_auto, do_cross,\
                                                marks1, marks2, wfunc,\
                                                _sample1_is_sample2,\
                                                permutate1, permutate2, randomize_marks)
            #take mean of the iterations
            R1R1_err = np.std(R1R1, axis=0)
            R1R1 = np.median(R1R1, axis=0)
            R1R2_err = np.std(R1R2, axis=0)
            R1R2 = np.median(R1R2, axis=0)
            R2R2_err = np.std(R2R2, axis=0)
            R2R2 = np.median(R2R2, axis=0)
        else:
            #get arrays to randomize marks
            permutate1 = np.random.permutation(np.arange(0,len(sample1)))
            permutate2 = np.random.permutation(np.arange(0,len(sample2)))
            R1R1,R1R2,R2R2 = random_counts(sample1, sample2, rbins, period,\
                                       num_threads, do_auto, do_cross,\
                                       marks1, marks2, wfunc,\
                                       _sample1_is_sample2, permutate1, permutate2,\
                                       randomize_marks)
    else: 
        msg = 'normalize_by parameter not recognized.'
        raise ValueError(msg)
    
    #return results
    if _sample1_is_sample2:
        M_11 = W1W1/R1R1
        return M_11
    else:
        if (do_auto==True) & (do_cross==True): 
            M_11 = W1W1/R1R1
            M_12 = W1W2/R1R2
            M_22 = W2W2/R2R2
            return M_11, M_12, M_22
        elif (do_cross==True):
            M_12 = W1W2/R1R2
            return M_12
        elif (do_auto==True):
            M_11 = W1W1/R1R1
            M_22 = W2W2/R2R2 
            return M_11, M_22


