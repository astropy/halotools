# -*- coding: utf-8 -*-

"""
functions to calculate clustering statistics, e.g. two point correlation functions.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
####import modules########################################################################
import sys
import numpy as np
from math import pi, gamma

from .pair_counters.rect_cuboid_pairs import npairs, xy_z_npairs, jnpairs, s_mu_npairs
from .pair_counters.double_tree_pairs import npairs
##########################################################################################


__all__=['redshift_space_tpcf','wp','s_mu_tpcf']
__author__ = ['Duncan Campbell']


np.seterr(divide='ignore', invalid='ignore') #ignore divide by zero in e.g. DD/RR


def redshift_space_tpcf(sample1, rp_bins, pi_bins, sample2=None, randoms=None,\
                        period=None, do_auto=True, do_cross=True, estimator='Natural',\
                        num_threads=1, max_sample_size=int(1e6)):
    """ 
    Calculate the redshift space correlation function, :math:`\\xi(r_{p}, \\pi)`.
    
    The first two dimensions define the plane for perpendicular distances.  The third 
    dimension is used for parallel distances.  i.e. x,y positions are on the plane of the
    sky, and z is the redshift coordinate.  This is the distant observer approximation.
    
    Parameters 
    ----------
    sample1 : array_like
        Npts x 3 numpy array containing 3-D positions of points. 
    
    rp_bins : array_like
        numpy array of boundaries defining the bins in which pairs are counted. 
    
    pi_bins : array_like
        numpy array of boundaries defining the bins in which pairs are counted. 
    
    sample2 : array_like, optional
        Npts x 3 numpy array containing 3-D positions of points.
    
    randoms : array_like, optional
        Nran x 3 numpy array containing 3-D positions of points.  If no randoms are 
        provided analytic randoms are used (only valid for periodic boundary conditions).
    
    period : array_like, optional
        length 3 array defining axis-aligned periodic boundary conditions. If only 
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*3).
        If none, PBCs are set to infinity.
    
    estimator : string, optional
        options: 'Natural', 'Davis-Peebles', 'Hewett' , 'Hamilton', 'Landy-Szalay'
    
    do_auto : boolean, optional
        do auto-correlation?  Default is True.
    
    do_cross : boolean, optional
        do cross-correlation?  Default is True.
    
    N_thread : int, optional
        number of threads to use in calculation. Default is 1. A string 'max' may be used
        to indicate that the pair counters should use all available cores on the machine.
    
    max_sample_size : int, optional
        Defines maximum size of the sample that will be passed to the pair counter. 
        
        If sample size exceeds `max_sample_size`, the sample will be randomly down-sampled 
        such that the subsample is equal to `max_sample_size`. 

    Returns 
    -------
    correlation_function : array_like
        ndarray containing correlation function :math:`\\xi(r_{p}, \\pi)` computed in each 
        of the len(`rp_bins`)-1 by len(`pi_bins`)-1 bins defined by input `rp_bins` and 
        `pi_bins`.

        :math:`1 + \\xi(r_{p},\\pi) = \\mathrm{DD} / \\mathrm{RR}`, is the 'Natural' 
        `estimator` is used, where :math:`\\mathrm{DD}` is calculated by the pair 
        counter, and :math:`\\mathrm{RR}` is counted internally analytic randoms if no 
        `randoms` are passed as an argument.

        If sample2 is passed as input, three ndarrays of shape 
        (len(`rp_bins`)-1,len(`pi_bins`)-1) are returned: 
        :math:`\\xi_{11}(rp, \\pi)`, :math:`\\xi_{12}(r_{p},\\pi)`, 
        :math:`\\xi_{22}(r_{p},\\pi)`,
        
        The autocorrelation of` sample1`, the cross-correlation between `sample1` 
        and `sample2`, and the autocorrelation of `sample2`.  If `do_auto` or `do_cross` 
        is set to False, the appropriate result(s) is not returned.

    """
    
    estimators = _list_estimators()
    
    #process input parameters
    sample1 = np.asarray(sample1)
    if sample2 is not None: 
        sample2 = np.asarray(sample2)
        if np.all(sample1==sample2):
            do_cross==False
            print("Warning: sample1 and sample2 are exactly the same, only the\
                   auto-correlation will be returned.")
    else: sample2 = sample1
    if randoms is not None: randoms = np.asarray(randoms)
    rp_bins = np.asarray(rp_bins)
    pi_bins = np.asarray(pi_bins)
    
    #Process period entry and check for consistency.
    if period is None:
            PBCs = False
            period = np.array([np.inf]*np.shape(sample1)[-1])
    else:
        PBCs = True
        period = np.asarray(period).astype("float64")
        if np.shape(period) == ():
            period = np.array([period]*np.shape(sample1)[-1])
        elif np.shape(period)[0] != np.shape(sample1)[-1]:
            raise ValueError("period should have shape (k,)")
            return None
    
    #down sample is sample size exceeds max_sample_size.
    if (len(sample2)>max_sample_size) & (not np.all(sample1==sample2)):
        inds = np.arange(0,len(sample2))
        np.random.shuffle(inds)
        inds = inds[0:max_sample_size]
        sample2 = sample2[inds]
        print('down sampling sample2...')
    if len(sample1)>max_sample_size:
        inds = np.arange(0,len(sample1))
        np.random.shuffle(inds)
        inds = inds[0:max_sample_size]
        sample1 = sample1[inds]
        print('down sampling sample1...')
    
    #check radial bins
    if np.shape(rp_bins) == ():
        rp_bins = np.array([rp_bins])
    if np.shape(pi_bins) == ():
        pi_bins = np.array([pi_bins])
    if rp_bins.ndim != 1:
        raise ValueError('rp bins must be a 1-D array')
    if pi_bins.ndim != 1:
        raise ValueError('pi bins must be a 1-D array')
    if len(rp_bins)<2:
        raise ValueError('rp bins must be of lenght >=2.')
    if len(pi_bins)<2:
        raise ValueError('pi bins must be of lenght >=2.')
    
    k = np.shape(sample1)[-1] #dimensionality of data
    if k!=3:
        raise ValueError('data must be 3-dimensional.')
    
    #check for input parameter consistency
    if (period is not None) & (np.max(rp_bins)>np.min(period[0:2])/2.0):
        raise ValueError('Cannot calculate for rp seperations larger than Lbox[0:2]/2.')
    if (period is not None) & (np.max(pi_bins)>np.min(period[2])/2.0):
        raise ValueError('Cannot calculate for pi seperations larger than Lbox[2]/2.')
    if (sample2 is not None) & (sample1.shape[-1]!=sample2.shape[-1]):
        raise ValueError('Sample 1 and sample 2 must have same dimension.')
    if (randoms is None) & (min(period)==np.inf):
        raise ValueError('If no PBCs are specified, randoms must be provided.')
    if estimator not in estimators: 
        raise ValueError('Must specify a supported estimator. Supported estimators \
        are:{0}'.value(estimators))
    if (PBCs==True) & (max(period)==np.inf):
        raise ValueError('If a non-infinte PBC specified, all PBCs must be non-infinte.')
    if (type(do_auto) is not bool) | (type(do_cross) is not bool):
        raise ValueError('do_auto and do_cross keywords must be of type boolean.')

    #If PBCs are defined, calculate the randoms analytically. Else, the user must specify
    #randoms and the pair counts are calculated the old fashion way.
    def random_counts(sample1, sample2, randoms, rp_bins, pi_bins, period,\
                      PBCs, k, num_threads, do_RR, do_DR):
        """
        Count random pairs.  There are three high level branches: 

            1. no PBCs w/ randoms.

            2. PBCs w/ randoms
            
            3. PBCs and analytical randoms

        There are also logical bits to do RR and DR pair counts, as not all estimators 
        need one or the other, and not doing these can save a lot of calculation.
        """
        def cylinder_volume(R,h):
            """
            Calculate the volume of a cylinder(s), used for the analytical randoms.
            """
            return pi*np.outer(R**2.0,h)
        
        #No PBCs, randoms must have been provided.
        if PBCs==False:
            RR = xy_z_npairs(randoms, randoms, rp_bins, pi_bins, period=period,\
                             num_threads=num_threads)
            RR = np.diff(np.diff(RR,axis=0),axis=1)
            D1R = xy_z_npairs(sample1, randoms, rp_bins, pi_bins, period=period,\
                              num_threads=num_threads)
            D1R = np.diff(np.diff(D1R,axis=0),axis=1)
            if np.all(sample1 == sample2): #calculating the cross-correlation
                D2R = None
            else:
                D2R = xy_z_npairs(sample2, randoms, rp_bins, pi_bins, period=period,\
                                  num_threads=num_threads)
                D2R = np.diff(np.diff(D2R,axis=0),axis=1)
            
            return D1R, D2R, RR
        #PBCs and randoms.
        elif randoms is not None:
            if do_RR==True:
                RR = xy_z_npairs(randoms, randoms, rp_bins, pi_bins, period=period,\
                                 num_threads=num_threads)
                RR = np.diff(np.diff(RR,axis=0),axis=1)
            else: RR=None
            if do_DR==True:
                D1R = xy_z_npairs(sample1, randoms, rp_bins, pi_bins, period=period,\
                                  num_threads=num_threads)
                D1R = np.diff(np.diff(D1R,axis=0),axis=1)
            else: D1R=None
            if np.all(sample1 == sample2): #calculating the cross-correlation
                D2R = None
            else:
                if do_DR==True:
                    D2R = xy_z_npairs(sample2, randoms, rp_bins, pi_bins, period=period,\
                                      num_threads=num_threads)
                    D2R = np.diff(np.diff(D2R,axis=0),axis=1)
                else: D2R=None
            
            return D1R, D2R, RR
        #PBCs and no randoms--calculate randoms analytically.
        elif randoms is None:
            #do volume calculations
            dv = cylinder_volume(rp_bins,2.0*pi_bins) #volume of spheres
            dv = np.diff(np.diff(dv, axis=0),axis=1) #volume of annuli
            global_volume = period.prod() #sexy
            
            #calculate randoms for sample1
            N1 = np.shape(sample1)[0]
            rho1 = N1/global_volume
            D1R = (N1)*(dv*rho1) #read note about pair counter
            
            #if not calculating cross-correlation, set RR exactly equal to D1R.
            if np.all(sample1 == sample2):
                D2R = None
                RR = D1R #in the analytic case, for the auto-correlation, DR==RR.
            else: #if there is a sample2, calculate randoms for it.
                N2 = np.shape(sample2)[0]
                rho2 = N2/global_volume
                D2R = N2*(dv*rho2) #read note about pair counter
                #calculate the random-random pairs.
                NR = N1*N2
                rhor = NR/global_volume
                RR = (dv*rhor) #RR is only the RR for the cross-correlation.

            return D1R, D2R, RR
        else:
            raise ValueError('Un-supported combination of PBCs and randoms provided.')
    
    def pair_counts(sample1, sample2, rp_bins, pi_bins, period,\
                    N_thread, do_auto, do_cross, do_DD):
        """
        Count data pairs.
        """
        D1D1 = xy_z_npairs(sample1, sample1, rp_bins, pi_bins, period=period,\
                           num_threads=num_threads)
        D1D1 = np.diff(np.diff(D1D1,axis=0),axis=1)
        if np.all(sample1 == sample2):
            D1D2 = D1D1
            D2D2 = D1D1
        else:
            D1D2 = xy_z_npairs(sample1, sample2, rp_bins, pi_bins, period=period,\
                               num_threads=num_threads)
            D1D2 = np.diff(np.diff(D1D2,axis=0),axis=1)
            D2D2 = xy_z_npairs(sample2, sample2, rp_bins, pi_bins, period=period,\
                               num_threads=num_threads)
            D2D2 = np.diff(np.diff(D2D2,axis=0),axis=1)

        return D1D1, D1D2, D2D2
    
    do_DD, do_DR, do_RR = _TP_estimator_requirements(estimator)
              
    if randoms is not None:
        N1 = len(sample1)
        N2 = len(sample2)
        NR = len(randoms)
    else: 
        N1 = 1.0
        N2 = 1.0
        NR = 1.0
    
    #count pairs
    D1D1,D1D2,D2D2 = pair_counts(sample1, sample2, rp_bins, pi_bins, period,\
                                 num_threads, do_auto, do_cross, do_DD)
    D1R, D2R, RR = random_counts(sample1, sample2, randoms, rp_bins, pi_bins, period,\
                                 PBCs, k, num_threads, do_RR, do_DR)
    
    if np.all(sample2==sample1):
        xi_11 = _TP_estimator(D1D1,D1R,RR,N1,N1,NR,NR,estimator)
        return xi_11
    else:
        if (do_auto==True) & (do_cross==True): 
            xi_11 = _TP_estimator(D1D1,D1R,RR,N1,N1,NR,NR,estimator)
            xi_12 = _TP_estimator(D1D2,D1R,RR,N1,N2,NR,NR,estimator)
            xi_22 = _TP_estimator(D2D2,D2R,RR,N2,N2,NR,NR,estimator)
            return xi_11, xi_12, xi_22
        elif (do_cross==True):
            xi_12 = _TP_estimator(D1D2,D1R,RR,N1,N2,NR,NR,estimator)
            return xi_12
        elif (do_auto==True):
            xi_11 = _TP_estimator(D1D1,D1R,D1R,N1,N1,NR,NR,estimator)
            xi_22 = _TP_estimator(D2D2,D2R,D2R,N2,N2,NR,NR,estimator)
            return xi_11


def wp(sample1, rp_bins, pi_bins, sample2=None, randoms=None, period=None,\
       do_auto=True, do_cross=True, estimator='Natural', num_threads=1,\
       max_sample_size=int(1e6)):
    """ 
    Calculate the projected correlation function, :math:`w_{p}(r_p)`.
    
    The first two dimensions define the plane for perpendicular distances.  The third 
    dimension is used for parallel distances.  i.e. x,y positions are on the plane of the
    sky, and z is the redshift coordinate. This is the 'distant observer' approximation.
    
    Parameters 
    ----------
    sample1 : array_like
        Npts x 3 numpy array containing 3-D positions of points. 
    
    rp_bins : array_like
        array of boundaries defining the perpendicular bins in which pairs are counted.
    
    pi_bins : array_like
        array of boundaries defining the parallel bins in which pairs are counted. 
    
    sample2 : array_like, optional
        Npts x 3 numpy array containing 3-D positions of points.
    
    randoms : array_like, optional
        Nran x 3 numpy array containing 3-D positions of points.
    
    period : array_like, optional
        length k array defining axis-aligned periodic boundary conditions. If only 
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*k).
        If none, PBCs are set to infinity.
    
    do_auto : boolean, optional
        do auto-correlation?  Default is True.
    
    do_cross : boolean, optional
        do cross-correlation?  Default is True.
    
    estimator : string, optional
        options: 'Natural', 'Davis-Peebles', 'Hewett' , 'Hamilton', 'Landy-Szalay'
    
    num_threads : int, optional
        number of threads to use in calculation. Default is 1. A string 'max' may be used
        to indicate that the pair counters should use all available cores on the machine.
    
    max_sample_size : int, optional
        Defines maximum size of the sample that will be passed to the pair counter. 
        
        If sample size exceeds max_sample_size, the sample will be randomly down-sampled 
        such that the subsample is equal to max_sample_size.

    Returns 
    -------
    correlation_function : numpy.array
        len(`rp_bins`)-1 length array containing the correlation function :math:`w_p(r_p)` 
        computed in each of the bins defined by input `rp_bins`.

        If `sample2` is passed as input, three arrays of length len(`rbins`)-1 are 
        returned: :math:`w_{p11}(r_p)`, :math:`w_{p12}(r_p)`, :math:`w_{p22}(r_p)`.

        The autocorrelation of `sample1`, the cross-correlation between `sample1` 
        and `sample2`, and the autocorrelation of `sample2`.  If `do_auto` or `do_cross` 
        is set to False, the appropriate result(s) is not returned.
    
    Notes
    -----
    The projected correlation function is calculated by:
    
    .. math:: `w_p{r_p} = \\int_0^{\\pi_{\\rm max}}\\xi(r_p,\\pi)\\mathrm{d}\\pi`
    
    where :math:`\\pi_{\\rm max} = \\mathrm{maximum(pi_bins)}` and :math:`\\xi(r_p,\\pi)` 
    is the redshift space correlation function.  See the documentation on 
    redshift_space_tpcf() for further details.
    
    Notice that the results will generally be sensitive to the choice of `pi_bins`.
    
    """
    
    #pass the arguments into the redshift space TPCF function
    result = redshift_space_tpcf(sample1, rp_bins, pi_bins,\
                                 sample2 = sample2, randoms=randoms,\
                                 period = period, do_auto=do_auto, do_cross=do_cross,\
                                 estimator=estimator, num_threads=num_threads,\
                                 max_sample_size=max_sample_size)
    
    #take care of some API issues
    if sample2 is None: 
        do_cross=False
    elif np.all(sample2==sample1): 
        do_cross=False
    
    #integrate the redshift space TPCF to get w_p
    def integrate_2D_xi(x,pi_bins):
        return 2.0*np.sum(x*np.diff(pi_bins),axis=1)

    #return the results.
    if (do_auto==True) & (do_cross==True):
        wp_D1D1 = integrate_2D_xi(result[0],pi_bins)
        wp_D1D2 = integrate_2D_xi(result[1],pi_bins)
        wp_D2D2 = integrate_2D_xi(result[2],pi_bins)
        return wp_D1D1, wp_D1D2, wp_D2D2
    elif (do_auto==False) & (do_cross==True):
        wp_D1D2 = integrate_2D_xi(result,pi_bins)
        return wp_D1D2
    elif (do_auto==True) & (do_cross==False):
        if sample2 is None: # do only sample 1 auto
            wp_D1D1 = integrate_2D_xi(result,pi_bins)
            return wp_D1D1
        elif np.all(sample2==sample1): # do only sample 1 auto
            wp_D1D1 = integrate_2D_xi(result,pi_bins)
            return wp_D1D1
        else: # do both auto for sample1 and sample2
            wp_D1D1 = integrate_2D_xi(result[0],pi_bins)  
            wp_D2D2 = integrate_2D_xi(result[1],pi_bins)
            return wp_D1D1, wp_D2D2


def s_mu_tpcf(sample1, s_bins, mu_bins, sample2=None, randoms=None,\
              period=None, do_auto=True, do_cross=True, estimator='Natural',\
              num_threads=1, max_sample_size=int(1e6)):
    """ 
    Calculate the redshift space correlation function, :math:`\\xi(s, \\mu)`, where
    .. math:: s^2 = r_{\\parallel}^2+r_{\\perp}^2
    and, 
    .. math:: `\\mu = r_{\\parallel}/s`
    
    Data must be 3-D.  The first two dimensions define the plane for perpendicular 
    distances.  The third dimension is used for parallel distances.  i.e. x,y positions 
    are on the plane of the sky, and z is the redshift coordinate.  This is the distant 
    observer approximation.
    
    This is a pre-step for calculating correlation function multipoles.
    
    Parameters 
    ----------
    sample1 : array_like
        Npts x 3 numpy array containing 3-D positions of points. 
    
    s_bins : array_like
        numpy array of boundaries defining the bins in which pairs are counted. 
    
    mu_bins : array_like
        numpy array of boundaries defining the bins in which pairs are counted. 
    
    sample2 : array_like, optional
        Npts x 3 numpy array containing 3-D positions of points.
    
    randoms : array_like, optional
        Nran x 3 numpy array containing 3-D positions of points.  If no randoms are 
        provided 'analytic randoms' are used (only valid for periodic boundary conditions).
    
    period : array_like, optional
        length 3 array defining axis-aligned periodic boundary conditions. If only 
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*3).
        If none, PBCs are set to infinity.
    
    estimator : string, optional
        options: 'Natural', 'Davis-Peebles', 'Hewett' , 'Hamilton', 'Landy-Szalay'
    
    do_auto : boolean, optional
        do auto-correlation?  Default is True.
    
    do_cross : boolean, optional
        do cross-correlation?  Default is True.
    
    num_threads : int, optional
        number of threads to use in calculation. Default is 1. A string 'max' may be used
        to indicate that the pair counters should use all available cores on the machine.
    
    max_sample_size : int, optional
        Defines maximum size of the sample that will be passed to the pair counter. 
        
        If sample size exeeds max_sample_size, the sample will be randomly down-sampled 
        such that the subsample length is equal to max_sample_size. 

    Returns 
    -------
    correlation_function : np.array
        ndarray containing correlation function :math:`\\xi(s, \\mu)` computed in each 
        of the len(`s_bins`)-1 by len(`mu_bins`)-1 bins defined by input `s_bins` and 
        `mu_bins`.

        :math:`1 + \\xi(s,\\mu) \\equiv \\mathrm{DD} / \\mathrm{RR}`, if the 'Natural' 
        `estimator` is used, where :math:`\\mathrm{DD}` is calculated by the pair counter, 
        and :math:`\\mathrm{RR}` is counted internally using analytic randoms if no 
        `randoms` are passed as an argument.

        If `sample2` is passed as input, three arrays of shape 
        len(`rp_bins`)-1 by len(`pi_bins`)-1 are returned: 
        :math:`\\xi_{11}(s, \\mu)`, :math:`\\xi_{12}(s,\\mu)`, :math:`\\xi_{22}(s,\\mu)`,
        the autocorrelation of `sample1`, the cross-correlation between `sample1` 
        and `sample2`, and the autocorrelation of sample2.  If `do_auto` or `do_cross` 
        is set to False, the appropriate result(s) is not returned.
    """
    
    estimators = _list_estimators()
    
    #process input parameters
    sample1 = np.asarray(sample1)
    if sample2 is not None: 
        sample2 = np.asarray(sample2)
        if np.all(sample1==sample2):
            do_cross==False
            print("Warning: sample1 and sample2 are exactly the same, only the\
                   auto-correlation will be returned.")
    else: sample2 = sample1
    if randoms is not None: randoms = np.asarray(randoms)
    s_bins = np.asarray(s_bins)
    mu_bins = np.asarray(mu_bins)
    
    #work with the sine of the angle between s and the LOS.  Only using cosine as the 
    #input because of convention.  sin(theta_los) increases as theta_los increases, which
    #is required in order to get the pair counter to work.  see note in cpairs s_mu_pairs.
    theta = np.arccos(mu_bins)
    mu_bins = np.sin(theta)[::-1] #must be increasing, remember to reverse result.
    
    #process the period parameter and check for consistency.
    if period is None:
            PBCs = False
            period = np.array([np.inf]*np.shape(sample1)[-1])
    else:
        PBCs = True
        period = np.asarray(period).astype("float64")
        if np.shape(period) == ():
            period = np.array([period]*np.shape(sample1)[-1])
        elif np.shape(period)[0] != np.shape(sample1)[-1]:
            raise ValueError("period should have shape (k,)")
    
    #downsample if sample size exceeds max_sample_size.
    if (len(sample1)>max_sample_size) & (np.all(sample1==sample2)):
        inds = np.arange(0,len(sample1))
        np.random.shuffle(inds)
        inds = inds[0:max_sample_size]
        sample1 = sample1[inds]
        sample2 = sample2[inds]
        print('downsampling sample1...')
    if len(sample2)>max_sample_size:
        inds = np.arange(0,len(sample2))
        np.random.shuffle(inds)
        inds = inds[0:max_sample_size]
        sample2 = sample2[inds]
        print('downsampling sample2...')
    if len(sample1)>max_sample_size:
        inds = np.arange(0,len(sample1))
        np.random.shuffle(inds)
        inds = inds[0:max_sample_size]
        sample1 = sample1[inds]
        print('downsampling sample1...')
    
    #check the bins parameters
    if np.shape(s_bins) == ():
        s_bins = np.array([s_bins])
    if np.shape(mu_bins) == ():
        mu_bins = np.array([mu_bins])
    if s_bins.ndim != 1:
        raise ValueError('s bins must be a 1-D array')
    if mu_bins.ndim != 1:
        raise ValueError('mu bins must be a 1-D array')
    if len(s_bins)<2:
        raise ValueError('s bins must be of lenght >=2.')
    if len(mu_bins)<2:
        raise ValueError('mu bins must be of lenght >=2.')
    if (np.min(mu_bins)<0.0) | (np.max(mu_bins)>1.0):
        raise ValueError('mu bins must be in the range [0,1]')
    
    #check dimensionality of data. currently, points must be 3D.
    k = np.shape(sample1)[-1]
    if k!=3:
        raise ValueError('data must be 3-dimensional.')
    
    #check for input parameter consistency
    if (period is not None) & (np.max(s_bins)>np.min(period)/2.0):
        raise ValueError('cannot calculate for s seperations larger than Lbox/2.')
    if (sample2 is not None) & (sample1.shape[-1]!=sample2.shape[-1]):
        raise ValueError('sample1 and sample2 must have same dimension.')
    if (randoms is None) & (min(period)==np.inf):
        raise ValueError('if no PBCs are specified, randoms must be provided!')
    if estimator not in estimators: 
        raise ValueError('user must specify a supported estimator. supported estimators \
        are:{0}'.value(estimators))
    if (PBCs==True) & (max(period)==np.inf):
        raise ValueError('If a non-infinte PBC is specified, all PBCs must be \
        non-infinte.')
    if (type(do_auto) is not bool) | (type(do_cross) is not bool):
        raise ValueError('do_auto and do_cross keywords must be of type boolean.')

    def random_counts(sample1, sample2, randoms, s_bins, mu_bins, period,\
                      PBCs, k, num_threads, do_RR, do_DR):
        """
        Count random pairs.  There are three high level branches: 

            1. no PBCs w/ randoms.

            2. PBCs w/ randoms

            3. PBCs and analytical randoms

        There are also logical bits to do RR and DR pair counts, as not all estimators 
        need one or the other, and not doing these can save a lot of calculation.
        
        If no randoms are passes, calculate analytical randoms; otherwise, do it the old 
        fashioned way.
        """
        def spherical_sector_volume(s,mu):
            """
            This function is used to calculate analytical randoms.
            
            Calculate the volume of a spherical sector, used for the analytical randoms.
            https://en.wikipedia.org/wiki/Spherical_sector
            
            Note that the extra *2 is to get the reflection.
            """
            theta = np.arcsin(mu)
            vol = (2.0*np.pi/3.0) * np.outer((s**3.0),(1.0-np.cos(theta)))*2.0
            return vol
        
        #No PBCs, randoms must have been provided.
        if PBCs==False:
            RR = s_mu_npairs(randoms, randoms, s_bins, mu_bins, period=period,\
                             num_threads=num_threads)
            RR = np.diff(np.diff(RR,axis=0),axis=1)
            D1R = s_mu_npairs(sample1, randoms, s_bins, mu_bins, period=period,\
                              num_threads=num_threads)
            D1R = np.diff(np.diff(D1R,axis=0),axis=1)
            if np.all(sample1 == sample2): #calculating the cross-correlation
                D2R = None
            else:
                D2R = s_mu_npairs(sample2, randoms, s_bins, mu_bins, period=period,\
                                  num_threads=num_threads)
                D2R = np.diff(np.diff(D2R,axis=0),axis=1)
            
            return D1R, D2R, RR
        #PBCs and randoms.
        elif randoms is not None:
            if do_RR==True:
                RR = s_mu_npairs(randoms, randoms, s_bins, mu_bins, period=period,\
                                 num_threads=num_threads)
                RR = np.diff(np.diff(RR,axis=0),axis=1)
            else: RR=None
            if do_DR==True:
                D1R = s_mu_npairs(sample1, randoms, s_bins, mu_bins, period=period,\
                                  num_threads=num_threads)
                D1R = np.diff(np.diff(D1R,axis=0),axis=1)
            else: D1R=None
            if np.all(sample1 == sample2): #calculating the cross-correlation
                D2R = None
            else:
                if do_DR==True:
                    D2R = s_mu_npairs(sample2, randoms, s_bins, mu_bins, period=period,\
                                      num_threads=num_threads)
                    D2R = np.diff(np.diff(D2R,axis=0),axis=1)
                else: D2R=None
            
            return D1R, D2R, RR
        #PBCs and no randoms--calculate randoms analytically.
        elif randoms is None:
            #do volume calculations
            dv = spherical_sector_volume(s_bins,mu_bins)
            dv = np.diff(dv, axis=1) #volume of wedges
            dv = np.diff(dv, axis=0) #volume of wedge 'pieces'
            global_volume = period.prod() #sexy
            
            #calculate randoms for sample1
            N1 = np.shape(sample1)[0]
            rho1 = N1/global_volume
            D1R = (N1-1.0)*(dv*rho1) #read note about pair counter
            
            #if not calculating cross-correlation, set RR exactly equal to D1R.
            if np.all(sample1 == sample2):
                D2R = None
                RR = D1R #in the analytic case, for the auto-correlation, DR==RR.
            else: #if there is a sample2, calculate randoms for it.
                N2 = np.shape(sample2)[0]
                rho2 = N2/global_volume
                D2R = (N2-1.0)*(dv*rho2) #read note about pair counter
                #calculate the random-random pairs.
                #RR is only the RR for the cross-correlation when using analytical randoms
                #for the non-cross case, DR==RR (in analytical world).
                NR = N1*N2
                rhor = NR/global_volume
                RR = (dv*rhor)

            return D1R, D2R, RR
        else:
            raise ValueError('Un-supported combination of PBCs and randoms provided.')
    
    def pair_counts(sample1, sample2, s_bins, mu_bins, period,\
                    N_thread, do_auto, do_cross, do_DD):
        """
        Count data pairs.
        """
        if do_auto==True:
            D1D1 = s_mu_npairs(sample1, sample1, s_bins, mu_bins, period=period,\
                               num_threads=num_threads)
            D1D1 = np.diff(np.diff(D1D1,axis=0),axis=1)
        else: 
            D1D1=None
            D2D2=None
            
        if np.all(sample1 == sample2):
            D1D2 = D1D1
            D2D2 = D1D1
        else:
            if do_cross==True:
                D1D2 = s_mu_npairs(sample1, sample2, s_bins, mu_bins, period=period,\
                                   num_threads=num_threads)
                D1D2 = np.diff(np.diff(D1D2,axis=0),axis=1)
            else: D1D2=None
            if do_auto==True:
                D2D2 = s_mu_npairs(sample2, sample2, s_bins, mu_bins, period=period,\
                                   num_threads=num_threads)
                D2D2 = np.diff(np.diff(D2D2,axis=0),axis=1)
            else: D2D2=None

        return D1D1, D1D2, D2D2
    
    #what needs to be done?
    do_DD, do_DR, do_RR = _TP_estimator_requirements(estimator)
    
    #how many points (for normalization purposes)
    if randoms is not None:
        N1 = len(sample1)
        N2 = len(sample2)
        NR = len(randoms)
    else: #this is taken care of in the random_pairs analytical randoms section.
        N1 = 1.0
        N2 = 1.0
        NR = 1.0
    
    #count pairs!
    D1D1,D1D2,D2D2 = pair_counts(sample1, sample2, s_bins, mu_bins, period,\
                                 num_threads, do_auto, do_cross, do_DD)
    D1R, D2R, RR = random_counts(sample1, sample2, randoms, s_bins, mu_bins, period,\
                                 PBCs, k, num_threads, do_RR, do_DR)
    
    #return results.  remember to reverse the final result because we used sin(theta_los)
    #bins instead of the user passed in mu = cos(theta_los). 
    if np.all(sample2==sample1):
        xi_11 = _TP_estimator(D1D1,D1R,RR,N1,N1,NR,NR,estimator)[:,::-1]
        return xi_11
    else:
        if (do_auto==True) & (do_cross==True): 
            xi_11 = _TP_estimator(D1D1,D1R,RR,N1,N1,NR,NR,estimator)[:,::-1]
            xi_12 = _TP_estimator(D1D2,D1R,RR,N1,N2,NR,NR,estimator)[:,::-1]
            xi_22 = _TP_estimator(D2D2,D2R,RR,N2,N2,NR,NR,estimator)[:,::-1]
            return xi_11, xi_12, xi_22
        elif (do_cross==True):
            xi_12 = _TP_estimator(D1D2,D1R,RR,N1,N2,NR,NR,estimator)[:,::-1]
            return xi_12
        elif (do_auto==True):
            xi_11 = _TP_estimator(D1D1,D1R,D1R,N1,N1,NR,NR,estimator)[:,::-1]
            xi_22 = _TP_estimator(D2D2,D2R,D2R,N2,N2,NR,NR,estimator)[:,::-1]
            return xi_11


def _list_estimators():
    """
    private internal function.
    
    list available tpcf estimators
    """
    estimators = ['Natural', 'Davis-Peebles', 'Hewett' , 'Hamilton', 'Landy-Szalay']
    return estimators


def _TP_estimator(DD,DR,RR,ND1,ND2,NR1,NR2,estimator):
    """
    private internal function.
    
    two point correlation function estimator
    
    note: jackknife_tpcf uses its own intenral version, this is not totally ideal.
    """
    if estimator == 'Natural':
        factor = ND1*ND2/(NR1*NR2)
        #DD/RR-1
        xi = (1.0/factor)*DD/RR - 1.0
    elif estimator == 'Davis-Peebles':
        factor = ND1*ND2/(ND1*NR2)
        #DD/DR-1
        xi = (1.0/factor)*DD/DR - 1.0
    elif estimator == 'Hewett':
        factor1 = ND1*ND2/(NR1*NR2)
        factor2 = ND1*NR2/(NR1*NR2)
        #(DD-DR)/RR
        xi = (1.0/factor1)*DD/RR - (1.0/factor2)*DR/RR
    elif estimator == 'Hamilton':
        #DDRR/DRDR-1
        xi = (DD*RR)/(DR*DR) - 1.0
    elif estimator == 'Landy-Szalay':
        factor1 = ND1*ND2/(NR1*NR2)
        factor2 = ND1*NR2/(NR1*NR2)
        #(DD - 2.0*DR + RR)/RR
        xi = (1.0/factor1)*DD/RR - (1.0/factor2)*2.0*DR/RR + 1.0
    else: 
        raise ValueError("unsupported estimator!")
    return xi


def _TP_estimator_requirements(estimator):
    """
    private internal function.
    
    return booleans indicating which pairs need to be counted for the chosen estimator
    """
    if estimator == 'Natural':
        do_DD = True
        do_DR = False
        do_RR = True
    elif estimator == 'Davis-Peebles':
        do_DD = True
        do_DR = True
        do_RR = False
    elif estimator == 'Hewett':
        do_DD = True
        do_DR = True
        do_RR = True
    elif estimator == 'Hamilton':
        do_DD = True
        do_DR = True
        do_RR = True
    elif estimator == 'Landy-Szalay':
        do_DD = True
        do_DR = True
        do_RR = True
    else: 
        raise ValueError("unsupported estimator!")
    return do_DD, do_DR, do_RR


