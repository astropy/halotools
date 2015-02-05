#Duncan Campbell
#August 27, 2014
#Yale University

#Contributions by Shany Danieli
#December 10, 2014
#Yale University

""" 
Functions that compute two-point statistics of a mock galaxy catalog. 
"""

from __future__ import division
import sys

__all__=['two_point_correlation_function','two_point_correlation_function_jackknife',
         'angular_two_point_correlation_function','Delta_Sigma']

####import modules########################################################################
import numpy as np
from math import pi, gamma
import spatial.geometry as geometry
from multiprocessing import Pool
try: from pair_counters.mpipairs import npairs, wnpairs, specific_wnpairs, jnpairs
except ImportError:
    print("MPI functionality not available.")
    from pair_counters.kdpairs import npairs, wnpairs, specific_wnpairs, jnpairs

####define wrapper functions for pair counters to facilitate parallelization##############
#straight pair counter
def _npairs_wrapper(tup):
    return npairs(*tup)
#weighted pair counter
def _wnpairs_wrapper(tup):
    return wnpairs(*tup)
#specific weighted pair counter
def _specific_wnpairs_wrapper(tup):
    return specific_wnpairs(*tup)
#jackknife pair counter
def _jnpairs_wrapper(tup):
    return jnpairs(*tup)
##########################################################################################

def two_point_correlation_function(sample1, rbins, sample2 = None, randoms=None, 
                                   period = None, max_sample_size=int(1e6),
                                   do_auto=True, do_cross=True, estimator='Natural', 
                                   N_threads=1, comm=None):
    """ Calculate the two-point correlation function. 
    
    Parameters 
    ----------
    sample1 : array_like
        Npts x k numpy array containing k-d positions of Npts. 
    
    rbins : array_like
        numpy array of boundaries defining the bins in which pairs are counted. 
        len(rbins) = Nrbins + 1.
    
    sample2 : array_like, optional
        Npts x k numpy array containing k-d positions of Npts.
    
    randoms : array_like, optional
        Nran x k numpy array containing k-d positions of Npts.
    
    period: array_like, optional
        length k array defining axis-aligned periodic boundary conditions. If only 
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*k).
        If none, PBCs are set to infinity.
    
    max_sample_size : int, optional
        Defines maximum size of the sample that will be passed to the pair counter. 
        
        If sample size exeeds max_sample_size, the sample will be randomly down-sampled 
        such that the subsamples are (roughly) equal to max_sample_size. 
        Subsamples will be passed to the pair counter in a simple loop, 
        and the correlation function will be estimated from the median pair counts in each bin.
    
    estimator: string, optional
        options: 'Natural', 'Davis-Peebles', 'Hewett' , 'Hamilton', 'Landy-Szalay'
    
    N_thread: int, optional
        number of threads to use in calculation.
    
    comm: mpi Intracommunicator object, optional
    
    do_auto: boolean, optional
        do auto-correlation?
    
    do_cross: boolean, optional
        do cross-correlation?

    Returns 
    -------
    correlation_function : array_like
        array containing correlation function :math:`\\xi` computed in each of the Nrbins 
        defined by input `rbins`.

        :math:`1 + \\xi(r) \equiv DD / RR`, 
        where `DD` is calculated by the pair counter, and RR is counted by the internally 
        defined `randoms` if no randoms are passed as an argument.

        If sample2 is passed as input, three arrays of length Nrbins are returned: two for
        each of the auto-correlation functions, and one for the cross-correlation function. 

    """
    #####notes#####
    #The pair counter returns all pairs, including self pairs and double counted pairs 
    #with separations less than r. If PBCs are set to none, then period=np.inf. This makes
    #all distance calculations equivalent to the non-periodic case, while using the same 
    #periodic distance functions within the pair counter.
    ###############
    
    #parallel processing things...
    if comm!=None:
        rank=comm.rank
    else: rank=0
    if N_threads>1:
        pool = Pool(N_threads)
    
    def list_estimators(): #I would like to make this accessible from the outside. Know how?
        estimators = ['Natural', 'Davis-Peebles', 'Hewett' , 'Hamilton', 'Landy-Szalay']
        return estimators
    estimators = list_estimators()
    
    #process input parameters
    sample1 = np.asarray(sample1)
    if sample2 != None: sample2 = np.asarray(sample2)
    else: sample2 = sample1
    if randoms != None: randoms = np.asarray(randoms)
    rbins = np.asarray(rbins)
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
    
    if np.shape(rbins) == ():
        rbins = np.array([rbins])
    
    k = np.shape(sample1)[-1] #dimensionality of data
    
    #check for input parameter consistency
    if (period != None) & (np.max(rbins)>np.min(period)/2.0):
        raise ValueError('Cannot calculate for seperations larger than Lbox/2.')
    if (sample2 != None) & (sample1.shape[-1]!=sample2.shape[-1]):
        raise ValueError('Sample 1 and sample 2 must have same dimension.')
    if (randoms == None) & (min(period)==np.inf):
        raise ValueError('If no PBCs are specified, randoms must be provided.')
    if estimator not in estimators: 
        raise ValueError('Must specify a supported estimator. Supported estimators are:{0}'
        .value(estimators))
    if (PBCs==True) & (max(period)==np.inf):
        raise ValueError('If a non-infinte PBC specified, all PBCs must be non-infinte.')

    #If PBCs are defined, calculate the randoms analytically. Else, the user must specify 
    #randoms and the pair counts are calculated the old fashion way.
    def random_counts(sample1, sample2, randoms, rbins, period, PBCs, k, N_threads, do_RR, do_DR, comm):
        """
        Count random pairs.  There are three high level branches: 
            1. no PBCs w/ randoms.
            2. PBCs w/ randoms
            3. PBCs and analytical randoms
        Within each of those branches there are 3 branches to use:
            1. MPI
            2. no threads
            3. threads
        There are also logical bits to do RR and DR pair counts, as not all estimators 
        need one or the other, and not doing these can save a lot of calculation.
        """
        def nball_volume(R,k):
            """
            Calculate the volume of a n-shpere.  This is used for the analytical randoms.
            """
            return (pi**(k/2.0)/gamma(k/2.0+1.0))*R**k
        
        #No PBCs, randoms must have been provided.
        if PBCs==False:
            if comm!=None:
                if do_RR==True:
                    if rank==0: print('Running MPI pair counter for RR with {0} processes.'.format(comm.size))
                    RR = npairs(randoms, randoms, rbins, comm=comm)
                    RR = np.diff(RR)
                else: RR=None
                if do_DR==True:
                    if rank==0: print('Running MPI pair counter for D1R with {0} processes.'.format(comm.size))
                    D1R = npairs(sample1, randoms, rbins, comm=comm)
                    D1R = np.diff(D1R)
                else: D1R=None
                if np.all(sample1 == sample2): #calculating the cross-correlation
                    D2R = None
                else:
                    if do_DR==True:
                        if rank==0: print('Running MPI pair counter for D2R with {0} processes.'.format(comm.size))
                        D2R = npairs(sample2, randoms, rbins, comm=comm)
                        D2R = np.diff(D2R)
                    else: D2R=None
            elif N_threads==1:
                RR = npairs(randoms, randoms, rbins, period=period)
                RR = np.diff(RR)
                D1R = npairs(sample1, randoms, rbins, period=period)
                D1R = np.diff(D1R)
                if np.all(sample1 == sample2): #calculating the cross-correlation
                    D2R = None
                else:
                    D2R = npairs(sample2, randoms, rbins, period=period)
                    D2R = np.diff(D2R)
            else:
                args = [[chunk,randoms,rbins,period] for chunk in np.array_split(randoms,N_threads)]
                RR = np.sum(pool.map(_npairs_wrapper,args),axis=0)
                RR = np.diff(RR)
                args = [[chunk,randoms,rbins,period] for chunk in np.array_split(sample1,N_threads)]
                D1R = np.sum(pool.map(_npairs_wrapper,args),axis=0)
                D1R = np.diff(D1R)
                if np.all(sample1 == sample2): #calculating the cross-correlation
                    D2R = None
                else:
                    args = [[chunk,randoms,rbins,period] for chunk in np.array_split(sample2,N_threads)]
                    D2R = np.sum(pool.map(_npairs_wrapper,args),axis=0)
                    D2R = np.diff(D2R)
            
            return D1R, D2R, RR
        #PBCs and randoms.
        elif randoms != None:
            if comm!=None:
                if do_RR==True:
                    if rank==0: print('Running MPI pair counter for RR with {0} processes.'.format(comm.size))
                    RR = npairs(randoms, randoms, rbins, period=period, comm=comm)
                    RR = np.diff(RR)
                else: RR=None
                if do_DR==True:
                    if rank==0: print('Running MPI pair counter for D1R with {0} processes.'.format(comm.size))
                    D1R = npairs(sample1, randoms, rbins, period=period, comm=comm)
                    D1R = np.diff(D1R)
                else: D1R=None
                if np.all(sample1 == sample2): #calculating the cross-correlation
                    D2R = None
                else:
                    if do_DR==True:
                        if rank==0: print('Running MPI pair counter for D2R with {0} processes.'.format(comm.size))
                        D2R = npairs(sample2, randoms, rbins, period=period, comm=comm)
                        D2R = np.diff(D2R)
                    else: D2R=None
            elif N_threads==1:
                if do_RR==True:
                    RR = npairs(randoms, randoms, rbins, period=period)
                    RR = np.diff(RR)
                else: RR=None
                if do_DR==True:
                    D1R = npairs(sample1, randoms, rbins, period=period)
                    D1R = np.diff(D1R)
                else: D1R=None
                if np.all(sample1 == sample2): #calculating the cross-correlation
                    D2R = None
                else:
                    if do_DR==True:
                        D2R = npairs(sample2, randoms, rbins, period=period)
                        D2R = np.diff(D2R)
                    else: D2R=None
            else:
                if do_RR==True:
                    args = [[chunk,randoms,rbins,period] for chunk in np.array_split(randoms,N_threads)]
                    RR = np.sum(pool.map(_npairs_wrapper,args),axis=0)
                    RR = np.diff(RR)
                else: RR=None
                if do_DR==True:
                    args = [[chunk,randoms,rbins,period] for chunk in np.array_split(sample1,N_threads)]
                    D1R = np.sum(pool.map(_npairs_wrapper,args),axis=0)
                    D1R = np.diff(D1R)
                else: D1R=None
                if np.all(sample1 == sample2): #calculating the cross-correlation
                    D2R = None
                else:
                    if do_DR==True:
                        args = [[chunk,randoms,rbins,period] for chunk in np.array_split(sample2,N_threads)]
                        D2R = np.sum(pool.map(_npairs_wrapper,args),axis=0)
                        D2R = np.diff(D2R)
                    else: D2R=None
            
            return D1R, D2R, RR
        #PBCs and no randoms--calculate randoms analytically.
        elif randoms == None:
            #do volume calculations
            dv = nball_volume(rbins,k) #volume of spheres
            dv = np.diff(dv) #volume of shells
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
    
    def pair_counts(sample1, sample2, rbins, period, N_thread, do_auto, do_cross, do_DD, comm):
        """
        Count data pairs.
        """
        if comm!=None:
            if do_auto==True:
                if rank==0: print('Running MPI pair counter for D1D1 with {0} processes.'.format(comm.size))
                D1D1 = npairs(sample1, sample1, rbins, period=period, comm=comm)
                D1D1 = np.diff(D1D1)
            else: D1D1=None
            if np.all(sample1 == sample2):
                D1D2 = D1D1
                D2D2 = D1D1
            else:
                if do_cross==True:
                    if rank==0: print('Running MPI pair counter for D1D2 with {0} processes.'.format(comm.size))
                    D1D2 = npairs(sample1, sample2, rbins, period=period, comm=comm)
                    D1D2 = np.diff(D1D2)
                else: D1D2=None
                if do_auto==True:
                    if rank==0: print('Running MPI pair counter for D2D2 with {0} processes.'.format(comm.size))
                    D2D2 = npairs(sample2, sample2, rbins, period=period, comm=comm)
                    D2D2 = np.diff(D2D2)
                else: D2D2=False
        elif N_threads==1:
            D1D1 = npairs(sample1, sample1, rbins, period=period)
            D1D1 = np.diff(D1D1)
            if np.all(sample1 == sample2):
                D1D2 = D1D1
                D2D2 = D1D1
            else:
                D1D2 = npairs(sample1, sample2, rbins, period=period)
                D1D2 = np.diff(D1D2)
                D2D2 = npairs(sample2, sample2, rbins, period=period)
                D2D2 = np.diff(D2D2)
        else:
            args = [[chunk,sample1,rbins,period] for chunk in np.array_split(sample1,N_threads)]
            D1D1 = np.sum(pool.map(_npairs_wrapper,args),axis=0)
            D1D1 = np.diff(D1D1)
            if np.all(sample1 == sample2):
                D1D2 = D1D1
                D2D2 = D1D1
            else:
                args = [[chunk,sample2,rbins,period] for chunk in np.array_split(sample1,N_threads)]
                D1D2 = np.sum(pool.map(_npairs_wrapper,args),axis=0)
                D1D2 = np.diff(D1D2)
                args = [[chunk,sample2,rbins,period] for chunk in np.array_split(sample2,N_threads)]
                D2D2 = np.sum(pool.map(_npairs_wrapper,args),axis=0)
                D2D2 = np.diff(D2D2)

        return D1D1, D1D2, D2D2
        
    def TP_estimator(DD,DR,RR,ND1,ND2,NR1,NR2,estimator):
        """
        two point correlation function estimator
        """
        if estimator == 'Natural':
            factor = ND1*ND2/(NR1*NR2)
            xi = (1.0/factor)*DD/RR - 1.0 #DD/RR-1
        elif estimator == 'Davis-Peebles':
            factor = ND1*ND2/(ND1*NR2)
            xi = (1.0/factor)*DD/DR - 1.0 #DD/DR-1
        elif estimator == 'Hewett':
            factor1 = ND1*ND2/(NR1*NR2)
            factor2 = ND1*NR2/(NR1*NR2)
            xi = (1.0/factor1)*DD/RR - (1.0/factor2)*DR/RR #(DD-DR)/RR
        elif estimator == 'Hamilton':
            xi = (DD*RR)/(DR*DR) - 1.0 #DDRR/DRDR-1
        elif estimator == 'Landy-Szalay':
            factor1 = ND1*ND2/(NR1*NR2)
            factor2 = ND1*NR2/(NR1*NR2)
            xi = (1.0/factor1)*DD/RR - (1.0/factor2)*2.0*DR/RR + 1.0 #(DD - 2.0*DR + RR)/RR
        else: 
            raise ValueError("unsupported estimator!")
        return xi
    
    def TP_estimator_requirements(estimator):
        """
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
    
    do_DD, do_DR, do_RR = TP_estimator_requirements(estimator)
              
    if np.all(randoms != None):
        N1 = len(sample1)
        N2 = len(sample2)
        NR = len(randoms)
    else: 
        N1 = 1.0
        N2 = 1.0
        NR = 1.0
    
    #count pairs
    if rank==0: print('counting data pairs...')
    D1D1,D1D2,D2D2 = pair_counts(sample1, sample2, rbins, period, N_threads, do_auto, do_cross, do_DD, comm)
    if rank==0: print('counting random pairs...')
    D1R, D2R, RR = random_counts(sample1, sample2, randoms, rbins, period, PBCs, k, N_threads, do_RR, do_DR, comm)
    if rank==0: print('done counting pairs') 
    
    if np.all(sample2==sample1):
        xi_11 = TP_estimator(D1D1,D1R,RR,N1,N1,NR,NR,estimator)
        return xi_11
    else:
        if (do_auto==True) & (do_cross==True): 
            xi_11 = TP_estimator(D1D1,D1R,RR,N1,N1,NR,NR,estimator)
            xi_12 = TP_estimator(D1D2,D1R,RR,N1,N2,NR,NR,estimator)
            xi_22 = TP_estimator(D2D2,D2R,RR,N2,N2,NR,NR,estimator)
            return xi_11, xi_12, xi_22
        elif (do_cross==True):
            xi_12 = TP_estimator(D1D2,D1R,RR,N1,N2,NR,NR,estimator)
            return xi_12
        elif (do_auto==True):
            xi_11 = TP_estimator(D1D1,D1R,D1R,N1,N1,NR,NR,estimator)
            xi_22 = TP_estimator(D2D2,D2R,D2R,N2,N2,NR,NR,estimator)
            return xi_11


def two_point_correlation_function_jackknife(sample1, randoms, rbins, Nsub=10,
                                             Lbox=[250.0,250.0,250.0], sample2 = None,
                                             period = None, max_sample_size=int(1e6),
                                             do_auto=True, do_cross=True,
                                             estimator='Natural', N_threads=1, comm=None):
    """
    Calculate the two-point correlation function with jackknife errors. 
    
    Parameters 
    ----------
    sample1 : array_like
        Npts x k numpy array containing k-d positions of Npts.
    
    randoms : array_like
        Nran x k numpy array containing k-d positions of Npts. 
    
    rbins : array_like
        numpy array of boundaries defining the bins in which pairs are counted. 
        len(rbins) = Nrbins + 1.
    
    Nsub : array_like, optional
        numpy array of number of divisions along each dimension defining jackknife subvolumes
        if single integer is given, assumed to be equivalent for each dimension
    
    Lbox : array_like, optional
        length of data volume along each dimension
    
    sample2 : array_like, optional
        Npts x k numpy array containing k-d positions of Npts.
    
    period: array_like, optional
        length k array defining axis-aligned periodic boundary conditions. If only 
        one number, Lbox, is specified, period is assumed to be np.array([Lbox]*k).
        If none, PBCs are set to infinity.
    
    max_sample_size : int, optional
        Defines maximum size of the sample that will be passed to the pair counter. 
        
        If sample size exeeds max_sample_size, the sample will be randomly down-sampled 
        such that the subsamples are (roughly) equal to max_sample_size. 
        Subsamples will be passed to the pair counter in a simple loop, 
        and the correlation function will be estimated from the median pair counts in each bin.
    
    estimator: string, optional
        options: 'Natural', 'Davis-Peebles', 'Hewett' , 'Hamilton', 'Landy-Szalay'
    
    N_thread: int, optional
        number of threads to use in calculation.
    
    comm: mpi Intracommunicator object, optional

    Returns 
    -------
    correlation_function(s), cov_matrix : array_like
        array containing correlation function :math:`\\xi(r)` computed in each of the Nrbins 
        defined by input `rbins`.
        Nrbins x Nrbins array containing the covariance matrix of `\\xi(r)`

    """
    
    #parallel processing things
    if comm!=None:
        rank=comm.rank
    else: rank=0
    if N_threads>1:
        pool = Pool(N_threads)
    
    def list_estimators(): #I would like to make this accessible from the outside. Know how?
        estimators = ['Natural', 'Davis-Peebles', 'Hewett' , 'Hamilton', 'Landy-Szalay']
        return estimators
    estimators = list_estimators()
    
    #process input parameters
    sample1 = np.asarray(sample1)
    if sample2 != None: sample2 = np.asarray(sample2)
    else: sample2 = sample1
    randoms = np.asarray(randoms)
    rbins = np.asarray(rbins)
    if type(Nsub) is int: Nsub = np.array([Nsub]*np.shape(sample1)[-1])
    else: Nsub = np.asarray(Nsub)
    if type(Lbox) in (int,float): Lbox = np.array([Lbox]*np.shape(sample1)[-1])
    else: Lbox = np.asarray(Lbox)
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
    if len(randoms)>max_sample_size:
        inds = np.arange(0,len(randoms))
        np.random.shuffle(inds)
        inds = inds[0:max_sample_size]
        sample1 = randoms[inds]
        print('down sampling randoms...')
    if np.shape(Nsub)[0]!=np.shape(sample1)[-1]:
        raise ValueError("Nsub should have shape (k,) or be a single integer")
    
    if np.shape(rbins) == ():
        rbins = np.array([rbins])
    
    k = np.shape(sample1)[-1] #dimensionality of data
    N1 = len(sample1)
    N2 = len(sample2)
    Nran = len(randoms)
    
    #check for input parameter consistency
    if (period != None) & (np.max(rbins)>np.min(period)/2.0):
        raise ValueError('Cannot calculate for seperations larger than Lbox/2.')
    if (sample2 != None) & (sample1.shape[-1]!=sample2.shape[-1]):
        raise ValueError('Sample 1 and sample 2 must have same dimension.')
    if estimator not in estimators: 
        raise ValueError('Must specify a supported estimator. Supported estimators are:{0}'
        .value(estimators))
    if (PBCs==True) & (max(period)==np.inf):
        raise ValueError('If a non-infinte PBC specified, all PBCs must be non-infinte.')
    
    def get_subvolume_labels(sample1, sample2, randoms, Nsub, Lbox):
        """
        Split volume into subvolumes, and tag points in subvolumes with integer labels for 
        use in the jackknife calculation.
        
        note: '0' tag should be reserved. It is used in the jackknife calculation to mean
        'full sample'
        """
        
        dL = Lbox/Nsub # length of subvolumes along each dimension
        N_sub_vol = np.prod(Nsub) # total the number of subvolumes
    
        #tag each particle with an integer indicating which jackknife subvolume it is in
        #subvolume indices for the sample1 particle's positions
        index_1 = np.sum(np.floor(sample1/dL)*np.hstack((1,np.cumprod(Nsub[:-1]))),axis=1)+1
        j_index_1 = index_1.astype(int)
        #subvolume indices for the random particle's positions
        index_random = np.sum(np.floor(randoms/dL)*np.hstack((1,np.cumprod(Nsub[:-1]))),axis=1)+1
        j_index_random = index_random.astype(int)
        #subvolume indices for the sample2 particle's positions
        index_2 = np.sum(np.floor(sample2/dL)*np.hstack((1,np.cumprod(Nsub[:-1]))),axis=1)+1
        j_index_2 = index_2.astype(int)
        
        return j_index_1, j_index_2, j_index_random, N_sub_vol
    
    def jnpair_counts(sample1, sample2, j_index_1, j_index_2, N_sub_vol, rbins,\
                      period, N_thread, do_auto, do_cross, do_DD, comm):
        """
        Count jackknife data pairs: DD
        """
        if comm!=None:
            if do_auto==True:
                if rank==0: print('Running MPI pair counter for D1D1 with {0} processes.'.format(comm.size))
                D1D1 = jnpairs(sample1, sample1, rbins, period=period,\
                               weights1=j_index_1, weights2=j_index_1,\
                               N_vol_elements=N_sub_vol, comm=comm)
                D1D1 = np.diff(D1D1,axis=1)
            else: D1D1=None
            if np.all(sample1 == sample2):
                D1D2 = D1D1
                D2D2 = D1D1
            else:
                if do_cross==True:
                    if rank==0: print('Running MPI pair counter for D1D2 with {0} processes.'.format(comm.size))
                    D1D2 = jnpairs(sample1, sample2, rbins, period=period,\
                                   weights1=j_index_2, weights2=j_index_2,\
                                   N_vol_elements=N_sub_vol, comm=comm)
                    D1D2 = np.diff(D1D2,axis=1)
                else: D1D2=None
                if do_auto==True:
                    if rank==0: print('Running MPI pair counter for D2D2 with {0} processes.'.format(comm.size))
                    D2D2 = jnpairs(sample2, sample2, rbins, period=period,\
                                   weights1=j_index_2, weights2=j_index_2,\
                                   N_vol_elements=N_sub_vol, comm=comm)
                    D2D2 = np.diff(D2D2,axis=1)
                else: D2D2=False
        elif N_threads==1:
            D1D1 = jnpairs(sample1, sample1, rbins, period=period,\
                       weights1=j_index_1, weights2=j_index_1, N_vol_elements=N_sub_vol)
            D1D1 = np.diff(D1D1,axis=1)
            if np.all(sample1 == sample2):
                D1D2 = D1D1
                D2D2 = D1D1
            else:
                D1D2 = D1D1
                D2D2 = D1D1
                D1D2 = jnpairs(sample1, sample2, rbins, period=period,\
                           weights1=j_index_1, weights2=j_index_2, N_vol_elements=N_sub_vol)
                D1D2 = np.diff(D1D2,axis=1)
                D2D2 = jnpairs(sample2, sample2, rbins, period=period,\
                           weights1=j_index_2, weights2=j_index_2, N_vol_elements=N_sub_vol)
                D2D2 = np.diff(D2D2,axis=1)
        else:
            inds1 = np.arange(0,len(sample1)) #array which is just indices into sample1
            inds2 = np.arange(0,len(sample2)) #array which is just indices into sample2
            args = [[sample1[chunk],sample1,rbins,period,j_index_1[chunk],j_index_1,N_sub_vol]\
                    for chunk in np.array_split(inds1,N_threads)]
            D1D1 = np.sum(pool.map(_jnpairs_wrapper,args),axis=0)
            D1D1 = np.diff(D1D1,axis=1)
            if np.all(sample1 == sample2):
                D1D2 = D1D1
                D2D2 = D1D1
            else:
                args = [[sample1[chunk],sample2,rbins,period,j_index_1[chunk],j_index_2,N_sub_vol]\
                        for chunk in np.array_split(inds1,N_threads)]
                D1D2 = np.sum(pool.map(_jnpairs_wrapper,args),axis=0)
                D1D2 = np.diff(D1D2,axis=1)
                args = [[sample2[chunk],sample2,rbins,period,j_index_2[chunk],j_index_2,N_sub_vol]\
                        for chunk in np.array_split(inds2,N_threads)]
                D2D2 = np.sum(pool.map(_jnpairs_wrapper,args),axis=0)
                D2D2 = np.diff(D2D2,axis=1)

        return D1D1, D1D2, D2D2
    
    def jrandom_counts(sample, randoms, j_index, j_index_randoms, N_sub_vol, rbins,\
                       period, N_thread, do_DR, do_RR, comm):
        """
        Count jackknife random pairs: DR, RR
        """
        
        if comm!=None:
            if do_DR==True:
                DR = jnpairs(sample, randoms, rbins, period=period,\
                           weights1=j_index, weights2=j_index_randoms,\
                           N_vol_elements=N_sub_vol, comm=comm)
                DR = np.diff(DR,axis=1)
            else: DR=None
            if do_RR==True:
                RR = jnpairs(randoms, randoms, rbins, period=period,\
                             weights1=j_index_randoms, weights2=j_index_randoms,\
                             N_vol_elements=N_sub_vol, comm=comm)
                RR = np.diff(RR,axis=1)
            else: RR=None
        elif N_threads==1:
            if do_DR==True:
                DR = jnpairs(sample, randoms, rbins, period=period,\
                           weights1=j_index, weights2=j_index_randoms,\
                           N_vol_elements=N_sub_vol)
                DR = np.diff(DR,axis=1)
            else: DR=None
            if do_RR==True:
                RR = jnpairs(randoms, randoms, rbins, period=period,\
                             weights1=j_index_randoms, weights2=j_index_randoms,\
                             N_vol_elements=N_sub_vol)
                RR = np.diff(RR,axis=1)
            else: RR=None
        else:
            inds1 = np.arange(0,len(sample)) #array which is just indices into sample1
            inds2 = np.arange(0,len(randoms)) #array which is just indices into sample2
            if do_DR == True:
                args = [[sample[chunk],randoms,rbins,period,j_index[chunk],j_index_randoms,N_sub_vol]\
                    for chunk in np.array_split(inds1,N_threads)]
                DR = np.sum(pool.map(_jnpairs_wrapper,args),axis=0)
                DR = np.diff(DR,axis=1)
            else: DR = None
            if do_RR==True:
                args = [[randoms[chunk],randoms,rbins,period,j_index_randoms[chunk],j_index_randoms,N_sub_vol]\
                       for chunk in np.array_split(inds2,N_threads)]
                RR = np.sum(pool.map(_jnpairs_wrapper,args),axis=0)
                RR = np.diff(RR,axis=1)
            else: RR=None

        return DR, RR
        
    def TP_estimator(DD,DR,RR,ND1,ND2,NR1,NR2,estimator):
        """
        two point correlation function estimator
        """
        if estimator == 'Natural':
            factor = ND1*ND2/(NR1*NR2)
            xi = (1.0/factor)*DD/RR - 1.0 #DD/RR-1
        elif estimator == 'Davis-Peebles':
            factor = ND1*ND2/(ND1*NR2)
            xi = (1.0/factor)*DD/DR - 1.0 #DD/DR-1
        elif estimator == 'Hewett':
            factor1 = ND1*ND2/(NR1*NR2)
            factor2 = ND1*NR2/(NR1*NR2)
            xi = (1.0/factor1)*DD/RR - (1.0/factor2)*DR/RR #(DD-DR)/RR
        elif estimator == 'Hamilton':
            xi = (DD*RR)/(DR*DR) - 1.0 #DDRR/DRDR-1
        elif estimator == 'Landy-Szalay':
            factor1 = ND1*ND2/(NR1*NR2)
            factor2 = ND1*NR2/(NR1*NR2)
            xi = (1.0/factor1)*DD/RR - (1.0/factor2)*2.0*DR/RR + 1.0 #(DD - 2.0*DR + RR)/RR
        else: 
            raise ValueError("unsupported estimator!")
        return xi
    
    def TP_estimator_requirements(estimator):
        """
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
    
    def jackknife_errors(sub,full,N_sub_vol):
        """
        Calculate jackknife errors.
        """
        after_subtraction =  sub - full
        squared = after_subtraction**2.0
        error2 = ((N_sub_vol-1)/N_sub_vol)*squared.sum(axis=0)
        error = error2**0.5
        
        return error
    
    def covariance_matrix(sub,full,N_sub_vol):
        """
        Calculate the covariance matrix.
        """
        Nr = full.shape[0] # Nr is the number of radial bins
        cov = np.zeros((Nr,Nr)) # 2D array that keeps the covariance matrix 
        after_subtraction = sub - full
        tmp = 0
        for i in range(Nr):
            for j in range(Nr):
                for k in range(N_sub_vol):
                    tmp = tmp + after_subtraction[k,i]*after_subtraction[k,j]
                cov[i,j] = (((N_sub_vol-1)/N_sub_vol)*tmp)
                tmp = 0
    
        return cov
    
    do_DD, do_DR, do_RR = TP_estimator_requirements(estimator)
    
    N1 = len(sample1)
    N2 = len(sample2)
    NR = len(randoms)
    
    j_index_1, j_index_2, j_index_random, N_sub_vol = \
                               get_subvolume_labels(sample1, sample2, randoms, Nsub, Lbox)
    
    #calculate all the pair counts
    D1D1, D1D2, D2D2 = jnpair_counts(sample1, sample2, j_index_1, j_index_2, N_sub_vol,\
                                     rbins, period, N_threads, do_auto, do_cross, do_DD, comm)
    D1D1_full = D1D1[0,:]
    D1D1_sub = D1D1[1:,:]
    D1D2_full = D1D2[0,:]
    D1D2_sub = D1D2[1:,:]
    D2D2_full = D2D2[0,:]
    D2D2_sub = D2D2[1:,:]
    D1R, RR = jrandom_counts(sample1, randoms, j_index_1, j_index_random, N_sub_vol,\
                             rbins, period, N_threads, do_DR, do_RR, comm)
    if np.all(sample1==sample2):
        D2R=D1R
    else:
        if do_DR==True:
            D2R, RR_dummy= jrandom_counts(sample2, randoms, j_index_2, j_index_random,\
                                      N_sub_vol, rbins, period, N_threads, do_DR, False, comm)
        else: D2R = None
    
    if do_DR==True:    
        D1R_full = D1R[0,:]
        D1R_sub = D1R[1:,:]
        D2R_full = D2R[0,:]
        D2R_sub = D2R[1:,:]
    else:
        D1R_full = None
        D1R_sub = None
        D2R_full = None
        D2R_sub = None
    if do_RR==True:
        RR_full = RR[0,:]
        RR_sub = RR[1:,:]
    else:
        RR_full = None
        RR_sub = None
    
    
    #calculate the correlation function for the full sample
    xi_11_full = TP_estimator(D1D1_full,D1R_full,RR_full,N1,N1,NR,NR,estimator)
    xi_12_full = TP_estimator(D1D2_full,D1R_full,RR_full,N1,N2,NR,NR,estimator)
    xi_22_full = TP_estimator(D2D2_full,D2R_full,RR_full,N2,N2,NR,NR,estimator)
    #calculate the correlation function for the subsamples
    xi_11_sub = TP_estimator(D1D1_sub,D1R_sub,RR_sub,N1,N1,NR,NR,estimator)
    xi_12_sub = TP_estimator(D1D2_sub,D1R_sub,RR_sub,N1,N2,NR,NR,estimator)
    xi_22_sub = TP_estimator(D2D2_sub,D2R_sub,RR_sub,N2,N2,NR,NR,estimator)
    
    #calculate the errors
    xi_11_err = jackknife_errors(xi_11_sub,xi_11_full,N_sub_vol)
    xi_12_err = jackknife_errors(xi_12_sub,xi_12_full,N_sub_vol)
    xi_22_err = jackknife_errors(xi_22_sub,xi_22_full,N_sub_vol)
    
    #calculate the covariance matrix
    xi_11_cov = jackknife_errors(xi_11_sub,xi_11_full,N_sub_vol)
    xi_12_cov = jackknife_errors(xi_12_sub,xi_12_full,N_sub_vol)
    xi_22_cov = jackknife_errors(xi_22_sub,xi_22_full,N_sub_vol)
    
    if np.all(sample1==sample2):
        return xi_11_full,xi_11_cov
    else:
        if (do_auto==True) & (do_cross==True):
            return xi_11_full,xi_12_full,xi_22_full,xi_11_cov,xi_12_cov,xi_22_cov
        elif du_auto==True:
            return xi_11_full,xi_22_full,xi_11_cov,xi_22_cov
        elif do_cross==True:
            return xi_12_full,xi_12_cov


def angular_two_point_correlation_function(sample1, theta_bins, sample2=None, randoms=None, 
                                           max_sample_size=int(1e6),estimator='Natural',
                                           do_auto=True, do_cross=True, N_threads=1, comm=None):
    """ Calculate the angular two-point correlation function. 
    
    Parameters 
    ----------
    sample1 : array_like
        Npts x 2 numpy array containing ra,dec positions of Npts. 
    
    theta_bins : array_like
        numpy array of boundaries defining the bins in which pairs are counted. 
        len(theta_bins) = N_theta_bins + 1.
    
    sample2 : array_like, optional
        Npts x 2 numpy array containing ra,dec positions of Npts.
    
    randoms : array_like, optional
        Nran x 2 numpy array containing ra,dec positions of Npts.
    
    max_sample_size : int, optional
        Defines maximum size of the sample that will be passed to the pair counter. 
        
        If sample size exeeds max_sample_size, the sample will be randomly down-sampled 
        such that the subsamples are (roughly) equal to max_sample_size. 
        Subsamples will be passed to the pair counter in a simple loop, 
        and the correlation function will be estimated from the median pair counts in each bin.
    
    estimator: string, optional
        options: 'Natural', 'Davis-Peebles', 'Hewett' , 'Hamilton', 'Landy-Szalay'
    
    N_thread: int, optional
        number of threads to use in calculation.

    comm: mpi Intracommunicator object, optional
    
    do_auto: boolean, optional
        do auto-correlation?
    
    do_cross: boolean, optional
        do cross-correlation?
    
    Returns 
    -------
    angular correlation_function : array_like
        array containing correlation function :math:`\\xi` computed in each of the Nrbins 
        defined by input `rbins`.

        :math:`1 + \\xi(r) \equiv DD / RR`, 
        where `DD` is calculated by the pair counter, and RR is counted by the internally 
        defined `randoms` if no randoms are passed as an argument.

        If sample2 is passed as input, three arrays of length Nrbins are returned: two for
        each of the auto-correlation functions, and one for the cross-correlation function. 

    """
    #####notes#####
    #The pair counter returns all pairs, including self pairs and double counted pairs 
    #with separations less than r. If PBCs are set to none, then period=np.inf. This makes
    #all distance calculations equivalent to the non-periodic case, while using the same 
    #periodic distance functions within the pair counter.
    ###############
    
    #parallel processing things...
    if comm!=None:
        rank=comm.rank
    else: rank=0
    if N_threads>1:
        pool = Pool(N_threads)
    
    def list_estimators(): #I would like to make this accessible from the outside. Know how?
        estimators = ['Natural', 'Davis-Peebles', 'Hewett' , 'Hamilton', 'Landy-Szalay']
        return estimators
    estimators = list_estimators()
    
    #process input parameters
    sample1 = np.asarray(sample1)
    if np.all(sample2 != None): sample2 = np.asarray(sample2)
    else: sample2 = sample1
    if np.all(randoms != None): 
        randoms = np.asarray(randoms)
        PBCs = False
    else: PBCs = True #assume full sky coverage
    theta_bins = np.asarray(theta_bins)
        
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
    
    if np.shape(theta_bins) == ():
        theta_bins = np.array([theta_bins])
    
    k = 2 #only 2-dimensions: ra,dec
    if np.shape(sample1)[-1] != k:
        raise ValueError('angular correlation function requires 2-dimensional data')
    
    #check for input parameter consistency
    if np.all(sample2 != None) & (sample1.shape[-1]!=sample2.shape[-1]):
        raise ValueError('Sample 1 and sample 2 must have same dimension.')
    if estimator not in estimators: 
        raise ValueError('Must specify a supported estimator. Supported estimators are:{0}'
        .value(estimators))

    #If PBCs are defined, calculate the randoms analytically. Else, the user must specify 
    #randoms and the pair counts are calculated the old fashion way.
    def random_counts(sample1, sample2, randoms, theta_bins, PBCs, N_threads, do_RR, do_DR, comm):
        """
        Count random pairs.
        """
        def cap_area(C):
            """
            Calculate angular area of a spherical cap with chord length c
            """
            theta = 2.0*np.arcsin(C/2.0)
            return 2.0*np.pi*(1.0-np.cos(theta))
        
        #No PBCs, randoms must have been provided.
        if PBCs==False:
            if comm!=None:
                if do_RR==True:
                    if rank==0: print('Running MPI pair counter for RR with {0} processes.'.format(comm.size))
                    RR = npairs(randoms, randoms, theta_bins, comm=comm)
                    RR = np.diff(RR)
                else: RR=None
                if do_DR==True:
                    if rank==0: print('Running MPI pair counter for D1R with {0} processes.'.format(comm.size))
                    D1R = npairs(sample1, randoms, theta_bins, comm=comm)
                    D1R = np.diff(D1R)
                else: D1R=None
                if np.all(sample1 == sample2): #calculating the cross-correlation
                    D2R = None
                else:
                    print('manually skipping D2R right now.')
                    if True==False:
                    #if do_DR==True:
                        if rank==0: print('Running MPI pair counter for D2R with {0} processes.'.format(comm.size))
                        D2R = npairs(sample2, randoms, theta_bins, comm=comm)
                        D2R = np.diff(D2R)
                    else: D2R=None
            elif N_threads==1:
                if do_RR==True:
                    RR = npairs(randoms, randoms, theta_bins)
                    RR = np.diff(RR)
                else: RR=None
                if do_DR==True:
                    D1R = npairs(sample1, randoms, theta_bins)
                    D1R = np.diff(D1R)
                else: D1R=None
                if np.all(sample1 == sample2): #calculating the cross-correlation
                    D2R = None
                else:
                    if do_DR==True:
                        D2R = npairs(sample2, randoms, theta_bins)
                        D2R = np.diff(D2R)
                    else: D2R=None
            else:
                if do_RR==True:
                    args = [[chunk,randoms,theta_bins] for chunk in np.array_split(randoms,N_threads)]
                    RR = np.sum(pool.map(_npairs_wrapper,args),axis=0)
                    RR = np.diff(RR)
                else: RR=None
                if do_DR==True:
                    args = [[chunk,randoms,theta_bins] for chunk in np.array_split(sample1,N_threads)]
                    D1R = np.sum(pool.map(_npairs_wrapper,args),axis=0)
                    D1R = np.diff(D1R)
                else: D1R=None
                if np.all(sample1 == sample2): #calculating the cross-correlation
                    D2R = None
                else:
                    if do_DR==True:
                        args = [[chunk,randoms,theta_bins] for chunk in np.array_split(sample2,N_threads)]
                        D2R = np.sum(pool.map(_npairs_wrapper,args),axis=0)
                        D2R = np.diff(D2R)
                    else: D2R=None
            
            return D1R, D2R, RR
        #PBCs and no randoms--calculate randoms analytically.
        elif PBCs==True:
            #do volume calculations
            dv = cap_area(theta_bins) #volume of spheres
            dv = np.diff(dv) #volume of shells
            global_area = 4.0*np.pi
            
            #calculate randoms for sample1
            N1 = np.shape(sample1)[0]
            rho1 = N1/global_area
            D1R = (N1)*(dv*rho1) #read note about pair counter
            
            #if not calculating cross-correlation, set RR exactly equal to D1R.
            if np.all(sample1 == sample2):
                D2R = None
                RR = D1R #in the analytic case, for the auto-correlation, DR==RR.
            else: #if there is a sample2, calculate randoms for it.
                N2 = np.shape(sample2)[0]
                rho2 = N2/global_area
                D2R = N2*(dv*rho2) #read note about pair counter
                #calculate the random-random pairs.
                NR = N1*N2
                rhor = NR/global_area
                RR = (dv*rhor) #RR is only the RR for the cross-correlation.

            return D1R, D2R, RR
        else:
            raise ValueError('Un-supported combination of PBCs and randoms provided.')
    
    def pair_counts(sample1, sample2, theta_bins, N_threads, do_auto, do_cross, do_DD, comm):
        """
        Count data pairs: D1D1, D1D2, D2D2.
        """
        if comm!=None:
            if do_auto==True:
                if rank==0: print('Running MPI pair counter for D1D1 with {0} processes.'.format(comm.size))
                D1D1 = npairs(sample1, sample1, theta_bins, period=None, comm=comm)
                D1D1 = np.diff(D1D1)
            else: D1D1=None
            if np.all(sample1 == sample2):
                D1D2 = D1D1
                D2D2 = D1D1
            else:
                if do_cross==True:
                    if rank==0: print('Running MPI pair counter for D1D2 with {0} processes.'.format(comm.size))
                    D1D2 = npairs(sample1, sample2, theta_bins, period=None, comm=comm)
                    D1D2 = np.diff(D1D2)
                else: D1D2=None
                if do_auto==True:
                    if rank==0: print('Running MPI pair counter for D2D2 with {0} processes.'.format(comm.size))
                    D2D2 = npairs(sample2, sample2, theta_bins, period=None, comm=comm)
                    D2D2 = np.diff(D2D2)
                else: D2D2=False
        elif N_threads==1:
            if do_auto==True:
                D1D1 = npairs(sample1, sample1, theta_bins, period=None)
                D1D1 = np.diff(D1D1)
            else: D1D1=None
            if np.all(sample1 == sample2):
                D1D2 = D1D1
                D2D2 = D1D1
            else:
                if do_cross==True:
                    D1D2 = npairs(sample1, sample2, theta_bins, period=None)
                    D1D2 = np.diff(D1D2)
                else: D1D2=None
                if do_auto==True:
                    D2D2 = npairs(sample2, sample2, theta_bins, period=None)
                    D2D2 = np.diff(D2D2)
                else: D2D2=False
        else:
            if do_auto==True:
                args = [[chunk,sample1,theta_bins] for chunk in np.array_split(sample1,N_threads)]
                D1D1 = np.sum(pool.map(_npairs_wrapper,args),axis=0)
                D1D1 = np.diff(D1D1)
            else: D1D1=None
            if np.all(sample1 == sample2):
                D1D2 = D1D1
                D2D2 = D1D1
            else:
                if do_cross==True:
                    args = [[chunk,sample2,theta_bins] for chunk in np.array_split(sample1,N_threads)]
                    D1D2 = np.sum(pool.map(_npairs_wrapper,args),axis=0)
                    D1D2 = np.diff(D1D2)
                else: D1D2=None
                if do_auto==True:
                    args = [[chunk,sample2,theta_bins] for chunk in np.array_split(sample2,N_threads)]
                    D2D2 = np.sum(pool.map(_npairs_wrapper,args),axis=0)
                    D2D2 = np.diff(D2D2)
                else: D2D2=None

        return D1D1, D1D2, D2D2
        
    def TP_estimator(DD,DR,RR,ND1,ND2,NR1,NR2,estimator):
        """
        two point correlation function estimator
        """
        if estimator == 'Natural':
            factor = ND1*ND2/(NR1*NR2)
            xi = (1.0/factor)*DD/RR - 1.0 #DD/RR-1
        elif estimator == 'Davis-Peebles':
            factor = ND1*ND2/(ND1*NR2)
            xi = (1.0/factor)*DD/DR - 1.0 #DD/DR-1
        elif estimator == 'Hewett':
            factor1 = ND1*ND2/(NR1*NR2)
            factor2 = ND1*NR2/(NR1*NR2)
            xi = (1.0/factor1)*DD/RR - (1.0/factor2)*DR/RR #(DD-DR)/RR
        elif estimator == 'Hamilton':
            xi = (DD*RR)/(DR*DR) - 1.0 #DDRR/DRDR-1
        elif estimator == 'Landy-Szalay':
            factor1 = ND1*ND2/(NR1*NR2)
            factor2 = ND1*NR2/(NR1*NR2)
            xi = (1.0/factor1)*DD/RR - (1.0/factor2)*2.0*DR/RR + 1.0 #(DD - 2.0*DR + RR)/RR
        else: 
            raise ValueError("unsupported estimator!")
        return xi
    
    def TP_estimator_requirements(estimator):
        """
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
              
    if np.all(randoms != None):
        N1 = len(sample1)
        N2 = len(sample2)
        NR = len(randoms)
    else: 
        N1 = 1.0
        N2 = 1.0
        NR = 1.0
    
    do_DD, do_DR, do_RR = TP_estimator_requirements(estimator)
    
    #convert angular coordinates into cartesian coordinates
    from halotools.utils.spherical_geometry import spherical_to_cartesian, chord_to_cartesian
    xyz_1 = np.empty((len(sample1),3))
    xyz_2 = np.empty((len(sample2),3))
    xyz_1[:,0],xyz_1[:,1],xyz_1[:,2] = spherical_to_cartesian(sample1[:,0], sample1[:,1])
    xyz_2[:,0],xyz_2[:,1],xyz_2[:,2] = spherical_to_cartesian(sample2[:,0], sample2[:,1])
    if PBCs==False:
        xyz_randoms = np.empty((len(randoms),3))
        xyz_randoms[:,0],xyz_randoms[:,1],xyz_randoms[:,2] = spherical_to_cartesian(randoms[:,0], randoms[:,1])
    else: xyz_randoms=None
    
    #convert angular bins to cartesian distances
    c_bins = chord_to_cartesian(theta_bins, radians=False)
    
    #count pairs
    if rank==0: print('counting data pairs...')
    D1D1,D1D2,D2D2 = pair_counts(xyz_1, xyz_2, c_bins, N_threads, do_auto, do_cross, do_DD, comm)
    if rank==0: print('counting random pairs...')
    D1R, D2R, RR = random_counts(xyz_1, xyz_2, xyz_randoms, c_bins, PBCs, N_threads, do_RR, do_DR, comm)
    if rank==0: print('done counting pairs')
    
    if rank==0:
        print(D1D2)
        print(D1R)
    
    if np.all(sample2==sample1):
        xi_11 = TP_estimator(D1D1,D1R,RR,N1,N1,NR,NR,estimator)
        return xi_11
    else:
        if (do_auto==True) & (do_cross==True):
            xi_11 = TP_estimator(D1D1,D1R,RR,N1,N1,NR,NR,estimator)
            xi_12 = TP_estimator(D1D2,D1R,RR,N1,N2,NR,NR,estimator)
            xi_22 = TP_estimator(D2D2,D2R,RR,N2,N2,NR,NR,estimator)
            return xi_11, xi_12, xi_22
        elif do_cross==True:
            xi_12 = TP_estimator(D1D2,D1R,RR,N1,N2,NR,NR,estimator)
            return xi_12
        elif do_auto==True:
            xi_11 = TP_estimator(D1D1,D1R,RR,N1,N1,NR,NR,estimator)
            xi_22 = TP_estimator(D2D2,D2R,RR,N2,N2,NR,NR,estimator)
            return xi_11, xi_22


def Delta_Sigma(centers, particles, rbins, bounds=[-0.1,0.1], normal=[0.0,0.0,1.0],
                randoms=None, period=[1.0,1.0,1.0], N_threads=1):
    """
    Calculate the galaxy-galaxy lensing signal, $\Delata\Sigma$.
    
    Parameters
    ----------
    centers: array_like
        N_galaxies x 3 array of locations of galaxies to calculate $\Delata\Sigma$ around.
    
    particles: array_like
        N_particles x 3 array of locations ofmatter particles
    
    rbins: array_like
        location of bin edges
    
    bounds: array_like, optional
        len(2) array defining how far in fornt and behind a galaxy to integrate over
    
    normal: array_like, optional
        len(3) normal vector defining observer - target direction
    
    period: array_like, optional
        period of simulation box
    
    N_threads: int, optional
        number of threads to use for calculation
    
    Returns
    -------
    
    Delata_Sigma: np.array
        len(rbins)-1 array of $\Delata\Sigma$
    """
    from halotools.mock_observables.spatial.geometry import inside_volume
    from halotools.mock_observables.spatial.geometry import cylinder
    from halotools.mock_observables.spatial.kdtrees.ckdtree import cKDTree
    
    if N_threads>1:
        pool = Pool(N_threads)
    
    if period is None:
            PBCs = False
            period = np.array([np.inf]*np.shape(sample1)[-1])
    else:
        PBCs = True
        period = np.asarray(period).astype("float64")
        if np.shape(period) == ():
            period = np.array([period]*np.shape(centers)[-1])
        elif np.shape(period)[0] != np.shape(centers)[-1]:
            raise ValueError("period should have shape (k,)")
    
    normal = np.asarray(normal)
    bounds = np.asarray(bounds)
    centers = np.asarray(centers)
    particles = np.asarray(particles)
    
    print np.shape(particles)
    
    particle_tree= cKDTree(particles)
    proj_particle_tree = cKDTree(particles[:,0:2])
    centers_tree = cKDTree(centers)
    proj_centers_tree = cKDTree(centers[:,0:2])
    
    proj_period = period[0:2]
    
    N_targets = len(centers)
    length = bounds[1]-bounds[0]
    print('length:',length)
    
    #create cylinders for each galaxy and each radial bin
    cyls = np.ndarray((N_targets,len(rbins)),dtype=object)
    for i in range(0,N_targets):
        for j in range(0,len(rbins)):
            cyls[i,j] = geometry.cylinder(center=centers[i], radius = rbins[j],\
                                          length=length, normal=normal)
    
    #calculate the number of particles inside each cylinder 
    tree = cKDTree(particles)
    N = np.ndarray((len(centers),len(rbins)))
    N.fill(0.0)
    periods = [period,]*len(centers)
    for j in range(0,len(rbins)):
        print(j,rbins[j])
        #dum1, dum2, dum3, N[:,j] = inside_volume(cyls[:,j].tolist(), tree, period=period)
        points_to_test = proj_centers_tree.query_ball_tree(proj_particle_tree,rbins[j],period=proj_period)
        N_test = [len(inds) for inds in points_to_test]
        print('mean number to test:',np.mean(N_test))
        """
        coordinates_to_test = [particles[inds] for inds in points_to_test]
        results = map(cylinder.inside,cyls[:,j].tolist(),coordinates_to_test,periods)
        N_inside = [np.sum(result) for result in results]
        N[:,j] = N_inside
        print('mean number inside:',np.mean(N_inside))
        """
        N[:,j] = N_test
        
    #numbers in annular bins, N
    N_diff = np.diff(N,axis=1)
    
    #area of an annular ring, A
    A_circle = np.pi*rbins**2.0
    A_annulus = np.diff(A_circle)
    
    #calculate the surface density in annular bins, Sigma
    Sigma_diff = np.mean(N_diff,axis=0)/A_annulus
    Sigma_inside = np.mean(N,axis=0)/A_circle
    
    Delta_Sigma = Sigma_inside[:-1]-Sigma_diff
    
    return Delta_Sigma

