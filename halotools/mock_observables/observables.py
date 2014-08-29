#Duncan Campbell
#August 27, 2014
#Yale University

""" 
Functions that compute statistics of a mock galaxy catalog in a periodic box. 
Still largely unused in its present form, and needs to be integrated with 
the pair counter and subvolume membership methods.
"""

from __future__ import division

__all__=['two_point_correlation_function','apparent_to_absolute_magnitude',
         'luminosity_to_absolute_magnitude','get_sun_mag','luminosity_function','HOD',
         'CLF','CSMF','isolatoion_criterion']

import numpy as np
from math import pi, gamma
from cpairs import npairs

def two_point_correlation_function(sample1, rbins, sample2 = None, randoms=None, 
                                   period = None, max_sample_size=int(1e6), 
                                   estimator='Natural'):
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
    #periodic distance functions within the pair counter..
    ###############
    
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
    def random_counts(sample1, sample2, randoms, rbins, period, PBCs, k=3):
        """
        Count random pairs.
        """
        def nball_volume(R,k):
            """
            Calculate the volume of a n-shpere.
            """
            return (pi**(k/2.0)/gamma(k/2.0+1.0))*R**k
        
        #No PBCs, randoms must have been provided.
        if PBCs==False:
            RR = npairs(randoms, randoms, rbins, period=period)
            RR = np.diff(RR)
            D1R = npairs(sample1, randoms, rbins, period=period)
            D1R = np.diff(D1R)
            if np.all(sample1 != sample2): #calculating the cross-correlation
                D2R = npairs(sample2, randoms, rbins, period=period)
                D2R = np.diff(D2R)
            else: D2R = None
            
            return D1R, D2R, RR
        #PBCs and randoms.
        elif randoms != None:
            RR = npairs(randoms, randoms, rbins, period=period)
            RR = np.diff(RR)
            D1R = npairs(sample1, randoms, rbins, period=period)
            D1R = np.diff(D1R)
            if np.all(sample1 != sample2): #calculating the cross-correlation
                D2R = npairs(sample2, randoms, rbins, period=period)
                D2R = np.diff(D2R)
            else: D2R = None
            
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
            
            #if there is a sample2, calculate randoms for it.
            if np.all(sample1 != sample2):
                N2 = np.shape(sample2)[0]
                rho2 = N2/global_volume
                D2R = N2*(dv*rho2) #read note about pair counter
                #calculate the random-random pairs.
                NR = N1*N2
                rhor = NR/global_volume
                RR = (dv*rhor) #RR is only the RR for the cross-correlation.
            else: #if not calculating cross-correlation, set RR exactly equal to D1R.
                D2R = None
                RR = D1R #in the analytic case, for the auto-correlation, DR==RR.

            return D1R, D2R, RR
        else:
            raise ValueError('Un-supported combination of PBCs and randoms provided.')
    
    def pair_counts(sample1, sample2, rbins, period):
        """
        Count data pairs.
        """
        D1D1 = npairs(sample1, sample1, rbins, period=period)
        D1D1 = np.diff(D1D1)
        if np.all(sample1 != sample2):
            D1D2 = npairs(sample1, sample2, rbins, period=period)
            D1D2 = np.diff(D1D2)
            D2D2 = npairs(sample2, sample2, rbins, period=period)
            D2D2 = np.diff(D2D2)
        else:
            D1D2 = D1D1
            D2D2 = D1D1

        return D1D1, D1D2, D2D2
        
    def TP_estimator(DD,DR,RR,factor,estimator):
        """
        two point correlation function estimator
        """
        if estimator == 'Natural':
            xi = (1.0/factor**2.0)*DD/RR - 1.0
        elif estimator == 'Davis-Peebles':
            xi = (1.0/factor)*DD/DR - 1.0
        elif estimator == 'Hewett':
            xi = (1.0/factor**2.0)*DD/RR - (1.0/factor)*DR/RR #(DD-DR)/RR
        elif estimator == 'Hamilton':
            xi = (DD*RR)/(DR*DR) - 1.0
        elif estimator == 'Landy-Szalay':
            xi = (1.0/factor**2.0)*DD/RR - (1.0/factor)*2.0*DR/RR + 1.0 #(DD - 2.0*DR + RR)/RR
        else: 
            raise ValueError("unsupported estimator!")
        return xi
              
    if randoms != None:
        factor1 = (len(sample1)*1.0)/len(randoms)
        factor2 = (len(sample2)*1.0)/len(randoms)
    else: 
        factor1 = 1.0
        factor2 = 1.0
    
    #count pairs
    D1D1,D1D2,D2D2 = pair_counts(sample1, sample2, rbins, period)
    D1R, D2R, RR = random_counts(sample1, sample2, randoms, rbins, period, PBCs, k=k) 
    
    if np.all(sample2==sample1):
        xi_11 = TP_estimator(D1D1,D1R,RR,factor1,estimator)
        return xi_11
    elif (PBCs==True) & (randoms == None): 
        #Analytical randoms used. D1R1=R1R1, D2R2=R2R2, and R1R2=RR. See random_counts().
        xi_11 = TP_estimator(D1D1,D1R,D1R,1.0,estimator)
        xi_12 = TP_estimator(D1D2,D1R,RR,1.0,estimator)
        xi_22 = TP_estimator(D2D2,D2R,D2R,1.0,estimator)
        return xi_11, xi_12, xi_22
    else:
        xi_11 = TP_estimator(D1D1,D1R,RR,factor1,estimator)
        xi_12 = TP_estimator(D1D2,D1R,RR,factor1,estimator)
        xi_22 = TP_estimator(D2D2,D2R,RR,factor2,estimator)
        return xi_11, xi_12, xi_22


def apparent_to_absolute_magnitude(m, d_L):
    """
    calculate the absolute magnitude
    
    Parameters
    ----------
    m: array_like
        apparent magnitude
    
    d_L: array_like
        luminosity distance to object
    
    Returns
    -------
    Mag: np.array of absolute magnitudes
    """
    
    M = m - 5.0*(np.log10(d_L)-1.0)
    
    return M


def luminosity_to_absolute_magnitude(L, band, system='SDSS_Blanton_2003_z0.1'):
    """
    calculate the absolute magnitude
    
    Parameters
    ----------
    L: array_like
        apparent magnitude
    
    band: string
       filter band
    
    system: string, optional
        filter systems: default is 'SDSS_Blanton_2003_z0.1'
          1. Binney_and_Merrifield_1998
          2. SDSS_Blanton_2003_z0.1
    
    Returns
    -------
    Mag: np.array of absolute magnitudes
    """
    
    Msun = get_sun_mag(band,system)
    Lsun = 1.0
    M = -2.5*np.log10(L/Lsun) + Msun
            
    return M


def get_sun_mag(filter,system):
    """
    get the solar value for a filter in a system.
    
    Parameters
    ----------
    filter: string
    
    system: string
    
    Returns
    -------
    Msun: float
    """
    if system=='Binney_and_Merrifield_1998':
    #see Binney and Merrifield 1998
        if filter=='U':
            return 5.61
        elif filter=='B':
            return 5.48
        elif filter=='V':
            return 4.83
        elif filter=='R':
            return 4.42
        elif filter=='I':
            return 4.08
        elif filter=='J':
            return 3.64
        elif filter=='H':
            return 3.32
        elif filter=='K':
            return 3.28
        else:
            raise ValueError('Filter does not exist in this system.')
    if system=='SDSS_Blanton_2003_z0.1':
    #see Blanton et al. 2003 equation 14
        if filter=='u':
            return 6.80
        elif filter=='g':
            return 5.45
        elif filter=='r':
            return 4.76
        elif filter=='i':
            return 4.58
        elif filter=='z':
            return 4.51
        else:
            raise ValueError('Filter does not exist in this system.')
    else:
        raise ValueError('Filter system not included in this package.')


def luminosity_function(m, z, band, cosmo, system='SDSS_Blanton_2003_z0.1', L_bins=None):
    """
    Calculate the galaxy luminosity function.
    
    Parameters
    ----------
    m: array_like
        apparent magnitude of galaxies
    
    z: array_like
        redshifts of galaxies
    
    band: string
        filter band
    
    cosmo: astropy.cosmology object 
        specifies the cosmology to use, default is FlatLambdaCDM(H0=70, Om0=0.3)
    
    system: string, optional
        filter systems: default is 'SDSS_Blanton_2003_z0.1'
          1. Binney_and_Merrifield_1998
          2. SDSS_Blanton_2003_z0.1
    
    L_bins: array_like, optional
        bin edges to use for for the luminosity function. If None is given, "Scott's rule"
        is used where delta_L = 3.5sigma/N**(1/3)
    
    Returns
    -------
    counts, L_bins: np.array, np.array
    """
    
    from astropy import cosmology
    d_L = cosmo.luminosity_distance(z)
    
    M = apparant_to_absolute_magnitude(m,d_L)
    Msun = get_sun_mag(filter,system)
    L = 10.0**((Msun-M)/2.5)
    
    #determine Luminosity bins
    if L_bins==None:
        delta_L = 3.5*np.std(L)/float(L.shape[0]) #scott's rule
        Nbins = np.ceil((np.max(L)-np.min(L))/delta_L)
        L_bins = np.linspace(np.min(L),np.max(L),Nbins)
    
    counts = np.histogram(L,L_bins)[0]
    
    return counts, L_bins


def HOD(mock,galaxy_mask=None, mass_bins=None):
    """
    Calculate the galaxy HOD.
    
    Parameters
    ----------
    mock: mock object
    
    galaxy_mask: array_like, optional
        boolean array specifying subset of galaxies for which to calculate the HOD.
    
    mass_bins: array_like, optional
        array indicating bin edges to use for HOD calculation
    
    Returns
    -------
    N_avg, mass_bins: np.array, np.array
        mean number of galaxies per halo within the bin defined by bins, bin edges
    """
    
    from halotools.utils import match
    
    if not hasattr(mock, 'halos'):
        raise ValueError('mock must contain halos.')
    if not hasattr(mock, 'galaxies'):
        raise ValueError('mock must contain galaxies. execute mock.populate().')
    
    if galaxy_mask != None:
        if len(galaxy_mask) != len(mock.galaxies):
            raise ValueError('galaxy mask be the same length as mock.galaxies')
        elif x.dtype != bool:
            raise TypeError('galaxy mask must be of type bool')
        else:
            galaxies = mock.galaxies[galaxy_mask]
    else:
        galaxies = np.array(mock.galaxies)
    
    galaxy_to_halo = match(galaxies['haloID'],halo['ID'])
    
    galaxy_halos = halos[galaxy_to_halo]
    unq_IDs, unq_inds = np.unique(galaxy_halos['ID'], return_index=True)
    Ngals_in_halo = np.bincount(galaxy_halos['ID'])
    Ngals_in_halo = Ngals_in_halo[galaxy_halos['ID']]
    
    Mhalo = galaxy_haloes[unq_inds]
    Ngals = Ngals_in_halo[unq_inds]
    
    inds_in_bins = np.digitize(Mhalo,mass_bins)
    
    N_avg = np.zeros((len(mass_bins)-1,))
    for i in range(0,len(N_avg)):
        inds = np.where(inds_in_bins==i+1)[0]
        Nhalos_in_bin = float(len(inds))
        Ngals_in_bin = float(sum(Ngal[inds]))
        if Nhalos_in_bin==0: N_avg[i]=0.0
        else: N_avg[i] = Ngals_in_bin/Nhalos_in_bin
    
    return N_avg, mass_bins
    
    pass


def CLF(mock):
    """
    Calculate the galaxy CLF.
    """
    pass


def CSMF(mock):
    """
    Calculate the galaxy CSMF.
    """
    pass


from halotools.mock_observables.spatial import geometry
class isolatoion_criterion(object):
    """
    A object that defines a galaxy isolation criterion.
    
    Parameters 
    ----------
    volume: geometry volume object
        e.g. sphere, cylinder
    
    vol_args: list or function
        arguments to initialize the volume objects defining the test region of isolated 
        candidates, or function taking a galaxy object which returns the vol arguments.
    
    test_prop: string
        mock property to test isolation against.  e.g. 'M_r', 'Mstar', etc.
        
    test_func: function
        python function defining the property isolation test.
    """
    
    def __init__(self, volume=geometry.sphere, vol_args=None,
                 test_prop='primary_galprop', test_func=None):
        #check to make sure the volume object passed is in fact a volume object 
        if not issubclass(volume,geometry.volume):
            raise ValueError('volume object must be a subclass of geometry.volume')
        else: self.volume = volume
        #check volume object arguments. Is it None, a function, or a list?
        if vol_args==None:
            #default only passes center argument to volume object
            def default_func(galaxy):
                center = galaxy['coords']
                return center
            self.vol_agrs = default_func
        elif hasattr(vol_args, '__call__'):
            self.vol_args= vol_args
            #check for compatibility with the mock in the method
        else:
            #else, return the list of values passes in every time.
            def default_func(galaxy):
                return vol_agrs
            self.vol_agrs = default_func
        #store these two and check if they are compatible with a mock later in the method.
        self.test_prop = test_prop
        self.test_func = test_func
    
    def make_volumes(self, galaxies, isolated_candidates):
        volumes = np.empty((len(isolated_candidates),))
        for i in range(0,len(isolated_candidates)):
            volumes[i] = self.volume(self.vol_args(galaxies[isolated_candidates[i]]))
        return volumes

    def apply_criterion(self, mock, isolated_candidates):
        """
        Return galaxies which pass isolation criterion. 
    
        Parameters 
        ----------
        mock: galaxy mock object
    
        isolated_candidates: array_like
            indices of mock galaxy candidates to test for isolation.
        
        Returns 
        -------
        inds: numpy.array
            indicies of galaxies in mock that pass the isolation criterion.

        """
        
        #check input
        if not hasattr(mock, 'galaxies'):
            raise ValueError('mock must contain galaxies. execute mock.populate()')
        if self.test_prop not in mock.galaxies.dtype.names:
            raise ValueError('test_prop not present in mock.galaxies table.')
        try: self.volume(self.vol_args(mock.galaxies[0]))
        except TypeError: print('vol_args are not compatable with the volume object.')
        
        volumes = make_volumes(self,mock.galaxies,isolated_candidates)
        
        points_inside_shapes = geometry.inside_volume(
                               volumes, mock.coords[neighbor_candidates], period=mock.Lbox
                               )[2]
        
        ioslated = np.array([True]*len(isolated_candidates))
        for i in range(0,len(isolated_candidates)):
            inside = points_inside_shapes[i] 
            isolated[i] = np.all(self.test_func(mock.galaxies[isolated_candidates[i]][self.test_prop],mock.galaxies[inside][self.test_prop]))
        
        return isolated



