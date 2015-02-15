#Duncan Campbell
#August 27, 2014
#Yale University

""" 
miscellaneous functions that compute observational statistics of a mock galaxy catalog. 
"""

from __future__ import division
import sys

__all__=['apparent_to_absolute_magnitude', 'luminosity_to_absolute_magnitude','get_sun_mag',
         'luminosity_function','HOD', 'CLF','CSMF']

####import modules########################################################################
import numpy as np
from math import pi, gamma
##########################################################################################

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
    
    M = m - 5.0*(np.log10(d_L)+5.0)
    
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


