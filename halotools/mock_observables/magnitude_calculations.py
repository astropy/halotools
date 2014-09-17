#Duncan Campbell
#September 17, 2014
#Yale University

""" 
Functions that compute quantities dealing with magnitudes.  
"""

from __future__ import division

__all__=['apparent_to_absolute_magnitude','luminosity_to_absolute_magnitude','get_sun_mag']

import numpy as np

def apparent_to_absolute_magnitude(m, d_L):
    """
    calculate the absolute magnitude
    
    Parameters
    ----------
    m: array_like
        apparent magnitude
    
    d_L: array_like
        luminosity distance to object in Mpc
    
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


