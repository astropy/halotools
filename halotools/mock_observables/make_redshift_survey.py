#Duncan Campbell
#November, 2014
#Yale University

"""
create mock redshift space coordinates
"""

from __future__ import division, print_function

__all__=['distant_observer','ra_dec_z']

import numpy as np

def distant_observer(mock,cosmo,los='z'):
    
    #from astropy.constants import c
    c = 299792.458 #speed of light in km/s
    from astropy import cosmology
    from scipy.interpolate import interp1d
    
    if los=='x': los=0
    if los=='y': los=1
    if los=='z': los=2
    
    #get the peculiar velocity component along the line of sight direction
    v_los = mock.galaxies['vel'][:,los]
    
    #compute cosmological redshift
    y = np.linspace(0,1,1000)
    x = cosmo.comoving_distance(y).value
    f = interp1d(x, y, kind='cubic')
    z_cos = f(mock.galaxies['coords'][:,los])
    
    #redshift is combination of cosmological and peculiar velocities
    z = z_cos+(v_los/c)*(1.0+z_cos)
    
    #reflect galaxies around redshift PBC
    flip = (z>f(mock.Lbox))
    z[flip] = z[flip]-f(mock.Lbox)
    
    mock.galaxies['redshift']=z

    return 0


def ra_dec_z():

    #from astropy.constants import c
    c = 299792.458 #speed of light in km/s
    from astropy import cosmology
    from math import pi
    from scipy.interpolate import interp1d
    
    #compute comoving distance from observer
    r = np.sqrt(mock.galaxies['coords'][:,0]**2+mock.galaxies['coords'][:,1]**2+mock.galaxies['coords'][:,2]**2)
    
    #compute radial velocity
    ct = mock.galaxies['coords'][:,2]/r
    st = np.sqrt(1.0-ct**2)
    cp = mock.galaxies['coords'][:,0]/np.sqrt(mock.galaxies['coords'][:,0]**2+mock.galaxies['coords'][:,1]**2)
    sp = mock.galaxies['coords'][:,1]/np.sqrt(mock.galaxies['coords'][:,0]**2+mock.galaxies['coords'][:,1]**2)
    vr = mock.galaxies['vel'][:,0]*st*cp + mock.galaxies['vel'][:,1]*st*sp + mock.galaxies['vel'][:,2]*ct

    #compute cosmological redshift and add contribution from perculiar velocity
    y = np.arange(0,2.0,0.01)
    x = cosmo.comoving_distance(y)
    f = interp1d(x, y, kind='cubic')
    z_cos = f(r)
    z = z_cos+(vr/c)*(1.0+z_cos)
    mock.galaxies['redshift']=z

    #calculate spherical coordinates
    theta = np.arccos(mock.galaxies['coords'][:,2]/r)
    phi   = np.arccos(cp) #atan(y/x)
    
    #convert spherical coordinates into ra,dec in radians
    mock.galaxies['ra']  = phi
    mock.galaxies['dec'] = (pi/2.0) - theta
    
    return 0


