# -*- coding: utf-8 -*-
"""
Modules to read and parse ASCII data of ROCKSTAR halo and merger tree catalogs, 
as well as subsequently derived value-added catalogs.

@author: aphearin
"""

from astropy.io import ascii
import pyfits

def read_barebones_ascii_halo_catalog_for_initial_mock_development(filename):
    """Read filename and return something called 'halos', 
    which I would like to be a numpty array, but it is not currently.
    Also, this takes forever."""
    column_names = ('id','mvir','x','y','z','vx','vy','vz')
#    types = ('long','float','float','float','float','float','float','float')
    halos = ascii.read(filename, delimiter='\s', names=column_names, data_start=0)

    print 'number of host halos read in:', len(halos)
    return halos

def load_bolshoi_host_halos_fits(filename):
    """Load .fits file containing host halo information 
    down to logMvir>10.5. """
    halos = pyfits.getdata(filename,0)
    
    return halos

