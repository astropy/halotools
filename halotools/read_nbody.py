# -*- coding: utf-8 -*-
"""
Methods to load halo catalogs into memory.
Not even close to the needed level of generality.
Currently is only useful at loading in a single pre-processed halo catalog: Bolshoi at z=0.
Adequate only while basic functionality of mock-making code is being developed.

"""

__all__=['read_barebones_ascii_halo_catalog_for_initial_mock_development','load_bolshoi_host_halos_fits']

#from __future__ import (absolute_import, division, print_function,
#                        unicode_literals)

from astropy.io import ascii
import pyfits
import defaults
from astropy.table import Table


def read_barebones_ascii_halo_catalog_for_initial_mock_development(filename):
    """ 

    Parameters
    ----------
    filename : string
        Name of file containing ASCII data

    Returns
    -------
    halos : Astropy Table
         Halo catalog information.

    Synopsis
    --------
    Read filename and return an astropy structured table called 'halos'. Currently a basically useless method.

    """
    
    column_names = ('id','mvir','x','y','z','vx','vy','vz')
#    types = ('long','float','float','float','float','float','float','float')
    halos = ascii.read(filename, delimiter='\s', names=column_names, data_start=0)

    return halos

def load_bolshoi_host_halos_fits(simulation_dict=None):
    """Placeholder method using pyfits to load a pre-processed .fits file containing host halo information.
    Will soon enough be replaced by a more general/flexible routine.

    Parameters
    ----------
    simulation_dict : dictionary
        The key 'halos' points to an astropy table containing halo catalog data.
        The key 'simulation_dict' points to a dictionary with keys for 
        simulation attributes such as box size, resolution, and scale factor of snapshot.

    Returns
    -------
    simulation : dictionary
        Halos key is a structured table containing halo catalog information. simulation_dict key is the input dictionary.

    Notes
    -----
    Default is Rockstar V1.5 Bolshoi halos at a=1.0003.

    """

    if simulation_dict == None:
        simulation_dict = defaults.default_simulation_dict

    halos = Table(pyfits.getdata(simulation_dict['catalog_filename'],0))
    #halos = pyfits.getdata(simulation_dict['catalog_filename'],0)


    # should be using astropy units!
    simulation = {'halos':halos,'simulation_dict':simulation_dict}
    return simulation


class simulation(object):
    """ Container class for properties of the simulation being used.
    
    Still unused.
    
    
    """
    
    def __init__(self,simulation_nickname=None):
        
        if simulation_nickname is None:
            self.halo_catalog_filename='/Users/aphearin/Dropbox/mock_for_surhud/VALUE_ADDED_HALOS/value_added_z0_halos.fits'
            self.simulation_dict = {
            'catalog_filename':default_halo_catalog_filename,
            'Lbox':250.0,
            'scale_factor':1.0003,
            'particle_mass':1.35e8,
            'softening':1.0
            }
        elif simulation_nickname is 'Bolshoi':
            self.halo_catalog_filename='/Users/aphearin/Dropbox/mock_for_surhud/VALUE_ADDED_HALOS/value_added_z0_halos.fits'
            self.simulation_dict = {
            'catalog_filename':default_halo_catalog_filename,
            'Lbox':250.0,
            'scale_factor':1.0003,
            'particle_mass':1.35e8,
            'softening':1.0
            }
        else:
            pass
        





















