# -*- coding: utf-8 -*-
"""
Modules to read and parse ASCII data of ROCKSTAR halo and merger tree catalogs, 
as well as subsequently derived value-added catalogs.

@author: aphearin
"""

from astropy.io import ascii
import pyfits
import defaults

def read_barebones_ascii_halo_catalog_for_initial_mock_development(filename):
    """ Read filename and return an astropy structured table called 'halos'.

    Parameters
    ----------
    filename : string
    Absolute path of filename containing ASCII data

    Returns
    ----------
    halos : astropy structured table containing halo catalog information

    Notes
    ----------
    This takes forever, and the code is not currently written to work with its output.
    Use load_bolshoi_host_halos_fits to load a pre-processed .fits file instead, which is virtually instantaneous.

    """
    
    column_names = ('id','mvir','x','y','z','vx','vy','vz')
#    types = ('long','float','float','float','float','float','float','float')
    halos = ascii.read(filename, delimiter='\s', names=column_names, data_start=0)

    print 'number of host halos read in:', len(halos)
    return halos

def load_bolshoi_host_halos_fits(simulation_dict=None):
    """Use pyfits to load a pre-processed .fits file containing host halo information.

    Parameters
    ----------
    simulation_dict : dictionary
    Contains keys for the filename, as well as simulation attributes such as 
    box size, resolution, and scale factor of snapshot.

    Returns
    ----------
    simulation : dictionary
    Halos key is a structured table containing halo catalog information.
    simulation_dict key is the input dictionary.

    Notes 
    ----------
    Default is Rockstar V1.5 Bolshoi halos at a=1.0003.

    """

    if simulation_dict == None:
        simulation_dict = defaults.default_simulation_dict

    halos = pyfits.getdata(simulation_dict['catalog_filename'],0)
    # should be using astropy units!
    simulation = {'halos':halos,'simulation_dict':simulation_dict}
    return simulation

