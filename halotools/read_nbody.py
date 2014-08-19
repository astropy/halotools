# -*- coding: utf-8 -*-
"""
Methods to load halo catalogs into memory.
Obviously this module needs to be generalized. 
Currently it's only useful at loading in a single pre-processed halo catalog: Bolshoi at z=0.
Adequate only while basic functionality of mock-making code is being developed.

"""

__all__=['read_barebones_ascii_halo_catalog_for_initial_mock_development','load_bolshoi_host_halos_fits']

#from __future__ import (absolute_import, division, print_function,
#                        unicode_literals)

from astropy.io import ascii
import defaults
from astropy.table import Table
from configuration import Config
import os
import urllib2
import warnings

def load_bolshoi_host_halos_fits(simulation_dict=None):
    """Placeholder method using pyfits to load a pre-processed .fits file containing host halo information.

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
    Will soon enough be replaced by a more general/flexible function 
    that can read in a variety of pre-processed
    halo catalogs. Ultimately wrapped up in a class with catalog I/O and analysis classes. 

    """
    import pyfits

    if simulation_dict == None:
        simulation_dict = defaults.default_simulation_dict

    halos = Table(pyfits.getdata(simulation_dict['catalog_filename'],0))
    #halos = pyfits.getdata(simulation_dict['catalog_filename'],0)


    # should be using astropy units!
    simulation = {'halos':halos,'simulation_dict':simulation_dict}
    return simulation



class simulation(object):
    """ Container class for the simulation being populated with mock galaxies.    
    
    """
    
    def __init__(self,simulation_name='bolshoi', scale_factor=1.0003, halo_finder='rockstar_V1.5', use_subhalos=False):


        self.simulation_name = simulation_name
        self.scale_factor = scale_factor
        self.halo_finder = halo_finder
        self.use_subhalos = use_subhalos

        self.halos = self.get_catalog()


    def get_catalog(self):

        import pyfits

        configobj = Config()
        self.catalog_path = configobj.catalog_pathname

        if self.use_subhalos==False:
            self.filename = (self.simulation_name+'_a'+
                str(self.scale_factor)+'_'+self.halo_finder+'_host_halos.fits' )
        else:
            self.filename = (self.simulation_name+'_a'+
                str(self.scale_factor)+'_'+self.halo_finder+'_subhalos.fits' )

        if os.path.isfile(self.catalog_path+self.filename)==True:
            halos = Table(pyfits.getdata(self.catalog_path+self.filename,0))
        else:
            warnings.warn("Host halo catalog not found in cache directory, downloading...")
            fileobj = urllib2.urlopen(configobj.hearin_url+self.filename)
            output = open(self.catalog_path+self.filename,'wb')
            output.write(fileobj.read())
            output.close()
            halos = Table(pyfits.getdata(self.catalog_path+self.filename,0))

        return halos














