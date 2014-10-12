# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""This module standardizes access to
various files used throughout the halotools package. 
Global scope functions have been modified from the 
paths methods of the astropy config sub-package.
"""
import os
from astropy.config.paths import get_cache_dir as get_astropy_cache_dir

class Config(object):
    """ Configuration object providing standardization of 
    a variety of cross-package settings. """

    def __init__(self):

        self.catalog_pathname = self.getCatalogDir()
        self.hearin_url="http://www.astro.yale.edu/aphearin/Data_files/"


    def get_halotools_cache_dir(self):
        """ Find the path to the halotools cache directory. 
        If the directory doesn't exist, make it, then return the path. 

        Returns
        -------
        dir : str
            The path to the halotools cache directory.

        """

        halotools_cache_dir = os.path.join(get_astropy_cache_dir(), 'halotools_cache')

        if not os.path.exists(halotools_cache_dir):
            try:
                os.mkdir(halotools_cache_dir)
            except OSError as e:
                if not os.path.exists(halotools_cache_dir):
                    raise
        elif not os.path.isdir(halotools_cache_dir):
            msg = 'Data cache directory {0} is not a directory'
            raise IOError(msg.format(halotools_cache_dir))

        return halotools_cache_dir

    def get_halotools_download_dir(self):
        """ Find the path to the halotools cache directory. 
        If the directory doesn't exist, make it, then return the path. 

        Returns
        -------
        dir : str
            The path to the halotools cache directory.

        """

        halotools_download_dir = os.path.join(self.get_halotools_cache_dir(), 'download')

        if not os.path.exists(halotools_download_dir):
            try:
                os.mkdir(halotools_download_dir)
            except OSError as e:
                if not os.path.exists(halotools_download_dir):
                    raise
        elif not os.path.isdir(halotools_download_dir):
            msg = 'Data cache directory {0} is not a directory'
            raise IOError(msg.format(halotools_download_dir))

        return halotools_download_dir

    def get_halotools_catalog_dir(self):
        """ Find the path to the halotools cache directory. 
        If the directory doesn't exist, make it, then return the path. 

        Returns
        -------
        dir : str
            The path to the halotools cache directory.

        """

        halotools_catalog_dir = os.path.join(self.get_halotools_cache_dir(), 'catalogs')

        if not os.path.exists(halotools_catalog_dir):
            try:
                os.mkdir(halotools_catalog_dir)
            except OSError as e:
                if not os.path.exists(halotools_catalog_dir):
                    raise
        elif not os.path.isdir(halotools_catalog_dir):
            msg = 'Data cache directory {0} is not a directory'
            raise IOError(msg.format(halotools_catalog_dir))

        return halotools_catalog_dir

    # Returns the path to this code file
    def getCodeDir(self):
        return os.path.dirname(os.path.realpath(__file__))

    # Returns the path to the directory storing simulation data
    def getCatalogDir(self):
        return os.path.dirname(os.path.realpath(__file__))+'/CATALOGS/'


    def getSimulationFilename(self,simulation_name,scale_factor,halo_finder,use_subhalos):

        if use_subhalos==False:
            fname = (simulation_name+'_a'+
                str(scale_factor)+'_'+halo_finder+'_host_halos.fits' )
        else:
            fname = (simulation_name+'_a'+
                str(scale_factor)+'_'+halo_finder+'_subhalos.fits' )

        return fname

    def getParticleFilename(self,simulation_name,scale_factor,num_ptcl):

        fname = simulation_name+'_'+num_ptcl+'_particles_a'+str(scale_factor)+'.fits'

        return fname













