# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""This module standardizes access to
various files used throughout the halotools package. 
Global scope functions have been modified from the 
paths methods of the astropy config sub-package.
"""
import os

class Config(object):
    """ Configuration object providing standardization of 
    a variety of cross-package settings. """

    def __init__(self):

        self.catalog_pathname = self.getCatalogDir()
        self.hearin_url="http://www.astro.yale.edu/aphearin/Data_files/"


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













