# -*- coding: utf-8 -*-
"""
Methods to load halo and particle catalogs into memory.

"""

__all__=['simulation','particles']

#from __future__ import (absolute_import, division, print_function,
#                        unicode_literals)

import astropy
import defaults
from astropy.table import Table
from configuration import Config
import os, sys, warnings, urllib2



class simulation(object):
    """ Container class for the simulation being populated with mock galaxies.    
    
    """
    
    def __init__(self,simulation_name=defaults.default_simulation_name, 
        scale_factor=defaults.default_scale_factor, 
        halo_finder=defaults.default_halo_finder, 
        use_subhalos=False,manual_dirname=None):


        self.simulation_name = simulation_name
        if simulation_name=='bolshoi':
            self.Lbox = 250.0
        else:
            self.Lbox = None
        
        self.scale_factor = scale_factor
        self.halo_finder = halo_finder
        self.use_subhalos = use_subhalos

        self.halos = self.get_catalog(manual_dirname)


    def get_catalog(self,manual_dirname=None):
        """ Method to load halo catalogs into memory. 

        If the halo catalog is already present in the cache directory, 
        simply use pyfits to load it. If the catalog is not there, 
        download it from www.astro.yale.edu/aphearin, and then load it into memory.

        """

        #import pyfits

        configobj = Config()
        if manual_dirname != None:
            warnings.warn("using hard-coded directory name to load simulation")
            cache_dir = manual_dirname
        else:
            cache_dir = configobj.getCatalogDir()

        #print("cache directory = "+cache_dir)

        self.filename = configobj.getSimulationFilename(
            self.simulation_name,self.scale_factor,self.halo_finder,self.use_subhalos)

        if os.path.isfile(cache_dir+self.filename)==True:
            hdulist = astropy.io.fits.open(cache_dir+self.filename)
            halos = Table(hdulist[1].data)
#            halos = Table(pyfits.getdata(cache_dir+self.filename,0))
        else:
            warnings.warn("\n Host halo catalog not found in cache directory")
            download_yes_or_no = raw_input(" \n Enter yes to download, "
                "any other key to exit:\n (File size is ~200Mb) \n\n ")

            if download_yes_or_no=='y' or download_yes_or_no=='yes':
                print("\n...downloading halo catalog from www.astro.yale.edu/aphearin")
                fileobj = urllib2.urlopen(configobj.hearin_url+self.filename)

                output = open(cache_dir+self.filename,'wb')
                output.write(fileobj.read())
                output.close()
                hdulist = astropy.io.fits.open(cache_dir+self.filename)
                halos = Table(hdulist[1].data)
                #halos = Table(pyfits.getdata(cache_dir+self.filename,0))
            else:
                warnings.warn("\n ...Exiting halotools \n")
                os._exit(1)

        return halos

###################################################################################################



class particles(object):
    """ Container class for the simulation particles.    
    
    """

    def __init__(self,simulation_name='bolshoi', scale_factor=1.0003):


        self.simulation_name = simulation_name
        self.scale_factor = scale_factor

        self.particles = self.get_particles()


    def get_particles(self):

        import pyfits

        configobj = Config()
        self.catalog_path = configobj.catalog_pathname

        self.filename = (self.simulation_name+'_2e5_particles_a'+
            str(self.scale_factor)+'.fits')

        if os.path.isfile(self.catalog_path+self.filename)==True:
            particles = Table(pyfits.getdata(self.catalog_path+self.filename,0))
        else:
            warnings.warn("Particle data not found in cache directory, downloading...")
            fileobj = urllib2.urlopen(configobj.hearin_url+self.filename)
            output = open(self.catalog_path+self.filename,'wb')
            output.write(fileobj.read())
            output.close()
            particles = Table(pyfits.getdata(self.catalog_path+self.filename,0))

        return particles












