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
        simply use astropy.io.fits to load it. If the catalog is not there, 
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


    def __init__(self,simulation_name=defaults.default_simulation_name, 
        scale_factor=defaults.default_scale_factor, 
        num_ptcl_string=defaults.default_size_particle_data,
        manual_dirname=None):


        self.simulation_name = simulation_name
        if simulation_name=='bolshoi':
            self.Lbox = 250.0
        else:
            self.Lbox = None
        
        self.scale_factor = scale_factor
        self.num_ptcl_string = num_ptcl_string

        self.particle_data = self.get_particles(manual_dirname)


    def get_particles(self,manual_dirname=None):
        """ Method to load simulation particle data into memory. 

        If the particle dataset is already present in the cache directory, 
        simply use astropy.io.fits to load it. If the catalog is not there, 
        download it from www.astro.yale.edu/aphearin, and then load it into memory.

        """

        configobj = Config()
        if manual_dirname != None:
            warnings.warn("using hard-coded directory name to load simulation")
            cache_dir = manual_dirname
        else:
            cache_dir = configobj.getCatalogDir()

        self.filename = configobj.getParticleFilename(
            self.simulation_name,self.scale_factor,self.num_ptcl_string)

        if os.path.isfile(cache_dir+self.filename)==True:
            hdulist = astropy.io.fits.open(cache_dir+self.filename)
            particle_data = Table(hdulist[1].data)
#            halos = Table(pyfits.getdata(cache_dir+self.filename,0))
        else:
            warnings.warn("\n Particle data not found in cache directory")
            download_yes_or_no = raw_input(" \n Enter yes to download, "
                "any other key to exit:\n (File size is ~10Mb) \n\n ")

            if download_yes_or_no=='y' or download_yes_or_no=='yes':
                print("\n...downloading particle data from www.astro.yale.edu/aphearin")
                fileobj = urllib2.urlopen(configobj.hearin_url+self.filename)

                output = open(cache_dir+self.filename,'wb')
                output.write(fileobj.read())
                output.close()
                hdulist = astropy.io.fits.open(cache_dir+self.filename)
                particle_data = Table(hdulist[1].data)
                #halos = Table(pyfits.getdata(cache_dir+self.filename,0))
            else:
                warnings.warn("\n ...Exiting halotools \n")
                os._exit(1)

        return particle_data

###################################################################################################





"""
        configobj = Config()
        if manual_dirname != None:
            warnings.warn("using hard-coded directory name to load simulation")
            cache_dir = manual_dirname
        else:
            cache_dir = configobj.getCatalogDir()
        self.cache_dir = cache_dir

        self.filename = configobj.getParticleFilename(
            self.simulation_name,self.scale_factor,self.num_ptcl_string)
"""





