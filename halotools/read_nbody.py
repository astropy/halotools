# -*- coding: utf-8 -*-
"""
Methods and classes to load halo and particle catalogs into memory.

"""

__all__=['simulation','particles']

#from __future__ import (absolute_import, division, print_function,
#                        unicode_literals)

from astropy.io import fits as fits
from astropy.table import Table
from astropy.utils.data import get_readable_fileobj
from astropy.utils.data import _get_download_cache_locs as get_download_cache_locs
from astropy.utils.data import _open_shelve as open_shelve


from configuration import Config
import os, sys, warnings, urllib2
import defaults



##bolshoi_a1.0003_rockstar_V1.5_host_halos.fits'


class processed_snapshot(object):
    """ Container class for halo and particle data taken from a single snapshot
    of some Nbody simulation.
    """

    def __init__(self,
        simulation_name=defaults.default_simulation_name,
        redshift=defaults.default_redshift,
        halo_finder=defaults.default_halo_finder):

        self.simulation_name = simulation_name
        self.redshift = redshift
        self.halo_finder = halo_finder

        halo_catalog_manager = catalog_manager()
        self.cache_dir = halo_catalog_manager.cache_dir


    @property
    def particles(self):
        """ Method to load simulation particle data into memory. 

        The behavior of this method is governed by the astropy utility get_readable_fileobj. 
        If the particle dataset is already present in the astropy cache directory, 
        get_readable_fileobj will detect it and the retrieve_particles method 
        will use astropy.io.fits to load the particles into memory. 
        If the catalog is not there, 
        get_readable_fileobj will download it from www.astro.yale.edu/aphearin, 
        and then load it into memory, again using astropy.io.fits.

        """

        with get_readable_fileobj(url_string,cache=True) as f: 
            fits_object = fits.HDUList.fromfile(f)
            particle_catalog = fits_object[1].data
            return particle_catalog



###################################################################################################

class catalog_manager(object):
    """ Container class for managing I/O of Rockstar catalogs hosted at SLAC.
    """

    def __init__(self):
        self.slac_urls = {'bolshoi_halos' : 'http://www.slac.stanford.edu/~behroozi/Bolshoi_Catalogs/',
        'bolshoi_bdm_halos' : 'http://www.slac.stanford.edu/~behroozi/Bolshoi_Catalogs_BDM/',
        'multidark_halos' : 'http://slac.stanford.edu/~behroozi/MultiDark_Hlists_Rockstar/',
        'consuelo_halos' : 'http://www.slac.stanford.edu/~behroozi/Consuelo_Catalogs/'
        }

        self.aph_url = 'http://www.astro.yale.edu/aphearin/Data_files/'

        self.cache_dir = get_download_cache_locs()[0].encode('utf-8')+'/'

    def retrieve_snapshot_filenames_from_slac_url(self,url,catalog_type='halos'):
        """ Get the full list of filenames available at the provided url.

        """
        from bs4 import BeautifulSoup
        import requests

        if catalog_type == 'halo' or 'halos': 
            expected_filename_prefix = 'hlist_'
        elif catalog_type == 'tree' or 'trees':
            expected_filename_prefix = 'tree_'
        else:
            raise TypeError("Input catalog type must either be 'halos' or 'trees'")

        soup = BeautifulSoup(requests.get(url).text)
        file_list = []

        for a in soup.find_all('a'):
            link = a['href']
            if link[0:len(expected_filename_prefix)]==expected_filename_prefix: 
                file_list.append(a['href'])

        return file_list

    def get_halo_catalog_filename(nickname=defaults.default_simulation_name,
        redshift=defaults.default_redshift,
        scale_factor=defaults.default_scale_factor,
        halo_finder=defaults.default_halo_finder,
        web_location=defaults.aph_web_location):
    pass
    #output_string = web_location+nickname+'_'+


###################################################################################################

class particles(object):
    """ Container class for the simulation particles.    
    
    """


    def __init__(self,simulation_name=defaults.default_simulation_name, 
        scale_factor=defaults.default_scale_factor, 
        num_ptcl_string=defaults.default_size_particle_data,
        manual_dirname=None,ask_permission=False):


        self.simulation_name = simulation_name
        if simulation_name=='bolshoi':
            self.Lbox = 250.0
        else:
            self.Lbox = None
        
        self.scale_factor = scale_factor
        self.num_ptcl_string = num_ptcl_string

        self.particle_data = self.get_particles(manual_dirname,ask_permission)

    def retrieve_particles(self,ask_permission=False):
        """ Method to load simulation particle data into memory. 

        The behavior of this method is governed by the astropy utility get_readable_fileobj. 
        If the particle dataset is already present in the astropy cache directory, 
        get_readable_fileobj will detect it and the retrieve_particles method 
        will use astropy.io.fits to load the particles into memory. 
        If the catalog is not there, 
        get_readable_fileobj will download it from www.astro.yale.edu/aphearin, 
        and then load it into memory, again using astropy.io.fits.

        """

        with get_readable_fileobj(url_string,cache=True) as f: 
            fits_object = fits.HDUList.fromfile(f)
            particle_catalog = fits_object[1].data
            return particle_catalog



        pass


    def get_particles(self,manual_dirname=None,ask_permission=False):
        """ Method to load simulation particle data into memory. 

        If the particle dataset is already present in the cache directory, 
        simply use astropy.io.fits to load it. If the catalog is not there, 
        download it from www.astro.yale.edu/aphearin, and then load it into memory.

        """
        download_yes_or_no = None

        configobj = Config()
        if manual_dirname != None:
            warnings.warn("using hard-coded directory name to load simulation")
            cache_dir = manual_dirname
        else:
            cache_dir = configobj.getCatalogDir()

        self.filename = configobj.getParticleFilename(
            self.simulation_name,self.scale_factor,self.num_ptcl_string)

        if os.path.isfile(cache_dir+self.filename)==True:
            hdulist = fits.open(cache_dir+self.filename)
            particle_data = Table(hdulist[1].data)
#            halos = Table(pyfits.getdata(cache_dir+self.filename,0))
        else:

            if ask_permission==True:
                warnings.warn("\n Particle data not found in cache directory")
                download_yes_or_no = raw_input(" \n Enter yes to download, "
                    "any other key to exit:\n (File size is ~10Mb) \n\n ")

            if (download_yes_or_no=='y') or (download_yes_or_no=='yes') or (ask_permission==False):
                print("\n...downloading particle data from www.astro.yale.edu/aphearin")
                fileobj = urllib2.urlopen(configobj.hearin_url+self.filename)

                output = open(cache_dir+self.filename,'wb')
                output.write(fileobj.read())
                output.close()
                hdulist = fits.open(cache_dir+self.filename)
                particle_data = Table(hdulist[1].data)
                #halos = Table(pyfits.getdata(cache_dir+self.filename,0))
            else:
                warnings.warn("\n ...Exiting halotools \n")
                os._exit(1)

        return particle_data

###################################################################################################


class simulation(object):
    """ Container class for the simulation being populated with mock galaxies.    
    
    """
    
    def __init__(self,simulation_name=defaults.default_simulation_name, 
        scale_factor=defaults.default_scale_factor, 
        halo_finder=defaults.default_halo_finder, 
        use_subhalos=False,manual_dirname=None,ask_permission=False):


        self.simulation_name = simulation_name
        if simulation_name=='bolshoi':
            self.Lbox = 250.0
        else:
            self.Lbox = None
        
        self.scale_factor = scale_factor
        self.halo_finder = halo_finder
        self.use_subhalos = use_subhalos

        self.halos = self.get_catalog(manual_dirname,ask_permission)


    def get_catalog(self,manual_dirname=None,ask_permission=False):
        """ Method to load halo catalogs into memory. 

        If the halo catalog is already present in the cache directory, 
        simply use astropy.io.fits to load it. If the catalog is not there, 
        download it from www.astro.yale.edu/aphearin, and then load it into memory.

        """

        download_yes_or_no = None

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
            hdulist = fits.open(cache_dir+self.filename)
            halos = Table(hdulist[1].data)
#            halos = Table(pyfits.getdata(cache_dir+self.filename,0))
        else:
            if ask_permission==True:
                warnings.warn("\n Host halo catalog not found in cache directory")
                download_yes_or_no = raw_input(" \n Enter yes to download, "
                    "any other key to exit:\n (File size is ~200Mb) \n\n ")

            if (download_yes_or_no=='y') or (download_yes_or_no=='yes') or (ask_permission==False):
                print("\n...downloading halo catalog from www.astro.yale.edu/aphearin")
                fileobj = urllib2.urlopen(configobj.hearin_url+self.filename)

                output = open(cache_dir+self.filename,'wb')
                output.write(fileobj.read())
                output.close()
                hdulist = fits.open(cache_dir+self.filename)
                halos = Table(hdulist[1].data)
                #halos = Table(pyfits.getdata(cache_dir+self.filename,0))
            else:
                warnings.warn("\n ...Exiting halotools \n")
                os._exit(1)

        return halos












