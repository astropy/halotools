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

import numpy as np

import configuration
import os, sys, warnings, urllib2
import defaults


class processed_snapshot(object):
    """ Class containing halo and particle data taken from 
    a single snapshot of some Nbody simulation.
    """

    def __init__(self,
        simname=defaults.default_simulation_name,
        redshift=defaults.default_redshift,
        halo_finder=defaults.default_halo_finder,
        ask_permission=True):

        self.simulation_name = simname
        self.redshift = redshift
        self.scale_factor = 1./(1.+self.redshift)
        self.halo_finder = halo_finder

        catman = Catalog_Manager()

        self.catalog_dir = configuration.get_halotools_cache_dir()

        halo_catalog_filename,closest_scale_factor = (
            catman.find_nearest_snapshot_in_cache(
                catalog_type='halos',scale_factor = self.scale_factor,
                simname=self.simulation_name,
                halo_finder=self.halo_finder)
            )
        self.halo_catalog_filename = (
            os.path.join(self.catalog_dir,halo_catalog_filename))

        particle_catalog_filename,closest_scale_factor = (
            catman.find_nearest_snapshot_in_cache(
                catalog_type='particles',scale_factor = self.scale_factor,
                simname=self.simulation_name,
                halo_finder=self.halo_finder)
            )
        self.particle_catalog_filename = (
            os.path.join(self.catalog_dir,particle_catalog_filename))

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

        pass

    @property
    def halos(self):
        """ Method to load simulation halo catalog into memory. 

        The behavior of this method is governed by the astropy utility get_readable_fileobj. 
        If the particle dataset is already present in the astropy cache directory, 
        get_readable_fileobj will detect it and the retrieve_particles method 
        will use astropy.io.fits to load the particles into memory. 
        If the catalog is not there, 
        get_readable_fileobj will download it from www.astro.yale.edu/aphearin, 
        and then load it into memory, again using astropy.io.fits.

        """

        pass
 

# Easy way to load fits data from a website using astropy.utils
#with get_readable_fileobj(url_string,cache=True) as f: 
#    fits_object = fits.HDUList.fromfile(f)
#    particle_catalog = fits_object[1].data
#    return particle_catalog



###################################################################################################

class Catalog_Manager(object):
    """ Container class for managing I/O of halo & particle catalogs.
    """

    def __init__(self):
        self.slac_urls = {'bolshoi_halos' : 'http://www.slac.stanford.edu/~behroozi/Bolshoi_Catalogs/',
        'bolshoi_bdm_halos' : 'http://www.slac.stanford.edu/~behroozi/Bolshoi_Catalogs_BDM/',
        'multidark_halos' : 'http://slac.stanford.edu/~behroozi/MultiDark_Hlists_Rockstar/',
        'consuelo_halos' : 'http://www.slac.stanford.edu/~behroozi/Consuelo_Catalogs/'
        }

        self.default_halo_catalog_filename = (
            defaults.default_simulation_name+'_a'+
            str(np.round(defaults.default_scale_factor,4))+'_'+
            defaults.default_halo_finder+'_halos.fits')

        self.default_particle_catalog_filename = (
            defaults.default_simulation_name+'_a'+
            str(np.round(defaults.default_scale_factor,4))+'_'+
            self.numptcl_to_string(defaults.default_numptcl)+'_particles.fits')


    def retrieve_catalog_filenames_from_url(self,url,catalog_type='halos'):
        """ Get the full list of filenames available at the provided url.

        This method uses BeautifulSoup to query the provided url for the list of files stored there. 
        Filenames of halo catalogs are expected to begin with 'hlist', or they will be ignored; 
        filenames of merger trees are expected to begin with 'tree'.

        Parameters 
        ----------
        url : string 
            Web address pointing to the directory to be searched.

        catalog_type : string 
            Specifies what type of catalog to look for at the provided url.

        Returns 
        -------
        file_list : list 
            List of strings of filenames available for downloaded at the provided url.
        """

        from bs4 import BeautifulSoup
        import requests

        soup = BeautifulSoup(requests.get(url).text)
        file_list = []

        ##################
        ### SLAC url case
        if url==defaults.behroozi_web_location:
            ### Set naming conventions of the files hosted at SLAC
            if (catalog_type == 'halo') or (catalog_type=='halos'): 
                expected_filename_prefix = 'hlist_'
            elif (catalog_type == 'tree') or (catalog_type == 'trees'):
                expected_filename_prefix = 'tree_'
            else:
                raise TypeError("Input catalog type must either be 'halos' or 'trees'")
            ### Grab all filenames with the assumed prefix
            for a in soup.find_all('a'):
                link = a['href']
                if link[0:len(expected_filename_prefix)]==expected_filename_prefix: 
                    file_list.append(a['href'])
        ##################
        ### APH url case (simpler, since only two default catalogs are hosted here)
        elif url==defaults.aph_web_location:
            #Currently broken due to permissions settings at Yale
            pass
            ### Set naming conventions of the files hosted at Yale
            #if (catalog_type == 'halo') or (catalog_type=='halos'): 
            #    expected_filename_suffix = 'halos.fits'
            #elif (catalog_type == 'particle') or (catalog_type=='particles'):
            #    expected_filename_suffix = 'particles.fits'
            #else:
            #    expected_filename_suffix = '.fits'
            ### Grab all filenames with the assumed suffix
            #for a in soup.find_all('a'):
            #    link = a['href']
                #if link[-len(expected_filename_suffix):]==expected_filename_suffix: 
            #    file_list.append(a['href'])
        ##################
        ### Some other url managed by the user
        else:
            for a in soup.find_all('a'):
                link = a['href']
                file_list.append(a['href'])

        return file_list
        ##################


    def identify_relevant_catalogs(self,catalog_type=None,
        simname=None,halo_finder=None):
        """ Look in cache for any catalog that matches the inputs.

        Parameters 
        ----------
        catalog_type : string 
            Specifies whether we are interested in halo or particle catalogs. 

        simname : string 
            Specifies the simulation of interest, e.g., 'bolshoi'.

        halo_finder : string 
            Specifies the halo-finder used to generate the catalog. 

        Returns 
        -------
        relevant_catalogs : array
            array of strings of the filenames of catalogs matching the input specifications.
        """

        # Fix possible pluralization mistake of user
        if catalog_type == 'halo': catalog_type='halos'
        if catalog_type == 'particle': catalog_type='particles'

        # Identify all catalogs currently stored in the cache directory
        available_catalogs = np.array(
            configuration.list_of_catalogs_in_cache(catalog_type=catalog_type))

        #########
        # The file_mask array will determine which of the available catalogs 
        # pass the input specifications
        file_mask  = np.ones(len(available_catalogs),dtype=bool)

        # Impose halos vs. particles restriction
        if catalog_type != None:
            last_characters_of_filename=catalog_type+'.fits'
            for ii,c in enumerate(available_catalogs):
                if c[-len(catalog_type)-5:] != last_characters_of_filename:
                    file_mask[ii]=False

        # Impose simulation name restriction
        if simname != None:
            first_characters_of_filename = simname
            for ii,c in enumerate(available_catalogs):
                if c[0:len(simname)] != first_characters_of_filename:
                    file_mask[ii]=False

        # Impose halo finder restriction
        if halo_finder != None:
            for ii,c in enumerate(available_catalogs):
                if c[-11-len(halo_finder):-11] != halo_finder:
                    file_mask[ii]=False
        #########

        relevant_catalogs = available_catalogs[file_mask]

        return relevant_catalogs


    def find_nearest_snapshot_in_cache(self,catalog_type,
        scale_factor=None,redshift=None,
        simname=defaults.default_simulation_name,
        halo_finder = defaults.default_halo_finder):

        """ Identify the catalog in the cache directory with the 
        closest redshift to the requested redshift.

        Returns 
        ------- 
        filename : string 
            filename of pre-processed catalog in cache directory with closest redshift to 
            the requested redshift

        nearest_snapshot : float
            Value of the scale factor of the returned catalog

        """

        # Fix possible pluralization mistake of user
        if catalog_type == 'halo': catalog_type='halos'
        if catalog_type == 'particle': catalog_type='particles'

        if (scale_factor == None):
            if (redshift == None):
                raise IOError("Must specify either a redshift or a scale factor")
            else:
                scale_factor = 1./(1.+redshift)
        else:
            if (redshift != None):
                raise IOError("Cannot specify both a redshift and a scale factor")

        # Ignore and over-write the halo_finder if looking for particle data
        # This is necessary or else the relevant_catalog finder will bail
        if catalog_type=='particles':
            halo_finder=None

        relevant_catalogs = self.identify_relevant_catalogs(
            catalog_type=catalog_type,simname=simname,halo_finder=halo_finder)

        if len(relevant_catalogs)==0:
            if catalog_type=='halos':
                warnings.warn("Zero halo catalogs in cache match the input simname & halo-finder")
                return None, None
            elif catalog_type=='particles':
                warnings.warn("Zero particle catalogs in cache match the input simname")
                return None, None

        first_scale_factor_index=len(simname)+2
        last_scale_factor_index = first_scale_factor_index + 6
        available_snapshots = (
            [float(a[first_scale_factor_index:last_scale_factor_index]) 
            for a in relevant_catalogs] )

        idx_nearest_snapshot = np.abs(np.array(available_snapshots)-scale_factor).argmin()
        nearest_snapshot = available_snapshots[idx_nearest_snapshot]
        filename_of_nearest_snapshot = relevant_catalogs[idx_nearest_snapshot]


        # Warn the user if the nearest scale factor differs by more than the 
        # tolerance value set in defaults module
        adiff_tol = defaults.scale_factor_difference_tol
        adiff = np.abs(nearest_snapshot - scale_factor)
        if adiff > adiff_tol:
            msg = "Closest match to desired snapshot has a scale factor of "+str(nearest_snapshot)
            warnings.warn(msg)

        return filename_of_nearest_snapshot,nearest_snapshot

    def numptcl_to_string(self,numptcl):
        """ Reduce the input number to a 3-character string used to encode 
        the number of particles in the particle catalog filenames.

        Parameters 
        ----------
        numptcl : float or int 
            Number specifying the number of particles in the downsampled catalog.

        Returns 
        -------
        output_string : string 
            3-character string used in the filename conventions of the particle data catalogs.

        """

        # First find the order of magnitude of numptcl (there must be a more elegant way than this)
        oom_tester=False
        ipower=0
        while oom_tester==False:
            powfactor = 10.**ipower
            reduced_numptcl = np.round(numptcl/powfactor)
            if reduced_numptcl<10:
                oom_tester=True
            ipower += 1

        # Now use the above to reduce numptcl to three characters
        power = ipower-1
        powfactor = 10.**power
        reduced_numptcl = np.round(numptcl/powfactor)
        ce = str(power)
        cp = str(int(np.floor(reduced_numptcl)))
        output_string = cp+'e'+ce

        return output_string

    def load_catalog(self,
        dirname=defaults.halo_catalog_dirname,filename=None,
        download_yn=False,url=defaults.aph_web_location):
        """ Use the astropy fits reader to load the halo or particle catalog into memory.

        Parameters 
        ----------
        dirname : string 
            Name of directory where filename is stored.

        filename : string 
            Name of file being loaded into memory. Method assumes .fits file format.

        download_yn : boolean, optional
            If set to True, and if filename is not already stored in the cache directory, 
            method will attempt to download the file from the provided url. If there is no corresponding 
            file at the input url, an exception will be raised.

        Returns 
        -------
        catalog : object
            Data structure located at the input filename.

        """
        if filename==None:
            filename = self.default_halo_catalog_filename

        if os.path.isfile(os.path.join(dirname,filename)):
            hdulist = fits.open(os.path.join(dirname,filename))
            catalog = Table(hdulist[1].data)
        else:
            ### Requested filename is not in cache, and external download is not requested
            if download_yn==False:
                return None
            else:
                # Download one of the default catalogs hosted at Yale
                if filename==self.default_halo_catalog_filename:
                    catalog_type='halos'
                if filename==self.default_particle_catalog_filename:
                    catalog_type='particles'
                else:
                    raise IOError("Input filename does not match one of the provided default catalogs")
                ###
                remote_filename = os.path.join(url,filename)
                fileobj = urllib2.urlopen(remote_filename)
                output_directory = configuration.get_catalogs_dir(catalog_type=catalog_type)
                output_filename = os.path.join(output_directory,filename)
                output = open(output_filename,'wb')
                output.write(fileobj.read())
                output.close()
                hdulist = fits.open(output_filename)
                catalog = Table(hdulist[1].data)


        return catalog

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












