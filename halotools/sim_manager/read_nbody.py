# -*- coding: utf-8 -*-
"""
Methods and classes to load halo and particle catalogs into memory.

"""

__all__=['ProcessedSnapshot','CatalogManager']

#from __future__ import (absolute_import, division, print_function,
#                        unicode_literals)

from astropy.io import fits as fits
from astropy.table import Table
from astropy.utils.data import get_readable_fileobj
from astropy.utils.data import _get_download_cache_locs as get_download_cache_locs
from astropy.utils.data import _open_shelve as open_shelve

from . import configuration
from ..utils.array_utils import find_idx_nearest_val
from ..utils.array_utils import array_like_length as aph_len
from ..utils.io_utils import download_file_from_url

import numpy as np

import os, sys, warnings, urllib2
import sim_defaults

HAS_SOUP = False
try:
    from bs4 import BeautifulSoup
    HAS_SOUP = True
except:
    pass

HAS_REQUESTS = False
try:
    import requests
    HAS_REQUESTS = True
except:
    pass


class ProcessedSnapshot(object):
    """ Class containing halo and particle data taken from 
    a single snapshot of some Nbody simulation.
    """

    def __init__(self,
        simname=sim_defaults.default_simulation_name,
        scale_factor=sim_defaults.default_scale_factor,
        halo_finder=sim_defaults.default_halo_finder,
        download_yn=True, **kwargs):

        self.simulation_name = simname
        self.scale_factor = scale_factor
        self.halo_finder = halo_finder

        self.download_yn = download_yn

        catman = CatalogManager()
        self.CatalogManager = catman

        Lbox, mp, softening = catman.get_simulation_properties(self.simulation_name)
        self.Lbox = Lbox
        self.particle_mass = mp
        self.softening_length = softening

        halo_catalog_filename,closest_scale_factor = (
            catman.find_nearest_snapshot_in_cache('subhalos',
                scale_factor = self.scale_factor,
                simname=self.simulation_name,
                halo_finder=self.halo_finder)
            )

        # If there are no matching halo catalogs in cache,
        # set the halo catalog to the default halo catalog
        if (halo_catalog_filename==None) or (closest_scale_factor != self.scale_factor):
            halo_catalog_filename = catman.default_halo_catalog_filename
            # Download the catalog, if desired
            if download_yn==True:
                catman.download_all_default_catalogs()

        self.halo_catalog_filename = halo_catalog_filename
        self.halo_catalog_dirname = configuration.get_catalogs_dir('subhalos')

        particle_catalog_filename,closest_scale_factor = (
            catman.find_nearest_snapshot_in_cache('particles',
                scale_factor = self.scale_factor,
                simname=self.simulation_name,
                halo_finder=self.halo_finder)
            )

        # If there are no matching particle catalogs in cache,
        # set the particle catalog to the default particle catalog
        if (particle_catalog_filename==None) or (closest_scale_factor != self.scale_factor):
            particle_catalog_filename = catman.default_particle_catalog_filename
            # Download the catalog, if desired
            if download_yn==True:
                catman.download_all_default_catalogs()

        self.particle_catalog_filename = particle_catalog_filename
        self.particle_catalog_dirname = configuration.get_catalogs_dir('particles')

    @property
    def particles(self):
        """ Method to load simulation particle data into memory. 

        The property decorator syntax allows this method to be called 
        as if it is an attribute.

        """

        catalog_type = 'particles'
        particles = self.CatalogManager.load_catalog(catalog_type,
            dirname = self.particle_catalog_dirname,
            filename=self.particle_catalog_filename,
            download_yn = self.download_yn)

        return particles


    @property
    def halos(self):
        """ Method to load simulation halo catalog into memory. 

        The property decorator syntax allows this method to be called 
        as if it is an attribute.

        """
        catalog_type = 'subhalos'
        halos = self.CatalogManager.load_catalog(catalog_type,
            dirname = self.halo_catalog_dirname,
            filename=self.halo_catalog_filename,
            download_yn = self.download_yn)

        return halos

###################################################################################################

class CatalogManager(object):
    """ Container class for managing I/O of halo & particle catalogs.
    """

    def __init__(self):
        self.slac_urls = {'bolshoi_halos' : 'http://www.slac.stanford.edu/~behroozi/Bolshoi_Catalogs/',
        'bolshoi_bdm_halos' : 'http://www.slac.stanford.edu/~behroozi/Bolshoi_Catalogs_BDM/',
        'multidark_halos' : 'http://slac.stanford.edu/~behroozi/MultiDark_Hlists_Rockstar/',
        'consuelo_halos' : 'http://www.slac.stanford.edu/~behroozi/Consuelo_Catalogs/'
        }

        self.default_halo_catalog_filename = (
            sim_defaults.default_simulation_name+'_a'+
            str(np.round(sim_defaults.default_scale_factor,4))+'_'+
            sim_defaults.default_halo_finder+'_subhalos.hdf5')

        self.default_particle_catalog_filename = (
            sim_defaults.default_simulation_name+'_a'+
            str(np.round(sim_defaults.default_scale_factor,4))+
            '_particles.fits')


    def download_raw_halocat(self, simname, halo_finder, input_redshift, 
        dz_tol=0.1, download_loc='halotools_cache', overwrite=False):
        """ Method to download publicly available ascii data of 
        raw halo catalog from web location. 

        Parameters 
        ----------
        simname : string 
            Nickname of the simulation, e.g. `bolshoi`. 

        halo_finder : string 
            Nickname of the halo-finder, e.g. `rockstar`. 

        input_redshift : float 
            Redshift of the requested snapshot. Must match one of the 
            available snapshots, or a prompt will be issued providing the nearest 
            available snapshots to choose from. 

        dz_tol : float, optional
            Tolerance value determining how close the requested redshift must be to 
            some available snapshot before issuing a warning. Default value is 0.1. 

        download_loc : string, optional
            Determines where the raw halo catalog will be stored. Default is the 
            halotools cache directory. Any value besides 'halotools_cache' 
            will be interpreted as an absolute path of the file being downloaded.

        overwrite : boolean, optional
            If a file with the same filename already exists 
            in the requested download location, the `overwrite` boolean determines 
            whether or not to overwrite the file. Default is False. 
        """

        if HAS_SOUP == False:
            print("Must have BeautifulSoup installed to use Halotools Catalog Manager")
            return 

        # Check the url for the (unprocessed) halo catalog 
        # that most closely matches the requested redshift
        list_of_available_sims = self.retrieve_available_raw_halocats(simname, halo_finder)
        closest_snapshot_fname = self.find_closest_raw_halocat(
            list_of_available_sims, input_redshift)
        scale_factor_of_closest_match = float(
            self.get_scale_factor_substring(
            closest_snapshot_fname))
        redshift_of_closest_match = (1./scale_factor_of_closest_match) - 1

        if abs(redshift_of_closest_match - input_redshift) > dz_tol:
            msg = (
                "No raw %s halo catalog has \na redshift within %.2f " + 
                "of the input_redshift = %.2f.\n The closest redshift for these catalogs is %.2f"
                )
            print(msg % (simname, dz_tol, input_redshift, redshift_of_closest_match))
            return 

        key = simname+'_'+halo_finder
        url = sim_defaults.raw_halocat_url[key]+closest_snapshot_fname

        if download_loc != 'halotools_cache':
            # We were given an explicit path to store the catalog
            # Check that this path actually exists, and if so, use it 
            if not os.path.exists(download_loc):
                raise IOError("Input directory name %s for download location"
                    "of raw halo catalog does not exist" % download_loc)
            else:
                output_fname = os.path.join(download_loc, closest_snapshot_fname)
                self.update_list_of_previously_used_dirnames(download_loc)
        else:
            # We were not given an explicit path, so use the default Halotools cache dir
            cache_dirname = configuration.get_catalogs_dir('raw_halos', 
                simname=simname, halo_finder=halo_finder)

            output_fname = os.path.join(cache_dirname, closest_snapshot_fname)

        if os.path.isfile(output_fname):
            if overwrite==True:
                warnings.warn("Downloading halo catalog and overwriting existing file %s" % output_fname)
            else:
                raise IOError("The following filename already exists: \n%s\n"
                    "If you really want to overwrite the file, \n"
                    "you must call this function again with the "
                    "keyword argument `overwrite` set to `True`" % output_fname)

        download_file_from_url(url, output_fname)

    def locate_raw_halocat(self, **kwargs):
        """ Return full path to the requested raw halo catalog. 

        Parameters 
        ----------
        fname : string, optional keyword argument 
            Absolute pathname to the raw halo catalog. 
            If not passed, must pass simname and redshift 
            keyword arguments.

        simname : string, optional keyword argument 
            Nickname of the simulation, e.g., `bolshoi`. 

        halo_finder : string, optional keyword argument 
            Nickname of the halo-finder, e.g., `rockstar` or `bdm`.
            Default is `rockstar`.  

        redshift : float, optional keyword argument 
            Redshift of the requested snapshot.

        Returns 
        -------
        halocat_fname : string 
            Absolute path to raw halo catalog

        Notes 
        -----
        If `fname`, `simname`, and `redshift` arguments are all supplied, 
        method will return `fname`. 

        """

        # First check that we were provided sufficient inputs
        if ('fname' not in kwargs.keys()):
            if ('simname' not in kwargs.keys()) or ('redshift' not in kwargs.keys()):
                msg = ("If not passing an absolute filename to locate_raw_halocat,\n"
                    "must pass both a simname and a redshift")
                raise IOError(msg)

        if 'fname' in kwargs.keys():
            return kwargs['fname']
        else:
            if 'halo_finder' not in kwargs.keys():
                halo_finder = sim_defaults.default_halo_finder
            else:
                halo_finder = kwargs['halo_finder']
            return self.full_fname_closest_raw_halocat_in_cache(
                kwargs['simname'], halo_finder, kwargs['redshift'])

    def get_raw_halocat_reader(self, simname):
        pass
 
    def update_list_of_previously_used_dirnames(self, 
        catalog_type, input_full_fname, input_simname):
        """ Method determines whether the input `dirname` is a new location 
        that has not been used before to store halo catalogs, and updates the 
        package memory as necessary.
        """
        dirname = os.path.dirname(input_full_fname)
        if not os.path.exists(dirname):
            warnings.warn("Cannot append a non-existent path %s \n"
                "to Halotools memory of preferred user cache directory locations")
            return

        cache_memory_loc = configuration.get_catalogs_dir(catalog_type)
        cache_memory_full_fname = os.path.join(cache_memory_loc, 
            sim_defaults.cache_memory_fname)
        # If the cache memory file does not already exist, 
        # issue a warning, create it, and add the header
        if not os.path.isfile(cache_memory_full_fname):
            warnings.warn("Creating the following file: \n %s \n "
                "This file is used to store Halotools' memory\n"
                "of user-preferred locations for data of type %s" % 
                (cache_memory_full_fname, catalog_type)
                )
            with open(cache_memory_full_fname, 'w') as f:
                header_line1 = (
                    "# This file lists all previously used locations storing "
                    "catalogs of the following type: "+catalog_type+"\n"
                    )
                header_line2 = ("# Halotools uses this file "
                    "to automatically detect a previously used disk\n")
                f.write(header_line1)
                f.write(header_line2)

        add_new_simloc = True
        with open(cache_memory_full_fname, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line[0] != '#':
                    existing_fname, existing_simname = line.strip()[0], line.strip()[1]
                    if input_full_fname == existing_fname:
                        add_new_simloc = False
                        if existing_simname != input_simname:
                            msg = ("There already exists filename:\n%s\n "
                                "The existing file pertains to the %s simulation,\n"
                                "but you requested the cache memory to update this file \n"
                                "to correspond to the %s simulation.\n" 
                                "Take care that this was intentional.")

                            warnings.warn(msg % (existing_fname, existing_simname, input_simname))
                else:
                    pass

        if add_new_simloc == True:
            with open(cache_memory_full_fname, 'a') as f:
                f.write(input_full_fname + '  ' + input_simname)


    def process_raw_halocat(self, input_fname, simname, halo_finder, 
        cuts, output_version_name):
        """ Method reads in a raw halo catalog, makes the desired cuts, 
        and stores the reduced catalog as an hdf5 file in the halo catalog 
        cache directory.
        """

        # First determine the simname from input_fname
        idx1 = len(input_fname) - input_fname[::-1].index('/') - 1
        temp_substring = input_fname[:idx1]
        idx2 = len(temp_substring) - temp_substring[::-1].index('/') - 1
        simname = temp_substring[idx2+1:]

        raw_halocat_cache_dir = configuration.get_catalogs_dir('raw_halo_catalog')
        simname_raw_halocat_cache_dir = os.path.join(raw_halocat_cache_dir, simname)
        raw_halocat_full_fname = os.path.join(simname_raw_halocat_cache_dir, input_fname)
        print("\nProcessing halo catalog with "
            "the following filename:\n %s \n" %raw_halocat_full_fname)

        manually_decompressed = False
        if input_fname[-3:] == '.gz':
            unzip_command = 'gunzip '+raw_halocat_full_fname
            os.system(unzip_command)
            raw_halocat_full_fname = raw_halocat_full_fname[:-3]
            manually_decompressed = True


        ### HALOCAT PROCESSING LINES HERE###

        if manually_decompressed == True:
            rezip_command = 'gzip '+raw_halocat_full_fname
            os.system(rezip_command)

    def full_fname_closest_raw_halocat_in_cache(
        self, simname, halo_finder, input_redshift):

        filename_list = self.full_fnames_all_raw_halocats_in_cache(simname, halo_finder)

        closest_fname = self.find_closest_raw_halocat(
            filename_list, input_redshift)
        if closest_fname == None:
            return None

        dirname = configuration.get_catalogs_dir('raw_halos', 
            simname=simname, halo_finder=halo_finder)
        output_full_fname = os.path.join(dirname, closest_fname)

        return output_full_fname


    def full_fnames_all_raw_halocats_in_cache(self, simname, halo_finder):
        """ Method returns all snapshots for the input simulation 
        that are available in the cache directory. 

        Parameters 
        ----------
        simname : string 
            Nickname of the simulation. Must match one of the keys in 
            the ``raw_halocat_url`` dictionary stored in the ``sim_defaults`` module. 

        Returns 
        -------
        file_list : list 
            List of strings of all raw catalogs in the cache directory. 
        """
        catalog_dirname = configuration.get_catalogs_dir('raw_halos')
        simname_raw_halocat_cache_dir = os.path.join(catalog_dirname, simname)
        if not os.path.exists(simname_raw_halocat_cache_dir):
            print("No raw halo catalogs exist in cache for simname %s" % simname)
            return
        catname_raw_halocat_cache_dir = os.path.join(simname_raw_halocat_cache_dir, halo_finder)
        if not os.path.exists(catname_raw_halocat_cache_dir):
            print("No %s simulation raw halo catalogs exist for halo-finder %s" % halo_finder)
            return

        file_list = (
            [ f for f in os.listdir(catname_raw_halocat_cache_dir) 
            if os.path.isfile(os.path.join(catname_raw_halocat_cache_dir,f)) ]
            )

        full_fname_list = [os.path.join(catname_raw_halocat_cache_dir, f) for f in file_list]

        return full_fname_list


    def find_closest_raw_halocat(self, filename_list, input_redshift):
        """ Method searches the url where the ``simname`` halo catalogs are stored, 
        and returns the filename of the closest matching snapshot to ``input_redshift``. 

        Parameters 
        ----------
        filename_list : list of strings
            Each entry of the list must be a filename of the type generated by Rockstar. 

        input_redshift : float
            Redshift of the requested snapshot.

        Returns
        -------
        output_fname : string 
            Filename of the closest matching snapshot. 
        """

        if aph_len(filename_list)==0:
            return None

        # First create a list of floats storing the scale factors of each hlist file
        scale_factor_list = []
        for full_fname in filename_list:
            fname = os.path.basename(full_fname)
            scale_factor_substring = self.get_scale_factor_substring(fname)
            scale_factor = float(scale_factor_substring)
            scale_factor_list.append(scale_factor)
        scale_factor_list = np.array(scale_factor_list)

        # Now use the array utils module to determine 
        # which scale factor is the closest
        input_scale_factor = 1./(1. + input_redshift)
        idx_closest_catalog = find_idx_nearest_val(scale_factor_list, input_scale_factor)

        closest_scale_factor = scale_factor_list[idx_closest_catalog]

        output_fname = filename_list[idx_closest_catalog]

        return output_fname


    def get_scale_factor_substring(self, fname):
        """ Method extracts the portion of the Rockstar hlist fname 
        that contains the scale factor of the snapshot. 

        Parameters 
        ----------
        fname : string 
            Filename of the hlist. 

        Returns 
        -------
        scale_factor_substring : string 
            The substring specifying the scale factor of the snapshot. 

        Notes 
        -----
        Assumes that the relevant substring always immediately follows 
        the first incidence of an underscore, and terminates immediately 
        preceding the second decimal. These assumptions are valid for 
        all catalogs currently on the hipacc website, including 
        'bolshoi', 'bolshoi_bdm', 'consuelo', and 'multidark'. 

        Examples
        --------
        >>> catman = CatalogManager()
        >>> fname = 'hlist_0.06630.list.gz'
        >>> scale_factor_string = catman.get_scale_factor_substring(fname)

        """
        first_index = fname.index('_')+1
        last_index = fname.index('.', fname.index('.')+1)
        scale_factor_substring = fname[first_index:last_index]
        return scale_factor_substring

    def retrieve_available_raw_halocats(self, simname, halo_finder):
        """ Method returns all snapshots for the input simulation 
        that are available at the host url. 

        Parameters 
        ----------
        simname : string 
            Nickname of the simulation, e.g., `bolshoi`. 

        halo_finder : string 
            Nickname of the halo-finder that generated the catalog, 
            e.g., `rockstar`. 

        Returns 
        -------
        file_list : list 
            List of strings of all raw catalogs available for the requested simulation. 

        Notes 
        ----- 
        Method assumes that the first characters of any halo catalog filename are `hlist_`. 

        """
        key = simname + '_' + halo_finder
        if key in sim_defaults.raw_halocat_url.keys():
            url = sim_defaults.raw_halocat_url[key]
        else:
            raise KeyError("Input simname + halo-finder does not correspond to "
                "any of the catalogs stored in sim_defaults.raw_halocat_url")

        soup = BeautifulSoup(requests.get(url).text)
        expected_filename_prefix = 'hlist_'
        file_list = []
        for a in soup.find_all('a'):
            link = a['href']
            if link[0:len(expected_filename_prefix)]==expected_filename_prefix:
                file_list.append(a['href'])

        return file_list

    def retrieve_catalog_filenames_from_url(self,url,catalog_type='subhalos'):
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
        if url==sim_defaults.behroozi_web_location:
            ### Set naming conventions of the files hosted at SLAC
            if (catalog_type == 'subhalo') or (catalog_type=='subhalos'): 
                expected_filename_prefix = 'hlist_'
            elif (catalog_type == 'tree') or (catalog_type == 'trees'):
                expected_filename_prefix = 'tree_'
            else:
                raise TypeError("Input catalog type must either be 'subhalos' or 'trees'")
            ### Grab all filenames with the assumed prefix
            for a in soup.find_all('a'):
                link = a['href']
                if link[0:len(expected_filename_prefix)]==expected_filename_prefix: 
                    file_list.append(a['href'])
        ##################
        ### APH url case (simpler, since only two default catalogs are hosted here)
        elif url==sim_defaults.aph_web_location:
            ### Set naming conventions of the files hosted at Yale
            if (catalog_type == 'subhalo') or (catalog_type=='subhalos'): 
                expected_filename_suffix = 'halos.fits'
            elif (catalog_type == 'particle') or (catalog_type=='particles'):
                expected_filename_suffix = 'particles.fits'
            else:
                expected_filename_suffix = '.fits'
            ### Grab all filenames with the assumed suffix
            for a in soup.find_all('a'):
                link = a['href']
                if link[-len(expected_filename_suffix):]==expected_filename_suffix: 
                    file_list.append(a['href'])
        ##################
        ### Some other url managed by the user
        else:
            for a in soup.find_all('a'):
                link = a['href']
                file_list.append(a['href'])

        return file_list
        ##################

    def get_simulation_properties(self,simname):
        """ Return a few characteristics of the input simulation.

        Parameters 
        ----------
        simname : string 
            Specifies the simulation of interest, e.g., 'bolshoi'.

        Returns 
        -------
        Lbox : float 
            Box size in Mpc/h.

        particle_mass : float 
            Particle mass in Msun/h.

        softening : float 
            Softening length in kpc/h.

        """

        Lbox, particle_mass, softening = None, None, None

        if (simname=='bolshoi'):
            Lbox = 250.0
            particle_mass = 1.35e8
            softening = 1.0

        return Lbox, particle_mass, softening 

    def id_rel_cats(self,catalog_type=None,
        simname=None,halo_finder=None):

        # Fix possible pluralization mistake of user
        if catalog_type == 'subhalo': catalog_type='subhalos'
        if catalog_type == 'particle': catalog_type='particles'

        # Identify all catalogs currently stored in the cache directory
        available_catalogs = np.array(
            configuration.list_of_catalogs_in_cache(catalog_type=catalog_type))

        ######### 
        # Define a simple string manipulating function that will be used to 
        # parse a filename into the features defining the catalog
        def parse_fname(s,extname):
            if extname[0] != '.':
                extname = '.'+extname
            s = s[:-len(extname)]

            segment_separations = []
            # Loop over the string to identify the segments
            for idx, char in enumerate(s):
                if char=='_':
                    segment_separations.append(idx)

            segments = []
            first_idx=0
            for idx in segment_separations:
                last_idx=idx
                segment = s[slice(first_idx,last_idx)]
                segments.append(segment)
                first_idx=last_idx+1
            last_idx=None
            segment = s[slice(first_idx,last_idx)]
            segments.append(segment)

            return segments

        #########
        # The file_mask array will determine which of the available catalogs 
        # pass the input specifications
        file_mask  = np.ones(len(available_catalogs),dtype=bool)

        extension = '.hdf5'
        for catidx,catname in enumerate(available_catalogs):
            parsed_string = parse_fname(catname,extension)
            ###
            sim_string = parsed_string[0]
            if (simname != None) & (sim_string != simname):
                file_mask[catidx] = False
            ###
            halo_finder_string = parsed_string[2]
            if (halo_finder != None) & (halo_finder != halo_finder_string) :
                file_mask[catidx] = False

        relevant_catalogs = available_catalogs[file_mask]
        return relevant_catalogs



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
        if catalog_type == 'subhalo': catalog_type='subhalos'
        if catalog_type == 'particle': catalog_type='particles'

        # Identify all catalogs currently stored in the cache directory
        available_catalogs = np.array(
            configuration.list_of_catalogs_in_cache(catalog_type=catalog_type))

        #########
        # The file_mask array will determine which of the available catalogs 
        # pass the input specifications
        file_mask  = np.ones(len(available_catalogs),dtype=bool)

        # Impose halos vs. particles restriction
        extension = '.hdf5'
        if catalog_type != None:
            last_characters_of_filename=catalog_type+extension
            for ii,c in enumerate(available_catalogs):
                if c[-len(catalog_type)-len(extension):] != last_characters_of_filename:
                    file_mask[ii]=False

       # Impose simulation name restriction
        if simname != None:
            first_characters_of_filename = simname
            for ii,c in enumerate(available_catalogs):
                if (c[0:len(simname)] != first_characters_of_filename) or (c[len(simname)] != '_'):
                    file_mask[ii]=False

        # Impose halo finder restriction
        if halo_finder != None:
            for ii,c in enumerate(available_catalogs):
                if c[-11-len(halo_finder):-11] != halo_finder:
                    file_mask[ii]=False
        relevant_catalogs = available_catalogs[file_mask]

        return relevant_catalogs


    def find_nearest_snapshot_in_cache(self,catalog_type,
        scale_factor=None,redshift=None,
        simname=sim_defaults.default_simulation_name,
        halo_finder = sim_defaults.default_halo_finder):

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
        if catalog_type == 'subhalo': catalog_type='subhalos'
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

        relevant_catalogs = self.id_rel_cats(
            catalog_type=catalog_type,simname=simname,halo_finder=halo_finder)

        if len(relevant_catalogs)==0:
            if catalog_type=='subhalos':
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
        adiff_tol = sim_defaults.scale_factor_difference_tol
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

    def load_catalog(self,catalog_type,
        dirname=None,filename=None,
        download_yn=False,url=sim_defaults.aph_web_location):
        """ Use the astropy reader to load the halo or particle catalog into memory.

        Parameters 
        ----------
        dirname : string 
            Name of directory where filename is stored.

        filename : string 
            Name of file being loaded into memory. 

        download_yn : boolean, optional
            If set to True, and if filename is not already stored in the cache directory, 
            method will attempt to download the file from the provided url. If there is no corresponding 
            file at the input url, an exception will be raised.

        url : string 
            Web location from which to download the catalog if it is not present in the cache directory.

        Returns 
        -------
        catalog : object
            Data structure located at the input filename.

        """
        if filename==None:
            if catalog_type=='subhalos':
                filename = self.default_halo_catalog_filename
            elif catalog_type=='particles':
                filename = self.default_particle_catalog_filename
            else:
                raise KeyError("Must supply catalog_type to be either "
                    "'particles' or 'subhalos'")
        if dirname==None:
            if catalog_type=='subhalos':
                dirname = self.halo_catalog_dirname
            elif catalog_type=='particles':
                dirname = self.particle_catalog_dirname
            else:
                raise KeyError("Must supply catalog_type to be either "
                    "'particles' or 'subhalos'")

        if os.path.isfile(os.path.join(dirname,filename)):
            catalog = Table.read(os.path.join(dirname,filename),path='data')
        else:
            ### Requested filename is not in cache, and external download is not requested
            if download_yn==False:
                return None
            else:
                # Download one of the default catalogs hosted at Yale
                if filename==self.default_halo_catalog_filename:
                    catalog_type='subhalos'
                if filename==self.default_particle_catalog_filename:
                    catalog_type='particles'
                else:
                    raise IOError("Input filename does not match one of the provided default catalogs")
                ###
                remote_filename = os.path.join(url,filename)
                fileobj = urllib2.urlopen(remote_filename)
                output_directory = configuration.get_catalogs_dir(catalog_type)
                output_filename = os.path.join(output_directory,filename)
                output = open(output_filename,'wb')
                output.write(fileobj.read())
                output.close()
                hdulist = fits.open(output_filename)
                catalog = Table(hdulist[1].data)

        return catalog


    def download_all_default_catalogs(self):
        """ If not already in cache, 
        download default particle and halo catalogs from Yale website.
        """

        url = sim_defaults.aph_web_location

        ### Download halo catalogs
        catalog_type = 'subhalos'
        output_directory = configuration.get_catalogs_dir(catalog_type)
        filename = self.default_halo_catalog_filename
        remote_filename = os.path.join(url,filename)
        if not os.path.isfile(os.path.join(output_directory,filename)):
            warnings.warn("Downloading default halo catalog")
            fileobj = urllib2.urlopen(remote_filename)
            output_filename = os.path.join(output_directory,filename)
            output = open(output_filename,'wb')
            output.write(fileobj.read())
            output.close()

        ### Download particle catalogs
        catalog_type = 'particles'
        output_directory = configuration.get_catalogs_dir(catalog_type)
        filename = self.default_particle_catalog_filename
        remote_filename = os.path.join(url,filename)
        if not os.path.isfile(os.path.join(output_directory,filename)):
            warnings.warn("Downloading default particle catalog")
            fileobj = urllib2.urlopen(remote_filename)
            output_filename = os.path.join(output_directory,filename)
            output = open(output_filename,'wb')
            output.write(fileobj.read())
            output.close()

    def clear_cache_memory(self, catalog_type):

        cache_memory_loc = configuration.get_catalogs_dir(catalog_type)
        cache_memory_fname = catalog_type+'_'+sim_defaults.cache_memory_fname
        cache_memory_full_fname = os.path.join(cache_memory_loc, cache_memory_fname)
        if os.path.exists(cache_memory_full_fname):
            os.system("rm "+cache_memory_full_fname)




###################################################################################################


###################################################################################################

