# -*- coding: utf-8 -*-
"""
Methods and classes for halo catalog I/O and organization.

"""

import numpy as np
from warnings import warn
from time import time

try:
    from bs4 import BeautifulSoup
except ImportError:
    raise("Must have bs4 package installed to use the catalog_manager module")

try:
    import requests
except ImportError:
    raise("Must have requests package installed to use the catalog_manager module")
import posixpath
import urlparse

from ..utils.array_utils import find_idx_nearest_val
from ..utils.array_utils import array_like_length as custom_len
from ..utils.io_utils import download_file_from_url

from astropy.tests.helper import remote_data


from . import supported_sims, cache_config, sim_defaults
import os, fnmatch, re
from functools import partial

unsupported_simname_msg = "Input simname ``%s`` is not recognized by Halotools"

class CatalogManager(object):
    """ Class used to scrape the web for simulation data  
    and manage the set of cached catalogs. 
    """

    def __init__(self):
        pass

    def _scrape_cache(self, catalog_type, **kwargs):
        """ Private method that is the workhorse behind 
        `processed_halocats_in_cache`, `raw_halocats_in_cache`, and `ptcl_cats_in_cache`. 

        Parameters 
        ----------
        catalog_type : string 
            Specifies which subdirectory of the Halotools cache to scrape for .hdf5 files. 
            Must be either 'halos', 'particles', or 'raw_halos'

        simname : string, optional
            Nickname of the simulation. Currently supported simulations are 
            Bolshoi  (simname = ``bolshoi``), Consuelo (simname = ``consuelo``), 
            MultiDark (simname = ``multidark``), and Bolshoi-Planck (simname = ``bolplanck``). 

            Argument is used to filter the output list of filenames. 
            Default is None, in which case `processed_halocats_in_cache` 
            will not filter the returned list of filenames by ``simname``. 

        halo_finder : string, optional
            Nickname of the halo-finder, e.g. ``rockstar``. 
            Argument is used to filter the output list of filenames. 
            Default is None, in which case `processed_halocats_in_cache` 
            will not filter the returned list of filenames by ``halo_finder``. 

        version_name : string, optional 
            String specifying the version of the processed halo catalog. 
            Argument is used to filter the output list of filenames. 
            Default is None, in which case `processed_halocats_in_cache` 
            will not filter the returned list of filenames by ``version_name``. 

        external_cache_loc : string, optional 
            Absolute path to an alternative source of halo catalogs. 

        Returns
        -------
        fname_list : list 
            List of strings of the filenames (including absolute path) of 
            processed halo stored in the cache directory, filtered according 
            to the input arguments. 
        """
        if 'simname' in kwargs.keys():
            if cache_config.simname_is_supported(kwargs['simname']) is False:
                raise KeyError(unsupported_simname_msg % kwargs['simname'])

        if ('version_name' in kwargs.keys()) & (catalog_type is not 'halos'):
            raise KeyError("The _scrape_cache method received a version_name = %s keyword "
                "argument, which should not be passed for catalog_type = %s" % (kwargs['version_name'], catalog_type))

        if 'external_cache_loc' in kwargs.keys():
            cachedir = os.path.abspath(kwargs['external_cache_loc'])
            if os.path.isdir(cachedir) is False:
                raise KeyError("Input external_cache_loc directory = %s \n Directory does not exist" % cachedir)
        else:
            cachedir = cache_config.get_catalogs_dir(catalog_type = catalog_type)

        fname_pattern = '.hdf5'
        if 'version_name' in kwargs.keys():
            fname_pattern = kwargs['version_name'] + fname_pattern
        if 'halo_finder' in kwargs.keys():
            fname_pattern = kwargs['halo_finder'] + '*' + fname_pattern
        if 'simname' in kwargs.keys():
            fname_pattern = kwargs['simname'] + '*' + fname_pattern
        fname_pattern = '*' + fname_pattern

        full_fname_list = []
        for path, dirlist, filelist in os.walk(cachedir):
            for name in filelist:
                full_fname_list.append(os.path.join(path,name))

        fname_list = fnmatch.filter(full_fname_list, fname_pattern)
                
        return fname_list

    def processed_halocats_in_cache(self, **kwargs):
        """
        Parameters 
        ----------
        simname : string, optional
            Nickname of the simulation. Currently supported simulations are 
            Bolshoi  (simname = ``bolshoi``), Consuelo (simname = ``consuelo``), 
            MultiDark (simname = ``multidark``), and Bolshoi-Planck (simname = ``bolplanck``). 
            
            Argument is used to filter the output list of filenames. 
            Default is None, in which case `processed_halocats_in_cache` 
            will not filter the returned list of filenames by ``simname``. 

        halo_finder : string, optional
            Nickname of the halo-finder, e.g. ``rockstar``. 
            Argument is used to filter the output list of filenames. 
            Default is None, in which case `processed_halocats_in_cache` 
            will not filter the returned list of filenames by ``halo_finder``. 

        version_name : string, optional 
            String specifying the version of the processed halo catalog. 
            Argument is used to filter the output list of filenames. 
            Default is None, in which case `processed_halocats_in_cache` 
            will not filter the returned list of filenames by ``version_name``. 

        external_cache_loc : string, optional 
            Absolute path to an alternative source of halo catalogs. 

        Returns
        -------
        fname_list : list 
            List of strings of the filenames (including absolute path) of 
            processed halo catalogs in the cache directory. 
            Filenames are filtered according to the input arguments. 
        """

        f = partial(self._scrape_cache, catalog_type='halos')
        return f(**kwargs)

    def raw_halocats_in_cache(self, **kwargs):
        """
        Parameters 
        ----------
        simname : string, optional
            Nickname of the simulation. Currently supported simulations are 
            Bolshoi  (simname = ``bolshoi``), Consuelo (simname = ``consuelo``), 
            MultiDark (simname = ``multidark``), and Bolshoi-Planck (simname = ``bolplanck``). 
            
            Argument is used to filter the output list of filenames. 
            Default is None, in which case `processed_halocats_in_cache` 
            will not filter the returned list of filenames by ``simname``. 

        halo_finder : string, optional
            Nickname of the halo-finder, e.g. ``rockstar``. 
            Argument is used to filter the output list of filenames. 
            Default is None, in which case `processed_halocats_in_cache` 
            will not filter the returned list of filenames by ``halo_finder``. 

        external_cache_loc : string, optional 
            Absolute path to an alternative source of halo catalogs. 

        Returns
        -------
        fname_list : list 
            List of strings of the filenames (including absolute path) of 
            all unprocessed halo catalog ASCII data tables in the cache directory. 
            Filenames are filtered according to the input arguments. 
        """
        f = partial(self._scrape_cache, catalog_type='raw_halos')
        return f(**kwargs)

    def ptcl_cats_in_cache(self, **kwargs):
        """
        Parameters 
        ----------
        simname : string, optional
            Nickname of the simulation. Currently supported simulations are 
            Bolshoi  (simname = ``bolshoi``), Consuelo (simname = ``consuelo``), 
            MultiDark (simname = ``multidark``), and Bolshoi-Planck (simname = ``bolplanck``). 
            
            Argument is used to filter the output list of filenames. 
            Default is None, in which case `processed_halocats_in_cache` 
            will not filter the returned list of filenames by ``simname``. 

        external_cache_loc : string, optional 
            Absolute path to an alternative source of halo catalogs. 

        Returns
        -------
        fname_list : list 
            List of strings of the filenames (including absolute path) of 
            all files in the cache directory storing a random downsampling of 
            dark matter particles. 
            Filenames are filtered according to the input arguments. 
        """
        f = partial(self._scrape_cache, catalog_type='particles')
        return f(**kwargs)

    def processed_halocats_available_for_download(self, **kwargs):
        """ Method searches the appropriate web location and 
        returns a list of the filenames of all reduced  
        halo catalog binaries processed by Halotools 
        that are available for download. 

        Parameters 
        ----------
        simname : string, optional
            Nickname of the simulation. Currently supported simulations are 
            Bolshoi  (simname = ``bolshoi``), Consuelo (simname = ``consuelo``), 
            MultiDark (simname = ``multidark``), and Bolshoi-Planck (simname = ``bolplanck``). 
            
            Argument is used to filter the output list of filenames. 
            Default is None, in which case `processed_halocats_in_cache` 
            will not filter the returned list of filenames by ``simname``. 

        halo_finder : string, optional
            Nickname of the halo-finder, e.g. ``rockstar``. 
            Argument is used to filter the output list of filenames. 
            Default is None, in which case `processed_halocats_in_cache` 
            will not filter the returned list of filenames by ``halo_finder``. 

        Returns 
        -------
        output : list 
            List of web locations of all pre-processed halo catalogs 
            matching the input arguments. 

        """
        if 'simname' in kwargs.keys():
            if cache_config.simname_is_supported(kwargs['simname']) is False:
                raise KeyError(unsupported_simname_msg % kwargs['simname'])

        baseurl = sim_defaults.processed_halocats_webloc
        soup = BeautifulSoup(requests.get(baseurl).text)
        simloclist = []
        for a in soup.find_all('a', href=True):
            dirpath = posixpath.dirname(urlparse.urlparse(a['href']).path)
            if dirpath and dirpath[0] != '/':
                simloclist.append(os.path.join(baseurl, dirpath))

        halocatloclist = []
        for simloc in simloclist:
            soup = BeautifulSoup(requests.get(simloc).text)
            for a in soup.find_all('a', href=True):
                dirpath = posixpath.dirname(urlparse.urlparse(a['href']).path)
                if dirpath and dirpath[0] != '/':
                    halocatloclist.append(os.path.join(simloc, dirpath))

        catlist = []
        for halocatdir in halocatloclist:
            soup = BeautifulSoup(requests.get(halocatdir).text)
            for a in soup.find_all('a'):
                catlist.append(os.path.join(halocatdir, a['href']))

        file_pattern = sim_defaults.default_version_name + '.hdf5'
        all_halocats = fnmatch.filter(catlist, '*'+file_pattern)

        # all_halocats a list of all pre-processed catalogs on the web
        # Now we apply our filter, if applicable

        if ('simname' in kwargs.keys()) & ('halo_finder' in kwargs.keys()):
            simname = kwargs['simname']
            halo_finder = kwargs['halo_finder']
            file_pattern = '*'+simname+'/'+halo_finder+'/*' + file_pattern
            output = fnmatch.filter(all_halocats, file_pattern)
        elif 'simname' in kwargs.keys():
            simname = kwargs['simname']
            file_pattern = '*'+simname+'/*' + file_pattern
            output = fnmatch.filter(all_halocats, file_pattern)
        elif 'halo_finder' in kwargs.keys():
            halo_finder = kwargs['halo_finder']
            file_pattern = '*/' + halo_finder + '/*' + file_pattern
            output = fnmatch.filter(all_halocats, file_pattern)
        else:
            output = all_halocats

        return output

    def _orig_halocat_web_location(self, **kwargs):
        """
        Parameters 
        ----------
        simname : string
            Nickname of the simulation. Currently supported simulations are 
            Bolshoi  (simname = ``bolshoi``), Consuelo (simname = ``consuelo``), 
            MultiDark (simname = ``multidark``), and Bolshoi-Planck (simname = ``bolplanck``). 
            
        halo_finder : string
            Nickname of the halo-finder, e.g. ``rockstar`` or ``bdm``.

        Returns 
        -------
        webloc : string 
            Web location from which the original halo catalogs were downloaded.  
        """
        try:
            simname = kwargs['simname']
            halo_finder = kwargs['halo_finder']
        except KeyError:
            raise("CatalogManager._orig_halocat_web_location method "
                "must be called with ``simname`` and ``halo_finder`` arguments")

        if simname == 'multidark':
            return 'http://slac.stanford.edu/~behroozi/MultiDark_Hlists_Rockstar/'
        elif simname == 'bolshoi':
            if halo_finder == 'rockstar':
                return 'http://www.slac.stanford.edu/~behroozi/Bolshoi_Catalogs/'
            elif halo_finder == 'bdm':
                return 'http://www.slac.stanford.edu/~behroozi/Bolshoi_Catalogs_BDM/' 
        elif simname == 'bolplanck':
            return 'http://www.slac.stanford.edu/~behroozi/BPlanck_Hlists/' 
        elif simname == 'consuelo':
            return 'http://www.slac.stanford.edu/~behroozi/Consuelo_Catalogs/'
        else:
            raise KeyError("Input simname %s and halo_finder %s do not "
                "have Halotools-recognized web locations" % (simname, halo_finder))

    def raw_halocats_available_for_download(self, **kwargs):
        """ Method searches the appropriate web location and 
        returns a list of the filenames of all relevant 
        raw halo catalogs that are available for download. 

        Parameters 
        ----------
        simname : string
            Nickname of the simulation. Currently supported simulations are 
            Bolshoi  (simname = ``bolshoi``), Consuelo (simname = ``consuelo``), 
            MultiDark (simname = ``multidark``), and Bolshoi-Planck (simname = ``bolplanck``). 
            
            Argument is used to filter the output list of filenames. 
            Default is None, in which case `processed_halocats_in_cache` 
            will not filter the returned list of filenames by ``simname``. 

        halo_finder : string
            Nickname of the halo-finder, e.g. ``rockstar``. 
            Argument is used to filter the output list of filenames. 
            Default is None, in which case `processed_halocats_in_cache` 
            will not filter the returned list of filenames by ``halo_finder``. 

        Returns 
        -------
        output : list 
            List of web locations of all pre-processed halo catalogs 
            matching the input arguments. 

        """
        if 'simname' in kwargs.keys():
            if cache_config.simname_is_supported(kwargs['simname']) is False:
                raise KeyError(unsupported_simname_msg % kwargs['simname'])

        simname = kwargs['simname']
        halo_finder = kwargs['halo_finder']

        url = self._orig_halocat_web_location(**kwargs)

        soup = BeautifulSoup(requests.get(url).text)
        file_list = []
        for a in soup.find_all('a'):
            file_list.append(os.path.join(url, a['href']))

        file_pattern = '*hlist_*'
        output = fnmatch.filter(file_list, file_pattern)

        return output

    def ptcl_cats_available_for_download(self, **kwargs):
        """ Method searches the appropriate web location and 
        returns a list of the filenames of all reduced  
        halo catalog binaries processed by Halotools 
        that are available for download. 

        Parameters 
        ----------
        simname : string, optional
            Nickname of the simulation. Currently supported simulations are 
            Bolshoi  (simname = ``bolshoi``), Consuelo (simname = ``consuelo``), 
            MultiDark (simname = ``multidark``), and Bolshoi-Planck (simname = ``bolplanck``). 
            
            Argument is used to filter the output list of filenames. 
            Default is None, in which case `processed_halocats_in_cache` 
            will not filter the returned list of filenames by ``simname``. 

        Returns 
        -------
        output : list 
            List of web locations of all catalogs of downsampled particles
            matching the input arguments. 

        """
        if 'simname' in kwargs.keys():
            if cache_config.simname_is_supported(kwargs['simname']) is False:
                raise KeyError(unsupported_simname_msg % kwargs['simname'])

        baseurl = sim_defaults.ptcl_cats_webloc
        soup = BeautifulSoup(requests.get(baseurl).text)
        simloclist = []
        for a in soup.find_all('a', href=True):
            dirpath = posixpath.dirname(urlparse.urlparse(a['href']).path)
            if dirpath and dirpath[0] != '/':
                simloclist.append(os.path.join(baseurl, dirpath))

        catlist = []
        for simloc in simloclist:
            soup = BeautifulSoup(requests.get(simloc).text)
            for a in soup.find_all('a'):
                catlist.append(os.path.join(simloc, a['href']))

        file_pattern = 'particles.hdf5'
        all_ptcl_cats = fnmatch.filter(catlist, '*'+file_pattern)

        if 'simname' in kwargs.keys():
            simname = kwargs['simname']
            file_pattern = '*'+simname+'/*' + file_pattern
            output = fnmatch.filter(all_ptcl_cats, file_pattern)
        else:
            output = all_ptcl_cats

        return output

    def _get_scale_factor_substring(self, fname):
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
        Assumes that the first character of the relevant substring 
        is the one immediately following the first incidence of an underscore, 
        and final character is the one immediately preceding the second decimal. 
        These assumptions are valid for all catalogs currently on the hipacc website. 

        """
        first_index = fname.index('_')+1
        last_index = fname.index('.', fname.index('.')+1)
        scale_factor_substring = fname[first_index:last_index]
        return scale_factor_substring

    def _closest_fname(self, filename_list, desired_redshift):

        if desired_redshift == -1:
            raise ValueError("desired_redshift of -1 is unphysical")
        else:
            input_scale_factor = 1./(1.+desired_redshift) 

        # First create a list of floats storing the scale factors of each hlist file
        scale_factor_list = []
        for full_fname in filename_list:
            fname = os.path.basename(full_fname)
            scale_factor_substring = self._get_scale_factor_substring(fname)
            scale_factor = float(scale_factor_substring)
            scale_factor_list.append(scale_factor)
        scale_factor_list = np.array(scale_factor_list)

        # Now use the array utils module to determine 
        # which scale factor is the closest
        input_scale_factor = 1./(1. + desired_redshift)
        idx_closest_catalog = find_idx_nearest_val(
            scale_factor_list, input_scale_factor)
        closest_scale_factor = scale_factor_list[idx_closest_catalog]
        output_fname = filename_list[idx_closest_catalog]

        closest_available_redshift = (1./closest_scale_factor) - 1

        return output_fname, closest_available_redshift

    def closest_catalog_in_cache(self, catalog_type, desired_redshift, 
        **kwargs):
        """
        Parameters 
        ----------
        desired_redshift : float 
            Redshift of the desired catalog. 

        catalog_type : string 
            Specifies which subdirectory of the Halotools cache to scrape for .hdf5 files. 
            Must be either ``halos``, ``particles``, or ``raw_halos``

        simname : string
            Nickname of the simulation. Currently supported simulations are 
            Bolshoi  (simname = ``bolshoi``), Consuelo (simname = ``consuelo``), 
            MultiDark (simname = ``multidark``), and Bolshoi-Planck (simname = ``bolplanck``). 
            
        halo_finder : string, optional
            Nickname of the halo-finder, e.g. ``rockstar``. 
            Required when input ``catalog_type`` is ``halos`` or ``raw_halos``. 

        version_name : string, optional 
            String specifying the version of the processed halo catalog. 
            Argument is used to filter the output list of filenames. 
            Default is set by ``~halotools.sim_manager.sim_defaults.default_version_name``. 

        external_cache_loc : string, optional 
            Absolute path to an alternative source of catalogs. This file searcher assumes 
            that ``external_cache_loc`` has the same organizational structure as the 
            default Halotools cache. So if you are searching for particles, 
            and ``external_cache_loc`` has particle data for some simulation ``simname``, 
            then there must be a directory ``external_cache_loc/simname`` where the hdf5 files 
            are stored. If you are searching for halos, there must be a directory 
            ``external_cache_loc/simname/halo_finder`` where the halo catalogs are stored. 

        Returns
        -------
        output_fname : list 
            String of the filename with the closest matching redshift. 

        redshift : float 
            Value of the redshift of the snapshot
        """
        if 'simname' in kwargs.keys():
            if cache_config.simname_is_supported(kwargs['simname']) is False:
                raise KeyError(unsupported_simname_msg % kwargs['simname'])

        # Verify arguments are as needed
        if catalog_type is not 'particles':
            try:
                halo_finder = kwargs['halo_finder']
            except KeyError:
                raise("If input catalog_type is not particles, must pass halo_finder argument")
        else:
            if 'halo_finder' in kwargs.keys():
                warn("There is no need to specify a halo-finder when requesting particle data")
                del kwargs['halo_finder']
        
        if (catalog_type == 'halos') & ('version_name' not in kwargs.keys()):
            kwargs['version_name'] = sim_defaults.default_version_name
        filename_list = self._scrape_cache(
            catalog_type = catalog_type, **kwargs)

        if custom_len(filename_list) == 0:
            print("\nNo matching catalogs found by closest_catalog_in_cache method of CatalogManager\n")
            return None

        output_fname, redshift = self._closest_fname(filename_list, desired_redshift)

        return output_fname, redshift

    def closest_catalog_on_web(self, **kwargs):
        """
        Parameters 
        ----------
        desired_redshift : float 
            Redshift of the desired catalog. 

        catalog_type : string 
            Specifies which subdirectory of the Halotools cache to scrape for .hdf5 files. 
            Must be either ``halos``, ``particles``, or ``raw_halos``

        simname : string
            Nickname of the simulation. Currently supported simulations are 
            Bolshoi  (simname = ``bolshoi``), Consuelo (simname = ``consuelo``), 
            MultiDark (simname = ``multidark``), and Bolshoi-Planck (simname = ``bolplanck``). 

        halo_finder : string, optional
            Nickname of the halo-finder, e.g. ``rockstar``. 
            Required when input ``catalog_type`` is ``halos`` or ``raw_halos``. 

        version_name : string, optional 
            String specifying the version of the processed halo catalog. 
            Argument is used to filter the output list of filenames. 
            Default is set by ``~halotools.sim_manager.sim_defaults.default_version_name``. 

        Returns
        -------
        output_fname : list 
            String of the filename with the closest matching redshift. 

        redshift : float 
            Value of the redshift of the snapshot

        Examples 
        --------
        >>> catman = CatalogManager()

        Suppose you would like to download a pre-processed halo catalog for the Bolshoi-Planck simulation for z=0.5. 
        To identify the filename of the available catalog that most closely matches your needs:

        >>> webloc_closest_match = catman.closest_catalog_on_web(catalog_type='halos', simname='bolplanck', halo_finder='rockstar', desired_redshift=0.5)  # doctest: +REMOTE_DATA

        You may also wish to have a collection of downsampled dark matter particles to accompany this snapshot:

        >>> webloc_closest_match = catman.closest_catalog_on_web(catalog_type='particles', simname='bolplanck', desired_redshift=0.5)  # doctest: +REMOTE_DATA

        """
        if 'simname' in kwargs.keys():
            if cache_config.simname_is_supported(kwargs['simname']) is False:
                raise KeyError(unsupported_simname_msg % kwargs['simname'])

        if 'version_name' not in kwargs.keys():
            kwargs['version_name'] = sim_defaults.default_version_name

        # Verify arguments are as needed
        try:
            simname = kwargs['simname']
        except KeyError:
            raise("Must supply input simname keyword argument")

        catalog_type = kwargs['catalog_type']
        if catalog_type is not 'particles':
            try:
                halo_finder = kwargs['halo_finder']
            except KeyError:
                raise("If input catalog_type is not particles, must pass halo_finder argument")
        else:
            if 'halo_finder' in kwargs.keys():
                warn("There is no need to specify a halo-finder when requesting particle data")
                del kwargs['halo_finder']

        if catalog_type is 'particles':
            filename_list = self.ptcl_cats_available_for_download(**kwargs)
        elif catalog_type is 'raw_halos':
            filename_list = self.raw_halocats_available_for_download(**kwargs)
        elif catalog_type is 'halos':
            filename_list = self.processed_halocats_available_for_download(**kwargs)

        desired_redshift = kwargs['desired_redshift']
        output_fname, redshift = self._closest_fname(filename_list, desired_redshift)

        return output_fname, redshift

    def download_raw_halocat(self, dz_tol = 0.1, overwrite=False, **kwargs):
        """ Method to download one of the pre-processed binary files 
        storing a reduced halo catalog.  

        Parameters 
        ----------
        simname : string 
            Nickname of the simulation. Currently supported simulations are 
            Bolshoi  (simname = ``bolshoi``), Consuelo (simname = ``consuelo``), 
            MultiDark (simname = ``multidark``), and Bolshoi-Planck (simname = ``bolplanck``). 
            
        halo_finder : string 
            Nickname of the halo-finder, e.g. `rockstar`. 

        desired_redshift : float 
            Redshift of the requested snapshot. Must match one of the 
            available snapshots, or a prompt will be issued providing the nearest 
            available snapshots to choose from. 

        dz_tol : float, optional
            Tolerance value determining how close the requested redshift must be to 
            some available snapshot before issuing a warning. Default value is 0.1. 

        overwrite : boolean, optional
            If a file with the same filename already exists 
            in the requested download location, the `overwrite` boolean determines 
            whether or not to overwrite the file. Default is False, in which case 
            no download will occur if a pre-existing file is detected. 

        external_cache_loc : string, optional 
            Absolute path to an alternative source of raw halo catalogs. 

        Returns 
        -------
        output_fname : string  
            Filename (including absolute path) of the location of the downloaded 
            halo catalog.  
        """
        if 'simname' in kwargs.keys():
            if cache_config.simname_is_supported(kwargs['simname']) is False:
                raise KeyError(unsupported_simname_msg % kwargs['simname'])

        desired_redshift = kwargs['desired_redshift']

        available_fnames_to_download = self.raw_halocats_available_for_download(**kwargs)

        url, closest_redshift = (
            self._closest_fname(available_fnames_to_download, desired_redshift))

        if abs(closest_redshift - desired_redshift) > dz_tol:
            msg = (
                "No raw %s halo catalog has \na redshift within %.2f " + 
                "of the desired_redshift = %.2f.\n The closest redshift for these catalogs is %.2f"
                )
            print(msg % (kwargs['simname'], dz_tol, kwargs['desired_redshift'], closest_redshift))
            return 

        if 'external_cache_loc' in kwargs.keys():
            external_cache_loc = kwargs['external_cache_loc']
            # We were given an explicit path to store the catalog
            # Check that this path actually exists, and if so, use it 
            if os.path.exists(external_cache_loc):
                output_fname = os.path.join(external_cache_loc, os.path.basename(url))
            else:
                raise IOError("Input directory name %s for download location"
                    "of raw halo catalog does not exist" % external_cache_loc)
                
        else:
            # We were not given an explicit path, so use the default Halotools cache dir
            cache_dirname = cache_config.get_catalogs_dir(catalog_type='raw_halos', **kwargs)
            output_fname = os.path.join(cache_dirname, os.path.basename(url))

        if overwrite == False:
            file_pattern = os.path.basename(url)
            # The file may already be decompressed, in which case we don't want to download it again
            file_pattern = re.sub('.tar.gz', '', file_pattern)
            file_pattern = re.sub('.gz', '', file_pattern)
            file_pattern = '*' + file_pattern + '*'

            for path, dirlist, filelist in os.walk(cache_dirname):
                 for fname in filelist:
                    if fnmatch.filter([fname], file_pattern) != []:
                        existing_fname = os.path.join(path, fname)
                        msg = ("The following filename already exists in your cache directory: \n\n%s\n\n"
                            "If you really want to overwrite the file, \n"
                            "you must call the same function again \n"
                            "with the keyword argument `overwrite` set to `True`")
                        print(msg % existing_fname)
                        return None

        start = time()
        download_file_from_url(url, output_fname)
        end = time()
        runtime = (end - start)
        print("\nTotal runtime to download snapshot = %.1f seconds\n" % runtime)
        return output_fname


    def download_processed_halocat(self, dz_tol = 0.1, overwrite=False, **kwargs):
        """ Method to download one of the pre-processed binary files 
        storing a reduced halo catalog.  

        Parameters 
        ----------
        simname : string 
            Nickname of the simulation. Currently supported simulations are 
            Bolshoi  (simname = ``bolshoi``), Consuelo (simname = ``consuelo``), 
            MultiDark (simname = ``multidark``), and Bolshoi-Planck (simname = ``bolplanck``). 
            
        halo_finder : string 
            Nickname of the halo-finder, e.g. `rockstar`. 

        desired_redshift : float 
            Redshift of the requested snapshot. Must match one of the 
            available snapshots, or a prompt will be issued providing the nearest 
            available snapshots to choose from. 

        dz_tol : float, optional
            Tolerance value determining how close the requested redshift must be to 
            some available snapshot before issuing a warning. Default value is 0.1. 

        overwrite : boolean, optional
            If a file with the same filename already exists 
            in the requested download location, the `overwrite` boolean determines 
            whether or not to overwrite the file. Default is False, in which case 
            no download will occur if a pre-existing file is detected. 

        external_cache_loc : string, optional 
            Absolute path to an alternative source of raw halo catalogs. 

        Returns 
        -------
        output_fname : string  
            Filename (including absolute path) of the location of the downloaded 
            halo catalog.  
        """
        if 'simname' in kwargs.keys():
            if cache_config.simname_is_supported(kwargs['simname']) is False:
                raise KeyError(unsupported_simname_msg % kwargs['simname'])

        desired_redshift = kwargs['desired_redshift']

        available_fnames_to_download = self.processed_halocats_available_for_download(**kwargs)

        url, closest_redshift = (
            self._closest_fname(available_fnames_to_download, desired_redshift))

        if abs(closest_redshift - desired_redshift) > dz_tol:
            msg = (
                "No raw %s halo catalog has \na redshift within %.2f " + 
                "of the desired_redshift = %.2f.\n The closest redshift for these catalogs is %.2f"
                )
            print(msg % (kwargs['simname'], dz_tol, kwargs['desired_redshift'], closest_redshift))
            return 

        if 'external_cache_loc' in kwargs.keys():
            external_cache_loc = kwargs['external_cache_loc']
            # We were given an explicit path to store the catalog
            # Check that this path actually exists, and if so, use it 
            if os.path.exists(external_cache_loc):
                output_fname = os.path.join(external_cache_loc, os.path.basename(url))
            else:
                raise IOError("Input directory name %s for download location"
                    "of raw halo catalog does not exist" % external_cache_loc)
        else:
            # We were not given an explicit path, so use the default Halotools cache dir
            cache_dirname = cache_config.get_catalogs_dir(catalog_type='halos', **kwargs)
            output_fname = os.path.join(cache_dirname, os.path.basename(url))

        if overwrite == False:
            file_pattern = os.path.basename(url)
            # The file may already be decompressed, in which case we don't want to download it again
            file_pattern = re.sub('.tar.gz', '', file_pattern)
            file_pattern = re.sub('.gz', '', file_pattern)
            file_pattern = '*' + file_pattern + '*'

            for path, dirlist, filelist in os.walk(cache_dirname):
                 for fname in filelist:
                    if fnmatch.filter([fname], file_pattern) != []:
                        existing_fname = os.path.join(path, fname)
                        msg = ("The following filename already exists in your cache directory: \n\n%s\n\n"
                            "If you really want to overwrite the file, \n"
                            "you must call the same function again \n"
                            "with the keyword argument `overwrite` set to `True`")
                        print(msg % existing_fname)
                        return None

        start = time()
        download_file_from_url(url, output_fname)
        end = time()
        runtime = (end - start)
        print("\nTotal runtime to download snapshot = %.1f seconds\n" % runtime)
        return output_fname


    def download_ptcl_cat(self, dz_tol = 0.1, overwrite=False, **kwargs):
        """ Method to download one of the pre-processed binary files 
        storing a reduced halo catalog.  

        Parameters 
        ----------
        simname : string 
            Nickname of the simulation. Currently supported simulations are 
            Bolshoi  (simname = ``bolshoi``), Consuelo (simname = ``consuelo``), 
            MultiDark (simname = ``multidark``), and Bolshoi-Planck (simname = ``bolplanck``). 
            
        halo_finder : string 
            Nickname of the halo-finder, e.g. `rockstar`. 

        desired_redshift : float 
            Redshift of the requested snapshot. Must match one of the 
            available snapshots, or a prompt will be issued providing the nearest 
            available snapshots to choose from. 

        dz_tol : float, optional
            Tolerance value determining how close the requested redshift must be to 
            some available snapshot before issuing a warning. Default value is 0.1. 

        overwrite : boolean, optional
            If a file with the same filename already exists 
            in the requested download location, the `overwrite` boolean determines 
            whether or not to overwrite the file. Default is False, in which case 
            no download will occur if a pre-existing file is detected. 

        external_cache_loc : string, optional 
            Absolute path to an alternative source of raw halo catalogs. 

        Returns 
        -------
        output_fname : string  
            Filename (including absolute path) of the location of the downloaded 
            halo catalog.  
        """
        if 'simname' in kwargs.keys():
            if cache_config.simname_is_supported(kwargs['simname']) is False:
                raise KeyError(unsupported_simname_msg % kwargs['simname'])

        desired_redshift = kwargs['desired_redshift']
        if 'halo_finder' in kwargs.keys():
            warn("It is not necessary to specify a halo catalog when downloading particles")

        available_fnames_to_download = self.ptcl_cats_available_for_download(**kwargs)

        url, closest_redshift = (
            self._closest_fname(available_fnames_to_download, desired_redshift))

        if abs(closest_redshift - desired_redshift) > dz_tol:
            msg = (
                "No raw %s halo catalog has \na redshift within %.2f " + 
                "of the desired_redshift = %.2f.\n The closest redshift for these catalogs is %.2f"
                )
            print(msg % (kwargs['simname'], dz_tol, kwargs['desired_redshift'], closest_redshift))
            return 

        if 'external_cache_loc' in kwargs.keys():
            external_cache_loc = kwargs['external_cache_loc']
            # We were given an explicit path to store the catalog
            # Check that this path actually exists, and if so, use it 
            if os.path.exists(external_cache_loc):
                output_fname = os.path.join(external_cache_loc, os.path.basename(url))
            else:
                raise IOError("Input directory name %s for download location"
                    "of raw halo catalog does not exist" % external_cache_loc)
        else:
            # We were not given an explicit path, so use the default Halotools cache dir
            cache_dirname = cache_config.get_catalogs_dir(catalog_type='particles', **kwargs)
            output_fname = os.path.join(cache_dirname, os.path.basename(url))

        if overwrite == False:
            file_pattern = os.path.basename(url)
            # The file may already be decompressed, in which case we don't want to download it again
            file_pattern = re.sub('.tar.gz', '', file_pattern)
            file_pattern = re.sub('.gz', '', file_pattern)
            file_pattern = '*' + file_pattern + '*'

            for path, dirlist, filelist in os.walk(cache_dirname):
                 for fname in filelist:
                    if fnmatch.filter([fname], file_pattern) != []:
                        existing_fname = os.path.join(path, fname)
                        msg = ("The following filename already exists in your cache directory: \n\n%s\n\n"
                            "If you really want to overwrite the file, \n"
                            "you must call the same function again \n"
                            "with the keyword argument `overwrite` set to `True`")
                        print(msg % existing_fname)
                        return None

        start = time()
        #download_file_from_url(url, output_fname)
        end = time()
        runtime = (end - start)
        print("\nTotal runtime to download snapshot = %.1f seconds\n" % runtime)
        return output_fname

    def retrieve_ptcl_cat_from_cache(self, **kwargs):
        pass

    def retrieve_processed_halocat_from_cache(self, **kwargs):
        pass

    def retrieve_raw_halocat_from_cache(self, **kwargs):
        pass

    def store_newly_processed_halocat(self, **kwargs):
        pass



class HaloCatalogProcessor(object):
    """ Class used to read halo catalog ASCII data, 
    produce a value-added halo catalog, and store the catalog  
    in the cache directory or other desired location. 
    """

    def __init__(self):
        pass

    def read_raw_halocat_ascii(self, **kwargs):
        pass




