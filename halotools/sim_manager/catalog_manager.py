# -*- coding: utf-8 -*-
"""
Methods and classes for halo catalog I/O and organization.

"""

import numpy as np
from warnings import warn

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


from . import supported_sims, cache_config, sim_defaults
import os, fnmatch
from functools import partial

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
            Nickname of the simulation, e.g. ``bolshoi``. 
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
            Nickname of the simulation, e.g. ``bolshoi``. 
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
            Nickname of the simulation, e.g. ``bolshoi``. 
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
            Nickname of the simulation, e.g. ``bolshoi``. 
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
            Nickname of the simulation, e.g. ``bolshoi``. 
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

    def _orig_halocat_web_location(self, simname, halo_finder):
        """
        Parameters 
        ----------
        simname : string, optional
            Nickname of the simulation, e.g. ``bolshoi``. 

        halo_finder : string, optional
            Nickname of the halo-finder, e.g. ``rockstar`` or ``bdm``.

        Returns 
        -------
        webloc : string 
            Web location from which the original halo catalogs were downloaded.  
        """
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

    def raw_halocats_available_for_download(self, simname, halo_finder):
        """ Method searches the appropriate web location and 
        returns a list of the filenames of all relevant 
        raw halo catalogs that are available for download. 

        Parameters 
        ----------
        simname : string
            Nickname of the simulation, e.g. ``bolshoi``. 
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
        url = self._orig_halocat_web_location(simname, halo_finder)

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
            Nickname of the simulation, e.g. ``bolshoi``. 
            Argument is used to filter the output list of filenames. 
            Default is None, in which case `processed_halocats_in_cache` 
            will not filter the returned list of filenames by ``simname``. 

        Returns 
        -------
        output : list 
            List of web locations of all catalogs of downsampled particles
            matching the input arguments. 

        """
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

    def _closest_fname(self, filename_list, redshift):

        if redshift == 0.:
            input_scale_factor == 1.
        else:
            input_scale_factor = (1./redshift) - 1

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
        input_scale_factor = 1./(1. + input_redshift)
        idx_closest_catalog = find_idx_nearest_val(
            scale_factor_list, input_scale_factor)
        closest_scale_factor = scale_factor_list[idx_closest_catalog]
        output_fname = filename_list[idx_closest_catalog]

        closest_available_redshift = (1./closest_scale_factor) - 1

        return output_fname, closest_available_redshift

    def closest_catalog_in_cache(self, version_name = sim_defaults.default_version_name, 
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
            Nickname of the simulation, e.g. ``bolshoi``. 

        halo_finder : string, optional
            Nickname of the halo-finder, e.g. ``rockstar``. 
            Required when input ``catalog_type`` is ``halos`` or ``raw_halos``. 

        version_name : string, optional 
            String specifying the version of the processed halo catalog. 
            Argument is used to filter the output list of filenames. 
            Default is set by ``~halotools.sim_manager.sim_defaults.default_version_name``. 

        external_cache_loc : string, optional 
            Absolute path to an alternative source of halo catalogs. 

        Returns
        -------
        output_fname : list 
            String of the filename with the closest matching redshift. 

        redshift : float 
            Value of the redshift of the snapshot
        """

        simname = kwargs['simname']
        input_redshift = kwargs['desired_redshift']

        # Verify arguments are as needed
        catalog_type = kwargs['catalog_type']
        del kwargs['catalog_type']
        if catalog_type is not 'particles':
            try:
                halo_finder = kwargs['halo_finder']
            except KeyError:
                raise("If input catalog_type is not particles, must pass halo_finder argument")
        else:
            if 'halo_finder' in kwargs.keys():
                warn("There is no need to specify a halo-finder when requesting particle data")
                del kwargs['halo_finder']
                
        filename_list = self._scrape_cache(
            catalog_type = catalog_type, **kwargs)

        if custom_len(filename_list) == 0:
            print("\nNo matching catalogs found by closest_catalog_in_cache method of CatalogManager\n")
            return None

        output_fname, redshift = self._closest_fname(filename_list, input_redshift)

        return output_fname, redshift


    def download_raw_halocat(self, **kwargs):
        """ Method to download one of the pre-processed binary files 
        storing a reduced halo catalog.  

        Parameters 
        ----------
        simname : string 
            Nickname of the simulation, e.g. `bolshoi`. 

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

        desired_redshift = kwargs['desired_redshift']
        simname = kwargs['simname']
        halo_finder = kwargs['halo_finder']

        if 'dz_tol' in kwargs.keys():
            dz_tol = kwargs['dz_tol']
        else:
            dz_tol = 0.1

        if 'overwrite' in kwargs.keys():
            overwrite = kwargs['overwrite']
        else:
            overwrite = False

        available_fnames_to_download = self.raw_halocats_available_for_download(
            simname=simname, halo_finder = halo_finder)

        closest_name, closest_redshift = (
            self._closest_fname(available_fnames_to_download, desired_redshift))

        if abs(closest_redshift - desired_redshift) > dz_tol:
            msg = (
                "No raw %s halo catalog has \na redshift within %.2f " + 
                "of the desired_redshift = %.2f.\n The closest redshift for these catalogs is %.2f"
                )
            print(msg % (simname, dz_tol, desired_redshift, closest_redshift))
            return 

        webloc = self._orig_halocat_web_location(simname = simname, halo_finder=halo_finder)
        url = os.path.join(webloc, closest_name)

        if 'external_cache_loc' in kwargs.keys():
            external_cache_loc = kwargs['external_cache_loc']
            # We were given an explicit path to store the catalog
            # Check that this path actually exists, and if so, use it 
            if not os.path.exists(external_cache_loc):
                raise IOError("Input directory name %s for download location"
                    "of raw halo catalog does not exist" % external_cache_loc)
            else:
                output_fname = os.path.join(external_cache_loc, closest_snapshot_fname)
        else:
            # We were not given an explicit path, so use the default Halotools cache dir
            cache_dirname = cache_config.get_catalogs_dir(catalog_type='raw_halos', 
                simname=simname, halo_finder=halo_finder)
            output_fname = os.path.join(cache_dirname, closest_snapshot_fname)

        # Check whether there are existing catalogs matching the file pattern 
        # that is about to be downloaded
        ### LEFT OFF HERE 



    def download_processed_halocat(self, **kwargs):
        pass

    def download_ptcl_cat(self, **kwargs):
        pass

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




