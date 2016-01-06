# -*- coding: utf-8 -*-
"""
Methods and classes for halo catalog I/O and organization.

"""

__all__ = ['CatalogManager']

import numpy as np
from warnings import warn
from time import time
from astropy.tests.helper import remote_data
from astropy.table import Table

from ..custom_exceptions import HalotoolsError

try:
    from bs4 import BeautifulSoup
except ImportError:
    raise HalotoolsError("Must have bs4 package installed to use the catalog_manager module")

try:
    import requests
except ImportError:
    raise HalotoolsError("Must have requests package installed to use the catalog_manager module")
import posixpath
import urlparse
import datetime

import os, fnmatch, re
from functools import partial

from ..custom_exceptions import *

try:
    import h5py
except ImportError:
    warn("Most of the functionality of the catalog_manager module requires h5py to be installed,\n"
        "which can be accomplished either with pip or conda")

from . import cache_config, sim_defaults

from ..utils.array_utils import find_idx_nearest_val
from ..utils.array_utils import custom_len, convert_to_ndarray
from ..utils.io_utils import download_file_from_url



unsupported_simname_msg = "Input simname ``%s`` is not recognized by Halotools"

class CatalogManager(object):
    """ Class used to scrape the web for simulation data
    and manage the set of cached catalogs.
    """

    def __init__(self):
        pass

    def _scrape_cache(self, catalog_type, **kwargs):
        """ Private method that is the workhorse behind
        `processed_halo_tables_in_cache`, `raw_halo_tables_in_cache`, and `ptcl_tables_in_cache`.

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
            Default is None, in which case `processed_halo_tables_in_cache`
            will not filter the returned list of filenames by ``simname``.

        halo_finder : string, optional
            Nickname of the halo-finder, e.g. ``rockstar``.
            Argument is used to filter the output list of filenames.
            Default is None, in which case `processed_halo_tables_in_cache`
            will not filter the returned list of filenames by ``halo_finder``.

        version_name : string, optional
            String specifying the version of the processed halo catalog.
            Argument is used to filter the output list of filenames.
            Default is None, in which case `processed_halo_tables_in_cache`
            will not filter the returned list of filenames by ``version_name``.

        external_cache_loc : string, optional
            Absolute path to an alternative source of halo catalogs.
            Method assumes that ``external_cache_loc`` is organized in the
            same way that the normal Halotools cache is. Specifically:

            * Particle tables should located in ``external_cache_loc/particle_catalogs/simname``

            * Processed halo tables should located in ``external_cache_loc/halo_catalogs/simname/halo_finder``

            * Raw halo tables (unprocessed ASCII) should located in ``external_cache_loc/raw_halo_catalogs/simname/halo_finder``

        Returns
        -------
        fname_list : list
            List of strings of the filenames (including absolute path) of
            processed halo stored in the cache directory, filtered according
            to the input arguments.
        """
        if 'simname' in kwargs.keys():
            if cache_config.simname_is_supported(kwargs['simname']) is False:
                raise UnsupportedSimError(kwargs['simname'])

        if ('version_name' in kwargs.keys()) & (catalog_type is not 'halos'):
            raise KeyError("The _scrape_cache method received a version_name = %s keyword "
                "argument, which should not be passed for catalog_type = %s" % (kwargs['version_name'], catalog_type))

        cachedir = cache_config.get_catalogs_dir(catalog_type = catalog_type, **kwargs)

        if catalog_type == 'raw_halos':
            fname_pattern = '.list*'
        else:
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

    def processed_halo_tables_in_cache(self, **kwargs):
        """
        Parameters
        ----------
        simname : string, optional
            Nickname of the simulation. Currently supported simulations are
            Bolshoi  (simname = ``bolshoi``), Consuelo (simname = ``consuelo``),
            MultiDark (simname = ``multidark``), and Bolshoi-Planck (simname = ``bolplanck``).

            Argument is used to filter the output list of filenames.
            Default is None, in which case `processed_halo_tables_in_cache`
            will not filter the returned list of filenames by ``simname``.

        halo_finder : string, optional
            Nickname of the halo-finder, e.g. ``rockstar``.
            Argument is used to filter the output list of filenames.
            Default is None, in which case `processed_halo_tables_in_cache`
            will not filter the returned list of filenames by ``halo_finder``.

        version_name : string, optional
            String specifying the version of the processed halo catalog.
            Argument is used to filter the output list of filenames.
            Default is None, in which case `processed_halo_tables_in_cache`
            will not filter the returned list of filenames by ``version_name``.

        external_cache_loc : string, optional
            Absolute path to an alternative source of halo catalogs.
            Method assumes that ``external_cache_loc`` is organized in the
            same way that the normal Halotools cache is. Specifically:

            * Particle tables should located in ``external_cache_loc/particle_catalogs/simname``

            * Processed halo tables should located in ``external_cache_loc/halo_catalogs/simname/halo_finder``

            * Raw halo tables (unprocessed ASCII) should located in ``external_cache_loc/raw_halo_catalogs/simname/halo_finder``

        Returns
        -------
        fname_list : list
            List of strings of the filenames (including absolute path) of
            processed halo catalogs in the cache directory.
            Filenames are filtered according to the input arguments.
        """

        f = partial(self._scrape_cache, catalog_type='halos')
        return f(**kwargs)

    def raw_halo_tables_in_cache(self, **kwargs):
        """
        Parameters
        ----------
        simname : string, optional
            Nickname of the simulation. Currently supported simulations are
            Bolshoi  (simname = ``bolshoi``), Consuelo (simname = ``consuelo``),
            MultiDark (simname = ``multidark``), and Bolshoi-Planck (simname = ``bolplanck``).

            Argument is used to filter the output list of filenames.
            Default is None, in which case `processed_halo_tables_in_cache`
            will not filter the returned list of filenames by ``simname``.

        halo_finder : string, optional
            Nickname of the halo-finder, e.g. ``rockstar``.
            Argument is used to filter the output list of filenames.
            Default is None, in which case `processed_halo_tables_in_cache`
            will not filter the returned list of filenames by ``halo_finder``.

        external_cache_loc : string, optional
            Absolute path to an alternative source of halo catalogs.
            Method assumes that ``external_cache_loc`` is organized in the
            same way that the normal Halotools cache is. Specifically:

            * Particle tables should located in ``external_cache_loc/particle_catalogs/simname``

            * Processed halo tables should located in ``external_cache_loc/halo_catalogs/simname/halo_finder``

            * Raw halo tables (unprocessed ASCII) should located in ``external_cache_loc/raw_halo_catalogs/simname/halo_finder``

        Returns
        -------
        fname_list : list
            List of strings of the filenames (including absolute path) of
            all unprocessed halo catalog ASCII data tables in the cache directory.
            Filenames are filtered according to the input arguments.
        """
        f = partial(self._scrape_cache, catalog_type='raw_halos')
        return f(**kwargs)

    def ptcl_tables_in_cache(self, **kwargs):
        """
        Parameters
        ----------
        simname : string, optional
            Nickname of the simulation. Currently supported simulations are
            Bolshoi  (simname = ``bolshoi``), Consuelo (simname = ``consuelo``),
            MultiDark (simname = ``multidark``), and Bolshoi-Planck (simname = ``bolplanck``).

            Argument is used to filter the output list of filenames.
            Default is None, in which case `processed_halo_tables_in_cache`
            will not filter the returned list of filenames by ``simname``.

        external_cache_loc : string, optional
            Absolute path to an alternative source of halo catalogs.
            Method assumes that ``external_cache_loc`` is organized in the
            same way that the normal Halotools cache is. Specifically:

            * Particle tables should located in ``external_cache_loc/particle_catalogs/simname``

            * Processed halo tables should located in ``external_cache_loc/halo_catalogs/simname/halo_finder``

            * Raw halo tables (unprocessed ASCII) should located in ``external_cache_loc/raw_halo_catalogs/simname/halo_finder``

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

    def processed_halo_tables_available_for_download(self, **kwargs):
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
            Default is None, in which case `processed_halo_tables_in_cache`
            will not filter the returned list of filenames by ``simname``.

        halo_finder : string, optional
            Nickname of the halo-finder, e.g. ``rockstar``.
            Argument is used to filter the output list of filenames.
            Default is None, in which case `processed_halo_tables_in_cache`
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

        baseurl = sim_defaults.processed_halo_tables_webloc
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

    def _orig_halo_table_web_location(self, **kwargs):
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
            raise HalotoolsError("CatalogManager._orig_halo_table_web_location method "
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

    def raw_halo_tables_available_for_download(self, **kwargs):
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
            Default is None, in which case `processed_halo_tables_in_cache`
            will not filter the returned list of filenames by ``simname``.

        halo_finder : string
            Nickname of the halo-finder, e.g. ``rockstar``.
            Argument is used to filter the output list of filenames.
            Default is None, in which case `processed_halo_tables_in_cache`
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

        url = self._orig_halo_table_web_location(**kwargs)

        soup = BeautifulSoup(requests.get(url).text)
        file_list = []
        for a in soup.find_all('a'):
            file_list.append(os.path.join(url, a['href']))

        file_pattern = '*hlist_*'
        output = fnmatch.filter(file_list, file_pattern)

        return output

    def ptcl_tables_available_for_download(self, **kwargs):
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
            Default is None, in which case `processed_halo_tables_in_cache`
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

        baseurl = sim_defaults.ptcl_tables_webloc
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
        all_ptcl_tables = fnmatch.filter(catlist, '*'+file_pattern)

        if 'simname' in kwargs.keys():
            simname = kwargs['simname']
            file_pattern = '*'+simname+'/*' + file_pattern
            output = fnmatch.filter(all_ptcl_tables, file_pattern)
        else:
            output = all_ptcl_tables

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
        """
        """

        if custom_len(filename_list) == 0:
            msg = "The _closest_fname method was passed an empty filename_list"
            raise HalotoolsError(msg)

        if desired_redshift <= -1:
            raise ValueError("desired_redshift of <= -1 is unphysical")
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

    def closest_catalog_in_cache(self, catalog_type, desired_redshift, **kwargs):
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
            Absolute path to an alternative source of halo catalogs.
            Method assumes that ``external_cache_loc`` is organized in the
            same way that the normal Halotools cache is. Specifically:

            * Particle tables should located in ``external_cache_loc/particle_catalogs/simname``

            * Processed halo tables should located in ``external_cache_loc/halo_catalogs/simname/halo_finder``

            * Raw halo tables (unprocessed ASCII) should located in ``external_cache_loc/raw_halo_catalogs/simname/halo_finder``

        Returns
        -------
        output_fname : list
            String of the filename with the closest matching redshift.

        redshift : float
            Value of the redshift of the snapshot
        """
        if 'simname' in kwargs.keys():
            if cache_config.simname_is_supported(kwargs['simname']) is False:
                raise UnsupportedSimError(kwargs['simname'])

        # Verify arguments are as needed
        if catalog_type is not 'particles':
            try:
                halo_finder = kwargs['halo_finder']
            except KeyError:
                raise HalotoolsError("If input catalog_type is not particles, must pass halo_finder argument")
        else:
            if 'halo_finder' in kwargs.keys():
                warn("There is no need to specify a halo-finder when requesting particle data")
                del kwargs['halo_finder']

        if (catalog_type == 'halos') & ('version_name' not in kwargs.keys()):
            kwargs['version_name'] = sim_defaults.default_version_name
        filename_list = self._scrape_cache(catalog_type = catalog_type, **kwargs)

        if custom_len(filename_list) == 0:
            msg = "\nNo matching catalogs found by closest_catalog_in_cache method of CatalogManager\n"
            raise HalotoolsCacheError(msg)

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

        >>> webloc_closest_match = catman.closest_catalog_on_web(catalog_type='particles', simname='bolplanck', desired_redshift=0.5)  # doctest: +SKIP

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
            raise HalotoolsError("Must supply input simname keyword argument")

        catalog_type = kwargs['catalog_type']
        if catalog_type is not 'particles':
            try:
                halo_finder = kwargs['halo_finder']
            except KeyError:
                raise HalotoolsError("If input catalog_type is not particles, must pass halo_finder argument")
        else:
            if 'halo_finder' in kwargs.keys():
                warn("There is no need to specify a halo-finder when requesting particle data")
                del kwargs['halo_finder']

        if catalog_type is 'particles':
            filename_list = self.ptcl_tables_available_for_download(**kwargs)
        elif catalog_type is 'raw_halos':
            filename_list = self.raw_halo_tables_available_for_download(**kwargs)
        elif catalog_type is 'halos':
            filename_list = self.processed_halo_tables_available_for_download(**kwargs)

        desired_redshift = kwargs['desired_redshift']
        output_fname, redshift = self._closest_fname(filename_list, desired_redshift)

        return output_fname, redshift

    def download_raw_halo_table(self, dz_tol = 0.1, overwrite=False, **kwargs):
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
            Absolute path to an alternative source of halo catalogs.
            Method assumes that ``external_cache_loc`` is organized in the
            same way that the normal Halotools cache is. Specifically:

            * Particle tables should located in ``external_cache_loc/particle_catalogs/simname``

            * Processed halo tables should located in ``external_cache_loc/halo_catalogs/simname/halo_finder``

            * Raw halo tables (unprocessed ASCII) should located in ``external_cache_loc/raw_halo_catalogs/simname/halo_finder``

        Returns
        -------
        output_fname : string
            Filename (including absolute path) of the location of the downloaded
            halo catalog.
        """
        if 'simname' in kwargs.keys():
            if cache_config.simname_is_supported(kwargs['simname']) is False:
                raise UnsupportedSimError(kwargs['simname'])

        desired_redshift = kwargs['desired_redshift']

        available_fnames_to_download = self.raw_halo_tables_available_for_download(**kwargs)

        url, closest_redshift = self._closest_fname(available_fnames_to_download, desired_redshift)

        if abs(closest_redshift - desired_redshift) > dz_tol:
            msg = (
                "No raw %s halo catalog has \na redshift within %.2f " +
                "of the desired_redshift = %.2f.\n The closest redshift for these catalogs is %.2f\n"
                )
            print(msg % (kwargs['simname'], dz_tol, kwargs['desired_redshift'], closest_redshift))
            return

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
        print("\nTotal runtime to download raw halo catalog = %.1f seconds\n" % runtime)
        if 'success_msg' in kwargs.keys():
            print(kwargs['success_msg'])
        return output_fname


    def download_processed_halo_table(self, dz_tol = 0.1, overwrite=False, **kwargs):
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
            Absolute path to an alternative source of halo catalogs.
            Method assumes that ``external_cache_loc`` is organized in the
            same way that the normal Halotools cache is. Specifically:

            * Particle tables should located in ``external_cache_loc/particle_catalogs/simname``

            * Processed halo tables should located in ``external_cache_loc/halo_catalogs/simname/halo_finder``

            * Raw halo tables (unprocessed ASCII) should located in ``external_cache_loc/raw_halo_catalogs/simname/halo_finder``

        Returns
        -------
        output_fname : string
            Filename (including absolute path) of the location of the downloaded
            halo catalog.
        """
        if 'simname' in kwargs.keys():
            if cache_config.simname_is_supported(kwargs['simname']) is False:
                raise UnsupportedSimError(kwargs['simname'])

        desired_redshift = kwargs['desired_redshift']

        available_fnames_to_download = self.processed_halo_tables_available_for_download(**kwargs)
        if available_fnames_to_download == []:
            msg = "You made the following request for a pre-processed halo catalog:\n"
            if 'simname' in kwargs:
                msg = msg + "simname = " + kwargs['simname'] + "\n"
            else:
                msg = msg + "simname = any simulation\n"
            if 'halo_finder' in kwargs:
                msg = msg + "halo-finder = " + kwargs['halo_finder'] + "\n"
            else:
                msg = msg + "halo-finder = any halo-finder\n"
            msg = msg + "There are no halo catalogs meeting your specifications"
            raise UnsupportedSimError(msg)

        url, closest_redshift = (
            self._closest_fname(available_fnames_to_download, desired_redshift))

        if abs(closest_redshift - desired_redshift) > dz_tol:
            msg = (
                "No pre-processed %s halo catalog has \na redshift within %.2f " +
                "of the desired_redshift = %.2f.\n The closest redshift for these catalogs is %.2f\n"
                )
            print(msg % (kwargs['simname'], dz_tol, kwargs['desired_redshift'], closest_redshift))
            return

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
                        if 'initial_download_script_msg' in kwargs.keys():
                            msg = kwargs['initial_download_script_msg']
                        else:
                            msg = ("The following filename already exists in your cache directory: \n\n%s\n\n"
                                "If you really want to overwrite the file, \n"
                                "you must call the same function again \n"
                                "with the keyword argument `overwrite` set to `True`")
                        raise HalotoolsCacheError(msg % existing_fname)

        start = time()
        download_file_from_url(url, output_fname)
        end = time()
        runtime = (end - start)
        print("\nTotal runtime to download pre-processed halo catalog = %.1f seconds\n" % runtime)
        if 'success_msg' in kwargs.keys():
            print(kwargs['success_msg'])
        return output_fname


    def download_ptcl_table(self, dz_tol = 0.1, overwrite=False, **kwargs):
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
            Absolute path to an alternative source of halo catalogs.
            Method assumes that ``external_cache_loc`` is organized in the
            same way that the normal Halotools cache is. Specifically:

            * Particle tables should located in ``external_cache_loc/particle_catalogs/simname``

            * Processed halo tables should located in ``external_cache_loc/halo_catalogs/simname/halo_finder``

            * Raw halo tables (unprocessed ASCII) should located in ``external_cache_loc/raw_halo_catalogs/simname/halo_finder``

        Returns
        -------
        output_fname : string
            Filename (including absolute path) of the location of the downloaded
            halo catalog.
        """
        if 'simname' in kwargs.keys():
            if cache_config.simname_is_supported(kwargs['simname']) is False:
                raise UnsupportedSimError(kwargs['simname'])

        desired_redshift = kwargs['desired_redshift']
        if 'halo_finder' in kwargs.keys():
            warn("It is not necessary to specify a halo catalog when downloading particles")

        available_fnames_to_download = self.ptcl_tables_available_for_download(**kwargs)

        url, closest_redshift = (
            self._closest_fname(available_fnames_to_download, desired_redshift))
        if available_fnames_to_download == []:
            msg = "You made the following request for a pre-processed halo catalog:\n"
            if 'simname' in kwargs:
                msg = msg + "simname = " + kwargs['simname'] + "\n"
            else:
                msg = msg + "simname = any simulation\n"
            msg = msg + "There are no simulations with this name with particles available for download"
            raise UnsupportedSimError(msg)

        if abs(closest_redshift - desired_redshift) > dz_tol:
            msg = (
                "No %s particle catalog has \na redshift within %.2f " +
                "of the desired_redshift = %.2f.\n The closest redshift for these catalogs is %.2f\n"
                )
            print(msg % (kwargs['simname'], dz_tol, kwargs['desired_redshift'], closest_redshift))
            return

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
                        if 'initial_download_script_msg' in kwargs.keys():
                            msg = kwargs['initial_download_script_msg']
                        else:
                            msg = ("The following filename already exists in your cache directory: \n\n%s\n\n"
                                "If you really want to overwrite the file, \n"
                                "you must call the same function again \n"
                                "with the keyword argument `overwrite` set to `True`")
                        raise HalotoolsCacheError(msg % existing_fname)

        start = time()
        download_file_from_url(url, output_fname)
        end = time()
        runtime = (end - start)
        print("\nTotal runtime to download particle data = %.1f seconds\n" % runtime)
        if 'success_msg' in kwargs.keys():
            print(kwargs['success_msg'])
        return output_fname

    def retrieve_ptcl_table_from_cache(self, simname, desired_redshift, **kwargs):
        """
        Parameters
        ----------
        desired_redshift : float
            Redshift of the desired catalog.

        simname : string
            Nickname of the simulation. Currently supported simulations are
            Bolshoi  (simname = ``bolshoi``), Consuelo (simname = ``consuelo``),
            MultiDark (simname = ``multidark``), and Bolshoi-Planck (simname = ``bolplanck``).

        external_cache_loc : string, optional
            Absolute path to an alternative source of halo catalogs.
            Method assumes that ``external_cache_loc`` is organized in the
            same way that the normal Halotools cache is. Specifically:

            * Particle tables should located in ``external_cache_loc/particle_catalogs/simname``

            * Processed halo tables should located in ``external_cache_loc/halo_catalogs/simname/halo_finder``

            * Raw halo tables (unprocessed ASCII) should located in ``external_cache_loc/raw_halo_catalogs/simname/halo_finder``

        Returns
        -------
        particles : Astropy Table
            `~astropy.table.Table` object storing position and velocity of particles.
        """
        try:
            import h5py
        except ImportError:
            raise HalotoolsError("Must have h5py package installed to use this feature")

        fname, z = self.closest_catalog_in_cache(simname = simname,
            catalog_type='particles', desired_redshift=desired_redshift, **kwargs)

        if abs(z-desired_redshift) > 0.01:
            print("\nClosest matching table of particles in cache is for z = %.3f" % z)
            print("Loading table with the following filename:\n %s" % fname)
            print("\nIf your science application requires particles an exact snapshot, "
                "be sure to double-check that this filename is as expected\n")

        return Table.read(fname, path='data')

    def retrieve_processed_halo_table_from_cache(self, **kwargs):
        pass

    def store_newly_processed_halo_table(self, halo_table, reader, version_name, **kwargs):
        """
        Parameters
        -----------
        halo_table : table
            `~astropy.table.Table` object storing the halo catalog

        reader : object
            `~halotools.sim_manager.BehrooziASCIIReader` object used to read the ascii data
            and produce the input ``halo_table``

        version_name : string
            String will be appended to the original hlist name when storing the hdf5 file.

        external_cache_loc : string, optional
            Absolute path to an alternative source of halo catalogs.
            Method assumes that ``external_cache_loc`` is organized in the
            same way that the normal Halotools cache is. Specifically:

            * Particle tables should located in ``external_cache_loc/particle_catalogs/simname``

            * Processed halo tables should located in ``external_cache_loc/halo_catalogs/simname/halo_finder``

            * Raw halo tables (unprocessed ASCII) should located in ``external_cache_loc/raw_halo_catalogs/simname/halo_finder``

        overwrite : boolean, optional
            Determines whether we will overwrite an existing file of the same name, if present

        notes : dict, optional
            Additional notes that will be appended to the stored hdf5 file as metadata.
            Each dict key of `notes` will be a metadata attribute of the hdf5 file, accessible
            via hdf5_fileobj.attrs[key]. The value attached to each key can be any string.

        Returns
        -------
        output_fname : string
            Filename (including absolute path) where the input ``halo_table`` will be stored.
        """
        output_dir = cache_config.get_catalogs_dir(catalog_type = 'halos',
            simname = reader.halocat.simname,
            halo_finder = reader.halocat.halo_finder, **kwargs)

        basename = os.path.basename(reader.fname) + '.' + version_name + '.hdf5'
        output_fname = os.path.join(output_dir, basename)


        ### notes to attach to output hdf5 as metadata ###
        if 'notes' in kwargs.keys():
            notes = kwargs['notes']
        else:
            notes = {}
        for key, value in notes.iteritems():
            if type(value) != str:
                raise HalotoolsIOError("Strings are the only permissible data types of values "
                    "attached to keys in the input notes dictionary")

        if 'overwrite' in kwargs.keys():
            overwrite = kwargs['overwrite']
        else:
            overwrite = False
        halo_table.write(output_fname, path='data', overwrite = overwrite, append = overwrite)

        ### Add metadata to the hdf5 file
        try:
            import h5py
        except ImportError:
            raise HalotoolsError("Must have h5py installed to use the "
                "store_newly_processed_halo_table method")
        f = h5py.File(output_fname)

        time_right_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.attrs['time_of_original_reduction'] = time_right_now

        f.attrs['original_data_source'] = self._orig_halo_table_web_location(
            simname=reader.halocat.simname,
            halo_finder=reader.halocat.halo_finder)

        f.attrs['simname'] = reader.halocat.simname
        f.attrs['halo_finder'] = reader.halocat.halo_finder
        f.attrs['redshift'] = str(reader.halocat.redshift)

        f.attrs['Lbox'] = str(reader.halocat.Lbox) + ' Mpc in h=1 units'
        f.attrs['particle_mass'] = str(reader.halocat.particle_mass) + ' Msun in h=1 units'
        f.attrs['softening_length'] = str(reader.halocat.softening_length) + ' kpc in h=1 units'

        f.attrs['cuts_description'] = reader._cuts_description

        for note_key, note in notes.iteritems():
            f.attrs[note_key] = note

        f.close()

        return output_fname





















