# -*- coding: utf-8 -*-
"""
Methods and classes for halo catalog I/O and organization.

"""

__all__ = ['DownloadManager']

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

from . import manipulate_cache_log

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

supported_sim_list = ('bolshoi', 'bolplanck', 'consuelo', 'multidark')

unsupported_simname_msg = "There are no web locations recognized by Halotools \n for simname ``%s``"

class DownloadManager(object):
    """ Class used to scrape the web for simulation data
    and manage the set of cached catalogs.
    """

    def __init__(self):
        pass

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

        version_name : string, optional 
            Version of the table. Default is set by `~halotools.sim_manager.sim_defaults` module. 

        Returns
        -------
        output : list
            List of web locations of all pre-processed halo catalogs
            matching the input arguments.

        """
        try:
            simname = kwargs['simname']
            if simname not in supported_sim_list:
                raise HalotoolsError(unsupported_simname_msg % simname)
        except KeyError:
            pass

        try:
            version_name = kwargs['version_name']
        except KeyError:
            version_name = sim_defaults.default_version_name

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

        file_pattern = version_name + '.hdf5'
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
            raise HalotoolsError("\nDownloadManager._orig_halo_table_web_location method "
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
            raise HalotoolsError("Input simname %s and halo_finder %s do not "
                "have Halotools-recognized web locations" % (simname, halo_finder))

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
        try:
            simname = kwargs['simname']
            if simname not in supported_sim_list:
                raise HalotoolsError(unsupported_simname_msg % simname)
        except KeyError:
            pass

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

    def closest_catalog_on_web(self, **kwargs):
        """
        Parameters
        ----------
        desired_redshift : float
            Redshift of the desired catalog.

        catalog_type : string
            Specifies which subdirectory of the Halotools cache to scrape for .hdf5 files.
            Must be either ``halos`` or ``particles``

        simname : string
            Nickname of the simulation. Currently supported simulations are
            Bolshoi  (simname = ``bolshoi``), Consuelo (simname = ``consuelo``),
            MultiDark (simname = ``multidark``), and Bolshoi-Planck (simname = ``bolplanck``).

        halo_finder : string, optional
            Nickname of the halo-finder, e.g. ``rockstar``.
            Required when input ``catalog_type`` is ``halos``.

        version_name : string, optional
            String specifying the version of the processed halo catalog.
            Argument is used to filter the output list of filenames.
            Default is set by `~halotools.sim_manager.sim_defaults` module.

        Returns
        -------
        output_fname : list
            String of the filename with the closest matching redshift.

        redshift : float
            Value of the redshift of the snapshot

        Examples
        --------
        >>> catman = DownloadManager()

        Suppose you would like to download a pre-processed halo catalog for the Bolshoi-Planck simulation for z=0.5.
        To identify the filename of the available catalog that most closely matches your needs:

        >>> webloc_closest_match = catman.closest_catalog_on_web(catalog_type='halos', simname='bolplanck', halo_finder='rockstar', desired_redshift=0.5)  # doctest: +REMOTE_DATA

        You may also wish to have a collection of downsampled dark matter particles to accompany this snapshot:

        >>> webloc_closest_match = catman.closest_catalog_on_web(catalog_type='particles', simname='bolplanck', desired_redshift=0.5)  # doctest: +REMOTE_DATA

        """
        try:
            simname = kwargs['simname']
            if simname not in supported_sim_list:
                raise HalotoolsError(unsupported_simname_msg % simname)
        except KeyError:
            pass

        if 'redshift' in kwargs.keys():
            msg = ("\nThe correct argument to use to specify the redshift \n"
                "you are searching for is with the ``desired_redshift`` keyword, \n"
                "not the ``redshift`` keyword.\n")
            raise HalotoolsError(msg)

        if 'version_name' not in kwargs.keys():
            kwargs['version_name'] = sim_defaults.default_version_name

        # Verify arguments are as needed
        try:
            simname = kwargs['simname']
        except KeyError:
            raise HalotoolsError("\nMust supply input simname keyword argument")

        catalog_type = kwargs['catalog_type']
        if catalog_type is not 'particles':
            try:
                halo_finder = kwargs['halo_finder']
            except KeyError:
                raise HalotoolsError("\nIf input catalog_type is not particles, must pass halo_finder argument")
        else:
            if 'halo_finder' in kwargs.keys():
                warn("There is no need to specify a halo-finder when requesting particle data")
                del kwargs['halo_finder']

        if catalog_type is 'particles':
            filename_list = self.ptcl_tables_available_for_download(**kwargs)
        elif catalog_type is 'halos':
            filename_list = self.processed_halo_tables_available_for_download(**kwargs)

        desired_redshift = kwargs['desired_redshift']
        output_fname, redshift = self._closest_fname(filename_list, desired_redshift)

        return output_fname, redshift


    def download_processed_halo_table(self, simname, halo_finder, desired_redshift, 
        dz_tol = 0.1, overwrite=False, version_name = sim_defaults.default_version_name, 
        download_dirname = 'std_cache_loc'):
        """ Method to download one of the pre-processed binary files
        storing a reduced halo catalog.

        Parameters
        ----------
        simname : string
            Nickname of the simulation. Currently supported simulations are
            Bolshoi  (simname = ``bolshoi``), Consuelo (simname = ``consuelo``),
            MultiDark (simname = ``multidark``), and Bolshoi-Planck (simname = ``bolplanck``).

        halo_finder : string
            Nickname of the halo-finder, e.g. `rockstar` or `bdm`. 

        desired_redshift : float
            Redshift of the requested snapshot. Must match one of the
            available snapshots, or a prompt will be issued providing the nearest
            available snapshots to choose from.

        download_dirname : str, optional 
            Absolute path to the directory where you want to download the catalog. 
            Default is `std_cache_loc`, which will store the catalog in the following directory:
            ``$HOME/.astropy/cache/halotools/halo_tables/simname/halo_finder/``

        dz_tol : float, optional
            Tolerance value determining how close the requested redshift must be to
            some available snapshot before issuing a warning. Default value is 0.1.

        overwrite : boolean, optional
            If a file with the same filename already exists
            in the requested download location, the `overwrite` boolean determines
            whether or not to overwrite the file. Default is False, in which case
            no download will occur if a pre-existing file is detected.

        Returns
        -------
        output_fname : string
            Filename (including absolute path) of the location of the downloaded
            halo catalog.
        """

        ###################################
        # Search for a file that matches the input specifications
        available_fnames_to_download = (
            self.processed_halo_tables_available_for_download(simname = simname, 
                halo_finder = halo_finder, version_name = version_name)
            )

        if available_fnames_to_download == []:
            msg = "You made the following request for a pre-processed halo catalog:\n"

            msg += "simname = " + simname + "\n"
            msg += "halo_finder = " + halo_finder + "\n"
            msg += "version_name = " + version_name + "\n"
            msg = msg + "There are no halo catalogs meeting your specifications"
            raise HalotoolsError(msg)

        url, closest_redshift = (
            self._closest_fname(available_fnames_to_download, desired_redshift))

        if abs(closest_redshift - desired_redshift) > dz_tol:
            msg = (
                "\nNo pre-processed %s halo catalog has \na redshift within %.2f " +
                "of the desired_redshift = %.2f.\n The closest redshift for these catalogs is %.2f\n"
                )
            raise HalotoolsError(msg % (simname, dz_tol, desired_redshift, closest_redshift))

        # At this point we have a candidate file to download that 
        # matches the input specifications. 
        ###################################

        ###################################
        # Determine the download directory
        if download_dirname == 'std_cache_loc':
            cache_log_fname = manipulate_cache_log.get_halo_table_cache_log_fname()
            cache_basedir = os.path.dirname(cache_log_fname)
            download_dirname = os.path.join(cache_basedir, 'halo_catalogs', simname, halo_finder)
            try:
                os.makedirs(std_cache_loc)
            except OSError:
                pass
        else:
            try:
                assert os.path.exists(download_dirname)
            except AssertionError:
                msg = ("\nThe input ``download_dirname`` is a non-existent path.\n")
                raise HalotoolsError(msg)
        output_fname = os.path.join(download_dirname, os.path.basename(url))
        ###################################


        ###################################
        # Now we check the cache log to see if there are any matching entries 

        ### LEFT OFF HERE 

        ###################################
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
        try:
            simname = kwargs['simname']
            if simname not in supported_sim_list:
                raise HalotoolsError(unsupported_simname_msg % simname)
        except KeyError:
            pass

        try:
            desired_redshift = kwargs['desired_redshift']
        except KeyError:
            msg = ("\n``desired_redshift`` is a required keyword argument.\n")
            raise HalotoolsError(msg)

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
            raise HalotoolsError(msg)

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











