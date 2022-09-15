"""
Module storing the DownloadManager class responsible for
retrieving the Halotools-provided simulation data from the web
and storing it in the Halotools cache.
"""

import numpy as np
from warnings import warn
from time import time

from ..custom_exceptions import HalotoolsError

try:
    from bs4 import BeautifulSoup
except ImportError:
    raise HalotoolsError("Must have bs4 package installed to use the DownloadManager")

try:
    import requests
except ImportError:
    raise HalotoolsError(
        "Must have requests package installed to use the DownloadManager"
    )

import posixpath
import urllib

import os
import fnmatch

from ..sim_manager import sim_defaults, supported_sims

from .halo_table_cache import HaloTableCache
from .ptcl_table_cache import PtclTableCache
from .halo_table_cache_log_entry import get_redshift_string


try:
    import h5py
except ImportError:
    warn(
        "Some of the functionality of the DownloadManager requires h5py to be installed,\n"
        "which can be accomplished either with pip or conda"
    )


from ..utils.array_utils import find_idx_nearest_val
from ..utils.array_utils import custom_len
from ..utils.io_utils import download_file_from_url


__all__ = ("DownloadManager",)

unsupported_simname_msg = (
    "There are no web locations recognized by Halotools \n for simname ``%s``"
)


class DownloadManager(object):
    """Class used to scrape the web for simulation data and cache the downloaded catalogs.

    For a list of available pre-processed halo catalogs provided by Halotools,
    see :ref:`supported_sim_list`.
    """

    def __init__(self):
        """ """
        self.halo_table_cache = HaloTableCache()
        self.ptcl_table_cache = PtclTableCache()

    def download_processed_halo_table(
        self,
        simname,
        halo_finder,
        redshift,
        dz_tol=0.1,
        overwrite=False,
        version_name=sim_defaults.default_version_name,
        download_dirname="std_cache_loc",
        ignore_nearby_redshifts=False,
        **kwargs
    ):
        """Method to download one of the pre-processed binary files
        storing a reduced halo catalog.

        Parameters
        ----------
        simname : string
            Nickname of the simulation. Currently supported simulations are
            Bolshoi  (simname = ``bolshoi``), Consuelo (simname = ``consuelo``),
            MultiDark (simname = ``multidark``), and Bolshoi-Planck (simname = ``bolplanck``).

        halo_finder : string
            Nickname of the halo-finder, e.g. `rockstar` or `bdm`.

        redshift : float
            Redshift of the requested snapshot.
            Must match one of theavailable snapshots within dz_tol,
            or a prompt will be issued providing the nearest
            available snapshots to choose from.

        version_name : string, optional
            Nickname of the version of the halo catalog used to differentiate
            between the same halo catalog processed in different ways.
            Default is set in `~halotools.sim_manager.sim_defaults`.

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

        ignore_nearby_redshifts : bool, optional
            Flag used to determine whether nearby redshifts in cache will be ignored.
            If there are existing halo catalogs in the Halotools cache with matching
            ``simname``, ``halo_finder`` and ``version_name``,
            and if one or more of those catalogs has a redshift within ``dz_tol``,
            then the ignore_nearby_redshifts flag must be set to True in order
            for the new halo catalog to be stored in cache.
            Default is False.

        Examples
        -----------
        >>> from halotools.sim_manager import sim_defaults
        >>>
        >>> dman = DownloadManager()
        >>> simname = 'bolplanck'
        >>> z = 2
        >>> version_name = sim_defaults.default_version_name
        >>> halo_finder = sim_defaults.default_halo_finder
        >>> dman.download_processed_halo_table(simname = 'bolplanck', halo_finder = halo_finder, version_name = version_name, redshift = z) # doctest: +SKIP

        Now that you have downloaded the catalog, it is stored in the default cache location:

        $HOME/.astropy/cache/halotools/halo_catalogs/
        Use the download_dirname keyword argument to store the catalog in an alternate location.
        Wherever you store it, after calling the `download_processed_halo_table` method
        you can load the catalog into memory as follows:

        >>> from halotools.sim_manager import CachedHaloCatalog
        >>> halocat = CachedHaloCatalog(simname = 'bolplanck', redshift = z) # doctest: +SKIP

        Since you chose default values for the ``version_name`` and ``halo_finder``,
        it is not necessary to specify these keyword arguments. The ``halocat`` has
        metadata attached to it describing the simulation, snapshot, catalog processing notes, etc.
        The actual halos are stored in the form of an Astropy `~astropy.table.Table` data structure
        and can be accessed as follows:

        >>> halos = halocat.halo_table # doctest: +SKIP
        >>> array_of_masses = halos['halo_mvir'] # doctest: +SKIP
        >>> array_of_x_position = halos['halo_x'] # doctest: +SKIP

        Notes
        -------
        If after downloading the catalog you decide that you want to move it
        to a new location on disk, you will need to be sure your cache directory
        is informed of the relocation.
        In this case, see :ref:`relocating_simulation_data` for instructions.

        """
        self.halo_table_cache.update_log_from_current_ascii()

        ############################################################
        # Identify candidate file to download

        available_fnames_to_download = (
            self._processed_halo_tables_available_for_download(
                simname=simname, halo_finder=halo_finder, version_name=version_name
            )
        )

        if len(available_fnames_to_download) == 0:
            msg = "You made the following request for a pre-processed halo catalog:\n"

            msg += "simname = " + simname + "\n"
            msg += "halo_finder = " + halo_finder + "\n"
            msg += "version_name = " + version_name + "\n"
            msg = msg + "There are no halo catalogs meeting your specifications"
            raise HalotoolsError(msg)

        url, closest_redshift = self._closest_fname(
            available_fnames_to_download, redshift
        )

        closest_redshift_string = get_redshift_string(closest_redshift)
        closest_redshift = float(closest_redshift_string)

        if abs(closest_redshift - redshift) > dz_tol:
            msg = (
                "\nNo pre-processed %s halo catalog has \na redshift within %.2f "
                + "of the redshift = %.2f.\n The closest redshift for these catalogs is %s \n"
            )
            raise HalotoolsError(
                msg % (simname, dz_tol, redshift, closest_redshift_string)
            )

        # At this point we have a candidate file to download that
        # matches the input specifications.
        ############################################################

        ############################################################
        # Determine the download directory,
        # passively creating the necessary directory tree
        if download_dirname == "std_cache_loc":
            cache_basedir = os.path.dirname(self.halo_table_cache.cache_log_fname)
            download_dirname = os.path.join(
                cache_basedir, "halo_catalogs", simname, halo_finder
            )
            try:
                os.makedirs(download_dirname)
            except OSError:
                pass
        else:
            try:
                assert os.path.exists(download_dirname)
            except AssertionError:
                msg = "\nYour input ``download_dirname`` is a non-existent path.\n"
                raise HalotoolsError(msg)
        output_fname = os.path.join(download_dirname, os.path.basename(url))
        ############################################################

        ############################################################
        # Now we check the cache log to see if there are any matching entries
        exact_match_generator = self.halo_table_cache.matching_log_entry_generator(
            simname=simname,
            halo_finder=halo_finder,
            version_name=version_name,
            redshift=closest_redshift,
            dz_tol=0.0,
        )
        exact_matches = list(exact_match_generator)

        if len(exact_matches) > 0:
            msg = (
                "\nThere already exists a halo catalog in your cache log with \n"
                "specifications that exactly match your inputs.\n"
            )
            if overwrite is False:
                if "initial_download_script_msg" in list(kwargs.keys()):
                    msg = kwargs["initial_download_script_msg"]
                    raise HalotoolsError(msg % output_fname)
                else:
                    msg += (
                        "If you want to overwrite this catalog with your download, \n"
                        "you must set the ``overwrite`` keyword argument to True. \n"
                        "Alternatively, you can delete the log entry using the \n"
                        "remove_entry_from_cache_log method of the HaloTableCache class.\n"
                    )
                    raise HalotoolsError(msg)
            else:
                msg += (
                    "Since you have set ``overwrite`` to True, \n"
                    "the download will proceed and the existing file will be overwritten.\n"
                )
                warn(msg)

        close_match_generator = self.halo_table_cache.matching_log_entry_generator(
            simname=simname,
            halo_finder=halo_finder,
            version_name=version_name,
            redshift=closest_redshift,
            dz_tol=dz_tol,
        )
        close_matches = list(close_match_generator)

        if (
            (len(close_matches) > 0)
            & (len(exact_matches) == 1)
            & (ignore_nearby_redshifts is False)
        ):

            entry = close_matches[0]
            msg = "\nThe following filename appears in the cache log. \n\n"
            msg += str(entry.fname) + "\n\n"
            msg += (
                "This log entry has exactly matching metadata "
                "and a redshift within the input ``dz_tol`` = "
                + str(dz_tol)
                + "\n of the redshift of the most closely matching catalog on the web.\n"
                "In order to proceed, you must either set "
                "the ``ignore_nearby_redshifts`` to True, or decrease ``dz_tol``. \n"
            )
            raise HalotoolsError(msg)

        # At this point there are no conflicts with the existing log
        ############################################################

        ############################################################
        # If the output_fname already exists, overwrite must be set to True
        # A special message is printed if this exception is raised by the
        # initial download script (hidden feature for developers only)
        if (overwrite is False) & (os.path.isfile(output_fname)):

            if "initial_download_script_msg" in list(kwargs.keys()):
                msg = kwargs["initial_download_script_msg"]
            else:
                msg = (
                    "The following filename already exists "
                    "in your cache directory: \n\n%s\n\n"
                    "If you really want to overwrite the file, \n"
                    "you must call the same function again \n"
                    "with the keyword argument `overwrite` set to `True`"
                )
            raise HalotoolsError(msg % output_fname)

        start = time()
        download_file_from_url(url, output_fname)
        end = time()
        runtime = end - start
        print(
            (
                "\nTotal runtime to download pre-processed "
                "halo catalog = %.1f seconds\n" % runtime
            )
        )

        # overwrite the fname metadata so that
        # it is consistent with the downloaded location
        try:
            import h5py
        except ImportError:
            msg = (
                "\nYou must have h5py installed to use "
                "the \ndownload_processed_halo_table method "
                "of the DownloadManager class.\n"
            )
            raise HalotoolsError(msg)

        f = h5py.File(output_fname, "a")
        f.attrs["fname"] = str(output_fname)
        f.close()

        new_log_entry = self.halo_table_cache.determine_log_entry_from_fname(
            output_fname
        )

        if new_log_entry.safe_for_cache is False:
            msg = (
                "\nThere is a problem with the file you downloaded.\n"
                "Please take note of the following filename "
                "and contact the Halotools developers.\n" + output_fname
            )
            raise HalotoolsError(msg)

        self.halo_table_cache.add_entry_to_cache_log(new_log_entry)

        # Print a special message if this download was triggered
        # by the downloading script (hidden feature for developers only)
        try:
            success_msg = kwargs["success_msg"]
        except KeyError:
            args_msg = "simname = '" + str(new_log_entry.simname) + "'"
            args_msg += ", halo_finder = '" + str(new_log_entry.halo_finder) + "'"
            args_msg += ", version_name = '" + str(new_log_entry.version_name) + "'"
            args_msg += ", redshift = " + str(new_log_entry.redshift)

            success_msg = (
                "\nThe halo catalog has been successfully downloaded "
                "to the following location:\n" + str(output_fname) + "\n\n"
                "This filename and its associated metadata have also been "
                "added to the Halotools cache log, \n"
                "as reflected by a newly added line to the following ASCII file:\n\n"
                + str(self.halo_table_cache.cache_log_fname)
                + "\n\n"
                "Since the catalog will now be recognized in cache, \n"
                "you can load it into memory using the following syntax:\n\n"
                ">>> from halotools.sim_manager import CachedHaloCatalog \n"
                ">>> halocat = CachedHaloCatalog(" + args_msg + ") \n\n"
                "For convenience, you can set this catalog to be your default catalog \n"
                "and omit the CachedHaloCatalog constructor arguments.\n"
                "To do that, change the following variables in the \n"
                "halotools.sim_manager.sim_defaults module:\n"
                "``default_simname``, ``default_halo_finder``, "
                "``default_version_name`` and ``default_redshift``.\n\n"
                "Tabular data storing the actual halos is bound to the ``halo_table`` \n"
                "attribute of ``halocat`` in the form of an Astropy Table:\n\n"
                ">>> halos = halocat.halo_table \n\n"
                "Halo properties are accessed in the same manner "
                "as a python dictionary or Numpy structured array:\n\n"
                ">>> mass_array = halos['halo_mvir'] \n\n"
                "The ``halocat`` object also contains additional metadata \n"
                "such as ``halocat.simname``, ``halocat.redshift`` and ``halocat.fname`` \n"
                "that you can use for sanity checks on your bookkeeping.\n\n"
                "Note that if you move this halo catalog into a new location on disk, \n"
                "you must update both the ``fname`` metadata of the hdf5 file \n"
                "as well as the``fname`` column of the corresponding entry in the log. \n"
                "You can accomplish this with the ``update_cached_file_location``"
                "method \nof the HaloTableCache class.\n\n"
            )

        if "initial_download_script_msg" in list(kwargs.keys()):
            return new_log_entry
        else:
            print(success_msg)

    def download_ptcl_table(
        self,
        simname,
        redshift,
        dz_tol=0.1,
        overwrite=False,
        version_name=sim_defaults.default_ptcl_version_name,
        download_dirname="std_cache_loc",
        ignore_nearby_redshifts=False,
        **kwargs
    ):
        """Method to download one of the binary files storing a
        random downsampling of dark matter particles.

        Parameters
        ----------
        simname : string
            Nickname of the simulation. Currently supported simulations are
            Bolshoi  (simname = ``bolshoi``), Consuelo (simname = ``consuelo``),
            MultiDark (simname = ``multidark``), and Bolshoi-Planck (simname = ``bolplanck``).

        redshift : float
            Redshift of the requested snapshot.
            Must match one of theavailable snapshots within dz_tol,
            or a prompt will be issued providing the nearest
            available snapshots to choose from.

        version_name : string, optional
            Nickname of the version of the halo catalog used to differentiate
            between the same halo catalog processed in different ways.
            Default is set in `~halotools.sim_manager.sim_defaults`.

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

        ignore_nearby_redshifts : bool, optional
            Flag used to determine whether nearby redshifts in cache will be ignored.
            If there are existing halo catalogs in the Halotools cache with matching
            ``simname``, ``halo_finder`` and ``version_name``,
            and if one or more of those catalogs has a redshift within ``dz_tol``,
            then the ignore_nearby_redshifts flag must be set to True in order
            for the new halo catalog to be stored in cache.
            Default is False.

        Examples
        -----------
        >>> dman = DownloadManager()
        >>> simname = 'bolplanck'
        >>> z = 2
        >>> version_name = sim_defaults.default_version_name
        >>> dman.download_ptcl_table(simname = 'bolplanck', version_name = version_name, redshift = z) # doctest: +SKIP

        Now that you have downloaded the particles, the data is stored in the default cache location:

        $HOME/.astropy/cache/halotools/particle_catalogs/
        Use the download_dirname keyword argument to store the catalog in an alternate location.
        Wherever you store it, after calling the `download_ptcl_table` method
        you can access particle data by loading the associated halo catalog into memory:

        >>> from halotools.sim_manager import CachedHaloCatalog
        >>> halocat = CachedHaloCatalog(simname = 'bolplanck', redshift = z, halo_finder = 'rockstar') # doctest: +SKIP

        Since you chose default values for the ``version_name``,
        it is not necessary to specify that keyword arguments. The ``halocat`` has
        metadata attached to it describing the simulation, snapshot, catalog processing notes, etc.
        The actual particles are stored in the form of an Astropy `~astropy.table.Table` data structure
        and can be accessed as follows:

        >>> particles = halocat.ptcl_table # doctest: +SKIP
        >>> array_of_x_position = particles['x'] # doctest: +SKIP

        Notes
        -------
        If after downloading the catalog you decide that you want to move it
        to a new location on disk, you will need to be sure your cache directory
        is informed of the relocation.
        In this case, see :ref:`relocating_simulation_data` for instructions.


        """
        self.halo_table_cache.update_log_from_current_ascii()
        ############################################################
        # Identify candidate file to download

        available_fnames_to_download = self._ptcl_tables_available_for_download(
            simname=simname, version_name=version_name
        )

        if len(available_fnames_to_download) == 0:
            msg = "You made the following request for a pre-processed halo catalog:\n"

            msg += "simname = " + simname + "\n"
            msg += "version_name = " + version_name + "\n"
            msg = msg + "There are no particle catalogs meeting your specifications"
            raise HalotoolsError(msg)

        url, closest_redshift = self._closest_fname(
            available_fnames_to_download, redshift
        )

        closest_redshift_string = get_redshift_string(closest_redshift)
        closest_redshift = float(closest_redshift_string)

        if abs(closest_redshift - redshift) > dz_tol:
            msg = (
                "\nNo particle catalog for the ``%s`` simulation has \na redshift within %.2f "
                + "of the redshift = %.2f.\n The closest redshift for these catalogs is %s \n"
            )
            raise HalotoolsError(
                msg % (simname, dz_tol, redshift, closest_redshift_string)
            )

        # At this point we have a candidate file to download that
        # matches the input specifications.
        ############################################################

        ############################################################
        # Determine the download directory,
        # passively creating the necessary directory tree
        if download_dirname == "std_cache_loc":
            cache_basedir = os.path.dirname(self.ptcl_table_cache.cache_log_fname)
            download_dirname = os.path.join(cache_basedir, "particle_catalogs", simname)
            try:
                os.makedirs(download_dirname)
            except OSError:
                pass
        else:
            try:
                assert os.path.exists(download_dirname)
            except AssertionError:
                msg = "\nYour input ``download_dirname`` is a non-existent path.\n"
                raise HalotoolsError(msg)
        output_fname = os.path.join(download_dirname, os.path.basename(url))
        ############################################################

        ############################################################
        # Now we check the cache log to see if there are any matching entries
        exact_match_generator = self.ptcl_table_cache.matching_log_entry_generator(
            simname=simname,
            version_name=version_name,
            redshift=closest_redshift,
            dz_tol=0.0,
        )
        exact_matches = list(exact_match_generator)

        if len(exact_matches) > 0:
            msg = (
                "\nThere already exists a particle catalog in your cache log with \n"
                "specifications that exactly match your inputs.\n"
            )
            if overwrite is False:
                if "initial_download_script_msg" in list(kwargs.keys()):
                    msg = kwargs["initial_download_script_msg"]
                    raise HalotoolsError(msg % output_fname)
                else:
                    msg += (
                        "If you want to overwrite this catalog with your download, \n"
                        "you must set the ``overwrite`` keyword argument to True. \n"
                        "Alternatively, you can delete the log entry using the \n"
                        "remove_entry_from_cache_log method of the PtclTableCache class.\n"
                    )
                    raise HalotoolsError(msg)
            else:
                msg += (
                    "Since you have set ``overwrite`` to True, \n"
                    "the download will proceed and the existing file will be overwritten.\n"
                )
                warn(msg)

        close_match_generator = self.ptcl_table_cache.matching_log_entry_generator(
            simname=simname,
            version_name=version_name,
            redshift=closest_redshift,
            dz_tol=dz_tol,
        )
        close_matches = list(close_match_generator)

        if (
            (len(close_matches) > 0)
            & (len(exact_matches) == 0)
            & (ignore_nearby_redshifts is False)
        ):

            entry = close_matches[0]
            msg = "\nThe following filename appears in the cache log. \n\n"
            msg += str(entry.fname) + "\n\n"
            msg += (
                "This log entry has exactly matching metadata "
                "and a redshift within the input ``dz_tol`` = "
                + str(dz_tol)
                + "\n of the redshift of the most closely matching catalog on the web.\n"
                "In order to proceed, you must either set "
                "the ``ignore_nearby_redshifts`` to True, or decrease ``dz_tol``. \n"
            )
            raise HalotoolsError(msg)

        # At this point there are no conflicts with the existing log
        ############################################################

        ############################################################
        # If the output_fname already exists, overwrite must be set to True
        # A special message is printed if this exception is raised by the
        # initial download script (hidden feature for developers only)
        if (overwrite is False) & (os.path.isfile(output_fname)):

            if "initial_download_script_msg" in list(kwargs.keys()):
                msg = kwargs["initial_download_script_msg"]
            else:
                msg = (
                    "The following filename already exists "
                    "in your cache directory: \n\n%s\n\n"
                    "If you really want to overwrite the file, \n"
                    "you must call the same function again \n"
                    "with the keyword argument `overwrite` set to `True`"
                )
            raise HalotoolsError(msg % output_fname)

        download_file_from_url(url, output_fname)

        # overwrite the fname metadata so that
        # it is consistent with the downloaded location
        try:
            import h5py
        except ImportError:
            msg = (
                "\nYou must have h5py installed to use "
                "the \ndownload_ptcl_table method "
                "of the DownloadManager class.\n"
            )
            raise HalotoolsError(msg)
        f = h5py.File(output_fname, "a")
        f.attrs["fname"] = str(output_fname)
        f.close()

        new_log_entry = self.ptcl_table_cache.determine_log_entry_from_fname(
            output_fname
        )

        if new_log_entry.safe_for_cache is False:
            msg = (
                "\nThere is a problem with the file you downloaded.\n"
                "Please take note of the following filename "
                "and contact the Halotools developers.\n" + output_fname
            )
            raise HalotoolsError(msg)

        self.ptcl_table_cache.add_entry_to_cache_log(new_log_entry)

        # Print a special message if this download was triggered
        # by the downloading script (hidden feature for developers only)
        try:
            success_msg = kwargs["success_msg"]
        except KeyError:
            args_msg = "simname = '" + str(new_log_entry.simname) + "'"
            args_msg += ", version_name = '" + str(new_log_entry.version_name) + "'"
            args_msg += ", redshift = " + str(new_log_entry.redshift)

            success_msg = (
                "\nThe particle catalog has been successfully downloaded "
                "to the following location:\n" + str(output_fname) + "\n\n"
                "This filename and its associated metadata have also been "
                "added to the Halotools cache log, \n"
                "as reflected by a newly added line to the following ASCII file:\n\n"
                + str(self.ptcl_table_cache.cache_log_fname)
                + "\n\n"
                "You can access the particle data with the following syntax:\n\n"
                ">>> from halotools.sim_manager import CachedHaloCatalog \n"
                ">>> halocat = CachedHaloCatalog(" + args_msg + ")\n"
                ">>> particles = halocat.ptcl_table \n\n"
                "Mock observable functions such as the galaxy-galaxy lensing signal \n"
                "can now be computed for mock galaxies populated into this simulation.\n\n"
                "Note that if you move this particle catalog into a new location on disk, \n"
                "you must update both the ``fname`` metadata of the hdf5 file \n"
                "as well as the``fname`` column of the corresponding entry in the log. \n"
                "You can accomplish this with the ``update_cached_file_location``"
                "method \nof the PtclTableCache class.\n\n"
            )

        if "initial_download_script_msg" in list(kwargs.keys()):
            return new_log_entry
        else:
            print(success_msg)

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
            simname = kwargs["simname"]
            halo_finder = kwargs["halo_finder"]
        except KeyError:
            raise HalotoolsError(
                "\nDownloadManager._orig_halo_table_web_location method "
                "must be called with ``simname`` and ``halo_finder`` arguments"
            )

        if simname == "multidark":
            return "http://slac.stanford.edu/~behroozi/MultiDark_Hlists_Rockstar/"
        elif simname == "bolshoi":
            if halo_finder == "rockstar":
                return "http://www.slac.stanford.edu/~behroozi/Bolshoi_Catalogs/"
            elif halo_finder == "bdm":
                return "http://www.slac.stanford.edu/~behroozi/Bolshoi_Catalogs_BDM/"
        elif simname == "bolplanck":
            return "http://www.slac.stanford.edu/~behroozi/BPlanck_Hlists/"
        elif simname == "consuelo":
            return "http://www.slac.stanford.edu/~behroozi/Consuelo_Catalogs/"
        else:
            raise HalotoolsError(
                "Input simname %s and halo_finder %s do not "
                "have Halotools-recognized web locations" % (simname, halo_finder)
            )

    def _get_scale_factor_substring(self, fname):
        """Method extracts the portion of the Rockstar hlist fname
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
        first_index = fname.index("_") + 1
        last_index = fname.index(".", fname.index(".") + 1)
        scale_factor_substring = fname[first_index:last_index]
        return scale_factor_substring

    def _closest_fname(self, filename_list, redshift):
        """ """

        if custom_len(filename_list) == 0:
            msg = "The _closest_fname method was passed an empty filename_list"
            raise HalotoolsError(msg)

        if redshift <= -1:
            raise ValueError("redshift of <= -1 is unphysical")
        else:
            input_scale_factor = 1.0 / (1.0 + redshift)

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
        input_scale_factor = 1.0 / (1.0 + redshift)
        idx_closest_catalog = find_idx_nearest_val(
            scale_factor_list, input_scale_factor
        )
        closest_scale_factor = scale_factor_list[idx_closest_catalog]
        output_fname = filename_list[idx_closest_catalog]

        closest_available_redshift = (1.0 / closest_scale_factor) - 1

        return output_fname, closest_available_redshift

    def _closest_catalog_on_web(
        self, catalog_type, simname, desired_redshift, **kwargs
    ):
        """
        Parameters
        ----------
        catalog_type : string
            Specifies which subdirectory of the Halotools cache to scrape for .hdf5 files.
            Must be either ``halos`` or ``particles``

        simname : string
            Nickname of the simulation. Currently supported simulations are
            Bolshoi  (simname = ``bolshoi``), Consuelo (simname = ``consuelo``),
            MultiDark (simname = ``multidark``), and Bolshoi-Planck (simname = ``bolplanck``).

        desired_redshift : float
            Redshift of the desired catalog.

        halo_finder : string, optional
            Nickname of the halo-finder, e.g. ``rockstar``.
            Required when input ``catalog_type`` is ``halos``.

        version_name : string, optional
            Nickname for the version of the catalog.
            Argument is used to filter the output list of filenames.
            Default is set by `~halotools.sim_manager.sim_defaults` module.

        Returns
        -------
        output_fname : list
            String of the filename with the closest matching redshift.

        actual_redshift : float
            Value of the redshift of the closest matching snapshot

        Examples
        --------
        >>> catman = DownloadManager()

        Suppose you would like to download a pre-processed halo catalog
        for the Bolshoi-Planck simulation for z=0.5.
        To identify the filename of the available catalog
        that most closely matches your needs:

        >>> webloc_closest_match = catman._closest_catalog_on_web(catalog_type='halos', simname='bolplanck', halo_finder='rockstar', desired_redshift=0.5)  # doctest: +REMOTE_DATA

        You may also wish to have a collection of downsampled
        dark matter particles to accompany this snapshot:

        >>> webloc_closest_match = catman._closest_catalog_on_web(catalog_type='particles', simname='bolplanck', desired_redshift=0.5)  # doctest: +REMOTE_DATA

        """
        if "redshift" in list(kwargs.keys()):
            msg = (
                "\nThe correct argument to use to specify the redshift \n"
                "you are searching for is with the ``desired_redshift`` keyword, \n"
                "not the ``redshift`` keyword.\n"
            )
            raise HalotoolsError(msg)

        try:
            assert catalog_type in ("particles", "halos")
        except AssertionError:
            msg = "Input ``catalog_type`` must be either ``particles`` or ``halos``"
            raise HalotoolsError(msg)

        if catalog_type is "halos":
            try:
                halo_finder = kwargs["halo_finder"]
            except KeyError:
                raise HalotoolsError(
                    "\nIf input catalog_type is ``halos``, "
                    "must pass ``halo_finder`` argument"
                )
        else:
            if "halo_finder" in list(kwargs.keys()):
                warn(
                    "There is no need to specify a halo-finder "
                    "when requesting particle data"
                )

        if simname not in supported_sims.supported_sim_list:
            raise HalotoolsError(unsupported_simname_msg % simname)

        try:
            version_name = kwargs["version_name"]
        except KeyError:
            version_name = sim_defaults.default_version_name

        try:
            ptcl_version_name = kwargs["ptcl_version_name"]
        except KeyError:
            ptcl_version_name = sim_defaults.default_ptcl_version_name

        if catalog_type is "particles":
            filename_list = self._ptcl_tables_available_for_download(
                simname=simname, version_name=ptcl_version_name
            )
        elif catalog_type is "halos":
            filename_list = self._processed_halo_tables_available_for_download(
                simname=simname, halo_finder=halo_finder, version_name=version_name
            )

        output_fname, actual_redshift = self._closest_fname(
            filename_list, desired_redshift
        )

        return output_fname, actual_redshift

    def _ptcl_tables_available_for_download(
        self, version_name=sim_defaults.default_ptcl_version_name, **kwargs
    ):
        """Method searches the appropriate web location and
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

        version_name : string, optional
            Nickname for the version of the catalog.
            Argument is used to filter the output list of filenames.
            Default is set by `~halotools.sim_manager.sim_defaults` module.

        Returns
        -------
        output : list
            List of web locations of all catalogs of downsampled particles
            matching the input arguments.

        """
        try:
            simname = kwargs["simname"]
            if simname not in supported_sims.supported_sim_list:
                raise HalotoolsError(unsupported_simname_msg % simname)
        except KeyError:
            pass

        baseurl = sim_defaults.ptcl_tables_webloc
        soup = BeautifulSoup(requests.get(baseurl).text, features="html.parser")
        simloclist = []
        for a in soup.find_all("a", href=True):
            dirpath = posixpath.dirname(urllib.parse.urlparse(a["href"]).path)
            if dirpath and dirpath[0] != "/":
                simloclist.append(baseurl + "/" + dirpath)

        catlist = []
        for simloc in simloclist:
            soup = BeautifulSoup(requests.get(simloc).text, features="html.parser")
            for a in soup.find_all("a"):
                catlist.append(simloc + "/" + a["href"])

        file_pattern = version_name
        all_ptcl_tables = fnmatch.filter(catlist, "*" + file_pattern + "*.hdf5")

        if "simname" in list(kwargs.keys()):
            simname = kwargs["simname"]
            file_pattern = "*" + simname + "*"
            output = fnmatch.filter(all_ptcl_tables, file_pattern)
        else:
            output = all_ptcl_tables

        return output

    def _processed_halo_tables_available_for_download(self, **kwargs):
        """Method searches the appropriate web location and
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
            Default is None, in which case `_processed_halo_tables_available_for_download`
            will not filter the returned list of filenames by ``simname``.

        halo_finder : string, optional
            Nickname of the halo-finder, e.g. ``rockstar``.

            Argument is used to filter the output list of filenames.
            Default is None, in which case `_processed_halo_tables_available_for_download`
            will not filter the returned list of filenames by ``halo_finder``.

        version_name : string, optional
            Nickname for the version of the catalog.

            Argument is used to filter the output list of filenames.
            Default is set by `~halotools.sim_manager.sim_defaults` module.

        Returns
        -------
        output : list
            List of web locations of all pre-processed halo catalogs
            matching the input arguments.

        """
        try:
            simname = kwargs["simname"]
            if simname not in supported_sims.supported_sim_list:
                raise HalotoolsError(unsupported_simname_msg % simname)
        except KeyError:
            pass

        try:
            version_name = kwargs["version_name"]
        except KeyError:
            version_name = sim_defaults.default_version_name

        baseurl = sim_defaults.processed_halo_tables_webloc
        soup = BeautifulSoup(requests.get(baseurl).text, features="html.parser")
        simloclist = []
        for a in soup.find_all("a", href=True):
            dirpath = posixpath.dirname(urllib.parse.urlparse(a["href"]).path)
            if dirpath and dirpath[0] != "/":
                simloclist.append(baseurl + "/" + dirpath)

        halocatloclist = []
        for simloc in simloclist:
            soup = BeautifulSoup(requests.get(simloc).text, features="html.parser")
            for a in soup.find_all("a", href=True):
                dirpath = posixpath.dirname(urllib.parse.urlparse(a["href"]).path)
                if dirpath and dirpath[0] != "/":
                    halocatloclist.append(simloc + "/" + dirpath)

        catlist = []
        for halocatdir in halocatloclist:
            soup = BeautifulSoup(requests.get(halocatdir).text, features="html.parser")
            for a in soup.find_all("a"):
                catlist.append(halocatdir + "/" + a["href"])

        file_pattern = version_name + ".hdf5"
        all_halocats = fnmatch.filter(catlist, "*" + file_pattern)

        # all_halocats a list of all pre-processed catalogs on the web
        # Now we apply our filter, if applicable

        if ("simname" in list(kwargs.keys())) & ("halo_finder" in list(kwargs.keys())):
            simname = kwargs["simname"]
            halo_finder = kwargs["halo_finder"]
            file_pattern = "*" + simname + "/" + halo_finder + "/*" + file_pattern
            output = fnmatch.filter(all_halocats, file_pattern)
        elif "simname" in list(kwargs.keys()):
            simname = kwargs["simname"]
            file_pattern = "*" + simname + "/*" + file_pattern
            output = fnmatch.filter(all_halocats, file_pattern)
        elif "halo_finder" in list(kwargs.keys()):
            halo_finder = kwargs["halo_finder"]
            file_pattern = "*/" + halo_finder + "/*" + file_pattern
            output = fnmatch.filter(all_halocats, file_pattern)
        else:
            output = all_halocats

        return output
