import numpy as np
import os, sys, urllib2, fnmatch
from warnings import warn 
import bisect 

from copy import deepcopy 
from astropy import cosmology
from astropy import units as u
from astropy.table import Table

try:
    import h5py
except ImportError:
    warn("Most of the functionality of the "
        "sim_manager sub-package requires h5py to be installed,\n"
        "which can be accomplished either with pip or conda. ")

from . import sim_defaults, catalog_manager, manipulate_cache_log

from .halo_table_cache import HaloTableCache
from .log_entry import HaloTableCacheLogEntry, get_redshift_string

from ..utils.array_utils import custom_len, convert_to_ndarray
from ..custom_exceptions import HalotoolsError, InvalidCacheLogEntry


__all__ = ('OverhauledHaloCatalog', )


class OverhauledHaloCatalog(object):
    """
    Container class for halo catalogs and particle data.  
    """

    def __init__(self, preload_halo_table = False, dz_tol = 0.05, 
        **kwargs):
        """
        """
        try:
            import h5py
        except ImportError:
            raise HalotoolsError("Must have h5py package installed "
                "to use OverhauledHaloCatalog objects")

        self._process_kwargs(**kwargs)

        self.log_entry = self._retrieve_matching_cache_log_entry(dz_tol)

        if preload_halo_table is True:
            self._load_halo_table()

        self._bind_metadata()


    def _process_kwargs(self, **kwargs):

        try:
            self.simname = kwargs['simname']
            self._default_simname_choice = False
        except KeyError:
            self.simname = sim_defaults.default_simname
            self._default_simname_choice = True
        try:
            self.halo_finder = kwargs['halo_finder']
            self._default_halo_finder_choice = False
        except KeyError:
            self.halo_finder = sim_defaults.default_halo_finder
            self._default_halo_finder_choice = True

        try:
            self.version_name = kwargs['version_name']
            self._default_version_name_choice = False
        except KeyError:
            self.version_name = sim_defaults.default_version_name
            self._default_version_name_choice = True
        
        try:
            self.redshift = kwargs['redshift']
            self._default_redshift_choice = False
        except KeyError:
            self.redshift = sim_defaults.default_redshift
            self._default_redshift_choice = True

    def _retrieve_matching_cache_log_entry(self, dz_tol):
        """
        """
        halo_table_cache = HaloTableCache()
        if len(halo_table_cache.log) == 0:
            msg = ("\nThe Halotools cache log is empty.\n"
                "If you have never used Haltools before, "
                "you should read the Getting Started guide on halotools.readthedocs.org.\n"
                "If you have previously used the package before, \n"
                "try running the halotools/scripts/rebuild_halo_table_cache_log.py script.\n")
            raise HalotoolsError(msg)


        gen0 = halo_table_cache.matching_log_entry_generator(
            simname = self.simname, halo_finder = self.halo_finder, 
            version_name = self.version_name, redshift = self.redshift, 
            dz_tol = dz_tol)
        gen1 = halo_table_cache.matching_log_entry_generator(
            simname = self.simname, 
            halo_finder = self.halo_finder, version_name = self.version_name)
        gen2 = halo_table_cache.matching_log_entry_generator(
            simname = self.simname, halo_finder = self.halo_finder)
        gen3 = halo_table_cache.matching_log_entry_generator(
            simname = self.simname)

        matching_entries = list(gen0)     

        msg = ("\nYou tried to load a cached halo catalog "
            "with the following characteristics:\n\n")

        if self._default_simname_choice is True:
            msg += ("simname = ``" + str(self.simname) 
                + "``  (set by sim_defaults.default_simname)\n")
        else:
            msg += "simname = ``" + str(self.simname) + "``\n"

        if self._default_halo_finder_choice is True:
            msg += ("halo_finder = ``" + str(self.halo_finder) 
                + "``  (set by sim_defaults.default_halo_finder)\n")
        else:
            msg += "halo_finder = ``" + str(self.halo_finder) + "``\n"

        if self._default_version_name_choice is True:
            msg += ("version_name = ``" + str(self.version_name) 
                + "``  (set by sim_defaults.default_version_name)\n")
        else:
            msg += "version_name = ``" + str(self.version_name) + "``\n"

        if self._default_redshift_choice is True:
            msg += ("redshift = ``" + str(self.redshift) 
                + "``  (set by sim_defaults.default_redshift)\n")
        else:
            msg += "redshift = ``" + str(self.redshift) + "``\n"

        msg += ("\nThere is no matching catalog in cache "
            "within dz_tol = "+str(dz_tol)+" of these inputs.\n"
            )

        if len(matching_entries) == 0:
            suggestion_preamble = ("\nThe following entries in the cache log "
                "most closely match your inputs:\n\n")
            alt_list1 = list(gen1) # discard the redshift requirement
            if len(alt_list1) > 0:
                msg += suggestion_preamble
                for entry in alt_list1: msg += str(entry) + "\n\n"
            else:
                alt_list2 = list(gen2) # discard the version_name requirement
                if len(alt_list2) > 0:
                    msg += suggestion_preamble
                    for entry in alt_list2: msg += str(entry) + "\n\n"
                else:
                    alt_list3 = list(gen3) # discard the halo_finder requirement
                    if len(alt_list3) > 0:
                        msg += suggestion_preamble
                        for entry in alt_list3: msg += str(entry) + "\n\n"
                    else:
                        msg += "There are no simulations matching your input simname.\n"
            raise InvalidCacheLogEntry(msg)

        elif len(matching_entries) == 1:
            return matching_entries[0]

        else:
            msg += ("There are multiple entries in the cache log \n"
                "within dz_tol = "+str(dz_tol)+" of your inputs. \n"
                "Try using the exact redshift and/or decreasing dz_tol.\n"
                "Now printing the matching entries:\n\n")
            for entry in matching_entries:
                msg += str(entry) + "\n"
            raise InvalidCacheLogEntry(msg)

    @property 
    def halo_table(self):
        """
        `~astropy.table.Table` object storing a catalog of dark matter halos. 
        """
        try:
            return self._halo_table
        except AttributeError:
            if self.log_entry.safe_for_cache == True:
                self._halo_table = Table.read(self.fname, path='data')
                return self._halo_table
            else:
                raise InvalidCacheLogEntry(self.log_entry._cache_safety_message)

    def _bind_metadata(self):
        """ Create convenience bindings of all metadata to the `HaloCatalog` instance. 
        """
        f = h5py.File(self.log_entry.fname)
        for attr_key in f.attrs.keys():
            if attr_key == 'redshift':
                setattr(self, attr_key, float(get_redshift_string(f.attrs[attr_key])))
            else:
                setattr(self, attr_key, f.attrs[attr_key])
        f.close()





