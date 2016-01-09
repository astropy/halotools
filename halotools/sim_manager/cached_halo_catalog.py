import numpy as np
import os
from warnings import warn 
from copy import deepcopy 

from astropy.table import Table

try:
    import h5py
except ImportError:
    warn("Most of the functionality of the "
        "sim_manager sub-package requires h5py to be installed,\n"
        "which can be accomplished either with pip or conda. ")

from . import sim_defaults, supported_sims

from .halo_table_cache import HaloTableCache
from .ptcl_table_cache import PtclTableCache
from .halo_table_cache_log_entry import get_redshift_string
from .ptcl_table_cache_log_entry import PtclTableCacheLogEntry

from ..custom_exceptions import HalotoolsError, InvalidCacheLogEntry


__all__ = ('CachedHaloCatalog', )


class CachedHaloCatalog(object):
    """
    Container class for halo catalogs and particle data.  
    """

    def __init__(self, preload_halo_table = False, dz_tol = 0.05, 
        update_cached_fname = False, **kwargs):
        """
        """
        try:
            import h5py
        except ImportError:
            raise HalotoolsError("Must have h5py package installed "
                "to use CachedHaloCatalog objects")

        self._dz_tol = dz_tol
        self._update_cached_fname = update_cached_fname

        self.halo_table_cache = HaloTableCache() 

        self.log_entry = self._determine_cache_log_entry(**kwargs)
        self.simname = self.log_entry.simname 
        self.halo_finder = self.log_entry.halo_finder 
        self.version_name = self.log_entry.version_name 
        self.redshift = self.log_entry.redshift 
        self.fname = self.log_entry.fname 

        self._bind_additional_metadata()

        if preload_halo_table is True:
            _ = self.halo_table
            del _

    def _determine_cache_log_entry(self, **kwargs):
        """
        """ 

        if 'fname' in kwargs:
            fname = kwargs['fname']

            if not os.path.isfile(fname):
                msg = ("\nThe ``fname`` you passed to the CachedHaloCatalog "
                    "constructor is a non-existent path.\n")
                raise HalotoolsError(msg)

            try:
                assert 'simname' not in kwargs 
            except AssertionError:
                msg = ("\nIf you specify an input ``fname``, "
                    "do not also specify ``simname``.\n")
                raise HalotoolsError(msg)

            try:
                assert 'halo_finder' not in kwargs 
            except AssertionError:
                msg = ("\nIf you specify an input ``fname``, "
                    "do not also specify ``halo_finder``.\n")
                raise HalotoolsError(msg)

            try:
                assert 'redshift' not in kwargs 
            except AssertionError:
                msg = ("\nIf you specify an input ``fname``, "
                    "do not also specify ``redshift``.\n")
                raise HalotoolsError(msg)

            try:
                assert 'version_name' not in kwargs 
            except AssertionError:
                msg = ("\nIf you specify an input ``fname``, "
                    "do not also specify ``version_name``.\n")
                raise HalotoolsError(msg)

            return self._retrieve_matching_log_entry_from_fname(fname)

        else:

            try:
                simname = kwargs['simname']
                self._default_simname_choice = False
            except KeyError:
                simname = sim_defaults.default_simname
                self._default_simname_choice = True

            try:
                halo_finder = kwargs['halo_finder']
                self._default_halo_finder_choice = False
            except KeyError:
                halo_finder = sim_defaults.default_halo_finder
                self._default_halo_finder_choice = True

            try:
                version_name = kwargs['version_name']
                self._default_version_name_choice = False
            except KeyError:
                version_name = sim_defaults.default_version_name
                self._default_version_name_choice = True
            
            try:
                redshift = kwargs['redshift']
                self._default_redshift_choice = False
            except KeyError:
                redshift = sim_defaults.default_redshift
                self._default_redshift_choice = True

            return self._retrieve_matching_log_entry_from_metadata(
                simname, halo_finder, version_name, redshift)

    def _retrieve_matching_log_entry_from_fname(self, fname):
        """
        """
        log_entry = self.halo_table_cache.determine_log_entry_from_fname(fname, 
            overwrite_fname_metadata = False)

        if log_entry.fname != fname:
            if self._update_cached_fname == True:
                old_fname = deepcopy(log_entry.fname)
                log_entry = (
                    self.halo_table_cache.determine_log_entry_from_fname(fname, 
                        overwrite_fname_metadata = self._update_cached_fname)
                    )
                self.halo_table_cache.update_cached_file_location(
                    fname, old_fname)
            else:
                msg = ("\nThe ``fname`` you passed as an input to the "
                    "CachedHaloCatalog class \ndoes not match the ``fname`` "
                    "stored as metadata in the hdf5 file.\n"
                    "This means that at some point you manually relocated the catalog on disk \n"
                    "after storing its location in cache, "
                    "but you did not yet update the Halotools cache log. \n"
                    "When possible, try to keep your halo catalogs "
                    "at a fixed disk location \n"
                    "as this helps ensure reproducibility. \n"
                    "If the ``fname`` you passed to CachedHaloCatalog is the "
                    "new location you want to store the catalog, \n"
                    "then you can update the cache by calling the CachedHaloCatalog \n"
                    "constructor again and setting the ``update_cached_fname`` variable to True.\n")
                raise HalotoolsError(msg)

        return log_entry


    def _retrieve_matching_ptcl_cache_log_entry(self):
        """
        """

        ptcl_table_cache = PtclTableCache()
        if len(ptcl_table_cache.log) == 0:
            msg = ("\nThe Halotools cache log has no record of any particle catalogs.\n"
                "If you have never used Halotools before, "
                "you should read the Getting Started guide on halotools.readthedocs.org.\n"
                "If you have previously used the package before, \n"
                "try running the halotools/scripts/rebuild_ptcl_table_cache_log.py script.\n")
            raise HalotoolsError(msg)

        gen0 = ptcl_table_cache.matching_log_entry_generator(
            simname = self.simname, version_name = self.version_name, 
            redshift = self.redshift, dz_tol = self._dz_tol)
        gen1 = ptcl_table_cache.matching_log_entry_generator(
            simname = self.simname, version_name = self.version_name)
        gen2 = ptcl_table_cache.matching_log_entry_generator(simname = self.simname)

        matching_entries = list(gen0)     

        msg = ("\nYou tried to load a cached halo catalog "
            "with the following characteristics:\n\n")

        if self._default_simname_choice is True:
            msg += ("simname = ``" + str(self.simname) 
                + "``  (set by sim_defaults.default_simname)\n")
        else:
            msg += "simname = ``" + str(self.simname) + "``\n"

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
            "within dz_tol = "+str(self._dz_tol)+" of these inputs.\n"
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
                    msg += "There are no simulations matching your input simname.\n"
            raise InvalidCacheLogEntry(msg)

        elif len(matching_entries) == 1:
            log_entry = matching_entries[0]
            return log_entry

        else:
            msg += ("There are multiple entries in the cache log \n"
                "within dz_tol = "+str(self._dz_tol)+" of your inputs. \n"
                "Try using the exact redshift and/or decreasing dz_tol.\n"
                "Now printing the matching entries:\n\n")
            for entry in matching_entries:
                msg += str(entry) + "\n"
            raise InvalidCacheLogEntry(msg)


    def _retrieve_matching_log_entry_from_metadata(self, 
        simname, halo_finder, version_name, redshift):
        """
        """
        
        if len(self.halo_table_cache.log) == 0:
            msg = ("\nThe Halotools cache log is empty.\n"
                "If you have never used Halotools before, "
                "you should read the Getting Started guide on halotools.readthedocs.org.\n"
                "If you have previously used the package before, \n"
                "try running the halotools/scripts/rebuild_halo_table_cache_log.py script.\n")
            raise HalotoolsError(msg)


        gen0 = self.halo_table_cache.matching_log_entry_generator(
            simname = simname, halo_finder = halo_finder, 
            version_name = version_name, redshift = redshift, 
            dz_tol = self._dz_tol)
        gen1 = self.halo_table_cache.matching_log_entry_generator(
            simname = simname, 
            halo_finder = halo_finder, version_name = version_name)
        gen2 = self.halo_table_cache.matching_log_entry_generator(
            simname = simname, halo_finder = halo_finder)
        gen3 = self.halo_table_cache.matching_log_entry_generator(
            simname = simname)

        matching_entries = list(gen0)     

        msg = ("\nYou tried to load a cached halo catalog "
            "with the following characteristics:\n\n")

        if self._default_simname_choice is True:
            msg += ("simname = ``" + str(simname) 
                + "``  (set by sim_defaults.default_simname)\n")
        else:
            msg += "simname = ``" + str(simname) + "``\n"

        if self._default_halo_finder_choice is True:
            msg += ("halo_finder = ``" + str(halo_finder) 
                + "``  (set by sim_defaults.default_halo_finder)\n")
        else:
            msg += "halo_finder = ``" + str(halo_finder) + "``\n"

        if self._default_version_name_choice is True:
            msg += ("version_name = ``" + str(version_name) 
                + "``  (set by sim_defaults.default_version_name)\n")
        else:
            msg += "version_name = ``" + str(version_name) + "``\n"

        if self._default_redshift_choice is True:
            msg += ("redshift = ``" + str(redshift) 
                + "``  (set by sim_defaults.default_redshift)\n")
        else:
            msg += "redshift = ``" + str(redshift) + "``\n"

        msg += ("\nThere is no matching catalog in cache "
            "within dz_tol = "+str(self._dz_tol)+" of these inputs.\n"
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
            log_entry = matching_entries[0]
            return log_entry

        else:
            msg += ("There are multiple entries in the cache log \n"
                "within dz_tol = "+str(self._dz_tol)+" of your inputs. \n"
                "Try using the exact redshift and/or decreasing dz_tol.\n"
                "Now printing the matching entries:\n\n")
            for entry in matching_entries:
                msg += str(entry) + "\n"
            raise InvalidCacheLogEntry(msg)

    @property 
    def halo_table(self):
        """
        Astropy `~astropy.table.Table` object storing a catalog of dark matter halos. 

        You can access the array storing, say, halo virial mass using the following syntax: 

        >>> halocat = CachedHaloCatalog() # doctest: +SKIP
        >>> mass_array = halocat.halo_table['halo_mvir'] # doctest: +SKIP

        To see what halo properties are available in the catalog:

        >>> print(halocat.halo_table.keys()) # doctest: +SKIP
        """
        try:
            return self._halo_table
        except AttributeError:
            if self.log_entry.safe_for_cache == True:
                self._halo_table = Table.read(self.fname, path='data')
                return self._halo_table
            else:
                raise InvalidCacheLogEntry(self.log_entry._cache_safety_message)

    def _bind_additional_metadata(self):
        """ Create convenience bindings of all metadata to the `CachedHaloCatalog` instance. 
        """
        f = h5py.File(self.log_entry.fname)
        for attr_key in f.attrs.keys():
            if attr_key == 'redshift':
                setattr(self, attr_key, float(get_redshift_string(f.attrs[attr_key])))
            else:
                setattr(self, attr_key, f.attrs[attr_key])
        f.close()

        matching_sim = self._retrieve_supported_sim()
        if matching_sim is not None:
            for attr in matching_sim._attrlist:
                if hasattr(self, attr):
                    try:
                        assert getattr(self, attr) == getattr(matching_sim, attr)
                    except AssertionError:
                        msg = ("The ``" + attr + "`` metadata of the hdf5 file \n"
                            "is inconsistent with the corresponding attribute of the \n" 
                            + matching_sim.__class__.__name__ + "class in the "
                            "sim_manager.supported_sims module.\n"
                            "Double-check the value of this attribute in the \n"
                            "NbodySimulation sub-class you added to the supported_sims module. \n"
                            )
                        raise HalotoolsError(msg)
                else:
                    setattr(self, attr, getattr(matching_sim, attr))
        else:
            msg = ("You have stored your own simulation in the Halotools cache \n"
                "but you have not added a corresponding NbodySimulation sub-class. \n"
                "This is permissible, but not recommended. \n"
                "See, for example, the Bolshoi sub-class for how to add your own simulation. \n")
            warn(msg)

    def _retrieve_supported_sim(self):
        """
        """
        matching_sim = None
        for clname in supported_sims.__all__:
            try:
                cl = getattr(supported_sims, clname)
                obj = cl()
                if isinstance(obj, supported_sims.NbodySimulation):
                    if self.simname == obj.simname:
                        matching_sim = obj
            except TypeError:
                pass
        return matching_sim

    @property 
    def ptcl_table(self):
        """
        Astropy `~astropy.table.Table` object storing a collection of ~1e6 randomly selected dark matter particles. 
        """
        try:
            return self._ptcl_table
        except AttributeError:
            try:
                ptcl_log_entry = self.ptcl_log_entry 
            except AttributeError:
                self.ptcl_log_entry = (
                    self._retrieve_matching_ptcl_cache_log_entry()
                    )
                ptcl_log_entry = self.ptcl_log_entry

            if ptcl_log_entry.safe_for_cache == True:
                self._ptcl_table = Table.read(ptcl_log_entry.fname, path='data')
                return self._ptcl_table
            else:
                raise InvalidCacheLogEntry(ptcl_log_entry._cache_safety_message)








