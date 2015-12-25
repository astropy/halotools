import os
from copy import copy
from astropy.config.paths import _find_home
from astropy.table import Table
from warnings import warn
import numpy as np 

from .log_entry import HaloTableCacheLogEntry
from ..custom_exceptions import InvalidCacheLogEntry, HalotoolsError

__all__ = ('HaloTableCache', )

class HaloTableCache(object):
    """ Object providing a collection of halo catalogs for use with Halotools. 
    """ 

    def __init__(self, **kwargs):
        self._standard_log_dirname = os.path.join(_find_home(), 
            '.astropy', 'cache', 'halotools')
        try:
            os.makedirs(os.path.dirname(self._standard_log_dirname))
        except OSError:
            pass
        self._standard_log_fname = os.path.join(self._standard_log_dirname, 'halo_table_cache_log.txt')
        
        try:
            self.cache_log_fname = kwargs['cache_log_fname']
        except KeyError:
            self.cache_log_fname = copy(self._standard_log_fname)
        self._cache_log_fname_exists = os.path.isfile(self.cache_log_fname)
        
        self.log = self.retrieve_log_from_ascii()
        
    def _overwrite_log_ascii(self, new_log):
        new_log.sort()
        log_table = self._log_table_from_log(new_log)
        log_table.write(self.cache_log_fname, format='ascii')
        
    def _clean_log_of_repeated_entries(self, input_log):
        cleaned_log = list(set(input_log))
        
        if len(cleaned_log) < len(input_log):
            msg = ("Detected multiple entries of the "
                "halo table cache log with identical entries.\n"
                "This is harmless. The log will be now be cleaned.\n")
            warn(msg)
            self._overwrite_log_ascii(cleaned_log)
        
        return cleaned_log
        
    def retrieve_log_from_ascii(self):
        """ Read '$HOME/.astropy/cache/halotools/halo_table_cache_log.txt', 
        clean the log of any repeated entries, sort the log, and return the resulting 
        list of `~halotools.sim_manager.HaloTableCacheLogEntry` instances. 
        """
        
        log_table = self._read_log_table_from_ascii()
        log_inferred_from_ascii = self._log_from_log_table(log_table)
        cleaned_log = self._clean_log_of_repeated_entries(log_inferred_from_ascii)
        cleaned_log.sort()
        return cleaned_log
        
    def _read_log_table_from_ascii(self):
            
        self._cache_log_fname_exists = os.path.isfile(self.cache_log_fname)
        
        if self._cache_log_fname_exists:
            try:
                log_table = Table.read(self.cache_log_fname, format='ascii')
                assert set(log_table.keys()) == set(
                    HaloTableCacheLogEntry.log_attributes)
                self._cache_log_fname_is_kosher = True
            except:
                log_table = self._get_empty_log_table()
                self._cache_log_fname_is_kosher = False
        else:
            log_table = self._get_empty_log_table()
        
        return log_table
    
    def _log_from_log_table(self, log_table):
        result = []
        for entry in log_table:
            constructor_kwargs = (
                {key:entry[key] for key in HaloTableCacheLogEntry.log_attributes})
            result.append(HaloTableCacheLogEntry(**constructor_kwargs))
        return result

            
    def _log_table_from_log(self, log):
        log_table = self._get_empty_log_table(len(log))
        for ii, entry in enumerate(log):
            for attr in HaloTableCacheLogEntry.log_attributes:
                log_table[attr][ii] = getattr(entry, attr)
        return log_table
        
    def _get_empty_log_table(self, num_entries = 0):
        if num_entries == 0:
            return Table(
                {'simname': [], 'halo_finder': [], 
                'redshift': [], 'version_name': [], 
                'fname': []}
                )
        else:
            return Table({'simname': np.zeros(num_entries, dtype=object), 
                'halo_finder': np.zeros(num_entries, dtype=object), 
                'redshift': np.zeros(num_entries, dtype=float), 
                'version_name': np.zeros(num_entries, dtype=object), 
                'fname': np.zeros(num_entries, dtype=object)})

    def matching_log_entry_generator(self, dz_tol = 0.0, **kwargs):
        """
        """

        try:
            assert set(kwargs.keys()).issubset(set(HaloTableCacheLogEntry.log_attributes))
        except AssertionError:
            msg = ("\nThe only acceptable keyword arguments to matching_log_entry_generator \n"
                "are elements of HaloTableCacheLogEntry.log_attributes:\n")
            for attr in HaloTableCacheLogEntry.log_attributes:
                msg += "``"+attr+"``, "
            msg = msg[:-2]
            raise KeyError(msg)

        for entry in self.log:
            yield_entry = True
            for key in kwargs.keys():
                if key == 'redshift':
                    requested_redshift = float(kwargs[key])
                    redshift_of_entry = float(getattr(entry, key))
                    yield_entry *=  abs(redshift_of_entry - requested_redshift) <= dz_tol
                else:
                    yield_entry *= kwargs[key] == getattr(entry, key)
            if yield_entry == True:
                yield entry

    def closest_matching_log_entries(self, simname, halo_finder, 
        version_name, redshift, dz_tol):
        """
        """
        gen = self.matching_log_entry_generator(
            simname = simname, halo_finder = halo_finder, 
            version_name = version_name, redshift = redshift, dz_tol = dz_tol)
        matching_fnames = [entry.fname for entry in gen]        

        gen2 = self.halo_table_cache.matching_log_entry_generator(simname = simname, 
            halo_finder = halo_finder, version_name = version_name)
        gen3 = self.halo_table_cache.matching_log_entry_generator(
            simname = simname, halo_finder = halo_finder)
        gen4 = self.halo_table_cache.matching_log_entry_generator(simname = simname)


    def add_entry_to_cache_log(self, log_entry, update_ascii = True):
        """
        """

        try:
            assert isinstance(log_entry, HaloTableCacheLogEntry)
        except AssertionError:
            msg = ("You can only add instances of HaloTableCacheLogEntry to the cache log")
            raise TypeError(msg)

        if log_entry.safe_for_cache == False:
            raise InvalidCacheLogEntry(log_entry._cache_safety_message)

        if log_entry in self.log:
            warn("The cache log already contains the entry")
        else:
            self.log.append(log_entry)
            self.log.sort()
            if update_ascii == True:
                self._overwrite_log_ascii(self.log)
                msg = ("The log has been updated on disk and in memory")
                print(msg)
            else:
                msg = ("The log has been updated in memory "
                    "but not on disk because \nthe update_ascii argument is set to False")
                print(msg)

    def remove_entry_from_cache_log(self, simname, halo_finder, version_name, redshift, fname, 
        raise_non_existence_exception = True):
        """
        If the log stores an entry matching the input metadata, the entry will be deleted and 
        the ascii file storing the log will be updated. If there is no match, 
        an exception will be raised according to the value of the input 
        ``raise_non_existence_exception``.  

        Parameters 
        -----------
        simname : string 
            Nickname of the simulation used as a shorthand way to keep track 
            of the halo catalogs in your cache. The simnames processed by Halotools are 
            'bolshoi', 'bolplanck', 'consuelo' and 'multidark'. 

        halo_finder : string 
            Nickname of the halo-finder, e.g., 'rockstar' or 'bdm'. 

        version_name : string 
            Nickname of the version of the halo catalog used to differentiate 
            between the same halo catalog processed in different ways. 

        redshift : string or float  
            Redshift of the halo catalog, rounded to 4 decimal places. 

        fname : string 
            Name of the hdf5 file storing the table of halos. 

        raise_non_existence_exception : bool, optional 
            If True, an exception will be raised if this function is called 
            and there is no matching entry in the log. Default is True. 
        """
        log_entry = HaloTableCacheLogEntry(simname = simname, 
            halo_finder = halo_finder, version_name = version_name, 
            redshift = redshift, fname = fname)

        try:
            self.log.remove(log_entry)
            self._overwrite_log_ascii(self.log)
            msg = ("The log has been updated on disk and in memory")
            print(msg)
        except ValueError:
            if raise_non_existence_exception == False:
                pass
            else:
                msg = ("\nYou requested that the following entry "
                    "be removed from the cache log:\n\n")
                msg += "simname = ``" + str(simname) + "``\n"
                msg += "halo_finder = ``" + str(halo_finder) + "``\n"
                msg += "version_name = ``" + str(version_name) + "``\n"
                msg += "redshift = ``" + str(redshift) + "``\n"
                msg += "fname = ``" + str(fname) + "``\n\n"
                msg += ("This entry does not appear in the log.\n"
                    "If you want to *passively* remove a log entry, \n"
                    "you must call this method again setting "
                    "`raise_non_existence_exception` to False.\n")
                raise HalotoolsError(msg)


    def determine_log_entry_from_fname(self, fname):
        """ Method tries to construct a `~halotools.sim_manager.HaloTableCacheLogEntry` 
        using the metadata that may be stored in the input file. An exception will be raised 
        if the determination is not possible. 

        Parameters 
        -----------
        fname : string 
            Name of the file 

        Returns 
        ---------
        log_entry : `~halotools.sim_manager.HaloTableCacheLogEntry` instance 
        """
        try:
            import h5py
        except ImportError:
            raise HalotoolsError("Must have h5py package installed "
                "to use the determine_log_entry_from_fname method. ")

        if not os.path.isfile(fname):
            msg = ("File does not exist")
            raise IOError(msg)

        if fname[-5:] != '.hdf5':
            msg = ("Can only self-determine the log entry of files with .hdf5 extension")
            raise IOError(msg)

        try:
            f = h5py.File(fname)
            required_set = set(HaloTableCacheLogEntry.required_metadata)
            actual_set = set(f.attrs.keys())
            assert required_set.issubset(actual_set)
        except AssertionError:
            missing_metadata = required_set - actual_set
            msg = ("The hdf5 file is missing the following metadata:\n")
            for elt in missing_metadata:
                msg += "``"+elt + "``\n"
            msg += "\n"
            raise InvalidCacheLogEntry(msg)
        finally:
            try:
                f.close()
            except:
                pass

        f = h5py.File(fname)
        constructor_kwargs = ({key: f.attrs[key] 
            for key in HaloTableCacheLogEntry.log_attributes})
        log_entry = HaloTableCacheLogEntry(**constructor_kwargs)
        f.close()
        return log_entry














        