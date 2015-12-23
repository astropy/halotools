import os
from copy import copy
from astropy.config.paths import _find_home
from astropy.table import Table
from log_entry import HaloTableCacheLogEntry
from warnings import warn

from custom_exceptions import InvalidCacheLogEntry

__all__ = ('HaloTableCache', )

class HaloTableCache(object):

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

    def add_entry_to_cache_log(self, log_entry, update_ascii = True):

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















        