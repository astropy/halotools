"""
"""
import os
from copy import copy
from astropy.table import Table
from warnings import warn
import numpy as np

from .ptcl_table_cache_log_entry import PtclTableCacheLogEntry

from ..sim_manager import halotools_cache_dirname
from ..custom_exceptions import InvalidCacheLogEntry, HalotoolsError
from ..utils.python_string_comparisons import compare_strings_py23_safe, _passively_decode_string


__all__ = ('PtclTableCache', )


class PtclTableCache(object):
    """ Object providing a collection of particle catalogs for use with Halotools.
    """

    def __init__(self, read_log_from_standard_loc=True, **kwargs):
        self._standard_log_dirname = _passively_decode_string(halotools_cache_dirname)
        try:
            os.makedirs(os.path.dirname(self._standard_log_dirname))
        except OSError:
            pass
        self._standard_log_fname = _passively_decode_string(
                os.path.join(self._standard_log_dirname, 'ptcl_table_cache_log.txt'))

        try:
            self.cache_log_fname = _passively_decode_string(kwargs['cache_log_fname'])
        except KeyError:
            self.cache_log_fname = _passively_decode_string(copy(self._standard_log_fname))
        self._cache_log_fname_exists = os.path.isfile(self.cache_log_fname)

        if read_log_from_standard_loc is True:
            self.log = self.retrieve_log_from_ascii()
        else:
            self.log = []

    def update_log_from_current_ascii(self):
        self.log = self.retrieve_log_from_ascii()

    def _overwrite_log_ascii(self, new_log):
        new_log.sort()
        log_table = self._log_table_from_log(new_log)
        try:
            os.remove(self.cache_log_fname)
        except OSError:
            pass
        finally:
            log_table.write(self.cache_log_fname, format='ascii')

    def _clean_log_of_repeated_entries(self, input_log):
        cleaned_log = list(set(input_log))

        if len(cleaned_log) < len(input_log):
            msg = ("Detected multiple entries of the "
                   "particle table cache log with identical entries.\n"
                   "This is harmless. The log will be now be cleaned.\n")
            warn(msg)
            self._overwrite_log_ascii(cleaned_log)

        return cleaned_log

    def retrieve_log_from_ascii(self):
        """ Read '$HOME/.astropy/cache/halotools/ptcl_table_cache_log.txt',
        clean the log of any repeated entries, sort the log, and return the resulting
        list of `~halotools.sim_manager.PtclTableCacheLogEntry` instances.
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
                log_table = Table.read(_passively_decode_string(self.cache_log_fname), format='ascii')
                assert set(log_table.keys()) == set(
                    PtclTableCacheLogEntry.log_attributes)
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
                {key: entry[key] for key in PtclTableCacheLogEntry.log_attributes})
            result.append(PtclTableCacheLogEntry(**constructor_kwargs))
        return result

    def _log_table_from_log(self, log):
        log_table = self._get_empty_log_table(len(log))
        for ii, entry in enumerate(log):
            for attr in PtclTableCacheLogEntry.log_attributes:
                log_table[attr][ii] = getattr(entry, attr)
        return log_table

    def _get_empty_log_table(self, num_entries=0):
        if num_entries == 0:
            return Table(
                {'simname': [],
                 'redshift': [], 'version_name': [],
                 'fname': []}
                )
        else:
            return Table({'simname': np.zeros(num_entries, dtype=object),
                          'redshift': np.zeros(num_entries, dtype=float),
                          'version_name': np.zeros(num_entries, dtype=object),
                          'fname': np.zeros(num_entries, dtype=object)})

    def matching_log_entry_generator(self, dz_tol=0.0, **kwargs):
        """
        """

        try:
            assert set(kwargs.keys()).issubset(set(PtclTableCacheLogEntry.log_attributes))
        except AssertionError:
            msg = ("\nThe only acceptable keyword arguments to matching_log_entry_generator \n"
                "are elements of PtclTableCacheLogEntry.log_attributes:\n")
            for attr in PtclTableCacheLogEntry.log_attributes:
                msg += "``"+attr+"``, "
            msg = msg[:-2]
            raise KeyError(msg)

        for entry in self.log:
            yield_entry = True
            for key in list(kwargs.keys()):
                if key == 'redshift':
                    requested_redshift = float(kwargs[key])
                    redshift_of_entry = float(getattr(entry, key))
                    yield_entry *= abs(redshift_of_entry - requested_redshift) <= dz_tol
                else:
                    yield_entry *= compare_strings_py23_safe(kwargs[key], getattr(entry, key))
                    # yield_entry *= kwargs[key] == getattr(entry, key)
            if bool(yield_entry) is True:
                yield entry

    def add_entry_to_cache_log(self, log_entry, update_ascii=True):
        """
        """

        try:
            assert isinstance(log_entry, PtclTableCacheLogEntry)
        except AssertionError:
            msg = ("You can only add instances of PtclTableCacheLogEntry to the cache log")
            raise TypeError(msg)

        if log_entry.safe_for_cache is False:
            raise InvalidCacheLogEntry(log_entry._cache_safety_message)

        if log_entry in self.log:
            warn("The cache log already contains the entry")
        else:
            self.log.append(log_entry)
            self.log.sort()
            if update_ascii is True:
                self._overwrite_log_ascii(self.log)

    def remove_entry_from_cache_log(self, simname, version_name,
                                    redshift, fname,
                                    raise_non_existence_exception=True,
                                    update_ascii=True,
                                    delete_corresponding_ptcl_catalog=False):
        """
        If the log stores an entry matching the input metadata, the entry
        will be deleted and the ascii file storing the log will be
        updated. If there is no match, an exception will be raised according
        to the value of the input ``raise_non_existence_exception``.

        Parameters
        -----------
        simname : string
            Nickname of the simulation used as a shorthand way to keep track
            of the particle catalogs in your cache. The simnames processed by Halotools are
            'bolshoi', 'bolplanck', 'consuelo' and 'multidark'.

        version_name : string
            Nickname of the version of the particle catalog.

        redshift : string or float
            Redshift of the particle catalog, rounded to 4 decimal places.

        fname : string
            Name of the hdf5 file storing the table of particles.

        raise_non_existence_exception : bool, optional
            If True, an exception will be raised if this function is called
            and there is no matching entry in the log. Default is True.

        delete_corresponding_ptcl_catalog : bool, optional
            If set to True, when the log entry is deleted,
            the corresponding hdf5 file will be deleted from your disk.
            Default is False.
        """
        ######################################################
        # update_ascii kwarg is for unit-testing only
        # this feature is intentionally hidden from the docstring
        if (update_ascii is False) & (delete_corresponding_ptcl_catalog is True):
            msg = ("\nIf ``delete_corresponding_ptcl_catalog`` is True, \n"
                "``update_ascii`` must also be set to True.\n")
            raise HalotoolsError(msg)
        ######################################################

        log_entry = PtclTableCacheLogEntry(simname=simname,
            version_name=version_name,
            redshift=redshift, fname=fname)

        try:
            self.log.remove(log_entry)

            if update_ascii is True:
                self._overwrite_log_ascii(self.log)
                msg = ("\nThe log has been updated on disk and in memory.\n")
            else:
                msg = ("\nThe log has been updated in memory "
                    "but not on disk because \n"
                    "the update_ascii argument is set to False.\n")

            if delete_corresponding_ptcl_catalog is True:
                try:
                    os.remove(log_entry.fname)
                    msg += (
                        "The corresponding hdf5 file storing the particle catalog "
                        "has also been deleted from disk.\n"
                        )
                except OSError:
                    pass

        except ValueError:
            if raise_non_existence_exception is False:
                pass
            else:
                msg = ("\nYou requested that the following entry "
                    "be removed from the cache log:\n\n")
                msg += "simname = ``" + str(simname) + "``\n"
                msg += "version_name = ``" + str(version_name) + "``\n"
                msg += "redshift = ``" + str(redshift) + "``\n"
                msg += "fname = ``" + str(fname) + "``\n\n"
                msg += ("This entry does not appear in the log.\n"
                    "If you want to *passively* remove a log entry, \n"
                    "you must call this method again setting "
                    "`raise_non_existence_exception` to False.\n")
                raise HalotoolsError(msg)

    def determine_log_entry_from_fname(self, fname, overwrite_fname_metadata=False):
        """ Method tries to construct a `~halotools.sim_manager.PtclTableCacheLogEntry`
        using the metadata that may be stored in the input file. An exception will be raised
        if the determination is not possible.

        Parameters
        -----------
        fname : string
            Name of the file

        Returns
        ---------
        log_entry : `~halotools.sim_manager.PtclTableCacheLogEntry` instance
        """
        try:
            import h5py
        except ImportError:
            raise HalotoolsError(
                "Must have h5py package installed \n"
                "to use the determine_log_entry_from_fname method "
                "of the PtclTableCache class.\n")

        if not os.path.isfile(fname):
            msg = "File does not exist"
            return str(msg)

        if fname[-5:] != '.hdf5':
            msg = "Can only self-determine the log entry of files with .hdf5 extension"
            return str(msg)

        try:
            f = h5py.File(fname, 'r')
            required_set = set(PtclTableCacheLogEntry.required_metadata)
            actual_set = set(f.attrs.keys())
            assert required_set.issubset(actual_set)
        except AssertionError:
            missing_metadata = required_set - actual_set
            msg = ("The hdf5 file is missing the following metadata:\n")
            for elt in missing_metadata:
                msg += "``"+elt + "``\n"
            msg += "\n"
            return str(msg)
        finally:
            try:
                f.close()
            except:
                pass

        f = h5py.File(fname, 'a')
        constructor_kwargs = {}

        # We need to get rid of the byte attributes here to avoid failures
        # later in the definition of PtclTableCacheLogEntry.__lt__
        for key in PtclTableCacheLogEntry.log_attributes:
            try:
                constructor_kwargs[key] = f.attrs[key].decode()
            except AttributeError:
                constructor_kwargs[key] = f.attrs[key]

        if overwrite_fname_metadata is True:
            constructor_kwargs['fname'] = fname
            f.attrs['fname'] = fname
        log_entry = PtclTableCacheLogEntry(**constructor_kwargs)
        f.close()
        return log_entry

    def update_cached_file_location(self, new_fname, old_fname, **kwargs):
        """
        Parameters
        -----------
        new_fname : string
            Name of the new location of the file

        old_fname : string
            Name of the old location of the file
        """

        ######################################################
        # update_ascii kwarg is for unit-testing only
        # this feature is intentionally hidden from the docstring
        try:
            update_ascii = kwargs['update_ascii']
        except KeyError:
            update_ascii = True
        ######################################################

        new_log_entry = self.determine_log_entry_from_fname(
            new_fname, overwrite_fname_metadata=True)

        self.add_entry_to_cache_log(new_log_entry, update_ascii=update_ascii)
        self.remove_entry_from_cache_log(
            simname=new_log_entry.simname,
            version_name=new_log_entry.version_name,
            redshift=new_log_entry.redshift,
            fname=old_fname,
            raise_non_existence_exception=False,
            update_ascii=update_ascii,
            delete_corresponding_ptcl_catalog=False)
