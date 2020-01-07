"""
"""
import os
from astropy.table import Table
import numpy as np
from warnings import warn

from .halo_table_cache_log_entry import get_redshift_string
from ..utils.python_string_comparisons import _passively_decode_string, compare_strings_py23_safe

try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False
    warn("Most of the functionality of the "
        "sim_manager sub-package requires h5py to be installed,\n"
        "which can be accomplished either with pip or conda. ")

__all__ = ('PtclTableCacheLogEntry', )


class PtclTableCacheLogEntry(object):
    """ Object serving as an entry in the `~halotools.sim_manager.PtclTableCache`.
    """

    log_attributes = ['simname', 'version_name', 'redshift', 'fname']
    required_metadata = ['Lbox', 'particle_mass']
    required_metadata.extend(log_attributes)

    def __init__(self, simname, version_name, redshift, fname):
        """
        Parameters
        -----------
        simname : string
            Nickname of the simulation used as a shorthand way to keep track
            of the particle catalogs in your cache. The simnames processed
            by Halotools are 'bolshoi', 'bolplanck', 'consuelo' and
            'multidark'.

        version_name : string
            Nickname of the version of the particle catalog.

        redshift : string or float
            Redshift of the particle catalog, rounded to 4 decimal places.

        fname : string
            Name of the hdf5 file storing the table of particles.

        Notes
        ------
        This class overrides the python built-in comparison functions __eq__, __lt__, etc.
        Equality holds only if all of the four constructor inputs are equal.
        Two class instances are compared by using a dictionary order
        defined by the same sequence as the constructor input positional arguments.

        See also
        ----------
        `~halotools.sim_manager.tests.TestPtclTableCacheLogEntry`.
        """
        msg = ("\nMust have the h5py package installed \n"
               "to instantiate the PtclTableCacheLogEntry class.\n")
        assert _HAS_H5PY, msg

        self.simname = _passively_decode_string(simname)
        self.version_name = _passively_decode_string(version_name)
        self.redshift = _passively_decode_string(get_redshift_string(redshift))
        self.fname = _passively_decode_string(fname)

    def __eq__(self, other):
        if type(other) is type(self):
            comparison_generator = (getattr(self, attr) == getattr(other, attr)
                for attr in PtclTableCacheLogEntry.log_attributes)
            return False not in tuple(comparison_generator)
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if type(other) == type(self):
            return self._key < other._key
        else:
            msg = ("\nYou cannot compare the order of a PtclTableCacheLogEntry instance \n"
                "to an object of a different type.\n")
            raise TypeError(msg)

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __gt__(self, other):
        return not self.__le__(other)

    def __ge__(self, other):
        return self.__eq__(other) or self.__gt__(other)

    def __hash__(self):
        return hash(self._key)

    def __str__(self):

        msg = "("
        for attr in PtclTableCacheLogEntry.log_attributes:
            msg += "'"+getattr(self, attr)+"', "
        msg = msg[0:-2] + ")"
        return msg

    @property
    def _key(self):
        """ Private property needed for the python built-in ``set``
        function to operate properly with the custom-defined
        comparison methods.
        """
        return (type(self), self.simname,
            self.version_name,
            self.redshift, self.fname)

    @property
    def safe_for_cache(self):
        """ Boolean determining whether the
        `~halotools.sim_manager.PtclTableCacheLogEntry` instance stores a valid
        particle catalog that can safely be added to the cache for future use.
        `safe_for_cache` is implemented as a property method, so that each request
        performs all the checks from scratch. A log entry is considered valid
        if it passes the following tests:

        1. The file exists.

        2. The filename has .hdf5 extension.

        3. The hdf5 file has ``simname``, ``version_name``, ``redshift``, ``fname``, ``Lbox`` and ``particle_mass`` metadata attributes.

        4. Each value in the above metadata is consistent with the corresponding value bound to the `~halotools.sim_manager.PtclTableCacheLogEntry` instance.

        5. The particle table data can be read in using the `~astropy.table.Table.read` method of the `~astropy.table.Table` class.

        6. The particle table has the following columns ``x``, ``y``, ``z``.

        7. Values for each of the ``x``, ``y``, ``z`` columns are all in the range [0, Lbox].

        Note in particular that `safe_for_cache` performs no checks whatsoever concerning
        the other log entries that may or may not be stored in the cache. Such checks are
        the responsibility of the `~halotools.sim_manager.PtclTableCacheCache` class.

        """
        self._cache_safety_message = "The particle catalog is safe to add to the cache log."

        message_preamble = ("The particle catalog and/or its associated metadata "
            "fail the following tests:\n\n")

        msg, num_failures = '', 0
        msg, num_failures = self._verify_file_exists(msg, num_failures)

        if num_failures > 0:
            self._cache_safety_message = message_preamble + msg
            self._num_failures = num_failures
            return False
        else:

            verification_sequence = ('_verify_h5py_extension',
                                     '_verify_hdf5_has_complete_metadata',
                                     '_verify_metadata_consistency',
                                     '_verify_table_read',
                                     '_verify_has_required_data_columns',
                                     '_verify_all_positions_inside_box')

            for verification_function in verification_sequence:
                func = getattr(self, verification_function)
                msg, num_failures = func(msg, num_failures)

            if num_failures > 0:
                self._cache_safety_message = message_preamble + msg

            self._num_failures = num_failures
            return num_failures == 0

    def _verify_table_read(self, msg, num_failures):
        """ Enforce that the data can be read using the usual Astropy syntax
        """
        try:
            data = Table.read(_passively_decode_string(self.fname), path='data')
        except:
            num_failures += 1
            msg += (str(num_failures)+". The hdf5 file must be readable with "
                "Astropy \nusing the following syntax:\n\n"
                ">>> ptcl_data = Table.read(fname, path='data')\n\n")
            pass
        return msg, num_failures

    def _verify_metadata_consistency(self, msg, num_failures):
        """ Enforce that the hdf5 metadata agrees with the
        values in the log entry.

        Note that we actually accept floats for the redshift hdf5 metadata,
        even though this should technically be a string.
        """

        try:
            f = h5py.File(self.fname, 'r')

            for key in PtclTableCacheLogEntry.log_attributes:
                ptcl_log_attr = getattr(self, key)
                try:

                    metadata = f.attrs[key]
                    if key != 'redshift':
                        assert compare_strings_py23_safe(metadata, ptcl_log_attr)
                    else:
                        metadata = float(get_redshift_string(metadata))
                        assert metadata == float(ptcl_log_attr)

                except AssertionError:

                    num_failures += 1
                    msg += (
                        str(num_failures)+". The hdf5 file has metadata "
                        "``"+key+"`` = "+str(metadata) +
                        ".\nThis does not match the " +
                        str(ptcl_log_attr)+" value in the log entry.\n\n"
                        )
                except KeyError:

                    pass
        except IOError:

            pass

        finally:
            # The file may or not still be open
            try:
                f.close()
            except:
                pass

        return msg, num_failures

    def _verify_has_required_data_columns(self, msg, num_failures):
        """
        """
        try:
            data = Table.read(_passively_decode_string(self.fname), path='data')
            keys = list(data.keys())
            try:
                assert 'x' in keys
                assert 'y' in keys
                assert 'z' in keys
            except AssertionError:
                num_failures += 1
                msg += (str(num_failures)+". The particle table "
                    "must at a minimum have the following columns:\n"
                    "``x``, ``y``, ``z``.\n\n"
                        )
        except:
            pass

        return msg, num_failures

    def _verify_all_positions_inside_box(self, msg, num_failures):
        """
        """
        try:
            data = Table.read(_passively_decode_string(self.fname), path='data')
            f = h5py.File(self.fname, 'r')
            Lbox = np.empty(3)
            Lbox[:] = f.attrs['Lbox']

            f.close()
            try:
                x = data['x']
                y = data['y']
                z = data['z']

                assert np.all(x >= 0)
                assert np.all(x <= Lbox[0])
                assert np.all(y >= 0)
                assert np.all(y <= Lbox[1])
                assert np.all(z >= 0)
                assert np.all(z <= Lbox[2])

            except AssertionError:
                num_failures += 1
                msg += (str(num_failures)+". All values of the "
                    "``x``, ``y``, ``z`` columns\n"
                    "must be bounded by [0, Lbox].\n\n"
                        )
        except:
            pass

        return msg, num_failures

    def _verify_file_exists(self, msg, num_failures):
        """
        """

        if os.path.isfile(self.fname):
            pass
        else:
            num_failures += 1
            msg += str(num_failures)+". The input filename does not exist.\n\n"
        return msg, num_failures

    def _verify_h5py_extension(self, msg, num_failures):
        """
        """

        if self.fname[-5:] == '.hdf5':
            pass
        else:
            num_failures += 1
            msg += str(num_failures) + ". The input file must have '.hdf5' extension.\n\n"
        return msg, num_failures

    def _verify_hdf5_has_complete_metadata(self, msg, num_failures):
        """
        """

        try:
            f = h5py.File(self.fname, 'r')
            required_set = set(PtclTableCacheLogEntry.required_metadata)
            actual_set = set(f.attrs.keys())

            if required_set.issubset(actual_set):
                pass
            else:
                missing_metadata = required_set - actual_set
                num_failures += 1
                msg += (str(num_failures) +
                    ". The hdf5 file is missing the following metadata:\n")
                for elt in missing_metadata:
                    msg += "``"+elt + "``\n"
                msg += "\n"

        except IOError:
            num_failures += 1
            msg += (str(num_failures) +
                ". Attempting to access the hdf5 metadata raised an exception.\n\n")
            pass

        finally:
            # The file may or not still be open
            try:
                f.close()
            except:
                pass

        return msg, num_failures
