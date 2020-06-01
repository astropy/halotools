"""
"""
import os
from astropy.table import Table
import numpy as np
from warnings import warn
from ..utils.python_string_comparisons import _passively_decode_string, compare_strings_py23_safe

try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False
    warn("Most of the functionality of the "
        "sim_manager sub-package requires h5py to be installed,\n"
        "which can be accomplished either with pip or conda. ")


__all__ = ('HaloTableCacheLogEntry', )


def get_redshift_string(redshift):
    return str('{0:.4f}'.format(float(redshift)))


class HaloTableCacheLogEntry(object):
    """ Object serving as an entry in the `~halotools.sim_manager.HaloTableCache`.
    """

    log_attributes = ['simname', 'halo_finder', 'version_name', 'redshift', 'fname']
    required_metadata = ['Lbox', 'particle_mass']
    required_metadata.extend(log_attributes)

    def __init__(self, simname, halo_finder, version_name, redshift, fname):
        """
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

        Notes
        ------
        This class overrides the python built-in comparison functions __eq__, __lt__, etc.
        Equality holds only if all of the five constructor inputs are equal.
        Two class instances are compared by using a dictionary order
        defined by the same sequence as the constructor input positional arguments.

        See also
        ----------
        `~halotools.sim_manager.tests.TestHaloTableCacheLogEntry`.
        """
        self.simname = _passively_decode_string(simname)
        self.halo_finder = _passively_decode_string(halo_finder)
        self.version_name = _passively_decode_string(version_name)
        self.redshift = _passively_decode_string(get_redshift_string(redshift))
        self.fname = _passively_decode_string(fname)

    def __eq__(self, other):
        if type(other) is type(self):
            comparison_generator = (getattr(self, attr) == getattr(other, attr)
                for attr in HaloTableCacheLogEntry.log_attributes)
            return False not in tuple(comparison_generator)
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if type(other) == type(self):
            return self._key < other._key
        else:
            msg = ("\nYou cannot compare the order of a HaloTableCacheLogEntry instance \n"
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
        for attr in HaloTableCacheLogEntry.log_attributes:
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
            self.halo_finder, self.version_name,
            self.redshift, self.fname)

    @property
    def safe_for_cache(self):
        """ Boolean determining whether the
        `~halotools.sim_manager.HaloTableCacheLogEntry` instance stores a valid
        halo catalog that can safely be added to the cache for future use.
        `safe_for_cache` is implemented as a property method, so that each request
        performs all the checks from scratch. A log entry is considered valid
        if it passes the following tests:

        1. The file exists.

        2. The filename has .hdf5 extension.

        3. The hdf5 file has ``simname``, ``halo_finder``, ``version_name``, ``redshift``, ``fname``, ``Lbox`` and ``particle_mass`` metadata attributes.

        4. Each value in the above metadata is consistent with the corresponding value bound to the `~halotools.sim_manager.HaloTableCacheLogEntry` instance.

        5. The halo table data can be read in using the `~astropy.table.Table.read` method of the `~astropy.table.Table` class.

        6. The halo table has the following columns ``halo_id``, ``halo_x``, ``halo_y``, ``halo_z``, plus at least one additional column storing a mass-like variable.

        7. The name of each column of the halo table begins with the substring ``halo_``.

        8. Values for each of the ``halo_x``, ``halo_y``, ``halo_z`` columns are all in the range [0, Lbox].

        9. The ``halo_id`` column stores a set of unique integers.

        Note in particular that `safe_for_cache` performs no checks whatsoever concerning
        the log other entries that may or may not be stored in the cache. Such checks are
        the responsibility of the `~halotools.sim_manager.HaloTableCache` class.

        """
        msg = ("\nCannot determine whether an hdf5 file "
            "is safe_for_cache without h5py installed.\n")
        assert _HAS_H5PY, msg

        self._cache_safety_message = "The halo catalog is safe to add to the cache log."

        message_preamble = ("The halo catalog and/or its associated metadata fail the following tests:\n\n")

        msg, num_failures = self._verify_file_exists()

        if num_failures > 0:
            self._cache_safety_message = message_preamble + msg
            self._num_failures = num_failures
            return False
        else:

            tmp_msg, num_failures = self._verify_h5py_extension(num_failures)
            msg += tmp_msg
            tmp_msg, num_failures = self._verify_hdf5_has_complete_metadata(num_failures)
            msg += tmp_msg
            tmp_msg, num_failures = self._verify_metadata_consistency(num_failures)
            msg += tmp_msg
            tmp_msg, num_failures, halo_table = self._verify_table_read(num_failures)
            msg += tmp_msg
            tmp_msg, num_failures = self._verify_has_required_data_columns(halo_table, num_failures)
            msg += tmp_msg
            tmp_msg, num_failures = self._verify_all_keys_begin_with_halo(halo_table, num_failures)
            msg += tmp_msg
            tmp_msg, num_failures = self._verify_all_positions_inside_box(halo_table, num_failures)
            msg += tmp_msg
            tmp_msg, num_failures = self._verify_halo_ids_are_unique(halo_table, num_failures)
            msg += tmp_msg
            tmp_msg, num_failures = self._verify_halo_rvir_mpc_units(halo_table, num_failures)
            msg += tmp_msg

            if num_failures > 0:
                self._cache_safety_message = message_preamble + msg

            self._num_failures = num_failures
            return num_failures == 0

    def _verify_table_read(self, num_failures):
        """ Enforce that the data can be read using the usual Astropy syntax
        """
        msg = ''

        try:
            halo_table = Table.read(_passively_decode_string(self.fname), path='data')
        except:
            num_failures += 1
            msg = (str(num_failures)+". The hdf5 file must be readable with "
                "Astropy \nusing the following syntax:\n\n"
                ">>> halo_data = Table.read(fname, path='data')\n\n")
            halo_table = Table()
        return msg, num_failures, halo_table

    def _verify_metadata_consistency(self, num_failures):
        """ Enforce that the hdf5 metadata agrees with the
        values in the log entry.

        Note that we actually accept floats for the redshift hdf5 metadata,
        even though this should technically be a string.
        """
        msg = ''

        try:
            import h5py
            f = h5py.File(self.fname, 'r')

            for key in HaloTableCacheLogEntry.log_attributes:
                try:
                    metadata = f.attrs[key]
                    if key != 'redshift':
                        assert compare_strings_py23_safe(metadata, getattr(self, key))
                    else:
                        metadata = float(get_redshift_string(metadata))
                        assert metadata == float(getattr(self, key))

                except AssertionError:

                    num_failures += 1
                    msg = (
                        str(num_failures)+". The hdf5 file has metadata "
                        "``"+key+"`` = "+str(metadata) +
                        ".\nThis does not match the " +
                        str(getattr(self, key))+" value in the log entry.\n\n"
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

    def _verify_all_keys_begin_with_halo(self, halo_table, num_failures):
        """
        """
        msg = ''

        try:
            for key in list(halo_table.keys()):
                try:
                    assert key[0:5] == 'halo_'
                except AssertionError:
                    num_failures += 1
                    msg = (str(num_failures)+". The column names "
                        "of all data in the halo catalog\n"
                        "must begin with the following five characters: `halo_`.\n\n")
        except:
            pass

        return msg, num_failures

    def _verify_has_required_data_columns(self, halo_table, num_failures):
        """
        """
        msg = ''

        try:
            keys = list(halo_table.keys())
            try:
                assert 'halo_x' in keys
                assert 'halo_y' in keys
                assert 'halo_z' in keys
                assert 'halo_id' in keys
                assert len(keys) >= 5
            except AssertionError:
                num_failures += 1
                msg = (str(num_failures)+". The halo table "
                    "must at a minimum have the following columns:\n"
                    "``halo_id``, ``halo_x``, ``halo_y``, ``halo_z``,\n"
                    "plus at least one additional column storing a mass-like variable.\n\n"
                       )
        except:
            pass

        return msg, num_failures

    def _verify_all_positions_inside_box(self, halo_table, num_failures):
        """
        """
        msg = ''

        try:
            f = h5py.File(self.fname, 'r')
            Lbox = f.attrs['Lbox']
            f.close()
            try:
                halo_x = halo_table['halo_x'].data
                halo_y = halo_table['halo_y'].data
                halo_z = halo_table['halo_z'].data

                assert np.all(halo_x >= 0)
                assert np.all(halo_x <= Lbox)
                assert np.all(halo_y >= 0)
                assert np.all(halo_y <= Lbox)
                assert np.all(halo_z >= 0)
                assert np.all(halo_z <= Lbox)

            except AssertionError:
                num_failures += 1
                msg = (str(num_failures)+". All values of the "
                    "``halo_x``, ``halo_y``, ``halo_z`` columns\n"
                    "must be bounded by [0, Lbox].\n\n"
                       )
        except:
            pass

        return msg, num_failures

    def _verify_halo_ids_are_unique(self, halo_table, num_failures):
        """
        """
        msg = ''

        try:
            try:
                halo_id = halo_table['halo_id'].data
                assert halo_id.dtype.str[1] in ('i', 'u')
                assert len(halo_id) == len(set(halo_id))
            except AssertionError:
                num_failures += 1
                msg = (str(num_failures)+". The ``halo_id`` column "
                    "must contain a unique set of integers.\n\n"
                       )
        except:
            pass

        return msg, num_failures

    def _verify_file_exists(self):
        """
        """
        msg = ''
        num_failures = 0

        if os.path.isfile(self.fname):
            pass
        else:
            num_failures += 1
            msg = str(num_failures)+". The input filename does not exist.\n\n"
        return msg, num_failures

    def _verify_h5py_extension(self, num_failures):
        """
        """
        msg = ''

        if self.fname[-5:] == '.hdf5':
            pass
        else:
            num_failures += 1
            msg = str(num_failures) + ". The input file must have '.hdf5' extension.\n\n"
        return msg, num_failures

    def _verify_halo_rvir_mpc_units(self, halo_table, num_failures):
        """ Require that all values stored in the halo_rvir column
        are less than 50, a crude way to ensure that units are not kpc.
        """
        msg = ''

        try:
            halo_rvir = halo_table['halo_rvir']
            assert np.all(halo_rvir.data < 50)
        except AssertionError:
            num_failures += 1
            msg = (str(num_failures)+". All values of the "
                "``halo_rvir`` column\n"
                "must be less than 50, crudely ensuring you used Mpc/h units.\n\n"
                   )
        except KeyError:
            pass

        return msg, num_failures

    def _verify_hdf5_has_complete_metadata(self, num_failures):
        """
        """
        msg = ''

        try:
            f = h5py.File(self.fname, 'r')
            required_set = set(HaloTableCacheLogEntry.required_metadata)
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
