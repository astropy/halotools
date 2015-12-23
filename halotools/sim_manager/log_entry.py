from collections import OrderedDict
import os

from ..custom_exceptions import InvalidCacheLogEntry

__all__ = ('HaloTableCacheLogEntry', )

def get_redshift_string(redshift):
    return str('{0:.4f}'.format(float(redshift)))

class HaloTableCacheLogEntry(object):

    import h5py
    log_attributes = ['simname', 'halo_finder', 'version_name', 'redshift', 'fname']
    required_metadata = ['Lbox', 'particle_mass']
    required_metadata.extend(log_attributes)

    def __init__(self, simname, halo_finder, version_name, redshift, fname):
        """
        """
        self.simname = simname
        self.halo_finder = halo_finder
        self.version_name = version_name
        self.redshift = get_redshift_string(redshift)
        self.fname = fname
    
    def __eq__(self, other):
        if isinstance(other, self.__class__):
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
    def _order(self):
        """ Private property used to define how log entries are ordered
        """
        return OrderedDict(
            (attrname, getattr(self, attrname)) 
            for attrname in HaloTableCacheLogEntry.log_attributes)

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
        """ This will do all the following checks:

        1. File exists 

        2. File has .hdf5 extension

        2. Metadata of hdf5 file is complete 

        3. Metadata of hdf5 file is consistent with HaloTableCacheLogEntry attributes

        4. Keys all begin with 'halo_'

        5. Positions are all in the range [0, Lbox]

        6. halo_id is a set of unique integers 

        7. There exists some mass-like variable

        """
        self._cache_safety_message = "The HaloTableCacheLogEntry is safe to add to the log"

        message_preamble = ("The HaloTableCacheLogEntry is not safe "
            "to add to the log\nfor the following reasons:\n\n")

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
                '_verify_all_keys_begin_with_halo', 
                '_verify_all_positions_inside_box', 
                '_verify_halo_ids_are_unique', 
                '_verify_exists_some_mass_like_variable')

            for verification_function in verification_sequence:
                func = getattr(self, verification_function)
                msg, num_failures = func(msg, num_failures)
            
            if num_failures > 0: self._cache_safety_message = message_preamble + msg
                
            self._num_failures = num_failures
            return num_failures == 0

    def _verify_metadata_consistency(self, msg, num_failures):
        """
        """
        try:
            f = h5py.File(self.fname)
            for key in HaloTableCacheLogEntry.log_attributes:
                try:
                    metadata = f.attr[key]
                    assert metadata == getattr(self, key)
                except AssertionError:
                    num_failures += 1
                    msg += (
                        str(num_failures)+". The ``"+key+"`` metadata of the hdf5 file = "+str(metadata)+
                        "does not match the "+str(getattr(self, key))+" value in the log entry.\n\n"
                        )
                except KeyError:
                    pass
        except:
            pass

        finally:
            # The file may or not still be open
            try:
                f.close()
            except:
                pass

        return msg, num_failures


    def _verify_all_keys_begin_with_halo(self, msg, num_failures):
        return msg, num_failures 

    def _verify_all_positions_inside_box(self, msg, num_failures):
        return msg, num_failures 

    def _verify_halo_ids_are_unique(self, msg, num_failures):
        return msg, num_failures 

    def _verify_exists_some_mass_like_variable(self, msg, num_failures):
        return msg, num_failures 

    def _verify_file_exists(self, msg, num_failures):

        if os.path.isfile(self.fname):
            pass
        else:
            num_failures += 1
            msg += str(num_failures)+". The input filename does not exist.\n\n"
        return msg, num_failures

    def _verify_h5py_extension(self, msg, num_failures):

        if self.fname[-5:]=='.hdf5':
            pass
        else:
            num_failures += 1
            msg += str(num_failures) + ". The input file must have '.hdf5' extension.\n\n"
        return msg, num_failures

    def _verify_hdf5_has_complete_metadata(self, msg, num_failures):

        try:
            f = h5py.File(self.fname)
            required_set = set(HaloTableCacheLogEntry.required_metadata)
            actual_set = set(f.attrs.keys())

            if required_set.issubset(actual_set):
                pass
            else:
                missing_metadata = required_set - actual_set
                num_failures += 1
                msg += (str(num_failures) + ". The hdf5 file is missing "
                    "the following metadata:\n")
                for elt in missing_metadata:
                    msg += "``"+elt + "``\n"
                msg += "\n"

        except:
            num_failures += 1
            msg += (str(num_failures) + ". Attempting to access the hdf5 metadata "
                "raised an exception.\n\n")
            pass

        finally:
            # The file may or not still be open
            try:
                f.close()
            except:
                pass

        return msg, num_failures

    def determine_log_entry_from_fname(self, fname):

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
            f.close()

        try:
            f = h5py.File(fname)
            required_set = set(HaloTableCacheLogEntry.required_metadata)
            for metadata_key in required_metadata - {'redshift'}:
                metadata = f.attrs[metadata_key]
                if type(metadata) not in (str, unicode):
                    msg = ("The metadata bound to ``"+metadata_key+"`` must be a string or unicode.\n"
                        "You will need to fix the metadata of the hdf5 file in order to proceed.\n"
                        "To overwrite the metadata of an hdf5 file, "
                        "you can use the following syntax:\n\n"
                        ">>> f = h5py.File(fname) \n"
                        ">>> f.attrs[metadata_key] = metadata_value \n"
                        ">>> f.close() \n")
                    raise InvalidCacheLogEntry(msg)
            redshift = f.attrs['redshift']
            if type(redshift) not in (str, unicode):
                msg = ("The ``redshift`` metadata must be a string.\n"
                    "You will need to fix the metadata of the hdf5 file in order to proceed.\n"
                    "To translate a float to the required format, "
                    "you can use the following syntax:\n\n"
                    ">>> redshift_string = str('{0:.4f}'.format(float(redshift)))\n\n"
                    "To overwrite the metadata of an hdf5 file, "
                    "you can use the following syntax:\n\n"
                    ">>> f = h5py.File(fname) \n"
                    ">>> f.attrs[metadata_key] = metadata_value \n"
                    ">>> f.close() \n")
                raise InvalidCacheLogEntry(msg)
        finally:
            f.close()

        f = h5py.File(fname)
        constructor_kwargs = ({key: f.attrs[key] 
            for key in HaloTableCacheLogEntry.required_metadata})
        log_entry = HaloTableCacheLogEntry(**constructor_kwargs)
        f.close()
        return log_entry





















