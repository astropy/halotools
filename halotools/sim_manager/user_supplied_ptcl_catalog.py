""" Module containing the UserSuppliedPtclCatalog class. 
"""

import numpy as np
import os, sys, urllib2, fnmatch
from warnings import warn 
import datetime 

from astropy import cosmology
from astropy import units as u
from astropy.table import Table

try:
    import h5py
except ImportError:
    warn("Most of the functionality of the sim_manager "
        "sub-package requires h5py to be installed,\n"
        "which can be accomplished either with pip or conda")

from .ptcl_table_cache import PtclTableCache 
from .ptcl_table_cache_log_entry import PtclTableCacheLogEntry
from .halo_table_cache_log_entry import get_redshift_string

from ..utils.array_utils import custom_len, convert_to_ndarray
from ..custom_exceptions import HalotoolsError

__all__ = ('UserSuppliedPtclCatalog', )

class UserSuppliedPtclCatalog(object):
    """ Class used to transform a user-provided particle catalog 
    into the standard form recognized by Halotools. 
    
    """
    def __init__(self, **kwargs):

        ptcl_table_dict, metadata_dict = self._parse_constructor_kwargs(**kwargs)
        self.ptcl_table = Table(ptcl_table_dict)

        self._test_metadata_dict(**metadata_dict)
        for key, value in metadata_dict.iteritems():
            setattr(self, key, value)

        self._passively_bind_ptcl_table(**kwargs)


    def _parse_constructor_kwargs(self, **kwargs):
        try:
            x = kwargs['x']
            assert type(x) is np.ndarray 
            y = kwargs['y']
            assert type(y) is np.ndarray 
            z = kwargs['z']
            assert type(z) is np.ndarray 

            Nptcls = custom_len(x)
            assert Nptcls >= 1e4
            assert Nptcls == len(y)
            assert Nptcls == len(z)
        except KeyError, AssertionError:
            msg = ("\nThe UserSuppliedHaloCatalog requires ``x``, ``y`` and ``z`` keyword arguments,\n "
                "each of which must store an ndarray of the same length Nptcls >= 1e4.\n")
            raise HalotoolsError(msg)

        ptcl_table_dict = (
            {key: kwargs[key] for key in kwargs 
            if (type(kwargs[key]) is np.ndarray) 
            and (custom_len(kwargs[key]) == Nptcls)}
            )

        metadata_dict = (
            {key: kwargs[key] for key in kwargs if key not in ptcl_table_dict}
            )

    	return ptcl_table_dict, metadata_dict 

    def _test_metadata_dict(self, **metadata_dict):
        try:
            assert 'Lbox' in metadata_dict
            assert custom_len(metadata_dict['Lbox']) == 1
            assert 'particle_mass' in metadata_dict
            assert custom_len(metadata_dict['particle_mass']) == 1
            assert 'redshift' in metadata_dict
        except AssertionError:
            msg = ("\nThe UserSuppliedPtclCatalog requires "
                "keyword arguments ``Lbox``, ``particle_mass`` and ``redshift``\n"
                "storing scalars that will be interpreted as metadata about the particle catalog.\n")
            raise HalotoolsError(msg)

        Lbox = metadata_dict['Lbox']
        assert Lbox > 0, "``Lbox`` must be a positive number"

        try:
            x, y, z = (
                self.ptcl_table['x'], 
                self.ptcl_table['x'], 
                self.ptcl_table['z']
                )
            assert np.all(x >= 0)
            assert np.all(x <= Lbox)
            assert np.all(y >= 0)
            assert np.all(y <= Lbox)
            assert np.all(z >= 0)
            assert np.all(z <= Lbox)
        except AssertionError:
            msg = ("The ``x``, ``y`` and ``z`` columns must only store arrays\n"
                "that are bound by 0 and the input ``Lbox``. \n")
            raise HalotoolsError(msg)

        redshift = metadata_dict['redshift']
        try:
            assert type(redshift) == float
        except AssertionError:
            msg = ("\nThe ``redshift`` metadata must be a float.\n")
            raise HalotoolsError(msg)


    def add_ptclcat_to_cache(self, 
        fname, simname, version_name, processing_notes, 
        overwrite = False, **additional_metadata):

        try:
            import h5py 
        except ImportError:
            msg = ("\nYou must have h5py installed if you want to \n"
                "store your catalog in the Halotools cache. \n")
            raise HalotoolsError(msg)

        ############################################################
        ## Perform some consistency checks in the fname
        if (os.path.isfile(fname)) & (overwrite == False):
            msg = ("\nYou attempted to store your particle catalog "
                "in the following location: \n\n" + str(fname) + 
                "\n\nThis path points to an existing file. \n"
                "Either choose a different fname or set ``overwrite`` to True.\n")
            raise HalotoolsError(msg)

        try:
            dirname = os.path.dirname(fname)
            assert os.path.exists(dirname)
        except:
            msg = ("\nThe directory you are trying to store the file does not exist. \n")
            raise HalotoolsError(msg)

        if fname[-5:] != '.hdf5':
            msg = ("\nThe fname must end with an ``.hdf5`` extension.\n")
            raise HalotoolsError(msg)

        ############################################################
        ## Perform consistency checks on the remaining log entry attributes
        try:
            _ = str(simname)
            _ = str(version_name)
            _ = str(processing_notes)
        except:
            msg = ("\nThe input ``simname``, ``version_name`` "
                "and ``processing_notes``\nmust all be strings.")
            raise HalotoolsError(msg)

        for key, value in additional_metadata.iteritems():
            try:
                _ = str(value)
            except:
                msg = ("\nIf you use ``additional_metadata`` keyword arguments \n"
                    "to provide supplementary metadata about your catalog, \n"
                    "all such metadata will be bound to the hdf5 file in the "
                    "format of a string.\nHowever, the value you bound to the "
                    "``"+key+"`` keyword is not representable as a string.\n")
                raise HalotoolsError(msg)

        ############################################################
        ## Now write the file to disk and add the appropriate metadata 

        self.ptcl_table.write(fname, path='data', overwrite = overwrite)

        f = h5py.File(fname)

        redshift_string = str(get_redshift_string(self.redshift))

        f.attrs.create('simname', str(simname))
        f.attrs.create('version_name', str(version_name))
        f.attrs.create('redshift', redshift_string)
        f.attrs.create('fname', str(fname))

        f.attrs.create('Lbox', self.Lbox)
        f.attrs.create('particle_mass', self.particle_mass)

        time_right_now = str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        f.attrs.create('time_catalog_was_originally_cached', time_right_now)

        f.attrs.create('processing_notes', str(processing_notes))

        for key, value in additional_metadata.iteritems():
            f.attrs.create(key, str(value))

        f.close()

        ############################################################
        # Now that the file is on disk, add it to the cache
        cache = PtclTableCache()

        log_entry = PtclTableCacheLogEntry(
            simname = simname, version_name = version_name, 
            redshift = self.redshift, fname = fname)

        cache.add_entry_to_cache_log(log_entry, update_ascii = True)
        self.log_entry = log_entry
















