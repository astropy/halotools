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
    	pass

    def add_ptclcat_to_cache(self, 
        fname, simname, version_name, processing_notes, 
        overwrite = False, **additional_metadata):
    	pass















