import numpy as np
import os, sys, urllib2, fnmatch
from warnings import warn 

from copy import deepcopy 
from astropy import cosmology
from astropy import units as u
from astropy.table import Table

try:
    import h5py
except ImportError:
    warn("Most of the functionality of the sim_manager sub-package requires h5py to be installed,\n"
        "which can be accomplished either with pip or conda")

from . import sim_defaults, catalog_manager, manipulate_cache_log

from ..utils.array_utils import custom_len, convert_to_ndarray
from ..custom_exceptions import HalotoolsError


__all__ = ('OverhauledHaloCatalog', )


class OverhauledHaloCatalog(object):
    """
    Container class for halo catalogs and particle data.  
    """

    def __init__(self, preload_halo_table = False, **kwargs):
        """
        """

        try:
            import h5py
        except ImportError:
            raise HalotoolsError("Must have h5py package installed "
                "to use OverhauledHaloCatalog objects")

        self.fname_halo_table = self._retrieve_fname_halo_table(**kwargs)

        if preload_halo_table is True:
            self._load_halo_table()

        self._bind_metadata()

    def _retrieve_fname_halo_table(self, **kwargs):
        """
        """
        if 'fname' in kwargs:
            fname = kwargs['fname']
            del kwargs['fname']
            return (
                manipulate_cache_log.return_halo_table_fname_after_verification(
                    fname, **kwargs)
                )
        else:
            return (
                manipulate_cache_log.return_halo_table_fname_from_simname_inputs(**kwargs)
                )


    @property 
    def halo_table(self):
        """
        `~astropy.table.Table` object storing a catalog of dark matter halos. 
        """
        if not hasattr(self, '_halo_table'):
            self._load_halo_table()
        return self._halo_table

    def _load_halo_table(self):
        """ Retrieve the halo_table from ``self.fname_halo_table`` and update the cache log. 
        """
        self._halo_table = Table.read(self.fname_halo_table, path='data')

    def _bind_metadata(self):
        """ Create convenience bindings of all metadata to the `HaloCatalog` instance. 
        """
        f = h5py.File(self.fname_halo_table)
        for attr_key in f.attrs.keys():
            setattr(self, attr_key, f.attrs[attr_key])
        f.close()

    def store_halocat_in_cache(self, fname_halo_table, overwrite = False, 
        neglect_supplementary_metadata = False, store_ptcl_table = True, **kwargs):
        """ 
        Parameters 
        ------------
        fname_halo_table : string 
            String providing the absolute path to the halo table. 

        overwrite : bool, optional 

        store_ptcl_table : bool, optional 

        neglect_supplementary_metadata : bool, optional 
        """
        pass





