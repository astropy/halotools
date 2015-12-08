import numpy as np
import os, sys, urllib2, fnmatch
from warnings import warn 

from astropy import cosmology
from astropy import units as u
from astropy.table import Table

try:
    import h5py
except ImportError:
    warn("Most of the functionality of the sim_manager sub-package requires h5py to be installed,\n"
        "which can be accomplished either with pip or conda")

from . import sim_defaults, catalog_manager

from ..utils.array_utils import custom_len, convert_to_ndarray
from ..custom_exceptions import HalotoolsError


__all__ = ('OverhauledHaloCatalog', )


class OverhauledHaloCatalog(object):
    """
    Container class for halo catalogs and particle data.  
    """

    def __init__(self, **kwargs):
        """
        """

        try:
            import h5py
        except ImportError:
            raise HalotoolsError("Must have h5py package installed to use this feature")
        self._catman = catalog_manager.CatalogManager()

        self._process_constructor_inputs(**kwargs)

        self.fname_halo_table = self._catman.get_halo_table_fname(
            simname = simname, halo_finder = halo_finder, 
            redshift = redshift, version_name = version_name, 
            dz_tol = dz_tol)

        self._check_metadata_consistency(
            simname = simname, halo_finder = halo_finder, 
            redshift = redshift, version_name = version_name, 
            )

        self._load_halo_table()
        self._bind_metadata()

    def _process_constructor_inputs(self, **kwargs):
        """
        """
        pass

    def _load_halo_table(self):
        """ Retrieve the halo_table from ``self.fname_halo_table`` and update the cache log. 
        """
        pass

    def _bind_metadata(self):
        """ Create convenience bindings of all metadata to the `HaloCatalog` instance. 
        """
        f = h5py.File(self.fname_halo_table)
        for attr_key in f.attrs.keys():
            setattr(self, attr_key, f.attrs[attr])

    def _check_metadata_consistency(self, **kwargs):
        """ Check that any metadata passed to the constructor 
        agrees with any metadata bound to the hdf5 file storing the table. 
        """
        pass

    def store_halocat_in_cache(self, halo_table_fname, overwrite = False, 
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





