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


__all__ = ('HaloCatalog', 'UserDefinedHaloCatalog')


class HaloCatalog(object):
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


class UserDefinedHaloCatalog(object):
    """ Class used to transform a user-provided halo catalog into the standard form recognized by Halotools. 
    """
    def __init__(self, **kwargs):
        """
        Parameters 
        ------------
        *metadata : dict 
            Keyword arguments storing catalog metadata. Both `Lbox` and `ptcl_mass` 
            are required and must be in Mpc/h and Msun/h units, respectively. 
            See Examples section for further notes. 

        *halo_catalog_columns : sequence of arrays 
            Sequence of length-*Nhalos* arrays passed in as keyword arguments. 

            Each key will be the column name attached to the input array. 
            All keys must begin with the substring ``halo_`` to help differentiate 
            halo property from mock galaxy properties. At a minimum, there must be a 
            'halo_id' keyword argument storing a unique integer for each halo, 
            as well as columns 'halo_x', 'halo_y' and 'halo_z'. 
            There must also be some additional mass-like variable, 
            for which you can use any name that begins with 'halo_'

        ptcl_table : table, optional 
            Astropy `~astropy.table.Table` object storing dark matter particles 
            randomly selected from the snapshot. At a minimum, the table must have 
            columns 'x', 'y' and 'z'. 

        Notes 
        -------
        This class is tested by 
        `~halotools.sim_manager.tests.test_user_defined_halo_catalog.TestUserDefinedHaloCatalog`. 

        """
        halocat_dict, metadata_dict = self._parse_constructor_kwargs(**kwargs)
        self.halo_table = Table(halocat_dict)
        for key, value in metadata_dict.iteritems():
            setattr(self, key, value)

        self._passively_bind_ptcl_table(**kwargs)

    def _parse_constructor_kwargs(self, **kwargs):
        """ Private method interprets constructor keyword arguments and returns two 
        dictionaries. One stores the halo catalog columns, the other stores the metadata. 

        Parameters 
        ------------
        **kwargs : keyword arguments passed to constructor 

        Returns 
        ----------
        halocat_dict : dictionary 
            Keys are the names of the halo catalog columns, values are length-*Nhalos* ndarrays. 

        metadata_dict : dictionary 
            Dictionary storing the catalog metadata. Keys will be attribute names bound 
            to the `UserDefinedHaloCatalog` instance. 
        """

        try:
            halo_id = kwargs['halo_id']
            assert type(halo_id) is np.ndarray 
            Nhalos = custom_len(halo_id)
            assert Nhalos > 1
        except KeyError, AssertionError:
            msg = ("\nThe UserDefinedHaloCatalog requires a ``halo_id`` keyword argument "
                "storing an ndarray of length Nhalos > 1.\n")
            raise HalotoolsError(msg)

        halocat_dict = (
            {key: kwargs[key] for key in kwargs 
            if (type(kwargs[key]) is np.ndarray) and (custom_len(kwargs[key]) == Nhalos)}
            )
        self._test_halocat_dict(halocat_dict)

        metadata_dict = (
            {key: kwargs[key] for key in kwargs
            if (key not in halocat_dict) and (key != 'ptcl_table')}
            )

        return halocat_dict, metadata_dict 


    def _test_halocat_dict(self, halocat_dict):
        """
        """ 
        try:
            assert 'halo_x' in halocat_dict 
            assert 'halo_y' in halocat_dict 
            assert 'halo_z' in halocat_dict 
            assert len(halocat_dict) >= 5
        except AssertionError:
            msg = ("\nThe UserDefinedHaloCatalog requires keyword arguments ``halo_x``, "
                "``halo_y`` and ``halo_z``,\nplus one additional column storing a mass-like variable.\n"
                "Each of these keyword arguments must storing an ndarray of the same length\n"
                "as the ndarray bound to the ``halo_id`` keyword argument.\n")
            raise HalotoolsError(msg)

        for key in halocat_dict:
            if key[:5] != 'halo_':
                msg = ("\nThe ``%s`` key passed to UserDefinedHaloCatalog stores \n"
                    "an ndarray of the same length as the ``halo_id`` keyword argument, \n"
                    "and so the ``%s`` key is interpreted as a halo catalog column.\n"
                    "All halo catalog column names must begin with ``halo_``\n"
                    "to help Halotools disambiguate between halo properties and mock galaxy properties.\n")
                raise HalotoolsError(msg)

    def _passively_bind_ptcl_table(self, **kwargs):
        """
        """

        try:
            ptcl_table = kwargs['ptcl_table']

            assert type(ptcl_table) is Table
            assert len(ptcl_table) > 1e4
            assert 'x' in ptcl_table
            assert 'y' in ptcl_table
            assert 'z' in ptcl_table

            self.ptcl_table = ptcl_table

        except AssertionError:
            msg = ("\nIf passing a ``ptcl_table`` to UserDefinedHaloCatalog, \n"
                "this argument must contain an Astropy Table object with at least 1e4 rows\n"
                "and ``x``, ``y`` and ``z`` columns. \n")
            raise HalotoolsError(msg)

        except KeyError:
            pass






