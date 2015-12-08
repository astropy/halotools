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

__all__ = ('UserDefinedHaloCatalog', )

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
        halo_table_dict, metadata_dict = self._parse_constructor_kwargs(**kwargs)
        self.halo_table = Table(halo_table_dict)

        self._test_metadata_dict(**metadata_dict)
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
        halo_table_dict : dictionary 
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

        halo_table_dict = (
            {key: kwargs[key] for key in kwargs 
            if (type(kwargs[key]) is np.ndarray) 
            and (custom_len(kwargs[key]) == Nhalos) 
            and (key[:5] == 'halo_')}
            )
        self._test_halo_table_dict(halo_table_dict)

        metadata_dict = (
            {key: kwargs[key] for key in kwargs
            if (key not in halo_table_dict) and (key != 'ptcl_table')}
            )

        return halo_table_dict, metadata_dict 


    def _test_halo_table_dict(self, halo_table_dict):
        """
        """ 
        try:
            assert 'halo_x' in halo_table_dict 
            assert 'halo_y' in halo_table_dict 
            assert 'halo_z' in halo_table_dict 
            assert len(halo_table_dict) >= 5
        except AssertionError:
            msg = ("\nThe UserDefinedHaloCatalog requires keyword arguments ``halo_x``, "
                "``halo_y`` and ``halo_z``,\nplus one additional column storing a mass-like variable.\n"
                "Each of these keyword arguments must storing an ndarray of the same length\n"
                "as the ndarray bound to the ``halo_id`` keyword argument.\n")
            raise HalotoolsError(msg)

    def _test_metadata_dict(self, **metadata_dict):
        """
        """
        try:
            assert 'Lbox' in metadata_dict
            assert custom_len(metadata_dict['Lbox']) == 1
            assert 'ptcl_mass' in metadata_dict
            assert custom_len(metadata_dict['ptcl_mass']) == 1
        except AssertionError:
            msg = ("\nThe UserDefinedHaloCatalog requires keyword arguments ``Lbox`` and ``ptcl_mass``\n"
                "storing scalars that will be interpreted as metadata about the halo catalog.\n")
            raise HalotoolsError(msg)

        Lbox = metadata_dict['Lbox']
        try:
            x, y, z = (
                self.halo_table['halo_x'], 
                self.halo_table['halo_y'], 
                self.halo_table['halo_z']
                )
            assert np.all(x >= 0)
            assert np.all(x <= Lbox)
            assert np.all(y >= 0)
            assert np.all(y <= Lbox)
            assert np.all(z >= 0)
            assert np.all(z <= Lbox)
        except AssertionError:
            msg = ("The ``halo_x``, ``halo_y`` and ``halo_z`` columns must only store arrays\n"
                "that are bound by 0 and the input ``Lbox``. \n")
            raise HalotoolsError(msg)

        for key, value in metadata_dict.iteritems():
            if (type(value) == np.ndarray):
                if custom_len(value) == len(self.halo_table['halo_id']):
                    msg = ("\nThe input ``" + key + "`` argument stores a length-Nhalos ndarray.\n"
                        "However, this key is being interpreted as metadata because \n"
                        "it does not begin with ``halo_``. If this is your intention, ignore this message.\n"
                        "Otherwise, rename this key to begin with ``halo_``. \n")
                    warn(msg, UserWarning)


    def _passively_bind_ptcl_table(self, **kwargs):
        """
        """

        try:
            ptcl_table = kwargs['ptcl_table']

            assert type(ptcl_table) is Table
            assert len(ptcl_table) >= 1e4
            assert 'x' in ptcl_table.keys()
            assert 'y' in ptcl_table.keys()
            assert 'z' in ptcl_table.keys()

            self.ptcl_table = ptcl_table

        except AssertionError:
            msg = ("\nIf passing a ``ptcl_table`` to UserDefinedHaloCatalog, \n"
                "this argument must contain an Astropy Table object with at least 1e4 rows\n"
                "and ``x``, ``y`` and ``z`` columns. \n")
            raise HalotoolsError(msg)

        except KeyError:
            pass



