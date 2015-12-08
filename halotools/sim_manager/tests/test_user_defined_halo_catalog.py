#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

from unittest import TestCase
import pytest 
import warnings 

import numpy as np 
from copy import copy, deepcopy 

from astropy.table import Table

from .. import UserDefinedHaloCatalog
from ...custom_exceptions import HalotoolsError

__all__ = ['TestUserDefinedHaloCatalog']

class TestUserDefinedHaloCatalog(TestCase):
    """ Class providing tests of the `~halotools.sim_manager.UserDefinedHaloCatalog`. 
    """

    def setup_class(self):
        """ Pre-load various arrays into memory for use by all tests. 
        """
        self.Nhalos = 1e2
        self.Lbox = 100
        self.halo_x = np.linspace(0, self.Lbox, self.Nhalos)
        self.halo_y = np.linspace(0, self.Lbox, self.Nhalos)
        self.halo_z = np.linspace(0, self.Lbox, self.Nhalos)
        self.halo_mass = np.logspace(10, 15, self.Nhalos)
        self.halo_id = np.arange(0, self.Nhalos)
        self.good_halocat_args = (
            {'halo_x': self.halo_x, 'halo_y': self.halo_y, 
            'halo_z': self.halo_z, 'halo_id': self.halo_id, 'halo_mass': self.halo_mass}
            )
        self.toy_list = [elt for elt in self.halo_x]
        self.ptcl_table = Table({'x': self.halo_x, 'y': self.halo_y, 'z': self.halo_z})

    def test_metadata(self):
        """ Method performs various existence and consistency tests on the input metadata. 

        * Enforces that ``Lbox`` and ``ptcl_mass`` are passed. 

        * Enforces that all ``halo_x``, ``halo_y`` and ``halo_z`` coordinates are between 0 and ``Lbox``. 

        * Enforces that all metadata get bound to the instance. 
        """
        with pytest.raises(HalotoolsError):
            halocat = UserDefinedHaloCatalog(Lbox = 200, **self.good_halocat_args)

        with pytest.raises(HalotoolsError):
            halocat = UserDefinedHaloCatalog(ptcl_mass = 200, **self.good_halocat_args)

        with pytest.raises(HalotoolsError):
            halocat = UserDefinedHaloCatalog(Lbox = 20, ptcl_mass = 100, 
                **self.good_halocat_args)

        halocat = UserDefinedHaloCatalog(Lbox = 200, ptcl_mass = 100, 
            **self.good_halocat_args)
        assert hasattr(halocat, 'Lbox')
        assert halocat.Lbox == 200
        assert hasattr(halocat, 'ptcl_mass')
        assert halocat.ptcl_mass == 100
        halocat = UserDefinedHaloCatalog(Lbox = 200, ptcl_mass = 100, 
            arnold_schwarzenegger = 'Stick around!', 
            **self.good_halocat_args)
        assert hasattr(halocat, 'arnold_schwarzenegger')
        assert halocat.arnold_schwarzenegger == 'Stick around!'

    def test_halo_table_data(self):
        """ Method performs various existence and consistency tests on the input halo catalog. 

        * Enforces that ``halo_id`` is passed to the constructor. 

        * Enforces that some mass-like variable is passed to the constructor. 

        * Enforces that ``halo_x``, ``halo_y`` and ``halo_z`` are all passed to the constructor. 

        * Enforces that all ``halo_x``, ``halo_y`` and ``halo_z`` coordinates are between 0 and ``Lbox``. 

        * Enforces that all length-*Nhalos* ndarray inputs have keywords that begin with ``halo_``. 
        """

        # All halo catalog columns must have length-Nhalos
        bad_halocat_args = deepcopy(self.good_halocat_args)
        with pytest.raises(HalotoolsError):
            bad_halocat_args['halo_x'][0] = -1
            halocat = UserDefinedHaloCatalog(Lbox = 200, ptcl_mass = 100, 
                **bad_halocat_args)

        # positions must be < Lbox
        bad_halocat_args = deepcopy(self.good_halocat_args)
        with pytest.raises(HalotoolsError):
            bad_halocat_args['halo_x'][0] = 10000
            halocat = UserDefinedHaloCatalog(Lbox = 200, ptcl_mass = 100, 
                **bad_halocat_args)

        # must have halo_x column 
        bad_halocat_args = deepcopy(self.good_halocat_args)
        with pytest.raises(HalotoolsError):
            del bad_halocat_args['halo_x']
            halocat = UserDefinedHaloCatalog(Lbox = 200, ptcl_mass = 100, 
                **bad_halocat_args)

        # Must have halo_id column 
        bad_halocat_args = deepcopy(self.good_halocat_args)
        with pytest.raises(HalotoolsError):
            del bad_halocat_args['halo_id']
            halocat = UserDefinedHaloCatalog(Lbox = 200, ptcl_mass = 100, 
                **bad_halocat_args)

        # Must have some column storing a mass-like variable
        bad_halocat_args = deepcopy(self.good_halocat_args)
        with pytest.raises(HalotoolsError):
            del bad_halocat_args['halo_mass']
            halocat = UserDefinedHaloCatalog(Lbox = 200, ptcl_mass = 100, 
                **bad_halocat_args)

        # Must raise warning if a length-Nhalos array is passed with 
        # a keyword argument that does not begin with 'halo_'
        bad_halocat_args = deepcopy(self.good_halocat_args)
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            bad_halocat_args['s'] = np.ones(self.Nhalos)
            halocat = UserDefinedHaloCatalog(Lbox = 200, ptcl_mass = 100, 
                **bad_halocat_args)
            assert len(w) == 2
            assert 'interpreted as metadata' in str(w[-1].message)














