#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

from unittest import TestCase
import pytest 

import numpy as np 
from copy import copy 

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

        * Enforces that all ``x``, ``y`` and ``z`` coordinates are between 0 and ``Lbox``. 

        """

        halocat = UserDefinedHaloCatalog(Lbox = 200, ptcl_mass = 100, 
            **self.good_halocat_args)

        with pytest.raises(HalotoolsError):
            halocat = UserDefinedHaloCatalog(Lbox = 200, **self.good_halocat_args)
            halocat = UserDefinedHaloCatalog(ptcl_mass = 200, **self.good_halocat_args)
            halocat = UserDefinedHaloCatalog(Lbox = 20, ptcl_mass = 100, 
                **self.good_halocat_args)













