#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

from unittest import TestCase
import pytest 
import warnings 

import numpy as np 
from copy import copy, deepcopy 

from astropy.table import Table

from .. import UserSuppliedHaloCatalog
from ...custom_exceptions import HalotoolsError

__all__ = ['TestUserSuppliedHaloCatalog']

class TestUserSuppliedHaloCatalog(TestCase):
    """ Class providing tests of the `~halotools.sim_manager.UserSuppliedHaloCatalog`. 
    """

    def setup_class(self):
        """ Pre-load various arrays into memory for use by all tests. 
        """
        self.Nhalos = 1e2
        self.Lbox = 100
        self.redshift = 0.0
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

        self.num_ptcl = 1e4
        self.good_ptcl_table = Table(
            {'x': np.zeros(self.num_ptcl), 
            'y': np.zeros(self.num_ptcl), 
            'z': np.zeros(self.num_ptcl)}
            )

    def test_particle_mass_requirement(self):

        with pytest.raises(HalotoolsError):
            halocat = UserSuppliedHaloCatalog(Lbox = 200, **self.good_halocat_args)

    def test_lbox_requirement(self):

        with pytest.raises(HalotoolsError):
            halocat = UserSuppliedHaloCatalog(particle_mass = 200, **self.good_halocat_args)

    def test_halos_contained_inside_lbox(self):

        with pytest.raises(HalotoolsError):
            halocat = UserSuppliedHaloCatalog(Lbox = 20, particle_mass = 100, 
                **self.good_halocat_args)

    def test_successful_load(self):

        halocat = UserSuppliedHaloCatalog(Lbox = 200, particle_mass = 100, redshift = self.redshift, 
            **self.good_halocat_args)
        assert hasattr(halocat, 'Lbox')
        assert halocat.Lbox == 200
        assert hasattr(halocat, 'particle_mass')
        assert halocat.particle_mass == 100

    def test_additional_metadata(self):

        halocat = UserSuppliedHaloCatalog(Lbox = 200, particle_mass = 100, redshift = self.redshift,
            arnold_schwarzenegger = 'Stick around!', 
            **self.good_halocat_args)
        assert hasattr(halocat, 'arnold_schwarzenegger')
        assert halocat.arnold_schwarzenegger == 'Stick around!'

    def test_all_halo_columns_have_length_nhalos(self):

        # All halo catalog columns must have length-Nhalos
        bad_halocat_args = deepcopy(self.good_halocat_args)
        with pytest.raises(HalotoolsError):
            bad_halocat_args['halo_x'][0] = -1
            halocat = UserSuppliedHaloCatalog(Lbox = 200, particle_mass = 100, redshift = self.redshift,
                **bad_halocat_args)

    def test_positions_contained_inside_lbox_alt_test(self):
        # positions must be < Lbox
        bad_halocat_args = deepcopy(self.good_halocat_args)
        with pytest.raises(HalotoolsError):
            bad_halocat_args['halo_x'][0] = 10000
            halocat = UserSuppliedHaloCatalog(Lbox = 200, particle_mass = 100, redshift = self.redshift,
                **bad_halocat_args)

    def test_has_halo_x_column(self):
        # must have halo_x column 
        bad_halocat_args = deepcopy(self.good_halocat_args)
        with pytest.raises(HalotoolsError):
            del bad_halocat_args['halo_x']
            halocat = UserSuppliedHaloCatalog(Lbox = 200, particle_mass = 100, redshift = self.redshift,
                **bad_halocat_args)

    def test_has_halo_id_column(self):
        # Must have halo_id column 
        bad_halocat_args = deepcopy(self.good_halocat_args)
        with pytest.raises(HalotoolsError):
            del bad_halocat_args['halo_id']
            halocat = UserSuppliedHaloCatalog(Lbox = 200, particle_mass = 100, redshift = self.redshift,
                **bad_halocat_args)

    def test_has_halo_mass_column(self):
        # Must have some column storing a mass-like variable
        bad_halocat_args = deepcopy(self.good_halocat_args)
        with pytest.raises(HalotoolsError):
            del bad_halocat_args['halo_mass']
            halocat = UserSuppliedHaloCatalog(Lbox = 200, particle_mass = 100, redshift = self.redshift,
                **bad_halocat_args)

    def test_halo_prefix_warning(self):
        # Must raise warning if a length-Nhalos array is passed with 
        # a keyword argument that does not begin with 'halo_'
        bad_halocat_args = deepcopy(self.good_halocat_args)
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            bad_halocat_args['s'] = np.ones(self.Nhalos)
            halocat = UserSuppliedHaloCatalog(Lbox = 200, particle_mass = 100, redshift = self.redshift,
                **bad_halocat_args)
            assert 'interpreted as metadata' in str(w[-1].message)

    def test_ptcl_table(self):
        """ Method performs various existence and consistency tests on the input ptcl_table.

        * Enforce that instances do *not* have ``ptcl_table`` attributes if none is passed. 

        * Enforce that instances *do* have ``ptcl_table`` attributes if a legitimate one is passed. 

        * Enforce that ptcl_table have ``x``, ``y`` and ``z`` columns. 

        * Enforce that ptcl_table input is an Astropy `~astropy.table.Table` object, not a Numpy recarray
        """

    def test_ptcl_table_dne(self):
        # Must not have a ptcl_table attribute when none is passed
        halocat = UserSuppliedHaloCatalog(Lbox = 200, particle_mass = 100, redshift = self.redshift,
            **self.good_halocat_args)
        assert not hasattr(halocat, 'ptcl_table')

    def test_ptcl_table_exists_when_given_goodargs(self):
   
        # Must have ptcl_table attribute when argument is legitimate
        halocat = UserSuppliedHaloCatalog(
            Lbox = 200, particle_mass = 100, redshift = self.redshift,
            ptcl_table = self.good_ptcl_table, **self.good_halocat_args)
        assert hasattr(halocat, 'ptcl_table')

    def test_min_numptcl_requirement(self):
        # Must have at least 1e4 particles
        num_ptcl2 = 1e3
        ptcl_table2 = Table(
            {'x': np.zeros(num_ptcl2), 
            'y': np.zeros(num_ptcl2), 
            'z': np.zeros(num_ptcl2)}
            )
        with pytest.raises(HalotoolsError):
            halocat = UserSuppliedHaloCatalog(
                Lbox = 200, particle_mass = 100, redshift = self.redshift,
                ptcl_table = ptcl_table2, **self.good_halocat_args)

    def test_ptcls_have_zposition(self):
        # Must have a 'z' column 
        num_ptcl2 = 1e4
        ptcl_table2 = Table(
            {'x': np.zeros(num_ptcl2), 
            'y': np.zeros(num_ptcl2)}
            )
        with pytest.raises(HalotoolsError):
            halocat = UserSuppliedHaloCatalog(
                Lbox = 200, particle_mass = 100, redshift = self.redshift,
                ptcl_table = ptcl_table2, **self.good_halocat_args)

    def test_ptcls_are_astropy_table(self):
        # Data structure must be an astropy table, not an ndarray
        ptcl_table2 = self.good_ptcl_table.as_array()
        with pytest.raises(HalotoolsError):
            halocat = UserSuppliedHaloCatalog(
                Lbox = 200, particle_mass = 100, redshift = self.redshift,
                ptcl_table = ptcl_table2, **self.good_halocat_args)










