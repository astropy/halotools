#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

from unittest import TestCase

import numpy as np

from ..fake_sim import FakeSim, FakeSimHalosNearBoundaries
from ...custom_exceptions import HalotoolsError

__all__ = ['TestFakeSim', 'TestFakeSimHalosNearBoundaries']


class TestFakeSim(TestCase):
    """
    """

    def setUp(self):
        self.fake_sim = FakeSim()

    def test_fake_sim_default_size(self):
        try:
            assert self.fake_sim.num_halos_per_massbin == 100
        except AssertionError:
            msg = ("\nThe test suite runs slowly and is excessively "
                "memory intensive if the default ``num_halos_per_massbin`` "
                "is larger than 100 or so.\nOnly change it if you have a good reason.\n")
            raise HalotoolsError(msg)

    def test_attrs(self):
        keylist = ['halo_hostid']
        for key in keylist:
            assert key in list(self.fake_sim.halo_table.keys())

    def test_halo_mvir_host_halo(self):
        assert 'halo_mvir_host_halo' in self.fake_sim.halo_table.keys()
        mask = self.fake_sim.halo_table['halo_upid'] == -1
        assert np.all(self.fake_sim.halo_table['halo_mvir'][mask] == self.fake_sim.halo_table['halo_mvir_host_halo'][mask])
        assert np.any(self.fake_sim.halo_table['halo_mvir'][~mask] != self.fake_sim.halo_table['halo_mvir_host_halo'][~mask])
        assert 0. not in self.fake_sim.halo_table['halo_mvir_host_halo'].data

    def tearDown(self):
        del self.fake_sim


class TestFakeSimHalosNearBoundaries(TestCase):
    """
    """

    def setUp(self):
        self.fake_sim = FakeSimHalosNearBoundaries()

    def test_attrs(self):
        keylist = ['halo_hostid']
        for key in keylist:
            assert key in list(self.fake_sim.halo_table.keys())

    def test_positions(self):
        assert not np.any((self.fake_sim.halo_table['halo_x'] > 1) &
            (self.fake_sim.halo_table['halo_x'] < self.fake_sim.Lbox - 1))
