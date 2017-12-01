"""
"""
from __future__ import (absolute_import, division, print_function)

import numpy as np
import pytest

from ..fake_sim import FakeSim, FakeSimHalosNearBoundaries
from ...custom_exceptions import HalotoolsError

__all__ = ('test_fake_sim_default_size', )


@pytest.mark.installation_test
def test_fake_sim_default_size():
    fake_sim = FakeSim()
    try:
        assert fake_sim.num_halos_per_massbin == 100
    except AssertionError:
        msg = ("\nThe test suite runs slowly and is excessively "
            "memory intensive if the default ``num_halos_per_massbin`` "
            "is larger than 100 or so.\nOnly change it if you have a good reason.\n")
        raise HalotoolsError(msg)


def test_attrs():
    fake_sim = FakeSim()
    keylist = ['halo_hostid']
    for key in keylist:
        assert key in list(fake_sim.halo_table.keys())


def test_stochasticity():
    fake_sim1 = FakeSim()
    fake_sim2 = FakeSim()
    fake_sim3 = FakeSim(seed=44)

    fake_sim3_is_identical = True
    for key in fake_sim1.halo_table.keys():
        col1 = fake_sim1.halo_table[key]
        col2 = fake_sim2.halo_table[key]
        assert np.all(col1 == col2)
        col3 = fake_sim3.halo_table[key]
        if np.any(col3 != col2):
            fake_sim3_is_identical = False

    assert fake_sim3_is_identical is False


def test_attrs2():
    fake_sim = FakeSimHalosNearBoundaries()
    keylist = ['halo_hostid']
    for key in keylist:
        assert key in list(fake_sim.halo_table.keys())

    def test_halo_mvir_host_halo(self):
        assert 'halo_mvir_host_halo' in self.fake_sim.halo_table.keys()
        mask = self.fake_sim.halo_table['halo_upid'] == -1
        assert np.all(self.fake_sim.halo_table['halo_mvir'][mask] == self.fake_sim.halo_table['halo_mvir_host_halo'][mask])
        assert np.any(self.fake_sim.halo_table['halo_mvir'][~mask] != self.fake_sim.halo_table['halo_mvir_host_halo'][~mask])
        assert 0. not in self.fake_sim.halo_table['halo_mvir_host_halo'].data

    def tearDown(self):
        del self.fake_sim


def test_positions2():
    fake_sim = FakeSimHalosNearBoundaries()
    assert not np.any((fake_sim.halo_table['halo_x'] > 1) &
        (fake_sim.halo_table['halo_x'] < fake_sim.Lbox[0] - 1))


def test_stochasticity2():
    fake_sim1 = FakeSimHalosNearBoundaries()
    fake_sim2 = FakeSimHalosNearBoundaries()
    fake_sim3 = FakeSimHalosNearBoundaries(seed=44)

    fake_sim3_is_identical = True
    for key in fake_sim1.halo_table.keys():
        col1 = fake_sim1.halo_table[key]
        col2 = fake_sim2.halo_table[key]
        assert np.all(col1 == col2)
        col3 = fake_sim3.halo_table[key]
        if np.any(col3 != col2):
            fake_sim3_is_identical = False

    assert fake_sim3_is_identical is False
