""" Module providing unit-testing for `~halotools.utils.value_added_halo_table_functions`.
"""

from __future__ import absolute_import, division, print_function

from copy import deepcopy

from collections import Counter

import numpy as np

import pytest

from ..value_added_halo_table_functions import (
    broadcast_host_halo_property,
    add_halo_hostid,
    compute_uber_hostid,
)
from ..crossmatch import crossmatch

from ...sim_manager import FakeSim

from ...custom_exceptions import HalotoolsError

__all__ = ("test_broadcast_host_halo_mass1",)


def test_broadcast_host_halo_mass1():
    """ """
    fake_sim = FakeSim()
    t = fake_sim.halo_table

    broadcast_host_halo_property(t, "halo_mvir", delete_possibly_existing_column=True)

    assert "halo_mvir_host_halo" in list(t.keys())

    hostmask = t["halo_hostid"] == t["halo_id"]
    assert np.all(t["halo_mvir_host_halo"][hostmask] == t["halo_mvir"][hostmask])
    assert np.any(t["halo_mvir_host_halo"][~hostmask] != t["halo_mvir"][~hostmask])

    # Verify that both the group_member_generator method and the
    # crossmatch method give identical results for calculation of host halo mass
    idx_table1, idx_table2 = crossmatch(t["halo_hostid"], t["halo_id"])
    t["tmp"] = np.zeros(len(t), dtype=t["halo_mvir"].dtype)
    t["tmp"][idx_table1] = t["halo_mvir"][idx_table2]
    assert np.all(t["tmp"] == t["halo_mvir_host_halo"])

    data = Counter(t["halo_hostid"])
    frequency_analysis = data.most_common()

    for igroup in range(0, 10):
        idx = np.where(t["halo_hostid"] == frequency_analysis[igroup][0])[0]
        idx_host = np.where(t["halo_id"] == frequency_analysis[igroup][0])[0]
        assert np.all(t["halo_mvir_host_halo"][idx] == t["halo_mvir"][idx_host])

    for igroup in range(-10, -1):
        idx = np.where(t["halo_hostid"] == frequency_analysis[igroup][0])[0]
        idx_host = np.where(t["halo_id"] == frequency_analysis[igroup][0])[0]
        assert np.all(t["halo_mvir_host_halo"][idx] == t["halo_mvir"][idx_host])

    del t


def test_broadcast_host_halo_mass2():
    """ """
    fake_sim = FakeSim()
    with pytest.raises(HalotoolsError) as err:
        broadcast_host_halo_property(4, "xxx")
    substr = "The input ``table`` must be an Astropy `~astropy.table.Table` object"
    assert substr in err.value.args[0]


def test_broadcast_host_halo_mass3():
    """ """
    fake_sim = FakeSim()
    t = fake_sim.halo_table
    with pytest.raises(HalotoolsError) as err:
        broadcast_host_halo_property(t, "xxx")
    substr = "The input table does not have the input ``halo_property_key``"
    assert substr in err.value.args[0]


def test_broadcast_host_halo_mass4():
    """ """
    fake_sim = FakeSim()
    t = fake_sim.halo_table

    with pytest.raises(HalotoolsError) as err:
        broadcast_host_halo_property(t, "halo_mvir")
    substr = "Your input table already has an existing new_colname column name."
    assert substr in err.value.args[0]

    broadcast_host_halo_property(t, "halo_mvir", delete_possibly_existing_column=True)


def test_add_halo_hostid1():
    """ """
    with pytest.raises(HalotoolsError) as err:
        add_halo_hostid(5, delete_possibly_existing_column=False)
    substr = "The input ``table`` must be an Astropy `~astropy.table.Table` object"
    assert substr in err.value.args[0]


def test_add_halo_hostid2():
    """ """
    fake_sim = FakeSim()
    t = fake_sim.halo_table

    del t["halo_id"]
    with pytest.raises(HalotoolsError) as err:
        add_halo_hostid(t, delete_possibly_existing_column=False)
    substr = "The input table must have ``halo_upid`` and ``halo_id`` keys"
    assert substr in err.value.args[0]


def test_add_halo_hostid3():
    """ """
    fake_sim = FakeSim()
    t = fake_sim.halo_table

    with pytest.raises(HalotoolsError) as err:
        add_halo_hostid(t, delete_possibly_existing_column=False)
    substr = "Your input table already has an existing ``halo_hostid`` column name."
    assert substr in err.value.args[0]

    existing_halo_hostid = deepcopy(t["halo_hostid"].data)
    del t["halo_hostid"]

    add_halo_hostid(t, delete_possibly_existing_column=False)

    assert np.all(t["halo_hostid"] == existing_halo_hostid)

    add_halo_hostid(t, delete_possibly_existing_column=True)
    assert np.all(t["halo_hostid"] == existing_halo_hostid)


def test_compute_uber_hostid():
    haloid = np.array((0, 1, 2, 3))
    upid = np.array((-1, 0, 25, 2))

    corrected_upid, uber_hostid, n_iter = compute_uber_hostid(upid, haloid)

    correct_upid = np.array((-1, 1, -1, 2))
    correct_hostid = np.array((0, 0, 2, 2))

    assert np.allclose(correct_upid, correct_upid)
    assert np.allclose(uber_hostid, correct_hostid)
