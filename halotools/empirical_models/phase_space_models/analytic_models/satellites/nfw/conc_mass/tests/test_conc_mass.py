"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.table import Table
from astropy.utils.misc import NumpyRNGContext
import pytest

from ..dutton_maccio14 import dutton_maccio14
from ..direct_from_halo_catalog import direct_from_halo_catalog

__all__ = ('test_dutton_maccio14', 'test_direct_from_halo_catalog1')

fixed_seed = 43


def test_dutton_maccio14():
    r"""
    """

    Npts = int(1e3)
    mass = np.logspace(10, 15, Npts)
    z = 0.
    conc = dutton_maccio14(mass, z)
    assert np.all(conc > 1)
    assert np.all(conc < 100)
    assert np.all(np.diff(conc) < 0)


def test_direct_from_halo_catalog1():
    r"""
    """
    Npts = int(10)
    mass = np.logspace(10, 15, Npts)
    with NumpyRNGContext(fixed_seed):
        conc = np.random.uniform(0, 100, Npts)
    t = Table({'conc': conc, 'halo_mvir': mass})
    conc_result = direct_from_halo_catalog(table=t, concentration_key='conc')
    assert np.allclose(conc, conc_result)


def test_direct_from_halo_catalog2():
    r"""
    """
    Npts = int(10)
    mass = np.logspace(10, 15, Npts)
    with NumpyRNGContext(fixed_seed):
        conc = np.random.uniform(0, 100, Npts)
    t = Table({'conc': conc, 'halo_mvir': mass})

    with pytest.raises(KeyError) as err:
        conc_result = direct_from_halo_catalog(table=t, concentration_key='Air Bud')
    substr = "The ``Air Bud`` key does not appear in the input halo catalog."
    assert substr in err.value.args[0]


def test_direct_from_halo_catalog3():
    r"""
    """
    Npts = int(10)
    mass = np.logspace(10, 15, Npts)
    with NumpyRNGContext(fixed_seed):
        conc = np.random.uniform(0, 100, Npts)
    t = Table({'conc': conc, 'halo_mvir': mass})

    with pytest.raises(KeyError) as err:
        conc_result = direct_from_halo_catalog(table=t)
    substr = "The ``direct_from_halo_catalog`` function accepts two keyword arguments"
    assert substr in err.value.args[0]
