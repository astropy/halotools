"""
"""
from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np
import pytest
from copy import deepcopy

from ..subhalo_phase_space import SubhaloPhaseSpace

from .....sim_manager import CachedHaloCatalog
from .....empirical_models import PrebuiltHodModelFactory, HodModelFactory
from .....utils import crossmatch
from .....mock_observables import relative_positions_and_velocities as rel_posvel

__all__ = ('test_composite_model', )

try:
    halocat = CachedHaloCatalog()
    HAS_DEFAULT_CATALOG = True
except:
    HAS_DEFAULT_CATALOG = False

fixed_seed = 43


@pytest.mark.skipif('not HAS_DEFAULT_CATALOG')
def test_composite_model():
    orig_model = PrebuiltHodModelFactory('leauthaud11')
    halocat = CachedHaloCatalog()

    model_dictionary = deepcopy(orig_model.model_dictionary)
    model_dictionary['satellites_profile'] = SubhaloPhaseSpace(
        'satellites', np.logspace(10.5, 15.1, 15))
    model = HodModelFactory(**model_dictionary)

    f = getattr(model, 'inherit_subhalo_properties_satellites')
    d = getattr(f, 'additional_kwargs')
    assert 'halo_table' in d
    model.populate_mock(halocat, seed=fixed_seed)

    satmask = model.mock.galaxy_table['gal_type'] == 'satellites'
    sats = model.mock.galaxy_table[satmask]

    assert np.all(sats['x'] >= 0.)
    assert np.all(sats['x'] <= halocat.Lbox[0])
    assert np.all(sats['x'] >= 0.)
    assert np.all(sats['y'] <= halocat.Lbox[1])
    assert np.all(sats['z'] >= 0.)
    assert np.all(sats['z'] <= halocat.Lbox[2])

    hostrvir = np.zeros(len(sats))
    hostx = np.zeros(len(sats))
    hosty = np.zeros(len(sats))
    hostz = np.zeros(len(sats))
    x = np.zeros(len(sats))
    y = np.zeros(len(sats))
    z = np.zeros(len(sats))
    idxA, idxB = crossmatch(sats['halo_hostid'], model.mock.halo_table['halo_id'])
    hostx[idxA] = model.mock.halo_table['halo_x'][idxB]
    hosty[idxA] = model.mock.halo_table['halo_y'][idxB]
    hostz[idxA] = model.mock.halo_table['halo_z'][idxB]
    hostrvir[idxA] = model.mock.halo_table['halo_rvir'][idxB]
    x, y, z = sats['x'], sats['y'], sats['z']

    dx = rel_posvel(hostx, x, period=halocat.Lbox[0])
    dy = rel_posvel(hosty, y, period=halocat.Lbox[1])
    dz = rel_posvel(hostz, z, period=halocat.Lbox[2])

    d = np.sqrt(dx**2 + dy**2 + dz**2)
    assert np.all(d < 2*hostrvir)
