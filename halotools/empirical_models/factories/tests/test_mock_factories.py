#!/usr/bin/env python

import numpy as np 

from astropy.tests.helper import pytest
from unittest import TestCase

from ...composite_models import *
from ...factories import *

from ....empirical_models import PrebuiltHodModelFactory
from ....sim_manager import FakeSim
from ....sim_manager.fake_sim import FakeSimHalosNearBoundaries

__all__ = ['test_preloaded_hod_mocks']


@pytest.mark.slow
def test_xyz_positions1():
    """ Loop over all pre-loaded HOD models, 
    and one-by-one test that mock instances created by 
    `~halotools.empirical_models.HodMockFactory`. 

    Notes 
    -----
    Any HOD-style model listed in the ``__all__`` built-in attribute of the 
    `~halotools.empirical_models.preloaded_models` module will be tested. 
    Test suite includes: 

        * Mock has appropriate properties when instantiated with default settings, as well as non-trivial entries for ``additional_haloprops``, ``new_haloprop_func_dict``, and ``create_astropy_table``. 

        * Galaxy positions satisfy :math:`0 < x, y, z < L_{\\rm box}`.  
    """
    model = PrebuiltHodModelFactory('zheng07', threshold = -21)
    halocat = FakeSimHalosNearBoundaries()
    model.populate_mock(halocat = halocat)

    pos = model.mock.xyz_positions()
    assert np.shape(pos) == (len(model.mock.galaxy_table), 3)

    mask = model.mock.galaxy_table['halo_mvir'] >= 10**13.5
    masked_pos = model.mock.xyz_positions(mask = mask)
    numgals = len(model.mock.galaxy_table[mask])
    assert np.shape(masked_pos) == (numgals, 3)

    assert masked_pos.shape[0] < pos.shape[0]

    pos_zdist = model.mock.xyz_positions(
        velocity_distortion_dimension = 'z')
    assert np.all(pos_zdist[:,0] == pos[:,0])
    assert np.all(pos_zdist[:,1] == pos[:,1])
    assert np.any(pos_zdist[:,2] != pos[:,2])
    assert np.all(abs(pos_zdist[:,2] - pos[:,2]) < 50)

    pos_zdist_pbc = model.mock.xyz_positions(
        velocity_distortion_dimension = 'z', 
        period = halocat.Lbox)
    assert np.all(pos_zdist[:,0] == pos[:,0])
    assert np.all(pos_zdist[:,1] == pos[:,1])
    assert np.any(pos_zdist[:,2] != pos[:,2])

    assert np.any(abs(pos_zdist_pbc[:,2] - pos[:,2]) > 50)









