""" Module providing unit-testing for the functions in 
the `~halotools.mock_observables.` module
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.tests.helper import pytest
from astropy.utils.misc import NumpyRNGContext

from .cf_helpers import generate_locus_of_3d_points

from ..error_estimation_tools import cuboid_subvolume_labels

fixed_seed = 43

__all__ = ('test_cuboid_subvolume_labels_bounds_checking', )

def test_cuboid_subvolume_labels_bounds_checking():
    Npts = 100
    with NumpyRNGContext(fixed_seed):
        good_sample = np.random.random((Npts, 3))
        bad_sample = np.random.random((Npts, 2))

    good_Nsub1 = 3
    good_Nsub2 = (4, 4, 4)
    bad_Nsub = (3, 3)

    good_Lbox = 1
    good_Lbox2 = (1, 1, 1)
    bad_Lbox = (3, 3)

    with pytest.raises(TypeError) as err:
        cuboid_subvolume_labels(bad_sample, good_Nsub1, good_Lbox)
    substr = "Input ``sample`` must have shape (Npts, 3)"
    assert substr in err.value.args[0]

    with pytest.raises(TypeError) as err:
        cuboid_subvolume_labels(good_sample, bad_Nsub, good_Lbox2)
    substr = "Input ``Nsub`` must be a scalar or length-3 sequence"
    assert substr in err.value.args[0]

    with pytest.raises(TypeError) as err:
        cuboid_subvolume_labels(good_sample, good_Nsub2, bad_Lbox)
    substr = "Input ``Lbox`` must be a scalar or length-3 sequence"
    assert substr in err.value.args[0]

def test_cuboid_subvolume_labels_correctness():
    Npts = 100
    Nsub = 2
    Lbox = 1

    sample = generate_locus_of_3d_points(Npts, xc=0.1, yc=0.1, zc=0.1, seed = fixed_seed)
    labels, N_sub_vol = cuboid_subvolume_labels(sample, Nsub, Lbox)
    assert np.all(labels == 1)

    sample = generate_locus_of_3d_points(Npts, xc=0.9, yc=0.9, zc=0.9, seed = fixed_seed)
    labels, N_sub_vol = cuboid_subvolume_labels(sample, Nsub, Lbox)
    assert np.all(labels == 8)

    sample = generate_locus_of_3d_points(Npts, xc=0.1, yc=0.1, zc=0.9, seed = fixed_seed)
    labels, N_sub_vol = cuboid_subvolume_labels(sample, Nsub, Lbox)
    assert np.all(labels == 2)

    sample = generate_locus_of_3d_points(Npts, xc=0.1, yc=0.9, zc=0.1, seed = fixed_seed)
    labels, N_sub_vol = cuboid_subvolume_labels(sample, Nsub, Lbox)
    assert np.all(labels == 3)










