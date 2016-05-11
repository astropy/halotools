""" Module providing unit-testing of `~halotools.mock_observables.radial_profile_3d`. 
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext

from ..radial_profile_3d import radial_profile_3d
from ...tests.cf_helpers import (generate_locus_of_3d_points, 
    generate_thin_shell_of_3d_points, generate_3d_regular_mesh)

import pytest 

__all__ = ('test_radial_profile_3d_test1', )

fixed_seed = 44

@pytest.mark.xfail
def test_radial_profile_3d_test1():
    """ For a tight localization of sample1 points surrounded by two concentric 
    shells of sample2 points, verify that both the counts and the primary result 
    of `~halotools.mock_observables.radial_profile_3d` are correct. 

    In this test, PBCs are turned off. 
    """
    npts1 = 100
    xc, yc, zc = 0.5, 0.5, 0.5
    sample1 = generate_locus_of_3d_points(npts1, xc, yc, zc, seed=fixed_seed)

    rbins = np.array([0.1, 0.2, 0.3, 0.4])
    midpoints = (rbins[:-1] + rbins[1:])/2.
    sample2a = generate_thin_shell_of_3d_points(npts1, midpoints[0], xc, yc, zc, seed=fixed_seed)
    sample2b = generate_thin_shell_of_3d_points(npts1, midpoints[1], xc, yc, zc, seed=fixed_seed)
    sample2c = generate_thin_shell_of_3d_points(npts1, midpoints[2], xc, yc, zc, seed=fixed_seed)
    sample2 = np.concatenate([sample2a, sample2b, sample2c])

    quantity_a, quantity_b, quantity_c = np.zeros(npts1) + 0.5, np.zeros(npts1) + 1.5, np.zeros(npts1) + 2.5
    quantity = np.concatenate([quantity_a, quantity_b, quantity_c])

    result = radial_profile_3d(sample1, sample2, quantity, rbins)
    assert len(result) == len(midpoints)
    assert np.all(result == [0.5, 1.5, 2.5])

@pytest.mark.xfail
def test_radial_profile_3d_test2():
    """ For two tight localizations of sample1 points each surrounded by two concentric 
    shells of sample2 points, verify that both the counts and the primary result 
    of `~halotools.mock_observables.radial_profile_3d` are correct. 

    In this test, PBCs have a non-trivial impact on the results.  
    """
    npts1a = 100
    xca1, yca1, zca1 = 0.3, 0.3, 0.3
    xca2, yca2, zca2 = 0.9, 0.9, 0.9

    sample1a = generate_locus_of_3d_points(npts1a, xca1, yca1, zca1, seed=fixed_seed)
    sample1b = generate_locus_of_3d_points(npts1a, xca2, yca2, zca2, seed=fixed_seed)
    sample1 = np.concatenate([sample1a, sample1b])
    npts1 = len(sample1)

    rbins = np.array([0.01, 0.03, 0.3])
    r1, r2 = 0.02, 0.2

    sample2_p1_r1 = generate_thin_shell_of_3d_points(npts1, r1, xca1, yca1, zca1, seed=fixed_seed, Lbox=1)
    sample2_p2_r1 = generate_thin_shell_of_3d_points(npts1, r1, xca2, yca2, zca2, seed=fixed_seed, Lbox=1)
    sample2_p1_r2 = generate_thin_shell_of_3d_points(npts1, r2, xca1, yca1, zca1, seed=fixed_seed, Lbox=1)
    sample2_p2_r2 = generate_thin_shell_of_3d_points(npts1, r2, xca2, yca2, zca2, seed=fixed_seed, Lbox=1)
    sample2 = np.concatenate([sample2_p1_r1, sample2_p2_r1, sample2_p1_r2, sample2_p2_r2])
    npts2 = len(sample2)

    quantity_a, quantity_b = np.zeros(npts2/2) + 0.5, np.zeros(npts2/2) + 1.5
    quantity = np.concatenate([quantity_a, quantity_b])

    result, counts = radial_profile_3d(sample1, sample2, quantity, rbins, return_counts=True, period=1)
    assert np.all(counts == (npts1/2.)*(npts2/2.))
    assert np.all(result == [0.5, 1.5])

@pytest.mark.xfail
def test_radial_profile_3d_test3():
    """ Create a regular mesh of ``sample1`` points and two concentric rings around 
    two different points in the mesh. Give random uniform weights to the rings, 
    and verify that `~halotools.mock_observables.radial_profile_3d` returns the 
    correct counts and results. 
    """
    npts1 = 100
    sample1 = generate_3d_regular_mesh(4) # coords = 0.125, 0.375, 0.625, 0.875

    rbins = np.array([0.04, 0.06, 0.1])
    r1, r2 = 0.05, 0.09

    xca1, yca1, zca1 = 0.125, 0.125, 0.125
    xca2, yca2, zca2 = 0.625, 0.625, 0.625
    sample2_p1_r1 = generate_thin_shell_of_3d_points(npts1, r1, xca1, yca1, zca1, seed=fixed_seed, Lbox=1)
    sample2_p2_r1 = generate_thin_shell_of_3d_points(npts1, r1, xca2, yca2, zca2, seed=fixed_seed, Lbox=1)
    sample2_p1_r2 = generate_thin_shell_of_3d_points(npts1, r2, xca1, yca1, zca1, seed=fixed_seed, Lbox=1)
    sample2_p2_r2 = generate_thin_shell_of_3d_points(npts1, r2, xca2, yca2, zca2, seed=fixed_seed, Lbox=1)
    sample2 = np.concatenate([sample2_p1_r1, sample2_p2_r1, sample2_p1_r2, sample2_p2_r2])
    npts2 = len(sample2)

    with NumpyRNGContext(fixed_seed):
        inner_ring_values = np.random.uniform(-1, 1, npts2/2)
        outer_ring_values = np.random.uniform(-1, 1, npts2/2)

    quantity = np.concatenate([inner_ring_values, outer_ring_values])

    result, counts = radial_profile_3d(sample1, sample2, quantity, rbins, period=1, return_counts=True)

    assert np.all(counts == npts2/2)
    assert np.allclose(result, [np.mean(inner_ring_values), np.mean(outer_ring_values)], rtol = 0.001)

@pytest.mark.xfail
def test_radial_profile_3d_test4():
    """ For two tight localizations of sample1 points each surrounded by two concentric 
    shells of sample2 points, verify that both the counts and the primary result 
    of `~halotools.mock_observables.radial_profile_3d` are correct. This test differs 
    from test_radial_profile_3d_test2 in that here the normalize_rbins_by is operative. 

    In this test, PBCs have a non-trivial impact on the results.  
    """
    npts1a = 100
    xca1, yca1, zca1 = 0.3, 0.3, 0.3
    xca2, yca2, zca2 = 0.9, 0.9, 0.9

    sample1a = generate_locus_of_3d_points(npts1a, xca1, yca1, zca1, seed=fixed_seed)
    sample1b = generate_locus_of_3d_points(npts1a, xca2, yca2, zca2, seed=fixed_seed)
    sample1 = np.concatenate([sample1a, sample1b])
    npts1 = len(sample1)

    rvir = 0.013
    rvir_array = np.zeros(npts1) + rvir
    rbins = np.array([0.5, 1, 2])
    r1, r2 = 0.75*rvir, 1.5*rvir

    sample2_p1_r1 = generate_thin_shell_of_3d_points(npts1, r1, xca1, yca1, zca1, seed=fixed_seed, Lbox=1)
    sample2_p2_r1 = generate_thin_shell_of_3d_points(npts1, r1, xca2, yca2, zca2, seed=fixed_seed, Lbox=1)
    sample2_p1_r2 = generate_thin_shell_of_3d_points(npts1, r2, xca1, yca1, zca1, seed=fixed_seed, Lbox=1)
    sample2_p2_r2 = generate_thin_shell_of_3d_points(npts1, r2, xca2, yca2, zca2, seed=fixed_seed, Lbox=1)
    sample2 = np.concatenate([sample2_p1_r1, sample2_p2_r1, sample2_p1_r2, sample2_p2_r2])
    npts2 = len(sample2)

    quantity_a, quantity_b = np.zeros(npts2/2) + 0.5, np.zeros(npts2/2) + 1.5
    quantity = np.concatenate([quantity_a, quantity_b])

    result, counts = radial_profile_3d(sample1, sample2, quantity, rbins, 
        return_counts=True, period=1, normalize_rbins_by=rvir_array)
    correct_counts = (npts1/2.)*(npts2/2.)
    assert np.all(counts == correct_counts)
    assert np.all(result == [0.5, 1.5])


@pytest.mark.xfail
def test_args_processing1():
    npts1, npts2 = 100, 200
    sample1 = np.random.random((npts1, 3))
    sample2 = np.random.random((npts2, 3))

    quantity = np.arange(5)
    rbins = np.linspace(4, 5, 5)

    with pytest.raises(ValueError) as err:
        result = radial_profile_3d(sample1, sample2, quantity, rbins)
    substr = "elements, but input ``sample2`` has"
    assert substr in err.value.args[0]

@pytest.mark.xfail
def test_args_processing2():
    npts1, npts2 = 100, 200
    sample1 = np.random.random((npts1, 3))
    sample2 = np.random.random((npts2, 3))

    quantity = np.zeros(npts2)
    rbins = np.linspace(4, 5, 5)

    norm = np.arange(5)

    with pytest.raises(ValueError) as err:
        result = radial_profile_3d(sample1, sample2, quantity, rbins, normalize_rbins_by=norm)
    substr = "elements, but input ``sample1`` has"
    assert substr in err.value.args[0]




