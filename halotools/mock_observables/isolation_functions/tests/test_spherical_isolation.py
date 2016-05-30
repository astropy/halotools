""" Module providing testing for the `~halotools.mock_observables.spherical_isolation` function.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext

from ..spherical_isolation import spherical_isolation

from ...tests.cf_helpers import generate_locus_of_3d_points, generate_3d_regular_mesh

__all__ = ('test_spherical_isolation1', 'test_spherical_isolation2',
    'test_spherical_isolation3', 'test_spherical_isolation4',
    'test_spherical_isolation_grid1', 'test_spherical_isolation_grid2',
    'test_shifted_randoms')

fixed_seed = 43


def test_spherical_isolation1():
    """ Verify that the `~halotools.mock_observables.spherical_isolation` function
    returns all points as isolated for two distant localizations of points.
    """
    sample1 = generate_locus_of_3d_points(100, xc=0.1, yc=0.1, zc=0.1, seed=fixed_seed)
    sample2 = generate_locus_of_3d_points(100, xc=0.9, seed=fixed_seed)
    r_max = 0.1
    iso = spherical_isolation(sample1, sample2, r_max, period=1)
    assert np.all(iso == True)

    iso = spherical_isolation(sample1, sample2, r_max)
    assert np.all(iso == True)


def test_spherical_isolation2():
    """ Verify that the `~halotools.mock_observables.spherical_isolation` function
    returns no points as isolated when a subset of ``sample2`` lies within ``sample1``
    """
    sample1 = generate_locus_of_3d_points(100, xc=0.1, yc=0.1, zc=0.1, seed=fixed_seed)
    sample2a = generate_locus_of_3d_points(100, xc=0.11, seed=fixed_seed)
    sample2b = generate_locus_of_3d_points(100, xc=1.11, seed=fixed_seed)
    sample2 = np.concatenate((sample2a, sample2b))
    r_max = 0.3
    iso = spherical_isolation(sample1, sample2, r_max)
    assert np.all(iso == False)


def test_spherical_isolation3():
    """ Verify that the `~halotools.mock_observables.spherical_isolation` function
    returns the correct subset of points in ``sample1`` as being isolated.

    For this test, PBCs are irrelevant.
    """
    sample1a = generate_locus_of_3d_points(100, xc=0.11, seed=fixed_seed)
    sample1b = generate_locus_of_3d_points(100, xc=1.11, seed=fixed_seed)
    sample1 = np.concatenate((sample1a, sample1b))

    sample2 = generate_locus_of_3d_points(100, xc=0.11, yc=0.1, zc=0.1, seed=fixed_seed)
    r_max = 0.2
    iso = spherical_isolation(sample1, sample2, r_max)

    assert np.all(iso[:100] == False)
    assert np.all(iso[100:] == True)


def test_spherical_isolation4():
    """ Verify that the `~halotools.mock_observables.spherical_isolation` function
    returns the correct subset of points in ``sample1`` as being isolated.

    For this test, PBCs have a non-trivial impact on the result.
    """
    sample1a = generate_locus_of_3d_points(100, xc=0.5, seed=fixed_seed)
    sample1b = generate_locus_of_3d_points(100, xc=0.99, seed=fixed_seed)
    sample1 = np.concatenate((sample1a, sample1b))

    sample2 = generate_locus_of_3d_points(100, xc=0.11, yc=0.1, zc=0.1, seed=fixed_seed)

    r_max = 0.2
    iso = spherical_isolation(sample1, sample2, r_max)
    assert np.all(iso == True)

    r_max = 0.2
    iso = spherical_isolation(sample1, sample2, r_max, period=1)
    assert np.all(iso[:100] == True)
    assert np.all(iso[100:] == False)


def test_spherical_isolation_grid1():
    """ Create a regular grid inside the unit box with points on each of the following
    nodes: 0.1, 0.3, 0.5, 0.7, 0.9. Demonstrate that all points in such a sample
    are isolated if r_max < 0.2, regardless of periodic boundary conditions.
    """
    sample1 = generate_3d_regular_mesh(5)

    r_max = 0.1
    iso = spherical_isolation(sample1, sample1, r_max)
    assert np.all(iso == True)
    iso = spherical_isolation(sample1, sample1, r_max, period=1)
    assert np.all(iso == True)

    r_max = 0.25
    iso2 = spherical_isolation(sample1, sample1, r_max)
    assert np.all(iso2 == False)
    iso2 = spherical_isolation(sample1, sample1, r_max, period=1)
    assert np.all(iso2 == False)


def test_spherical_isolation_grid2():
    """ Create two regular grids inside the unit box with points on each of the following nodes:
    0.1, 0.3, 0.5, 0.7, 0.9 for sample1, and 0.1, 0.2, 0.3, ..., 0.9 for sample2.

    Demonstrate that the `~halotools.mock_observables.spherical_isolation` function
    behaves properly in all appropriate limits of ``r_max``.
    """
    sample1 = generate_3d_regular_mesh(5)
    sample2 = generate_3d_regular_mesh(10)

    r_max = 0.001
    iso = spherical_isolation(sample1, sample2, r_max, period=1)
    assert np.all(iso == True)

    r_max = 0.2
    iso = spherical_isolation(sample1, sample2, r_max, period=1)
    assert np.all(iso == False)

    r_max = 0.001
    iso = spherical_isolation(sample1, sample2, r_max)
    assert np.all(iso == True)

    r_max = 0.2
    iso = spherical_isolation(sample1, sample2, r_max)
    assert np.all(iso == False)


def test_shifted_randoms():
    """ Begin with a set of randomly distributed points in the unit box.
    Create sample2 by applying a tiny shift to these random points.

    Demonstrate that the `~halotools.mock_observables.spherical_isolation` function
    behaves properly in all appropriate limits of ``r_max``.
    """
    npts = 1000
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts, 3))
    epsilon = 0.001
    sample2 = sample1 + epsilon

    r_max = epsilon/10.
    iso = spherical_isolation(sample1, sample2, r_max)
    assert np.all(iso == True)

    r_max = 2*epsilon
    iso = spherical_isolation(sample1, sample2, r_max)
    assert np.all(iso == False)
