""" Module providing testing of isolation functions by comparing to naive serial algorithms
implemented in pure python
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext

from .pure_python_isolation import naive_spherical_isolation, naive_cylindrical_isolation

from ..spherical_isolation import spherical_isolation
from ..cylindrical_isolation import cylindrical_isolation


__all__ = ('test_spherical_isolation1', )
fixed_seed = 43


def test_spherical_isolation1():
    """
    """
    npts1, npts2 = 100, 103
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts1, 3))
        sample2 = np.random.random((npts2, 3))

    r_max = 0.2
    iso1 = spherical_isolation(sample1, sample2, r_max, period=1)

    x1arr, y1arr, z1arr = sample1[:, 0], sample1[:, 1], sample1[:, 2]
    x2arr, y2arr, z2arr = sample2[:, 0], sample2[:, 1], sample2[:, 2]
    xperiod, yperiod, zperiod = 1, 1, 1
    iso2 = naive_spherical_isolation(x1arr, y1arr, z1arr, x2arr, y2arr, z2arr,
        r_max, xperiod, yperiod, zperiod)

    assert np.all(iso1 == iso2)
    #  ensure the test is non-trivial
    assert np.any(iso1)
    assert not np.all(iso1)


def test_spherical_isolation2():
    """
    """
    npts1, npts2 = 100, 103
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts1, 3))
        sample2 = np.random.random((npts2, 3))

    r_max = 0.2
    iso1 = spherical_isolation(sample1, sample2, r_max, period=None)

    x1arr, y1arr, z1arr = sample1[:, 0], sample1[:, 1], sample1[:, 2]
    x2arr, y2arr, z2arr = sample2[:, 0], sample2[:, 1], sample2[:, 2]
    xperiod, yperiod, zperiod = np.inf, np.inf, np.inf
    iso2 = naive_spherical_isolation(x1arr, y1arr, z1arr, x2arr, y2arr, z2arr,
        r_max, xperiod, yperiod, zperiod)

    assert np.all(iso1 == iso2)
    #  ensure the test is non-trivial
    assert np.any(iso1)
    assert not np.all(iso1)


def test_cylindrical_isolation1():
    """
    """
    npts1, npts2 = 100, 103
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts1, 3))
        sample2 = np.random.random((npts2, 3))

    rp_max, pi_max = 0.2, 0.2
    iso1 = cylindrical_isolation(sample1, sample2, rp_max, pi_max, period=1)

    x1arr, y1arr, z1arr = sample1[:, 0], sample1[:, 1], sample1[:, 2]
    x2arr, y2arr, z2arr = sample2[:, 0], sample2[:, 1], sample2[:, 2]
    xperiod, yperiod, zperiod = 1, 1, 1
    iso2 = naive_cylindrical_isolation(x1arr, y1arr, z1arr, x2arr, y2arr, z2arr,
        rp_max, pi_max, xperiod, yperiod, zperiod)

    assert np.all(iso1 == iso2)
    #  ensure the test is non-trivial
    assert np.any(iso1)
    assert not np.all(iso1)


def test_cylindrical_isolation2():
    """
    """
    npts1, npts2 = 100, 103
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((npts1, 3))
        sample2 = np.random.random((npts2, 3))

    rp_max, pi_max = 0.2, 0.2
    iso1 = cylindrical_isolation(sample1, sample2, rp_max, pi_max, period=None)

    x1arr, y1arr, z1arr = sample1[:, 0], sample1[:, 1], sample1[:, 2]
    x2arr, y2arr, z2arr = sample2[:, 0], sample2[:, 1], sample2[:, 2]
    xperiod, yperiod, zperiod = np.inf, np.inf, np.inf
    iso2 = naive_cylindrical_isolation(x1arr, y1arr, z1arr, x2arr, y2arr, z2arr,
        rp_max, pi_max, xperiod, yperiod, zperiod)

    assert np.all(iso1 == iso2)
    #  ensure the test is non-trivial
    assert np.any(iso1)
    assert not np.all(iso1)

