""" Module providing unit-testing for the `~halotools.mock_observables.angular_tpcf` function.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pytest
from astropy.utils.misc import NumpyRNGContext

from ..angular_tpcf import angular_tpcf

from ....utils import sample_spherical_surface
from ....custom_exceptions import HalotoolsError

slow = pytest.mark.slow

__all__ = ('test_angular_tpcf1', )

fixed_seed = 43


def test_angular_tpcf1():
    Npts1 = 1000
    angular_coords1 = sample_spherical_surface(Npts1, seed=fixed_seed)

    theta_bins = np.logspace(-1, 1, 5)
    __ = angular_tpcf(angular_coords1, theta_bins)


def test_angular_tpcf2():
    Npts1, Npts2 = 1000, 500
    angular_coords1 = sample_spherical_surface(Npts1, seed=fixed_seed)
    angular_coords2 = sample_spherical_surface(Npts2, seed=fixed_seed+1)

    theta_bins = np.logspace(-1, 1, 5)
    w1 = angular_tpcf(angular_coords1, theta_bins, sample2=angular_coords1, do_cross=False)
    w2 = angular_tpcf(angular_coords1, theta_bins, sample2=angular_coords2, do_auto=False)
    assert not np.all(w1 == w2)


def test_angular_tpcf3():
    Npts1, Npts2, Nran = 1000, 500, 4000
    angular_coords1 = sample_spherical_surface(Npts1, seed=fixed_seed)
    angular_coords2 = sample_spherical_surface(Npts2, seed=fixed_seed+1)
    randoms = sample_spherical_surface(Nran, seed=fixed_seed)

    theta_bins = np.logspace(-1, 1, 5)
    __ = angular_tpcf(angular_coords1, theta_bins, sample2=angular_coords2, randoms=randoms)


def test_angular_tpcf_auto_consistency():
    Npts1, Npts2, Nran = 1000, 500, 4000
    angular_coords1 = sample_spherical_surface(Npts1, seed=fixed_seed)
    angular_coords2 = sample_spherical_surface(Npts2, seed=fixed_seed+1)
    randoms = sample_spherical_surface(Nran, seed=fixed_seed)

    theta_bins = np.logspace(-1, 1, 5)
    result_a = angular_tpcf(angular_coords1, theta_bins, sample2=angular_coords2,
        randoms=randoms, do_auto=True, do_cross=True)
    result_b = angular_tpcf(angular_coords1, theta_bins, sample2=angular_coords2,
        randoms=randoms, do_auto=True, do_cross=False)
    w11_a, w12_a, w22_a = result_a
    w11_b, w22_b = result_b
    assert np.allclose(w11_a, w11_b)
    assert np.allclose(w22_a, w22_b)


def test_angular_tpcf_cross_consistency():
    Npts1, Npts2, Nran = 1000, 500, 4000
    angular_coords1 = sample_spherical_surface(Npts1, seed=fixed_seed)
    angular_coords2 = sample_spherical_surface(Npts2, seed=fixed_seed+1)
    randoms = sample_spherical_surface(Nran, seed=fixed_seed)

    theta_bins = np.logspace(-1, 1, 5)
    result_a = angular_tpcf(angular_coords1, theta_bins, sample2=angular_coords2,
        randoms=randoms, do_auto=True, do_cross=True)
    result_b = angular_tpcf(angular_coords1, theta_bins, sample2=angular_coords2,
        randoms=randoms, do_auto=False, do_cross=True)
    w11_a, w12_a, w22_a = result_a
    w12_b = result_b
    assert np.allclose(w12_a, w12_b)


def test_exception_handling1():
    Npts1 = 1000
    angular_coords1 = sample_spherical_surface(Npts1, seed=fixed_seed)

    theta_bins = np.logspace(-1, 200, 5)
    with pytest.raises(HalotoolsError) as err:
        __ = angular_tpcf(angular_coords1, theta_bins)
    substr = "cannot be larger than 180.0 deg."
    assert substr in err.value.args[0]


def test_exception_handling2():
    Npts1 = 1000
    angular_coords1 = sample_spherical_surface(Npts1, seed=fixed_seed)

    theta_bins = np.logspace(-1, 1, 5)
    with pytest.raises(HalotoolsError) as err:
        __ = angular_tpcf(angular_coords1, theta_bins, do_auto='yes')
    substr = "`do_auto` and `do_cross` keywords must be of type boolean."
    assert substr in err.value.args[0]


def test_exception_handling3():
    Npts1 = 1000
    angular_coords1 = sample_spherical_surface(Npts1, seed=fixed_seed)

    theta_bins = [0.01, 0.03, 0.02]
    with pytest.raises(HalotoolsError) as err:
        __ = angular_tpcf(angular_coords1, theta_bins)
    substr = "must be a monotonically increasing 1-D"
    assert substr in err.value.args[0]
