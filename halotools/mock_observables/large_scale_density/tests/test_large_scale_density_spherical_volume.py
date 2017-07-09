"""
"""
from __future__ import (absolute_import, division, print_function)
import numpy as np
import pytest

from ..large_scale_density_spherical_volume import large_scale_density_spherical_volume

from ...tests.cf_helpers import generate_locus_of_3d_points
from ....custom_exceptions import HalotoolsError

__all__ = ('test_large_scale_density_spherical_volume1', )

fixed_seed = 43


def test_large_scale_density_spherical_volume_exception_handling():
    """
    """
    npts1, npts2 = 100, 200
    sample = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.1, seed=fixed_seed)
    tracers = generate_locus_of_3d_points(npts2, xc=0.15, yc=0.1, zc=0.1, seed=fixed_seed)
    radius = 0.1

    with pytest.raises(HalotoolsError) as err:
        result = large_scale_density_spherical_volume(
            sample, tracers, radius)
    substr = "If period is None, you must pass in ``sample_volume``."
    assert substr in err.value.args[0]

    with pytest.raises(HalotoolsError) as err:
        result = large_scale_density_spherical_volume(
            sample, tracers, radius, period=[1, 1])
    substr = "Input ``period`` must either be a float or length-3 sequence"
    assert substr in err.value.args[0]

    with pytest.raises(HalotoolsError) as err:
        result = large_scale_density_spherical_volume(
            sample, tracers, radius, period=1, sample_volume=0.4)
    substr = "If period is not None, do not pass in sample_volume"
    assert substr in err.value.args[0]


def test_large_scale_density_spherical_volume1():
    """
    """
    npts1, npts2 = 100, 200
    sample = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.1, seed=fixed_seed)
    tracers = generate_locus_of_3d_points(npts2, xc=0.15, yc=0.1, zc=0.1, seed=fixed_seed)
    radius = 0.1
    result = large_scale_density_spherical_volume(
        sample, tracers, radius, period=1)

    environment_volume = (4/3.)*np.pi*radius**3
    correct_answer = 200/environment_volume
    assert np.allclose(result, correct_answer, rtol=0.001)


def test_large_scale_density_spherical_volume2():
    """
    """
    npts1, npts2 = 100, 200
    sample = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.1, seed=fixed_seed)
    tracers = generate_locus_of_3d_points(npts2, xc=0.95, yc=0.1, zc=0.1, seed=fixed_seed)
    radius = 0.2
    result = large_scale_density_spherical_volume(
        sample, tracers, radius, period=[1, 1, 1], norm_by_mean_density=True)
    mean_density = float(npts2)

    environment_volume = (4/3.)*np.pi*radius**3
    correct_answer = 200/environment_volume/mean_density
    assert np.allclose(result, correct_answer, rtol=0.001)
