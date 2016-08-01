""" Module providing unit-testing for the `~halotools.mock_observables.angular_tpcf` function.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.tests.helper import pytest
from astropy.utils.misc import NumpyRNGContext

from ..angular_tpcf import angular_tpcf

from ....utils import sample_spherical_surface
from ....custom_exceptions import HalotoolsError

slow = pytest.mark.slow

__all__ = ('test_angular_tpcf1', )

fixed_seed = 43


def test_angular_tpcf1():
    Npts = 1000
    angular_coords = sample_spherical_surface(Npts)

    theta_bins = np.logspace(-2, 1, 10)
    w = angular_tpcf(angular_coords, theta_bins)


