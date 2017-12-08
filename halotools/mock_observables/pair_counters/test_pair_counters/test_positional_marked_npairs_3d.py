"""
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
import pytest
from astropy.utils.misc import NumpyRNGContext
from astropy.config.paths import _find_home

from ..pairs import wnpairs as pure_python_weighted_pairs
from ..marked_npairs_3d import positional_marked_npairs_3d, _func_signature_int_from_wfunc

from ....custom_exceptions import HalotoolsError

slow = pytest.mark.slow

error_msg = ("\nThe `test_positional_marked_npairs_wfuncs_behavior` function performs \n"
    "non-trivial checks on the returned values of marked correlation functions\n"
    "calculated on a set of points with uniform weights.\n"
    "One such check failed.\n")

__all__ = ('test_positional_marked_npairs_3d_periodic', )

fixed_seed = 43


# Determine whether the machine is mine
# This will be used to select tests whose
# returned values depend on the configuration
# of my personal cache directory files
aph_home = '/Users/aphearin'
detected_home = _find_home()
if aph_home == detected_home:
    APH_MACHINE = True
else:
    APH_MACHINE = False


def retrieve_mock_data(Npts, Npts2, Lbox):

    # set up a regular grid of points to test pair counters
    epsilon = 0.001
    gridx = np.linspace(0, Lbox-epsilon, Npts2)
    gridy = np.linspace(0, Lbox-epsilon, Npts2)
    gridz = np.linspace(0, Lbox-epsilon, Npts2)
    xx, yy, zz = np.array(np.meshgrid(gridx, gridy, gridz))
    xx = xx.flatten()
    yy = yy.flatten()
    zz = zz.flatten()

    grid_points = np.vstack([xx, yy, zz]).T
    rbins = np.array([0.0, 0.1, 0.2, 0.3])
    period = np.array([Lbox, Lbox, Lbox])

    return grid_points, rbins, period


def test_marked_npairs_3d_periodic():
    """
    Function tests marked_npairs with periodic boundary conditions.
    """
    Npts = 1000
    with NumpyRNGContext(fixed_seed):
        random_sample = np.random.random((Npts, 3))
        ran_weights1 = np.random.random((Npts, 1))

    period = np.array([1.0, 1.0, 1.0])
    rbins = np.array([0.0, 0.1, 0.2, 0.3])

    result = marked_npairs_3d(random_sample, random_sample,
        rbins, period=period, weights1=ran_weights1, weights2=ran_weights1, weight_func_id=1)

    test_result = pure_python_weighted_pairs(random_sample, random_sample, rbins,
        period=period, weights1=ran_weights1, weights2=ran_weights1)

    assert np.allclose(test_result, result, rtol=1e-05), "pair counts are incorrect"


