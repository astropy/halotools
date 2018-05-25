""" Module providing testing for the `~halotools.mock_observables.idx_in_cylinders` function.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pytest
from astropy.utils.misc import NumpyRNGContext

from .pure_python_counts_in_cells import pure_python_idx_in_cylinders

from ..counts_in_cylinders import idx_in_cylinders

from ...tests.cf_helpers import generate_locus_of_3d_points, generate_3d_regular_mesh

fixed_seed = 43
seed_list = np.arange(5).astype(int)


def test_idx_in_cylinders_brute_force1():
    """
    """
    npts1 = 100
    npts2 = 90

    for seed in seed_list:
        with NumpyRNGContext(seed):
            sample1 = np.random.random((npts1, 3))
            sample2 = np.random.random((npts2, 3))

        rp_max = np.zeros(npts1) + 0.2
        pi_max = np.zeros(npts1) + 0.2
        brute_force_result = pure_python_idx_in_cylinders(sample1, sample2, rp_max, pi_max)
        result = idx_in_cylinders(sample1, sample2, rp_max, pi_max, num_threads=2)

        assert np.all(_sort(result) == _sort(brute_force_result))

def _sort(indexes):
    return np.sort(indexes, order=["i1", "i2"])

