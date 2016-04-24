#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function,
    unicode_literals)

import numpy as np
import pytest 

from ..npairs_xy_z import npairs_xy_z
from ..pairs import xy_z_npairs as pure_python_brute_force_xy_z_npairs

from ...tests.cf_helpers import generate_locus_of_3d_points
from ...tests.cf_helpers import generate_3d_regular_mesh

__all__ = ('test_npairs_xy_z_tight_locus_xy1', )

def test_npairs_xy_z_tight_locus_xy1():
    """ Verify that `halotools.mock_observables.npairs_xy_z` returns 
    the correct counts for two tight loci of points. 

    In this test, PBCs are irrelevant
    """
    npts1, npts2 = 100, 90
    data1 = generate_locus_of_3d_points(npts1, xc=0.1, yc=0.1, zc=0.1)
    data2 = generate_locus_of_3d_points(npts2, xc=0.1, yc=0.2, zc=0.1)

    rp_bins = np.array((0.05, 0.15, 0.3))
    pi_bins = np.array([0, np.max(rp_bins)])

    result = npairs_xy_z(data1, data2, rp_bins, pi_bins, period=1)
    assert np.all(result[:,1] == [0, npts1*npts2, npts1*npts2])

def test_xy_z_npairs_periodic():
    """
    test xy_z_npairs with periodic boundary conditions.
    """
    npts1, npts2 = 100, 90
    data1 = np.random.random((npts1, 3))
    data2 = np.random.random((npts2, 3))

    rp_bins = np.arange(0,0.31,0.1)
    pi_bins = np.arange(0,0.31,0.1)

    result = npairs_xy_z(data1, data2, rp_bins, pi_bins, period=1)
    test_result = pure_python_brute_force_xy_z_npairs(data1, data2, rp_bins, pi_bins, period=1)

    assert np.shape(result)==(len(rp_bins),len(pi_bins))
    assert np.all(result == test_result)
