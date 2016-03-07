#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)
import numpy as np 
from astropy.tests.helper import pytest

from .cf_helpers import generate_locus_of_3d_points
from ..nearest_neighbor import nearest_neighbor

__all__ = ('test_diffusely_distributed_points', 'test_tight_locus')

npts = 100

def test_diffusely_distributed_points():
    """ Verify that the `~halotools.mock_observables.nearest_neighbor` 
    function returns a result without raising an exception when passed 
    two sets of points that evenly sample the full box at random. 
    """
    sample1 = np.random.rand(npts, 3)
    sample2 = np.random.rand(npts, 3)
    r_max = 0.2
    nth_nearest = 1
    nn = nearest_neighbor(sample1, sample2, r_max, 
        nth_nearest=nth_nearest)
    nn2 = nearest_neighbor(sample1, sample2, r_max, 
        nth_nearest=2, period=1.)




def test_tight_locus():
    """ Verify that the `~halotools.mock_observables.nearest_neighbor` 
    function returns a result without raising an exception when passed 
    two distant sets of points, each in a tight locus. 
    """

    npts1, npts2 = npts, npts
    sample1 = generate_locus_of_3d_points(npts1, 
        xc=0.1, yc=0.1, zc=0.1, epsilon=0.01)
    assert sample1.shape == (npts1, 3)
    assert np.all(sample1 >= 0.09)
    assert np.all(sample1 <= 0.11)
    sample2 = generate_locus_of_3d_points(npts2, 
        xc=0.9, yc=0.9, zc=0.9, epsilon=0.01)

    r_max = 0.5
    nn = nearest_neighbor(sample1, sample2, r_max, nth_nearest=1)






