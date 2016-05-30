""" Module providing unit-testing for the
`~halotools.mock_observables.tpcf_one_two_halo_decomp` function.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext

from ..tpcf_one_two_halo_decomp import tpcf_one_two_halo_decomp

from astropy.tests.helper import pytest
slow = pytest.mark.slow

__all__=['test_tpcf_one_two_halo_auto_periodic', 'test_tpcf_one_two_halo_cross_periodic']

#create toy data to test functions
period = np.array([1.0, 1.0, 1.0])
rbins = np.linspace(0.001, 0.3, 5)
rmax = rbins.max()

fixed_seed = 43


@slow
def test_tpcf_one_two_halo_auto_periodic():
    """
    test the tpcf_one_two_halo autocorrelation with periodic boundary conditions
    """
    Npts = 100
    with NumpyRNGContext(fixed_seed):
        IDs1 = np.random.randint(0, 11, Npts)
        sample1 = np.random.random((Npts, 3))

    result = tpcf_one_two_halo_decomp(sample1, IDs1, rbins, sample2=None,
      randoms=None, period=period,
      max_sample_size=int(1e4), estimator='Natural')

    assert len(result)==2, "wrong number of correlation functions returned."


@slow
def test_tpcf_one_two_halo_cross_periodic():
    """
    test the tpcf_one_two_halo cross-correlation with periodic boundary conditions
    """
    Npts = 100
    with NumpyRNGContext(fixed_seed):
        IDs1 = np.random.randint(0, 11, Npts)
        IDs2 = np.random.randint(0, 11, Npts)
        sample1 = np.random.random((Npts, 3))
        sample2 = np.random.random((Npts, 3))

    result = tpcf_one_two_halo_decomp(sample1, IDs1, rbins, sample2=sample2,
      sample2_host_halo_id=IDs2, randoms=None,
      period=period, max_sample_size=int(1e4),
      estimator='Natural', approx_cell1_size=[rmax, rmax, rmax],
      approx_cell2_size=[rmax, rmax, rmax],
      approx_cellran_size=[rmax, rmax, rmax])

    assert len(result)==6, "wrong number of correlation functions returned."
