""" Unit-testing module for weighted_npairs_s_mu
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import pytest
import numpy as np
from astropy.utils.misc import NumpyRNGContext

from ..npairs_s_mu import npairs_s_mu
from ..weighted_npairs_s_mu import weighted_npairs_s_mu


__all__ = ('test1', )

fixed_seed = 43


def test1():
    """
    """
    Npts = 1000
    with NumpyRNGContext(fixed_seed):
        random_sample = np.random.random((Npts, 3))
    period = np.array([1.0, 1.0, 1.0])
    # define bins
    s_bins = np.array([0.0, 0.1, 0.2, 0.3])
    N_mu_bins = 100
    mu_bins = np.linspace(0, 1.0, N_mu_bins)
    Npts = len(random_sample)

    weights1 = np.ones(Npts)
    weights2 = np.ones(Npts)

    # count pairs using optimized double tree pair counter
    unweighted_counts1 = npairs_s_mu(random_sample, random_sample, s_bins, mu_bins, period=period)
    unweighted_counts2, weighted_counts = weighted_npairs_s_mu(random_sample, random_sample,
            weights1, weights2, s_bins, mu_bins, period=period)

    assert np.all(unweighted_counts1 == unweighted_counts2)
    assert np.all(unweighted_counts1 == weighted_counts)


def test2():
    """
    """
    Npts = 1000
    with NumpyRNGContext(fixed_seed+1):
        random_sample = np.random.random((Npts, 3))
        weights1 = np.random.rand(Npts)
        weights2 = np.random.rand(Npts)
    period = np.array([1.0, 1.0, 1.0])
    # define bins
    s_bins = np.array([0.0, 0.1, 0.2, 0.3])
    N_mu_bins = 100
    mu_bins = np.linspace(0, 1.0, N_mu_bins)
    Npts = len(random_sample)

    # count pairs using optimized double tree pair counter
    unweighted_counts1 = npairs_s_mu(random_sample, random_sample, s_bins, mu_bins, period=period)
    unweighted_counts2, weighted_counts = weighted_npairs_s_mu(random_sample, random_sample,
            weights1, weights2, s_bins, mu_bins, period=period)

    assert np.all(unweighted_counts1 == unweighted_counts2)
    assert np.all(unweighted_counts1 != weighted_counts)

def test_weight_consistency():
    """
    """
    Npts = 1000
    with NumpyRNGContext(fixed_seed):
        random_sample = np.random.random((Npts, 3))
    period = np.array([1.0, 1.0, 1.0])
    # define bins
    s_bins = np.array([0.0, 0.1, 0.2, 0.3])
    N_mu_bins = 100
    mu_bins = np.linspace(0, 1.0, N_mu_bins)
    Npts = len(random_sample)

    weights1 = np.ones(Npts)
    weights2 = np.ones(Npts)

    unweighted_counts_serial, weighted_counts_serial = weighted_npairs_s_mu(random_sample, random_sample,
            weights1, weights2, s_bins, mu_bins, period=period, num_threads=1)

    unweighted_counts_parallel, weighted_counts_parallel = weighted_npairs_s_mu(random_sample, random_sample,
            weights1, weights2, s_bins, mu_bins, period=period, num_threads=2)
    
    #since weights are all equal to 1, weighted and unweighted counts should be the same
    assert np.all(unweighted_counts_serial == weighted_counts_serial)
    assert np.all(unweighted_counts_parallel == weighted_counts_parallel)

def test_parallel_serial_consistency():
    """
    """
    Npts = 1000
    with NumpyRNGContext(fixed_seed):
        random_sample = np.random.random((Npts, 3))
    period = np.array([1.0, 1.0, 1.0])
    # define bins
    s_bins = np.array([0.0, 0.1, 0.2, 0.3])
    N_mu_bins = 100
    mu_bins = np.linspace(0, 1.0, N_mu_bins)
    Npts = len(random_sample)

    weights1 = np.ones(Npts)
    weights2 = np.ones(Npts)

    unweighted_counts_serial, weighted_counts_serial = weighted_npairs_s_mu(random_sample, random_sample,
            weights1, weights2, s_bins, mu_bins, period=period, num_threads=1)

    unweighted_counts_parallel, weighted_counts_parallel = weighted_npairs_s_mu(random_sample, random_sample,
            weights1, weights2, s_bins, mu_bins, period=period, num_threads=3)
    
    #parallel and serial approaches should return the same answer
    assert np.all(unweighted_counts_serial == unweighted_counts_parallel)
    assert np.all(weighted_counts_serial == weighted_counts_parallel)
