""" Module providing unit-testing for the `~halotools.mock_observables.marked_tpcf` function.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from astropy.tests.helper import pytest
from astropy.utils.misc import NumpyRNGContext

from ..marked_tpcf import marked_tpcf

slow = pytest.mark.slow

fixed_seed = 43


__all__ = ('test_marked_tpcf_auto_periodic',
           'test_marked_tpcf_auto_nonperiodic',
           'test_marked_tpcf_cross1', 'test_marked_tpcf_cross_consistency')


def test_marked_tpcf_auto_periodic():
    """
    test marked_tpcf auto correlation with periodic boundary conditions
    """
    Npts = 100
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts, 3))

    rbins = np.linspace(0.001, 0.25, 5)
    period = 1

    weight_func_id = 1
    weights1 = np.random.random(Npts)

    #with randoms
    result = marked_tpcf(sample1, rbins, sample2=None, marks1=weights1, marks2=None,
        period=period, num_threads=1, weight_func_id=weight_func_id)

    assert result.ndim == 1, "More than one correlation function returned erroneously."


def test_marked_tpcf_auto_nonperiodic():
    """
    test marked_tpcf auto correlation without periodic boundary conditions
    """
    Npts = 100
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts, 3))

    rbins = np.linspace(0.001, 0.25, 5)

    weight_func_id = 1
    weights1 = np.random.random(Npts)

    #with randoms
    result = marked_tpcf(sample1, rbins, sample2=None, marks1=weights1, marks2=None,
        period=None, num_threads=1, weight_func_id=weight_func_id)

    assert result.ndim == 1, "More than one correlation function returned erroneously."


def test_marked_tpcf_cross1():
    """
    """
    Npts = 100
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts, 3))
        sample2 = np.random.random((Npts, 3))

    rbins = np.linspace(0.001, 0.25, 5)
    period = 1

    weights1 = np.random.random(Npts)
    weights2 = np.random.random(Npts)
    weight_func_id = 1

    result = marked_tpcf(sample1, rbins, sample2=sample2,
        marks1=weights1, marks2=weights2,
        period=period, num_threads='max', weight_func_id=weight_func_id)


def test_marked_tpcf_cross_consistency():
    """
    """
    Npts = 100
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts, 3))
        sample2 = np.random.random((Npts, 3))

    rbins = np.linspace(0.001, 0.25, 5)
    period = 1

    weights1 = np.random.random(Npts)
    weights2 = np.random.random(Npts)
    weight_func_id = 1

    cross_mark1 = marked_tpcf(sample1, rbins, sample2=sample2,
        marks1=weights1, marks2=weights2,
        period=period, num_threads=1, weight_func_id=weight_func_id,
        do_auto=False, normalize_by='number_counts')

    auto1, cross_mark2, auto2 = marked_tpcf(sample1, rbins, sample2=sample2,
        marks1=weights1, marks2=weights2,
        period=period, num_threads=1, weight_func_id=weight_func_id, normalize_by='number_counts')

    auto1b, auto2b = marked_tpcf(sample1, rbins, sample2=sample2,
        marks1=weights1, marks2=weights2,
        period=period, num_threads=1, weight_func_id=weight_func_id,
        do_cross=False, normalize_by='number_counts')

    assert np.all(cross_mark1 == cross_mark2)
