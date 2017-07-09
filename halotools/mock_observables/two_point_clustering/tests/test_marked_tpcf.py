""" Module providing unit-testing for the `~halotools.mock_observables.marked_tpcf` function.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import pytest
from astropy.utils.misc import NumpyRNGContext

from ..marked_tpcf import marked_tpcf

from ....custom_exceptions import HalotoolsError

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
    with NumpyRNGContext(fixed_seed):
        weights1 = np.random.random(Npts)

    # with randoms
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
    with NumpyRNGContext(fixed_seed):
        weights1 = np.random.random(Npts)

    # with randoms
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

    with NumpyRNGContext(fixed_seed):
        weights1 = np.random.random(Npts)
        weights2 = np.random.random(Npts)
    weight_func_id = 1

    result = marked_tpcf(sample1, rbins, sample2=sample2,
        marks1=weights1, marks2=weights2,
        period=period, num_threads=1, weight_func_id=weight_func_id)


def test_marked_tpcf_cross_consistency():
    """
    """
    Npts = 100
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts, 3))
        sample2 = np.random.random((Npts, 3))

    rbins = np.linspace(0.001, 0.25, 5)
    period = 1

    with NumpyRNGContext(fixed_seed):
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


def test_iterations():
    Npts1, Npts2 = 100, 90
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts1, 3))
        sample2 = np.random.random((Npts2, 3))

    rbins = np.linspace(0.001, 0.25, 5)
    period = 1

    weight_func_id = 1
    with NumpyRNGContext(fixed_seed):
        weights1 = np.random.random(Npts1)
        weights2 = np.random.random(Npts2)

    result1 = marked_tpcf(sample1, rbins, sample2=sample2, marks1=weights1, marks2=weights2,
        period=period, num_threads=1, weight_func_id=weight_func_id, seed=fixed_seed)
    result2 = marked_tpcf(sample1, rbins, sample2=sample2, marks1=weights1, marks2=weights2,
        period=period, num_threads=1, weight_func_id=weight_func_id, seed=fixed_seed, iterations=3)


def test_exception_handling1():
    Npts1, Npts2 = 100, 90
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts1, 3))
        sample2 = np.random.random((Npts2, 3))

    rbins = np.linspace(0.001, 0.25, 5)
    period = 1

    weight_func_id = 1
    with NumpyRNGContext(fixed_seed):
        weights1 = np.random.random(Npts1)
        weights2 = np.random.random(Npts2)

    with pytest.raises(ValueError) as err:
        __ = marked_tpcf(sample1, rbins, sample2=sample2, marks1=weights1, marks2=weights2,
            period=period, num_threads=1, weight_func_id=weight_func_id, seed=fixed_seed,
            normalize_by='Arnold Schwarzenegger')
    substr = "`normalize_by` parameter not recognized"
    assert substr in err.value.args[0]


def test_exception_handling2():
    Npts1, Npts2 = 100, 90
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts1, 3))
        sample2 = np.random.random((Npts2, 3))

    rbins = np.linspace(0.001, 0.25, 5)
    period = 1

    weight_func_id = 1
    with NumpyRNGContext(fixed_seed):
        weights1 = np.random.random(Npts1)
        weights2 = np.random.random(Npts2)

    with pytest.raises(ValueError) as err:
        __ = marked_tpcf(sample1, rbins, sample2=sample2, marks1=weights1, marks2=weights2,
            period=period, num_threads=1, weight_func_id=1.1, seed=fixed_seed)
    substr = "weight_func_id parameter must be an integer ID of a weighting function"
    assert substr in err.value.args[0]


def test_exception_handling3():
    Npts1, Npts2 = 100, 90
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts1, 3))
        sample2 = np.random.random((Npts2, 3))

    rbins = np.linspace(0.001, 0.25, 5)
    period = 1

    weight_func_id = 1
    with NumpyRNGContext(fixed_seed):
        weights1 = np.random.random((Npts1, 4))
        weights2 = np.random.random((Npts2, 4))

    with pytest.raises(HalotoolsError) as err:
        __ = marked_tpcf(sample1, rbins, sample2=sample2, marks1=weights1, marks2=weights2,
            period=period, num_threads=1, weight_func_id=weight_func_id, seed=fixed_seed)
    substr = "does not have a consistent shape with `sample1`"
    assert substr in err.value.args[0]


def test_exception_handling4():
    Npts1, Npts2 = 100, 90
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts1, 3))
        sample2 = np.random.random((Npts2, 3))

    rbins = np.linspace(0.001, 0.25, 5)
    period = 1

    weight_func_id = 1
    with NumpyRNGContext(fixed_seed):
        weights1 = np.random.random(Npts1)
        weights2 = np.random.random((Npts2, 4))

    with pytest.raises(HalotoolsError) as err:
        __ = marked_tpcf(sample1, rbins, sample2=sample2, marks1=weights1, marks2=weights2,
            period=period, num_threads=1, weight_func_id=weight_func_id, seed=fixed_seed)
    substr = "does not have a consistent shape with `sample2`."
    assert substr in err.value.args[0]


def test_exception_handling5():
    Npts1, Npts2 = 100, 90
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts1, 3))
        sample2 = np.random.random((Npts2, 3))

    rbins = np.linspace(0.001, 0.25, 5)
    period = 1

    weight_func_id = 1
    with NumpyRNGContext(fixed_seed):
        weights1 = np.random.random(Npts2)
        weights2 = np.random.random(Npts2)

    with pytest.raises(HalotoolsError) as err:
        __ = marked_tpcf(sample1, rbins, sample2=sample2, marks1=weights1, marks2=weights2,
            period=period, num_threads=1, weight_func_id=1, seed=fixed_seed)
    substr = "`marks1` must have same length as `sample1`."
    assert substr in err.value.args[0]


def test_exception_handling6():
    Npts1, Npts2 = 100, 90
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts1, 3))
        sample2 = np.random.random((Npts2, 3))

    rbins = np.linspace(0.001, 0.25, 5)
    period = 1

    weight_func_id = 1
    with NumpyRNGContext(fixed_seed):
        weights1 = np.random.random(Npts1)
        weights2 = np.random.random(Npts1)

    with pytest.raises(HalotoolsError) as err:
        __ = marked_tpcf(sample1, rbins, sample2=sample2, marks1=weights1, marks2=weights2,
            period=period, num_threads=1, weight_func_id=1, seed=fixed_seed)
    substr = "`marks2` must have same length as `sample2`."
    assert substr in err.value.args[0]


def test_exception_handling7():
    Npts1, Npts2 = 100, 90
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts1, 3))
        sample2 = np.random.random((Npts2, 3))

    rbins = np.linspace(0.001, 0.25, 5)
    period = 1

    weight_func_id = 1
    with NumpyRNGContext(fixed_seed):
        weights1 = np.random.random(Npts1)
        weights2 = np.random.random(Npts2)
        randomize_marks = np.random.random(Npts2)

    with pytest.raises(HalotoolsError) as err:
        __ = marked_tpcf(sample1, rbins, sample2=sample2, marks1=weights1, marks2=weights2,
            period=period, num_threads=1, weight_func_id=1, seed=fixed_seed, randomize_marks=randomize_marks)
    substr = "`randomize_marks` must have same length"
    assert substr in err.value.args[0]


def test_exception_handling8():
    """
    test marked_tpcf auto correlation with periodic boundary conditions
    """
    Npts = 100
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((Npts, 3))

    rbins = np.linspace(0.001, 0.25, 5)
    period = 1

    weight_func_id = 1
    with NumpyRNGContext(fixed_seed):
        weights1 = np.random.random(Npts)

    with pytest.raises(ValueError) as err:
        result = marked_tpcf(sample1, rbins, sample2=None, marks1=weights1, marks2=None,
            period=period, num_threads=1, weight_func_id=weight_func_id, do_auto='yes')
    substr = "`do_auto` and `do_cross` keywords must be boolean-valued."
    assert substr in err.value.args[0]
