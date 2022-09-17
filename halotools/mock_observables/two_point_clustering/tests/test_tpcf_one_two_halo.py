""" Module providing unit-testing for the
`~halotools.mock_observables.tpcf_one_two_halo_decomp` function.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext
import pytest

from ..tpcf_one_two_halo_decomp import tpcf_one_two_halo_decomp

from ....custom_exceptions import HalotoolsError

__all__ = [
    "test_tpcf_one_two_halo_auto_periodic",
    "test_tpcf_one_two_halo_cross_periodic",
]

# create toy data to test functions
period = np.array([1.0, 1.0, 1.0])
rbins = np.linspace(0.001, 0.3, 5)
rmax = rbins.max()

fixed_seed = 43


def test_tpcf_one_two_halo_auto_periodic():
    """
    test the tpcf_one_two_halo autocorrelation with periodic boundary conditions
    """
    Npts = 100
    with NumpyRNGContext(fixed_seed):
        IDs1 = np.random.randint(0, 11, Npts)
        sample1 = np.random.random((Npts, 3))

    result = tpcf_one_two_halo_decomp(
        sample1,
        IDs1,
        rbins,
        sample2=None,
        randoms=None,
        period=period,
        estimator="Natural",
    )

    assert len(result) == 2, "wrong number of correlation functions returned."


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

    result = tpcf_one_two_halo_decomp(
        sample1,
        IDs1,
        rbins,
        sample2=sample2,
        sample2_host_halo_id=IDs2,
        randoms=None,
        period=period,
        estimator="Natural",
        approx_cell1_size=[rmax, rmax, rmax],
        approx_cell2_size=[rmax, rmax, rmax],
        approx_cellran_size=[rmax, rmax, rmax],
    )

    assert len(result) == 6, "wrong number of correlation functions returned."


def test_tpcf_one_two_halo_auto_nonperiodic():
    """
    test the tpcf_one_two_halo autocorrelation with periodic boundary conditions
    """
    Npts, Nran = 100, 1000
    with NumpyRNGContext(fixed_seed):
        IDs1 = np.random.randint(0, 11, Npts)
        sample1 = np.random.random((Npts, 3))
        randoms = np.random.random((Nran, 3))

    result = tpcf_one_two_halo_decomp(
        sample1,
        IDs1,
        rbins,
        sample2=None,
        randoms=randoms,
        period=period,
        estimator="Natural",
    )

    assert len(result) == 2, "wrong number of correlation functions returned."


def test_tpcf_one_two_halo_cross_nonperiodic():
    """
    test the tpcf_one_two_halo cross-correlation with periodic boundary conditions
    """
    Npts, Nran = 100, 1000
    with NumpyRNGContext(fixed_seed):
        IDs1 = np.random.randint(0, 11, Npts)
        IDs2 = np.random.randint(0, 11, Npts)
        sample1 = np.random.random((Npts, 3))
        sample2 = np.random.random((Npts, 3))
        randoms = np.random.random((Nran, 3))

    result = tpcf_one_two_halo_decomp(
        sample1,
        IDs1,
        rbins,
        sample2=sample2,
        sample2_host_halo_id=IDs2,
        randoms=randoms,
        period=period,
        estimator="Natural",
        approx_cell1_size=[rmax, rmax, rmax],
        approx_cell2_size=[rmax, rmax, rmax],
        approx_cellran_size=[rmax, rmax, rmax],
    )

    assert len(result) == 6, "wrong number of correlation functions returned."


def test_tpcf_decomposition_process_args1():
    Npts = 100
    with NumpyRNGContext(fixed_seed):
        IDs1 = np.random.randint(0, 11, Npts)
        IDs2 = np.random.randint(0, 11, Npts)
        sample1 = np.random.random((Npts, 3))
        sample2 = np.random.random((Npts, 3))

    with pytest.raises(ValueError) as err:
        result = tpcf_one_two_halo_decomp(
            sample1,
            IDs1,
            rbins,
            sample2=sample2,
            sample2_host_halo_id=None,
            randoms=None,
            period=period,
            estimator="Natural",
            approx_cell1_size=[rmax, rmax, rmax],
            approx_cell2_size=[rmax, rmax, rmax],
            approx_cellran_size=[rmax, rmax, rmax],
        )
    substr = "If passing an input ``sample2``, must also pass sample2_host_halo_id"
    assert substr in err.value.args[0]


def test_tpcf_decomposition_process_args2():
    Npts = 100
    with NumpyRNGContext(fixed_seed):
        IDs1 = np.random.randint(0, 11, Npts - 1)
        IDs2 = np.random.randint(0, 11, Npts)
        sample1 = np.random.random((Npts, 3))
        sample2 = np.random.random((Npts, 3))

    with pytest.raises(HalotoolsError) as err:
        result = tpcf_one_two_halo_decomp(
            sample1,
            IDs1,
            rbins,
            sample2=sample2,
            sample2_host_halo_id=IDs2,
            randoms=None,
            period=period,
            estimator="Natural",
            approx_cell1_size=[rmax, rmax, rmax],
            approx_cell2_size=[rmax, rmax, rmax],
            approx_cellran_size=[rmax, rmax, rmax],
        )
    substr = "same length as `sample1`"
    assert substr in err.value.args[0]


def test_tpcf_decomposition_process_args3():
    Npts = 100
    with NumpyRNGContext(fixed_seed):
        IDs1 = np.random.randint(0, 11, Npts)
        IDs2 = np.random.randint(0, 11, Npts - 1)
        sample1 = np.random.random((Npts, 3))
        sample2 = np.random.random((Npts, 3))

    with pytest.raises(HalotoolsError) as err:
        result = tpcf_one_two_halo_decomp(
            sample1,
            IDs1,
            rbins,
            sample2=sample2,
            sample2_host_halo_id=IDs2,
            randoms=None,
            period=period,
            estimator="Natural",
            approx_cell1_size=[rmax, rmax, rmax],
            approx_cell2_size=[rmax, rmax, rmax],
            approx_cellran_size=[rmax, rmax, rmax],
        )
    substr = "same length as `sample2`"
    assert substr in err.value.args[0]


def test_tpcf_decomposition_process_args4():
    Npts = 100
    with NumpyRNGContext(fixed_seed):
        IDs1 = np.random.randint(0, 11, Npts)
        IDs2 = np.random.randint(0, 11, Npts)
        sample1 = np.random.random((Npts, 3))
        sample2 = np.random.random((Npts, 3))

    with pytest.raises(ValueError) as err:
        result = tpcf_one_two_halo_decomp(
            sample1,
            IDs1,
            rbins,
            sample2=sample2,
            sample2_host_halo_id=IDs2,
            randoms=None,
            period=period,
            estimator="Natural",
            approx_cell1_size=[rmax, rmax, rmax],
            approx_cell2_size=[rmax, rmax, rmax],
            approx_cellran_size=[rmax, rmax, rmax],
            do_auto="yes",
        )
    substr = "`do_auto` and `do_cross` keywords must be boolean-valued."
    assert substr in err.value.args[0]


def test_tpcf_decomposition_cross_consistency():
    Npts1, Npts2 = 100, 200
    with NumpyRNGContext(fixed_seed):
        IDs1 = np.random.randint(0, 11, Npts1)
        IDs2 = np.random.randint(0, 11, Npts2)
        sample1 = np.random.random((Npts1, 3))
        sample2 = np.random.random((Npts2, 3))

    result_a = tpcf_one_two_halo_decomp(
        sample1,
        IDs1,
        rbins,
        sample2=sample2,
        sample2_host_halo_id=IDs2,
        randoms=None,
        period=period,
        estimator="Natural",
        approx_cell1_size=[rmax, rmax, rmax],
        approx_cell2_size=[rmax, rmax, rmax],
        approx_cellran_size=[rmax, rmax, rmax],
        do_auto=True,
        do_cross=True,
    )
    result_1h_12a, result_2h_12a = result_a[2:4]

    result_b = tpcf_one_two_halo_decomp(
        sample1,
        IDs1,
        rbins,
        sample2=sample2,
        sample2_host_halo_id=IDs2,
        randoms=None,
        period=period,
        estimator="Natural",
        approx_cell1_size=[rmax, rmax, rmax],
        approx_cell2_size=[rmax, rmax, rmax],
        approx_cellran_size=[rmax, rmax, rmax],
        do_auto=False,
        do_cross=True,
    )
    result_1h_12b, result_2h_12b = result_b[0:2]

    assert np.allclose(result_1h_12a, result_1h_12b)
    assert np.allclose(result_2h_12a, result_2h_12b)


def test_tpcf_decomposition_auto_consistency():
    Npts1, Npts2 = 100, 200
    with NumpyRNGContext(fixed_seed):
        IDs1 = np.random.randint(0, 11, Npts1)
        IDs2 = np.random.randint(0, 11, Npts2)
        sample1 = np.random.random((Npts1, 3))
        sample2 = np.random.random((Npts2, 3))

    result_a = tpcf_one_two_halo_decomp(
        sample1,
        IDs1,
        rbins,
        sample2=sample2,
        sample2_host_halo_id=IDs2,
        randoms=None,
        period=period,
        estimator="Natural",
        approx_cell1_size=[rmax, rmax, rmax],
        approx_cell2_size=[rmax, rmax, rmax],
        approx_cellran_size=[rmax, rmax, rmax],
        do_auto=True,
        do_cross=True,
    )
    result_1h_11a, result_2h_11a = result_a[0:2]
    result_1h_22a, result_2h_22a = result_a[4:6]

    (
        result_1h_11b,
        result_2h_11b,
        result_1h_22b,
        result_2h_22b,
    ) = tpcf_one_two_halo_decomp(
        sample1,
        IDs1,
        rbins,
        sample2=sample2,
        sample2_host_halo_id=IDs2,
        randoms=None,
        period=period,
        estimator="Natural",
        approx_cell1_size=[rmax, rmax, rmax],
        approx_cell2_size=[rmax, rmax, rmax],
        approx_cellran_size=[rmax, rmax, rmax],
        do_auto=True,
        do_cross=False,
    )

    assert np.allclose(result_1h_11a, result_1h_11b)
    assert np.allclose(result_2h_11a, result_2h_11b)
    assert np.allclose(result_1h_22a, result_1h_22b)
    assert np.allclose(result_2h_22a, result_2h_22b)
