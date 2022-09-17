""" Module providing unit-testing for the `~halotools.mock_observables.tpcf` function.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import warnings
import pytest
from astropy.utils.misc import NumpyRNGContext

from .locate_external_unit_testing_data import tpcf_corrfunc_comparison_files_exist

from ..tpcf import tpcf

from ....custom_exceptions import HalotoolsError

slow = pytest.mark.slow

__all__ = (
    "test_tpcf_auto",
    "test_tpcf_cross",
    "test_tpcf_estimators",
    "test_tpcf_sample_size_limit",
    "test_tpcf_randoms",
    "test_tpcf_period_API",
    "test_tpcf_cross_consistency_w_auto",
)

fixed_seed = 43
TPCF_CORRFUNC_FILES_EXIST = tpcf_corrfunc_comparison_files_exist()


def test_tpcf_auto():
    """
    test the tpcf auto-correlation functionality
    """
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((100, 3))
    with NumpyRNGContext(fixed_seed + 2):
        randoms = np.random.random((102, 3))
    period = np.array([1.0, 1.0, 1.0])
    rbins = np.linspace(0.001, 0.3, 5)
    rmax = rbins.max()

    # with randoms
    result = tpcf(
        sample1,
        rbins,
        sample2=None,
        randoms=randoms,
        period=None,
        estimator="Natural",
        approx_cell1_size=[rmax, rmax, rmax],
        approx_cellran_size=[rmax, rmax, rmax],
    )
    assert result.ndim == 1, "More than one correlation function returned erroneously."

    # with out randoms
    result = tpcf(
        sample1,
        rbins,
        sample2=None,
        randoms=None,
        period=period,
        estimator="Natural",
        approx_cell1_size=[rmax, rmax, rmax],
        num_threads=1,
    )
    assert result.ndim == 1, "More than one correlation function returned erroneously."


def test_tpcf_cross():
    """
    test the tpcf cross-correlation functionality
    """
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((100, 3))
    with NumpyRNGContext(fixed_seed + 1):
        sample2 = np.random.random((101, 3))
    with NumpyRNGContext(fixed_seed + 2):
        randoms = np.random.random((102, 3))
    period = np.array([1.0, 1.0, 1.0])
    rbins = np.linspace(0.001, 0.3, 5)
    rmax = rbins.max()

    # with randoms
    result = tpcf(
        sample1,
        rbins,
        sample2=sample2,
        randoms=randoms,
        period=None,
        estimator="Natural",
        do_auto=False,
        approx_cell1_size=[rmax, rmax, rmax],
    )
    assert result.ndim == 1, "More than one correlation function returned erroneously."

    # with out randoms
    result = tpcf(
        sample1,
        rbins,
        sample2=sample2,
        randoms=None,
        period=period,
        estimator="Natural",
        do_auto=False,
        approx_cell1_size=[rmax, rmax, rmax],
    )
    assert result.ndim == 1, "More than one correlation function returned erroneously."


def test_tpcf_natural_estimator():
    """
    test the tpcf different estimators functionality
    """
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((100, 3))
    with NumpyRNGContext(fixed_seed + 1):
        sample2 = np.random.random((101, 3))
    with NumpyRNGContext(fixed_seed + 2):
        randoms = np.random.random((102, 3))
    rbins = np.linspace(0.001, 0.3, 5)
    rmax = rbins.max()
    msg = "wrong number of correlation functions returned erroneously."

    result_1 = tpcf(
        sample1,
        rbins,
        sample2=sample2,
        randoms=randoms,
        period=None,
        estimator="Natural",
        approx_cell1_size=[rmax, rmax, rmax],
        approx_cellran_size=[rmax, rmax, rmax],
    )
    assert len(result_1) == 3, msg


@pytest.mark.xfail
def test_tpcf_davis_peebles_estimator():
    """
    test the tpcf different estimators functionality
    """
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((100, 3))
    with NumpyRNGContext(fixed_seed + 1):
        sample2 = np.random.random((101, 3))
    with NumpyRNGContext(fixed_seed + 2):
        randoms = np.random.random((102, 3))
    rbins = np.linspace(0.001, 0.3, 5)
    rmax = rbins.max()
    msg = "wrong number of correlation functions returned erroneously."

    result = tpcf(
        sample1,
        rbins,
        sample2=sample2,
        randoms=randoms,
        period=None,
        estimator="Davis-Peebles",
        approx_cell1_size=[rmax, rmax, rmax],
        approx_cellran_size=[rmax, rmax, rmax],
    )
    assert len(result) == 3, msg


@pytest.mark.xfail
def test_tpcf_hewett_estimator():
    """
    test the tpcf different estimators functionality
    """
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((100, 3))
        sample2 = np.random.random((100, 3))
        randoms = np.random.random((100, 3))
    rbins = np.linspace(0.001, 0.3, 5)
    rmax = rbins.max()
    msg = "wrong number of correlation functions returned erroneously."

    result = tpcf(
        sample1,
        rbins,
        sample2=sample2,
        randoms=randoms,
        period=None,
        estimator="Hewett",
        approx_cell1_size=[rmax, rmax, rmax],
        approx_cellran_size=[rmax, rmax, rmax],
    )
    assert len(result) == 3, msg


def test_tpcf_hamilton_estimator():
    """
    test the tpcf different estimators functionality
    """
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((100, 3))
    with NumpyRNGContext(fixed_seed + 1):
        sample2 = np.random.random((101, 3))
    with NumpyRNGContext(fixed_seed + 2):
        randoms = np.random.random((102, 3))
    rbins = np.linspace(0.001, 0.3, 5)
    rmax = rbins.max()
    msg = "wrong number of correlation functions returned erroneously."

    result = tpcf(
        sample1,
        rbins,
        sample2=sample2,
        randoms=randoms,
        period=None,
        estimator="Hamilton",
        approx_cell1_size=[rmax, rmax, rmax],
        approx_cellran_size=[rmax, rmax, rmax],
    )
    assert len(result) == 3, msg


def test_tpcf_landy_szalay_estimator():
    """
    test the tpcf different estimators functionality
    """
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((100, 3))
    with NumpyRNGContext(fixed_seed + 1):
        sample2 = np.random.random((101, 3))
    with NumpyRNGContext(fixed_seed + 2):
        randoms = np.random.random((102, 3))
    rbins = np.linspace(0.001, 0.3, 5)
    rmax = rbins.max()
    msg = "wrong number of correlation functions returned erroneously."

    result = tpcf(
        sample1,
        rbins,
        sample2=sample2,
        randoms=randoms,
        period=None,
        estimator="Landy-Szalay",
        approx_cell1_size=[rmax, rmax, rmax],
        approx_cellran_size=[rmax, rmax, rmax],
    )
    assert len(result) == 3, msg


def test_tpcf_randoms():
    """
    test the tpcf possible randoms + PBCs combinations
    """
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((100, 3))
    with NumpyRNGContext(fixed_seed + 1):
        sample2 = np.random.random((101, 3))
    with NumpyRNGContext(fixed_seed + 2):
        randoms = np.random.random((102, 3))
    period = np.array([1.0, 1.0, 1.0])
    rbins = np.linspace(0.001, 0.3, 5)
    rmax = rbins.max()

    # No PBCs w/ randoms
    result_1 = tpcf(
        sample1,
        rbins,
        sample2=sample2,
        randoms=randoms,
        period=None,
        estimator="Natural",
        approx_cell1_size=[rmax, rmax, rmax],
        approx_cellran_size=[rmax, rmax, rmax],
    )
    # PBCs w/o randoms
    result_2 = tpcf(
        sample1,
        rbins,
        sample2=sample2,
        randoms=None,
        period=period,
        estimator="Natural",
        approx_cell1_size=[rmax, rmax, rmax],
        approx_cellran_size=[rmax, rmax, rmax],
    )
    # PBCs w/ randoms
    result_3 = tpcf(
        sample1,
        rbins,
        sample2=sample2,
        randoms=randoms,
        period=period,
        estimator="Natural",
        approx_cell1_size=[rmax, rmax, rmax],
        approx_cellran_size=[rmax, rmax, rmax],
    )

    # No PBCs and no randoms should throw an error.
    with pytest.raises(ValueError) as err:
        tpcf(
            sample1,
            rbins,
            sample2=sample2,
            randoms=None,
            period=None,
            estimator="Natural",
            approx_cell1_size=[rmax, rmax, rmax],
            approx_cellran_size=[rmax, rmax, rmax],
        )
    substr = "If no PBCs are specified, randoms must be provided."
    assert substr in err.value.args[0]

    msg = "wrong number of correlation functions returned erroneously."
    assert len(result_1) == 3, msg
    assert len(result_2) == 3, msg
    assert len(result_3) == 3, msg


def test_tpcf_period_API():
    """
    test the tpcf period API functionality.
    """
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((100, 3))
    with NumpyRNGContext(fixed_seed + 1):
        sample2 = np.random.random((101, 3))
    with NumpyRNGContext(fixed_seed + 2):
        randoms = np.random.random((102, 3))
    period = np.array([1.0, 1.0, 1.0])
    rbins = np.linspace(0.001, 0.3, 5)
    rmax = rbins.max()

    result_1 = tpcf(
        sample1,
        rbins,
        sample2=sample2,
        randoms=randoms,
        period=period,
        estimator="Natural",
        approx_cell1_size=[rmax, rmax, rmax],
    )

    period = 1.0
    result_2 = tpcf(
        sample1,
        rbins,
        sample2=sample2,
        randoms=randoms,
        period=period,
        estimator="Natural",
        approx_cell1_size=[rmax, rmax, rmax],
    )

    # should throw an error.  period must be positive!
    period = np.array([1.0, 1.0, -1.0])
    with pytest.raises(ValueError) as err:
        tpcf(
            sample1,
            rbins,
            sample2=sample2,
            randoms=randoms,
            period=period,
            estimator="Natural",
            approx_cell1_size=[rmax, rmax, rmax],
        )
    substr = "All values must bounded positive numbers."
    assert substr in err.value.args[0]

    assert (
        len(result_1) == 3
    ), "wrong number of correlation functions returned erroneously."
    assert (
        len(result_2) == 3
    ), "wrong number of correlation functions returned erroneously."


def test_tpcf_cross_consistency_w_auto():
    """
    test the tpcf cross-correlation mode consistency with auto-correlation mode
    """
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((200, 3))
        sample2 = np.random.random((100, 3))
        randoms = np.random.random((300, 3))
    period = np.array([1.0, 1.0, 1.0])
    rbins = np.linspace(0.001, 0.3, 5)
    rmax = rbins.max()

    # with out randoms
    result1 = tpcf(
        sample1,
        rbins,
        sample2=None,
        randoms=None,
        period=period,
        estimator="Natural",
        approx_cell1_size=[rmax, rmax, rmax],
    )

    result2 = tpcf(
        sample2,
        rbins,
        sample2=None,
        randoms=None,
        period=period,
        estimator="Natural",
        approx_cell1_size=[rmax, rmax, rmax],
    )

    result1_p, result12, result2_p = tpcf(
        sample1,
        rbins,
        sample2=sample2,
        randoms=None,
        period=period,
        estimator="Natural",
        approx_cell1_size=[rmax, rmax, rmax],
    )

    msg = "cross mode and auto mode are not the same"
    assert np.allclose(result1, result1_p), msg
    assert np.allclose(result2, result2_p), msg

    # with randoms
    result1 = tpcf(
        sample1,
        rbins,
        sample2=None,
        randoms=randoms,
        period=period,
        estimator="Natural",
        approx_cell1_size=[rmax, rmax, rmax],
    )

    result2 = tpcf(
        sample2,
        rbins,
        sample2=None,
        randoms=randoms,
        period=period,
        estimator="Natural",
        approx_cell1_size=[rmax, rmax, rmax],
    )

    result1_p, result12, result2_p = tpcf(
        sample1,
        rbins,
        sample2=sample2,
        randoms=randoms,
        period=period,
        estimator="Natural",
        approx_cell1_size=[rmax, rmax, rmax],
    )

    assert np.allclose(result1, result1_p), msg
    assert np.allclose(result2, result2_p), msg


def test_RR_precomputed_exception_handling1():
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((1000, 3))
        sample2 = np.random.random((100, 3))
        randoms = np.random.random((100, 3))
    period = np.array([1.0, 1.0, 1.0])
    rbins = np.linspace(0.001, 0.3, 5)
    rmax = rbins.max()

    RR_precomputed = rmax
    with pytest.raises(HalotoolsError) as err:
        _ = tpcf(
            sample1,
            rbins,
            sample2=sample2,
            randoms=randoms,
            period=period,
            estimator="Natural",
            approx_cell1_size=[rmax, rmax, rmax],
            RR_precomputed=RR_precomputed,
        )
    substr = "``RR_precomputed`` and ``NR_precomputed`` arguments, or neither\n"
    assert substr in err.value.args[0]


def test_RR_precomputed_exception_handling2():
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((1000, 3))
        sample2 = np.random.random((100, 3))
        randoms = np.random.random((100, 3))
    period = np.array([1.0, 1.0, 1.0])
    rbins = np.linspace(0.001, 0.3, 5)
    rmax = rbins.max()

    RR_precomputed = rbins[:-2]
    NR_precomputed = randoms.shape[0]
    with pytest.raises(HalotoolsError) as err:
        _ = tpcf(
            sample1,
            rbins,
            sample2=sample2,
            randoms=randoms,
            period=period,
            estimator="Natural",
            approx_cell1_size=[rmax, rmax, rmax],
            RR_precomputed=RR_precomputed,
            NR_precomputed=NR_precomputed,
        )
    substr = "\nLength of ``RR_precomputed`` must match length of ``rbins``\n"
    assert substr in err.value.args[0]


def test_RR_precomputed_exception_handling3():
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((1000, 3))
        sample2 = np.random.random((100, 3))
        randoms = np.random.random((100, 3))
    period = np.array([1.0, 1.0, 1.0])
    rbins = np.linspace(0.001, 0.3, 5)
    rmax = rbins.max()

    RR_precomputed = rbins[:-1]
    NR_precomputed = 5
    with pytest.raises(HalotoolsError) as err:
        _ = tpcf(
            sample1,
            rbins,
            sample2=sample2,
            randoms=randoms,
            period=period,
            estimator="Natural",
            approx_cell1_size=[rmax, rmax, rmax],
            RR_precomputed=RR_precomputed,
            NR_precomputed=NR_precomputed,
        )
    substr = "the value of NR_precomputed must agree with the number of randoms"
    assert substr in err.value.args[0]


def test_RR_precomputed_natural_estimator_auto():
    """Strategy here is as follows. First, we adopt the same setup
    with randomly generated points as used in the rest of the test suite.
    First, we just compute the tpcf in the normal way.
    Then we break apart the tpcf innards so that we can
    compute RR in the exact same way that it is computed within tpcf.
    We will then pass in this RR using the RR_precomputed keyword,
    and verify that the tpcf computed in this second way gives
    exactly the same results as if we did not pre-compute RR.

    """
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((1000, 3))
        sample2 = np.copy(sample1)
        randoms = np.random.random((100, 3))
    period = np.array([1.0, 1.0, 1.0])
    rbins = np.linspace(0.001, 0.3, 5)
    rmax = rbins.max()

    approx_cell1_size = [rmax, rmax, rmax]
    approx_cell2_size = approx_cell1_size
    approx_cellran_size = [rmax, rmax, rmax]

    substr = "`sample1` and `sample2` are exactly the same"
    with warnings.catch_warnings(record=True) as w:
        normal_result = tpcf(
            sample1,
            rbins,
            sample2=sample2,
            randoms=randoms,
            period=period,
            estimator="Natural",
            approx_cell1_size=approx_cell1_size,
            approx_cellran_size=approx_cellran_size,
        )
        assert substr in str(w[-1].message)

    # The following quantities are computed inside the
    # tpcf namespace. We reproduce them here because they are
    # necessary inputs to the _random_counts and _pair_counts
    # functions called by tpcf
    _sample1_is_sample2 = True
    PBCs = True
    num_threads = 1
    do_DD, do_DR, do_RR = True, True, True
    do_auto, do_cross = True, False

    from ..tpcf import _random_counts, _pair_counts

    # count data pairs
    D1D1, D1D2, D2D2 = _pair_counts(
        sample1,
        sample2,
        rbins,
        period,
        num_threads,
        do_auto,
        do_cross,
        _sample1_is_sample2,
        approx_cell1_size,
        approx_cell2_size,
    )

    # count random pairs
    D1R, D2R, RR = _random_counts(
        sample1,
        sample2,
        randoms,
        rbins,
        period,
        PBCs,
        num_threads,
        do_RR,
        do_DR,
        _sample1_is_sample2,
        approx_cell1_size,
        approx_cell2_size,
        approx_cellran_size,
    )

    N1 = len(sample1)
    NR = len(randoms)

    factor = N1 * N1 / (NR * NR)

    def mult(x, y):
        return x * y

    xi_11 = mult(1.0 / factor, D1D1 / RR) - 1.0

    # The following assertion implies that the RR
    # computed within this testing namespace is the same RR
    # as computed in the tpcf namespace
    assert np.all(xi_11 == normal_result)

    # Now we will pass in the above RR as an argument
    # and verify that we get an identical tpcf
    with warnings.catch_warnings(record=True) as w:
        result_with_RR_precomputed = tpcf(
            sample1,
            rbins,
            sample2=sample2,
            randoms=randoms,
            period=period,
            estimator="Natural",
            approx_cell1_size=approx_cell1_size,
            approx_cellran_size=approx_cellran_size,
            RR_precomputed=RR,
            NR_precomputed=NR,
        )
        assert substr in str(w[-1].message)

    assert np.all(result_with_RR_precomputed == normal_result)


def test_RR_precomputed_Landy_Szalay_estimator_auto():
    """Strategy here is as follows. First, we adopt the same setup
    with randomly generated points as used in the rest of the test suite.
    First, we just compute the tpcf in the normal way.
    Then we break apart the tpcf innards so that we can
    compute RR in the exact same way that it is computed within tpcf.
    We will then pass in this RR using the RR_precomputed keyword,
    and verify that the tpcf computed in this second way gives
    exactly the same results as if we did not pre-compute RR.

    """
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((1000, 3))
        sample2 = sample1
        randoms = np.random.random((100, 3))
    period = np.array([1.0, 1.0, 1.0])
    rbins = np.linspace(0.001, 0.3, 5)
    rmax = rbins.max()

    approx_cell1_size = [rmax, rmax, rmax]
    approx_cell2_size = approx_cell1_size
    approx_cellran_size = [rmax, rmax, rmax]

    substr = "`sample1` and `sample2` are exactly the same"
    with warnings.catch_warnings(record=True) as w:
        normal_result = tpcf(
            sample1,
            rbins,
            sample2=sample2,
            randoms=randoms,
            period=period,
            estimator="Landy-Szalay",
            approx_cell1_size=approx_cell1_size,
            approx_cellran_size=approx_cellran_size,
        )
        assert substr in str(w[-1].message)

    # The following quantities are computed inside the
    # tpcf namespace. We reproduce them here because they are
    # necessary inputs to the _random_counts and _pair_counts
    # functions called by tpcf
    _sample1_is_sample2 = True
    PBCs = True
    num_threads = 1
    do_DD, do_DR, do_RR = True, True, True
    do_auto, do_cross = True, False

    from ..tpcf import _random_counts, _pair_counts

    # count data pairs
    D1D1, D1D2, D2D2 = _pair_counts(
        sample1,
        sample2,
        rbins,
        period,
        num_threads,
        do_auto,
        do_cross,
        _sample1_is_sample2,
        approx_cell1_size,
        approx_cell2_size,
    )

    # count random pairs
    D1R, D2R, RR = _random_counts(
        sample1,
        sample2,
        randoms,
        rbins,
        period,
        PBCs,
        num_threads,
        do_RR,
        do_DR,
        _sample1_is_sample2,
        approx_cell1_size,
        approx_cell2_size,
        approx_cellran_size,
    )

    ND1 = len(sample1)
    ND2 = len(sample2)
    NR1 = len(randoms)
    NR2 = len(randoms)

    factor1 = ND1 * ND2 / (NR1 * NR2)
    factor2 = ND1 * NR2 / (NR1 * NR2)

    def mult(x, y):
        return x * y

    xi_11 = mult(1.0 / factor1, D1D1 / RR) - mult(1.0 / factor2, 2.0 * D1R / RR) + 1.0

    # # The following assertion implies that the RR
    # # computed within this testing namespace is the same RR
    # # as computed in the tpcf namespace
    assert np.all(xi_11 == normal_result)

    # Now we will pass in the above RR as an argument
    # and verify that we get an identical tpcf
    with warnings.catch_warnings(record=True) as w:
        result_with_RR_precomputed = tpcf(
            sample1,
            rbins,
            sample2=sample2,
            randoms=randoms,
            period=period,
            estimator="Landy-Szalay",
            approx_cell1_size=approx_cell1_size,
            approx_cellran_size=approx_cellran_size,
            RR_precomputed=RR,
            NR_precomputed=NR1,
        )
        assert substr in str(w[-1].message)

    assert np.all(result_with_RR_precomputed == normal_result)


def test_tpcf_raises_exception_for_non_monotonic_rbins():
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((1000, 3))
    period = np.array([1.0, 1.0, 1.0])
    rbins = np.linspace(10, 0.3, 5)

    with pytest.raises(TypeError) as err:
        normal_result = tpcf(sample1, rbins, period=period)
    substr = "Input separation bins must be a monotonically increasing"
    assert substr in err.value.args[0]


def test_tpcf_raises_exception_for_large_search_length():
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((1000, 3))
    period = np.array([1.0, 1.0, 1.0])
    rbins = np.linspace(0.1, 0.5, 5)

    with pytest.raises(ValueError) as err:
        normal_result = tpcf(sample1, rbins, period=period)
    substr = "Either decrease your search length or use a larger simulation"
    assert substr in err.value.args[0]


def test_tpcf_raises_exception_for_incompatible_data_shapes():
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((1000, 3))
        sample2 = np.random.random((1000, 2))
    period = np.array([1.0, 1.0, 1.0])
    rbins = np.linspace(0.1, 0.3, 5)

    with pytest.raises(TypeError) as err:
        normal_result = tpcf(sample1, rbins, sample2=sample2, period=period)
    substr = "Input sample of points must be a Numpy ndarray of shape (Npts, 3)."
    assert substr in err.value.args[0]


def test_tpcf_raises_exception_for_bad_do_auto_instructions():
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((1000, 3))
        sample2 = np.random.random((1000, 3))
    period = np.array([1.0, 1.0, 1.0])
    rbins = np.linspace(0.1, 0.3, 5)

    with pytest.raises(ValueError) as err:
        normal_result = tpcf(
            sample1, rbins, sample2=sample2, period=period, do_auto="Jose Canseco"
        )
    substr = "`do_auto` and `do_cross` keywords must be boolean-valued."
    assert substr in err.value.args[0]


def test_tpcf_raises_exception_for_unavailable_estimator():
    with NumpyRNGContext(fixed_seed):
        sample1 = np.random.random((1000, 3))
        sample2 = np.random.random((1000, 3))
    period = np.array([1.0, 1.0, 1.0])
    rbins = np.linspace(0.1, 0.3, 5)

    with pytest.raises(ValueError) as err:
        normal_result = tpcf(sample1, rbins, period=period, estimator="Jose Canseco")
    substr = "is not in the list of available estimators:"
    assert substr in err.value.args[0]


@pytest.mark.skipif("not TPCF_CORRFUNC_FILES_EXIST")
def test_tpcf_vs_corrfunc():
    """ """
    msg = (
        "This unit-test compares the tpcf results from halotools \n"
        "against the results derived from the Corrfunc code managed by \n"
        "Manodeep Sinha. "
    )
    (
        __,
        aph_fname1,
        aph_fname2,
        aph_fname3,
        deep_fname1,
        deep_fname2,
    ) = tpcf_corrfunc_comparison_files_exist(return_fnames=True)

    sinha_sample1_xi = np.load(deep_fname1)[:, 0]
    sinha_sample2_xi = np.load(deep_fname2)[:, 0]

    sample1 = np.load(aph_fname1)
    sample2 = np.load(aph_fname2)
    rbins = np.load(aph_fname3)

    halotools_result1 = tpcf(sample1, rbins, period=250.0)
    assert np.allclose(halotools_result1, sinha_sample1_xi, rtol=1e-5), msg

    halotools_result2 = tpcf(sample2, rbins, period=250.0)
    assert np.allclose(halotools_result2, sinha_sample2_xi, rtol=1e-5), msg
