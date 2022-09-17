"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pytest
from astropy.utils.misc import NumpyRNGContext
from astropy.config.paths import _find_home

from ..pairs import xy_z_wnpairs as pure_python_weighted_pairs
from ..marked_npairs_xy_z import marked_npairs_xy_z
from ..npairs_xy_z import npairs_xy_z
from ..marked_npairs_3d import _func_signature_int_from_wfunc

from ....custom_exceptions import HalotoolsError

slow = pytest.mark.slow

error_msg = (
    "\nThe `test_marked_npairs_wfuncs_behavior` function performs \n"
    "non-trivial checks on the returned values of marked correlation functions\n"
    "calculated on a set of points with uniform weights.\n"
    "One such check failed.\n"
)

__all__ = ("test_marked_npairs_xy_z_periodic",)

fixed_seed = 43


# Determine whether the machine is mine
# This will be used to select tests whose
# returned values depend on the configuration
# of my personal cache directory files
aph_home = "/Users/aphearin"
detected_home = _find_home()
if aph_home == detected_home:
    APH_MACHINE = True
else:
    APH_MACHINE = False


def retrieve_mock_data(Npts, Npts2, Lbox):

    # set up a regular grid of points to test pair counters
    epsilon = 0.001
    gridx = np.linspace(0, Lbox - epsilon, Npts2)
    gridy = np.linspace(0, Lbox - epsilon, Npts2)
    gridz = np.linspace(0, Lbox - epsilon, Npts2)
    xx, yy, zz = np.array(np.meshgrid(gridx, gridy, gridz))
    xx = xx.flatten()
    yy = yy.flatten()
    zz = zz.flatten()

    grid_points = np.vstack([xx, yy, zz]).T
    rp_bins = np.array([0.0, 0.1, 0.2, 0.3])
    period = np.array([Lbox, Lbox, Lbox])

    return grid_points, rp_bins, period


def test_marked_npairs_xy_z_periodic():
    """
    Function tests marked_npairs with periodic boundary conditions.
    """
    Npts = 1000
    with NumpyRNGContext(fixed_seed):
        random_sample = np.random.random((Npts, 3))
        ran_weights1 = np.random.random((Npts, 1))

    period = np.array([1.0, 1.0, 1.0])
    rp_bins = np.array([0.0, 0.1, 0.2, 0.3])
    pi_bins = np.array([0, 0.15])

    result = marked_npairs_xy_z(
        random_sample,
        random_sample,
        rp_bins,
        pi_bins,
        period=period,
        weights1=ran_weights1,
        weights2=ran_weights1,
        weight_func_id=1,
    )

    test_result = pure_python_weighted_pairs(
        random_sample,
        random_sample,
        rp_bins,
        pi_bins,
        period=period,
        weights1=ran_weights1,
        weights2=ran_weights1,
    )

    assert np.allclose(test_result, result, rtol=1e-05), "pair counts are incorrect"


def test_marked_npairs_xy_z_nonperiodic():
    """
    Function tests marked_npairs without periodic boundary conditions.
    """

    Npts = 1000
    with NumpyRNGContext(fixed_seed):
        random_sample = np.random.random((Npts, 3))
        ran_weights1 = np.random.random((Npts, 1))

    rp_bins = np.array([0.0, 0.1, 0.2, 0.3])
    pi_bins = np.array([0, 0.15])

    result = marked_npairs_xy_z(
        random_sample,
        random_sample,
        rp_bins,
        pi_bins,
        period=None,
        weights1=ran_weights1,
        weights2=ran_weights1,
        weight_func_id=1,
    )

    test_result = pure_python_weighted_pairs(
        random_sample,
        random_sample,
        rp_bins,
        pi_bins,
        period=None,
        weights1=ran_weights1,
        weights2=ran_weights1,
    )

    assert np.allclose(test_result, result, rtol=1e-05), "pair counts are incorrect"


@pytest.mark.skipif("not APH_MACHINE")
def test_marked_npairs_parallelization():
    """
    Function tests marked_npairs_3d with periodic boundary conditions.
    """

    Npts = 1000
    with NumpyRNGContext(fixed_seed):
        random_sample = np.random.random((Npts, 3))
        ran_weights1 = np.random.random((Npts, 1))

    period = np.array([1.0, 1.0, 1.0])
    rp_bins = np.array([0.0, 0.1, 0.2, 0.3])
    pi_bins = np.array([0, 0.15])

    serial_result = marked_npairs_xy_z(
        random_sample,
        random_sample,
        rp_bins,
        pi_bins,
        period=period,
        weights1=ran_weights1,
        weights2=ran_weights1,
        weight_func_id=1,
    )

    parallel_result2 = marked_npairs_xy_z(
        random_sample,
        random_sample,
        rp_bins,
        pi_bins,
        period=period,
        weights1=ran_weights1,
        weights2=ran_weights1,
        weight_func_id=1,
        num_threads=2,
    )

    parallel_result7 = marked_npairs_xy_z(
        random_sample,
        random_sample,
        rp_bins,
        pi_bins,
        period=period,
        weights1=ran_weights1,
        weights2=ran_weights1,
        weight_func_id=1,
        num_threads=3,
    )

    assert np.allclose(
        serial_result, parallel_result2, rtol=1e-05
    ), "pair counts are incorrect"
    assert np.allclose(
        serial_result, parallel_result7, rtol=1e-05
    ), "pair counts are incorrect"


def test_marked_npairs_3d_wfuncs_signatures():
    """
    Loop over all wfuncs and ensure that the wfunc signature is handled correctly.
    """
    Npts = 1000
    with NumpyRNGContext(fixed_seed):
        random_sample = np.random.random((Npts, 3))
    rp_bins = np.array([0.0, 0.1, 0.2, 0.3])
    pi_bins = np.array([0, 0.15])
    rmax = rp_bins.max()
    period = np.array([1.0, 1.0, 1.0])

    # Determine how many wfuncs have currently been implemented
    wfunc_index = 1
    while True:
        try:
            _ = _func_signature_int_from_wfunc(wfunc_index)
            wfunc_index += 1
        except HalotoolsError:
            break
    num_wfuncs = np.copy(wfunc_index)

    # Now loop over all all available weight_func_id indices
    for wfunc_index in range(1, num_wfuncs):
        signature = _func_signature_int_from_wfunc(wfunc_index)
        with NumpyRNGContext(fixed_seed):
            weights = np.random.random(Npts * signature).reshape(Npts, signature) - 0.5
        result = marked_npairs_xy_z(
            random_sample,
            random_sample,
            rp_bins,
            pi_bins,
            period=period,
            weights1=weights,
            weights2=weights,
            weight_func_id=wfunc_index,
            approx_cell1_size=[rmax, rmax, rmax],
        )

        with pytest.raises(HalotoolsError):
            signature = _func_signature_int_from_wfunc(wfunc_index) + 1
            with NumpyRNGContext(fixed_seed):
                weights = (
                    np.random.random(Npts * signature).reshape(Npts, signature) - 0.5
                )
            result = marked_npairs_xy_z(
                random_sample,
                random_sample,
                rp_bins,
                pi_bins,
                period=period,
                weights1=weights,
                weights2=weights,
                weight_func_id=wfunc_index,
            )


def test_marked_npairs_behavior_weight_func_id1():
    """
    Verify the behavior of a few wfunc-weighted counters by comparing pure python,
    unmarked pairs to the returned result from a uniformly weighted set of points.
    """

    Npts, Npts2 = 1000, 10
    grid_points, rp_bins, period = retrieve_mock_data(Npts, Npts2, 1)
    rmax = rp_bins.max()
    pi_bins = np.array([0, 0.15])

    # wfunc = 1
    weights = np.ones(Npts) * 3
    result = marked_npairs_xy_z(
        grid_points,
        grid_points,
        rp_bins,
        pi_bins,
        period=period,
        weights1=weights,
        weights2=weights,
        weight_func_id=1,
        approx_cell1_size=[rmax, rmax, rmax],
    )

    test_result = pure_python_weighted_pairs(
        grid_points,
        grid_points,
        rp_bins,
        pi_bins,
        period=period,
        weights1=np.ones((Npts, 1)),
        weights2=np.ones((Npts, 1)),
    )
    assert np.all(result == 9.0 * test_result)


def test_marked_npairs_behavior_weight_func_id2():
    """
    Verify the behavior of a few wfunc-weighted counters by comparing pure python,
    unmarked pairs to the returned result from a uniformly weighted set of points.
    """

    Npts, Npts2 = 1000, 10
    grid_points, rp_bins, period = retrieve_mock_data(Npts, Npts2, 1)
    rmax = rp_bins.max()
    pi_bins = np.array([0, 0.15])

    test_result = pure_python_weighted_pairs(
        grid_points,
        grid_points,
        rp_bins,
        pi_bins,
        weights1=np.ones((Npts, 1)),
        weights2=np.ones((Npts, 1)),
        period=period,
    )

    # wfunc = 2
    weights = np.ones(Npts) * 3
    result = marked_npairs_xy_z(
        grid_points,
        grid_points,
        rp_bins,
        pi_bins,
        period=period,
        weights1=weights,
        weights2=weights,
        weight_func_id=2,
        approx_cell1_size=[rmax, rmax, rmax],
    )
    assert np.all(result == 6.0 * test_result), error_msg


def test_marked_npairs_behavior_weight_func_id3():
    """
    Verify the behavior of a few wfunc-weighted counters by comparing pure python,
    unmarked pairs to the returned result from a uniformly weighted set of points.
    """

    Npts, Npts2 = 1000, 10
    grid_points, rp_bins, period = retrieve_mock_data(Npts, Npts2, 1)
    rmax = rp_bins.max()
    pi_bins = np.array([0, 0.15])

    test_result = pure_python_weighted_pairs(
        grid_points,
        grid_points,
        rp_bins,
        pi_bins,
        weights1=np.ones((Npts, 1)),
        weights2=np.ones((Npts, 1)),
        period=period,
    )

    # wfunc = 3
    weights2 = np.ones(Npts) * 2
    weights3 = np.ones(Npts) * 3

    weights = np.vstack([weights2, weights3]).T
    result = marked_npairs_xy_z(
        grid_points,
        grid_points,
        rp_bins,
        pi_bins,
        period=period,
        weights1=weights,
        weights2=weights,
        weight_func_id=3,
        approx_cell1_size=[rmax, rmax, rmax],
    )
    assert np.all(result == 9.0 * test_result), error_msg

    weights = np.vstack([weights3, weights2]).T
    result = marked_npairs_xy_z(
        grid_points,
        grid_points,
        rp_bins,
        pi_bins,
        period=period,
        weights1=weights,
        weights2=weights,
        weight_func_id=3,
        approx_cell1_size=[rmax, rmax, rmax],
    )
    assert np.all(result == 4.0 * test_result), error_msg


def test_marked_npairs_behavior_weight_func_id4():
    """
    Verify the behavior of a few wfunc-weighted counters by comparing pure python,
    unmarked pairs to the returned result from a uniformly weighted set of points.
    """

    Npts, Npts2 = 1000, 10
    grid_points, rp_bins, period = retrieve_mock_data(Npts, Npts2, 1)
    rmax = rp_bins.max()
    pi_bins = np.array([0, 0.15])

    # wfunc = 4
    weights2 = np.ones(Npts) * 2
    weights3 = np.ones(Npts) * 3

    weights = np.vstack([weights2, weights3]).T
    result = marked_npairs_xy_z(
        grid_points,
        grid_points,
        rp_bins,
        pi_bins,
        period=period,
        weights1=weights,
        weights2=weights,
        weight_func_id=4,
        approx_cell1_size=[rmax, rmax, rmax],
    )
    assert np.all(result == 0), error_msg


def test_marked_npairs_behavior_weight_func_id5():
    """
    Verify the behavior of a few wfunc-weighted counters by comparing pure python,
    unmarked pairs to the returned result from a uniformly weighted set of points.
    """

    Npts, Npts2 = 1000, 10
    grid_points, rp_bins, period = retrieve_mock_data(Npts, Npts2, 1)
    rmax = rp_bins.max()
    pi_bins = np.array([0, 0.15])

    # wfunc = 5
    weights2 = np.ones(Npts) * 2
    weights3 = np.ones(Npts) * 3

    weights = np.vstack([weights2, weights3]).T
    result = marked_npairs_xy_z(
        grid_points,
        grid_points,
        rp_bins,
        pi_bins,
        period=period,
        weights1=weights,
        weights2=weights,
        weight_func_id=5,
        approx_cell1_size=[rmax, rmax, rmax],
    )
    assert np.all(result == 0), error_msg


def test_marked_npairs_behavior_weight_func_id6():
    """
    Verify the behavior of a few wfunc-weighted counters by comparing pure python,
    unmarked pairs to the returned result from a uniformly weighted set of points.
    """

    Npts, Npts2 = 1000, 10
    grid_points, rp_bins, period = retrieve_mock_data(Npts, Npts2, 1)
    rmax = rp_bins.max()
    pi_bins = np.array([0, 0.15])

    # wfunc = 6
    weights2 = np.ones(Npts) * 2
    weights3 = np.ones(Npts) * 3

    weights = np.vstack([weights2, weights3]).T
    result = marked_npairs_xy_z(
        grid_points,
        grid_points,
        rp_bins,
        pi_bins,
        period=period,
        weights1=weights,
        weights2=weights,
        weight_func_id=6,
        approx_cell1_size=[rmax, rmax, rmax],
    )
    assert np.all(result == 0), error_msg


def test_marked_npairs_behavior_weight_func_id7():
    """
    Verify the behavior of a few wfunc-weighted counters by comparing pure python,
    unmarked pairs to the returned result from a uniformly weighted set of points.
    """

    Npts, Npts2 = 1000, 10
    grid_points, rp_bins, period = retrieve_mock_data(Npts, Npts2, 1)
    rmax = rp_bins.max()
    pi_bins = np.array([0, 0.15])

    test_result = pure_python_weighted_pairs(
        grid_points,
        grid_points,
        rp_bins,
        pi_bins,
        weights1=np.ones((Npts, 1)),
        weights2=np.ones((Npts, 1)),
        period=period,
    )

    # wfunc = 7
    weights2 = np.ones(Npts)
    weights3 = np.zeros(Npts) - 1

    weights = np.vstack([weights2, weights3]).T
    result = marked_npairs_xy_z(
        grid_points,
        grid_points,
        rp_bins,
        pi_bins,
        period=period,
        weights1=weights,
        weights2=weights,
        weight_func_id=7,
        approx_cell1_size=[rmax, rmax, rmax],
    )
    assert np.all(result == -test_result), error_msg


def test_marked_npairs_behavior_weight_func_id8():
    """
    Verify the behavior of a few wfunc-weighted counters by comparing pure python,
    unmarked pairs to the returned result from a uniformly weighted set of points.
    """

    Npts, Npts2 = 1000, 10
    grid_points, rp_bins, period = retrieve_mock_data(Npts, Npts2, 1)
    rmax = rp_bins.max()
    pi_bins = np.array([0, 0.15])

    test_result = pure_python_weighted_pairs(
        grid_points,
        grid_points,
        rp_bins,
        pi_bins,
        weights1=np.ones((Npts, 1)),
        weights2=np.ones((Npts, 1)),
        period=period,
    )

    # wfunc = 8
    weights2 = np.ones(Npts)
    weights3 = np.ones(Npts) * 3

    weights = np.vstack([weights2, weights3]).T
    result = marked_npairs_xy_z(
        grid_points,
        grid_points,
        rp_bins,
        pi_bins,
        period=period,
        weights1=weights,
        weights2=weights,
        weight_func_id=8,
        approx_cell1_size=[rmax, rmax, rmax],
    )
    assert np.all(result == 3 * test_result), error_msg


def test_marked_npairs_behavior_weight_func_id9():
    """
    Verify the behavior of a few wfunc-weighted counters by comparing pure python,
    unmarked pairs to the returned result from a uniformly weighted set of points.
    """

    Npts, Npts2 = 1000, 10
    grid_points, rp_bins, period = retrieve_mock_data(Npts, Npts2, 1)
    rmax = rp_bins.max()
    pi_bins = np.array([0, 0.15])

    test_result = pure_python_weighted_pairs(
        grid_points,
        grid_points,
        rp_bins,
        pi_bins,
        weights1=np.ones((Npts, 1)),
        weights2=np.ones((Npts, 1)),
        period=period,
    )

    # wfunc = 9
    weights2 = np.ones(Npts)
    weights3 = np.ones(Npts) * 3

    weights = np.vstack([weights2, weights3]).T
    result = marked_npairs_xy_z(
        grid_points,
        grid_points,
        rp_bins,
        pi_bins,
        period=period,
        weights1=weights,
        weights2=weights,
        weight_func_id=9,
        approx_cell1_size=[rmax, rmax, rmax],
    )
    assert np.all(result == 3 * test_result), error_msg


def test_marked_npairs_behavior_weight_func_id10():
    """
    Verify the behavior of a few wfunc-weighted counters by comparing pure python,
    unmarked pairs to the returned result from a uniformly weighted set of points.
    """

    Npts, Npts2 = 1000, 10
    grid_points, rp_bins, period = retrieve_mock_data(Npts, Npts2, 1)
    rmax = rp_bins.max()
    pi_bins = np.array([0, 0.15])

    test_result = pure_python_weighted_pairs(
        grid_points,
        grid_points,
        rp_bins,
        pi_bins,
        weights1=np.ones((Npts, 1)),
        weights2=np.ones((Npts, 1)),
        period=period,
    )

    # wfunc = 10
    weights2 = np.ones(Npts)
    weights3 = np.ones(Npts) * 3

    weights = np.vstack([weights2, weights3]).T
    result = marked_npairs_xy_z(
        grid_points,
        grid_points,
        rp_bins,
        pi_bins,
        period=period,
        weights1=weights,
        weights2=weights,
        weight_func_id=10,
        approx_cell1_size=[rmax, rmax, rmax],
    )
    assert np.all(result == 0), error_msg

    weights2 = np.ones(Npts)
    weights3 = -np.ones(Npts) * 3

    weights = np.vstack([weights2, weights3]).T
    result = marked_npairs_xy_z(
        grid_points,
        grid_points,
        rp_bins,
        pi_bins,
        period=period,
        weights1=weights,
        weights2=weights,
        weight_func_id=10,
        approx_cell1_size=[rmax, rmax, rmax],
    )
    assert np.all(result == -3 * test_result), error_msg


def test_marked_npairs_behavior_weight_func_id11():
    """
    weight_func_id=11
    """

    Npts, Npts2 = 1000, 10
    grid_points, rp_bins, period = retrieve_mock_data(Npts, Npts2, 1)

    pi_bins = np.array([0, 0.15])

    weights1 = np.tile((1.0, 0.0), Npts).reshape((Npts, 2))
    weights2 = np.ones(Npts * 2).reshape((Npts, 2))

    result1 = marked_npairs_xy_z(
        grid_points,
        grid_points,
        rp_bins,
        pi_bins,
        period=period,
        weights1=weights1,
        weights2=weights2,
        weight_func_id=11,
    )

    result2 = npairs_xy_z(grid_points, grid_points, rp_bins, pi_bins, period=period)
    assert np.all(result1 == result2)

    weights2 = np.tile((1.0, 2.0), Npts).reshape((Npts, 2))
    result3 = marked_npairs_xy_z(
        grid_points,
        grid_points,
        rp_bins,
        pi_bins,
        period=period,
        weights1=weights1,
        weights2=weights2,
        weight_func_id=11,
    )
    assert np.all(result3 == 2 * result2)
