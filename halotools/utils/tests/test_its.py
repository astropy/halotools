"""
"""
import numpy as np
from astropy.utils import NumpyRNGContext
import pytest
import warnings

from ..inverse_transformation_sampling import (
    _sorted_rank_order_array,
    rank_order_percentile,
)
from ..inverse_transformation_sampling import (
    build_cdf_lookup,
    monte_carlo_from_cdf_lookup,
)

__all__ = ("test_sorted_rank_order_array",)


def test_sorted_rank_order_array():
    result = _sorted_rank_order_array(4)
    assert np.all(result == (1 / 5.0, 2 / 5.0, 3 / 5.0, 4 / 5.0))


def test_rank_order_percentile1():
    result = rank_order_percentile((1, 2, 3, 4))
    assert np.all(result == (1 / 5.0, 2 / 5.0, 3 / 5.0, 4 / 5.0))


def test_rank_order_percentile2():
    result = rank_order_percentile((4, 3, 2, 1))
    assert np.all(result == (4 / 5.0, 3 / 5.0, 2 / 5.0, 1 / 5.0))


def test_rank_order_percentile3():
    with NumpyRNGContext(43):
        randoms = np.random.rand(100)
    result1 = rank_order_percentile(randoms)
    result2 = rank_order_percentile(0.5 * randoms)
    assert np.all(result1 == result2)


def test_rank_order_percentile4():
    assert np.all(rank_order_percentile((10, 2, 5)) == (0.75, 0.25, 0.5))


def test_build_cdf_lookup1():
    npts = 100000
    y = np.linspace(0, 20, npts)
    x_table, y_table = build_cdf_lookup(y)


def test_build_cdf_lookup2():
    npts = 100
    y = np.random.normal(size=npts)
    x_table, y_table = build_cdf_lookup(y, npts_lookup_table=20)
    assert len(x_table == 20)
    assert len(y_table == 20)


def test_build_cdf_lookup3():
    npts = int(1e5)
    with NumpyRNGContext(43):
        y = np.random.normal(size=npts, loc=900)
    x_table, y_table = build_cdf_lookup(y, npts_lookup_table=npts // 10)
    assert np.allclose(y_table.mean(), 900.0, atol=0.01)


def test_monte_carlo_from_cdf_lookup1():
    npts = int(1e5)
    mean = 50.0
    std = 0.5
    with NumpyRNGContext(43):
        y = np.random.normal(size=npts, loc=mean, scale=std)
    x_table, y_table = build_cdf_lookup(y)
    mc_y = monte_carlo_from_cdf_lookup(x_table, y_table, num_draws=npts + 5, seed=43)

    assert len(mc_y) == npts + 5
    assert np.allclose(mc_y.mean(), mean, rtol=0.01)


def test_monte_carlo_from_cdf_lookup2():
    """
    """
    npts = int(1e4)
    y = 0.3 * np.random.normal(size=npts) + 0.7 * np.random.power(2, size=npts)
    x_table, y_table = build_cdf_lookup(y)

    with pytest.raises(ValueError) as err:
        monte_carlo_from_cdf_lookup(x_table, y_table)
    substr = "If input ``mc_input`` is set to ``random``"
    assert substr in err.value.args[0]


def test_monte_carlo_from_cdf_lookup3():
    """
    """
    npts = int(1e4)
    y = 0.3 * np.random.normal(size=npts) + 0.7 * np.random.power(2, size=npts)
    x_table, y_table = build_cdf_lookup(y)

    with pytest.raises(ValueError) as err:
        monte_carlo_from_cdf_lookup(
            x_table, y_table, mc_input=np.array((0.1, 0.4, 0.25)), num_draws=10
        )
    substr = "If input ``mc_input`` is specified"
    assert substr in err.value.args[0]


def test_build_cdf_lookup_raises_npts_warning():
    npts_y, npts_lookup_table = 100, 200
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        y = np.random.normal(size=npts_y)
        __ = build_cdf_lookup(y, npts_lookup_table=npts_lookup_table)
    substr = "However, the number of data points in your data table npts_y "
    assert substr in str(w[-1].message)
