"""
"""
import numpy as np
from astropy.utils import NumpyRNGContext

from ..conditional_abunmatch_bin_based import conditional_abunmatch_bin_based


def test_conditional_abunmatch_bin_based1():
    with NumpyRNGContext(43):
        x = np.random.normal(loc=0, scale=0.1, size=100)
    y = np.linspace(10, 20, 100)
    model_y = conditional_abunmatch_bin_based(x, y, seed=43, npts_lookup_table=len(y))
    msg = "monotonic cam does not preserve mean"
    assert np.allclose(model_y.mean(), y.mean(), rtol=0.1), msg


def test_conditional_abunmatch_bin_based2():
    with NumpyRNGContext(43):
        x = np.random.normal(loc=0, scale=0.1, size=100)
    y = np.linspace(10, 20, 100)
    model_y = conditional_abunmatch_bin_based(x, y, seed=43, npts_lookup_table=len(y))
    idx_x_sorted = np.argsort(x)
    msg = "monotonic cam does not preserve correlation"
    high = model_y[idx_x_sorted][-50:].mean()
    low = model_y[idx_x_sorted][:50].mean()
    mean = model_y.mean()
    high_low_fracdiff = (high-low)/mean
    assert high_low_fracdiff > 0.1


def test_conditional_abunmatch_bin_based3():
    with NumpyRNGContext(43):
        x = np.random.normal(loc=0, scale=0.1, size=100)
    y = np.linspace(10, 20, 100)
    model_y = conditional_abunmatch_bin_based(x, y, sigma=0.01, seed=43, npts_lookup_table=len(y))
    idx_x_sorted = np.argsort(x)
    msg = "low-noise cam does not preserve correlation"
    high = model_y[idx_x_sorted][-50:].mean()
    low = model_y[idx_x_sorted][:50].mean()
    mean = model_y.mean()
    high_low_fracdiff = (high-low)/mean
    assert high_low_fracdiff > 0.1
