"""
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext

from ..distribution_matching import distribution_matching_indices, resample_x_to_match_y
from ..distribution_matching import bijective_distribution_matching


__all__ = ('test_distribution_matching_indices1', )

fixed_seed = 43


def test_distribution_matching_indices1():
    npts1, npts2 = int(1e5), int(1e4)
    nselect = int(2e4)
    with NumpyRNGContext(fixed_seed):
        input_distribution = np.random.normal(loc=0, scale=1, size=npts1)
        output_distribution = np.random.normal(loc=1, scale=0.5, size=npts2)
    xmin = min(input_distribution.min(), output_distribution.min())
    xmax = min(input_distribution.max(), output_distribution.max())
    nbins = 50
    bins = np.linspace(xmin, xmax, nbins)
    indices = distribution_matching_indices(
            input_distribution, output_distribution, nselect, bins, seed=fixed_seed)
    result = input_distribution[indices]

    percentile_table = np.linspace(0.01, 0.99, 25)
    result_percentiles = np.percentile(result, percentile_table)
    correct_percentiles = np.percentile(output_distribution, percentile_table)
    assert np.allclose(result_percentiles, correct_percentiles, rtol=0.1)


def test_resample_x_to_match_y():
    """
    """
    nx, ny = int(9.9999e5), int(1e6)
    with NumpyRNGContext(fixed_seed):
        x = np.random.normal(loc=0, size=nx, scale=1)
        y = np.random.normal(loc=0.5, size=ny, scale=0.25)
    bins = np.linspace(y.min(), y.max(), 100)
    indices = resample_x_to_match_y(x, y, bins, seed=fixed_seed)
    rescaled_x = x[indices]

    idx_x_sorted = np.argsort(x)
    assert np.all(np.diff(rescaled_x[idx_x_sorted]) >= 0)

    try:
        result, __ = np.histogram(rescaled_x, bins, density=True)
        correct_result, __ = np.histogram(y, bins, density=True)
        assert np.allclose(result, correct_result, atol=0.02)
    except TypeError:
        pass


def test_bijective_distribution_matching():
    npts = int(1e5)
    with NumpyRNGContext(fixed_seed):
        x_in = np.random.normal(loc=0, scale=0.5, size=npts)
        x_desired = np.random.normal(loc=2, scale=1, size=npts)

    x_out = bijective_distribution_matching(x_in, x_desired)
    assert np.allclose(np.sort(x_out), np.sort(x_desired))

    idx_x_in_sorted = np.argsort(x_in)
    assert np.all(np.diff(x_out[idx_x_in_sorted])>=0)
