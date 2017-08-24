"""
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext

from ..distribution_matching import distribution_matching_indices

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
