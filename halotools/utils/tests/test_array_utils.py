"""
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext

from .. import array_utils

__all__ = ('test_find_idx_nearest_val', )

fixed_seed = 43


def test_find_idx_nearest_val():

    # Create an array of randomly sorted integers
    x = np.arange(-10, 10)
    with NumpyRNGContext(fixed_seed):
        randomizer = np.random.random(len(x))
    ransort = np.argsort(randomizer)
    x = x[ransort]

    # Check that you always get an exactly matching element
    # when the inputs are elements of x
    v = np.arange(-10, 10)
    result = []
    for elt in v:
        closest_val = x[array_utils.find_idx_nearest_val(x, elt)]
        result.append(closest_val-elt)
    assert np.allclose(result, 0)

    # Check that you never differ by more than 0.5 when
    # your inputs are within the range spanned by x
    Npts = int(1e4)
    with NumpyRNGContext(fixed_seed):
        v = np.random.uniform(x.min()-0.5, x.max()+0.5, Npts)
    result = np.empty(Npts)
    for ii, elt in enumerate(v):
        closest_val = x[array_utils.find_idx_nearest_val(x, elt)]
        result[ii] = abs(closest_val - elt)
    assert np.all(result <= 0.5)

    # Check that values beyond the upper bound are handled correctly
    Npts = 10
    with NumpyRNGContext(fixed_seed):
        v = np.random.uniform(x.max()+10, x.max()+11, Npts)
    result = np.empty(Npts)
    for ii, elt in enumerate(v):
        closest_val = x[array_utils.find_idx_nearest_val(x, elt)]
        result[ii] = abs(closest_val - elt)
    assert np.all(result >= 10)
    assert np.all(result <= 11)

    # Check that values beyond the lower bound are handled correctly
    with NumpyRNGContext(fixed_seed):
        v = np.random.uniform(x.min()-11, x.min()-10, Npts)
    result = np.empty(Npts)
    for ii, elt in enumerate(v):
        closest_val = x[array_utils.find_idx_nearest_val(x, elt)]
        result[ii] = abs(closest_val - elt)
    assert np.all(result >= 10)
    assert np.all(result <= 11)
