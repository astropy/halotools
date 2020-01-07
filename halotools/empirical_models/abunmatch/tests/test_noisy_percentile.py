"""
"""
import numpy as np
from scipy.stats import spearmanr
from astropy.utils.misc import NumpyRNGContext

from ..noisy_percentile import noisy_percentile


__all__ = ('test_noisy_percentile1', )


fixed_seed = 43
npts = int(1e5)
with NumpyRNGContext(fixed_seed):
    u = np.random.uniform(0, 1, npts)
    uran = np.random.uniform(0, 1, npts)


def test_noisy_percentile1():

    r_desired = 0.5
    u2 = noisy_percentile(u, r_desired, seed=fixed_seed+1)
    r_achieved = spearmanr(u, u2)[0]
    assert np.allclose(r_desired, r_achieved, atol=0.02)


def test_noisy_percentile2():

    r_desired = 0.1
    u2 = noisy_percentile(u, r_desired, seed=fixed_seed+1)
    r_achieved = spearmanr(u, u2)[0]
    assert np.allclose(r_desired, r_achieved, atol=0.02)


def test_noisy_percentile3():

    r_desired = 0.99
    u2 = noisy_percentile(u, r_desired, random_percentile=uran)
    r_achieved = spearmanr(u, u2)[0]
    assert np.allclose(r_desired, r_achieved, atol=0.02)


def test_negative_unity_noisy_percentile():
    """Regression test for #939.
    """
    r_desired = -1
    u2 = noisy_percentile(u, r_desired, seed=fixed_seed)
    assert np.allclose(u2, u[::-1])
