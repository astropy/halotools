""" Module provides unit-testing for the `~halotools.mock_observables.tpcf_multipole` function.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext
import pytest

from ..tpcf_multipole import tpcf_multipole

__all__ = ['test_tpcf_multipole_monopole', 'test_tpcf_multipole_odd_multipoles']

fixed_seed = 43


def test_tpcf_multipole_monopole():
    """
    test s_mu_tpcf autocorrelation without periodic boundary conditons.
    """
    
    xi_mu_s_fake = np.ones((49, 49))
    mu_bins = np.linspace(0.0, 1.0, 50)
    result = tpcf_multipole(xi_mu_s_fake, mu_bins, order=0)
    
    
    assert np.all(result) == 1.0, "monopole should be exactly 1 for all mu"
    
def test_tpcf_multipole_odd_monopoles():
    """
    test s_mu_tpcf autocorrelation without periodic boundary conditons.
    """
    
    xi_mu_s_fake = np.ones((49, 49))
    mu_bins = np.linspace(0.0, 1.0, 50)
    result_1 = tpcf_multipole(xi_mu_s_fake, mu_bins, order=1)
    
    result_3 = tpcf_multipole(xi_mu_s_fake, mu_bins, order=3)
    
    result_5 = tpcf_multipole(xi_mu_s_fake, mu_bins, order=5)
    
    exact_result = np.zeros(49)
    
    assert np.allclose(result_1,exact_result, atol=1e-14), "l=1 should be exactly 0 for all mu"
    assert np.allclose(result_3,exact_result, atol=1e-14), "l=3 should be exactly 0 for all mu"
    assert np.allclose(result_5,exact_result, atol=1e-14), "l-5 should be exactly 0 for all mu"
    

