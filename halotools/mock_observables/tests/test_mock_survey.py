""" Module providing unit-testing for the functions in
the `~halotools.mock_observables.mock_survey` module
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pytest
from astropy.utils.misc import NumpyRNGContext

from ..mock_survey import ra_dec_z

__all__ = ('test_ra_dec_z', )

fixed_seed = 43

# create some toy data to test functions
N = 100
with NumpyRNGContext(fixed_seed):
    x = np.random.random((N, 3))
    v = np.random.random((N, 3))*0.1
period = np.array([1.0, 1.0, 1.0])


@pytest.mark.slow
def test_ra_dec_z():
    """
    test ra_dec_z function
    """
    from astropy import cosmology
    cosmo = cosmology.FlatLambdaCDM(H0=0.7, Om0=0.3)

    ra, dec, z = ra_dec_z(x, v, cosmo=cosmo)

    assert len(ra) == N
    assert len(dec) == N
    assert len(z) == N
    assert np.all(ra < 2.0*np.pi) & np.all(ra > 0.0), "ra range is incorrect"
    assert np.all(dec > -1.0*np.pi/2.0) & np.all(dec < np.pi/2.0), "ra range is incorrect"
