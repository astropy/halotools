""" Module providing unit-testing for the functions in
the `~halotools.mock_observables.mock_survey` module
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.tests.helper import pytest

from ..mock_survey import distant_observer_redshift, ra_dec_z

__all__=['test_distant_observer', 'test_ra_dec_z']

#create some toy data to test functions
N=100
x = np.random.random((N, 3))
v = np.random.random((N, 3))*0.1
period = np.array([1.0, 1.0, 1.0])


@pytest.mark.slow
def test_distant_observer():
    """
    test distant observer function
    """
    redshifts = distant_observer_redshift(x, v)

    assert len(redshifts)==N, "redshift array is not the correct size"

    redshifts = distant_observer_redshift(x, v, period=period)

    from astropy.constants import c
    c_km_s = c.to('km/s').value
    z_cos_max = period[2]*100.00/c_km_s

    assert len(redshifts)==N, "redshift array is not the correct size"
    assert np.max(redshifts)<=z_cos_max, "PBC is not handeled correctly for redshifts"


@pytest.mark.slow
def test_ra_dec_z():
    """
    test ra_dec_z function
    """
    from astropy import cosmology
    cosmo = cosmology.FlatLambdaCDM(H0=0.7, Om0=0.3)

    ra, dec, z = ra_dec_z(x, v, cosmo=cosmo)

    assert len(ra)==N
    assert len(dec)==N
    assert len(z)==N
    assert np.all(ra<2.0*np.pi) & np.all(ra>0.0), "ra range is incorrect"
    assert np.all(dec>-1.0*np.pi/2.0) & np.all(dec<np.pi/2.0), "ra range is incorrect"
