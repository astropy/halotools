#!/usr/bin/env python

from __future__ import division, print_function
import numpy as np
import sys

from ..mock_survey import distant_observer_redshift, ra_dec_z

def test_distant_observer():
    
    N=100
    x = np.random.random((N,3))
    v = np.random.random((N,3))*0.1
    period = np.array([1,1,1])
    
    redshifts = distant_observer_redshift(x,v)
    
    assert len(redshifts)==N, "redshift array is not the correct size"
    
    redshifts = distant_observer_redshift(x,v,period=period)
    
    from astropy.constants import c
    c_km_s = c.to('km/s').value
    z_cos_max = period[2]*100.00/c_km_s
    
    assert len(redshifts)==N, "redshift array is not the correct size"
    assert np.max(redshifts)<=z_cos_max, "PBC is not handeled correctly for redshifts"


def test_ra_dec_z():
    
    
    N=100
    x = np.random.random((N,3))
    v = np.random.random((N,3))*0.1
    period = np.array([1,1,1])
    from astropy import cosmology
    cosmo = cosmology.FlatLambdaCDM(H0=0.7, Om0=0.3)
    
    ra, dec, z = ra_dec_z(x,v,cosmo=cosmo)
    
    assert len(ra)==N
    assert len(dec)==N
    assert len(z)==N
    assert np.all(ra<2.0*np.pi) & np.all(ra>0.0), "ra range is incorrect"
    assert np.all(dec>-1.0*np.pi/2.0) & np.all(dec<np.pi/2.0), "ra range is incorrect"
