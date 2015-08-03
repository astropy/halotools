#!/usr/bin/env python

import numpy as np 
from astropy.table import Table 

from ..sfr_components import BinaryGalpropInterpolModel as BinaryModel
from .. import model_defaults

def test_BinaryGalpropInterpolModel():
    """ Function testing the initialization of 
    `~halotools.empirical_models.sfr_components.BinaryGalpropInterpolModel`. 
    """
    abcissa, ordinates = [12, 15], [1/3., 0.9]
    m = BinaryModel(galprop_key='late_type', 
        galprop_abcissa = abcissa, galprop_ordinates = ordinates, 
        prim_haloprop_key = 'vpeak_host', gal_type = 'sats')

    Npts = 5e3
    testmass12 = np.ones(Npts)*1e12
    testmass135 = np.ones(Npts)*10.**13.5
    testmass15 = np.ones(Npts)*1e15

    frac12 = m.mean_late_type_fraction(prim_haloprop = testmass12)
    frac135 = m.mean_late_type_fraction(prim_haloprop = testmass135)
    frac15 = m.mean_late_type_fraction(prim_haloprop = testmass15)

    midval = 0.5*np.sum(ordinates)
    assert np.all(frac12 == ordinates[0])
    assert np.all(frac135 == midval)
    assert np.all(frac15 == ordinates[1])

    mean_mcfrac12 = np.mean(m.mc_late_type(prim_haloprop = testmass12, seed=43))
    mean_mcfrac135 = np.mean(m.mc_late_type(prim_haloprop = testmass135, seed=43))
    mean_mcfrac15 = np.mean(m.mc_late_type(prim_haloprop = testmass15, seed=43))

    np.testing.assert_allclose(mean_mcfrac12, frac12, rtol=1e-2, atol=1.e-2)
    np.testing.assert_allclose(mean_mcfrac135, frac135, rtol=1e-2, atol=1.e-2)
    np.testing.assert_allclose(mean_mcfrac15, frac15, rtol=1e-2, atol=1.e-2)







