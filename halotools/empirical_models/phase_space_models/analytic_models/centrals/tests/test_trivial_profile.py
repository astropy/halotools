"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.cosmology import WMAP9

from ..trivial_profile import TrivialProfile

from ..... import model_defaults

from ......sim_manager import sim_defaults


__all__ = ('test_enclosed_mass1', 'test_enclosed_mass2')


def test_enclosed_mass1():
    """
    """
    model = TrivialProfile()
    m = model.enclosed_mass(0.01, 1e12)
    assert np.all(m == 1e12)


def test_enclosed_mass2():
    """
    """
    model = TrivialProfile()
    m = model.enclosed_mass([0.1, 0.2], 1e12)
    assert np.all(m == 1e12)


def test_dimensionless_mass_density1():
    model = TrivialProfile()
    d = model.dimensionless_mass_density(0.1, 1e12)


def test_dimensionless_mass_density2():
    model = TrivialProfile()
    d = model.dimensionless_mass_density([0.1, 0.2], 1e12)


def test_instance_attrs():
    """ Require that all model variants have ``cosmology``, ``redshift`` and ``mdef`` attributes.
    """
    default_model = TrivialProfile()
    wmap9_model = TrivialProfile(cosmology=WMAP9)
    m200_model = TrivialProfile(mdef='200m')
    assert default_model.cosmology == sim_defaults.default_cosmology
    assert m200_model.cosmology == sim_defaults.default_cosmology
    assert wmap9_model.cosmology == WMAP9

    assert default_model.redshift == sim_defaults.default_redshift
    assert m200_model.redshift == sim_defaults.default_redshift
    assert wmap9_model.redshift == sim_defaults.default_redshift

    assert default_model.mdef == model_defaults.halo_mass_definition
    assert m200_model.mdef == '200m'
    assert wmap9_model.mdef == model_defaults.halo_mass_definition
