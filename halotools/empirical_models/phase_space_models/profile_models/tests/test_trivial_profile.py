"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from unittest import TestCase
from astropy.cosmology import WMAP9

from ..trivial_profile import TrivialProfile

from .... import model_defaults

from .....sim_manager import sim_defaults


__all__ = ('TestTrivialProfile', 'test_enclosed_mass1')


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


class TestTrivialProfile(TestCase):
    """ Tests of `~halotools.empirical_models.TrivialProfile`.

    The TrivialProfile has very little functionality. It is mostly just a standard-form class
    called during mock population to assign the positions of central galaxies to be equal
    to the positions of their host halos. So `TestTrivialProfile` mostly
    just enforces that the necessary attributes of ``TrivialProfile`` instances exist.

    """

    def setup_class(self):
        """ Pre-load various arrays into memory for use by all tests.
        """
        self.default_model = TrivialProfile()
        self.wmap9_model = TrivialProfile(cosmology=WMAP9)
        self.m200_model = TrivialProfile(mdef='200m')

    def test_instance_attrs(self):
        """ Require that all model variants have ``cosmology``, ``redshift`` and ``mdef`` attributes.
        """
        assert self.default_model.cosmology == sim_defaults.default_cosmology
        assert self.m200_model.cosmology == sim_defaults.default_cosmology
        assert self.wmap9_model.cosmology == WMAP9

        assert self.default_model.redshift == sim_defaults.default_redshift
        assert self.m200_model.redshift == sim_defaults.default_redshift
        assert self.wmap9_model.redshift == sim_defaults.default_redshift

        assert self.default_model.mdef == model_defaults.halo_mass_definition
        assert self.m200_model.mdef == '200m'
        assert self.wmap9_model.mdef == model_defaults.halo_mass_definition
