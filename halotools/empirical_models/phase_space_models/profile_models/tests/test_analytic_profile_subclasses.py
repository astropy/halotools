#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function,
    unicode_literals)


from unittest import TestCase

from astropy.cosmology import WMAP9, FLRW

from ... import profile_models

__all__ = ['TestAnalyticDensityProf']


class TestAnalyticDensityProf(TestCase):
    """ Test the existence and reasonableness of all instances of
    sub-classes of `~halotools.empirical_models.AnalyticDensityProf`.
    """

    def setup_class(self):
        """ Pre-load various arrays into memory for use by all tests.
        """
        self.prof_model_list = (
            profile_models.NFWProfile, profile_models.TrivialProfile
            )

    def test_attr_inheritance(self):
        """ Test that all sub-classes of
        `~halotools.empirical_models.AnalyticDensityProf`
        correctly inherit the necessary attributes and methods.
        """

        # Test that all sub-classes inherit the correct attributes
        for model_class in self.prof_model_list:
            model_instance = model_class(cosmology=WMAP9, redshift=2, mdef='vir')

            assert hasattr(model_instance, 'cosmology')
            assert isinstance(model_instance.cosmology, FLRW)

            assert hasattr(model_instance, 'redshift')
            assert model_instance.redshift == 2

            assert hasattr(model_instance, 'mdef')
            assert model_instance.mdef == 'vir'

            assert hasattr(model_instance, 'halo_boundary_key')
            assert model_instance.halo_boundary_key == 'halo_rvir'

            assert hasattr(model_instance, 'prim_haloprop_key')
            assert model_instance.prim_haloprop_key == 'halo_mvir'

            assert hasattr(model_instance, 'param_dict')
            assert hasattr(model_instance, 'publications')
            assert hasattr(model_instance, 'prof_param_keys')

            assert hasattr(model_instance, 'virial_velocity')
            vvir = model_instance.virial_velocity(total_mass=1e12)
            # For fixed mdef, the value of vvir should not depend on the profile model
            try:
                assert vvir == vvir_last
            except NameError:
                vvir_last = vvir
