#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function, 
    unicode_literals)

import numpy as np 

from unittest import TestCase
import pytest 

from astropy.cosmology import WMAP9, Planck13
from astropy import units as u

from ..profile_helpers import *
from ..nfw_profile import NFWProfile

from .....custom_exceptions import HalotoolsError


__all__ = ['TestNFWProfile']

class TestNFWProfile(TestCase):
    """ Tests of `~halotools.empirical_models.phase_space_models.profile_models.NFWProfile`. 

    Basic summary of tests:

        * Default settings for lookup table arrays all have reasonable values and ranges. 

        * Discretization of NFW Profile with lookup table attains better than 0.1 percent accuracy for all relevant radii and concentrations

        * Lookup table recomputes properly when manually passed alternate discretizations 
    """

    def setup_class(self):
        self.default_nfw = NFWProfile()
        self.wmap9_nfw = NFWProfile(cosmology = WMAP9)
        self.m200_nfw = NFWProfile(mdef = '200m')

    def test_behavior_consistency(self):
        """
        """
        



    # The following tests are useful but need to be rewritten according to 
    # the changes made by the prof_overhaul branch 

    # # Check that the initialized attributes are correct
    # model_instance = hpc.NFWProfile()
    # assert hasattr(model_instance, 'cosmology')
    # assert isinstance(model_instance.cosmology, cosmology.FlatLambdaCDM)


    # # Check that the lookup table attributes are correct
    # model_instance.build_inv_cumu_lookup_table()

    # assert np.all(model_instance.NFWmodel_conc_lookup_table_bins > 0)
    # assert np.all(model_instance.NFWmodel_conc_lookup_table_bins < 1000)


    # assert len(model_instance.NFWmodel_conc_lookup_table_bins) >= 10
    # assert (len(model_instance.cumu_inv_func_table) == 
    #     len(model_instance.func_table_indices) )

    # # Verify accuracy of lookup table
    # def get_lookup_table_frac_error(model, conc, test_radius):
    #     """ Function returns the fractional difference between the 
    #     exact value of the inverse cumulative mass PDF, and the 
    #     value inferred from the discretized lookup table. 
    #     """
    #     exact_result = model.cumulative_mass_PDF(test_radius, conc)
    #     conc_table = model.NFWmodel_conc_lookup_table_bins        
    #     digitized_conc_index = np.digitize(np.array([conc]), conc_table)
    #     digitized_conc = conc_table[digitized_conc_index]
    #     func = model.cumu_inv_func_table[digitized_conc_index[0]]
    #     approximate_result = 10.**func(np.log10(exact_result))
    #     fracdiff = abs((approximate_result - test_radius)/test_radius)
    #     return fracdiff
    # # Now we will verify that the lookup table method attains 
    # # better than 0.1% accuracy at all relevant radii and concentrations
    # radius = np.logspace(-3, 0, 15)
    # test_conc_array = np.linspace(1, 25, 5)
    # for test_conc in test_conc_array:
    #     frac_error = get_lookup_table_frac_error(
    #         model_instance, test_conc, radius)
    #     assert np.allclose(frac_error, 0, rtol = 1e-3, atol = 1e-3)

    # # The lookup table should adjust properly when passed an input_dict
    # initial_NFWmodel_conc_lookup_table_min = copy(model_instance.NFWmodel_conc_lookup_table_min)
    # initial_NFWmodel_conc_lookup_table_max = copy(model_instance.NFWmodel_conc_lookup_table_max)
    # initial_NFWmodel_conc_lookup_table_spacing = copy(model_instance.NFWmodel_conc_lookup_table_spacing)
    # initial_NFWmodel_conc_lookup_table_bins = copy(model_instance.NFWmodel_conc_lookup_table_bins)

    # model_instance.NFWmodel_conc_lookup_table_min -= 0.05
    # model_instance.NFWmodel_conc_lookup_table_min += 0.05
    # model_instance.NFWmodel_conc_lookup_table_spacing *= 0.9
    # model_instance.build_inv_cumu_lookup_table()
    # assert model_instance.NFWmodel_conc_lookup_table_bins != initial_NFWmodel_conc_lookup_table_bins

    # model_instance.NFWmodel_conc_lookup_table_min = initial_NFWmodel_conc_lookup_table_min
    # model_instance.NFWmodel_conc_lookup_table_max = initial_NFWmodel_conc_lookup_table_max
    # model_instance.NFWmodel_conc_lookup_table_spacing = initial_NFWmodel_conc_lookup_table_spacing
    # model_instance.build_inv_cumu_lookup_table()
    # assert np.all(model_instance.NFWmodel_conc_lookup_table_bins == initial_NFWmodel_conc_lookup_table_bins)















