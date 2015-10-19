#!/usr/bin/env python
from astropy import cosmology
import numpy as np
from copy import copy

__all__ = ['test_HaloProfileModel', 'test_TrivialProfile','test_NFWProfile']

def test_HaloProfileModel():
    """ Method testing the abstract base class 
    `~halotools.empirical_models.HaloProfileModel`. 
    """
    pass 
    # The following tests are useful but need to be rewritten according to 
    # the changes made by the prof_overhaul branch 

    # prof_model_list = hpc.__all__
    # parent_class = hpc.HaloProfileModel

    # # First create a list of all sub-classes to test
    # component_models_to_test = []
    # for clname in prof_model_list:
    #     cl = getattr(hpc, clname)

    #     if (issubclass(cl, parent_class)) & (cl != parent_class):
    #         component_models_to_test.append(cl)

    # # Now we will test that all sub-classes inherit the correct behavior
    # for model_class in component_models_to_test:
    #     model_instance = model_class(cosmology=cosmology.WMAP7, redshift=2)

    #     assert hasattr(model_instance, 'cosmology')
    #     assert isinstance(model_instance.cosmology, cosmology.FlatLambdaCDM)
    #     assert hasattr(model_instance, 'redshift')

    #     assert hasattr(model_instance, 'build_inv_cumu_lookup_table')
    #     model_instance.build_inv_cumu_lookup_table()
    #     assert hasattr(model_instance, 'cumu_inv_func_table')
    #     assert type(model_instance.cumu_inv_func_table) == np.ndarray
    #     assert hasattr(model_instance, 'func_table_indices')
    #     assert type(model_instance.func_table_indices) == np.ndarray


def test_TrivialProfile():
    """ Tests of `~halotools.empirical_models.halo_prof_components.TrivialProfile`. 

    Mostly this function checks that the each of the following attributes is present, 
    and is an empty array, list, or dictionary:

        * ``cumu_inv_func_table``

        * ``cumu_inv_func_table_dict``

        * ``cumu_inv_param_table``

    """
    pass 
    # The following tests are useful but need to be rewritten according to 
    # the changes made by the prof_overhaul branch 

    # # Check that the initialized attributes are correct
    # model_instance = hpc.TrivialProfile()
    # assert model_instance.prof_param_keys == []
    
    # # Check that the lookup table attributes are correct
    # model_instance.build_inv_cumu_lookup_table()
    # assert len(model_instance.cumu_inv_func_table) == 0
    # assert len(model_instance.func_table_indices) == 0


def test_NFWProfile():
    """ Tests of `~halotools.empirical_models.halo_prof_components.NFWProfile`. 

    Basic summary of tests:

        * Default settings for lookup table arrays all have reasonable values and ranges. 

        * Discretization of NFW Profile with lookup table attains better than 0.1 percent accuracy for all relevant radii and concentrations

        * Lookup table recomputes properly when manually passed alternate discretizations 
    """
    pass 
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
















