#!/usr/bin/env python
from .. import halo_prof_components as hpc
from astropy import cosmology
import numpy as np
from copy import copy

__all__ = ['test_HaloProfileModel', 'test_TrivialProfile','test_NFWProfile']

def test_HaloProfileModel():
    """ Method testing the abstract base class 
    `~halotools.empirical_models.HaloProfileModel`. 
    """
    prof_model_list = hpc.__all__
    parent_class = hpc.HaloProfileModel

    # First create a list of all sub-classes to test
    component_models_to_test = []
    for clname in prof_model_list:
        cl = getattr(hpc, clname)

        if (issubclass(cl, parent_class)) & (cl != parent_class):
            component_models_to_test.append(cl)

    # Now we will test that all sub-classes inherit the correct behavior
    for model_class in component_models_to_test:
        model_instance = model_class()

        assert hasattr(model_instance, 'cosmology')
        assert isinstance(model_instance.cosmology, cosmology.FlatLambdaCDM)

        assert hasattr(model_instance, '_set_prof_param_table_dict')
        input_dict = {}
        model_instance._set_prof_param_table_dict(input_dict)
        input_dict = model_instance.prof_param_table_dict
        model_instance._set_prof_param_table_dict(input_dict)

        assert hasattr(model_instance, 'build_inv_cumu_lookup_table')
        model_instance.build_inv_cumu_lookup_table()
        assert hasattr(model_instance, 'cumu_inv_func_table')
        assert type(model_instance.cumu_inv_func_table) == np.ndarray
        assert hasattr(model_instance, 'cumu_inv_param_table_dict')
        assert type(model_instance.cumu_inv_param_table_dict) == dict
        assert hasattr(model_instance, 'func_table_indices')
        assert type(model_instance.func_table_indices) == np.ndarray


def test_TrivialProfile():
    """ Tests of `~halotools.empirical_models.halo_prof_components.TrivialProfile`. 

    Mostly this function checks that the each of the following attributes is present, 
    and is an empty array, list, or dictionary:

        * ``cumu_inv_func_table``

        * ``cumu_inv_func_table_dict``

        * ``cumu_inv_param_table``

        * ``cumu_inv_param_table_dict``

        * ``halo_prof_func_dict``

        * ``haloprop_key_dict``
    """

    # Check that the initialized attributes are correct
    model_instance = hpc.TrivialProfile()
    assert model_instance.halo_prof_func_dict == {}
    assert model_instance.haloprop_key_dict == {}
    
    # Check that the lookup table attributes are correct
    model_instance.build_inv_cumu_lookup_table()
    assert len(model_instance.cumu_inv_func_table) == 0
    assert model_instance.cumu_inv_param_table_dict == {}
    assert len(model_instance.func_table_indices) == 0


def test_NFWProfile():
    """ Tests of `~halotools.empirical_models.halo_prof_components.NFWProfile`. 

    Basic summary of tests:

        * Default settings for lookup table arrays all have reasonable values and ranges. 

        * Discretization of NFW Profile with lookup table attains better than 0.1 percent accuracy for all relevant radii and concentrations

        * Lookup table recomputes properly when manually passed alternate discretizations 
    """

    # Check that the initialized attributes are correct
    model_instance = hpc.NFWProfile()
    assert hasattr(model_instance, 'cosmology')
    assert isinstance(model_instance.cosmology, cosmology.FlatLambdaCDM)
    assert model_instance._conc_parname == 'NFWmodel_conc'

    # Check that the lookup table attributes are correct
    model_instance.build_inv_cumu_lookup_table()
    assert np.all(model_instance.cumu_inv_param_table_dict[model_instance._conc_parname] > 0)
    assert np.all(model_instance.cumu_inv_param_table_dict[model_instance._conc_parname] < 1000)
    assert len(model_instance.cumu_inv_param_table_dict[model_instance._conc_parname]) >= 10
    assert (len(model_instance.cumu_inv_func_table) == 
        len(model_instance.func_table_indices) )

    # Verify accuracy of lookup table
    def get_lookup_table_frac_error(model, conc, test_radius):
        """ Function returns the fractional difference between the 
        exact value of the inverse cumulative mass PDF, and the 
        value inferred from the discretized lookup table. 
        """
        exact_result = model.cumulative_mass_PDF(test_radius, conc)
        conc_table = model.cumu_inv_param_table_dict[model._conc_parname]
        digitized_conc_index = np.digitize(np.array([conc]), conc_table)
        digitized_conc = conc_table[digitized_conc_index]
        func = model.cumu_inv_func_table[digitized_conc_index[0]]
        approximate_result = 10.**func(np.log10(exact_result))
        fracdiff = abs((approximate_result - test_radius)/test_radius)
        return fracdiff
    # Now we will verify that the lookup table method attains 
    # better than 0.1% accuracy at all relevant radii and concentrations
    radius = np.logspace(-3, 0, 15)
    test_conc_array = np.linspace(1, 25, 5)
    for test_conc in test_conc_array:
        frac_error = get_lookup_table_frac_error(
            model_instance, test_conc, radius)
        assert np.allclose(frac_error, 0, rtol = 1e-3, atol = 1e-3)

    # The lookup table should adjust properly when passed an input_dict
    input_dict = copy(model_instance.prof_param_table_dict)
    input_dict[model_instance._conc_parname] = (1.0, 25.0, 0.04)
    model_instance._set_prof_param_table_dict(input_dict)
    assert model_instance.prof_param_table_dict == input_dict
    input_dict[model_instance._conc_parname] = (2.0, 20.0, 0.03)
    assert model_instance.prof_param_table_dict != input_dict
    model_instance.build_inv_cumu_lookup_table(
        prof_param_table_dict=input_dict)
    assert model_instance.prof_param_table_dict == input_dict
    dict_persistence_check = copy(model_instance.prof_param_table_dict)
    input_dict['some_irrelevant_key'] = 4
    model_instance.build_inv_cumu_lookup_table(
        prof_param_table_dict=input_dict)
    assert dict_persistence_check == model_instance.prof_param_table_dict














