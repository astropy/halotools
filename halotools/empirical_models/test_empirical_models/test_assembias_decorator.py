#!/usr/bin/env python

from unittest import TestCase
import pytest
from copy import copy

import numpy as np 
from astropy.table import Table

from ..assembias_decorator import HeavisideAssembiasComponent
from .. import model_defaults
from ..hod_components import Zheng07Cens, Leauthaud11Cens
from ..sfr_components import BinaryGalpropInterpolModel
from ...sim_manager import FakeSim
from ...utils.table_utils import SampleSelector
from ...utils.array_utils import array_like_length as custom_len


class TestAssembiasDecorator(TestCase):
    """
    """

    def setup_class(self):
        """
        """
    	Npts = 1e4
    	mass = np.zeros(Npts) + 1e12
    	zform = np.linspace(0, 10, Npts)
        halo_designation = np.zeros(Npts, dtype=bool)
        halo_designation[Npts/2:] = True
    	d = {'halo_mvir': mass, 'halo_zform': zform, 'halo_is_old': halo_designation}
    	self.toy_halo_table = Table(d)

        fakesim = FakeSim()
        self.fake_halo_table = fakesim.halo_table

    def test_setup(self):
        """
        """
        is_old_mask = self.toy_halo_table['halo_is_old'] == True
        young_halos = self.toy_halo_table[np.invert(is_old_mask)]
        old_halos = self.toy_halo_table[is_old_mask]
        assert len(young_halos) == len(old_halos) == len(self.toy_halo_table)/2
        assert np.all(young_halos['halo_zform'] <= 5)
        assert np.all(old_halos['halo_zform'] >= 5)

    def constructor_tests(self, model):
    	"""
    	"""
    	assert hasattr(model, '_assembias_strength_abcissa')
    	assert hasattr(model, 'param_dict')
        assert hasattr(model, '_method_name_to_decorate')
    	param_key = model._method_name_to_decorate + '_assembias_param1'
    	assert param_key in model.param_dict
    	keys = [key for key in model.param_dict.keys() if model._method_name_to_decorate + '_assembias_param' in key]
    	assert len(keys) == len(model._assembias_strength_abcissa) 


    def splitting_func_tests(self, model, **kwargs):
        """
        """
        output_split = model.percentile_splitting_function(halo_table=self.toy_halo_table)
        if 'split' in kwargs:
            assert np.all(output_split == kwargs['split'])


    def perturbation_bound_tests(self, model, **kwargs):
        """
        """
        upper_bound = model.upper_bound_galprop_perturbation(halo_table = self.toy_halo_table)
        assert np.all(upper_bound == kwargs['upper_bound'])
        lower_bound = model.lower_bound_galprop_perturbation(halo_table = self.toy_halo_table)
        assert np.all(lower_bound == kwargs['lower_bound'])


    def assembias_strength_tests(self, model, **kwargs):
        """
        """
        strength = model.assembias_strength(halo_table = self.toy_halo_table)
        assert np.all(strength == kwargs['strength'])


    def decorated_method_tests(self, model, **kwargs):
        """
        """
        result = model.mean_quiescent_fraction(halo_table = self.toy_halo_table)
        assert np.all(result >= model._lower_bound)
        assert np.all(result <= model._upper_bound)

        young_result = result[young_mask]
        assert np.all(young_result == 0)
        old_result = result[np.invert(young_mask)]
        assert np.all(old_result == 1)

        model2 = HeavisideAssembiasComponent(baseline_model=baseline_model, 
            galprop_abcissa = galprop_abcissa, galprop_ordinates = galprop_ordinates, galprop_key = galprop_key,
            method_name_to_decorate=method_name_to_decorate, 
            lower_bound = 0, upper_bound = 1, 
            split = 0.5, prim_haloprop_key = 'halo_mvir', sec_haloprop_key = 'halo_zform', 
            assembias_strength = 1
            )

        result2 = model2.mean_quiescent_fraction(halo_table = self.toy_halo_table)
        young_result2 = result2[young_mask]
        assert np.all(young_result2 == 0)
        old_result2 = result2[np.invert(young_mask)]
        assert np.all(old_result2 == 1)


    def test_binary_galprop_models(self):
        """
        """
        baseline_model = BinaryGalpropInterpolModel
        galprop_key='quiescent'
        galprop_abcissa = [12]
        galprop_ordinates = [0.5]
        method_name_to_decorate='mean_'+galprop_key+'_fraction'

        def split_func(**kwargs):
            return np.zeros(custom_len(kwargs['halo_table'])) + 0.5

        halo_type_tuple = ('halo_is_old', True, False)

        model = HeavisideAssembiasComponent(baseline_model=baseline_model, 
            galprop_abcissa = galprop_abcissa, galprop_ordinates = galprop_ordinates, 
            galprop_key = galprop_key,
            method_name_to_decorate=method_name_to_decorate, 
            lower_bound = 0, upper_bound = 1, 
            split_func = split_func, halo_type_tuple = halo_type_tuple, 
            prim_haloprop_key = 'halo_mvir', sec_haloprop_key = 'halo_zform', 
            assembias_strength = 1
            )

        self.constructor_tests(model)
        self.perturbation_bound_tests(model, 
            upper_bound = 0.5, lower_bound = -0.5)
        self.assembias_strength_tests(model, strength = 1)
        self.splitting_func_tests(model, split=0.5)


        # baseline_model = Zheng07Cens
        # method_name_to_decorate='mean_occupation'

        # model = HeavisideAssembiasComponent(baseline_model=baseline_model, 
        #   method_name_to_decorate=method_name_to_decorate, 
        #   lower_bound = 0, upper_bound = 1, 
        #   )














