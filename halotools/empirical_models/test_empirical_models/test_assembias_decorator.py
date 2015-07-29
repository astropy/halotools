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

@pytest.mark.slow
def test_silly():
    assert 4 < 5


class TestAssembiasDecorator(TestCase):

    def setup_class(self):
    	Npts = 1e4
    	mass = np.zeros(Npts) + 1e12
    	zform = np.linspace(0, 10, Npts)
        halo_designation = np.zeros(Npts, dtype=bool)
        halo_designation[Npts/2:] = True
    	d = {'halo_mvir': mass, 'halo_zform': zform, 'halo_is_old': halo_designation}
    	self.toy_halo_table = Table(d)

        fakesim = FakeSim()
        self.fake_halo_table = fakesim.halo_table

    def test_initialization(self):
    	"""
    	"""
    	baseline_model = Zheng07Cens
    	method_name_to_decorate='mean_occupation'

    	model = HeavisideAssembiasComponent(baseline_model=baseline_model, 
    		method_name_to_decorate=method_name_to_decorate, 
    		lower_bound = 0, upper_bound = 1, 
    		)

    	assert isinstance(model.baseline_model_instance, baseline_model)
    	assert hasattr(model, '_split_abcissa')
    	assert hasattr(model, '_split_ordinates')
    	assert hasattr(model, '_assembias_strength_abcissa')
    	assert hasattr(model, 'param_dict')
    	param_key = method_name_to_decorate + '_assembias_param1'
    	assert param_key in model.param_dict
    	keys = [key for key in model.param_dict.keys() if method_name_to_decorate + '_assembias_param' in key]
    	assert len(keys) == len(model._assembias_strength_abcissa)

    	output_split = model.percentile_splitting_function(halo_table=self.toy_halo_table)
    	assert np.all(output_split == 0.5)

    	model._split_ordinates = [0.75]
    	output_split = model.percentile_splitting_function(halo_table=self.toy_halo_table)
    	assert np.all(output_split == 0.75)

    	output_strength = model.assembias_strength(halo_table=self.toy_halo_table)
    	assert np.all(output_strength == 0.5)

        assert hasattr(model, 'mean_occupation')

    def test_behavior_default_model(self):
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
            galprop_abcissa = galprop_abcissa, galprop_ordinates = galprop_ordinates, galprop_key = galprop_key,
            method_name_to_decorate=method_name_to_decorate, 
            lower_bound = 0, upper_bound = 1, 
            split_func = split_func, halo_type_tuple = halo_type_tuple, 
            prim_haloprop_key = 'halo_mvir', sec_haloprop_key = 'halo_zform', 
            assembias_strength = 1
            )

        upper_bound = model.upper_bound_galprop_perturbation(halo_table = self.toy_halo_table)
        assert np.all(upper_bound == 0.5)

        lower_bound = model.lower_bound_galprop_perturbation(halo_table = self.toy_halo_table)
        assert np.all(lower_bound == -0.5)

        strength = model.assembias_strength(halo_table = self.toy_halo_table)
        assert np.all(strength == 1)

        result = model.mean_quiescent_fraction(halo_table = self.toy_halo_table)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

        young_mask = self.toy_halo_table['halo_is_old'] == False
        young_halos = self.toy_halo_table[young_mask]
        old_halos = self.toy_halo_table[np.invert(young_mask)]
        assert len(young_halos) == len(old_halos) == len(self.toy_halo_table)/2
        assert np.all(young_halos['halo_zform'] <= 5)
        assert np.all(old_halos['halo_zform'] >= 5)

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















