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
from ...utils.table_utils import SampleSelector, compute_conditional_percentiles
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

        d1 = {'halo_mvir': mass, 'halo_zform': zform}
        self.toy_halo_table1 = Table(d1)

        halo_zform_percentile = (np.arange(Npts)+1) / float(Npts)
        halo_zform_percentile = 1. - halo_zform_percentile[::-1]
        d2 = {'halo_mvir': mass, 'halo_zform': zform, 'halo_zform_percentile': halo_zform_percentile}
        self.toy_halo_table2 = Table(d2)

        fakesim = FakeSim()
        self.fake_halo_table = fakesim.halo_table


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


    def splitting_func_tests(self, model, halo_table, **kwargs):
        """
        """
        output_split = model.percentile_splitting_function(halo_table=halo_table)
        assert np.all(output_split == kwargs['correct_split'])


    def perturbation_bound_tests(self, model, halo_table, **kwargs):
        """
        """
        upper_bound = model.upper_bound_galprop_perturbation(halo_table = halo_table)
        assert np.all(upper_bound == kwargs['correct_upper_pert_bound'])
        lower_bound = model.lower_bound_galprop_perturbation(halo_table = halo_table)
        assert np.all(lower_bound == kwargs['correct_lower_pert_bound'])


    def assembias_strength_tests(self, model, halo_table, **kwargs):
        """
        """
        strength = model.assembias_strength(halo_table = halo_table)
        assert np.all(strength == kwargs['assembias_strength'])


    def decorated_method_tests(self, model, halo_table, **kwargs):
        """
        """
        method = getattr(model, model._method_name_to_decorate)
        baseline_method = getattr(model.baseline_model_instance, model._method_name_to_decorate)
        result = method(halo_table = halo_table)
        baseline_result = baseline_method(halo_table = halo_table)

        assert np.all(result >= model._lower_bound)
        assert np.all(result <= model._upper_bound)
        assert np.all(baseline_result == 0.5)

        halo_zform_percentile = (np.arange(len(halo_table))+1) / float(len(halo_table)) 
        halo_zform_percentile = 1. - halo_zform_percentile[::-1]

        old_mask = halo_zform_percentile >= kwargs['correct_split']
        old_halos = halo_table[old_mask]
        assert len(old_halos) == kwargs['correct_split']*len(halo_table)

        young_halos = halo_table[np.invert(old_mask)]
        assert len(young_halos) == (1-kwargs['correct_split'])*len(halo_table)

        if 'assembias_strength' and 'split' in kwargs:
            strength = kwargs['assembias_strength']
            split = kwargs['split']
            if strength > 0:
                dx1 = strength*model.upper_bound_galprop_perturbation(
                    halo_table = halo_table)
            else:
                dx1 = strength*model.lower_bound_galprop_perturbation(
                    halo_table = halo_table)
            dx2 = -split*dx1/(1-split)

            assert np.all(result[old_mask] == baseline_result[old_mask] + dx1[old_mask])

            assert np.all(result[np.invert(old_mask)] == 
                baseline_result[np.invert(old_mask)] + dx2[np.invert(old_mask)])





    def test_binary_galprop_models(self):
        """
        """

        def execute_all_behavior_tests(correct_upper_pert_bound, 
            correct_lower_pert_bound, 
            correct_split, **kwargs):

            model = HeavisideAssembiasComponent(**kwargs)

            is_old = self.toy_halo_table2['halo_zform_percentile'] >= correct_split
            table3 = self.toy_halo_table1
            table3['halo_is_old'] = is_old

            halo_table_gen = (self.toy_halo_table1, self.toy_halo_table2, table3)
            for ii, halo_table in enumerate(halo_table_gen):
                if ii == 2:
                    kwargs['halo_type_tuple'] = ('halo_is_old', True, False)

                model = HeavisideAssembiasComponent(**kwargs)
                self.constructor_tests(model)
                self.splitting_func_tests(model, halo_table = halo_table, 
                    correct_split=correct_split, **kwargs)
                self.assembias_strength_tests(model, halo_table = halo_table, **kwargs)
                self.perturbation_bound_tests(model, halo_table = halo_table, 
                    correct_upper_pert_bound = correct_upper_pert_bound, 
                    correct_lower_pert_bound = correct_lower_pert_bound, **kwargs)
                self.decorated_method_tests(model, halo_table = halo_table, 
                    correct_split = correct_split, **kwargs)





        baseline_model = BinaryGalpropInterpolModel
        galprop_abcissa = [12]
        galprop_ordinates = [0.5]
        galprop_key='quiescent'
        method_name_to_decorate='mean_'+galprop_key+'_fraction'
        lower_bound = 0
        upper_bound = 1
        split = 0.5
        def split_func(**kwargs):
            return np.zeros(len(kwargs['halo_table'])) + split
        halo_type_tuple = ('halo_is_old', True, False)
        prim_haloprop_key = 'halo_mvir'
        sec_haloprop_key = 'halo_zform'
        assembias_strength = 1

        kwargs = (
            {'baseline_model': baseline_model, 
            'galprop_abcissa': galprop_abcissa, 
            'galprop_ordinates': galprop_ordinates, 
            'galprop_key': galprop_key, 
            'method_name_to_decorate': method_name_to_decorate, 
            'lower_bound': lower_bound, 
            'upper_bound': upper_bound, 
            'split_func': split_func, 
            'prim_haloprop_key': prim_haloprop_key, 
            'sec_haloprop_key': sec_haloprop_key, 
            'assembias_strength': assembias_strength
            }
            )

        correct_upper_pert_bound = 0.5
        correct_lower_pert_bound = -0.5
        correct_split = 0.5
        # print("...Working on input split_func case")
        execute_all_behavior_tests(correct_upper_pert_bound, 
            correct_lower_pert_bound, correct_split, **kwargs)
        del kwargs['split_func']
        kwargs['split'] = split
        # print("...working on input scalar split case")
        execute_all_behavior_tests(correct_upper_pert_bound, 
            correct_lower_pert_bound, correct_split, **kwargs)




        # assembias_strength = 0.5
        # kwargs = (
        #     {'baseline_model': baseline_model, 
        #     'galprop_abcissa': galprop_abcissa, 
        #     'galprop_ordinates': galprop_ordinates, 
        #     'galprop_key': galprop_key, 
        #     'method_name_to_decorate': method_name_to_decorate, 
        #     'lower_bound': lower_bound, 
        #     'upper_bound': upper_bound, 
        #     'split': split, 
        #     'prim_haloprop_key': prim_haloprop_key, 
        #     'sec_haloprop_key': sec_haloprop_key, 
        #     'assembias_strength': assembias_strength
        #     }
        #     )
        # execute_all_behavior_tests(correct_upper_pert_bound, 
        #     correct_lower_pert_bound, correct_split, **kwargs)





        # split = 0.25
        # correct_upper_pert_bound = 0.25
        # correct_lower_pert_bound = -0.25
        # correct_split = 0.25
        # kwargs['split'] = split
        # execute_all_behavior_tests(correct_upper_pert_bound, 
        #     correct_lower_pert_bound, correct_split, **kwargs)

        
        







