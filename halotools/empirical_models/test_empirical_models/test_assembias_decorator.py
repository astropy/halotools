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

        d1 = {'halo_mvir': mass, 'halo_zform': zform}
        self.toy_halo_table1 = Table(d1)

        halo_zform_percentile = (np.arange(Npts)+1) / float(Npts)
        d2 = {'halo_mvir': mass, 'halo_zform': zform, 'halo_zform_percentile': halo_zform_percentile}
        self.toy_halo_table2 = Table(d2)


     #    halo_designation = np.zeros(Npts, dtype=bool)
     #    halo_designation[Npts/2:] = True
    	# d = {'halo_mvir': mass, 'halo_zform': zform, 'halo_is_old': halo_designation}
    	# self.toy_halo_table = Table(d)
     #    is_old = self.toy_halo_table['halo_is_old'] == True
     #    self.young_toy_halos = self.toy_halo_table[np.invert(is_old)]
     #    self.old_toy_halos = self.toy_halo_table[is_old]

        fakesim = FakeSim()
        self.fake_halo_table = fakesim.halo_table

    # def test_setup(self):
    #     """
    #     """
    #     assert len(self.young_toy_halos) == len(self.old_toy_halos) == len(self.toy_halo_table)/2
    #     assert np.all(self.young_toy_halos['halo_zform'] <= 5)
    #     assert np.all(self.old_toy_halos['halo_zform'] >= 5)

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
        result = method(halo_table = halo_table)

        assert np.all(result >= model._lower_bound)
        assert np.all(result <= model._upper_bound)

        # young_result = result[young_mask]
        # assert np.all(young_result == 0)
        # old_result = result[np.invert(young_mask)]
        # assert np.all(old_result == 1)


    def test_binary_galprop_models(self):
        """
        """

        def execute_all_behavior_tests(model, halo_table, 
            correct_upper_pert_bound, correct_lower_pert_bound, 
            correct_split, **kwargs):

            self.constructor_tests(model)
            self.perturbation_bound_tests(model, halo_table = halo_table, 
                correct_upper_pert_bound = correct_upper_pert_bound, 
                correct_lower_pert_bound = correct_lower_pert_bound, **kwargs)
            self.assembias_strength_tests(model, halo_table = halo_table, **kwargs)
            self.splitting_func_tests(model, halo_table = halo_table, 
                correct_split=correct_split, **kwargs)
            self.decorated_method_tests(model, halo_table = halo_table, **kwargs)


        baseline_model = BinaryGalpropInterpolModel
        galprop_abcissa = [12]
        galprop_ordinates = [0.5]
        galprop_key='quiescent'
        method_name_to_decorate='mean_'+galprop_key+'_fraction'
        lower_bound = 0
        upper_bound = 1
        def split_func(**kwargs):
            return np.zeros(custom_len(kwargs['halo_table'])) + 0.5
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

        model = HeavisideAssembiasComponent(**kwargs)
        correct_upper_pert_bound = 0.5
        correct_lower_pert_bound = -0.5
        correct_split = 0.5

        execute_all_behavior_tests(model, self.toy_halo_table1, 
            correct_upper_pert_bound, correct_lower_pert_bound, 
            correct_split, **kwargs)
        execute_all_behavior_tests(model, self.toy_halo_table2, 
            correct_upper_pert_bound, correct_lower_pert_bound, 
            correct_split, **kwargs)
        is_old = self.toy_halo_table2['halo_zform_percentile'] > correct_split
        table3 = self.toy_halo_table1
        table3['halo_is_old'] = is_old
        kwargs['halo_type_tuple'] = ('halo_is_old', True, False)
        model = HeavisideAssembiasComponent(**kwargs)
        execute_all_behavior_tests(model, table3, 
            correct_upper_pert_bound, correct_lower_pert_bound, 
            correct_split, **kwargs)

        
        
        

        


        # kwargs['assembias_strength'] = 0.5
        # model = HeavisideAssembiasComponent(**kwargs)
        # self.constructor_tests(model)
        # self.perturbation_bound_tests(model, 
        #     correct_upper_pert_bound = 0.5, correct_lower_pert_bound = -0.5, **kwargs)
        # self.assembias_strength_tests(model, **kwargs)
        # self.splitting_func_tests(model, correct_split=0.5, **kwargs)






        # baseline_model = Zheng07Cens
        # method_name_to_decorate='mean_occupation'

        # model = HeavisideAssembiasComponent(baseline_model=baseline_model, 
        #   method_name_to_decorate=method_name_to_decorate, 
        #   lower_bound = 0, upper_bound = 1, 
        #   )














