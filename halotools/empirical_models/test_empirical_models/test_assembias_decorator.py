#!/usr/bin/env python

from unittest import TestCase
import pytest
from copy import copy

import numpy as np 
from astropy.table import Table

from ..assembias_decorator import HeavisideAssembiasComponent
from ...sim_manager import FakeSim
from ..hod_components import Zheng07Cens, Leauthaud11Cens
from .. import model_defaults
from ...utils.table_utils import SampleSelector


class TestAssembiasDecorator(TestCase):

    def setup_class(self):
        Npts = 1e4
        mass = np.zeros(Npts) + 1e12
        zform = np.linspace(0, 10, Npts)
        d = {'halo_mvir': mass, 'halo_zform': zform}
        self.halo_table = Table(d)

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

        output_split = model.percentile_splitting_function(halo_table=self.halo_table)
        assert np.all(output_split == 0.5)

        model._split_ordinates = [0.75]
        output_split = model.percentile_splitting_function(halo_table=self.halo_table)
        assert np.all(output_split == 0.75)

        output_strength = model.assembias_strength(halo_table=self.halo_table)
        assert np.all(output_strength == 0.5)


        model2 = HeavisideAssembiasComponent(baseline_model=baseline_model, 
        method_name_to_decorate=method_name_to_decorate, 
        assembias_strength_abcissa = [1e10, 1e14], 
        assembias_strength_ordinates = [1, 0], 
        lower_bound = 0, upper_bound = 1, split=0.45
        )
        output_strength = model.assembias_strength(halo_table=self.halo_table)
        assert np.all(output_strength == 0.5)

        l = [key for key in model2.param_dict if 'assembias_param' in key]
        assert len(l) == 2

        d = {key: model2.param_dict[key] for key in model2.param_dict if 'assembias_param' in key}
        assert len(d) == 2
        assert model2._method_name_to_decorate+'_assembias_param1' in d
        assert model2._method_name_to_decorate+'_assembias_param2' in d
        assert d[model2._method_name_to_decorate+'_assembias_param1'] == 1
        assert d[model2._method_name_to_decorate+'_assembias_param2'] == 0

        self.halo_table[model2.prim_haloprop_key] = 1e10
        output_strength = model2.assembias_strength(halo_table=self.halo_table)
        assert np.all(output_strength == 1)

        self.halo_table[model2.prim_haloprop_key] = 1e14
        output_strength = model2.assembias_strength(halo_table=self.halo_table)
        assert np.all(output_strength == 0)

        output_split = model2.percentile_splitting_function(halo_table=self.halo_table)
        assert np.all(output_split == 0.45)














