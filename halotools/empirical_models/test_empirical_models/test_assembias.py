#!/usr/bin/env python

from unittest import TestCase
import pytest
from copy import copy

import numpy as np 
from astropy.table import Table

from ..hod_components import AssembiasZheng07Cens, AssembiasZheng07Sats, AssembiasLeauthaud11Cens, AssembiasLeauthaud11Sats

from .. import model_defaults
from ..hod_components import Zheng07Cens, Leauthaud11Cens
from ..sfr_components import BinaryGalpropInterpolModel
from ...sim_manager import FakeSim
from ...utils.table_utils import SampleSelector, compute_conditional_percentiles
from ...utils.array_utils import custom_len

class TestAssembias(TestCase):
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

    def init_test(self, model):

        assert hasattr(model, 'prim_haloprop_key')
        assert hasattr(model, 'sec_haloprop_key')
        assert hasattr(model, '_method_name_to_decorate')
        assert hasattr(model, 'gal_type')

        lower_bound_key = 'lower_bound_' + model._method_name_to_decorate + '_' + model.gal_type
        assert hasattr(model, lower_bound_key)

    def baseline_recovery_test(self, model):

        baseline_method = getattr(model, 'baseline_'+model._method_name_to_decorate)
        baseline_result = baseline_method(halo_table = self.toy_halo_table2)

        method = getattr(model, model._method_name_to_decorate)
        result = method(halo_table = self.toy_halo_table2)

        mask = self.toy_halo_table2['halo_zform_percentile'] >= model._split_ordinates[0]
        oldmean = result[mask].mean()
        youngmean = result[np.invert(mask)].mean()
        baseline_mean = baseline_result.mean()
        assert oldmean != youngmean
        assert oldmean != baseline_mean
        assert youngmean != baseline_mean 

        param_key = model._get_assembias_param_dict_key(0)
        param = model.param_dict[param_key]
        if param > 0:
            assert oldmean > youngmean 
        elif param < 0: 
            assert oldmean < youngmean
        else:
            assert oldmean == youngmean 

        split = model.percentile_splitting_function(halo_table = self.toy_halo_table2)
        split = np.where(mask, split, 1-split)
        derived_result = split*oldmean
        derived_result[np.invert(mask)] = split[np.invert(mask)]*youngmean
        derived_mean = derived_result[mask].mean() + derived_result[np.invert(mask)].mean()
        baseline_mean = baseline_result.mean()
        np.testing.assert_allclose(baseline_mean, derived_mean, rtol=1e-3)

    def test_assembias_zheng07_cens(self):
        abz = AssembiasZheng07Cens(sec_haloprop_key = 'halo_zform')

        self.init_test(abz)
        self.baseline_recovery_test(abz)

        abz2 = AssembiasZheng07Cens(sec_haloprop_key = 'halo_zform', 
            split=0.75, assembias_strength = -0.25)
        self.init_test(abz2)
        self.baseline_recovery_test(abz2)

    def test_assembias_zheng07_sats(self):
        abz = AssembiasZheng07Sats(sec_haloprop_key = 'halo_zform')

        self.init_test(abz)
        self.baseline_recovery_test(abz)

        abz2 = AssembiasZheng07Sats(sec_haloprop_key = 'halo_zform', 
            split=0.25, assembias_strength = -0.7)
        self.init_test(abz2)
        self.baseline_recovery_test(abz2)

    def test_assembias_leauthaud11_cens(self):
        abl = AssembiasLeauthaud11Cens(sec_haloprop_key = 'halo_zform')

        self.init_test(abl)
        self.baseline_recovery_test(abl)

        abl2 = AssembiasLeauthaud11Cens(sec_haloprop_key = 'halo_zform', 
            split=0.25, assembias_strength = -0.7)
        self.init_test(abl2)
        self.baseline_recovery_test(abl2)

    def test_assembias_leauthaud11_sats(self):
        abl = AssembiasLeauthaud11Sats(sec_haloprop_key = 'halo_zform')

        self.init_test(abl)
        self.baseline_recovery_test(abl)

        abl2 = AssembiasLeauthaud11Sats(sec_haloprop_key = 'halo_zform', 
            split=0.25, assembias_strength = -0.7)
        self.init_test(abl2)
        self.baseline_recovery_test(abl2)










