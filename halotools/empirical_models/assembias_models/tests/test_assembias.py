"""
"""
import numpy as np
from astropy.table import Table

from ...occupation_models import AssembiasZheng07Cens
from ...occupation_models import AssembiasZheng07Sats
from ...occupation_models import AssembiasLeauthaud11Cens
from ...occupation_models import AssembiasLeauthaud11Sats
from ...occupation_models import AssembiasTinker13Cens

from ....sim_manager import FakeSim

__all__ = ('test_preloaded_assembiased_occupation_models', )
__author__ = ('Andrew Hearin', )


Npts = int(1e4)
mass = np.zeros(Npts) + 1e12
zform = np.linspace(0, 10, Npts)

d1 = {'halo_mvir': mass, 'halo_zform': zform}
toy_halo_table1 = Table(d1)

halo_zform_percentile = (np.arange(Npts)+1) / float(Npts)
halo_zform_percentile = 1. - halo_zform_percentile[::-1]
d2 = {'halo_mvir': mass, 'halo_zform': zform, 'halo_zform_percentile': halo_zform_percentile}
toy_halo_table2 = Table(d2)

fakesim = FakeSim()
fake_halo_table = fakesim.halo_table

model_class_list = (AssembiasZheng07Cens, AssembiasZheng07Sats,
    AssembiasLeauthaud11Cens, AssembiasLeauthaud11Sats,
    AssembiasTinker13Cens)

prim_haloprop = np.logspace(10, 15, 6)


def init_test(model):

    assert hasattr(model, 'prim_haloprop_key')
    assert hasattr(model, 'sec_haloprop_key')
    assert hasattr(model, '_method_name_to_decorate')
    assert hasattr(model, 'gal_type')

    lower_bound_key = 'lower_bound_' + model._method_name_to_decorate + '_' + model.gal_type
    assert hasattr(model, lower_bound_key)


def assembias_sign_effect(model):

    decorated_method = getattr(model, model._method_name_to_decorate)
    decorated_result_type1 = decorated_method(
        prim_haloprop=prim_haloprop,
        sec_haloprop_percentile=1)
    decorated_result_type2 = decorated_method(
        prim_haloprop=prim_haloprop,
        sec_haloprop_percentile=0)

    assembias_sign = model.assembias_strength(prim_haloprop)
    positive_assembias_idx = assembias_sign > 0
    negative_assembias_idx = assembias_sign < 0
    diff = decorated_result_type1 - decorated_result_type2
    assert np.all(diff[positive_assembias_idx] >= 0)
    assert np.all(diff[negative_assembias_idx] <= 0)
    assert np.any(diff != 0)


def baseline_preservation_test(model):

    prim_haloprop = np.logspace(10, 15, 6)

    baseline_method = getattr(model, 'baseline_'+model._method_name_to_decorate)
    baseline_result = baseline_method(prim_haloprop=prim_haloprop)

    decorated_method = getattr(model, model._method_name_to_decorate)
    decorated_result_type1 = decorated_method(
        prim_haloprop=prim_haloprop,
        sec_haloprop_percentile=1)
    decorated_result_type2 = decorated_method(
        prim_haloprop=prim_haloprop,
        sec_haloprop_percentile=0)
    type1_frac = 1 - model.percentile_splitting_function(prim_haloprop)
    type2_frac = 1 - type1_frac

    derived_result = type1_frac*decorated_result_type1 + type2_frac*decorated_result_type2
    np.testing.assert_allclose(baseline_result, derived_result, rtol=1e-3)


def model_variation_generator(model_class):
    yield model_class()
    yield model_class(split=0.75)
    yield model_class(split=0.25, assembias_strength=-0.5)
    yield model_class(split_abscissa=[1e10, 1e15],
        split=[-0.25, 1.75],
        assembias_strength_abscissa=[1e10, 1e13, 1e15],
        assembias_strength=[-1.25, 0.25, -5.75])


def test_preloaded_assembiased_occupation_models():

    for model_class in model_class_list:
        for model in model_variation_generator(model_class):
            init_test(model)
            assembias_sign_effect(model)
            baseline_preservation_test(model)
