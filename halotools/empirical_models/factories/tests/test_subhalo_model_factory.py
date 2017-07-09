"""
"""
from __future__ import (absolute_import, division, print_function)

import pytest

import numpy as np
from copy import deepcopy

from ...smhm_models import Behroozi10SmHm, Moster13SmHm
from ...component_model_templates import BinaryGalpropInterpolModel

from ...factories import SubhaloModelFactory, PrebuiltSubhaloModelFactory
from ...composite_models.smhm_models.behroozi10 import behroozi10_model_dictionary
from ...composite_models.sfr_models.smhm_binary_sfr import smhm_binary_sfr_model_dictionary


from ....sim_manager import FakeSim
from ....custom_exceptions import HalotoolsError

__all__ = ('test_behroozi10_dictionary', )

fixed_seed = 43


def test_behroozi10_dictionary():
    """
    """
    model_dictionary = behroozi10_model_dictionary()
    model = SubhaloModelFactory(**model_dictionary)


def test_absent_model_feature_calling_sequence():
    """ Verify that an exception is raised if the
    ``model_feature_calling_sequence`` keyword argument contains
    an entry that was not a keyword argument passed to the constructor.
    """
    model_dictionary = behroozi10_model_dictionary()
    model = SubhaloModelFactory(**model_dictionary)

    behroozi = Behroozi10SmHm(redshift=0)
    model2 = SubhaloModelFactory(stellar_mass=behroozi,
        model_feature_calling_sequence=['stellar_mass'])
    with pytest.raises(HalotoolsError):
        model3 = SubhaloModelFactory(stellar_mass=behroozi,
            model_feature_calling_sequence=['stellar_mass', 'quiescent'])


def test_baseline_model_instance1():
    """
    """
    behroozi = Behroozi10SmHm(redshift=0)

    # The following instantiation methods should give the same results
    model1 = SubhaloModelFactory(stellar_mass=behroozi)
    model2 = SubhaloModelFactory(baseline_model_instance=model1)
    assert model1._model_feature_calling_sequence == model2._model_feature_calling_sequence


def test_baseline_model_instance2():
    """
    """
    behroozi = Behroozi10SmHm(redshift=0)

    # The following instantiation methods should give the same results
    model1 = SubhaloModelFactory(stellar_mass=behroozi)
    quenching1 = BinaryGalpropInterpolModel(galprop_name='quiescent',
        galprop_abscissa=[12, 15], galprop_ordinates=[0.25, 0.75])

    # The _model_feature_calling_sequence should still have stellar_mass appear first as normal
    # when using the baseline_model_instance feature
    model3 = SubhaloModelFactory(baseline_model_instance=model1, quenching=quenching1)
    assert model3._model_feature_calling_sequence == ['stellar_mass', 'quenching']


def test_baseline_model_instance3():
    """
    """
    behroozi = Behroozi10SmHm(redshift=0)
    model1 = SubhaloModelFactory(stellar_mass=behroozi)
    quenching1 = BinaryGalpropInterpolModel(galprop_name='quiescent',
        galprop_abscissa=[12, 15], galprop_ordinates=[0.25, 0.75])

    # The input model_feature_calling_sequence should also propagate when using
    # the baseline_model_instance feature
    model4 = SubhaloModelFactory(baseline_model_instance=model1, quenching=quenching1,
        model_feature_calling_sequence=['quenching', 'stellar_mass'])
    assert model4._model_feature_calling_sequence == ['quenching', 'stellar_mass']


def test_baseline_model_instance4():
    """
    """
    behroozi = Behroozi10SmHm(redshift=0)
    model1 = SubhaloModelFactory(stellar_mass=behroozi)
    quenching1 = BinaryGalpropInterpolModel(galprop_name='quiescent',
        galprop_abscissa=[12, 15], galprop_ordinates=[0.25, 0.75])

    # We should raise an exception if we're missing a feature in the model_feature_calling_sequence
    with pytest.raises(HalotoolsError):
        model5 = SubhaloModelFactory(baseline_model_instance=model1, quenching=quenching1,
            model_feature_calling_sequence=['quenching'])


def test_baseline_model_instance_behavior1():
    """
    """
    behroozi = Behroozi10SmHm(redshift=0)
    moster = Moster13SmHm(redshift=0)

    # The following instantiation methods should give the same results for <M*>(Mvir)
    model1 = SubhaloModelFactory(stellar_mass=behroozi)
    model2 = SubhaloModelFactory(stellar_mass=moster)
    model3 = SubhaloModelFactory(baseline_model_instance=model1, stellar_mass=moster)
    mvir_array = np.logspace(10, 15, 100)
    result2 = model2.mean_stellar_mass(prim_haloprop=mvir_array)
    result3 = model3.mean_stellar_mass(prim_haloprop=mvir_array)
    assert np.all(result2 == result3)


def test_baseline_model_instance_behavior2():
    """
    """
    behroozi = Behroozi10SmHm(redshift=0)
    model1 = SubhaloModelFactory(stellar_mass=behroozi)
    quenching = BinaryGalpropInterpolModel(galprop_name='quiescent',
        galprop_abscissa=[12, 15], galprop_ordinates=[0.25, 0.75])

    mvir_array = np.logspace(10, 15, 100)

    # Tacking on a quenching model should not change the stellar_mass behavior
    model2 = SubhaloModelFactory(baseline_model_instance=model1, quenching=quenching)
    result1 = model1.mean_stellar_mass(prim_haloprop=mvir_array)
    result2 = model2.mean_stellar_mass(prim_haloprop=mvir_array)
    assert np.all(result1 == result1)


def test_model_feature_naming_conventions1():
    """
    """
    behroozi = Behroozi10SmHm(redshift=0)
    model1 = SubhaloModelFactory(mstar=behroozi)
    assert hasattr(model1, 'mc_stellar_mass')
    assert hasattr(model1, 'mean_stellar_mass')

    assert not hasattr(model1, 'mock')
    halocat = FakeSim()
    model1.populate_mock(halocat)
    assert hasattr(model1, 'mock')
    assert 'stellar_mass' in list(model1.mock.galaxy_table.keys())


def test_empty_arguments():
    with pytest.raises(HalotoolsError) as err:
        model = SubhaloModelFactory()
    substr = "You did not pass any model features to the factory"
    assert substr in err.value.args[0]


def test_unavailable_haloprop():
    halocat = FakeSim()
    m = PrebuiltSubhaloModelFactory('behroozi10')
    m._haloprop_list.append("Jose Canseco")
    with pytest.raises(HalotoolsError) as err:
        m.populate_mock(halocat)
    substr = "this column is not available in the catalog you attempted to populate"
    assert substr in err.value.args[0]
    assert "``Jose Canseco``" in err.value.args[0]


def test_deterministic_mock_making():
    """ Test ensuring that mock population is purely deterministic
    when using the seed keyword.

    This is a regression test associated with https://github.com/astropy/halotools/issues/551.
    """
    model = PrebuiltSubhaloModelFactory('behroozi10', threshold=-21)
    halocat = FakeSim(seed=fixed_seed)
    model.populate_mock(halocat, seed=fixed_seed)
    mask = model.mock.galaxy_table['stellar_mass'] > 1e10
    h1 = deepcopy(model.mock.galaxy_table[mask])
    del model
    del halocat

    model = PrebuiltSubhaloModelFactory('behroozi10', threshold=-21)
    halocat = FakeSim(seed=fixed_seed)
    model.populate_mock(halocat, seed=fixed_seed)
    mask = model.mock.galaxy_table['stellar_mass'] > 1e10
    h2 = deepcopy(model.mock.galaxy_table[mask])
    del model
    del halocat

    model = PrebuiltSubhaloModelFactory('behroozi10', threshold=-21)
    halocat = FakeSim(seed=fixed_seed)
    model.populate_mock(halocat, seed=fixed_seed+1)
    mask = model.mock.galaxy_table['stellar_mass'] > 1e10
    h3 = deepcopy(model.mock.galaxy_table[mask])
    del model
    del halocat

    assert len(h1) == len(h2)
    assert len(h1) != len(h3)

    for key in h1.keys():
        try:
            assert np.allclose(h1[key], h2[key], rtol=0.001)
        except TypeError:
            pass


def test_raises_appropriate_exception1():
    """
    """
    model_dict_no_redshift = smhm_binary_sfr_model_dictionary()
    model = SubhaloModelFactory(**model_dict_no_redshift)

    for i, component_model in enumerate(model.model_dictionary.values()):
        component_model.redshift = i

    with pytest.raises(HalotoolsError) as err:
        model.set_model_redshift()
    substr = "Inconsistency between the redshifts of the component models"
    assert substr in err.value.args[0]


def test_restore_init_param_dict():
    """
    """
    model_dict_no_redshift = smhm_binary_sfr_model_dictionary()
    model = SubhaloModelFactory(**model_dict_no_redshift)
    orig_param_dict = deepcopy(model.param_dict)

    for key in model.param_dict.keys():
        model.param_dict[key] += 1

    assert orig_param_dict != model.param_dict

    model.restore_init_param_dict()
    assert orig_param_dict == model.param_dict


def test_test_dictionary_consistency():
    """
    """
    model_dict_no_redshift = behroozi10_model_dictionary()
    model = SubhaloModelFactory(**model_dict_no_redshift)
    for component_model in model.model_dictionary.values():
        del component_model._methods_to_inherit

    with pytest.raises(HalotoolsError) as err:
        model._test_dictionary_consistency()
    substr = "At a minimum, all component models must have this attribute"
    assert substr in err.value.args[0]
