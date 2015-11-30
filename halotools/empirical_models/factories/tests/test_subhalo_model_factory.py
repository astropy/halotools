#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

from unittest import TestCase
import pytest 

import numpy as np 
from copy import copy 

from ...smhm_models import Behroozi10SmHm, Moster13SmHm
from ...sfr_models import BinaryGalpropInterpolModel

from ...factories import SubhaloModelFactory
from ...composite_models.smhm_models.behroozi10 import behroozi10_model_dictionary


from ....sim_manager import FakeSim
from ....custom_exceptions import HalotoolsError

__all__ = ['TestSubhaloModelFactory']

class TestSubhaloModelFactory(TestCase):
    """ Class providing tests of the `~halotools.empirical_models.factories.SubhaloModelFactory`. 
    """

    def setup_class(self):
        """ Pre-load various arrays into memory for use by all tests. 
        """

    def test_exception_handling(self):
        """ Using `~halotools.empirical_models.composite_models.smhm_models.behroozi10_model_dictionary` 
        as a specific example, require that the `~halotools.empirical_models.SubhaloModelFactory` raises a HalotoolsError 
        if any of its component models do not have `_galprop_dtypes_to_allocate` and `_methods_to_inherit` attributes. 
        """

        model_dictionary = behroozi10_model_dictionary()

        model = SubhaloModelFactory(**model_dictionary)

        tmp = copy(model_dictionary['stellar_mass']._galprop_dtypes_to_allocate)
        del model_dictionary['stellar_mass']._galprop_dtypes_to_allocate
        with pytest.raises(HalotoolsError):
            model = SubhaloModelFactory(**model_dictionary)
        model_dictionary['stellar_mass']._galprop_dtypes_to_allocate = tmp

        tmp = copy(model_dictionary['stellar_mass']._methods_to_inherit)
        del model_dictionary['stellar_mass']._methods_to_inherit
        with pytest.raises(HalotoolsError):
            model = SubhaloModelFactory(**model_dictionary)

        behroozi = Behroozi10SmHm(redshift = 0)
        model2 = SubhaloModelFactory(stellar_mass = behroozi, 
            model_feature_calling_sequence = ['stellar_mass'])
        with pytest.raises(HalotoolsError):
            model3 = SubhaloModelFactory(stellar_mass = behroozi, 
                model_feature_calling_sequence = ['stellar_mass', 'quiescent'])

    def test_baseline_model_instance_feature(self):
        """ Use the ``baseline_model_instance`` feature to create permutations of composite models, 
        performing consistency checks in all cases. 

        * Identical models should result when passing in baseline_model_instance or the original model dictionary

        * The _model_feature_calling_sequence mechanism should work as it normally does when not using the baseline_model_instance feature 

        * Adding on additional, orthogonal behavior should not change the original behavior
        """

        behroozi = Behroozi10SmHm(redshift = 0)
        moster = Moster13SmHm(redshift = 0)
        quenching1 = BinaryGalpropInterpolModel(galprop_name = 'quiescent')
        quenching2 = BinaryGalpropInterpolModel(galprop_name = 'quiescent', 
            galprop_abcissa = [13], galprop_ordinates = [0.5])

        # The following instantiation methods should give the same results
        model1 = SubhaloModelFactory(stellar_mass = behroozi)
        model2 = SubhaloModelFactory(baseline_model_instance = model1)
        assert model1._model_feature_calling_sequence == model2._model_feature_calling_sequence

        # The _model_feature_calling_sequence should still have stellar_mass appear first as normal 
        # when using the baseline_model_instance feature
        model3 = SubhaloModelFactory(baseline_model_instance = model1, quenching = quenching1)
        assert model3._model_feature_calling_sequence == ['stellar_mass', 'quenching']

        # The input model_feature_calling_sequence should also propagate when using 
        # the baseline_model_instance feature
        model4 = SubhaloModelFactory(baseline_model_instance = model1, quenching = quenching1, 
            model_feature_calling_sequence = ['quenching', 'stellar_mass'])
        assert model4._model_feature_calling_sequence == ['quenching', 'stellar_mass']

        # We should raise an exception if we're missing a feature in the model_feature_calling_sequence
        with pytest.raises(HalotoolsError):
            model5 = SubhaloModelFactory(baseline_model_instance = model1, quenching = quenching1, 
                model_feature_calling_sequence = ['quenching'])

        # The following instantiation methods should give the same results for <M*>(Mvir)
        model1 = SubhaloModelFactory(stellar_mass = behroozi)
        model2 = SubhaloModelFactory(stellar_mass = moster)
        model3 = SubhaloModelFactory(baseline_model_instance = model1, stellar_mass = moster)
        mvir_array = np.logspace(10, 15, 100)
        result2 = model2.mean_stellar_mass(prim_haloprop = mvir_array)
        result3 = model3.mean_stellar_mass(prim_haloprop = mvir_array)
        assert np.all(result2 == result3)

        # Tacking on a quenching model should not change the stellar_mass behavior
        model4 = SubhaloModelFactory(baseline_model_instance = model3, quenching = quenching1)
        result4 = model4.mean_stellar_mass(prim_haloprop = mvir_array)
        assert np.all(result3 == result4)

















