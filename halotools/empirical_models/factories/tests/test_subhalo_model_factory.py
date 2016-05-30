#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

from unittest import TestCase
from astropy.tests.helper import pytest

import numpy as np
from copy import copy

from ...smhm_models import Behroozi10SmHm, Moster13SmHm
from ...component_model_templates import BinaryGalpropInterpolModel

from ...factories import SubhaloModelFactory, PrebuiltSubhaloModelFactory
from ...composite_models.smhm_models.behroozi10 import behroozi10_model_dictionary


from ....sim_manager import FakeSim
from ....custom_exceptions import HalotoolsError

__all__ = ['TestSubhaloModelFactory']


class TestSubhaloModelFactory(TestCase):
    """ Class providing tests of the `~halotools.empirical_models.SubhaloModelFactory`.
    """

    def setUp(self):
        """ Pre-load various arrays into memory for use by all tests.
        """
        pass

    def test_behroozi10_dictionary(self):
        """
        """
        model_dictionary = behroozi10_model_dictionary()
        model = SubhaloModelFactory(**model_dictionary)

    def test_absent_model_feature_calling_sequence(self):
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

    def test_baseline_model_instance1(self):
        """
        """
        behroozi = Behroozi10SmHm(redshift=0)

        # The following instantiation methods should give the same results
        model1 = SubhaloModelFactory(stellar_mass=behroozi)
        model2 = SubhaloModelFactory(baseline_model_instance=model1)
        assert model1._model_feature_calling_sequence == model2._model_feature_calling_sequence

    def test_baseline_model_instance2(self):
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

    def test_baseline_model_instance3(self):
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

    def test_baseline_model_instance4(self):
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

    def test_baseline_model_instance_behavior1(self):
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

    def test_baseline_model_instance_behavior2(self):
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

    def test_model_feature_naming_conventions1(self):
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

    def test_empty_arguments(self):
        with pytest.raises(HalotoolsError) as err:
            model = SubhaloModelFactory()
        substr = "You did not pass any model features to the factory"
        assert substr in err.value.args[0]

    def test_unavailable_haloprop(self):
        halocat = FakeSim()
        m = PrebuiltSubhaloModelFactory('behroozi10')
        m._haloprop_list.append("Jose Canseco")
        with pytest.raises(HalotoolsError) as err:
            m.populate_mock(halocat)
        substr = "this column is not available in the catalog you attempted to populate"
        assert substr in err.value.args[0]
        assert "``Jose Canseco``" in err.value.args[0]

    def tearDown(self):
        pass
