""" Module providing unit-testing for
the `~halotools.empirical_models.PrebuiltSubhaloModelFactory` class
"""
from __future__ import absolute_import, division, print_function

from unittest import TestCase
from astropy.tests.helper import pytest

from ...factories import PrebuiltSubhaloModelFactory, SubhaloModelFactory

from ....sim_manager import FakeSim
from ....custom_exceptions import HalotoolsError

__all__ = ['TestPrebuiltSubhaloModelFactory']


class TestPrebuiltSubhaloModelFactory(TestCase):
    """ Class providing tests of the `~halotools.empirical_models.PrebuiltSubhaloModelFactory`.
    """

    def test_behroozi_composite(self):
        """ Require that the `~halotools.empirical_models.behroozi10_model_dictionary`
        model dictionary builds without raising an exception.
        """
        model = PrebuiltSubhaloModelFactory('behroozi10')
        alt_model = SubhaloModelFactory(**model.model_dictionary)

    def test_smhm_binary_sfr_composite(self):
        """ Require that the `~halotools.empirical_models.smhm_binary_sfr_model_dictionary`
        model dictionary builds without raising an exception.
        """
        model = PrebuiltSubhaloModelFactory('smhm_binary_sfr')
        alt_model = SubhaloModelFactory(**model.model_dictionary)

    @pytest.mark.slow
    def test_fake_mock_population(self):
        halocat = FakeSim()
        for modelname in PrebuiltSubhaloModelFactory.prebuilt_model_nickname_list:
            model = PrebuiltSubhaloModelFactory(modelname)
            model.populate_mock(halocat)
            model.populate_mock(halocat)

            model2 = PrebuiltSubhaloModelFactory(modelname, redshift=2.)
            with pytest.raises(HalotoolsError) as err:
                model2.populate_mock(halocat)
            substr = "Inconsistency between the model redshift"
            assert substr in err.value.args[0]
            halocat = FakeSim(redshift=2.)
            model2.populate_mock(halocat)

    @pytest.mark.slow
    def test_fake_mock_observations1(self):
        for modelname in PrebuiltSubhaloModelFactory.prebuilt_model_nickname_list:
            model = PrebuiltSubhaloModelFactory(modelname)
            model.compute_average_galaxy_clustering(num_iterations=1, simname='fake')
        model.compute_average_galaxy_clustering(num_iterations=2, simname='fake')

    @pytest.mark.slow
    def test_fake_mock_observations2(self):
        modelname = 'behroozi10'
        model = PrebuiltSubhaloModelFactory(modelname)
        model.compute_average_galaxy_matter_cross_clustering(
            num_iterations=1, simname='fake')
