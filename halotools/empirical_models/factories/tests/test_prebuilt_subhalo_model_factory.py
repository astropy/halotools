""" Module providing unit-testing for
the `~halotools.empirical_models.PrebuiltSubhaloModelFactory` class
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import pytest

from ...factories import PrebuiltSubhaloModelFactory, SubhaloModelFactory

from ....sim_manager import FakeSim
from ....custom_exceptions import HalotoolsError

__all__ = ("test_behroozi_composite",)


def test_behroozi_composite():
    """Require that the `~halotools.empirical_models.behroozi10_model_dictionary`
    model dictionary builds without raising an exception.
    """
    model = PrebuiltSubhaloModelFactory("behroozi10")
    alt_model = SubhaloModelFactory(**model.model_dictionary)


def test_smhm_binary_sfr_composite():
    """Require that the `~halotools.empirical_models.smhm_binary_sfr_model_dictionary`
    model dictionary builds without raising an exception.
    """
    model = PrebuiltSubhaloModelFactory("smhm_binary_sfr")
    alt_model = SubhaloModelFactory(**model.model_dictionary)


def test_fake_mock_population():
    halocat = FakeSim()
    for modelname in PrebuiltSubhaloModelFactory.prebuilt_model_nickname_list:
        model = PrebuiltSubhaloModelFactory(modelname)
        model.populate_mock(halocat)
        model.populate_mock(halocat)

        model2 = PrebuiltSubhaloModelFactory(modelname, redshift=2.0)
        with pytest.raises(HalotoolsError) as err:
            model2.populate_mock(halocat)
        substr = "Inconsistency between the model redshift"
        assert substr in err.value.args[0]
        halocat = FakeSim(redshift=2.0)
        model2.populate_mock(halocat)


def test_fake_mock_observations1():
    for modelname in PrebuiltSubhaloModelFactory.prebuilt_model_nickname_list:
        model = PrebuiltSubhaloModelFactory(modelname)
        model.compute_average_galaxy_clustering(num_iterations=1, simname="fake")
    model.compute_average_galaxy_clustering(num_iterations=2, simname="fake")


def test_fake_mock_observations2():
    modelname = "behroozi10"
    model = PrebuiltSubhaloModelFactory(modelname)

    model.compute_average_galaxy_matter_cross_clustering(
        num_iterations=1, simname="fake"
    )

    def mask_function(t):
        return t["halo_upid"] == -1

    result = model.compute_average_galaxy_matter_cross_clustering(
        num_iterations=1,
        simname="fake",
        redshift=0,
        halo_finder="rockstar",
        rbins=np.array((0.1, 0.2, 0.3)),
        mask_function=mask_function,
        include_complement=True,
        summary_statistic="mean",
    )
    assert np.shape(result) == (3, 2)
    xi = result[0]
    assert len(xi) == 2
