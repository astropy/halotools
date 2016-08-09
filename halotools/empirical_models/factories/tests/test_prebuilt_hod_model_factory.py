"""
"""
from __future__ import (absolute_import, division, print_function)

from unittest import TestCase
from astropy.tests.helper import pytest

import numpy as np

from ...factories import PrebuiltHodModelFactory


from ....sim_manager import FakeSim

__all__ = ['TestPrebuiltHodModelFactory']


class TestPrebuiltHodModelFactory(TestCase):
    """ Class providing tests of the `~halotools.empirical_models.PrebuiltHodModelFactory`.
    """
    @pytest.mark.slow
    def test_fake_mock_population(self):
        halocat = FakeSim()
        for modelname in PrebuiltHodModelFactory.prebuilt_model_nickname_list:
            model = PrebuiltHodModelFactory(modelname)
            model.populate_mock(halocat)
        model.populate_mock(halocat)

    @pytest.mark.slow
    def test_fake_mock_observations1(self):
        for modelname in PrebuiltHodModelFactory.prebuilt_model_nickname_list:
            model = PrebuiltHodModelFactory(modelname)
            result = model.compute_average_galaxy_clustering(num_iterations=1, simname='fake')

        result = model.compute_average_galaxy_clustering(
            num_iterations=1, simname='fake', summary_statistic='mean',
            gal_type='centrals', include_crosscorr=True, rbins=np.array((0.1, 0.2, 0.3)),
            redshift=0, halo_finder='rockstar')
