#!/usr/bin/env python
import numpy as np

from astropy.tests.helper import pytest
from unittest import TestCase

from ...custom_exceptions import HalotoolsError

class TestHodModelFactoryTutorial(TestCase):
    """
    """

    @pytest.mark.slow
    def test_hod_modeling_tutorial1(self):

        from ...empirical_models import HodModelFactory

        from ...empirical_models import TrivialPhaseSpace, Zheng07Cens

        cens_occ_model =  Zheng07Cens(threshold = -20.5)
        cens_prof_model = TrivialPhaseSpace()

        from ...empirical_models import NFWPhaseSpace, Zheng07Sats
        sats_occ_model =  Zheng07Sats(threshold = -20.5)
        sats_prof_model = NFWPhaseSpace()

        model_instance = HodModelFactory(
            centrals_occupation = cens_occ_model, 
            centrals_profile = cens_prof_model, 
            satellites_occupation = sats_occ_model, 
            satellites_profile = sats_prof_model)

        # The model_instance is a composite model 
        # All composite models can directly populate N-body simulations 
        # with mock galaxy catalogs using the populate_mock method:

        model_instance.populate_mock(simname = 'fake')

        # Setting simname to 'fake' populates a mock into a fake halo catalog 
        # that is generated on-the-fly, but you can use the populate_mock 
        # method with any Halotools-formatted catalog 





