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

        cens_occ_model =  Zheng07Cens()
        cens_prof_model = TrivialPhaseSpace()

        from ...empirical_models import NFWPhaseSpace, Zheng07Sats
        sats_occ_model =  Zheng07Sats()
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

    @pytest.mark.slow
    def test_hod_modeling_tutorial2(self):

        from ...empirical_models import HodModelFactory

        from ...empirical_models import TrivialPhaseSpace, Leauthaud11Cens
        another_cens_occ_model =  Leauthaud11Cens()
        another_cens_prof_model = TrivialPhaseSpace()

        from ...empirical_models import NFWPhaseSpace, Leauthaud11Sats
        another_sats_occ_model =  Leauthaud11Sats()
        another_sats_prof_model = NFWPhaseSpace()

        from ...empirical_models import HaloMassInterpolQuenching
        sat_quenching = HaloMassInterpolQuenching('halo_mvir', [1e12, 1e13, 1e14, 1e15], [0.35, 0.5, 0.6, 0.9], gal_type = 'satellites')
        cen_quenching = HaloMassInterpolQuenching('halo_mvir', [1e12, 1e15], [0.25, 0.95], gal_type = 'centrals')

        model_instance = HodModelFactory(
            centrals_occupation = another_cens_occ_model, 
            centrals_profile = another_cens_prof_model, 
            satellites_occupation = another_sats_occ_model, 
            satellites_profile = another_sats_prof_model, 
            centrals_quenching = cen_quenching, 
            satellites_quenching = sat_quenching
            )

        assert hasattr(model_instance, 'mean_quiescent_fraction_centrals')
        assert hasattr(model_instance, 'mean_quiescent_fraction_satellites')

        assert 'centrals_quiescent_ordinates_param1' in model_instance.param_dict.keys()
        assert 'centrals_quiescent_ordinates_param2' in model_instance.param_dict.keys()
        assert 'satellites_quiescent_ordinates_param1' in model_instance.param_dict.keys()
        assert 'satellites_quiescent_ordinates_param4' in model_instance.param_dict.keys()

        model_instance.populate_mock(simname = 'fake')

        assert 'quiescent' in model_instance.mock.galaxy_table.keys()
        assert set(model_instance.mock.galaxy_table['quiescent']) == {True, False}

        cenmask = model_instance.mock.galaxy_table['gal_type'] == 'centrals'
        cens = model_instance.mock.galaxy_table[cenmask]
        assert set(cens['quiescent']) == {True, False}

        satmask = model_instance.mock.galaxy_table['gal_type'] == 'satellites'
        sats = model_instance.mock.galaxy_table[satmask]
        assert set(sats['quiescent']) == {True, False}









