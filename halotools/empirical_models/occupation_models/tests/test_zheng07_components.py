#!/usr/bin/env python
import numpy as np
from astropy.table import Table
from copy import copy, deepcopy

from astropy.tests.helper import pytest
from unittest import TestCase
import warnings

from .. import occupation_model_template, zheng07_components

from ... import model_defaults

from ....custom_exceptions import HalotoolsError

__all__ = ('TestZheng07Cens', 'TestZheng07Sats')


class TestZheng07Cens(TestCase):
    """ Class providing testing of the `~halotools.empirical_models.Zheng07Cens` model.

    The following list provides a brief summary of the tests performed:

        * The basic metadata of the model is correct, e.g., ``self._upper_occupation_bound = 1``

        * The `mean_occupation` function is bounded by zero and unity for the full range of reasonable input masses, :math:`0 <= \\langle N_{\mathrm{cen}}(M) \\rangle <=1` for :math:`\\log_{10}M/M_{\odot} \\in [10, 16]`

        * The `mean_occupation` function increases monotonically for the full range of reasonable input masses, :math:`\\langle N_{\mathrm{cen}}(M_{2}) \\rangle > \\langle N_{\mathrm{cen}}(M_{1}) \\rangle` for :math:`M_{2}>M_{1}`

        * The model correctly navigates having either array or halo catalog arguments, and returns the identical result regardless of how the inputs are bundled

        * The `mean_occupation` function scales properly as a function of variations in :math:`\\sigma_{\\mathrm{log}M}`, and also variations in :math:`\\log M_{\mathrm{min}}`, for both low and high halo masses.

    """

    def setUp(self):
        self.supported_thresholds = np.arange(-22, -17.5, 0.5)

        self.default_model = zheng07_components.Zheng07Cens()

        self.lowmass = (10.**self.default_model.param_dict['logMmin'])/1.1
        self.highmass = (10.**self.default_model.param_dict['logMmin'])*1.1

        self.model2 = zheng07_components.Zheng07Cens()
        self.model2.param_dict['logMmin'] += np.log10(2.)

        self.model3 = zheng07_components.Zheng07Cens()
        self.model3.param_dict['sigma_logM'] *= 2.0

    def test_setup(self):

        # Check that the alternate dictionaries were correctly implemented
        assert self.model2.param_dict['logMmin'] > self.default_model.param_dict['logMmin']
        assert self.model3.param_dict['sigma_logM'] > self.default_model.param_dict['sigma_logM']

    def test_default_model(self):
        self.enforce_required_attributes(self.default_model)
        self.enforce_mean_occupation_behavior(self.default_model)
        self.enforce_mc_occupation_behavior(self.default_model)
        self.enforce_correct_argument_inference_from_halo_catalog(self.default_model)

    def test_alternate_threshold_models(self):

        for threshold in self.supported_thresholds:
            thresh_model = zheng07_components.Zheng07Cens(threshold=threshold)
            self.enforce_required_attributes(thresh_model)
            self.enforce_mean_occupation_behavior(thresh_model)
            self.enforce_mc_occupation_behavior(thresh_model)

    def test_mean_ncen_scaling1(self):

        ### Now make sure the value of <Ncen> scales reasonably with the parameters

        defocc_lowmass = self.default_model.mean_occupation(prim_haloprop=self.lowmass)
        occ2_lowmass = self.model2.mean_occupation(prim_haloprop=self.lowmass)
        occ3_lowmass = self.model3.mean_occupation(prim_haloprop=self.lowmass)
        assert occ3_lowmass > defocc_lowmass
        assert defocc_lowmass > occ2_lowmass

    def test_mean_ncen_scaling2(self):

        defocc_highmass = self.default_model.mean_occupation(prim_haloprop=self.highmass)
        occ2_highmass = self.model2.mean_occupation(prim_haloprop=self.highmass)
        occ3_highmass = self.model3.mean_occupation(prim_haloprop=self.highmass)
        assert defocc_highmass > occ3_highmass
        assert occ3_highmass > occ2_highmass

    def test_param_dict_propagation1(self):

        ### Verify that directly changing model parameters
        # without a new instantiation also behaves properly
        defocc_lowmass = self.default_model.mean_occupation(prim_haloprop=self.lowmass)

        alt_model = zheng07_components.Zheng07Cens()
        alt_model.param_dict['sigma_logM'] *= 2.
        updated_defocc_lowmass = alt_model.mean_occupation(prim_haloprop=self.lowmass)

        assert updated_defocc_lowmass > defocc_lowmass

    def test_param_dict_propagation2(self):

        ### Verify that directly changing model parameters
        # without a new instantiation also behaves properly
        defocc_highmass = self.default_model.mean_occupation(prim_haloprop=self.highmass)

        alt_model = zheng07_components.Zheng07Cens()
        alt_model.param_dict['sigma_logM'] *= 2.
        updated_defocc_highmass = alt_model.mean_occupation(prim_haloprop=self.highmass)

        assert updated_defocc_highmass < defocc_highmass

    def test_param_dict_propagation3(self):

        ### Verify that directly changing model parameters
        # without a new instantiation also behaves properly
        defocc_lowmass = self.default_model.mean_occupation(prim_haloprop=self.lowmass)

        alt_model = zheng07_components.Zheng07Cens()
        alt_model.param_dict['sigma_logM'] *= 2.
        updated_defocc_lowmass = alt_model.mean_occupation(prim_haloprop=self.lowmass)

        assert updated_defocc_lowmass > defocc_lowmass

    def test_param_dict_propagation4(self):

        ### Verify that directly changing model parameters
        # without a new instantiation also behaves properly

        defocc_highmass = self.default_model.mean_occupation(prim_haloprop=self.highmass)

        alt_model = zheng07_components.Zheng07Cens()
        alt_model.param_dict['sigma_logM'] *= 2.
        updated_defocc_highmass = alt_model.mean_occupation(prim_haloprop=self.highmass)

        assert updated_defocc_highmass < defocc_highmass

    def enforce_required_attributes(self, model):
        assert isinstance(model, occupation_model_template.OccupationComponent)
        assert model.gal_type == 'centrals'

        assert hasattr(model, 'prim_haloprop_key')

        assert model._upper_occupation_bound == 1

    def enforce_mean_occupation_behavior(self, model):

        assert hasattr(model, 'mean_occupation')

        mvir_array = np.logspace(10, 16, 10)
        mean_occ = model.mean_occupation(prim_haloprop=mvir_array)

        # Check that the range is in [0,1]
        assert np.all(mean_occ<= 1)
        assert np.all(mean_occ >= 0)

        # The mean occupation should be monotonically increasing
        assert np.all(np.diff(mean_occ) >= 0)

    def enforce_mc_occupation_behavior(self, model):

        ### Check the Monte Carlo realization method
        assert hasattr(model, 'mc_occupation')

        # First check that the mean occuation is ~0.5 when model is evaulated at Mmin
        mvir_midpoint = 10.**model.param_dict['logMmin']
        Npts = int(1e3)
        masses = np.ones(Npts)*mvir_midpoint
        mc_occ = model.mc_occupation(prim_haloprop=masses, seed=43)
        assert set(mc_occ).issubset([0, 1])
        expected_result = 0.48599999
        np.testing.assert_allclose(mc_occ.mean(), expected_result, rtol=1e-5, atol=1.e-5)

        # Now check that the model is ~ 1.0 when evaluated for a cluster
        masses = np.ones(Npts)*5.e15
        mc_occ = model.mc_occupation(prim_haloprop=masses, seed=43)
        assert set(mc_occ).issubset([0, 1])
        expected_result = 1.0
        np.testing.assert_allclose(mc_occ.mean(), expected_result, rtol=1e-2, atol=1.e-2)

        # Now check that the model is ~ 0.0 when evaluated for a tiny halo
        masses = np.ones(Npts)*1.e10
        mc_occ = model.mc_occupation(prim_haloprop=masses, seed=43)
        assert set(mc_occ).issubset([0, 1])
        expected_result = 0.0
        np.testing.assert_allclose(mc_occ.mean(), expected_result, rtol=1e-2, atol=1.e-2)

    def enforce_correct_argument_inference_from_halo_catalog(self, model):
        mvir_array = np.logspace(10, 16, 10)
        key = model.prim_haloprop_key
        mvir_dict = {key: mvir_array}
        halo_catalog = Table(mvir_dict)
        # First test mean occupations
        meanocc_from_array = model.mean_occupation(prim_haloprop=mvir_array)
        meanocc_from_halos = model.mean_occupation(table=halo_catalog)
        assert np.all(meanocc_from_array == meanocc_from_halos)
        # Now test Monte Carlo occupations
        mcocc_from_array = model.mc_occupation(prim_haloprop=mvir_array, seed=43)
        mcocc_from_halos = model.mc_occupation(table=halo_catalog, seed=43)
        assert np.all(mcocc_from_array == mcocc_from_halos)

    def test_raises_correct_exception(self):
        with pytest.raises(HalotoolsError) as err:
            _ = self.default_model.mean_occupation(x=4)
        substr = "You must pass either a ``table`` or ``prim_haloprop`` argument"
        assert substr in err.value.args[0]

    def test_get_published_parameters1(self):
        d1 = self.default_model.get_published_parameters(self.default_model.threshold)

    def test_get_published_parameters2(self):
        with pytest.raises(KeyError) as err:
            d2 = self.default_model.get_published_parameters(self.default_model.threshold,
                publication='Parejko13')
        substr = "For Zheng07Cens, only supported best-fit models are currently Zheng et al. 2007"
        assert substr == err.value.args[0]

    def test_get_published_parameters3(self):
        with warnings.catch_warnings(record=True) as w:

            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")

            d1 = self.default_model.get_published_parameters(-11.3)

            assert "does not match any of the Table 1 values" in str(w[-1].message)

            d2 = self.default_model.get_published_parameters(self.default_model.threshold)

            assert d1 == d2


class TestZheng07Sats(TestCase):
    """ Class providing testing of the `~halotools.empirical_models.Zheng07Sats` model.

        The following list provides a brief summary of the tests performed:

        * The basic metadata of the model is correct, e.g., ``self._upper_occupation_bound = 1``

        * The `mean_occupation` function is bounded by zero and unity for the full range of reasonable input masses, :math:`0 <= \\langle N_{\mathrm{cen}}(M) \\rangle <=1` for :math:`\\log_{10}M/M_{\odot} \\in [10, 16]`

        * The `mean_occupation` function increases monotonically for the full range of reasonable input masses, :math:`\\langle N_{\mathrm{cen}}(M_{2}) \\rangle > \\langle N_{\mathrm{cen}}(M_{1}) \\rangle` for :math:`M_{2}>M_{1}`

        * The model correctly navigates having either array or halo catalog arguments, and returns the identical result regardless of how the inputs are bundled

        * The `mean_occupation` function scales properly as a function of variations in :math:`\\sigma_{\\mathrm{log}M}`, and also variations in :math:`\\log M_{\mathrm{min}}`, for both low and high halo masses.
    """

    def setUp(self):

        self.default_model = zheng07_components.Zheng07Sats()

        self.model2 = zheng07_components.Zheng07Sats()
        self.model2.param_dict['alpha'] *= 1.25

        self.model3 = zheng07_components.Zheng07Sats()
        self.model3.param_dict['logM0'] += np.log10(2)

        self.model4 = zheng07_components.Zheng07Sats()
        self.model4.param_dict['logM1'] += np.log10(2)

        self.supported_thresholds = np.arange(-22, -17.5, 0.5)

    def test_default_model(self):

        ### First test the model with all default settings

        self.enforce_required_attributes(self.default_model)
        self.enforce_mean_occupation_behavior(self.default_model)
        self.enforce_mc_occupation_behavior(self.default_model)

    def test_alternate_threshold_models(self):

        for threshold in self.supported_thresholds:
            thresh_model = zheng07_components.Zheng07Sats(threshold=threshold)
            self.enforce_required_attributes(thresh_model)
            self.enforce_mean_occupation_behavior(thresh_model)

    def enforce_required_attributes(self, model):
        assert isinstance(model, occupation_model_template.OccupationComponent)
        assert model.gal_type == 'satellites'

        assert hasattr(model, 'prim_haloprop_key')

        assert model._upper_occupation_bound == float("inf")

    def enforce_mean_occupation_behavior(self, model):

        assert hasattr(model, 'mean_occupation')

        mvir_array = np.logspace(10, 16, 10)
        mean_occ = model.mean_occupation(prim_haloprop=mvir_array)

        # Check non-negative
        assert np.all(mean_occ >= 0)
        # The mean occupation should be monotonically increasing
        assert np.all(np.diff(mean_occ) >= 0)

    def enforce_mc_occupation_behavior(self, model):

        ### Check the Monte Carlo realization method
        assert hasattr(model, 'mc_occupation')

        model.param_dict['alpha'] = 1
        model.param_dict['logM0'] = 11.25
        model.param_dict['logM1'] = model.param_dict['logM0'] + np.log10(20.)

        Npts = int(1e3)
        masses = np.ones(Npts)*10.**model.param_dict['logM1']
        mc_occ = model.mc_occupation(prim_haloprop=masses, seed=43)
        # We chose a specific seed that has been pre-tested,
        # so we should always get the same result
        expected_result = 1.0
        np.testing.assert_allclose(mc_occ.mean(), expected_result, rtol=1e-2, atol=1.e-2)

    def test_ncen_inheritance_behavior(self):
        satmodel_nocens = zheng07_components.Zheng07Sats()
        cenmodel = zheng07_components.Zheng07Cens()
        satmodel_cens = zheng07_components.Zheng07Sats(modulate_with_cenocc=True)

        Npts = 100
        masses = np.logspace(10, 15, Npts)
        mean_occ_satmodel_nocens = satmodel_nocens.mean_occupation(prim_haloprop=masses)
        mean_occ_satmodel_cens = satmodel_cens.mean_occupation(prim_haloprop=masses)
        assert np.all(mean_occ_satmodel_cens <= mean_occ_satmodel_nocens)

        diff = mean_occ_satmodel_cens - mean_occ_satmodel_nocens
        assert diff.sum() < 0

        mean_occ_cens = satmodel_cens.central_occupation_model.mean_occupation(prim_haloprop=masses)
        assert np.all(mean_occ_satmodel_cens == mean_occ_satmodel_nocens*mean_occ_cens)

    def test_alpha_scaling1_mean_occupation(self):

        logmass = self.model2.param_dict['logM1'] + np.log10(5)
        mass = 10.**logmass
        assert (self.model2.mean_occupation(prim_haloprop=mass) >
            self.default_model.mean_occupation(prim_haloprop=mass))

    def test_alpha_scaling2_mc_occupation(self):
        logmass = self.model2.param_dict['logM1'] + np.log10(5)
        mass = 10.**logmass
        Npts = 1000
        masses = np.ones(Npts)*mass

        assert (self.model2.mc_occupation(prim_haloprop=masses, seed=43).mean() >
            self.default_model.mc_occupation(prim_haloprop=masses, seed=43).mean())

    def test_alpha_propagation(self):
        logmass = self.model2.param_dict['logM1'] + np.log10(5)
        mass = 10.**logmass
        Npts = 1000
        masses = np.ones(Npts)*mass

        alt_default_model = deepcopy(self.default_model)

        alt_default_model.param_dict['alpha'] = self.model2.param_dict['alpha']

        assert (self.model2.mc_occupation(prim_haloprop=masses, seed=43).mean() ==
            alt_default_model.mc_occupation(prim_haloprop=masses, seed=43).mean())

    def test_logM0_scaling1_mean_occupation(self):

        # At very low mass, both models should have zero satellites
        lowmass = 1e10
        assert (self.model3.mean_occupation(prim_haloprop=lowmass) ==
            self.default_model.mean_occupation(prim_haloprop=lowmass))

    def test_logM0_scaling2_mean_occupation(self):

        # At intermediate masses, there should be fewer satellites for larger M0
        midmass = 1e12
        assert (self.model3.mean_occupation(prim_haloprop=midmass) <
            self.default_model.mean_occupation(prim_haloprop=midmass)
            )

    def test_logM0_scaling3_mean_occupation(self):

        # At high masses, the difference should be negligible
        highmass = 1e15
        np.testing.assert_allclose(
            self.model3.mean_occupation(prim_haloprop=highmass),
            self.default_model.mean_occupation(prim_haloprop=highmass),
            rtol=1e-3, atol=1.e-3)

    def test_logM1_scaling1_mean_occupation(self):

        # At very low mass, both models should have zero satellites
        lowmass = 1e10
        assert (self.model4.mean_occupation(prim_haloprop=lowmass) ==
            self.default_model.mean_occupation(prim_haloprop=lowmass))

    def test_logM1_scaling2_mean_occupation(self):

        # At intermediate masses, there should be fewer satellites for larger M1
        midmass = 1e12
        fracdiff_midmass = ((self.model4.mean_occupation(prim_haloprop=midmass) -
            self.default_model.mean_occupation(prim_haloprop=midmass)) /
            self.default_model.mean_occupation(prim_haloprop=midmass))
        assert fracdiff_midmass < 0

        highmass = 1e14
        fracdiff_highmass = ((self.model4.mean_occupation(prim_haloprop=highmass) -
            self.default_model.mean_occupation(prim_haloprop=highmass)) /
            self.default_model.mean_occupation(prim_haloprop=highmass))
        assert fracdiff_highmass < 0

        # The fractional change due to alterations of logM1 should be identical at all mass
        assert fracdiff_highmass == fracdiff_midmass

    def test_raises_correct_exception(self):
        with pytest.raises(HalotoolsError) as err:
            _ = self.default_model.mean_occupation(x=4)
        substr = "You must pass either a ``table`` or ``prim_haloprop`` argument"
        assert substr in err.value.args[0]

    def test_get_published_parameters1(self):
        d1 = self.default_model.get_published_parameters(self.default_model.threshold)

    def test_get_published_parameters2(self):
        with pytest.raises(KeyError) as err:
            d2 = self.default_model.get_published_parameters(self.default_model.threshold,
                publication='Parejko13')
        substr = "For Zheng07Sats, only supported best-fit models are currently Zheng et al. 2007"
        assert substr == err.value.args[0]

    def test_get_published_parameters3(self):
        with warnings.catch_warnings(record=True) as w:

            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")

            d1 = self.default_model.get_published_parameters(-11.3)

            assert "does not match any of the Table 1 values" in str(w[-1].message)

            d2 = self.default_model.get_published_parameters(self.default_model.threshold)

            assert d1 == d2
