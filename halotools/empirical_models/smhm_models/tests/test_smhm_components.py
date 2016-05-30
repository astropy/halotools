#!/usr/bin/env python

import numpy as np
from astropy.table import Table
from unittest import TestCase

from ...component_model_templates import LogNormalScatterModel
from ...smhm_models import Moster13SmHm, Behroozi10SmHm

from ... import model_defaults

__all__ = ['test_Moster13SmHm_initialization', 'test_LogNormalScatterModel_initialization']


def test_Moster13SmHm_initialization():
    """ Function testing the initialization of
    `~halotools.empirical_models.Moster13SmHm`.
    Summary of tests:

        * Class successfully instantiates when called with no arguments.

        * Class successfully instantiates when constructor is passed ``redshift``, ``prim_haloprop_key``.

        * When the above arguments are passed to the constructor, the instance is correctly initialized with the input values.

        * The scatter model bound to Moster13SmHm correctly inherits each of the above arguments.
    """

    default_model = Moster13SmHm()
    assert default_model.prim_haloprop_key == model_defaults.default_smhm_haloprop
    assert default_model.scatter_model.prim_haloprop_key == model_defaults.default_smhm_haloprop
    assert hasattr(default_model, 'redshift') is False
    assert isinstance(default_model.scatter_model, LogNormalScatterModel)

    keys = ['m10', 'm11', 'n10', 'n11', 'beta10', 'beta11', 'gamma10', 'gamma11', 'scatter_model_param1']
    for key in keys:
        assert key in list(default_model.param_dict.keys())
    assert default_model.param_dict['scatter_model_param1'] == model_defaults.default_smhm_scatter

    default_scatter_dict = {'scatter_model_param1': model_defaults.default_smhm_scatter}
    assert default_model.scatter_model.param_dict == default_scatter_dict
    assert default_model.scatter_model.ordinates == [model_defaults.default_smhm_scatter]

    z0_model = Moster13SmHm(redshift=0)
    assert z0_model.redshift == 0
    z1_model = Moster13SmHm(redshift=1)
    assert z1_model.redshift == 1

    macc_model = Moster13SmHm(prim_haloprop_key='macc')
    assert macc_model.prim_haloprop_key == 'macc'
    assert macc_model.scatter_model.prim_haloprop_key == 'macc'


def test_Moster13SmHm_behavior():
    """
    """
    default_model = Moster13SmHm()
    mstar1 = default_model.mean_stellar_mass(prim_haloprop=1.e12)
    ratio1 = mstar1/3.4275e10
    np.testing.assert_array_almost_equal(ratio1, 1.0, decimal=3)

    default_model.param_dict['n10'] *= 1.1
    mstar2 = default_model.mean_stellar_mass(prim_haloprop=1.e12)
    assert mstar2 > mstar1

    default_model.param_dict['n11'] *= 1.1
    mstar3 = default_model.mean_stellar_mass(prim_haloprop=1.e12)
    assert mstar3 == mstar2

    mstar4_z1 = default_model.mean_stellar_mass(prim_haloprop=1.e12, redshift=1)
    default_model.param_dict['n11'] *= 1.1
    mstar5_z1 = default_model.mean_stellar_mass(prim_haloprop=1.e12, redshift=1)
    assert mstar5_z1 != mstar4_z1

    mstar_realization1 = default_model.mc_stellar_mass(prim_haloprop=1.e12*np.ones(int(1e4)), seed=43)
    mstar_realization2 = default_model.mc_stellar_mass(prim_haloprop=1.e12*np.ones(int(1e4)), seed=43)
    mstar_realization3 = default_model.mc_stellar_mass(prim_haloprop=1.e12*np.ones(int(1e4)), seed=44)
    assert np.array_equal(mstar_realization1, mstar_realization2)
    assert not np.array_equal(mstar_realization1, mstar_realization3)

    measured_scatter1 = np.std(np.log10(mstar_realization1))
    model_scatter = default_model.param_dict['scatter_model_param1']
    np.testing.assert_allclose(measured_scatter1, model_scatter, rtol=1e-3)

    default_model.param_dict['scatter_model_param1'] = 0.3
    mstar_realization4 = default_model.mc_stellar_mass(prim_haloprop=1e12*np.ones(int(1e4)), seed=43)
    measured_scatter4 = np.std(np.log10(mstar_realization4))
    np.testing.assert_allclose(measured_scatter4, 0.3, rtol=1e-3)


def test_LogNormalScatterModel_initialization():
    """ Function testing the initialization of
    `~halotools.empirical_models.LogNormalScatterModel`.
    Summary of tests:

        * Class successfully instantiates when called with no arguments.

        * Class successfully instantiates when constructor is passed ``ordinates`` and ``abscissa``.

        * When the above arguments are passed to the constructor, the instance is correctly initialized with the input values.

    """
    default_scatter_model = LogNormalScatterModel()
    assert default_scatter_model.prim_haloprop_key == model_defaults.default_smhm_haloprop
    assert default_scatter_model.abscissa == [12]
    assert default_scatter_model.ordinates == [model_defaults.default_smhm_scatter]
    default_param_dict = {'scatter_model_param1': model_defaults.default_smhm_scatter}
    assert default_scatter_model.param_dict == default_param_dict

    input_abscissa = [12, 15]
    input_ordinates = [0.3, 0.1]
    scatter_model2 = LogNormalScatterModel(
        scatter_abscissa=input_abscissa, scatter_ordinates=input_ordinates)

    assert np.all(scatter_model2.abscissa == input_abscissa)
    assert np.all(scatter_model2.ordinates == input_ordinates)
    model2_param_dict = {'scatter_model_param1': 0.3, 'scatter_model_param2': 0.1}
    assert scatter_model2.param_dict == model2_param_dict


def test_LogNormalScatterModel_behavior():
    """ Function testing the behavior of
    `~halotools.empirical_models.LogNormalScatterModel`.

    Summary of tests:

        * The default model returns the default scatter, both the mean_scatter method and the scatter_realization method.

        * A model defined by interpolation between 12 and 15 returns the input scatter at the input abscissa, both the mean_scatter method and the scatter_realization method.

        * The 12-15 model returns the correct intermediate level of scatter at the halfway point between 12 and 15, both the mean_scatter method and the scatter_realization method.

        * All the above results apply equally well to cases where ``mass`` or ``halos`` is used as input.

        * When the param_dict of a model is updated (as it would be during an MCMC), the behavior is correctly adjusted.
    """

    testing_seed = 43

    default_scatter_model = LogNormalScatterModel()

    Npts = int(1e4)
    testmass12 = 1e12
    mass12 = np.zeros(Npts) + testmass12
    masskey = model_defaults.default_smhm_haloprop
    d = {masskey: mass12}
    halos12 = Table(d)

    # Test the mean_scatter method of the default model
    scatter = default_scatter_model.mean_scatter(prim_haloprop=testmass12)
    assert np.allclose(scatter, model_defaults.default_smhm_scatter)
    scatter_array = default_scatter_model.mean_scatter(prim_haloprop=mass12)
    assert np.allclose(scatter_array, model_defaults.default_smhm_scatter)
    scatter_array = default_scatter_model.mean_scatter(table=halos12)
    assert np.allclose(scatter_array, model_defaults.default_smhm_scatter)

    # Test the scatter_realization method of the default model
    scatter_realization = default_scatter_model.scatter_realization(seed=testing_seed, prim_haloprop=mass12)
    disp = np.std(scatter_realization)
    np.testing.assert_almost_equal(disp, model_defaults.default_smhm_scatter, decimal=2)
    scatter_realization = default_scatter_model.scatter_realization(seed=testing_seed, table=halos12)
    disp = np.std(scatter_realization)
    np.testing.assert_almost_equal(disp, model_defaults.default_smhm_scatter, decimal=2)

    input_abscissa = [12, 15]
    input_ordinates = [0.3, 0.1]
    scatter_model2 = LogNormalScatterModel(
        scatter_abscissa=input_abscissa, scatter_ordinates=input_ordinates)

    assert len(scatter_model2.abscissa) == 2
    assert len(scatter_model2.param_dict) == 2
    assert set(scatter_model2.param_dict.keys()) == set(['scatter_model_param1', 'scatter_model_param2'])
    assert set(scatter_model2.param_dict.values()) == set(input_ordinates)

    # Test the mean_scatter method of a non-trivial model at the first abscissa
    scatter_array = scatter_model2.mean_scatter(prim_haloprop=mass12)
    assert np.allclose(scatter_array, 0.3)
    scatter_array = scatter_model2.mean_scatter(table=halos12)
    assert np.allclose(scatter_array, 0.3)

    # Test the scatter_realization method of a non-trivial model at the first abscissa
    scatter_realization = scatter_model2.scatter_realization(seed=testing_seed, prim_haloprop=mass12)
    disp = np.std(scatter_realization)
    np.testing.assert_almost_equal(disp, 0.3, decimal=2)
    scatter_realization = scatter_model2.scatter_realization(seed=testing_seed, table=halos12)
    disp = np.std(scatter_realization)
    np.testing.assert_almost_equal(disp, 0.3, decimal=2)

    # Test the mean_scatter method of a non-trivial model at the second abscissa
    testmass15 = 1e15
    mass15 = np.zeros(Npts) + testmass15
    masskey = model_defaults.default_smhm_haloprop
    d = {masskey: mass15}
    halos15 = Table(d)

    scatter_array = scatter_model2.mean_scatter(prim_haloprop=mass15)
    assert np.allclose(scatter_array, 0.1)
    scatter_array = scatter_model2.mean_scatter(table=halos15)
    assert np.allclose(scatter_array, 0.1)

    # Test the scatter_realization method of a non-trivial model at the second abscissa
    scatter_realization = scatter_model2.scatter_realization(seed=testing_seed, prim_haloprop=mass15)
    disp = np.std(scatter_realization)
    np.testing.assert_almost_equal(disp, 0.1, decimal=2)
    scatter_realization = scatter_model2.scatter_realization(seed=testing_seed, table=halos15)
    disp = np.std(scatter_realization)
    np.testing.assert_almost_equal(disp, 0.1, decimal=2)

    # Test the mean_scatter method of a non-trivial model at an intermediate value
    testmass135 = 10.**13.5
    mass135 = np.zeros(Npts) + testmass135
    masskey = model_defaults.default_smhm_haloprop
    d = {masskey: mass135}
    halos135 = Table(d)

    scatter_array = scatter_model2.mean_scatter(prim_haloprop=mass135)
    assert np.allclose(scatter_array, 0.2)
    scatter_array = scatter_model2.mean_scatter(table=halos135)
    assert np.allclose(scatter_array, 0.2)

    # Test the scatter_realization method of a non-trivial model at an intermediate value
    scatter_realization = scatter_model2.scatter_realization(seed=testing_seed, prim_haloprop=mass135)
    disp = np.std(scatter_realization)
    np.testing.assert_almost_equal(disp, 0.2, decimal=2)
    scatter_realization = scatter_model2.scatter_realization(seed=testing_seed, table=halos135)
    disp = np.std(scatter_realization)
    np.testing.assert_almost_equal(disp, 0.2, decimal=2)

    # Update the parameter dictionary that defines the non-trivial model
    scatter_model2.param_dict['scatter_model_param2'] = 0.5

    # Test the mean_scatter method of the updated non-trivial model
    scatter_array = scatter_model2.mean_scatter(prim_haloprop=mass12)
    assert np.allclose(scatter_array, 0.3)
    scatter_array = scatter_model2.mean_scatter(prim_haloprop=mass15)
    assert np.allclose(scatter_array, 0.5)
    scatter_array = scatter_model2.mean_scatter(prim_haloprop=mass135)
    assert np.allclose(scatter_array, 0.4)

    # Test the scatter_realization method of the updated non-trivial model
    scatter_realization = scatter_model2.scatter_realization(seed=testing_seed, prim_haloprop=mass15)
    disp = np.std(scatter_realization)
    np.testing.assert_almost_equal(disp, 0.5, decimal=2)
    scatter_realization = scatter_model2.scatter_realization(seed=testing_seed, prim_haloprop=mass135)
    disp = np.std(scatter_realization)
    np.testing.assert_almost_equal(disp, 0.4, decimal=2)
    scatter_realization = scatter_model2.scatter_realization(seed=testing_seed, prim_haloprop=mass12)
    disp = np.std(scatter_realization)
    np.testing.assert_almost_equal(disp, 0.3, decimal=2)


class TestBehroozi10SmHm(TestCase):

    def setup_class(self):
        """ Use tabular data provided by Peter Behroozi
        as a blackbox test of the implementation.
        """

        self.model = Behroozi10SmHm()

        self.logmratio_z1 = np.array(
            [-2.145909, -2.020974, -1.924020, -1.852937,
            -1.804730, -1.776231, -1.764455, -1.766820,
            -1.781140, -1.805604, -1.838727, -1.879292,
            -1.926290, -1.978890, -2.036405, -2.098245,
            -2.163930, -2.233045, -2.305230, -2.380185,
            -2.457643, -2.537377, -2.619191, -2.702901]
            )

        self.logmh_z1 = np.array(
            [11.368958, 11.493958, 11.618958, 11.743958,
            11.868958, 11.993958, 12.118958, 12.243958,
            12.368958, 12.493958, 12.618958, 12.743958,
            12.868958, 12.993958, 13.118958, 13.243958,
            13.368958, 13.493958, 13.618958, 13.743958,
            13.868958, 13.993958, 14.118958, 14.243958]
            )

        self.logmh_z01 = np.array(
            [10.832612, 10.957612, 11.082612, 11.207612,
            11.332612, 11.457612, 11.582612, 11.707612,
            11.832612, 11.957612, 12.082612, 12.207612,
            12.332612, 12.457612, 12.582612, 12.707612,
            12.832612, 12.957612, 13.082612, 13.207612,
            13.332612, 13.457612, 13.582612, 13.707612,
            13.832612, 13.957612, 14.082612, 14.207612,
            14.332612, 14.457612, 14.582612, 14.707612,
            14.832612, 14.957612, 15.082612, 15.207612])

        self.logmratio_z01 = np.array(
            [-2.532613, -2.358159, -2.184308, -2.012586,
            -1.847878, -1.702718, -1.596036, -1.537164,
            -1.518895, -1.529237, -1.558904, -1.601876,
            -1.654355, -1.713868, -1.778768, -1.84792,
            -1.920522, -1.995988, -2.07388, -2.153878,
            -2.235734, -2.319242, -2.404256, -2.490647,
            -2.578321, -2.66718, -2.757161, -2.848199,
            -2.94024, -3.033235, -3.127133, -3.221902,
            -3.317498, -3.413892, -3.511041, -3.608918])

        self.logmratio_z05 = np.array([
            -2.375180, -2.183537, -2.015065, -1.879960,
            -1.782708, -1.720799, -1.688169, -1.678521,
            -1.686669, -1.708703, -1.741731, -1.783616,
            -1.832761, -1.887952, -1.948255, -2.012940,
            -2.081414, -2.153203, -2.227921, -2.305249,
            -2.384912, -2.466680, -2.550359, -2.635785,
            -2.722806, -2.811296, -2.901139, -2.992246,
            -3.084516, -3.177873]
            )

        self.logmh_z05 = np.array([
            11.066248, 11.191248, 11.316248, 11.441248,
            11.566248, 11.691248, 11.816248, 11.941248,
            12.066248, 12.191248, 12.316248, 12.441248,
            12.566248, 12.691248, 12.816248, 12.941248,
            13.066248, 13.191248, 13.316248, 13.441248,
            13.566248, 13.691248, 13.816248, 13.941248,
            14.066248, 14.191248, 14.316248, 14.441248,
            14.566248, 14.691248]
            )

        self.logmh_z01 = np.log10((10.**self.logmh_z01)/self.model.littleh)
        self.logmratio_z05 = np.log10((10.**self.logmratio_z05)/self.model.littleh)
        self.logmh_z1 = np.log10((10.**self.logmh_z1)/self.model.littleh)
        self.logmratio_z01 = np.log10((10.**self.logmratio_z01)/self.model.littleh)
        self.logmh_z05 = np.log10((10.**self.logmh_z05)/self.model.littleh)
        self.logmratio_z1 = np.log10((10.**self.logmratio_z1)/self.model.littleh)

    def test_behroozi10_smhm_blackbox(self):
        """
        """

        halo_mass_z01 = 10.**self.logmh_z01
        z01_sm = self.model.mean_stellar_mass(prim_haloprop=halo_mass_z01, redshift=0.1)
        z01_ratio = z01_sm / halo_mass_z01
        z01_result = np.log10(z01_ratio)
        assert np.allclose(z01_result, self.logmratio_z01, rtol=0.02)

        halo_mass_z05 = 10.**self.logmh_z05
        z05_sm = self.model.mean_stellar_mass(prim_haloprop=halo_mass_z05, redshift=0.5)
        z05_ratio = z05_sm / halo_mass_z05
        z05_result = np.log10(z05_ratio)
        assert np.allclose(z05_result, self.logmratio_z05, rtol=0.02)

        halo_mass_z1 = 10.**self.logmh_z1
        z1_sm = self.model.mean_stellar_mass(prim_haloprop=halo_mass_z1, redshift=1)
        z1_ratio = z1_sm / halo_mass_z1
        z1_result = np.log10(z1_ratio)
        assert np.allclose(z1_result, self.logmratio_z1, rtol=0.02)
