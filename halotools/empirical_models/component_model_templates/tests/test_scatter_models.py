""" Module providing unit-testing for the functions in
the `~halotools.empirical_models.component_model_templates.scatter_models` module
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.table import Table

from ..scatter_models import LogNormalScatterModel

from ... import model_defaults

from ....sim_manager import FakeSim

__all__ = ['test_nonzero_scatter', 'test_zero_scatter']


def test_nonzero_scatter():
    halocat = FakeSim()
    halo_table = halocat.halo_table

    scatter_model = LogNormalScatterModel(scatter_abscissa=[10**10, 10**12],
        scatter_ordinates=[0.1, 0.1])

    scatter = scatter_model.scatter_realization(table=halo_table)

    assert len(scatter) == len(halo_table)


def test_zero_scatter():
    halocat = FakeSim()
    halo_table = halocat.halo_table

    scatter_model = LogNormalScatterModel(scatter_abscissa=[10**10, 10**12],
        scatter_ordinates=[0.0, 0.0])

    scatter = scatter_model.scatter_realization(table=halo_table)

    assert len(scatter) == len(halo_table)
    assert np.all(scatter == 0.0)


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
