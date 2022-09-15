"""
"""

import numpy as np

from ...component_model_templates import LogNormalScatterModel
from ...smhm_models import Moster13SmHm

from ... import model_defaults

__all__ = ("test_Moster13SmHm_initialization", "test_Moster13SmHm_behavior")


def test_Moster13SmHm_initialization():
    """Function testing the initialization of
    `~halotools.empirical_models.Moster13SmHm`.
    Summary of tests:

        * Class successfully instantiates when called with no arguments.

        * Class successfully instantiates when constructor is passed ``redshift``, ``prim_haloprop_key``.

        * When the above arguments are passed to the constructor, the instance is correctly initialized with the input values.

        * The scatter model bound to Moster13SmHm correctly inherits each of the above arguments.
    """

    default_model = Moster13SmHm()
    assert default_model.prim_haloprop_key == model_defaults.default_smhm_haloprop
    assert (
        default_model.scatter_model.prim_haloprop_key
        == model_defaults.default_smhm_haloprop
    )
    assert hasattr(default_model, "redshift") is False
    assert isinstance(default_model.scatter_model, LogNormalScatterModel)

    keys = [
        "m10",
        "m11",
        "n10",
        "n11",
        "beta10",
        "beta11",
        "gamma10",
        "gamma11",
        "scatter_model_param1",
    ]
    for key in keys:
        assert key in list(default_model.param_dict.keys())
    assert (
        default_model.param_dict["scatter_model_param1"]
        == model_defaults.default_smhm_scatter
    )

    default_scatter_dict = {"scatter_model_param1": model_defaults.default_smhm_scatter}
    assert default_model.scatter_model.param_dict == default_scatter_dict
    assert default_model.scatter_model.ordinates == [
        model_defaults.default_smhm_scatter
    ]

    z0_model = Moster13SmHm(redshift=0)
    assert z0_model.redshift == 0
    z1_model = Moster13SmHm(redshift=1)
    assert z1_model.redshift == 1

    macc_model = Moster13SmHm(prim_haloprop_key="macc")
    assert macc_model.prim_haloprop_key == "macc"
    assert macc_model.scatter_model.prim_haloprop_key == "macc"


def test_Moster13SmHm_behavior():
    """ """
    default_model = Moster13SmHm(redshift=0.0)
    mstar1 = default_model.mean_stellar_mass(
        prim_haloprop=1.0e12 * default_model.littleh
    )
    ratio1 = mstar1 / ((3.4275e10) * default_model.littleh**2)
    np.testing.assert_array_almost_equal(ratio1, 1.0, decimal=3)

    default_model.param_dict["n10"] *= 1.1
    mstar2 = default_model.mean_stellar_mass(prim_haloprop=1.0e12)
    assert mstar2 > mstar1

    default_model.param_dict["n11"] *= 1.1
    mstar3 = default_model.mean_stellar_mass(prim_haloprop=1.0e12)
    assert mstar3 == mstar2

    mstar4_z1 = default_model.mean_stellar_mass(prim_haloprop=1.0e12, redshift=1)
    default_model.param_dict["n11"] *= 1.1
    mstar5_z1 = default_model.mean_stellar_mass(prim_haloprop=1.0e12, redshift=1)
    assert mstar5_z1 != mstar4_z1

    mstar_realization1 = default_model.mc_stellar_mass(
        prim_haloprop=1.0e12 * np.ones(int(1e4)), seed=43
    )
    mstar_realization2 = default_model.mc_stellar_mass(
        prim_haloprop=1.0e12 * np.ones(int(1e4)), seed=43
    )
    mstar_realization3 = default_model.mc_stellar_mass(
        prim_haloprop=1.0e12 * np.ones(int(1e4)), seed=44
    )
    assert np.array_equal(mstar_realization1, mstar_realization2)
    assert not np.array_equal(mstar_realization1, mstar_realization3)

    measured_scatter1 = np.std(np.log10(mstar_realization1))
    model_scatter = default_model.param_dict["scatter_model_param1"]
    np.testing.assert_allclose(measured_scatter1, model_scatter, rtol=1e-3)

    default_model.param_dict["scatter_model_param1"] = 0.3
    mstar_realization4 = default_model.mc_stellar_mass(
        prim_haloprop=1e12 * np.ones(int(1e4)), seed=43
    )
    measured_scatter4 = np.std(np.log10(mstar_realization4))
    np.testing.assert_allclose(measured_scatter4, 0.3, rtol=1e-3)
