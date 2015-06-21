# -*- coding: utf-8 -*-
"""

Module containing some commonly used subhalo-based models.

"""

from . import model_defaults
from .mock_factories import SubhaloMockFactory
from . import smhm_components
from . import sfr_components
from . import abunmatch
from ..sim_manager import sim_defaults
from ..sim_manager import FakeMock

__all__ = ['SmHmBinarySFR_blueprint', 'Campbell15_blueprint']


def SmHmBinarySFR_blueprint(
    prim_haloprop_key = model_defaults.default_smhm_haloprop, 
    smhm_model=smhm_components.Moster13SmHm, 
    scatter_level = 0.2, 
    redshift = sim_defaults.default_redshift, 
    sfr_abcissa = [12, 15], sfr_ordinates = [0.25, 0.75], logparam=True, 
    **kwargs):
    """ Blueprint for a simple model assigning stellar mass and 
    quiescent/active designation to a subhalo catalog. 

    Parameters 
    ----------
    prim_haloprop_key : string, optional keyword argument 
        String giving the column name of the primary halo property governing 
        the galaxy propery being modeled.  
        Default is set in the `~halotools.empirical_models.model_defaults` module. 

    smhm_model : object, optional keyword argument 
        Sub-class of `~halotools.empirical_models.smhm_components.PrimGalpropModel` governing 
        the stellar-to-halo-mass relation. Default is `Moster13SmHm`. 

    scatter_level : float, optional keyword argument 
        Constant amount of scatter in stellar mass, in dex. Default is 0.2. 

    redshift : float, optional keyword argument
        Redshift of the halo hosting the galaxy. Used to evaluate the 
        stellar-to-halo-mass relation. Default is set in `~halotools.sim_manager.sim_defaults`. 

    sfr_abcissa : array, optional keyword argument 
        Values of the primary halo property at which the quiescent fraction is specified. 
        Default is [12, 15], in accord with the default True value for ``logparam``. 

    sfr_ordinates : array, optional keyword argument 
        Values of the quiescent fraction when evaluated at the input abcissa. 
        Default is [0.25, 0.75]

    logparam : bool, optional keyword argument
        If set to True, the interpolation will be done 
        in the base-10 logarithm of the primary halo property, 
        rather than linearly. Default is True. 

    Returns 
    -------
    blueprint : dict 
        Dictionary containing instructions for how to build the model. 
        When model_blueprint is passed to `~halotools.empirical_models.SubhaloModelFactory`, 
        the factory returns the SmHmBinarySFR model object. 
    """

    sfr_model = sfr_components.BinaryGalpropInterpolModel(
        galprop_key='quiescent', prim_haloprop_key=prim_haloprop_key, 
        abcissa=sfr_abcissa, ordinates=sfr_ordinates, logparam=logparam)

    sm_model = smhm_components.Moster13SmHm(
        prim_haloprop_key=prim_haloprop_key, redshift=redshift, 
        scatter_abcissa = [12], scatter_ordinates = [scatter_level])

    blueprint = {sm_model.galprop_key: sm_model, sfr_model.galprop_key: sfr_model, 'mock_factory': SubhaloMockFactory}

    return blueprint


def Campbell15_blueprint(
    prim_haloprop_key = model_defaults.default_smhm_haloprop, 
    sec_galprop_key = 'ssfr', 
    smhm_model=smhm_components.Moster13SmHm, 
    scatter_level = 0.2, 
    redshift = sim_defaults.default_redshift, **kwargs):
    """ Blueprint for conditional abundance matching models based on Campbell et al. (2015). 

    Parameters
    -----------
    prim_haloprop_key : string, optional keyword argument 
        String giving the column name of the primary halo property governing 
        the galaxy propery being modeled.  
        Default is set in the `~halotools.empirical_models.model_defaults` module. 

    sec_haloprop_key : string, required keyword argument 
        Column name of the subhalo property that CAM models as 
        being correlated with ``galprop_key`` at fixed ``prim_galprop_key``. 

    prim_galprop_key : string, required keyword argument 
        Column name such as ``stellar_mass`` or ``luminosity`` 
        where the primary galaxy property is stored in 
        ``input_galaxy_table``. 

    sec_galprop_key : string, optional keyword argument 
        Column name such as ``gr_color`` or ``ssfr`` 
        of the secondary galaxy property being modeled. 
        Can be any column of ``input_galaxy_table`` other than 
        ``prim_galprop_key``. Default is ``ssfr``. 

    input_galaxy_table : data table, required keyword argument 
        Astropy Table object storing the input galaxy population 
        upon which the CAM model is based.  

    prim_galprop_bins : array, required keyword argument 
        Array used to bin ``input_galaxy_table`` by ``prim_galprop_key``. 

    smhm_model : object, optional keyword argument 
        Sub-class of `~halotools.empirical_models.smhm_components.PrimGalpropModel` governing 
        the stellar-to-halo-mass relation. Default is `Moster13SmHm`. 

    scatter_level : float, optional keyword argument 
        Constant amount of scatter in dex in ``prim_galprop_key`` 
        at fixed ``prim_haloprop_key``. Default is 0.2. 

    redshift : float, optional keyword argument
        Redshift of the halo hosting the galaxy. Used to evaluate the 
        stellar-to-halo-mass relation. Default is set in `~halotools.sim_manager.sim_defaults`. 

    correlation_strength : float or array, optional keyword argument 
        Specifies the absolute value of the desired 
        Spearman rank-order correlation coefficient 
        between ``sec_haloprop_key`` and ``galprop_key``. 
        If a float, the correlation strength will be assumed constant 
        for all values of ``prim_galprop_key``. If an array, the i^th entry 
        specifies the correlation strength when ``prim_galprop_key`` equals  
        ``prim_galprop_bins[i]``. Entries must be in the range [-1, 1], 
        with negative values corresponding to anti-correlations; 
        the endpoints signify maximum correlation, zero signifies 
        that ``sec_haloprop_key`` and ``galprop_key`` are uncorrelated. 
        Default is maximum (positive) correlation strength of 1. 

    correlation_strength_abcissa : float or array, optional keyword argument 
        Specifies the value of ``prim_galprop_key`` at which 
        the input ``correlation_strength`` applies. ``correlation_strength_abcissa`` 
        need only be specified if a ``correlation_strength`` array is passed. 
        Intermediary values of the correlation strength at values 
        between the abcissa are solved for by spline interpolation. 
    """

    stellar_mass_model = smhm_model(
        prim_haloprop_key=prim_haloprop_key, redshift=redshift, 
        scatter_abcissa = [12], scatter_ordinates = [scatter_level])

    ssfr_model = abunmatch.ConditionalAbunMatch(
        galprop_key='ssfr', 
        prim_galprop_key=stellar_mass_model.galprop_key, 
        **kwargs)

    fake_mock = FakeMock(approximate_ngals = 1e5)

    ssfr_model = ConditionalAbunMatch(input_galaxy_table=input_galaxy_table, 
                                      sec_haloprop_key=sec_haloprop_key, 
                                      galprop_key=sec_galprop_key, 
                                      prim_galprop_key=prim_galprop_key, 
                                      prim_galprop_bins=prim_galprop_bins, 
                                      correlation_strength=correlation_strength)
    blueprint = ({
        stellar_mass_model.galprop_key: stellar_mass_model, 
        ssfr_model.galprop_key: ssfr_model, 
        'mock_factory': SubhaloMockFactory})

    return blueprint













