# -*- coding: utf-8 -*-
"""

Module containing some commonly used subhalo-based models.

"""

from . import model_defaults
from .mock_factories import SubhaloMockFactory
from . import smhm_components
from . import sfr_components
from ..sim_manager import sim_defaults

__all__ = ['SmHmBinarySFR_blueprint']


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












