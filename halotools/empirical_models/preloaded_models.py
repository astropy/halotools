# -*- coding: utf-8 -*-
"""

Module containing some commonly used composite HOD models.

"""
from . import model_factories
from . import preloaded_subhalo_model_blueprints
from . import preloaded_hod_blueprints

from .. import sim_manager

__all__ = ['Kravtsov04', 'SmHmBinarySFR']

def Kravtsov04(**kwargs):
    """ Simple HOD-style model based on Kravtsov et al. (2004). 

    There are two populations, centrals and satellites. 
    Central occupation statistics are given by a nearest integer distribution 
    with first moment given by an ``erf`` function. 
    Satellite occupation statistics are given by a Poisson distribution 
    with first moment given by a power law that has been truncated at the low-mass end. 

    Under the hood, this model is built from a set of component models whose 
    behavior is coded up elsewhere. The behavior of the central occupations 
    derives from the `~halotools.empirical_models.hod_components.Zheng07Cens` class, while for 
    satellites the relevant class is `~halotools.empirical_models.hod_components.Kravtsov04Sats`. 

    This composite model was built by the `~halotools.empirical_models.model_factories.HodModelFactory`, 
    which followed the instructions contained in 
    `~halotools.empirical_models.Kravtsov04_blueprint`. 

    Parameters 
    ----------
    threshold : float, optional keyword argument
        Luminosity threshold of the galaxy sample being modeled. 

    Returns 
    -------
    model : object 
        Instance of `~halotools.empirical_models.model_factories.HodModelFactory`

    Examples 
    --------
    >>> model = Kravtsov04()
    >>> model = Kravtsov04(threshold = -20.5)

    To use our model to populate a simulation with mock galaxies, we only need to 
    load a snapshot into memory and call the built-in ``populate_mock`` method. 
    For illustration purposes, we'll use a small, fake simulation:

    >>> fake_snapshot = sim_manager.FakeSim()
    >>> model.populate_mock(snapshot = fake_snapshot)

    """
    blueprint = preloaded_hod_blueprints.Kravtsov04_blueprint(**kwargs)
    return model_factories.HodModelFactory(blueprint, **kwargs)

def SmHmBinarySFR(**kwargs):
    """ Blueprint for a very simple model assigning stellar mass and 
    quiescent/active designation to a subhalo catalog. 

    Stellar masses are assigned according to a parameterized relation, 
    Moster et al. (2013) by default. SFR designation is determined by 
    interpolating between a set of input control points, with default 
    behavior being a 25% quiescent fraction for galaxies 
    residing in Milky Way halos, and 75% for cluster galaxies. 

    Since `SmHmBinarySFR` does not discriminate between centrals and satellites 
    in the SFR assignment, this model is physically unrealistic and is 
    included here primarily for demonstration purposes. 

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

    threshold : float, optional keyword argument 
        Stellar mass threshold of mock galaxy catalog. Default is None, 
        in which case the lower bound on stellar mass will be entirely determined 
        by the resolution of the N-body simulation and the model parameters. 
        
    Returns 
    -------
    model : object 
        Instance of `~halotools.empirical_models.model_factories.SubhaloModelFactory`

    Examples 
    --------
    >>> model = SmHmBinarySFR()
    >>> model = SmHmBinarySFR(threshold = 10**10.5)

    To use our model to populate a simulation with mock galaxies, we only need to 
    load a snapshot into memory and call the built-in ``populate_mock`` method. 
    For illustration purposes, we'll use a small, fake simulation:

    >>> fake_snapshot = sim_manager.FakeSim()
    >>> model.populate_mock(snapshot = fake_snapshot)

    """

    blueprint = preloaded_subhalo_model_blueprints.SmHmBinarySFR_blueprint(**kwargs)

    if 'threshold' in kwargs.keys():
        galaxy_selection_func = lambda x: x['stellar_mass'] > kwargs['threshold']
        model = model_factories.SubhaloModelFactory(blueprint, 
            galaxy_selection_func=galaxy_selection_func)
    else:
        model = model_factories.SubhaloModelFactory(blueprint)

    return model













