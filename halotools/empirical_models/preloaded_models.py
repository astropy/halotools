# -*- coding: utf-8 -*-
"""

Module containing some commonly used composite HOD models.

"""
from . import model_factories
from . import preloaded_subhalo_model_blueprints
from . import preloaded_hod_blueprints

from .. import sim_manager

__all__ = ['Zheng07', 'SmHmBinarySFR', 'Campbell15']

def Zheng07(**kwargs):
    """ Simple HOD-style model based on Kravtsov et al. (2004). 

    There are two populations, centrals and satellites. 
    Central occupation statistics are given by a nearest integer distribution 
    with first moment given by an ``erf`` function. 
    Satellite occupation statistics are given by a Poisson distribution 
    with first moment given by a power law that has been truncated at the low-mass end. 

    Under the hood, this model is built from a set of component models whose 
    behavior is coded up elsewhere. The behavior of the central occupations 
    derives from the `~halotools.empirical_models.hod_components.Zheng07Cens` class, while for 
    satellites the relevant class is `~halotools.empirical_models.hod_components.Zheng07Sats`. 

    This composite model was built by the `~halotools.empirical_models.model_factories.HodModelFactory`, 
    which followed the instructions contained in 
    `~halotools.empirical_models.Zheng07_blueprint`. 

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
    >>> model = Zheng07()
    >>> model = Zheng07(threshold = -20.5)

    To use our model to populate a simulation with mock galaxies, we only need to 
    load a snapshot into memory and call the built-in ``populate_mock`` method. 
    For illustration purposes, we'll use a small, fake simulation:

    >>> fake_snapshot = sim_manager.FakeSim()
    >>> model.populate_mock(snapshot = fake_snapshot)

    """
    blueprint = preloaded_hod_blueprints.Zheng07_blueprint(**kwargs)
    return model_factories.HodModelFactory(blueprint, **kwargs)

def Leauthaud11(**kwargs):
    """ 
    """
    blueprint = preloaded_hod_blueprints.Leauthaud11_blueprint(**kwargs)
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


def Campbell15(**kwargs):
    """ Conditional abundance matching model based on Campbell et al. (2015). 

    Parameters
    -----------
    prim_haloprop_key : string, optional keyword argument 
        String giving the column name of the primary halo property governing 
        the galaxy propery being modeled.  
        Default is set in the `~halotools.empirical_models.model_defaults` module. 

    sec_haloprop_key : string, optional keyword argument 
        Column name of the subhalo property that CAM models as 
        being correlated with ``galprop_key`` at fixed ``prim_galprop_key``. 
        Default is ``vpeak``. 

    sec_galprop_key : string, optional keyword argument 
        Column name such as ``gr_color`` or ``ssfr`` 
        of the secondary galaxy property being modeled. 
        Can be any column of ``input_galaxy_table`` other than 
        ``prim_galprop_key``. Default is ``ssfr``. 

    input_galaxy_table : data table, optional keyword argument 
        Astropy Table object storing the input galaxy population 
        upon which the CAM model is based.  
        Default behavior is to use `~halotools.sim_manager.FakeMock`. 

    prim_galprop_bins : array, optional keyword argument 
        Array used to bin ``input_galaxy_table`` by ``prim_galprop_key``. 
        Default is 15 bins logarithmically spaced between 
        :math:`10^{8}M_{\odot}` and :math:`10^{12}M_{\odot}`. 

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
        Default is constant maximum (positive) correlation strength of 1. 

    correlation_strength_abcissa : float or array, optional keyword argument 
        Specifies the value of ``prim_galprop_key`` at which 
        the input ``correlation_strength`` applies. ``correlation_strength_abcissa`` 
        need only be specified if a ``correlation_strength`` array is passed. 
        Intermediary values of the correlation strength at values 
        between the abcissa are solved for by spline interpolation. 
        Default is constant maximum (positive) correlation strength of 1. 

    threshold : float, optional keyword argument 
        Stellar mass threshold of mock galaxy catalog. Default is None, 
        in which case the lower bound on stellar mass will be entirely determined 
        by the resolution of the N-body simulation and the model parameters. 

    Examples 
    --------
    To load the Campbell et al. (2015) model object with all default settings, simply call 
    the `Campbell15` function with no arguments:

    >>> model = Campbell15()

    To use our model to populate a simulation with mock galaxies, we only need to 
    load a snapshot into memory and call the built-in ``populate_mock`` method. 
    For illustration purposes, we'll use a small, fake simulation, though you 
    can populate a real simulation by instead calling the 
    `~halotools.sim_manager.ProcessedSnapshot` class. 

    >>> fake_snapshot = sim_manager.FakeSim()
    >>> model.populate_mock(snapshot = fake_snapshot)

    We can easily build alternative versions of models and mocks by calling the 
    `Campbell15` function with different arguments:

    >>> model_with_scatter = Campbell15(correlation_strength = 0.8, sec_haloprop_key = 'halo_zhalf')
    >>> model_with_scatter.populate_mock(snapshot = fake_snapshot)


    """

    blueprint = preloaded_subhalo_model_blueprints.Campbell15_blueprint(**kwargs)

    if 'threshold' in kwargs.keys():
        galaxy_selection_func = lambda x: x['stellar_mass'] > kwargs['threshold']
        model = model_factories.SubhaloModelFactory(blueprint, 
            galaxy_selection_func=galaxy_selection_func)
    else:
        model = model_factories.SubhaloModelFactory(blueprint)

    return model













