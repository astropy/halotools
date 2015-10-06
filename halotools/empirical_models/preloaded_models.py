# -*- coding: utf-8 -*-
"""

Module containing some commonly used composite HOD models.

"""
import numpy as np
from . import model_factories, model_defaults, smhm_components
from . import hod_components as hoc
from . import smhm_components
from . import sfr_components
from .phase_space_models import NFWPhaseSpace, TrivialPhaseSpace
from .abunmatch import ConditionalAbunMatch

from ..sim_manager import FakeMock, FakeSim, sim_defaults


__all__ = ['Zheng07', 'SmHmBinarySFR', 'Leauthaud11', 'Campbell15', 'Hearin15', 'Tinker13']

def Zheng07(threshold = model_defaults.default_luminosity_threshold, **kwargs):
    """ Simple HOD-style based on Zheng et al. (2007), arXiv:0703457. 

    There are two populations, centrals and satellites. 
    Central occupation statistics are given by a nearest integer distribution 
    with first moment given by an ``erf`` function; the class governing this 
    behavior is `~halotools.empirical_models.hod_components.Zheng07Cens`. 
    Central galaxies are assumed to reside at the exact center of the host halo; 
    the class governing this behavior is `~halotools.empirical_models.TrivialPhaseSpace`. 

    Satellite occupation statistics are given by a Poisson distribution 
    with first moment given by a power law that has been truncated at the low-mass end; 
    the class governing this behavior is `~halotools.empirical_models.hod_components.Zheng07Sats`; 
    satellites in this model follow an (unbiased) NFW profile, as governed by the 
    `~halotools.empirical_models.NFWPhaseSpace` class. 

    This composite model was built by the `~halotools.empirical_models.model_factories.HodModelFactory`.

    Parameters 
    ----------
    threshold : float, optional 
        Luminosity threshold of the galaxy sample being modeled. 
        Default is set in the `~halotools.empirical_models.model_defaults` module. 

    Returns 
    -------
    model : object 
        Instance of `~halotools.empirical_models.model_factories.HodModelFactory`

    Examples 
    --------
    Calling the `Zheng07` class with no arguments instantiates a model based on the 
    default luminosity threshold: 

    >>> model = Zheng07()

    The default settings are set in the `~halotools.empirical_models.model_defaults` module. 
    To load a model based on a different threshold, use the ``threshold`` keyword argument:

    >>> model = Zheng07(threshold = -20.5)

    This call will create a model whose parameter values are set according to the best-fit 
    values given in Table 1 of arXiv:0703457. 

    To use our model to populate a simulation with mock galaxies, we only need to 
    load a snapshot into memory and call the built-in ``populate_mock`` method. 
    For illustration purposes, we'll use a small, fake simulation:

    >>> fake_snapshot = FakeSim() # doctest: +SKIP
    >>> model.populate_mock(snapshot = fake_snapshot) # doctest: +SKIP

    """

    ### Build model for centrals
    cen_key = 'centrals'
    cen_model_dict = {}
    # Build the occupation model
    occu_cen_model = hoc.Zheng07Cens(threshold = threshold, **kwargs)
    cen_model_dict['occupation'] = occu_cen_model
    # Build the profile model
    
    cen_profile = TrivialPhaseSpace(**kwargs)
    cen_model_dict['profile'] = cen_profile

    ### Build model for satellites
    sat_key = 'satellites'
    sat_model_dict = {}
    # Build the occupation model
    occu_sat_model = hoc.Zheng07Sats(threshold = threshold, **kwargs)
    occu_sat_model._suppress_repeated_param_warning = True
    sat_model_dict['occupation'] = occu_sat_model
    # Build the profile model
    sat_profile = NFWPhaseSpace(**kwargs)    
    sat_model_dict['profile'] = sat_profile

    model_blueprint = {
        occu_cen_model.gal_type : cen_model_dict,
        occu_sat_model.gal_type : sat_model_dict 
        }

    composite_model = model_factories.HodModelFactory(model_blueprint)
    return composite_model

def Leauthaud11(threshold = model_defaults.default_stellar_mass_threshold, 
    central_velocity_bias = False, satellite_velocity_bias = False, **kwargs):
    """ HOD-style based on Leauthaud et al. (2011), arXiv:1103.2077. 
    The behavior of this model is governed by an assumed underlying stellar-to-halo-mass relation. 

    There are two populations, centrals and satellites. 
    Central occupation statistics are given by a nearest integer distribution 
    with first moment given by an ``erf`` function; the class governing this 
    behavior is `~halotools.empirical_models.hod_components.Leauthaud11Cens`. 
    Central galaxies are assumed to reside at the exact center of the host halo; 
    the class governing this behavior is `~halotools.empirical_models.TrivialPhaseSpace`. 

    Satellite occupation statistics are given by a Poisson distribution 
    with first moment given by a power law that has been truncated at the low-mass end; 
    the class governing this behavior is `~halotools.empirical_models.hod_components.Leauthaud11Sats`; 
    satellites in this model follow an (unbiased) NFW profile, as governed by the 
    `~halotools.empirical_models.NFWPhaseSpace` class. 

    This composite model was built by the `~halotools.empirical_models.model_factories.HodModelFactory`, 
    which followed the instructions contained in `~halotools.empirical_models.Leauthaud11_blueprint`. 

    Parameters 
    ----------
    threshold : float, optional 
        Stellar mass threshold of the mock galaxy sample. 
        Default value is specified in the `~halotools.empirical_models.model_defaults` module.

    concentration_binning : tuple, optional 
        Three-element tuple. The first entry will be the minimum 
        value of the concentration in the lookup table for the satellite NFW profile, 
        the second entry the maximum, the third entry 
        the linear spacing of the grid. 
        Default is set in `~halotools.empirical_models.model_defaults`.  
        If high-precision is not required, the lookup tables will build much faster if 
        ``concentration_binning`` is set to (1, 25, 0.5).

    central_velocity_bias : bool, optional 
        Boolean specifying whether the central galaxy velocities are biased 
        with respect to the halo velocities. If True, ``param_dict`` will have a 
        parameter called ``velbias_centrals`` that multiplies the underlying 
        halo velocities. Default is False. 

    satellite_velocity_bias : bool, optional 
        Boolean specifying whether the satellite galaxy velocities are biased 
        with respect to the halo velocities. If True, ``param_dict`` will have a 
        parameter called ``velbias_satellites`` that multiplies the underlying 
        Jeans solution for the halo radial velocity dispersion by an overall factor. 
        Default is False. 

    Returns 
    -------
    model : object 
        Instance of `~halotools.empirical_models.model_factories.HodModelFactory`

    Examples 
    --------
    Calling the `Leauthaud11` class with no arguments instantiates a model based on the 
    default stellar mass threshold: 

    >>> model = Leauthaud11()

    The default settings are set in the `~halotools.empirical_models.model_defaults` module. 
    To load a model based on a different threshold, use the ``threshold`` keyword argument:

    >>> model = Leauthaud11(threshold = 11.25)

    To use our model to populate a simulation with mock galaxies, we only need to 
    load a snapshot into memory and call the built-in ``populate_mock`` method. 
    For illustration purposes, we'll use a small, fake simulation:

    >>> fake_snapshot = FakeSim()
    >>> model.populate_mock(snapshot = fake_snapshot) # doctest: +SKIP

    """
    ### Build model for centrals
    cen_key = 'centrals'
    cen_model_dict = {}
    # Build the occupation model
    occu_cen_model = hoc.Leauthaud11Cens(threshold = threshold, **kwargs)
    occu_cen_model._suppress_repeated_param_warning = True
    cen_model_dict['occupation'] = occu_cen_model
    # Build the profile model
    
    cen_profile = TrivialPhaseSpace(velocity_bias = central_velocity_bias, **kwargs)

    cen_model_dict['profile'] = cen_profile

    ### Build model for satellites
    sat_key = 'satellites'
    sat_model_dict = {}
    # Build the occupation model
    occu_sat_model = hoc.Leauthaud11Sats(threshold = threshold, **kwargs)
    sat_model_dict['occupation'] = occu_sat_model
    # Build the profile model
    sat_profile = NFWPhaseSpace(velocity_bias = satellite_velocity_bias, **kwargs)    
    sat_model_dict['profile'] = sat_profile

    model_blueprint = {
        occu_cen_model.gal_type : cen_model_dict,
        occu_sat_model.gal_type : sat_model_dict
        }

    composite_model = model_factories.HodModelFactory(model_blueprint)
    return composite_model


def SmHmBinarySFR(
    prim_haloprop_key = model_defaults.default_smhm_haloprop, 
    smhm_model=smhm_components.Moster13SmHm, 
    scatter_level = 0.2, 
    redshift = sim_defaults.default_redshift, 
    sfr_abcissa = [12, 15], sfr_ordinates = [0.25, 0.75], logparam=True, 
    **kwargs):
    """ Very simple model assigning stellar mass and 
    quiescent/active designation to a subhalo catalog. 

    Stellar masses are assigned according to a parameterized relation, 
    Behroozi et al. (2010) by default. SFR designation is determined by 
    interpolating between a set of input control points, with default 
    behavior being a 25% quiescent fraction for galaxies 
    residing in Milky Way halos, and 75% for cluster galaxies. 

    Since `SmHmBinarySFR` does not discriminate between centrals and satellites 
    in the SFR assignment, this model is physically unrealistic and is 
    included here primarily for demonstration purposes. 

    Parameters 
    ----------
    prim_haloprop_key : string, optional  
        String giving the column name of the primary halo property governing 
        the galaxy propery being modeled.  
        Default is set in the `~halotools.empirical_models.model_defaults` module. 

    smhm_model : object, optional  
        Sub-class of `~halotools.empirical_models.smhm_components.PrimGalpropModel` governing 
        the stellar-to-halo-mass relation. Default is `Moster13SmHm`. 

    scatter_level : float, optional  
        Constant amount of scatter in stellar mass, in dex. Default is 0.2. 

    redshift : float, optional 
        Redshift of the halo hosting the galaxy. Used to evaluate the 
        stellar-to-halo-mass relation. Default is set in `~halotools.sim_manager.sim_defaults`. 

    sfr_abcissa : array, optional  
        Values of the primary halo property at which the quiescent fraction is specified. 
        Default is [12, 15], in accord with the default True value for ``logparam``. 

    sfr_ordinates : array, optional  
        Values of the quiescent fraction when evaluated at the input abcissa. 
        Default is [0.25, 0.75]

    logparam : bool, optional 
        If set to True, the interpolation will be done 
        in the base-10 logarithm of the primary halo property, 
        rather than linearly. Default is True. 

    threshold : float, optional  
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

    >>> fake_snapshot = FakeSim()
    >>> model.populate_mock(snapshot = fake_snapshot) # doctest: +SKIP

    """

    sfr_model = sfr_components.BinaryGalpropInterpolModel(
        galprop_key='quiescent', prim_haloprop_key=prim_haloprop_key, 
        abcissa=sfr_abcissa, ordinates=sfr_ordinates, logparam=logparam, **kwargs)

    sm_model = smhm_components.Behroozi10SmHm(
        prim_haloprop_key=prim_haloprop_key, redshift=redshift, 
        scatter_abcissa = [12], scatter_ordinates = [scatter_level], **kwargs)

    blueprint = {sm_model.galprop_key: sm_model, sfr_model.galprop_key: sfr_model}

    if 'threshold' in kwargs.keys():
        galaxy_selection_func = lambda x: x['stellar_mass'] > kwargs['threshold']
        model = model_factories.SubhaloModelFactory(blueprint, 
            galaxy_selection_func=galaxy_selection_func)
    else:
        model = model_factories.SubhaloModelFactory(blueprint)

    return model

def Hearin15(central_assembias_strength = 1, 
    central_assembias_strength_abcissa = [1e12], 
    satellite_assembias_strength = 0.2, 
    satellite_assembias_strength_abcissa = [1e12], 
    **kwargs):
    """ 
    HOD-style model in which central and satellite occupations statistics are assembly-biased. 

    Parameters 
    ----------
    threshold : float, optional
        Stellar mass threshold of the mock galaxy sample. 
        Default value is specified in the `~halotools.empirical_models.model_defaults` module.

    sec_haloprop_key : string, optional  
        String giving the column name of the secondary halo property modulating 
        the occupation statistics of the galaxies. 
        Default value is specified in the `~halotools.empirical_models.model_defaults` module.

    central_assembias_strength : float or list, optional 
        Fraction or list of fractions between -1 and 1 defining 
        the assembly bias correlation strength. Default is a constant strength of 0.5. 

    central_assembias_strength_abcissa : list, optional 
        Values of the primary halo property at which the assembly bias strength is specified. 
        Default is a constant strength of 0.5. 

    satellite_assembias_strength : float or list, optional 
        Fraction or list of fractions between -1 and 1 defining 
        the assembly bias correlation strength. Default is a constant strength of 0.5. 

    satellite_assembias_strength_abcissa : list, optional 
        Values of the primary halo property at which the assembly bias strength is specified. 
        Default is a constant strength of 0.5. 

    split : float, optional 
        Fraction between 0 and 1 defining how 
        we split halos into two groupings based on 
        their conditional secondary percentiles. 
        Default is 0.5 for a constant 50/50 split. 

    redshift : float, optional  
        Default is set in the `~halotools.sim_manager.sim_defaults` module. 

    concentration_binning : tuple, optional 
        Three-element tuple. The first entry will be the minimum 
        value of the concentration in the lookup table for the satellite NFW profile, 
        the second entry the maximum, the third entry 
        the linear spacing of the grid. 
        Default is set in `~halotools.empirical_models.model_defaults`.  
        If high-precision is not required, the lookup tables will build much faster if 
        ``concentration_binning`` is set to (1, 25, 0.5).

    Examples 
    --------
    >>> model = Hearin15()

    To use our model to populate a simulation with mock galaxies, we only need to 
    load a snapshot into memory and call the built-in ``populate_mock`` method:

    >>> model.populate_mock() # doctest: +SKIP

    """     
    ##############################
    ### Build the occupation model
    if central_assembias_strength == 0:
        cen_ab_component = hoc.Leauthaud11Cens(**kwargs)
    else:
        cen_ab_component = hoc.AssembiasLeauthaud11Cens(
            assembias_strength = central_assembias_strength, 
            assembias_strength_abcissa = central_assembias_strength_abcissa, 
            **kwargs)
    cen_model_dict = {}
    cen_model_dict['occupation'] = cen_ab_component

    # Build the profile model
    cen_profile = TrivialPhaseSpace(**kwargs)
    cen_model_dict['profile'] = cen_profile

    ##############################
    ### Build the occupation model
    if satellite_assembias_strength == 0:
        sat_ab_component = hoc.Leauthaud11Sats(**kwargs)
    else:
        sat_ab_component = hoc.AssembiasLeauthaud11Sats(
            assembias_strength = satellite_assembias_strength, 
            assembias_strength_abcissa = satellite_assembias_strength_abcissa, 
            **kwargs)
        # There is no need for a redundant new_haloprop_func_dict 
        # if this is already possessed by the central model
        if hasattr(cen_ab_component, 'new_haloprop_func_dict'):
            del sat_ab_component.new_haloprop_func_dict

    sat_model_dict = {}
    sat_model_dict['occupation'] = sat_ab_component

    # Build the profile model
    sat_profile = NFWPhaseSpace(**kwargs) 
    sat_profile._suppress_repeated_param_warning = True   
    sat_model_dict['profile'] = sat_profile

    model_blueprint = {'centrals': cen_model_dict, 'satellites': sat_model_dict}
    composite_model = model_factories.HodModelFactory(model_blueprint)
    return composite_model


def Campbell15(
    prim_haloprop_key = model_defaults.default_smhm_haloprop, 
    sec_galprop_key = 'ssfr', sec_haloprop_key = 'halo_vpeak', 
    smhm_model=smhm_components.Moster13SmHm, 
    scatter_level = 0.2, 
    redshift = sim_defaults.default_redshift, **kwargs):
    """ Conditional abundance matching model based on Campbell et al. (2015). 

    Parameters
    -----------
    prim_haloprop_key : string, optional  
        String giving the column name of the primary halo property governing 
        the galaxy propery being modeled.  
        Default is set in the `~halotools.empirical_models.model_defaults` module. 

    sec_haloprop_key : string, optional  
        Column name of the subhalo property that CAM models as 
        being correlated with ``galprop_key`` at fixed ``prim_galprop_key``. 
        Default is ``vpeak``. 

    sec_galprop_key : string, optional  
        Column name such as ``gr_color`` or ``ssfr`` 
        of the secondary galaxy property being modeled. 
        Can be any column of ``input_galaxy_table`` other than 
        ``prim_galprop_key``. Default is ``ssfr``. 

    input_galaxy_table : data table, optional  
        Astropy Table object storing the input galaxy population 
        upon which the CAM model is based.  
        Default behavior is to use `~halotools.sim_manager.FakeMock`. 

    prim_galprop_bins : array, optional  
        Array used to bin ``input_galaxy_table`` by ``prim_galprop_key``. 
        Default is 15 bins logarithmically spaced between 
        :math:`10^{8}M_{\odot}` and :math:`10^{12}M_{\odot}`. 

    smhm_model : object, optional  
        Sub-class of `~halotools.empirical_models.smhm_components.PrimGalpropModel` governing 
        the stellar-to-halo-mass relation. Default is `Moster13SmHm`. 

    scatter_level : float, optional  
        Constant amount of scatter in dex in ``prim_galprop_key`` 
        at fixed ``prim_haloprop_key``. Default is 0.2. 

    redshift : float, optional 
        Redshift of the halo hosting the galaxy. Used to evaluate the 
        stellar-to-halo-mass relation. Default is set in `~halotools.sim_manager.sim_defaults`. 

    correlation_strength : float or array, optional  
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

    correlation_strength_abcissa : float or array, optional  
        Specifies the value of ``prim_galprop_key`` at which 
        the input ``correlation_strength`` applies. ``correlation_strength_abcissa`` 
        need only be specified if a ``correlation_strength`` array is passed. 
        Intermediary values of the correlation strength at values 
        between the abcissa are solved for by spline interpolation. 
        Default is constant maximum (positive) correlation strength of 1. 

    threshold : float, optional  
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
    `~halotools.sim_manager.HaloCatalog` class. 

    >>> fake_snapshot = FakeSim()
    >>> model.populate_mock(snapshot = fake_snapshot) # doctest: +SKIP

    We can easily build alternative versions of models and mocks by calling the 
    `Campbell15` function with different arguments:

    >>> model_with_scatter = Campbell15(correlation_strength = 0.8, sec_haloprop_key = 'halo_zhalf')
    >>> model_with_scatter.populate_mock(snapshot = fake_snapshot) # doctest: +SKIP


    """

    stellar_mass_model = smhm_model(
        prim_haloprop_key=prim_haloprop_key, redshift=redshift, 
        scatter_abcissa = [12], scatter_ordinates = [scatter_level])

    fake_mock = FakeMock(approximate_ngals = 1e5)
    input_galaxy_table = fake_mock.galaxy_table
    prim_galprop_bins = np.logspace(8, 12, num=15)    

    ssfr_model = ConditionalAbunMatch(input_galaxy_table=input_galaxy_table, 
                                      prim_galprop_key=stellar_mass_model.galprop_key, 
                                      galprop_key=sec_galprop_key, 
                                      sec_haloprop_key=sec_haloprop_key, 
                                      prim_galprop_bins=prim_galprop_bins, 
                                      **kwargs)
    blueprint = (
        {stellar_mass_model.galprop_key: stellar_mass_model, 
        ssfr_model.galprop_key: ssfr_model}
        )

    if 'threshold' in kwargs.keys():
        galaxy_selection_func = lambda x: x['stellar_mass'] > kwargs['threshold']
        model = model_factories.SubhaloModelFactory(blueprint, 
            galaxy_selection_func=galaxy_selection_func)
    else:
        model = model_factories.SubhaloModelFactory(blueprint)

    return model


def Tinker13(threshold = model_defaults.default_stellar_mass_threshold, 
    central_velocity_bias = False, satellite_velocity_bias = False, **kwargs):
    """
    """
    cen_key = 'centrals'
    cen_model_dict = {}
    # Build the occupation model
    occu_cen_model = hoc.Tinker13Cens(threshold = threshold, **kwargs)
    occu_cen_model._suppress_repeated_param_warning = True
    cen_model_dict['occupation'] = occu_cen_model
    # Build the profile model
    
    cen_profile = TrivialPhaseSpace(velocity_bias = central_velocity_bias, **kwargs)

    cen_model_dict['profile'] = cen_profile
    
    sat_key1 = 'quiescent_satellites'
    sat_model_dict1 = {}
    # Build the occupation model
    occu_sat_model1 = hoc.Tinker13QuiescentSats(threshold = threshold, **kwargs)
    sat_model_dict1['occupation'] = occu_sat_model1
    # Build the profile model
    sat_profile1 = NFWPhaseSpace(velocity_bias = satellite_velocity_bias, 
                                 concentration_binning = (1, 35, 1), **kwargs)    
    sat_model_dict1['profile'] = sat_profile1

    sat_key2 = 'active_satellites'
    sat_model_dict2 = {}
    # Build the occupation model
    occu_sat_model2 = hoc.Tinker13ActiveSats(threshold = threshold, **kwargs)
    sat_model_dict2['occupation'] = occu_sat_model2
    # Build the profile model
    sat_profile2 = NFWPhaseSpace(velocity_bias = satellite_velocity_bias, 
                                 concentration_binning = (1, 35, 1), **kwargs)  
    del sat_profile2.new_haloprop_func_dict
    sat_model_dict2['profile'] = sat_profile2
    
    blueprint = {cen_key: cen_model_dict, 
                 sat_key1: sat_model_dict1, 
                 sat_key2: sat_model_dict2}
    
    return model_factories.HodModelFactory(blueprint)











