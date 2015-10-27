# -*- coding: utf-8 -*-
"""

Module containing some commonly used composite HOD models.

"""
from __future__ import (
    division, print_function, absolute_import, unicode_literals)

import numpy as np

from .. import factories, model_defaults
from ..occupation_models import hod_components as hoc
from ..occupation_models import zheng07_components
from ..occupation_models import leauthaud11_components 
from ..occupation_models import tinker13_components 

from ..smhm_models import Moster13SmHm, Behroozi10SmHm
from ..sfr_models import BinaryGalpropInterpolModel
from ..phase_space_models import NFWPhaseSpace, TrivialPhaseSpace
from ..abunmatch import ConditionalAbunMatch

from ...sim_manager import FakeMock, FakeSim, sim_defaults


__all__ = ['SmHmBinarySFR', 'Campbell15']


def SmHmBinarySFR(
    prim_haloprop_key = model_defaults.default_smhm_haloprop, 
    smhm_model=Moster13SmHm, 
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
        Instance of `~halotools.empirical_models.factories.SubhaloModelFactory`

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

    sfr_model = BinaryGalpropInterpolModel(
        galprop_key='quiescent', prim_haloprop_key=prim_haloprop_key, 
        abcissa=sfr_abcissa, ordinates=sfr_ordinates, logparam=logparam, **kwargs)

    sm_model = Behroozi10SmHm(
        prim_haloprop_key=prim_haloprop_key, redshift=redshift, 
        scatter_abcissa = [12], scatter_ordinates = [scatter_level], **kwargs)

    blueprint = {sm_model.galprop_key: sm_model, sfr_model.galprop_key: sfr_model}

    if 'threshold' in kwargs.keys():
        galaxy_selection_func = lambda x: x['stellar_mass'] > kwargs['threshold']
        model = factories.SubhaloModelFactory(blueprint, 
            galaxy_selection_func=galaxy_selection_func)
    else:
        model = factories.SubhaloModelFactory(blueprint)

    return model


def Campbell15(
    prim_haloprop_key = model_defaults.default_smhm_haloprop, 
    sec_galprop_key = 'ssfr', sec_haloprop_key = 'halo_vpeak', 
    smhm_model = Moster13SmHm, 
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
        model = factories.SubhaloModelFactory(blueprint, 
            galaxy_selection_func=galaxy_selection_func)
    else:
        model = factories.SubhaloModelFactory(blueprint)

    return model


