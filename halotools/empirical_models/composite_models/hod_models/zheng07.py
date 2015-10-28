# -*- coding: utf-8 -*-
"""

Module containing the HOD-style composite model published in Zheng et al. (2007)

"""
from __future__ import (
    division, print_function, absolute_import, unicode_literals)

import numpy as np

from ... import factories, model_defaults
from ...occupation_models import zheng07_components
from ...phase_space_models import NFWPhaseSpace, TrivialPhaseSpace

from ....sim_manager import FakeSim


__all__ = ['Zheng07']

def Zheng07(threshold = model_defaults.default_luminosity_threshold, **kwargs):
    """ Simple HOD-style based on Zheng et al. (2007), arXiv:0703457. 

    There are two populations, centrals and satellites. 
    Central occupation statistics are given by a nearest integer distribution 
    with first moment given by an ``erf`` function; the class governing this 
    behavior is `~halotools.empirical_models.occupation_components.Zheng07Cens`. 
    Central galaxies are assumed to reside at the exact center of the host halo; 
    the class governing this behavior is `~halotools.empirical_models.TrivialPhaseSpace`. 

    Satellite occupation statistics are given by a Poisson distribution 
    with first moment given by a power law that has been truncated at the low-mass end; 
    the class governing this behavior is `~halotools.empirical_models.occupation_components.Zheng07Sats`; 
    satellites in this model follow an (unbiased) NFW profile, as governed by the 
    `~halotools.empirical_models.NFWPhaseSpace` class. 

    This composite model was built by the `~halotools.empirical_models.factories.HodModelFactory`.

    Parameters 
    ----------
    threshold : float, optional 
        Luminosity threshold of the galaxy sample being modeled. 
        Default is set in the `~halotools.empirical_models.model_defaults` module. 

    Returns 
    -------
    model : object 
        Instance of `~halotools.empirical_models.factories.HodModelFactory`

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

    ####################################
    ### Build subpopulation blueprint for centrals
    subpopulation_blueprint_centrals = {}

    # Build the `occupation` feature
    centrals_occupation = zheng07_components.Zheng07Cens(threshold = threshold, **kwargs)

    # Build the `profile` feature
    centrals_profile = TrivialPhaseSpace(**kwargs)

    ####################################
    ### Build subpopulation blueprint for satellites
    subpopulation_blueprint_satellites = {}

    # Build the occupation model
    satellites_occupation = zheng07_components.Zheng07Sats(threshold = threshold, **kwargs)
    satellites_occupation._suppress_repeated_param_warning = True

    # Build the profile model
    satellites_profile = NFWPhaseSpace(**kwargs)    


    composite_model = factories.HodModelFactory(centrals_occupation = centrals_occupation, 
        centrals_profile = centrals_profile, satellites_occupation = satellites_occupation, 
        satellites_profile = satellites_profile)

    return composite_model




    
