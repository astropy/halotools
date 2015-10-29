# -*- coding: utf-8 -*-
"""

Module containing the HOD-style composite model based on Leauthaud et al. (2011).

"""
from __future__ import (
    division, print_function, absolute_import, unicode_literals)

import numpy as np

from ... import model_defaults
from ...occupation_models import leauthaud11_components 
from ... import factories
from ...smhm_models import Behroozi10SmHm
from ...phase_space_models import NFWPhaseSpace, TrivialPhaseSpace

from ....sim_manager import FakeSim, sim_defaults


__all__ = ['return_leauthaud11_model_dictionary']
    
def return_leauthaud11_model_dictionary(threshold = model_defaults.default_stellar_mass_threshold, 
    central_velocity_bias = False, satellite_velocity_bias = False, **kwargs):
    """ Blueprint for an HOD-style based on Leauthaud et al. (2011), arXiv:1103.2077. 
    The behavior of this model is governed by an assumed underlying stellar-to-halo-mass relation. 

    There are two populations, centrals and satellites. 
    Central occupation statistics are given by a nearest integer distribution 
    with first moment given by an ``erf`` function; the class governing this 
    behavior is `~halotools.empirical_models.occupation_components.Leauthaud11Cens`. 
    Central galaxies are assumed to reside at the exact center of the host halo; 
    the class governing this behavior is `~halotools.empirical_models.TrivialPhaseSpace`. 

    Satellite occupation statistics are given by a Poisson distribution 
    with first moment given by a power law that has been truncated at the low-mass end; 
    the class governing this behavior is `~halotools.empirical_models.occupation_components.Leauthaud11Sats`; 
    satellites in this model follow an (unbiased) NFW profile, as governed by the 
    `~halotools.empirical_models.NFWPhaseSpace` class. 

    This composite model is built by the `~halotools.empirical_models.factories.HodModelFactory`, 
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
    model_blueprint : dict 
        Dictionary passed to `~halotools.empirical_models.factories.HodModelFactory`

    Examples 
    --------

    >>> model_blueprint = return_leauthaud11_model_dictionary()
    >>> model_instance = factories.HodModelFactory(**model_blueprint)

    The default settings are set in the `~halotools.empirical_models.model_defaults` module. 
    To load a model based on a different threshold and redshift:

    >>> model_blueprint = return_leauthaud11_model_dictionary(threshold = 11, redshift = 1)
    >>> model_instance = factories.HodModelFactory(**model_blueprint)

    For this model, you can also use the following syntax candy, 
    which accomplishes the same task as the above:

    >>> model_instance = factories.HodModelFactory('leauthaud11', threshold = 11, redshift = 1)

    As with all instances of the `~halotools.empirical_models.HodModelFactory`, 
    you can populate a mock with one line of code: 

    >>> model_instance.populate_mock(simname = 'bolshoi', redshift = 1) # doctest: +SKIP

    """
    ### Build model for centrals
    # Build the occupation model
    centrals_occupation = leauthaud11_components.Leauthaud11Cens(threshold = threshold, **kwargs)
    centrals_occupation._suppress_repeated_param_warning = True
    # Build the profile model
    
    centrals_profile = TrivialPhaseSpace(velocity_bias = central_velocity_bias, **kwargs)

    ### Build model for satellites
    # Build the occupation model
    satellites_occupation = leauthaud11_components.Leauthaud11Sats(threshold = threshold, **kwargs)
    # Build the profile model
    satellites_profile = NFWPhaseSpace(velocity_bias = satellite_velocity_bias, **kwargs)    


    return ({'centrals_occupation': centrals_occupation, 
        'centrals_profile': centrals_profile, 
        'satellites_occupation': satellites_occupation, 
        'satellites_profile': satellites_profile})

    

