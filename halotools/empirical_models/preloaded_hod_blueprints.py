# -*- coding: utf-8 -*-
"""

Module containing some commonly used composite HOD model blueprints.

"""

from . import model_defaults, mock_factories, smhm_components
from . import hod_components as hoc
from . import gal_prof_factory as gpf
from . import halo_prof_components as hpc
from . import gal_prof_components as gpc

from ..sim_manager import sim_defaults

__all__ = ['Zheng07_blueprint', 'Leauthaud11_blueprint']


def Zheng07_blueprint(threshold = model_defaults.default_luminosity_threshold, **kwargs):
    """ Blueprint for the simplest pre-loaded HOD model. 
    There are two populations, 
    centrals and satellites, with occupation statistics, 
    positions and velocities based on Kravtsov et al. (2004). 

    Documentation of the test suite of this blueprint can be found at 
    `~halotools.empirical_models.test_empirical_models.test_Zheng07_blueprint`

    Parameters 
    ----------
    threshold : float, optional 
        Luminosity threshold of the galaxy sample being modeled. 

    Returns 
    -------
    model_blueprint : dict 
        Dictionary containing instructions for how to build the model. 
        When model_blueprint is passed to `~halotools.empirical_models.HodModelFactory`, 
        the factory returns the Zheng07 model object. 

    Examples 
    --------
    >>> from halotools.empirical_models import preloaded_hod_blueprints
    >>> blueprint = preloaded_hod_blueprints.Zheng07_blueprint()
    >>> blueprint  = preloaded_hod_blueprints.Zheng07_blueprint(threshold = -19)
    """     

    ### Build model for centrals
    cen_key = 'centrals'
    cen_model_dict = {}
    # Build the occupation model
    occu_cen_model = hoc.Zheng07Cens(gal_type=cen_key, 
        threshold = threshold)
    cen_model_dict['occupation'] = occu_cen_model
    # Build the profile model
    
    cen_profile = gpf.IsotropicGalProf(
        gal_type=cen_key, halo_prof_model=hpc.TrivialProfile)

    cen_model_dict['profile'] = cen_profile

    ### Build model for satellites
    sat_key = 'satellites'
    sat_model_dict = {}
    # Build the occupation model
    occu_sat_model = hoc.Zheng07Sats(gal_type=sat_key, 
        threshold = threshold)
    sat_model_dict['occupation'] = occu_sat_model
    # Build the profile model
    sat_profile = gpf.IsotropicGalProf(
        gal_type=sat_key, halo_prof_model=hpc.NFWProfile)
    sat_model_dict['profile'] = sat_profile

    model_blueprint = {
        occu_cen_model.gal_type : cen_model_dict,
        occu_sat_model.gal_type : sat_model_dict, 
        'mock_factory' : mock_factories.HodMockFactory
        }

    return model_blueprint


def Leauthaud11_blueprint(threshold = model_defaults.default_stellar_mass_threshold, **kwargs):
    """ Blueprint for a Leauthaud11-style HOD model. 

    Parameters 
    ----------
    threshold : float, optional 
        Stellar mass threshold of the galaxy sample being modeled, 
        in ``logarithmic units``. 

    Returns 
    -------
    model_blueprint : dict 
        Dictionary containing instructions for how to build the model. 
        When model_blueprint is passed to `~halotools.empirical_models.HodModelFactory`, 
        the factory returns the Leauthaud11 model object. 

    Examples 
    --------
    >>> from halotools.empirical_models import preloaded_hod_blueprints
    >>> blueprint = preloaded_hod_blueprints.Leauthaud11_blueprint()
    >>> blueprint  = preloaded_hod_blueprints.Leauthaud11_blueprint(threshold = 11.25)
    """     

    ### Build model for centrals
    cen_key = 'centrals'
    cen_model_dict = {}
    # Build the occupation model
    occu_cen_model = hoc.Leauthaud11Cens(gal_type=cen_key, 
        threshold = threshold)
    cen_model_dict['occupation'] = occu_cen_model
    # Build the profile model
    
    cen_profile = gpf.IsotropicGalProf(
        gal_type=cen_key, halo_prof_model=hpc.TrivialProfile)

    cen_model_dict['profile'] = cen_profile

    ### Build model for satellites
    sat_key = 'satellites'
    sat_model_dict = {}
    # Build the occupation model
    occu_sat_model = hoc.Leauthaud11Sats(gal_type=sat_key, 
        threshold = threshold)
    sat_model_dict['occupation'] = occu_sat_model
    # Build the profile model
    sat_profile = gpf.IsotropicGalProf(
        gal_type=sat_key, halo_prof_model=hpc.NFWProfile)
    sat_model_dict['profile'] = sat_profile

    model_blueprint = {
        occu_cen_model.gal_type : cen_model_dict,
        occu_sat_model.gal_type : sat_model_dict, 
        'mock_factory' : mock_factories.HodMockFactory
        }

    return model_blueprint


def Zentner15_blueprint(threshold = model_defaults.default_stellar_mass_threshold, 
    smhm_model=smhm_components.Moster13SmHm, 
    prim_haloprop_key=model_defaults.prim_haloprop_key, 
    sec_haloprop_key=model_defaults.sec_haloprop_key,
    redshift = sim_defaults.default_redshift, 
    **kwargs):
    """ 

    Parameters 
    ----------
    threshold : float, optional keyword argument
        Stellar mass threshold of the mock galaxy sample. 
        Default value is specified in the `~halotools.empirical_models.model_defaults` module.

    smhm_model : object, optional keyword argument 
        Sub-class of `~halotools.empirical_models.smhm_components.PrimGalpropModel` governing 
        the stellar-to-halo-mass relation. Default is `Moster13SmHm`. 

    prim_haloprop_key : string, optional keyword argument 
        String giving the column name of the primary halo property governing 
        the occupation statistics of gal_type galaxies. 
        Default value is specified in the `~halotools.empirical_models.model_defaults` module.

    sec_haloprop_key : string, optional keyword argument 
        String giving the column name of the secondary halo property modulating 
        the occupation statistics of the galaxies. 
        Default value is specified in the `~halotools.empirical_models.model_defaults` module.

    redshift : float, optional keyword argument 
        Redshift of the stellar-to-halo-mass relation. Default is 0. 

    Returns 
    -------
    model_blueprint : dict 
        Dictionary containing instructions for how to build the model. 
        When model_blueprint is passed to `~halotools.empirical_models.HodModelFactory`, 
        the factory returns the Zentner15 model object. 

    """     

    ### Build model for centrals
    cen_key = 'centrals'
    cen_model_dict = {}
    # Build the occupation model
    occu_cen_model = hoc.Leauthaud11Cens(
        gal_type=cen_key, 
        threshold = threshold, 
        smhm_model = smhm_model, 
        prim_haloprop_key = prim_haloprop_key, 
        redshift = redshift
        )
    cen_model_dict['occupation'] = occu_cen_model
    # Build the profile model
    
    cen_profile = gpf.IsotropicGalProf(
        gal_type=cen_key, halo_prof_model=hpc.TrivialProfile)

    cen_model_dict['profile'] = cen_profile

"""
    ### Build model for satellites
    sat_key = 'satellites'
    sat_model_dict = {}
    # Build the occupation model
    occu_sat_model = hoc.Zheng07Sats(gal_type=sat_key, 
        threshold = threshold)
    sat_model_dict['occupation'] = occu_sat_model
    # Build the profile model
    sat_profile = gpf.IsotropicGalProf(
        gal_type=sat_key, halo_prof_model=hpc.NFWProfile)
    sat_model_dict['profile'] = sat_profile

    model_blueprint = {
        occu_cen_model.gal_type : cen_model_dict,
        occu_sat_model.gal_type : sat_model_dict, 
        'mock_factory' : mock_factories.HodMockFactory
        }
"""

    return model_blueprint












