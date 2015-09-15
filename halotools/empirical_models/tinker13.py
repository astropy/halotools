# -*- coding: utf-8 -*-
"""
This module contains various component features used by 
HOD-style models of the galaxy-halo connection. 

"""

__all__ = (['OccupationComponent','Zheng07Cens','Zheng07Sats', 
    'Leauthaud11Cens', 'Leauthaud11Sats', 'AssembiasZheng07Cens', 'AssembiasZheng07Sats', 
    'AssembiasLeauthaud11Cens', 'AssembiasLeauthaud11Sats']
    )

from functools import partial
from copy import copy
import numpy as np
import math
from scipy.special import erf 
from scipy.stats import poisson
from scipy.optimize import brentq
from scipy.interpolate import InterpolatedUnivariateSpline as spline

from . import model_defaults, model_helpers, smhm_components
from .assembias import HeavisideAssembias

from ..utils.array_utils import custom_len
from ..utils.table_utils import compute_conditional_percentiles
from ..  import sim_manager
from ..custom_exceptions import HalotoolsModelInputError

from astropy.extern import six
from abc import ABCMeta, abstractmethod, abstractproperty
import warnings

from .hod_components import *

class Tinker13Cens(OccupationComponent):
    """ HOD-style model for a central galaxy occupation that derives from 
    two distinct active/passive stellar-to-halo-mass relations. 
    """
    def __init__(self, threshold = model_defaults.default_stellar_mass_threshold, 
        prim_haloprop_key=model_defaults.prim_haloprop_key,
        redshift = sim_manager.sim_defaults.default_redshift, **kwargs):
        """
        Parameters 
        ----------
        threshold : float, optional 
            Stellar mass threshold of the mock galaxy sample in h=1 solar mass units. 
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        prim_haloprop_key : string, optional  
            String giving the column name of the primary halo property governing 
            the occupation statistics of gal_type galaxies. 
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        redshift : float, optional  
            Redshift of the stellar-to-halo-mass relation. 
            Default is set in `~halotools.sim_manager.sim_defaults`. 

        """
        upper_bound = 1.0

        # Call the super class constructor, which binds all the 
        # arguments to the instance.  
        super(Tinker13Cens, self).__init__(
            gal_type='centrals', threshold=threshold, 
            upper_bound=upper_bound, 
            prim_haloprop_key = prim_haloprop_key, 
            **kwargs)
        self.redshift = redshift

        self.smhm_model = smhm_components.Behroozi10SmHm(
            prim_haloprop_key = prim_haloprop_key, **kwargs)

        for key, value in self.smhm_model.param_dict.iteritems():
            active_key = key + '_active'
            passive_key = key + '_passive'
            self.param_dict[active_key] = value
            self.param_dict[passive_key] = value

        self.publications = ['arXiv:1308.2974', 'arXiv:1103.2077', 'arXiv:1104.0928']


    def passive_central_fraction(self, **kwargs):
        """
        """
        pass

    def mean_occupation(self, **kwargs):
        """ 
        """
        pass

    def mean_stellar_mass_active(self, **kwargs):
        """ 
        """
        self._update_smhm_param_dict('active')
        return self.smhm_model.mean_stellar_mass(redshift = self.redshift, **kwargs)

    def mean_stellar_mass_quiescent(self, **kwargs):
        """ 
        """
        self._update_smhm_param_dict('quiescent')
        return self.smhm_model.mean_stellar_mass(redshift = self.redshift, **kwargs)

    def mean_log_halo_mass_active(self, log_stellar_mass):
        """ 
        """
        self._update_smhm_param_dict('active')
        return self.smhm_model.mean_log_halo_mass(log_stellar_mass, 
            redshift = self.redshift)

    def mean_log_halo_mass_quiescent(self, log_stellar_mass):
        """ 
        """
        self._update_smhm_param_dict('quiescent')
        return self.smhm_model.mean_log_halo_mass(log_stellar_mass, 
            redshift = self.redshift)

    def _update_smhm_param_dict(self, sfr_key):
        for key, value in self.param_dict.iteritems():
            stripped_key = key[[:-len('sfr_key')-1]]
            if stripped_key in self.smhm_model.param_dict:
                self.smhm_model.param_dict[stripped_key] = value 












