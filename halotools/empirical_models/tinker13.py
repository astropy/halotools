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
from ..custom_exceptions import *

from astropy.extern import six
from abc import ABCMeta, abstractmethod, abstractproperty
import warnings

from .hod_components import OccupationComponent

class Tinker13Cens(OccupationComponent):
    """ HOD-style model for a central galaxy occupation that derives from 
    two distinct active/quiescent stellar-to-halo-mass relations. 
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
            quiescent_key = key + '_quiescent'
            self.param_dict[active_key] = value
            self.param_dict[quiescent_key] = value

        self.sfr_designation_key = 'sfr_designation'

        self.publications = ['arXiv:1308.2974', 'arXiv:1103.2077', 'arXiv:1104.0928']

        # The _methods_to_inherit determines which methods will be directly callable 
        # by the composite model built by the HodModelFactory
        # Here we are overriding this attribute that is normally defined 
        # in the OccupationComponent super class
        self._methods_to_inherit = (
            ['mc_occupation', 'mean_occupation', 'mean_occupation_active', 'mean_occupation_quiescent', 
            'mean_stellar_mass_active', 'mean_stellar_mass_quiescent', 
            'mean_log_halo_mass_active', 'mean_log_halo_mass_quiescent']
            )

        # The _mock_generation_calling_sequence determines which methods 
        # will be called during mock population, as well as in what order they will be called
        self._mock_generation_calling_sequence = ['mc_sfr_designation', 'mc_occupation']
        self._galprop_dtypes_to_allocate = np.dtype([
            ('halo_num_'+ self.gal_type, 'i4'), 
            ('central_sfr_designation', object), 
            ])

    def quiescent_central_fraction(self, **kwargs):
        """
        """
        pass

    def mc_sfr_designation(self, **kwargs):
        """
        """
        quiescent_fraction = self.quiescent_central_fraction(**kwargs)

        if 'seed' in kwargs:
            np.random.seed(seed=kwargs['seed'])
        mc_generator = np.random.random(custom_len(quiescent_fraction))

        result = np.where(mc_generator < quiescent_fraction, 'quiescent', 'active')
        if 'halo_table' in kwargs:
            kwargs['halo_table'][central_sfr_designation] = result
        return result

    def mean_occupation(self, **kwargs):
        """
        """
        if 'halo_table' in kwargs:
            halo_table = kwargs['halo_table']
            try:
                prim_haloprop = halo_table[self.prim_haloprop_key]
            except KeyError:
                msg = ("The ``halo_table`` passed as a keyword argument to the mean_occupation method\n"
                    "does not have the requested ``%s`` key")
                raise HalotoolsError(msg % self.prim_haloprop_key)
            try:
                sfr_designation = halo_table[self.sfr_designation_key]
            except KeyError:
                msg = ("The ``halo_table`` passed as a keyword argument to the mean_occupation method\n"
                    "does not have the requested ``%s`` key")
                raise HalotoolsError(msg % self.sfr_designation_key)
        else:
            try:
                prim_haloprop = kwargs['prim_haloprop']
                sfr_designation = kwargs['sfr_designation']
            except KeyError:
                msg = ("If not passing a ``halo_table`` keyword argument to the mean_occupation method,\n"
                    "you must pass both ``prim_haloprop`` and ``sfr_designation`` keyword arguments")

        result = np.zeros(custom_len(prim_haloprop))

        quiescent_central_idx = np.where(sfr_designation == 'quiescent')[0]
        result[quiescent_central_idx] = self.mean_occupation_quiescent(
            prim_haloprop = prim_haloprop[quiescent_central_idx])        

        active_central_idx = np.invert(quiescent_central_idx)
        result[active_central_idx] = self.mean_occupation_active(
            prim_haloprop = prim_haloprop[active_central_idx])        

        return result


    def mean_occupation_active(self, **kwargs):
        """ 
        """
        self._update_smhm_param_dict('active')

        logmstar = np.log10(self.smhm_model.mean_stellar_mass(
            redshift = self.redshift, **kwargs))
        logscatter = math.sqrt(2)*self.smhm_model.mean_scatter(**kwargs)

        mean_ncen = 0.5*(1.0 - 
            erf((self.threshold - logmstar)/logscatter))
        mean_ncen *= (1. - self.quiescent_central_fraction(**kwargs))

        return mean_ncen

    def mean_occupation_quiescent(self, **kwargs):
        """ 
        """
        self._update_smhm_param_dict('quiescent')

        logmstar = np.log10(self.smhm_model.mean_stellar_mass(
            redshift = self.redshift, **kwargs))
        logscatter = math.sqrt(2)*self.smhm_model.mean_scatter(**kwargs)

        mean_ncen = 0.5*(1.0 - 
            erf((self.threshold - logmstar)/logscatter))
        mean_ncen *= self.quiescent_central_fraction(**kwargs)

        return mean_ncen

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












