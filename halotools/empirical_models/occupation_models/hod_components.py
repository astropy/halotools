# -*- coding: utf-8 -*-
"""
This module contains various component features used by 
HOD-style models of the galaxy-halo connection. 

"""

__all__ = (['OccupationComponent']
    )

from functools import partial
from copy import copy
import numpy as np
import math
from scipy.special import erf 
from scipy.stats import poisson
from scipy.optimize import brentq
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from astropy.extern import six
from abc import ABCMeta, abstractmethod, abstractproperty
import warnings

from .. import model_defaults, model_helpers
from ..smhm_models import smhm_components
from ..assembias_models import HeavisideAssembias
from ..model_helpers import bounds_enforcing_decorator_factory

from ...utils.array_utils import custom_len
from ...utils.table_utils import compute_conditional_percentiles
from ...  import sim_manager
from ...custom_exceptions import *

@six.add_metaclass(ABCMeta)
class OccupationComponent(object):
    """ Abstract base class of any occupation model. 
    Functionality is mostly trivial. 
    The sole purpose of the base class is to 
    standardize the attributes and methods 
    required of any HOD-style model for halo occupation statistics. 
    """
    def __init__(self, **kwargs):
        """ 
        Parameters 
        ----------
        gal_type : string, keyword argument 
            Name of the galaxy population whose occupation statistics is being modeled. 

        threshold : float, keyword argument
            Threshold value defining the selection function of the galaxy population 
            being modeled. Typically refers to absolute magnitude or stellar mass. 

        upper_occupation_bound : float, keyword argument
            Upper bound on the number of gal_type galaxies per halo. 
            The only currently supported values are unity or infinity. 

        prim_haloprop_key : string, keyword argument 
            String giving the column name of the primary halo property governing 
            the occupation statistics of gal_type galaxies, e.g., ``halo_mvir``. 
        """
        required_kwargs = ['gal_type', 'threshold', 'prim_haloprop_key']
        model_helpers.bind_required_kwargs(required_kwargs, self, **kwargs)

        self._upper_occupation_bound = kwargs['upper_occupation_bound']
        self._lower_occupation_bound = 0

        self.param_dict = {}

        # Enforce the requirement that sub-classes have been configured properly
        required_method_name = 'mean_occupation'
        if not hasattr(self, required_method_name):
            raise SyntaxError("Any sub-class of OccupationComponent must "
                "implement a method named %s " % required_method_name)

        # The _methods_to_inherit determines which methods will be directly callable 
        # by the composite model built by the HodModelFactory
        try:
            self._methods_to_inherit.extend(['mc_occupation', 'mean_occupation'])
        except AttributeError:
            self._methods_to_inherit = ['mc_occupation', 'mean_occupation']

        # The _attrs_to_inherit determines which methods will be directly bound  
        # to the composite model built by the HodModelFactory
        try:
            self._attrs_to_inherit.append('threshold')
        except AttributeError:
            self._attrs_to_inherit = ['threshold']

        if not hasattr(self, 'publications'):
            self.publications = []

        # The _mock_generation_calling_sequence determines which methods 
        # will be called during mock population, as well as in what order they will be called
        self._mock_generation_calling_sequence = ['mc_occupation']
        self._galprop_dtypes_to_allocate = np.dtype([('halo_num_'+ self.gal_type, 'i4')])

    def mc_occupation(self, seed=None, **kwargs):
        """ Method to generate Monte Carlo realizations of the abundance of galaxies. 

        Parameters
        ----------        
        prim_haloprop : array, optional  
            Array of mass-like variable upon which occupation statistics are based. 
            If ``prim_haloprop`` is not passed, then ``halo_table`` keyword argument must be passed. 

        halo_table : object, optional  
            Data table storing halo catalog. 
            If ``halo_table`` is not passed, then ``prim_haloprop`` keyword argument must be passed. 

        seed : int, optional  
            Random number seed used to generate the Monte Carlo realization. 
            Default is None. 

        Returns
        -------
        mc_abundance : array
            Integer array giving the number of galaxies in each of the input halo_table.     
        """ 
        first_occupation_moment = self.mean_occupation(**kwargs)
        if self._upper_occupation_bound == 1:
            return self._nearest_integer_distribution(first_occupation_moment, seed=seed, **kwargs)
        elif self._upper_occupation_bound == float("inf"):
            return self._poisson_distribution(first_occupation_moment, seed=seed, **kwargs)
        else:
            raise KeyError("The only permissible values of upper_occupation_bound for instances "
                "of OccupationComponent are unity and infinity.")

    def _nearest_integer_distribution(self, first_occupation_moment, seed=None, **kwargs):
        """ Nearest-integer distribution used to draw Monte Carlo occupation statistics 
        for central-like populations with only permissible galaxy per halo.

        Parameters 
        ----------
        first_occupation_moment : array
            Array giving the first moment of the occupation distribution function. 

        seed : int, optional  
            Random number seed used to generate the Monte Carlo realization. 
            Default is None. 

        Returns
        -------
        mc_abundance : array
            Integer array giving the number of galaxies in each of the input halo_table. 
        """
        np.random.seed(seed=seed)
        mc_generator = np.random.random(custom_len(first_occupation_moment))

        result = np.where(mc_generator < first_occupation_moment, 1, 0)
        if 'halo_table' in kwargs:
            kwargs['halo_table']['halo_num_'+self.gal_type] = result
        return result

    def _poisson_distribution(self, first_occupation_moment, seed=None, **kwargs):
        """ Poisson distribution used to draw Monte Carlo occupation statistics 
        for satellite-like populations in which per-halo abundances are unbounded. 

        Parameters 
        ----------
        first_occupation_moment : array
            Array giving the first moment of the occupation distribution function. 

        seed : int, optional  
            Random number seed used to generate the Monte Carlo realization. 
            Default is None. 

        Returns
        -------
        mc_abundance : array
            Integer array giving the number of galaxies in each of the input halo_table. 
        """
        np.random.seed(seed=seed)
        # The scipy built-in Poisson number generator raises an exception 
        # if its input is zero, so here we impose a simple workaround
        first_occupation_moment = np.where(first_occupation_moment <=0, 
            model_defaults.default_tiny_poisson_fluctuation, first_occupation_moment)

        result = poisson.rvs(first_occupation_moment)
        if 'halo_table' in kwargs:
            kwargs['halo_table']['halo_num_'+self.gal_type] = result
        return result




