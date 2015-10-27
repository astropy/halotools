# -*- coding: utf-8 -*-
"""
Module containing classes used to model the mapping between 
stellar mass and subhalo_table. 
"""
from __future__ import (
    division, print_function, absolute_import, unicode_literals)

import numpy as np
from scipy.interpolate import UnivariateSpline
from astropy.extern import six
from abc import ABCMeta, abstractmethod, abstractproperty
from astropy import cosmology
from warnings import warn
from functools import partial

from .. import model_defaults
from .. import model_helpers as model_helpers

from ...utils.array_utils import custom_len
from ...sim_manager import sim_defaults 


__all__ = ['LogNormalScatterModel']

class LogNormalScatterModel(object):
    """ Simple model used to generate log-normal scatter 
    in a stellar-to-halo-mass type relation. 

    """

    def __init__(self, 
        prim_haloprop_key=model_defaults.default_smhm_haloprop, 
        **kwargs):
        """
        Parameters 
        ----------
        prim_haloprop_key : string, optional  
            String giving the column name of the primary halo property governing 
            the level of scatter. 
            Default is set in the `~halotools.empirical_models.model_defaults` module. 

        scatter_abcissa : array_like, optional  
            Array of values giving the abcissa at which
            the level of scatter will be specified by the input ordinates.
            Default behavior will result in constant scatter at a level set in the 
            `~halotools.empirical_models.model_defaults` module. 

        scatter_ordinates : array_like, optional  
            Array of values defining the level of scatter at the input abcissa.
            Default behavior will result in constant scatter at a level set in the 
            `~halotools.empirical_models.model_defaults` module. 

        Examples 
        ---------
        >>> scatter_model = LogNormalScatterModel()
        >>> scatter_model = LogNormalScatterModel(prim_haloprop_key='halo_mvir')

        To implement variable scatter, we need to define the level 
        of log-normal scatter at a set of control values 
        of the primary halo property. Here we give an example of a model 
        in which the scatter is 0.3 dex for Milky Way halo_table and 0.1 dex in cluster halo_table:

        >>> scatter_abcissa = [12, 15]
        >>> scatter_ordinates = [0.3, 0.1]
        >>> scatter_model = LogNormalScatterModel(scatter_abcissa=scatter_abcissa, scatter_ordinates=scatter_ordinates)

        """
        
        default_scatter = model_defaults.default_smhm_scatter
        self.prim_haloprop_key = prim_haloprop_key

        if ('scatter_abcissa' in kwargs.keys()) and ('scatter_ordinates' in kwargs.keys()):
            self.abcissa = kwargs['scatter_abcissa']
            self.ordinates = kwargs['scatter_ordinates']
        else:
            self.abcissa = [12]
            self.ordinates = [default_scatter]

        self._initialize_param_dict()

        self._update_interpol()

    def mean_scatter(self, **kwargs):
        """ Return the amount of log-normal scatter that should be added 
        to the galaxy property as a function of the input halo_table. 

        Parameters 
        ----------
        prim_haloprop : array, optional  
            Array of mass-like variable upon which occupation statistics are based. 
            If ``prim_haloprop`` is not passed, then ``halo_table`` keyword argument must be passed. 

        halo_table : object, optional  
            Data table storing halo catalog. 
            If ``halo_table`` is not passed, then ``prim_haloprop`` keyword argument must be passed. 

        Returns 
        -------
        scatter : array_like 
            Array containing the amount of log-normal scatter evaluated 
            at the input halo_table. 
        """
        # Retrieve the array storing the mass-like variable
        if 'halo_table' in kwargs.keys():
            mass = kwargs['halo_table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in kwargs.keys():
            mass = kwargs['prim_haloprop']
        else:
            raise KeyError("Must pass one of the following keyword arguments to mean_occupation:\n"
                "``halo_table`` or ``prim_haloprop``")

        self._update_interpol()

        return self.spline_function(np.log10(mass))

    def scatter_realization(self, seed=None, **kwargs):
        """ Return the amount of log-normal scatter that should be added 
        to the galaxy property as a function of the input halo_table. 

        Parameters 
        ----------
        prim_haloprop : array, optional  
            Array of mass-like variable upon which occupation statistics are based. 
            If ``prim_haloprop`` is not passed, then ``halo_table`` keyword argument must be passed. 

        halo_table : object, optional  
            Data table storing halo catalog. 
            If ``halo_table`` is not passed, then ``prim_haloprop`` keyword argument must be passed. 

        seed : int, optional  
            Random number seed. Default is None. 

        Returns 
        -------
        scatter : array_like 
            Array containing a random variable realization that should be summed 
            with the galaxy property to add scatter.  
        """

        scatter_scale = self.mean_scatter(**kwargs)

        np.random.seed(seed=seed)
            
        return np.random.normal(loc=0, scale=scatter_scale)

    def _update_interpol(self):
        """ Private method that updates the interpolating functon used to 
        define the level of scatter as a function of the input halo_table. 
        If this method is not called after updating ``self.param_dict``, 
        changes in ``self.param_dict`` will not alter the model behavior. 
        """

        scipy_maxdegree = 5
        degree_list = [scipy_maxdegree, custom_len(self.abcissa)-1]
        self.spline_degree = np.min(degree_list)

        self.ordinates = [self.param_dict[self._get_param_key(i)] for i in range(len(self.abcissa))]

        self.spline_function = model_helpers.custom_spline(
            self.abcissa, self.ordinates, k=self.spline_degree)

    def _initialize_param_dict(self):
        """ Private method used to initialize ``self.param_dict``. 
        """
        self.param_dict={}
        for ipar, val in enumerate(self.ordinates):
            key = self._get_param_key(ipar)
            self.param_dict[key] = val

    def _get_param_key(self, ipar):
        """ Private method used to retrieve the key of self.param_dict 
        that corresponds to the appropriately selected i^th ordinate 
        defining the behavior of the scatter model. 
        """
        return 'scatter_model_param'+str(ipar+1)




        

