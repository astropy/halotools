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

from .scatter_models import LogNormalScatterModel
from .smhm_model_template import PrimGalpropModel

from .. import model_defaults
from .. import model_helpers as model_helpers

from ...utils.array_utils import custom_len
from ...sim_manager import sim_defaults 


__all__ = ['Moster13SmHm']

class Moster13SmHm(PrimGalpropModel):
    """ Stellar-to-halo-mass relation based on 
    Moster et al. (2013), arXiv:1205.5807. 
    """

    def __init__(self, **kwargs):
        """
        Parameters 
        ----------
        prim_haloprop_key : string, optional  
            String giving the column name of the primary halo property governing stellar mass. 
            Default is set in the `~halotools.empirical_models.model_defaults` module. 

        scatter_model : object, optional  
            Class governing stochasticity of stellar mass. Default scatter is log-normal, 
            implemented by the `LogNormalScatterModel` class. 

        scatter_abcissa : array_like, optional  
            Array of values giving the abcissa at which
            the level of scatter will be specified by the input ordinates.
            Default behavior will result in constant scatter at a level set in the 
            `~halotools.empirical_models.model_defaults` module. 

        scatter_ordinates : array_like, optional  
            Array of values defining the level of scatter at the input abcissa.
            Default behavior will result in constant scatter at a level set in the 
            `~halotools.empirical_models.model_defaults` module. 

        new_haloprop_func_dict : function object, optional  
            Dictionary of function objects used to create additional halo properties 
            that may be needed by the model component. 
            Used strictly by the `MockFactory` during call to the `process_halo_catalog` method. 
            Each dict key of ``new_haloprop_func_dict`` will 
            be the name of a new column of the halo catalog; each dict value is a function 
            object that returns a length-N numpy array when passed a length-N Astropy table 
            via the ``halo_table`` keyword argument. 
            The input ``model`` model object has its own new_haloprop_func_dict; 
            if the keyword argument ``new_haloprop_func_dict`` passed to `MockFactory` 
            contains a key that already appears in the ``new_haloprop_func_dict`` bound to 
            ``model``, and exception will be raised. 

        """

        super(Moster13SmHm, self).__init__(
            galprop_key='stellar_mass', **kwargs)

        self.publications = ['arXiv:0903.4682', 'arXiv:1205.5807']

    def mean_stellar_mass(self, **kwargs):
        """ Return the stellar mass of a central galaxy as a function 
        of the input halo_table.  

        Parameters 
        ----------
        prim_haloprop : array, optional  
            Array of mass-like variable upon which occupation statistics are based. 
            If ``prim_haloprop`` is not passed, then ``halo_table`` keyword argument must be passed. 

        halo_table : object, optional  
            Data table storing halo catalog. 
            If ``halo_table`` is not passed, then ``prim_haloprop`` keyword argument must be passed. 

        redshift : float or array, optional 
            Redshift of the halo hosting the galaxy. 
            Default is set in `~halotools.sim_manager.sim_defaults`. 
            If passing an array, must be of the same length as 
            the ``prim_haloprop`` or ``halo_table`` argument. 

        Returns 
        -------
        mstar : array_like 
            Array containing stellar masses living in the input halo_table. 
        """

        # Retrieve the array storing the mass-like variable
        if 'halo_table' in kwargs.keys():
            mass = kwargs['halo_table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in kwargs.keys():
            mass = kwargs['prim_haloprop']
        else:
            raise KeyError("Must pass one of the following keyword arguments to mean_occupation:\n"
                "``halo_table`` or ``prim_haloprop``")

        if 'redshift' in kwargs.keys():
            redshift = kwargs['redshift']
        elif hasattr(self, 'redshift'):
            redshift = self.redshift
        else:
            redshift = sim_defaults.default_redshift

        # compute the parameter values that apply to the input redshift
        a = 1./(1+redshift)

        m1 = self.param_dict['m10'] + self.param_dict['m11']*(1-a)
        n = self.param_dict['n10'] + self.param_dict['n11']*(1-a)
        beta = self.param_dict['beta10'] + self.param_dict['beta11']*(1-a)
        gamma = self.param_dict['gamma10'] + self.param_dict['gamma11']*(1-a)

        # Calculate each term contributing to Eqn 2
        norm = 2.*n*mass
        m_by_m1 = mass/(10.**m1)
        denom_term1 = m_by_m1**(-beta)
        denom_term2 = m_by_m1**gamma

        mstar = norm / (denom_term1 + denom_term2)
        return mstar

    def retrieve_default_param_dict(self):
        """ Method returns a dictionary of all model parameters 
        set to the values in Table 1 of Moster et al. (2013). 

        Returns 
        -------
        d : dict 
            Dictionary containing parameter values. 
        """

        d = {
        'm10': 11.590, 
        'm11': 1.195, 
        'n10': 0.0351, 
        'n11': -0.0247, 
        'beta10': 1.376, 
        'beta11': -0.826, 
        'gamma10': 0.608, 
        'gamma11': 0.329
        }

        return d








