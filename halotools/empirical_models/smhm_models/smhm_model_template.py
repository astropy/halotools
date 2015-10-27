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

from .. import model_defaults
from .. import model_helpers as model_helpers

from ...utils.array_utils import custom_len
from ...sim_manager import sim_defaults 


__all__ = ['PrimGalpropModel']


@six.add_metaclass(ABCMeta)
class PrimGalpropModel(model_helpers.GalPropModel):
    """ Abstract container class for models connecting halo_table to their primary
    galaxy property, e.g., stellar mass or luminosity. 
    """

    def __init__(self, galprop_key = 'stellar_mass', 
        prim_haloprop_key = model_defaults.default_smhm_haloprop, 
        scatter_model = LogNormalScatterModel, 
        **kwargs):
        """
        Parameters 
        ----------
        galprop_key : string, optional  
            Name of the galaxy property being assigned. Default is ``stellar mass``, 
            though another common case may be ``luminosity``. 

        prim_haloprop_key : string, optional  
            String giving the column name of the primary halo property governing 
            stellar mass.  
            Default is set in the `~halotools.empirical_models.model_defaults` module. 

        scatter_model : object, optional  
            Class governing stochasticity of stellar mass. Default scatter is log-normal, 
            implemented by the `LogNormalScatterModel` class. 

        redshift : float, optional  
            Redshift of the stellar-to-halo-mass relation. Default is 0. 

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
        self.galprop_key = galprop_key
        self.prim_haloprop_key = prim_haloprop_key

        if 'redshift' in kwargs.keys():
            self.redshift = kwargs['redshift']

        if 'new_haloprop_func_dict' in kwargs.keys():
            self.new_haloprop_func_dict = kwargs['new_haloprop_func_dict']

        self.scatter_model = scatter_model(
            prim_haloprop_key=self.prim_haloprop_key, **kwargs)

        self._build_param_dict(**kwargs)

        # Enforce the requirement that sub-classes have been configured properly
        required_method_name = 'mean_'+self.galprop_key
        if not hasattr(self, required_method_name):
            raise SyntaxError("Any sub-class of PrimGalpropModel must "
                "implement a method named %s " % required_method_name)

        # If the sub-class did not implement their own Monte Carlo method mc_galprop, 
        # then use _mc_galprop and give it the usual name
        if not hasattr(self, 'mc_'+self.galprop_key):
            setattr(self, 'mc_'+self.galprop_key, self._mc_galprop)

        super(PrimGalpropModel, self).__init__(galprop_key=self.galprop_key)

        # The _mock_generation_calling_sequence determines which methods 
        # will be called during mock population, as well as in what order they will be called
        self._mock_generation_calling_sequence = ['mc_stellar_mass']
        key = str(self.galprop_key)
        self._galprop_dtypes_to_allocate = np.dtype([(key, 'f4')])


    def mean_scatter(self, **kwargs):
        """ Use the ``param_dict`` of `PrimGalpropModel` to update the ``param_dict`` 
        of the scatter model, and then call the `mean_scatter` method of 
        the scatter model. 
        """
        for key in self.scatter_model.param_dict.keys():
            self.scatter_model.param_dict[key] = self.param_dict[key]

        return self.scatter_model.mean_scatter(**kwargs)

    def scatter_realization(self, **kwargs):
        """ Use the ``param_dict`` of `PrimGalpropModel` to update the ``param_dict`` 
        of the scatter model, and then call the `scatter_realization` method of 
        the scatter model. 
        """
        for key in self.scatter_model.param_dict.keys():
            self.scatter_model.param_dict[key] = self.param_dict[key]

        return self.scatter_model.scatter_realization(**kwargs)

    def _build_param_dict(self, **kwargs):
        """ Method combines the parameter dictionaries of the 
        smhm model and the scatter model. 
        """

        if hasattr(self, 'retrieve_default_param_dict'):
            self.param_dict = self.retrieve_default_param_dict()
        else:
            self.param_dict = {}

        scatter_param_dict = self.scatter_model.param_dict

        for key, value in scatter_param_dict.iteritems():
            self.param_dict[key] = value

    def _mc_galprop(self, include_scatter = True, **kwargs):
        """ Return the prim_galprop of the galaxies living in the input halo_table. 

        Parameters 
        ----------
        prim_haloprop : array, optional  
            Array of mass-like variable upon which occupation statistics are based. 
            If ``prim_haloprop`` is not passed, then ``halo_table`` keyword argument must be passed. 

        halo_table : object, optional  
            Data table storing halo catalog. 
            If ``halo_table`` is not passed, then ``prim_haloprop`` keyword argument must be passed. 

        redshift : float, optional 
            Redshift of the halo hosting the galaxy. 

        include_scatter : boolean, optional  
            Determines whether or not the scatter model is applied to add stochasticity 
            to the galaxy property assignment. Default is True. 
            If False, model is purely deterministic, and the behavior is determined 
            by the ``mean_galprop`` method of the sub-class. 

        Returns 
        -------
        prim_galprop : array_like 
            Array storing the values of the primary galaxy property 
            of the galaxies living in the input halo_table. 
        """

        # Interpret the inputs to determine the appropriate redshift
        if 'redshift' not in kwargs.keys():
            if hasattr(self, 'redshift'):
                kwargs['redshift'] = self.redshift
            else:
                warn("\nThe PrimGalpropModel class was not instantiated with a redshift,\n"
                "nor was a redshift passed to the primary function call.\n"
                "Choosing the default redshift z = %.2f\n" % sim_defaults.default_redshift)
                kwargs['redshift'] = sim_defaults.default_redshift

        prim_galprop_func = getattr(self, 'mean_'+self.galprop_key)
        galprop_first_moment = prim_galprop_func(**kwargs)

        if include_scatter is False:
            return galprop_first_moment
        else:
            log10_galprop_with_scatter = (
                np.log10(galprop_first_moment) + 
                self.scatter_realization(**kwargs)
                )
            return 10.**log10_galprop_with_scatter


