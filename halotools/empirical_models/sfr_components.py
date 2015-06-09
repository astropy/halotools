# -*- coding: utf-8 -*-
"""

"""

__all__ = ['BinaryGalpropInterpolModel']

from functools import partial
from copy import copy
import numpy as np
from scipy.interpolate import UnivariateSpline as spline

import model_defaults
from ..utils.array_utils import array_like_length as custom_len
import occupation_helpers as occuhelp

from astropy.extern import six
from abc import ABCMeta, abstractmethod, abstractproperty
import warnings


@six.add_metaclass(ABCMeta)
class BinaryGalpropModel(object):
    """
    Container class for any component model of a binary-valued galaxy property. 

    """

    def __init__(self,  prim_haloprop_key = 'mvir', **kwargs):
        """ 
        Parameters 
        ----------
        galprop_key : array, keyword argument 
            String giving the name of galaxy property being assigned a binary value. 
            Default is 'quiescent'. 

        prim_haloprop_key : string, optional keyword argument 
            String giving the key name used to access the primary halo property 
            from an input halo or galaxy catalog. Default is 'mvir'. 

        gal_type : string, keyword argument
            Name of the galaxy population being modeled, e.g., 'centrals'. 
            This is only necessary to specify in cases where 
            the `BinaryGalpropInterpolModel` instance is part of a composite model, 
            with multiple population types. Default is None. 

        .. automethod:: _mc_galprop
        """

        self.prim_haloprop_key = prim_haloprop_key
        self.galprop_key = kwargs['galprop_key']
        self.gal_type = kwargs['gal_type']

        setattr(self, 'mc_'+self.galprop_key, self._mc_galprop)

        # Enforce the requirement that sub-classes have been configured properly
        required_method_name = 'mean_'+self.galprop_key+'_fraction'
        if not hasattr(self, required_method_name):
            raise SyntaxError("Any sub-class of BinaryGalpropModel must "
                "implement a method named %s " % required_method_name)


    def _mc_galprop(self, **kwargs):
        """
        Monte Carlo realization of the binary-valued galprop. 

        Parameters 
        ----------
        prim_haloprop : array_like, optional keyword argument
            Array of primary halo property, e.g., `mvir`, 
            at which the galprop fraction is being computed. 
            Functionality is equivalent to using the mass_like keyword argument. 

        halos : table, optional keyword argument
            Astropy Table containing a halo catalog. 
            If the ``halos`` keyword argument is passed, 
            ``self.prim_haloprop_key`` must be a column of the halo catalog. 

        galaxy_table : table, optional keyword argument
            Astropy Table containing a galaxy catalog. 
            If the ``galaxy_table`` keyword argument is passed, 
            ``self.prim_haloprop_key`` must be a column of the galaxy catalog. 

        input_param_dict : dict, optional keyword argument 
            If passed, the model will first update 
            its param_dict with input_param_dict, altering the behavior of the model. 

        Returns 
        -------
        mc_galprop : array_like
            Boolean value of whether or not the mock galaxy is posses the galprop, 
            where the Monte Carlo realization is drawn from a nearest-integer 
            distribution determined by the user-defined `mean_galprop_fraction` method. 

        """
        mean_func = getattr(self, 'mean_'+self.galprop_key+'_fraction')
        mean_galprop_fraction = mean_func(**kwargs)
        mc_generator = np.random.random(custom_len(mean_galprop_fraction))
        mc_galprop = np.zeros_like(mean_galprop_fraction, dtype=bool)
        mc_galprop[mc_generator<mean_galprop_fraction] = True
        return mc_galprop


class BinaryGalpropInterpolModel(BinaryGalpropModel):
    """
    Component model for any binary-valued galaxy property 
    whose assignment is determined by interpolating between points on a grid. 

    One example of a model of this class could be used to help build is 
    Tinker et al. (2013), arXiv:1308.2974. 
    In this model, a galaxy is either active or quiescent, 
    and the quiescent fraction is purely a function of halo mass, with
    separate parameters for centrals and satellites. The value of the quiescent 
    fraction is computed by interpolating between a grid of values of mass. 
    `BinaryGalpropInterpolModel` is quite flexible, and can be used as a template for 
    any binary-valued galaxy property whatsoever. See the examples below for usage instructions. 

    """

    def __init__(self, abcissa = [12, 15], ordinates = [0.25, 0.75], 
        logparam=True, prim_haloprop_key = 'mvir',
        interpol_method='spline', **kwargs):
        """ 
        Parameters 
        ----------
        galprop_key : array, required keyword argument
            String giving the name of galaxy property being assigned a binary value. 

        abcissa : array, optional keyword argument 
            Values of the primary halo property at which the galprop fraction is specified. 
            Default is [12, 15], in accord with the default True value for ``logparam``. 

        ordinates : array, optional keyword argument 
            Values of the galprop fraction when evaluated at the input abcissa. 
            Default is [0.25, 0.75]

        logparam : bool, optional keyword argument
            If set to True, the interpolation will be done 
            in the base-10 logarithm of the primary halo property, 
            rather than linearly. Default is True. 

        prim_haloprop_key : string, optional keyword argument 
            String giving the key name used to access the primary halo property 
            from an input halo or galaxy catalog. Default is 'mvir'. 

        gal_type : string, optional keyword argument
            Name of the galaxy population being modeled, e.g., 'centrals'. 
            This is only necessary to specify in cases where 
            the `BinaryGalpropInterpolModel` instance is part of a composite model, 
            with multiple population types. Default is None. 

        interpol_method : string, optional keyword argument 
            Keyword specifying how `mean_galprop_fraction` 
            evaluates input values of the primary halo property. 
            The default spline option interpolates the 
            model's abcissa and ordinates. 
            The polynomial option uses the unique, degree N polynomial 
            passing through the ordinates, where N is the number of supplied ordinates. 

        input_spline_degree : int, optional keyword argument
            Degree of the spline interpolation for the case of interpol_method='spline'. 
            If there are k abcissa values specifying the model, input_spline_degree 
            is ensured to never exceed k-1, nor exceed 5. Default is 3. 

        Examples
        -----------       
        Suppose we wish to construct a model for whether a central galaxy is 
        star-forming or quiescent. We want to set the quiescent fraction to 1/3 
        for Milky Way-type centrals (:math:`M_{\\mathrm{vir}}=10^{12}M_{\odot}`), 
        and 90% for massive cluster centrals (:math:`M_{\\mathrm{vir}}=10^{15}M_{\odot}`).

        >>> abcissa, ordinates = [12, 15], [1/3., 0.9]
        >>> cen_quiescent_model = BinaryGalpropInterpolModel(galprop_key='quiescent', abcissa=abcissa, ordinates=ordinates, prim_haloprop_key='mvir', gal_type='cens')

        The ``cen_quiescent_model`` has a built-in method that computes the quiescent fraction 
        as a function of mass:

        >>> quiescent_frac = cen_quiescent_model.mean_quiescent_fraction(prim_haloprop=1e12)

        There is also a built-in method to return a Monte Carlo realization of quiescent/star-forming galaxies:

        >>> masses = np.logspace(10, 15, num=100)
        >>> quiescent_realization = cen_quiescent_model.mc_quiescent(prim_haloprop=masses)

        Now ``quiescent_realization`` is a boolean-valued array of the same length as ``masses``. 
        Entries of ``quiescent_realization`` that are ``True`` correspond to central galaxies that are quiescent. 

        Here is another example of how you could use `BinaryGalpropInterpolModel` 
        to construct a simple model for satellite morphology, where the early- vs. late-type 
        of the satellite depends on :math:`V_{\\mathrm{peak}}` value of the host halo

        >>> sat_morphology_model = BinaryGalpropInterpolModel(galprop_key='late_type', abcissa=abcissa, ordinates=ordinates, prim_haloprop_key='vpeak_host', gal_type='sats')
        >>> vmax_array = np.logspace(2, 3, num=100)
        >>> morphology_realization = sat_morphology_model.mc_late_type(prim_haloprop=vmax_array)

        .. automethod:: _mean_galprop_fraction
        .. automethod:: _mc_galprop
        """

        self.galprop_key = kwargs['galprop_key']
        self._interpol_method = interpol_method
        self._logparam = logparam
        self._abcissa = abcissa
        self._ordinates = ordinates
        self.prim_haloprop_key = prim_haloprop_key
        setattr(self, self.galprop_key+'_abcissa', self._abcissa)

        if 'gal_type' in kwargs.keys():
            self.gal_type = kwargs['gal_type']
            self._abcissa_key = self.galprop_key+'_abcissa_'+self.gal_type
            self._ordinates_key_prefix = self.galprop_key+'_ordinates_'+self.gal_type
        else:
            self._abcissa_key = self.galprop_key+'_abcissa'
            self._ordinates_key_prefix = self.galprop_key+'_ordinates'

        self._build_param_dict()

        if self._interpol_method=='spline':
            if 'input_spline_degree' in kwargs.keys():
                self._input_spine_degree = kwargs['input_spline_degree']
            else:
                self._input_spline_degree = 3
            scipy_maxdegree = 5
            self._spline_degree = np.min(
                [scipy_maxdegree, self._input_spline_degree, 
                custom_len(self._abcissa)-1])

        setattr(self, 'mean_'+self.galprop_key+'_fraction', self._mean_galprop_fraction)
        setattr(self, 'mc_'+self.galprop_key, self._mc_galprop)

    def _build_param_dict(self):

        self._ordinates_keys = [self._ordinates_key_prefix + '_param' + str(i+1) for i in range(custom_len(self._abcissa))]
        self.param_dict = {key:value for key, value in zip(self._ordinates_keys, self._ordinates)}

    def _mean_galprop_fraction(self, **kwargs):
        """
        Expectation value of the galprop for galaxies living in the input halos.  

        Parameters 
        ----------
        mass_like : array_like, optional keyword argument
            Array of primary halo property, e.g., `mvir`, 
            at which the galprop fraction is being computed. 

        prim_haloprop : array_like, optional keyword argument
            Array of primary halo property, e.g., `mvir`, 
            at which the galprop fraction is being computed. 
            Functionality is equivalent to using the mass_like keyword argument. 

        halos : table, optional keyword argument
            Astropy Table containing a halo catalog. 
            If the ``halos`` keyword argument is passed, 
            ``self.prim_haloprop_key`` must be a column of the halo catalog. 

        galaxy_table : table, optional keyword argument
            Astropy Table containing a galaxy catalog. 
            If the ``galaxy_table`` keyword argument is passed, 
            ``self.prim_haloprop_key`` must be a column of the galaxy catalog. 

        input_param_dict : dict, optional keyword argument 
            If passed, the model will first update 
            its param_dict with input_param_dict, altering the behavior of the model. 

        Returns 
        -------
        mean_galprop_fraction : array_like
            Values of the galprop fraction evaluated at the input primary halo properties. 

        """

        # If requested, update the parameter dictionary defining the behavior of the model
        if 'input_param_dict' in kwargs.keys():
            input_param_dict = kwargs['input_param_dict']
            for key in self.param_dict.keys():
                if key in input_param_dict.keys():
                    self.param_dict[key] = input_param_dict[key]

        if 'mass_like' in kwargs.keys():
            mass_like = kwargs['mass_like']
        elif 'prim_haloprop' in kwargs.keys():
            mass_like = kwargs['prim_haloprop']
        elif 'halos' in kwargs.keys():
            mass_like = kwargs['halos'][self.prim_haloprop_key]
        elif 'galaxy_table' in kwargs.keys():
            mass_like = kwargs['galaxy_table'][self.prim_haloprop_key]            
        else:
            raise KeyError("Must pass mean_galprop_fraction one of the "
                "following keyword arguments:\n'mass_like', 'prim_haloprop', 'halos', or 'galaxy_table'\n"
                "Received none of these.")

        if self._logparam is True:
            mass_like = np.log10(mass_like)

        model_ordinates = [self.param_dict[ordinate_key] for ordinate_key in self._ordinates_keys]
        if self._interpol_method=='polynomial':
            mean_galprop_fraction = occuhelp.polynomial_from_table(
                self._abcissa,model_ordinates,mass_like)
        elif self._interpol_method=='spline':
            spline_function = occuhelp.custom_spline(
                self._abcissa,model_ordinates,
                    k=self._spline_degree)
            mean_galprop_fraction = spline_function(mass_like)
        else:
            raise IOError("Input interpol_method must be 'polynomial' or 'spline'.")

        # Enforce boundary conditions 
        mean_galprop_fraction[mean_galprop_fraction<0]=0
        mean_galprop_fraction[mean_galprop_fraction>1]=1

        return mean_galprop_fraction

    def _mc_galprop(self, **kwargs):
        """
        Monte Carlo realization of the binary-valued galprop. 

        Parameters 
        ----------
        mass_like : array_like, optional keyword argument
            Array of primary halo property, e.g., `mvir`, 
            at which the galprop fraction is being computed. 

        prim_haloprop : array_like, optional keyword argument
            Array of primary halo property, e.g., `mvir`, 
            at which the galprop fraction is being computed. 
            Functionality is equivalent to using the mass_like keyword argument. 

        halos : table, optional keyword argument
            Astropy Table containing a halo catalog. 
            If the ``halos`` keyword argument is passed, 
            ``self.prim_haloprop_key`` must be a column of the halo catalog. 

        galaxy_table : table, optional keyword argument
            Astropy Table containing a galaxy catalog. 
            If the ``galaxy_table`` keyword argument is passed, 
            ``self.prim_haloprop_key`` must be a column of the galaxy catalog. 

        input_param_dict : dict, optional keyword argument 
            If passed, the model will first update 
            its param_dict with input_param_dict, altering the behavior of the model. 

        Returns 
        -------
        mc_galprop_fraction : array_like
            Boolean value of whether or not the mock galaxy is posses the galprop, 
            where the Monte Carlo realization is drawn from a nearest-integer 
            distribution determined by `_mean_galprop_fraction`. 

        """
        mean_galprop_fraction = self._mean_galprop_fraction(**kwargs)
        mc_generator = np.random.random(custom_len(mean_galprop_fraction))
        mc_galprop = np.zeros_like(mean_galprop_fraction, dtype=bool)
        mc_galprop[mc_generator<mean_galprop_fraction] = True
        return mc_galprop









