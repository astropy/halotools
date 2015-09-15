# -*- coding: utf-8 -*-
"""

"""

__all__ = ['BinaryGalpropModel', 'BinaryGalpropInterpolModel']

from functools import partial
from copy import copy
import numpy as np
from scipy.interpolate import UnivariateSpline as spline

import model_defaults
from ..utils.array_utils import custom_len
import model_helpers as model_helpers

from astropy.extern import six
from abc import ABCMeta, abstractmethod, abstractproperty
import warnings

from ..custom_exceptions import HalotoolsError

@six.add_metaclass(ABCMeta)
class BinaryGalpropModel(object):
    """
    Container class for any component model of a binary-valued galaxy property. 

    """

    def __init__(self, 
        prim_haloprop_key = model_defaults.default_binary_galprop_haloprop,
        **kwargs):
        """
        Parameters 
        ----------
        galprop_key : string, keyword argument 
            Name of the galaxy property being assigned. 

        prim_haloprop_key : string, optional  
            String giving the column name of the primary halo property governing 
            the galaxy propery being modeled.  
            Default is set in the `~halotools.empirical_models.model_defaults` module. 

        new_haloprop_func_dict : function object, optional  
            Dictionary of function objects used to create additional halo properties 
            that may be needed by the model component. 
            Used strictly by the `MockFactory` during call to the `process_halo_catalog` method. 
            Each dict key of ``new_haloprop_func_dict`` will 
            be the name of a new column of the halo catalog; each dict value is a function 
            object that returns a length-N numpy array when passed a length-N Astropy table 
            via the ``halos`` keyword argument. 
            The input ``model`` model object has its own new_haloprop_func_dict; 
            if the keyword argument ``new_haloprop_func_dict`` passed to `MockFactory` 
            contains a key that already appears in the ``new_haloprop_func_dict`` bound to 
            ``model``, and exception will be raised. 

        """
        required_kwargs = ['galprop_key']
        model_helpers.bind_required_kwargs(required_kwargs, self, **kwargs)

        self.prim_haloprop_key = prim_haloprop_key

        if 'sec_haloprop_key' in kwargs.keys():
            self.sec_haloprop_key = kwargs['sec_haloprop_key']

        if 'new_haloprop_func_dict' in kwargs.keys():
            self.new_haloprop_func_dict = kwargs['new_haloprop_func_dict']

        # Enforce the requirement that sub-classes have been configured properly
        required_method_name = 'mean_'+self.galprop_key+'_fraction'
        if not hasattr(self, required_method_name):
            raise HalotoolsError("Any sub-class of BinaryGalpropModel must "
                "implement a method named %s " % required_method_name)

        setattr(self, 'mc_'+self.galprop_key, self._mc_galprop)


    def _mc_galprop(self, seed=None, **kwargs):
        """ Return a Monte Carlo realization of the galaxy property 
        based on draws from a nearest-integer distribution. 

        Parameters 
        ----------
        prim_haloprop : array, optional  
            Array of mass-like variable governing the galaxy property. 
            If ``prim_haloprop`` is not passed, then either ``halos`` or ``galaxy_table`` 
            keyword arguments must be passed. 

        halos : object, optional  
            Data table storing halo catalog. 
            If ``halos`` is not passed, then either ``prim_haloprop`` or ``galaxy_table`` 
            keyword arguments must be passed. 

        galaxy_table : object, optional  
            Data table storing galaxy catalog. 
            If ``galaxy_table`` is not passed, then either ``prim_haloprop`` or ``halos`` 
            keyword arguments must be passed. 

        seed : int, optional  
            Random number seed used to generate the Monte Carlo realization.
            Default is None. 

        Returns 
        -------
        mc_galprop : array_like 
            Array storing the values of the primary galaxy property 
            of the galaxies living in the input halos. 
        """
        np.random.seed(seed=seed)

        mean_func = getattr(self, 'mean_'+self.galprop_key+'_fraction')
        mean_galprop_fraction = mean_func(**kwargs)
        mc_generator = np.random.random(custom_len(mean_galprop_fraction))
        return np.where(mc_generator < mean_galprop_fraction, True, False)

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

    def __init__(self, galprop_abcissa = [12, 15], galprop_ordinates = [0.25, 0.75], 
        logparam=True, interpol_method='spline', **kwargs):
        """ 
        Parameters 
        ----------
        galprop_key : array, keyword argument
            String giving the name of galaxy property being assigned a binary value. 

        prim_haloprop_key : string, optional  
            String giving the column name of the primary halo property governing 
            stellar mass.  
            Default is set in the `~halotools.empirical_models.model_defaults` module. 

        galprop_abcissa : array, optional  
            Values of the primary halo property at which the galprop fraction is specified. 
            Default is [12, 15], in accord with the default True value for ``logparam``. 

        galprop_ordinates : array, optional  
            Values of the galprop fraction when evaluated at the input abcissa. 
            Default is [0.25, 0.75]

        logparam : bool, optional 
            If set to True, the interpolation will be done 
            in the base-10 logarithm of the primary halo property, 
            rather than linearly. Default is True. 

        interpol_method : string, optional  
            Keyword specifying how `mean_galprop_fraction` 
            evaluates input values of the primary halo property. 
            The default spline option interpolates the 
            model's abcissa and ordinates. 
            The polynomial option uses the unique, degree N polynomial 
            passing through the ordinates, where N is the number of supplied ordinates. 

        input_spline_degree : int, optional 
            Degree of the spline interpolation for the case of interpol_method='spline'. 
            If there are k abcissa values specifying the model, input_spline_degree 
            is ensured to never exceed k-1, nor exceed 5. Default is 3. 

        Examples
        -----------       
        Suppose we wish to construct a model for whether a central galaxy is 
        star-forming or quiescent. We want to set the quiescent fraction to 1/3 
        for Milky Way-type centrals (:math:`M_{\\mathrm{vir}}=10^{12}M_{\odot}`), 
        and 90% for massive cluster centrals (:math:`M_{\\mathrm{vir}}=10^{15}M_{\odot}`). 
        We can use the `BinaryGalpropInterpolModel` to implement this as follows:

        >>> abcissa, ordinates = [12, 15], [1/3., 0.9]
        >>> cen_quiescent_model = BinaryGalpropInterpolModel(galprop_key='quiescent', galprop_abcissa=abcissa, galprop_ordinates=ordinates, prim_haloprop_key='mvir')

        The ``cen_quiescent_model`` has a built-in method that computes the quiescent fraction 
        as a function of mass:

        >>> quiescent_frac = cen_quiescent_model.mean_quiescent_fraction(prim_haloprop =1e12)

        There is also a built-in method to return a Monte Carlo realization of quiescent/star-forming galaxies:

        >>> masses = np.logspace(10, 15, num=100)
        >>> quiescent_realization = cen_quiescent_model.mc_quiescent(prim_haloprop = masses)

        Now ``quiescent_realization`` is a boolean-valued array of the same length as ``masses``. 
        Entries of ``quiescent_realization`` that are ``True`` correspond to central galaxies that are quiescent. 

        Here is another example of how you could use `BinaryGalpropInterpolModel` 
        to construct a simple model for satellite morphology, where the early- vs. late-type 
        of the satellite depends on :math:`V_{\\mathrm{peak}}` value of the host halo

        >>> sat_morphology_model = BinaryGalpropInterpolModel(galprop_key='late_type', galprop_abcissa=abcissa, galprop_ordinates=ordinates, prim_haloprop_key='vpeak_host')
        >>> vmax_array = np.logspace(2, 3, num=100)
        >>> morphology_realization = sat_morphology_model.mc_late_type(prim_haloprop =vmax_array)

        .. automethod:: _mean_galprop_fraction
        """
        try:
            galprop_key = kwargs['galprop_key']
        except KeyError:
            raise HalotoolsError("\nAll sub-classes of BinaryGalpropInterpolModel must pass "
                "a ``galprop_key`` keyword argument to the constructor\n")

        setattr(self, 'mean_'+galprop_key+'_fraction', self._mean_galprop_fraction)
        super(BinaryGalpropInterpolModel, self).__init__(**kwargs)

        self._interpol_method = interpol_method
        self._logparam = logparam
        self._abcissa = galprop_abcissa
        self._ordinates = galprop_ordinates

        if self._interpol_method=='spline':
            if 'input_spline_degree' in kwargs.keys():
                self._input_spine_degree = kwargs['input_spline_degree']
            else:
                self._input_spline_degree = 3
            scipy_maxdegree = 5
            self._spline_degree = np.min(
                [scipy_maxdegree, self._input_spline_degree, 
                custom_len(self._abcissa)-1])

        self._abcissa_key = self.galprop_key+'_abcissa'
        self._ordinates_key_prefix = self.galprop_key+'_ordinates'
        self._build_param_dict()

        setattr(self, self.galprop_key+'_abcissa', self._abcissa)

    def _build_param_dict(self):

        self._ordinates_keys = [self._ordinates_key_prefix + '_param' + str(i+1) for i in range(custom_len(self._abcissa))]
        self.param_dict = {key:value for key, value in zip(self._ordinates_keys, self._ordinates)}

    def _mean_galprop_fraction(self, **kwargs):
        """
        Expectation value of the galprop for galaxies living in the input halos.  

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
        mean_galprop_fraction : array_like
            Values of the galprop fraction evaluated at the input primary halo properties. 

        """
        # Retrieve the array storing the mass-like variable
        if 'halo_table' in kwargs.keys():
            prim_haloprop = kwargs['halo_table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in kwargs.keys():
            prim_haloprop = kwargs['prim_haloprop']
        else:
            raise KeyError("Must pass one of the following keyword arguments to mean_occupation:\n"
                "``halo_table`` or  ``prim_haloprop``")

        if self._logparam is True:
            prim_haloprop = np.log10(prim_haloprop)

        # Update self._abcissa, in case the user has changed it
        self._abcissa = getattr(self, self.galprop_key+'_abcissa')

        model_ordinates = [self.param_dict[ordinate_key] for ordinate_key in self._ordinates_keys]
        if self._interpol_method=='polynomial':
            mean_galprop_fraction = model_helpers.polynomial_from_table(
                self._abcissa, model_ordinates, prim_haloprop)
        elif self._interpol_method=='spline':
            spline_function = model_helpers.custom_spline(
                self._abcissa, model_ordinates,
                    k=self._spline_degree)
            mean_galprop_fraction = spline_function(prim_haloprop)
        else:
            raise HalotoolsError("Input interpol_method must be 'polynomial' or 'spline'.")

        # Enforce boundary conditions 
        mean_galprop_fraction[mean_galprop_fraction<0]=0
        mean_galprop_fraction[mean_galprop_fraction>1]=1

        return mean_galprop_fraction

