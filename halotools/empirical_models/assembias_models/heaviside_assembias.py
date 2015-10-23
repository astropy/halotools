# -*- coding: utf-8 -*-
"""
Decorator class for implementing generalized assembly bias
"""

__all__ = ['HeavisideAssembias']

import numpy as np 
from warnings import warn 
from time import time

from .. import model_defaults, model_helpers

from ...utils.array_utils import custom_len
from ...custom_exceptions import HalotoolsError
from ...utils.table_utils import compute_conditional_percentiles

class HeavisideAssembias(object):
    """ Class used as an orthogonal mix-in to introduce 
    assembly-biased behavior into the class whose behavior is being supplemented class.  

    """
    def __init__(self, **kwargs):
        """
        Parameters 
        ----------
        method_name_to_decorate : string 
            Name of the method in the primary class whose behavior is being decorated 

        lower_assembias_bound : float 
            lower bound on the method being decorated with assembly bias 

        upper_assembias_bound : float 
            upper bound on the method being decorated with assembly bias 

        sec_haloprop_key : string, optional 
            String giving the column name of the secondary halo property 
            governing the assembly bias. Must be a key in the halo_table 
            passed to the methods of `HeavisideAssembiasComponent`. 
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        split : float or list, optional 
            Fraction or list of fractions between 0 and 1 defining how 
            we split halos into two groupings based on 
            their conditional secondary percentiles. 
            Default is 0.5 for a constant 50/50 split. 

        split_abcissa : list, optional 
            Values of the primary halo property at which the halos are split as described above in 
            the ``split`` argument. If ``loginterp`` is set to True (the default behavior), 
            the interpolation will be done in the logarithm of the primary halo property. 
            Default is to assume a constant 50/50 split. 

        splitting_model : object, optional 
            Model instance with a method called ``splitting_method_name``  
            used to split the input halos into two types.

        splitting_method_name : string, optional 
            Name of method bound to ``splitting_model`` used to split the input halos into 
            two types. 

        halo_type_tuple : tuple, optional 
            Tuple providing the information about how elements of the input ``halo_table`` 
            have been pre-divided into types. The first tuple entry must be a 
            string giving the column name of the input ``halo_table`` that provides the halo-typing. 
            The second entry gives the value that will be stored in this column for halos 
            that are above the percentile split, the third entry gives the value for halos 
            below the split. 

            If provided, you must ensure that the splitting of the ``halo_table`` 
            was self-consistently performed with the 
            input ``split``, or ``split_abcissa`` and ``split_ordinates``, or 
            ``split_func`` keyword arguments. 

        assembias_strength : float or list, optional 
            Fraction or sequence of fractions between -1 and 1 
            defining the assembly bias correlation strength. 
            Default is 0.5. 

        assembias_strength_abcissa : list, optional 
            Values of the primary halo property at which the assembly bias strength is specified. 
            Default is to assume a constant strength of 0.5. If passing a list, the strength 
            will interpreted at the input ``assembias_strength_abcissa``.
            Default is to assume a constant strength of 0.5. 

        loginterp : bool, optional
            If set to True, the interpolation will be done 
            in the logarithm of the primary halo property, 
            rather than linearly. Default is True. 

        """
        self._interpret_constructor_inputs(**kwargs)

        self._decorate_baseline_method()

        self._bind_new_haloprop_func_dict()


    def _interpret_constructor_inputs(self, loginterp = True, 
        sec_haloprop_key = model_defaults.sec_haloprop_key, **kwargs):
        """
        """
        self._loginterp = loginterp
        self.sec_haloprop_key = sec_haloprop_key

        required_attr_list = ['prim_haloprop_key', 'gal_type']
        for attr in required_attr_list:
            if not hasattr(self, attr):
                msg = ("In order to use the HeavisideAssembias class " 
                    "to decorate your model component with assembly bias, \n"
                    "the component instance must have a %s attribute")
                raise HalotoolsError(msg % attr)

        try:
            self._method_name_to_decorate = kwargs['method_name_to_decorate']
        except KeyError:
            msg = ("The constructor to the HeavisideAssembiasComponent class "
                "must be called with the following keyword arguments:\n" 
                "``%s``")
            raise HalotoolsError(msg % ('_method_name_to_decorate'))

        try:
            lower_bound = float(kwargs['lower_assembias_bound'])
            lower_bound_key = 'lower_bound_' + self._method_name_to_decorate + '_' + self.gal_type 
            setattr(self, lower_bound_key, lower_bound)
            upper_bound = float(kwargs['upper_assembias_bound'])
            upper_bound_key = 'upper_bound_' + self._method_name_to_decorate + '_' + self.gal_type 
            setattr(self, upper_bound_key, upper_bound)
        except KeyError:
            msg = ("The constructor to the HeavisideAssembiasComponent class "
                "must be called with the following keyword arguments:\n" 
                "``%s``, ``%s``")
            raise HalotoolsError(msg % ('lower_assembias_bound', 'upper_assembias_bound'))

        self._set_percentile_splitting(**kwargs)
        self._initialize_assembias_param_dict(**kwargs)

        if 'halo_type_tuple' in kwargs:
            self.halo_type_tuple = kwargs['halo_type_tuple']

    def _set_percentile_splitting(self, split = 0.5, **kwargs):
        """
        Method interprets the arguments passed to the constructor 
        and sets up the interpolation scheme for how halos will be 
        divided into two types as a function of the primary halo property. 
        """

        if 'splitting_model' in kwargs:
            self.splitting_model = kwargs['splitting_model']
            self.ancillary_model_dependencies = ['splitting_model']
            func = getattr(self.splitting_model, kwargs['splitting_method_name'])
            if callable(func):
                self._input_split_func = func
            else:
                raise HalotoolsError("Input ``splitting_model`` must have a callable function "
                    "named ``%s``" % kwargs['splitting_method_name'])
        elif 'split_abcissa' in kwargs.keys():
            if custom_len(kwargs['split_abcissa']) != custom_len(split):
                raise HalotoolsError("``split`` and ``split_abcissa`` must have the same length")
            self._split_abcissa = kwargs['split_abcissa']
            self._split_ordinates = split
        else:
            try:
                self._split_abcissa = [2]
                self._split_ordinates = [split]
            except KeyError:
                msg = ("The _set_percentile_splitting method must at least be called with a ``split``" 
                    "keyword argument, or alternatively ``split`` and ``split_abcissa`` arguments.")
                raise HalotoolsError(msg)

    def _initialize_assembias_param_dict(self, assembias_strength = 0.5, **kwargs):
        """
        """
        if not hasattr(self, 'param_dict'):
            self.param_dict = {}

        # Make sure the code behaves properly whether or not we were passed an iterable
        strength = assembias_strength
        try:
            iterator = iter(strength)
            strength = list(strength)
        except TypeError:
            strength = [strength]

        if 'assembias_strength_abcissa' in kwargs:
            abcissa = kwargs['assembias_strength_abcissa']
            try:
                iterator = iter(abcissa)
                abcissa = list(abcissa)
            except TypeError:
                abcissa = [abcissa]
        else:
            abcissa = [2]

        if custom_len(abcissa) != custom_len(strength):
            raise HalotoolsError("``assembias_strength`` and ``assembias_strength_abcissa`` must have the same length")

        self._assembias_strength_abcissa = abcissa
        for ipar, val in enumerate(strength):
            self.param_dict[self._get_assembias_param_dict_key(ipar)] = val

    def _decorate_baseline_method(self):
        """
        """

        try:
            baseline_method = getattr(self, self._method_name_to_decorate)
            setattr(self, 'baseline_'+self._method_name_to_decorate, 
                baseline_method)
            decorated_method = self.assembias_decorator(baseline_method)
            setattr(self, self._method_name_to_decorate, decorated_method)
        except AttributeError:
            msg = ("The baseline model constructor must be called before "
                "calling the HeavisideAssembias constructor, \n"
                "and the baseline model must have a method named ``%s``")
            raise HalotoolsError(msg % self._method_name_to_decorate)

    def _bind_new_haloprop_func_dict(self):
        """
        """

        def assembias_percentile_calculator(halo_table):
            return compute_conditional_percentiles(
                halo_table = halo_table, 
                prim_haloprop_key = self.prim_haloprop_key, 
                sec_haloprop_key = self.sec_haloprop_key
                )

        key = self.sec_haloprop_key + '_percentile'
        try:
            self.new_haloprop_func_dict[key] = assembias_percentile_calculator
        except AttributeError:
            self.new_haloprop_func_dict = {}
            self.new_haloprop_func_dict[key] = assembias_percentile_calculator

        self._methods_to_inherit.extend(['assembias_strength'])


    def percentile_splitting_function(self, **kwargs):
        """
        Method returns the fraction of halos that are ``type1`` 
        as a function of the input primary halo property. 

        Parameters 
        -----------
        halo_table : object, optional  
            Data table storing halo catalog. 

        Returns 
        -------
        split : float
            Fraction of ``type1`` halos at the input primary halo property. 
        """
        try:
            halo_table = kwargs['halo_table']
            prim_haloprop = halo_table[self.prim_haloprop_key]
        except KeyError:
            msg = ("The ``percentile_splitting_function`` method requires a "
                "``halo_table`` input keyword argument.\n"
                "The input halo_table must have a column with name %s")
            raise HalotoolsError(msg % self.prim_haloprop_key)

        if hasattr(self, '_input_split_func'):
            result = self._input_split_func(halo_table = halo_table)

            if np.any(result < 0):
                msg = ("The input split_func passed to the HeavisideAssembias class"
                    "must not return negative values")
                raise HalotoolsError(msg)
            if np.any(result > 1):
                msg = ("The input split_func passed to the HeavisideAssembias class"
                    "must not return values exceeding unity")
                raise HalotoolsError(msg)

            return result

        elif self._loginterp is True:
            spline_function = model_helpers.custom_spline(
                np.log10(self._split_abcissa), self._split_ordinates)
            result = spline_function(np.log10(prim_haloprop))
        else:
            model_abcissa = self._split_abcissa
            spline_function = model_helpers.custom_spline(
                self._split_abcissa, self._split_ordinates)
            result = spline_function(prim_haloprop)

        result = np.where(result < 0, 0, result)
        result = np.where(result > 1, 1, result)

        return result



    def assembias_strength(self, **kwargs):
        """
        Method returns the strength of assembly bias as a function of the input halos, 
        where the strength varies between -1 and 1. 

        Parameters 
        ----------
        prim_haloprop : array_like 
            Array storing the primary halo property. 

        Returns 
        -------
        strength : array_like 
            Strength of assembly bias as a function of the input halo property. 
        """
        try:
            prim_haloprop = kwargs['prim_haloprop']
        except KeyError:
            raise HalotoolsError("The ``assembias_strength`` method requires a "
                "``prim_haloprop`` input keyword argument")

        model_ordinates = (self.param_dict[self._get_assembias_param_dict_key(ipar)] 
            for ipar in range(len(self._assembias_strength_abcissa)))
        spline_function = model_helpers.custom_spline(
            self._assembias_strength_abcissa, list(model_ordinates))

        if self._loginterp is True:
            result = spline_function(np.log10(prim_haloprop))
        else:
            result = spline_function(prim_haloprop)

        result = np.where(result > 1, 1, result)
        result = np.where(result < -1, -1, result)

        return result


    def _get_assembias_param_dict_key(self, ipar):
        """
        """
        return self._method_name_to_decorate + '_' + self.gal_type + '_assembias_param' + str(ipar+1)

    def _galprop_perturbation(self, **kwargs):
        """
        Method determines how much to boost the baseline function 
        according to the strength of assembly bias and the min/max 
        boost allowable by the requirement that the all-halo baseline 
        function be preserved. The returned perturbation applies to type-1 halos. 
        """
        lower_bound_key = 'lower_bound_' + self._method_name_to_decorate + '_' + self.gal_type
        baseline_lower_bound = getattr(self, lower_bound_key)
        upper_bound_key = 'upper_bound_' + self._method_name_to_decorate + '_' + self.gal_type
        baseline_upper_bound = getattr(self, upper_bound_key)

        try:
            baseline_result = kwargs['baseline_result']
            prim_haloprop = kwargs['prim_haloprop']
            splitting_result = kwargs['splitting_result']
        except KeyError:
            msg = ("Must call _galprop_perturbation method of the" 
                "HeavisideAssembias class with the following keyword arguments:\n"
                "``baseline_result``, ``splitting_result`` and ``prim_haloprop``")
            raise HalotoolsError(msg)

        result = np.zeros(len(prim_haloprop))

        strength = self.assembias_strength(prim_haloprop=prim_haloprop)
        positive_strength_idx = strength > 0
        negative_strength_idx = ~positive_strength_idx

        if len(baseline_result[positive_strength_idx]) > 0:
            base_pos = baseline_result[positive_strength_idx]
            split_pos = splitting_result[positive_strength_idx]
            strength_pos = strength[positive_strength_idx]

            upper_bound1 = baseline_upper_bound - base_pos
            upper_bound2 = ((1 - split_pos)/split_pos)*(base_pos - baseline_lower_bound)
            upper_bound = np.minimum(upper_bound1, upper_bound2)
            result[positive_strength_idx] = strength_pos*upper_bound

        if len(baseline_result[negative_strength_idx]) > 0:
            base_neg = baseline_result[negative_strength_idx]
            split_neg = splitting_result[negative_strength_idx]
            strength_neg = strength[negative_strength_idx]

            lower_bound1 = baseline_lower_bound - base_neg
            lower_bound2 = (1 - split_neg)/split_neg*(base_neg - baseline_upper_bound)
            lower_bound = np.maximum(lower_bound1, lower_bound2)
            result[negative_strength_idx] = np.abs(strength_neg)*lower_bound

        return result

    def assembias_decorator(self, func):
        """ Primary behavior of the `HeavisideAssembias` class. 

        This method is used to introduce a boost/decrement of the baseline 
        function in a manner that preserves the all-halo result. 

        Parameters 
        -----------
        func : function object 
            Baseline function whose behavior is being decorated with assembly bias. 

        Returns 
        -------
        wrapper : function object 
            Decorated function that includes assembly bias effects. 
        """
        lower_bound_key = 'lower_bound_' + self._method_name_to_decorate + '_' + self.gal_type
        baseline_lower_bound = getattr(self, lower_bound_key)
        upper_bound_key = 'upper_bound_' + self._method_name_to_decorate + '_' + self.gal_type
        baseline_upper_bound = getattr(self, upper_bound_key)

        def wrapper(*args, **kwargs):
            try:
                halo_table = kwargs['halo_table']
                prim_haloprop = halo_table[self.prim_haloprop_key]
            except KeyError:
                msg = ("The ``assembias_decorator`` method requires a "
                    "``halo_table`` input keyword argument.\n"
                    "The input halo_table must have a column with name %s")
                raise HalotoolsError(msg % self.prim_haloprop_key)

            split = self.percentile_splitting_function(halo_table = halo_table)
            result = func(*args, **kwargs)

            # We will only decorate values that are not edge cases
            no_edge_mask = (
                (split > 0) & (split < 1) & 
                (result > baseline_lower_bound) & (result < baseline_upper_bound)
                )
            no_edge_result = result[no_edge_mask]
            no_edge_split = split[no_edge_mask]
            # print("\nPrinting no-edge-split min/max")
            # print(no_edge_split.min(), no_edge_split.max())
            # print("\n")

            # Determine the type1_mask that divides the halo sample into two subsamples
            if hasattr(self, 'halo_type_tuple'):
                halo_type_key = self.halo_type_tuple[0]
                halo_type1_val = self.halo_type_tuple[1]
                type1_mask = halo_table[halo_type_key][no_edge_mask] == halo_type1_val
            elif self.sec_haloprop_key + '_percentile' in halo_table.keys():
                no_edge_percentiles = halo_table[self.sec_haloprop_key + '_percentile'][no_edge_mask]
                type1_mask = no_edge_percentiles > no_edge_split
            else:
                msg = ("\nThe HeavisideAssembias class implements assembly bias \n" 
                    "by altering the behavior of the model according to the value of " 
                    "``%s``.\n This quantity can be pre-computed using the "
                    "new_haloprop_func_dict mechanism, making your mock population run faster.\n"
                    "See the MockFactory documentation for detailed instructions.\n "
                    "Now computing %s from scratch.\n")
                key = self.sec_haloprop_key + '_percentile'
                warn(msg % (key, key))

                percentiles = compute_conditional_percentiles(
                    halo_table = halo_table, 
                    prim_haloprop_key = self.prim_haloprop_key, 
                    sec_haloprop_key = self.sec_haloprop_key
                    )
                no_edge_percentiles = percentiles[no_edge_mask]
                type1_mask = no_edge_percentiles > no_edge_split

            perturbation = self._galprop_perturbation(
                    prim_haloprop = halo_table[self.prim_haloprop_key][no_edge_mask], 
                    baseline_result = no_edge_result, 
                    splitting_result = no_edge_split)

            # This is the version in master that does not preserve the HOD
            ### but passes my test suite
            # perturbation[~type1_mask] *= (-no_edge_split[~type1_mask]/
            #     (1 - no_edge_split[~type1_mask]))

            # This is the new fix that seems to give correct results 
            ### but fails the test suite
            perturbation[type1_mask] *= (-no_edge_split[type1_mask]/
                (1 - no_edge_split[type1_mask]))

            no_edge_result += perturbation

            result[no_edge_mask] = no_edge_result
            return result

        return wrapper
























