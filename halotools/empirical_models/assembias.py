# -*- coding: utf-8 -*-
"""
Decorator class for implementing generalized assembly bias
"""

__all__ = ['HeavisideAssembias']

import numpy as np 
from warnings import warn 
from time import time

from . import model_defaults, model_helpers

from ..halotools_exceptions import HalotoolsError
from ..utils.table_utils import compute_conditional_percentiles

class HeavisideAssembias(object):
    """
    """
    def __init__(self, sec_haloprop_key = model_defaults.sec_haloprop_key, 
        loginterp = True, split = 0.5, assembias_strength = 0.5, **kwargs):
        """
        Parameters 
        ----------
        split : float, optional 
            Fraction between 0 and 1 defining how we split halos into two groupings based on 
            their conditional secondary percentiles. Default is 0.5 for a constant 50/50 split. 

        split_abcissa : list, optional 
            Values of the primary halo property at which the halos are split as described above in 
            the ``split`` argument. If ``loginterp`` is set to True (the default behavior), 
            the interpolation will be done in the logarithm of the primary halo property. 
            Default is to assume a constant 50/50 split. 

        split_ordinates : list, optional 
            Values of the fraction between 0 and 1 defining how we split halos into two groupings in a 
            fashion that varies with the value of ``prim_haloprop``. This fraction will equal 
            the input ``split_ordinates`` for halos whose ``prim_haloprop`` 
            equals the input ``split_abcissa``. Default is to assume a constant 50/50 split. 

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

        assembias_strength : float, optional 
            Fraction between -1 and 1 defining the assembly bias correlation strength. 
            Default is 0.5. 

        assembias_strength_abcissa : list, optional 
            Values of the primary halo property at which the assembly bias strength is specified. 
            Default is to assume a constant strength of 0.5. 

        assembias_strength_ordinates : list, optional 
            Values of the assembly bias strength when evaluated at the input ``assembias_strength_abcissa``. 
            Default is to assume a constant strength of 0.5. 

        sec_haloprop_key : string, optional 
            String giving the column name of the secondary halo property 
            governing the assembly bias. Must be a key in the halo_table 
            passed to the methods of `HeavisideAssembiasComponent`. 
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        loginterp : bool, optional
            If set to True, the interpolation will be done 
            in the logarithm of the primary halo property, 
            rather than linearly. Default is True. 

        """
        try:
            self._method_name_to_decorate = kwargs['method_name_to_decorate']
        except KeyError:
            msg = ("The constructor to the HeavisideAssembiasComponent class "
                "must be called with the following keyword arguments:\n" 
                "``%s``")
            raise HalotoolsError(msg % ('_method_name_to_decorate'))

        required_attr_list = ['prim_haloprop_key', '_lower_bound', '_upper_bound']
        for attr in required_attr_list:
            if not hasattr(self, attr):
                msg = ("In order to use the HeavisideAssembias class " 
                    "to decorate your model component with assembly bias, \n"
                    "the component instance must have a %s attribute")
                raise HalotoolsError(msg % attr)

        self._loginterp = loginterp
        self.sec_haloprop_key = sec_haloprop_key

        if 'splitting_model' in kwargs:
            self.splitting_model = kwargs['splitting_model']
            self.ancillary_model_dependencies = ['splitting_model']
            self.set_percentile_splitting(
                splitting_method_name = kwargs['splitting_method_name'])
        elif 'split_abcissa' and 'split_ordinates' in kwargs:
            self.set_percentile_splitting(split_abcissa=kwargs['split_abcissa'], 
                split_ordinates=kwargs['split_ordinates'])
        else:
            self.set_percentile_splitting(split = split)

        if 'assembias_strength_abcissa' and 'assembias_strength_ordinates' in kwargs:
            self._initialize_assembias_param_dict(split_abcissa=kwargs['assembias_strength_abcissa'], 
                split_ordinates=kwargs['assembias_strength_abcissa'])
        else:
            self._initialize_assembias_param_dict(assembias_strength=assembias_strength)

        if 'halo_type_tuple' in kwargs:
            self.halo_type_tuple = kwargs['halo_type_tuple']

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


    def set_percentile_splitting(self, **kwargs):
        """
        """
        if 'splitting_method_name' in kwargs.keys():
            func = getattr(self.splitting_model, kwargs['splitting_method_name'])
            if callable(func):
                self._input_split_func = func
            else:
                raise HalotoolsError("Input ``splitting_model`` must have a callable function "
                    "named ``%s``" % kwargs['splitting_method_name'])
        elif 'split' in kwargs.keys():
            self._split_abcissa = [2]
            self._split_ordinates = [kwargs['split']]
        elif ('split_ordinates' in kwargs.keys()) & ('split_abcissa' in kwargs.keys()):
            self._split_abcissa = kwargs['split_abcissa']
            self._split_ordinates = kwargs['split_ordinates']
        else:
            msg = ("The constructor to the HeavisideAssembias class "
                "must be called with either the ``split`` keyword argument,\n"
                " or both the ``split_abcissa`` and ``split_ordinates`` keyword arguments" )
            raise HalotoolsError(msg)

    def percentile_splitting_function(self, **kwargs):
        """
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


    def _initialize_assembias_param_dict(self, **kwargs):
        """
        """
        if not hasattr(self, 'param_dict'):
            self.param_dict = {}

        if 'assembias_strength' in kwargs.keys():
            self._assembias_strength_abcissa = [2]
            self.param_dict[self._get_assembias_param_dict_key(0)] = kwargs['assembias_strength']
        elif 'assembias_strength_ordinates' and 'assembias_strength_abcissa' in kwargs:
            self._assembias_strength_abcissa = kwargs['assembias_strength_abcissa']
            for ipar, val in enumerate(kwargs['assembias_strength_ordinates']):
                self.param_dict[self._get_assembias_param_dict_key(ipar)] = val
        else:
            msg = ("The constructor to the HeavisideAssembias class "
                "must be called with either the ``assembias_strength`` keyword argument,\n"
                " or both the ``assembias_strength_abcissa`` and ``assembias_strength_ordinates`` keyword arguments" )
            raise HalotoolsError(msg)

    def assembias_strength(self, **kwargs):
        """
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
        return self._method_name_to_decorate + '_assembias_param' + str(ipar+1)

    def galprop_perturbation(self, **kwargs):
        """
        """
        try:
            baseline_result = kwargs['baseline_result']
            prim_haloprop = kwargs['prim_haloprop']
            splitting_result = kwargs['splitting_result']
        except KeyError:
            msg = ("Must call galprop_perturbation method of the" 
                "HeavisideAssembias class with the following keyword arguments:\n"
                "``baseline_result``, ``splitting_result`` and ``prim_haloprop``")
            raise HalotoolsError(msg)

        result = np.zeros(len(prim_haloprop))

        strength = self.assembias_strength(prim_haloprop=prim_haloprop)
        positive_strength_idx = strength > 0
        negative_strength_idx = np.invert(positive_strength_idx)

        if len(baseline_result[positive_strength_idx]) > 0:
            base_pos = baseline_result[positive_strength_idx]
            split_pos = splitting_result[positive_strength_idx]
            strength_pos = strength[positive_strength_idx]

            upper_bound1 = self._upper_bound - base_pos
            upper_bound2 = ((1 - split_pos)/split_pos)*(base_pos - self._lower_bound)
            upper_bound = np.minimum(upper_bound1, upper_bound2)
            result[positive_strength_idx] = strength_pos*upper_bound

        if len(baseline_result[negative_strength_idx]) > 0:
            base_neg = baseline_result[negative_strength_idx]
            split_neg = splitting_result[negative_strength_idx]
            strength_neg = strength[negative_strength_idx]

            lower_bound1 = self._lower_bound - base_neg
            lower_bound2 = (1 - split_neg)/split_neg*(base_neg - self._upper_bound)
            lower_bound = np.maximum(lower_bound1, lower_bound2)
            result[negative_strength_idx] = strength_neg*lower_bound

        return result

    def assembias_decorator(self, func):
        """
        """

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

            # We will only apply decorate values that are not edge cases
            no_edge_mask = (
                (split > 0) & (split < 1) & 
                (result > self._lower_bound) & (result < self._upper_bound)
                )
            no_edge_result = result[no_edge_mask]
            no_edge_split = split[no_edge_mask]

            # Determine the type1_mask that divides the halo sample into two subsamples
            if hasattr(self, 'halo_type_tuple'):
                halo_type_key = self.halo_type_tuple[0]
                halo_type1_val = self.halo_type_tuple[1]
                type1_mask = halo_table[halo_type_key][no_edge_mask] == halo_type1_val
            elif self.sec_haloprop_key + '_percentile' in halo_table.keys():
                no_edge_percentiles = halo_table[self.sec_haloprop_key + '_percentile'][no_edge_mask]
                type1_mask = no_edge_percentiles >= no_edge_split
            else:
                msg = ("Computing ``%s`` quantity from scratch - \n"
                    "Method is much faster if this quantity is pre-computed")
                key = self.sec_haloprop_key + '_percentile'
                warn(msg % key)

                percentiles = compute_conditional_percentiles(
                    halo_table = halo_table, 
                    prim_haloprop_key = self.prim_haloprop_key, 
                    sec_haloprop_key = self.sec_haloprop_key
                    )
                no_edge_percentiles = percentiles[no_edge_mask]
                type1_mask = no_edge_percentiles >= no_edge_split

            perturbation = self.galprop_perturbation(
                    prim_haloprop = halo_table[self.prim_haloprop_key][no_edge_mask], 
                    baseline_result = no_edge_result, 
                    splitting_result = no_edge_split)
            perturbation[np.invert(type1_mask)] *= (-no_edge_split[np.invert(type1_mask)]/
                (1 - no_edge_split[np.invert(type1_mask)]))

            no_edge_result += perturbation

            result[no_edge_mask] = no_edge_result
            return result

        return wrapper
























