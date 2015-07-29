# -*- coding: utf-8 -*-
"""
Decorator class for implementing generalized assembly bias
"""

__all__ = ['HeavisideAssembiasComponent']

import numpy as np 
from warnings import warn 

from . import model_defaults, model_helpers

from ..halotools_exceptions import HalotoolsError
from ..utils.table_utils import compute_conditional_percentiles

class HeavisideAssembiasComponent(object):
    """
    """
    def __init__(self, sec_haloprop_key = model_defaults.sec_haloprop_key, 
        loginterp = True, split = 0.5, assembias_strength = 0.5, **kwargs):
        """
        Parameters 
        ----------
        baseline_model : Class object 
            Class governing the underlying behavior which is being 
            decorated with assembly bias. An instance of this class will 
            be built by passing on the full set of keyword arguments that 
            were passed to `HeavisideAssembiasComponent`. 

        method_name_to_decorate : string 
            Name of the method bound instances of ``baseline_model`` that 
            we are decorating with assembly bias.

        lower_bound : float 
            Smallest physically meaningful value for the property being modeled. 

            For example, if modeling the first occupation moment of any galaxy type, 
            ``lower_bound`` = 0; the same value would be used if modeling 
            the quenched fraction. 

            Can be set to -float("inf") provided that 
            the ``upper_bound`` keyword argument is set to a bounded value. 

        upper_bound : float 
            Largest physically meaningful value for the property being modeled. 

            For example, if modeling the first occupation moment of any central galaxies, 
            ``upper_bound`` = 1, whereas for satellites we would set ``upper_bound`` = float("inf"). 

            Can be set to float("inf") provided that 
            the ``lower_bound`` keyword argument is set to a bounded value. 

        split : float, optional 
            Fraction between 0 and 1 defining how we split halos into two groupings based on 
            their conditional secondary percentiles. Default is 0.5
            their conditional secondary percentiles. Default is 0.5 for a constant 50/50 split. 

        split_abcissa : list, optional 
            Values of the primary halo property at which the halos are split as described above in 
            the ``split`` argument. 
            If ``loginterp`` is set to True (the default behavior), the interpolation will be done 
            in the logarithm of the primary halo property. Default is to assume a constant 50/50 split. 

        split_ordinates : list, optional 
            Values of the fraction between 0 and 1 defining how we split halos into two groupings in a 
            fashion that varies with the value of ``prim_haloprop``. 
            This fraction will equal the input ``split_ordinates`` for halos whose ``prim_haloprop`` 
            equals the input ``split_abcissa``. Default is to assume a constant 50/50 split. 

        split_func : function, optional 
            Function object used to split the input halos into two types.

        halo_type_tuple : tuple, optional 
            Tuple providing the information about how elements of the input ``halo_table`` 
            have been pre-divided into types. The first tuple entry must be a 
            string giving the column name of the input ``halo_table`` that provides the halo-typing. 
            Second and third entries gives the value of this column for type-1 and type-2 halos, respectively. 

            If provided, you must ensure that the splitting of the ``halo_table`` 
            was self-consistently performed with the 
            input ``split``, or ``split_abcissa`` and ``split_ordinates``, or 
            ``split_func`` keyword arguments. 

        assembias_strength : float, optional 
            Fraction between -1 and 1 defining the assembly bias correlation strength. 
            Default is 0.5. 

        assembias_strength_abcissa : list, optional 
            Values of the primary halo property at which the assembly bias strength is specified. 
            Default is to assume a constant strength given by ``assembias_strength`` keyword argument. 

        assembias_strength_ordinates : list, optional 
            Values of the assembly bias strength when evaluated at the input ``assembias_strength_abcissa``. 
            Default is to assume a constant strength given by ``assembias_strength`` keyword argument. 

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
            baseline_model_instance = kwargs['baseline_model'](**kwargs)
            self.baseline_model_instance = baseline_model_instance
            self._method_name_to_decorate = kwargs['method_name_to_decorate']
            self._lower_bound = kwargs['lower_bound']
            self._upper_bound = kwargs['upper_bound']
        except KeyError:
            msg = ("The constructor to the HeavisideAssembiasComponent class "
                "must be called with the following keyword arguments:\n" 
                "``%s``, ``%s``, ``%s``, ``%s``")
            raise HalotoolsError(msg % ('lower_bound', 'upper_bound', 
                'method_name_to_decorate', 'baseline_model_instance'))

        self._loginterp = loginterp
        self.sec_haloprop_key = sec_haloprop_key
        self.prim_haloprop_key = self.baseline_model_instance.prim_haloprop_key

        if 'split_func' in kwargs:
            self.set_percentile_splitting(split_func = kwargs['split_func'])
        elif 'split_abcissa' and 'split_ordinates' in kwargs:
            self.set_percentile_splitting(split_abcissa=kwargs['split_abcissa'], 
                split_ordinates=kwargs['split_ordinates'])
        else:
            self.set_percentile_splitting(split = split)

        if 'assembias_strength_abcissa' and 'assembias_strength_ordinates' in kwargs:
            self._initialize_param_dict(split_abcissa=kwargs['assembias_strength_abcissa'], 
                split_ordinates=kwargs['assembias_strength_abcissa'])
        else:
            self._initialize_param_dict(assembias_strength=assembias_strength)

        if 'halo_type_tuple' in kwargs:
            self.halo_type_tuple = kwargs['halo_type_tuple']

        
    def __getattr__(self, attr):
        return getattr(self.baseline_model_instance, attr)

    def set_percentile_splitting(self, **kwargs):
        """
        """
        if 'split_func' in kwargs.keys():
            func = kwargs['split_func']
            if callable(func):
                self._input_split_func = func
            else:
                raise HalotoolsError("Input ``split_func`` must be a callable function")
        elif 'split' in kwargs.keys():
            self._split_abcissa = [0]
            self._split_ordinates = [kwargs['split']]
        elif ('split_ordinates' in kwargs.keys()) & ('split_abcissa' in kwargs.keys()):
            self._split_abcissa = kwargs['split_abcissa']
            self._split_ordinates = kwargs['split_ordinates']
        else:
            msg = ("The constructor to the HeavisideAssembiasComponent class "
                "must be called with either the ``split`` keyword argument,\n"
                " or both the ``split_abcissa`` and ``split_ordinates`` keyword arguments" )
            raise HalotoolsError(msg)

    def percentile_splitting_function(self, **kwargs):
        """
        """
        try:
            halo_table = kwargs['halo_table']
        except KeyError:
            raise HalotoolsError("The ``percentile_splitting_function`` method requires a "
                "``halo_table`` input keyword argument")
        try:
            prim_haloprop = halo_table[self.prim_haloprop_key]
        except KeyError:
            raise HalotoolsError("prim_haloprop_key = %s is not a column of the input halo_table" % self.prim_haloprop_key)

        if hasattr(self, '_input_split_func'):
            return self._input_split_func(halo_table = halo_table)
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


    def _initialize_param_dict(self, **kwargs):
        """
        """
        if hasattr(self.baseline_model_instance, 'param_dict'):
            self.param_dict = self.baseline_model_instance.param_dict
        else:
            self.param_dict = {}

        if 'assembias_strength' in kwargs.keys():
            self._assembias_strength_abcissa = [1]
            self.param_dict[self._get_param_dict_key(0)] = kwargs['assembias_strength']
        elif 'assembias_strength_ordinates' and 'assembias_strength_abcissa' in kwargs:
            self._assembias_strength_abcissa = kwargs['assembias_strength_abcissa']
            for ipar, val in enumerate(kwargs['assembias_strength_ordinates']):
                self.param_dict[self._get_param_dict_key(ipar)] = val
        else:
            msg = ("The constructor to the HeavisideAssembiasComponent class "
                "must be called with either the ``assembias_strength`` keyword argument,\n"
                " or both the ``assembias_strength_abcissa`` and ``assembias_strength_ordinates`` keyword arguments" )
            raise HalotoolsError(msg)

    def assembias_strength(self, **kwargs):
        """
        """

        try:
            halo_table = kwargs['halo_table']
        except KeyError:
            raise HalotoolsError("The ``percentile_splitting_function`` method requires a "
                "``halo_table`` input keyword argument")

        try:
            prim_haloprop = halo_table[self.prim_haloprop_key]
        except KeyError:
            raise HalotoolsError("prim_haloprop_key = %s is not a column of the input halo_table" % self.prim_haloprop_key)

        model_ordinates = [self.param_dict[self._get_param_dict_key(ipar)] for ipar in range(len(self._assembias_strength_abcissa))]

        spline_function = model_helpers.custom_spline(self._assembias_strength_abcissa, model_ordinates)


        if self._loginterp is True:
            result = spline_function(np.log10(prim_haloprop))
        else:
            result = spline_function(prim_haloprop)

        result = np.where(result > 1, 1, result)
        result = np.where(result < -1, -1, result)

        return result


    def _get_param_dict_key(self, ipar):
        """
        """
        return self._method_name_to_decorate + '_assembias_param' + str(ipar+1)


    def lower_bound_galprop_perturbation(self, **kwargs):
        """
        """

        baseline_func = getattr(self.baseline_model_instance, self._method_name_to_decorate)

        lower_bound1 = self._lower_bound - baseline_func(**kwargs)
        lower_bound2_prefactor = (
            (1 - self.percentile_splitting_function(**kwargs))/
            self.percentile_splitting_function(**kwargs))
        lower_bound2 = lower_bound2_prefactor*(baseline_func(**kwargs) - self._upper_bound)

        return np.maximum(lower_bound1, lower_bound2)

    def upper_bound_galprop_perturbation(self, **kwargs):
        """
        """

        baseline_func = getattr(self.baseline_model_instance, self._method_name_to_decorate)

        upper_bound1 = self._upper_bound - baseline_func(**kwargs)
        upper_bound2_prefactor = (
            (1 - self.percentile_splitting_function(**kwargs))/
            self.percentile_splitting_function(**kwargs))
        upper_bound2 = upper_bound2_prefactor*(baseline_func(**kwargs) - self._lower_bound)

        return np.minimum(upper_bound1, upper_bound2)


    def galprop_perturbation(self, **kwargs):
        """
        """

        try:
            halo_table = kwargs['halo_table']
        except KeyError:
            raise HalotoolsError("The ``percentile_splitting_function`` method requires a "
                "``halo_table`` input keyword argument")

        result = np.zeros(len(halo_table))

        strength = self.assembias_strength(halo_table=halo_table)
        positive_strength_idx = strength > 0
        negative_strength_idx = np.invert(positive_strength)

        if len(halo_table[positive_strength_idx] > 0):
            result[positive_strength_idx] = (
                strength[positive_strength_idx]*
                self.upper_bound_galprop_perturbation(halo_table = halo_table[positive_strength_idx])
                )

        if len(halo_table[negative_strength_idx] > 0):
            result[negative_strength_idx] = (
                strength[negative_strength_idx]*
                self.lower_bound_galprop_perturbation(halo_table = halo_table[negative_strength_idx])
                )

        return result

    def dx2(self, *args, **kwargs):
        split = self.split_func(*args, **kwargs)
        dx1 = self.dx1(*args, **kwargs)
        return -split*dx1/(1-split)

    def assembias_decorator(self, func):

        def wrapper(*args, **kwargs):

            try:
                halo_table = kwargs['halo_table']
            except KeyError:
                raise HalotoolsError("The ``percentile_splitting_function`` method requires a "
                    "``halo_table`` input keyword argument")

            split = self.percentile_splitting_function(halo_table = halo_table)
            result = func(*args, **kwargs)

            # We will only apply decorate values that are not edge cases
            no_edge_mask = (
                (split > 0) & (split < 1) & 
                (result > self._lower_bound) & (result < self._upper_bound)
                )
            no_edge_result = result[no_edge_mask]
            no_edge_halos = halo_table[no_edge_mask]

            # Determine the type1_mask that divides the halo sample into two subsamples
            if hasattr(self, halo_type_tuple):
                halo_type_key = self.halo_type_tuple[0]
                halo_type1_val = self.halo_type_tuple[1]
                type1_mask = no_edge_halos[halo_type_key] == halo_type1_val
            elif self.sec_haloprop_key + '_percentile' in no_edge_halos.keys():
                no_edge_percentiles = no_edge_halos[self.sec_haloprop_key + '_percentile']
                no_edge_split = split[no_edge_mask]
                type1_mask = no_edge_percentiles < no_edge_split
            else:
                no_edge_percentiles = compute_conditional_percentiles(
                    halo_table = no_edge_halos, 
                    prim_haloprop_key = self.prim_haloprop_key, 
                    sec_haloprop_key = self.sec_haloprop_key
                    )
                no_edge_split = split[no_edge_mask]
                type1_mask = no_edge_percentiles < no_edge_split

            dx1 = self.dx1(no_edge_halos[case1_mask])
            no_edge_result[case1_mask] += dx1
            dx2 = self.dx1(no_edge_halos[np.invert(case1_mask)])
            no_edge_result[np.invert(case1_mask)] += dx2

            result[no_edge_mask] = no_edge_result
            return result

        return wrapper




