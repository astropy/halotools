# -*- coding: utf-8 -*-
"""
Decorator class for implementing generalized assembly bias
"""

__all__ = ['HeavisideAssembias']

import numpy as np 
from warnings import warn 
from time import time

from . import model_defaults, model_helpers, hod_components, sfr_components

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
        prim_haloprop_key : string 
            String giving the column name of the primary halo property governing 
            the galaxy property function that `HeavisideAssembias` decorates with assembly bias. 

        method_name_to_decorate : string 
            Name of the method bound to instances of the baseline_model that 
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
            self.prim_haloprop_key = kwargs['prim_haloprop_key']
            self._method_name_to_decorate = kwargs['method_name_to_decorate']
            self._lower_bound = kwargs['lower_bound']
            self._upper_bound = kwargs['upper_bound']
        except KeyError:
            msg = ("The constructor to the HeavisideAssembiasComponent class "
                "must be called with the following keyword arguments:\n" 
                "``%s``, ``%s``, ``%s``, ``%s``")
            raise HalotoolsError(msg % ('prim_haloprop_key', '_method_name_to_decorate', 
                'lower_bound', 'upper_bound'))

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
            self._initialize_param_dict(split_abcissa=kwargs['assembias_strength_abcissa'], 
                split_ordinates=kwargs['assembias_strength_abcissa'])
        else:
            self._initialize_param_dict(assembias_strength=assembias_strength)

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
        ta = time()
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

        tb = (time() - ta)*1000.
        print("percentile_splitting_function calculation runtime = %.2f ms" % tb)
        return result


    def _initialize_param_dict(self, **kwargs):
        """
        """
        if not hasattr(self, 'param_dict'):
            self.param_dict = {}

        if 'assembias_strength' in kwargs.keys():
            self._assembias_strength_abcissa = [2]
            self.param_dict[self._get_param_dict_key(0)] = kwargs['assembias_strength']
        elif 'assembias_strength_ordinates' and 'assembias_strength_abcissa' in kwargs:
            self._assembias_strength_abcissa = kwargs['assembias_strength_abcissa']
            for ipar, val in enumerate(kwargs['assembias_strength_ordinates']):
                self.param_dict[self._get_param_dict_key(ipar)] = val
        else:
            msg = ("The constructor to the HeavisideAssembias class "
                "must be called with either the ``assembias_strength`` keyword argument,\n"
                " or both the ``assembias_strength_abcissa`` and ``assembias_strength_ordinates`` keyword arguments" )
            raise HalotoolsError(msg)

    def assembias_strength(self, **kwargs):
        """
        """
        ta = time()

        if 'prim_haloprop' in kwargs.keys():
            prim_haloprop = kwargs['prim_haloprop']
        else:
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

        tb = (time() - ta)*1000.
        print("Assembias_strength calculation runtime = %.2f ms" % tb)
        return result


    def _get_param_dict_key(self, ipar):
        """
        """
        return self._method_name_to_decorate + '_assembias_param' + str(ipar+1)


    def lower_bound_galprop_perturbation(self, **kwargs):
        """
        """
        ta = time()
        try:
            baseline_result = kwargs['baseline_result']
        except KeyError:
            msg = ("Must call lower_bound_galprop_perturbation method of the" 
                "HeavisideAssembias class with the baseline_result keyword argument")
            raise HalotoolsError(msg)

        try:
            split = kwargs['splitting_result']
        except KeyError:
            raise HalotoolsError("The ``lower_bound_galprop_perturbation`` method requires a "
                "``splitting_result`` input keyword argument")

        lower_bound1 = self._lower_bound - baseline_result
        lower_bound2_prefactor = (1 - split)/split
        lower_bound2 = lower_bound2_prefactor*(baseline_result - self._upper_bound)

        tb = time() - ta
        print("lower_bound_galprop_perturbation runtime = %.2f ms" % (tb*1000.))

        return np.maximum(lower_bound1, lower_bound2)

    def upper_bound_galprop_perturbation(self, **kwargs):
        """
        """
        ta = time()
        try:
            baseline_result = kwargs['baseline_result']
        except KeyError:
            msg = ("Must call upper_bound_galprop_perturbation method of the" 
                "HeavisideAssembias class with the baseline_result keyword argument")
            raise HalotoolsError(msg)

        try:
            split = kwargs['splitting_result']
        except KeyError:
            raise HalotoolsError("The ``upper_bound_galprop_perturbation`` method requires a "
                "``splitting_result`` input keyword argument")

        upper_bound1 = self._upper_bound - baseline_result
        upper_bound2_prefactor = (1 - split)/split
        upper_bound2 = upper_bound2_prefactor*(baseline_result - self._lower_bound)
        tb = time() - ta
        print("upper_bound_galprop_perturbation runtime = %.2f ms" % (tb*1000.))

        return np.minimum(upper_bound1, upper_bound2)


    def galprop_perturbation(self, **kwargs):
        """
        """
        ta = time()
        try:
            baseline_result = kwargs['baseline_result']
        except KeyError:
            msg = ("Must call galprop_perturbation method of the" 
                "HeavisideAssembias class with the baseline_result keyword argument")
            raise HalotoolsError(msg)

        if 'prim_haloprop' in kwargs.keys():
            prim_haloprop = kwargs['prim_haloprop']
        else:
            try:
                halo_table = kwargs['halo_table']
            except KeyError:
                raise HalotoolsError("The ``percentile_splitting_function`` method requires a "
                    "``halo_table`` input keyword argument")

            try:
                prim_haloprop = halo_table[self.prim_haloprop_key]
            except KeyError:
                raise HalotoolsError("prim_haloprop_key = %s is not a column of the input halo_table" % self.prim_haloprop_key)

        try:
            splitting_result = kwargs['splitting_result']
        except KeyError:
            raise HalotoolsError("The ``percentile_splitting_function`` method requires a "
                "``splitting_result`` input keyword argument")

        result = np.zeros(len(prim_haloprop))

        strength = self.assembias_strength(prim_haloprop=prim_haloprop)
        positive_strength_idx = strength > 0
        negative_strength_idx = np.invert(positive_strength_idx)

        if len(baseline_result[positive_strength_idx]) > 0:
            result[positive_strength_idx] = (
                strength[positive_strength_idx]*
                self.upper_bound_galprop_perturbation(
                    baseline_result = baseline_result[positive_strength_idx], 
                    splitting_result = splitting_result[positive_strength_idx])
                )

        if len(baseline_result[negative_strength_idx]) > 0:
            result[negative_strength_idx] = (
                strength[negative_strength_idx]*
                self.lower_bound_galprop_perturbation(
                    baseline_result = baseline_result[negative_strength_idx], 
                    splitting_result = splitting_result[negative_strength_idx])
                )

        tb = time() - ta
        print("galprop_perturbation runtime = %.2f ms" % (tb*1000.))
        return result

    def complementary_galprop_perturbation(self, **kwargs):
        """
        """
        ta = time()

        try:
            split = kwargs['splitting_result']
        except KeyError:
            raise HalotoolsError("The ``upper_bound_galprop_perturbation`` method requires a "
                "``splitting_result`` input keyword argument")

        galprop_perturbation = self.galprop_perturbation(**kwargs)

        result = -split*galprop_perturbation/(1-split)
        tb = time() - ta
        print("complementary_galprop_perturbation runtime = %.2f ms" % (tb*1000.))
        return result

    def assembias_decorator(self, func):
        """
        """

        def wrapper(*args, **kwargs):

            t0 = time()
            try:
                halo_table = kwargs['halo_table']
            except KeyError:
                raise HalotoolsError("The ``percentile_splitting_function`` method requires a "
                    "``halo_table`` input keyword argument")

            try:
                prim_haloprop = halo_table[self.prim_haloprop_key]
            except KeyError:
                raise HalotoolsError("prim_haloprop_key = %s is not a column of the input halo_table" % self.prim_haloprop_key)

            t1 = time()
            t = (t1 - t0)*1000.
            print("t1 = %.2f ms" % t)

            split = self.percentile_splitting_function(halo_table = halo_table)
            t2 = time()
            t = (t2 - t1)*1000.
            print("t2 = %.2f ms" % t)

            result = func(*args, **kwargs)
            t3 = time()
            t = (t3 - t2)*1000.
            print("t3 = %.2f ms" % t)


            # We will only apply decorate values that are not edge cases
            no_edge_mask = (
                (split > 0) & (split < 1) & 
                (result > self._lower_bound) & (result < self._upper_bound)
                )
            t4 = time()
            t = (t4 - t3)*1000.
            print("t4 = %.2f ms" % t)
            no_edge_result = result[no_edge_mask]
            t5 = time()
            t = (t5 - t4)*1000.
            print("t5 = %.2f ms" % t)

            # no_edge_halos = halo_table[no_edge_mask]
            t6 = time()
            t = (t6 - t5)*1000.
            print("t6 = %.2f ms" % t)

            # Determine the type1_mask that divides the halo sample into two subsamples
            if hasattr(self, 'halo_type_tuple'):
                halo_type_key = self.halo_type_tuple[0]
                halo_type1_val = self.halo_type_tuple[1]
                type1_mask = halo_table[halo_type_key][no_edge_mask] == halo_type1_val
                # type1_mask = no_edge_halos[halo_type_key] == halo_type1_val
            elif self.sec_haloprop_key + '_percentile' in halo_table.keys():
                no_edge_percentiles = halo_table[self.sec_haloprop_key + '_percentile'][no_edge_mask]
                no_edge_split = split[no_edge_mask]
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
                t7a = time()
                t = (t7a - t6)*1000.
                print("t7a = %.2f ms" % t)
                no_edge_percentiles = percentiles[no_edge_mask]
                no_edge_split = split[no_edge_mask]
                type1_mask = no_edge_percentiles >= no_edge_split

            t7 = time()
            t = (t7 - t6)*1000.
            print("t7 = %.2f ms" % t)


            # no_edge_halos_type1 = no_edge_halos[type1_mask]
            t8 = time()
            t = (t8 - t7)*1000.
            print("t8 = %.2f ms" % t)

            no_edge_result[type1_mask] += (
                self.galprop_perturbation(
                    prim_haloprop = halo_table[self.prim_haloprop_key][no_edge_mask][type1_mask], 
                    baseline_result = no_edge_result[type1_mask], 
                    splitting_result = no_edge_split[type1_mask])
                )
            t9 = time()
            t = (t9 - t8)*1000.
            print("t9 = %.2f ms" % t)


            # no_edge_halos_type2 = no_edge_halos[np.invert(type1_mask)]
            t10 = time()
            t = (t10 - t9)*1000.
            print("t10 = %.2f ms" % t)

            no_edge_result[np.invert(type1_mask)] += (
                self.complementary_galprop_perturbation(
                    prim_haloprop = halo_table[self.prim_haloprop_key][no_edge_mask][np.invert(type1_mask)], 
                    baseline_result = no_edge_result[np.invert(type1_mask)], 
                    splitting_result = no_edge_split[np.invert(type1_mask)])
                )
            t11 = time()
            t = (t11 - t10)*1000.
            print("t11 = %.2f ms" % t)


            result[no_edge_mask] = no_edge_result
            t12 = time()
            t = (t12 - t11)*1000.
            print("t12 = %.2f ms" % t)

            return result

        return wrapper


class AssembiasZheng07Cens(hod_components.Zheng07Cens, HeavisideAssembias):
    """
    """
    def __init__(self, **kwargs):
        """
        Parameters 
        ----------
        gal_type : string, optional keyword argument
            Name of the galaxy population being modeled. Default is ``centrals``.  

        threshold : float, optional keyword argument
            Luminosity threshold of the mock galaxy sample. If specified, 
            input value must agree with one of the thresholds used in Zheng07 to fit HODs: 
            [-18, -18.5, -19, -19.5, -20, -20.5, -21, -21.5, -22].
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        prim_haloprop_key : string, optional keyword argument 
            String giving the column name of the primary halo property governing 
            the occupation statistics of gal_type galaxies. 
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        split : float, optional 
            Fraction between 0 and 1 defining how we split halos into two groupings based on 
            their conditional secondary percentiles. Default is 0.5 for a constant 50/50 split. 

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

        """
        hod_components.Zheng07Cens.__init__(self, **kwargs)
        HeavisideAssembias.__init__(self, method_name_to_decorate = 'mean_occupation', 
            prim_haloprop_key = self.prim_haloprop_key, 
            lower_bound = 0, upper_bound = 1, **kwargs)




















