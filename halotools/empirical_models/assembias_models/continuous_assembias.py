r"""
This module contains the `~halotools.empirical_models.ContinuousAssembias` class.
The purpose of this class is to introduce a sigmoid-shaped assembly bias into
any method of any component model. It subclasses `HeavisideAssembias` and
 extends its features. Details can be found in
`McLaughlin et al 2017 (in prep)`_.
"""

from functools import wraps
import numpy as np

from . import HeavisideAssembias, FreeSplitAssembias
from .. import model_helpers
from ...custom_exceptions import HalotoolsError
from ...utils.array_utils import custom_len
from ...utils.table_utils import compute_conditional_percentile_values, compute_conditional_averages
from ...utils.table_utils import compute_conditional_percentiles

__all__ = ('ContinuousAssembias', 'FreeSplitContinuousAssembias' )
__author__ = ('Sean McLaughlin', )

F = 0.99  # slope tolerance


class ContinuousAssembias(HeavisideAssembias):
    """
    Class used to extend the behavior of `HeavisideAssembias` for continuous distributions.
    """
    def _disp_func(self, sec_haloprop, slope):
        """
        Function define the value of \delta N_max as a function of the sec_haloprop distribution and a slope param

        Parameters
        ----------
        sec_haloprop : float or ndarray
            The table of secondary haloprops.
            In general, the median (or some other percentile) is subtracted
            from this value first, and it is normalized by the most extreme value.

        slope : float
            The "slope" parameter, which controls the rate of change between
            negative and positive displacements.

            The name "slope" is a bit of a misnomer,
            but generally controls the rate of change of the distribution.

        Returns
        -------
        displacement : float or ndarray
            The displacement as a function of the sec_haloprop
        """
        return np.reciprocal(1 + np.exp(-(10 ** slope) * (sec_haloprop)))

    def _initialize_assembias_param_dict(self, assembias_strength=0.5, assembias_slope=1.0, **kwargs):
        r"""
        For full documentation, see the Heaviside Assembias Declaration in Halotools.

        This function calls the superclass's version.
        Then, it adds the parameters from disp_func to the dict as well.

        Parameters
        ----------
        assembias_strength : float, optional
            Strength of assembias. Default is 1.0 for maximum strength

        assembias_slope : float, optional
            Effective slopes of disp_func.
            Can be an iterator of float. Uses the same abscissa as assembias_strength
        """
        super(ContinuousAssembias, self)._initialize_assembias_param_dict(
                assembias_strength=assembias_strength, **kwargs)
        #  Accept float or iterable
        slope = assembias_slope
        try:
            iterator = iter(slope)
            slope = list(slope)
        except TypeError:
            slope = [slope]

        # assert it has the proper length
        if custom_len(self._assembias_strength_abscissa) != custom_len(slope):
            raise HalotoolsError("``assembias_strength`` and ``assembias_slope`` "
                                 "must have the same length")

        for ipar, val in enumerate(slope):
            self.param_dict[self._get_continuous_assembias_param_dict_key(ipar)] = val

    #  This formula ensures that the most extreme value will experience F times the maximum displacement
    #  Otherwise, slope could arbitrarily cancel out strength.
    #@model_helpers.bounds_enforcing_decorator_factory(np.log10(np.log((1+F)/(1-F))), 10)
    def assembias_slope(self, prim_haloprop):
        """ Returns the slope of disp_func as a function of prim_haloprop

        Parameters
        ----------
        prim_haloprop : float or ndarray
            the primary haloprop, e.g., halo mass or vmax

        Returns
        -------
        slope : float or ndarray
            Value of the slope at each prim_haloprop
        """
        model_ordinates = (self.param_dict[self._get_continuous_assembias_param_dict_key(ipar)]
                           for ipar in xrange(len(self._assembias_strength_abscissa)))
        spline_function = model_helpers.custom_spline(
            self._assembias_strength_abscissa, list(model_ordinates), k=3)

        if self._loginterp is True:
            result = spline_function(np.log10(prim_haloprop))
        else:
            result = spline_function(prim_haloprop)

        return result

    def _get_continuous_assembias_param_dict_key(self, ipar):
        """
        """
        return self._method_name_to_decorate + '_' + self.gal_type + '_assembias_slope' + str(ipar + 1)

    def _galprop_perturbation(self, **kwargs):
        r"""
        Method determines hwo much to boost the baseline function
        according to the strength of assembly bias and the min/max
        boost allowable by the requirement that the all-halo baseline
        function be preserved. The returned perturbation applies to all halos.

        Required kwargs are: ``baseline_result``, ``prim_haloprop``, ``sec_haloprop``.

        Note that this is defined where sec_haloprop has had the p-th percentile subtracted
        and is normalized by the maximum value.

        Returns ndarray with dimensions of prim_haloprop detailing the perturbation.
        """

        lower_bound_key = 'lower_bound_' + self._method_name_to_decorate + '_' + self.gal_type
        baseline_lower_bound = getattr(self, lower_bound_key)
        upper_bound_key = 'upper_bound_' + self._method_name_to_decorate + '_' + self.gal_type
        baseline_upper_bound = getattr(self, upper_bound_key)

        try:
            baseline_result = kwargs['baseline_result']
            prim_haloprop = kwargs['prim_haloprop']
            sec_haloprop = kwargs['sec_haloprop']
        except KeyError:
            msg = ("Must call _galprop_perturbation method of the"
                   "HeavisideAssembias class with the following keyword arguments:\n"
                   "``baseline_result``, ``splitting_result`` and ``prim_haloprop``")
            raise HalotoolsError(msg)

        #  evaluate my continuous modification
        strength = self.assembias_strength(prim_haloprop)
        slope = self.assembias_slope(prim_haloprop)

        #  the average displacement acts as a normalization we need.
        max_displacement = self._disp_func(sec_haloprop=sec_haloprop, slope=slope)
        disp_average = compute_conditional_averages(vals=max_displacement,prim_haloprop=prim_haloprop)
        #disp_average = np.ones((prim_haloprop.shape[0], ))*0.5

        result = np.zeros(len(prim_haloprop))

        greater_than_half_avg_idx = disp_average > 0.5
        less_than_half_avg_idx = disp_average <= 0.5

        #print max_displacement
        #print strength, slope

        if len(max_displacement[greater_than_half_avg_idx]) > 0:
            base_pos = baseline_result[greater_than_half_avg_idx]
            strength_pos = strength[greater_than_half_avg_idx]
            avg_pos = disp_average[greater_than_half_avg_idx]

            upper_bound1 = (base_pos - baseline_lower_bound)/avg_pos
            upper_bound2 = (baseline_upper_bound - base_pos)/(1-avg_pos)
            upper_bound = np.minimum(upper_bound1, upper_bound2)
            result[greater_than_half_avg_idx] = strength_pos*upper_bound*(max_displacement[greater_than_half_avg_idx]-avg_pos)

        if len(max_displacement[less_than_half_avg_idx]) > 0:
            base_neg = baseline_result[less_than_half_avg_idx]
            strength_neg = strength[less_than_half_avg_idx]
            avg_neg = disp_average[less_than_half_avg_idx]

            lower_bound1 = (base_neg-baseline_lower_bound)/avg_neg#(1- avg_neg)
            lower_bound2 = (baseline_upper_bound - base_neg)/(1-avg_neg)#avg_neg
            lower_bound = np.minimum(lower_bound1, lower_bound2)
            result[less_than_half_avg_idx] = strength_neg*lower_bound*(max_displacement[less_than_half_avg_idx]-avg_neg)

        return result

    def assembias_decorator(self, func):
        r""" Primary behavior of the `ContinuousAssembias` class.
        This method is used to introduce a boost/decrement of the baseline
        function in a manner that preserves the all-halo result.
        Any function with a semi-bounded range can be decorated with
        `assembias_decorator`. The baseline behavior can be anything
        whatsoever, such as mean star formation rate or
        mean halo occupation, provided it has a semi-bounded range.

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

        @wraps(func)
        def wrapper(*args, **kwargs):

            #################################################################################
            #  Retrieve the arrays storing prim_haloprop and sec_haloprop
            #  The control flow below is what permits accepting an input
            #  table or a directly inputting prim_haloprop and sec_haloprop arrays

            _HAS_table = False
            if 'table' in kwargs:
                try:
                    table = kwargs['table']
                    prim_haloprop = table[self.prim_haloprop_key]
                    sec_haloprop = table[self.sec_haloprop_key]
                    _HAS_table = True
                except KeyError:
                    msg = ("When passing an input ``table`` to the "
                           " ``assembias_decorator`` method,\n"
                           "the input table must have a column with name ``%s``"
                           "and a column with name ``%s``.\n")
                    raise HalotoolsError(msg % (self.prim_haloprop_key), self.sec_haloprop_key)
            else:
                try:
                    prim_haloprop = np.atleast_1d(kwargs['prim_haloprop'])
                except KeyError:
                    msg = ("\nIf not passing an input ``table`` to the "
                           "``assembias_decorator`` method,\n"
                           "you must pass ``prim_haloprop`` argument.\n")
                    raise HalotoolsError(msg)
                try:
                    sec_haloprop = np.atleast_1d(kwargs['sec_haloprop'])
                except KeyError:
                    msg = ("\nIf not passing an input ``table`` to the "
                           "``assembias_decorator`` method,\n"
                           "you must pass ``sec_haloprop`` argument")
                    raise HalotoolsError(msg)

            #################################################################################

            #  Compute the percentile to split on as a function of the input prim_haloprop
            split = self.percentile_splitting_function(prim_haloprop)

            #  Compute the baseline, undecorated result
            result = func(*args, **kwargs)

            #  We will only decorate values that are not edge cases,
            #  so first compute the mask for non-edge cases
            no_edge_mask = (
                (split > 0) & (split < 1) &
                (result > baseline_lower_bound) & (result < baseline_upper_bound)
            )
            #  Now create convenient references to the non-edge-case sub-arrays
            no_edge_result = result[no_edge_mask]
            no_edge_split = split[no_edge_mask]

            #  Retrieve percentile values (medians) if they've been precomputed. Else, compute them.
            if _HAS_table is True:
                if self.sec_haloprop_key + '_percentile_values' in table.keys():
                    no_edge_percentile_values = table[self.sec_haloprop_key + '_percentile_value'][no_edge_mask]
                else:
                    #  the value of sec_haloprop_percentile will be computed from scratch
                    no_edge_percentile_values = compute_conditional_percentile_values( p=no_edge_split,
                        prim_haloprop=prim_haloprop[no_edge_mask],
                        sec_haloprop=sec_haloprop[no_edge_mask]
                    )
            else:
                try:
                    percentiles = kwargs['sec_haloprop_percentile_values']
                    if custom_len(percentiles) == 1:
                        percentiles = np.zeros(custom_len(prim_haloprop)) + percentiles
                    no_edge_percentile_values = percentiles[no_edge_mask]
                except KeyError:
                    no_edge_percentile_values = compute_conditional_percentile_values(p=no_edge_split,
                        prim_haloprop=prim_haloprop[no_edge_mask],
                        sec_haloprop=sec_haloprop[no_edge_mask]
                    )

            #  NOTE I've removed the type 1 mask as it is not well-defined in this implementation
            #  this has all been rolled into the galprop_perturbation function

            #  normalize by max value away from percentile
            #  This ensures that the "slope" definition and boundaries are universal
            pv_sub_sec_haloprop = sec_haloprop[no_edge_mask] - no_edge_percentile_values

            if prim_haloprop[no_edge_mask].shape[0] == 0:
                perturbation = np.zeros_like(no_edge_result)
            else:
                perturbation = self._galprop_perturbation(
                    prim_haloprop=prim_haloprop[no_edge_mask],
                    sec_haloprop=pv_sub_sec_haloprop/np.max(np.abs(pv_sub_sec_haloprop)),
                    #sec_haloprop = compute_conditional_percentiles(prim_haloprop = prim_haloprop[no_edge_mask], sec_haloprop=sec_haloprop[no_edge_mask])-no_edge_split,
                    baseline_result=no_edge_result)

            no_edge_result += perturbation
            result[no_edge_mask] = no_edge_result

            return result

        return wrapper


class FreeSplitContinuousAssembias(ContinuousAssembias, FreeSplitAssembias):
    pass


