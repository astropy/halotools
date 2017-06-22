"""
This module contains the `~halotools.empirical_models.HeavisideAssembias` class.
The purpose of this class is to introduce step function-type assembly bias into
any method of any component model, as in
`Hearin et al 2015 decorated HODs <http://arxiv.org/abs/1512.03050>`_.
"""

import numpy as np
from warnings import warn

from .. import model_defaults, model_helpers

from ...utils.array_utils import custom_len
from ...custom_exceptions import HalotoolsError
from ...utils.table_utils import compute_conditional_percentiles
import collections

__all__ = ('HeavisideAssembias', )
__author__ = ('Andrew Hearin', )


class HeavisideAssembias(object):
    """ Class used as an orthogonal mix-in to introduce step function-style
    assembly-biased behavior into any component model.

    """
    def __init__(self, **kwargs):
        """
        No positional arguments accepted; all argument are strictly keyword arguments.

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
            governing the assembly bias. Must be a key in the table
            passed to the methods of `HeavisideAssembiasComponent`.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        split : float or list, optional
            Fraction or list of fractions between 0 and 1 defining how
            we split halos into two groupings based on
            their conditional secondary percentiles.
            Default is 0.5 for a constant 50/50 split.

        split_abscissa : list, optional
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
            Tuple providing the information about how elements of the input ``table``
            have been pre-divided into types. The first tuple entry must be a
            string giving the column name of the input ``table`` that provides the halo-typing.
            The second entry gives the value that will be stored in this column for halos
            that are above the percentile split, the third entry gives the value for halos
            below the split.

            If provided, you must ensure that the splitting of the ``table``
            was self-consistently performed with the
            input ``split``, or ``split_abscissa`` and ``split_ordinates``, or
            ``split_func`` keyword arguments.

        assembias_strength : float or list, optional
            Fraction or sequence of fractions between -1 and 1
            defining the assembly bias correlation strength.
            Default is 0.5.

        assembias_strength_abscissa : list, optional
            Values of the primary halo property at which the assembly bias strength is specified.
            Default is to assume a constant strength of 0.5. If passing a list, the strength
            will interpreted at the input ``assembias_strength_abscissa``.
            Default is to assume a constant strength of 0.5.

        loginterp : bool, optional
            If set to True, the interpolation will be done
            in the logarithm of the primary halo property,
            rather than linearly. Default is True.

        """
        self._interpret_constructor_inputs(**kwargs)

        self._decorate_baseline_method()

        self._methods_to_inherit.extend(['assembias_strength'])

        try:
            self.publications.append('arXiv:1512.03050')
        except:
            self.publications = ['arXiv:1512.03050']

    def _interpret_constructor_inputs(self, loginterp=True,
            sec_haloprop_key=model_defaults.sec_haloprop_key, **kwargs):
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

    def _set_percentile_splitting(self, split=0.5, **kwargs):
        """
        Method interprets the arguments passed to the constructor
        and sets up the interpolation scheme for how halos will be
        divided into two types as a function of the primary halo property.
        """

        if 'splitting_model' in kwargs:
            self.splitting_model = kwargs['splitting_model']
            func = getattr(self.splitting_model, kwargs['splitting_method_name'])
            if isinstance(func, collections.Callable):
                self._input_split_func = func
            else:
                raise HalotoolsError("Input ``splitting_model`` must have a callable function "
                    "named ``%s``" % kwargs['splitting_method_name'])
        elif 'split_abscissa' in list(kwargs.keys()):
            if custom_len(kwargs['split_abscissa']) != custom_len(split):
                raise HalotoolsError("``split`` and ``split_abscissa`` must have the same length")
            self._split_abscissa = kwargs['split_abscissa']
            self._split_ordinates = split
        else:
            try:
                self._split_abscissa = [2]
                self._split_ordinates = [split]
            except KeyError:
                msg = ("The _set_percentile_splitting method must at least be called with a ``split``"
                    "keyword argument, or alternatively ``split`` and ``split_abscissa`` arguments.")
                raise HalotoolsError(msg)

    def _initialize_assembias_param_dict(self, assembias_strength=0.5, **kwargs):
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

        if 'assembias_strength_abscissa' in kwargs:
            abscissa = kwargs['assembias_strength_abscissa']
            try:
                iterator = iter(abscissa)
                abscissa = list(abscissa)
            except TypeError:
                abscissa = [abscissa]
        else:
            abscissa = [2]

        if custom_len(abscissa) != custom_len(strength):
            raise HalotoolsError("``assembias_strength`` and ``assembias_strength_abscissa`` "
                "must have the same length")

        self._assembias_strength_abscissa = abscissa
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

    @model_helpers.bounds_enforcing_decorator_factory(0, 1)
    def percentile_splitting_function(self, prim_haloprop):
        """
        Method returns the fraction of halos that are ``type-2``
        as a function of the input primary halo property.

        Parameters
        -----------
        prim_haloprop : array_like
            Array storing the primary halo property.

        Returns
        -------
        split : float
            Fraction of ``type2`` halos at the input primary halo property.
        """

        if hasattr(self, '_input_split_func'):
            result = self._input_split_func(prim_haloprop=prim_haloprop)

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
                np.log10(self._split_abscissa), self._split_ordinates, k=3)
            result = spline_function(np.log10(prim_haloprop))
        else:
            spline_function = model_helpers.custom_spline(
                self._split_abscissa, self._split_ordinates, k=3)
            result = spline_function(prim_haloprop)

        return result

    @model_helpers.bounds_enforcing_decorator_factory(-1, 1)
    def assembias_strength(self, prim_haloprop):
        """
        Method returns the strength of assembly bias as a function of the primary halo property.

        The `bounds_enforcing_decorator_factory` guarantees that the assembly bias
        strength is enforced to be between -1 and 1.

        Parameters
        ----------
        prim_haloprop : array_like
            Array storing the primary halo property.

        Returns
        -------
        strength : array_like
            Strength of assembly bias as a function of the input halo property.
        """
        model_ordinates = (self.param_dict[self._get_assembias_param_dict_key(ipar)]
            for ipar in range(len(self._assembias_strength_abscissa)))
        spline_function = model_helpers.custom_spline(
            self._assembias_strength_abscissa, list(model_ordinates), k=3)

        if self._loginterp is True:
            result = spline_function(np.log10(prim_haloprop))
        else:
            result = spline_function(prim_haloprop)

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

        strength = self.assembias_strength(prim_haloprop)
        positive_strength_idx = strength > 0
        negative_strength_idx = strength < 0

        if len(baseline_result[positive_strength_idx]) > 0:
            base_pos = baseline_result[positive_strength_idx]
            split_pos = splitting_result[positive_strength_idx]
            type1_frac_pos = 1 - split_pos
            strength_pos = strength[positive_strength_idx]

            upper_bound1 = baseline_upper_bound - base_pos
            upper_bound2 = ((1 - type1_frac_pos)/type1_frac_pos)*(base_pos - baseline_lower_bound)
            upper_bound = np.minimum(upper_bound1, upper_bound2)
            result[positive_strength_idx] = strength_pos*upper_bound

        if len(baseline_result[negative_strength_idx]) > 0:
            base_neg = baseline_result[negative_strength_idx]
            split_neg = splitting_result[negative_strength_idx]
            type1_frac_neg = 1 - split_neg
            strength_neg = strength[negative_strength_idx]

            lower_bound1 = baseline_lower_bound - base_neg
            lower_bound2 = (1 - type1_frac_neg)/type1_frac_neg*(base_neg - baseline_upper_bound)
            lower_bound = np.maximum(lower_bound1, lower_bound2)
            result[negative_strength_idx] = np.abs(strength_neg)*lower_bound

        return result

    def assembias_decorator(self, func):
        """ Primary behavior of the `HeavisideAssembias` class.

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

        def wrapper(*args, **kwargs):

            #################################################################################
            # Retrieve the arrays storing prim_haloprop and sec_haloprop
            # The control flow below is what permits accepting an input
            # table or a directly inputting prim_haloprop and sec_haloprop arrays
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
                    if 'sec_haloprop_percentile' not in kwargs:
                        msg = ("\nIf not passing an input ``table`` to the "
                            "``assembias_decorator`` method,\n"
                            "you must pass either a ``sec_haloprop`` or "
                            "``sec_haloprop_percentile`` argument.\n")
                        raise HalotoolsError(msg)

            #################################################################################

            # Compute the fraction of type-2 halos as a function of the input prim_haloprop
            split = self.percentile_splitting_function(prim_haloprop)

            # Compute the baseline, undecorated result
            result = func(*args, **kwargs)

            # We will only decorate values that are not edge cases,
            # so first compute the mask for non-edge cases
            no_edge_mask = (
                (split > 0) & (split < 1) &
                (result > baseline_lower_bound) & (result < baseline_upper_bound)
                )
            # Now create convenient references to the non-edge-case sub-arrays
            no_edge_result = result[no_edge_mask]
            no_edge_split = split[no_edge_mask]

            #################################################################################
            # Compute the array type1_mask
            # This array will serve as a boolean mask that divides the halo sample into two subsamples
            # There are several possible ways that the type1_mask can be computed, depending on
            # what the decorator was passed as input

            if _HAS_table is True:
                # we were passed halo_type_tuple:
                if hasattr(self, 'halo_type_tuple'):
                    halo_type_key = self.halo_type_tuple[0]
                    halo_type1_val = self.halo_type_tuple[1]
                    type1_mask = table[halo_type_key][no_edge_mask] == halo_type1_val

                # the value of sec_haloprop_percentile is already stored as a column of the table
                elif self.sec_haloprop_key + '_percentile' in list(table.keys()):
                    no_edge_percentiles = table[self.sec_haloprop_key + '_percentile'][no_edge_mask]
                    type1_mask = no_edge_percentiles > no_edge_split
                else:
                    # the value of sec_haloprop_percentile will be computed from scratch
                    percentiles = compute_conditional_percentiles(
                        prim_haloprop=prim_haloprop,
                        sec_haloprop=sec_haloprop
                        )
                    no_edge_percentiles = percentiles[no_edge_mask]
                    type1_mask = no_edge_percentiles > no_edge_split
            else:
                try:
                    percentiles = kwargs['sec_haloprop_percentile']
                    if custom_len(percentiles) == 1:
                        percentiles = np.zeros(custom_len(prim_haloprop)) + percentiles
                except KeyError:
                    percentiles = compute_conditional_percentiles(
                        prim_haloprop=prim_haloprop,
                        sec_haloprop=sec_haloprop
                        )
                no_edge_percentiles = percentiles[no_edge_mask]
                type1_mask = no_edge_percentiles > no_edge_split

            # type1_mask has now been computed for all possible branchings
            #################################################################################

            perturbation = self._galprop_perturbation(
                    prim_haloprop=prim_haloprop[no_edge_mask],
                    baseline_result=no_edge_result,
                    splitting_result=no_edge_split)

            frac_type1 = 1 - no_edge_split
            frac_type2 = 1 - frac_type1
            perturbation[~type1_mask] *= (-frac_type1[~type1_mask] /
                (frac_type2[~type1_mask]))

            no_edge_result += perturbation
            result[no_edge_mask] = no_edge_result

            return result

        return wrapper
