"""
This module contains the `~halotools.empirical_models.ContinuousAssembias` class.
The purpose of this class is to introduce a sigmoid-shaped assembly bias into
any method of any component model. It subclasses `HeavisideAssembias` and
 extends its features. Details can be found in
`McLaughlin et al 2017 (in prep)`_.
"""

from itertools import izip
from functools import wraps
from warnings import warn
from math import ceil

import numpy as np
from . import HeavisideAssembias
from .. import model_helpers
from ...custom_exceptions import HalotoolsError
from ...utils.array_utils import custom_len
#from ...utils.table_utils import compute_conditional_percentiles

__all__ = ('ContinuousAssembias', )
__author__ = ('Sean McLaughlin', )




def compute_prim_haloprop_bins(dlog10_prim_haloprop=0.05, **kwargs):
    """
    Parameters
    ----------
    prim_haloprop : array
        Array storing the value of the primary halo property column of the ``table``
        passed to ``compute_conditional_percentiles``.
    prim_haloprop_bin_boundaries : array, optional
        Array defining the boundaries by which we will bin the input ``table``.
        Default is None, in which case the binning will be automatically determined using
        the ``dlog10_prim_haloprop`` keyword.
    dlog10_prim_haloprop : float, optional
        Logarithmic spacing of bins of the mass-like variable within which
        we will assign secondary property percentiles. Default is 0.05.
    Returns
    --------
    output : array
        Numpy array of integers storing the bin index of the prim_haloprop bin
        to which each halo in the input table was assigned.
    """
    try:
        prim_haloprop = kwargs['prim_haloprop']
    except KeyError:
        msg = ("The ``compute_prim_haloprop_bins`` method "
               "requires the ``prim_haloprop`` keyword argument")
        raise HalotoolsError(msg)

    try:
        prim_haloprop_bin_boundaries = kwargs['prim_haloprop_bin_boundaries']
    except KeyError:
        lg10_min_prim_haloprop = np.log10(np.min(prim_haloprop)) - 0.001
        lg10_max_prim_haloprop = np.log10(np.max(prim_haloprop)) + 0.001
        num_prim_haloprop_bins = (lg10_max_prim_haloprop - lg10_min_prim_haloprop) / dlog10_prim_haloprop
        prim_haloprop_bin_boundaries = np.logspace(
            lg10_min_prim_haloprop, lg10_max_prim_haloprop,
            num=ceil(num_prim_haloprop_bins))

    # digitize the masses so that we can access them bin-wise
    # print "PHP",np.max(prim_haloprop), prim_haloprop_bin_boundaries[-1]
    output = np.digitize(prim_haloprop, prim_haloprop_bin_boundaries)

    # Use the largest bin for any points larger than the largest bin boundary,
    # and raise a warning if such points are found
    Nbins = len(prim_haloprop_bin_boundaries)
    if Nbins in output:
        msg = ("\n\nThe ``compute_prim_haloprop_bins`` function detected points in the \n"
               "input array of primary halo property that were larger than the largest value\n"
               "of the input ``prim_haloprop_bin_boundaries``. All such points will be assigned\n"
               "to the largest bin.\nBe sure that this is the behavior you expect for your application.\n\n")
        warn(msg)
        output = np.where(output == Nbins, Nbins - 1, output)

    return output

# TOOD update docs
def compute_conditional_averages(vals, **kwargs):
    """
    In bins of the ``prim_haloprop``, compute the average value of disp_func given
    the input ``table`` based on the value of ``sec_haloprop``.
    Parameters
    ----------
    disp_func: function, optional
        A kwarg that is the function to calculate the conditional average of.
        Default is 'lambda x: x' which will compute the average value of sec_haloprop
        in bins of prim_haloprop
    disp_func_kwargs: dictionary, optional
        kwargs for the disp_func. Default is an empty dictionary
    table : astropy table, optional
        a keyword argument that stores halo catalog being used to make mock galaxy population
        If a `table` is passed, the `prim_haloprop_key` and `sec_haloprop_key` keys
        must also be passed. If not passing a `table`, you must directly pass the
        `prim_haloprop` and `sec_haloprop` keyword arguments.
    prim_haloprop_key : string, optional
        Name of the column of the input ``table`` that will be used to access the
        primary halo property. `compute_conditional_percentiles` bins the ``table`` by
        ``prim_haloprop_key`` when computing the result.
    sec_haloprop_key : string, optional
        Name of the column of the input ``table`` that will be used to access the
        secondary halo property. `compute_conditional_percentiles` bins the ``table`` by
        ``prim_haloprop_key``, and in each bin uses the value stored in ``sec_haloprop_key``
        to compute the ``prim_haloprop``-conditioned rank-order percentile.
    prim_haloprop : array_like, optional
        Array storing the primary halo property used to bin the input points.
        If a `prim_haloprop` is passed, you must also pass a `sec_haloprop`.
    sec_haloprop : array_like, optional
        Array storing the secondary halo property used to define the conditional percentiles
        in each bin of `prim_haloprop`.
    prim_haloprop_bin_boundaries : array, optional
        Array defining the boundaries by which we will bin the input ``table``.
        Default is None, in which case the binning will be automatically determined using
        the ``dlog10_prim_haloprop`` keyword.
    dlog10_prim_haloprop : float, optional
        Logarithmic spacing of bins of the mass-like variable within which
        we will assign secondary property percentiles. Default is 0.2.
    Examples
    --------
    >>> from halotools.sim_manager import FakeSim
    >>> fakesim = FakeSim()
    >>> result = compute_conditional_percentiles(table = fakesim.halo_table, prim_haloprop_key = 'halo_mvir', sec_haloprop_key = 'halo_vmax')
    Notes
    -----
    The sign of the result is such that in bins of the primary property,
    *smaller* values of the secondary property
    receive *smaller* values of the returned percentile.
    """

    if 'table' in kwargs:
        table = kwargs['table']
        try:
            prim_haloprop_key = kwargs['prim_haloprop_key']
            prim_haloprop = table[prim_haloprop_key]
        except KeyError:
            msg = ("\nWhen passing an input ``table`` to the ``compute_conditional_percentiles`` method,\n"
                   "you must also pass ``prim_haloprop_key`` and ``sec_haloprop_key`` keyword arguments\n"
                   "whose values are column keys of the input ``table``\n")
            raise HalotoolsError(msg)
    else:
        try:
            prim_haloprop = kwargs['prim_haloprop']
        except KeyError:
            msg = ("\nIf not passing an input ``table`` to the ``compute_conditional_percentiles`` method,\n"
                   "you must pass a ``prim_haloprop`` and ``sec_haloprop`` arguments\n")
            raise HalotoolsError(msg)

    compute_prim_haloprop_bins_dict = {}
    compute_prim_haloprop_bins_dict['prim_haloprop'] = prim_haloprop
    try:
        compute_prim_haloprop_bins_dict['prim_haloprop_bin_boundaries'] = (
            kwargs['prim_haloprop_bin_boundaries'])
    except KeyError:
        pass
    try:
        compute_prim_haloprop_bins_dict['dlog10_prim_haloprop'] = kwargs['dlog10_prim_haloprop']
    except KeyError:
        pass
    prim_haloprop_bins = compute_prim_haloprop_bins(**compute_prim_haloprop_bins_dict)

    output = np.zeros_like(prim_haloprop)

    # sort on secondary property only with each mass bin
    bins_in_halocat = set(prim_haloprop_bins)
    for ibin in bins_in_halocat:
        indices_of_prim_haloprop_bin = np.where(prim_haloprop_bins == ibin)[0]
        # place the averages into the catalog
        output[indices_of_prim_haloprop_bin] = np.mean(vals[indices_of_prim_haloprop_bin])
    return output


def compute_conditional_percentile_values(p=0.5, **kwargs):
    """
    In bins of the ``prim_haloprop``, compute the percentile given by p of the input
     ``table`` based on the value of ``sec_haloprop``.
    Parameters
    ----------
    p: float or array
        Percentile to find. Float or array of floats between 0 and 1. Default is 0.5, the median.
    table : astropy table, optional
        a keyword argument that stores halo catalog being used to make mock galaxy population
        If a `table` is passed, the `prim_haloprop_key` and `sec_haloprop_key` keys
        must also be passed. If not passing a `table`, you must directly pass the
        `prim_haloprop` and `sec_haloprop` keyword arguments.
    prim_haloprop_key : string, optional
        Name of the column of the input ``table`` that will be used to access the
        primary halo property. `compute_conditional_percentiles` bins the ``table`` by
        ``prim_haloprop_key`` when computing the result.
    sec_haloprop_key : string, optional
        Name of the column of the input ``table`` that will be used to access the
        secondary halo property. `compute_conditional_percentiles` bins the ``table`` by
        ``prim_haloprop_key``, and in each bin uses the value stored in ``sec_haloprop_key``
        to compute the ``prim_haloprop``-conditioned rank-order percentile.
    prim_haloprop : array_like, optional
        Array storing the primary halo property used to bin the input points.
        If a `prim_haloprop` is passed, you must also pass a `sec_haloprop`.
    sec_haloprop : array_like, optional
        Array storing the secondary halo property used to define the conditional percentiles
        in each bin of `prim_haloprop`.
    prim_haloprop_bin_boundaries : array, optional
        Array defining the boundaries by which we will bin the input ``table``.
        Default is None, in which case the binning will be automatically determined using
        the ``dlog10_prim_haloprop`` keyword.
    dlog10_prim_haloprop : float, optional
        Logarithmic spacing of bins of the mass-like variable within which
        we will assign secondary property percentiles. Default is 0.2.
    Examples
    --------
    >>> from halotools.sim_manager import FakeSim
    >>> fakesim = FakeSim()
    >>> result = compute_conditional_percentiles(table = fakesim.halo_table, prim_haloprop_key = 'halo_mvir', sec_haloprop_key = 'halo_vmax')
    Notes
    -----
    The sign of the result is such that in bins of the primary property,
    *smaller* values of the secondary property
    receive *smaller* values of the returned percentile.
    """

    try:
        assert np.all(0 < p) and np.all(1 > p)
    except AssertionError:
        raise HalotoolsError("p must be a floating number between 0 and 1. ")

    if 'table' in kwargs:
        table = kwargs['table']
        try:
            prim_haloprop_key = kwargs['prim_haloprop_key']
            prim_haloprop = table[prim_haloprop_key]
            sec_haloprop_key = kwargs['sec_haloprop_key']
            sec_haloprop = table[sec_haloprop_key]
        except KeyError:
            msg = ("\nWhen passing an input ``table`` to the ``compute_conditional_percentiles`` method,\n"
                   "you must also pass ``prim_haloprop_key`` and ``sec_haloprop_key`` keyword arguments\n"
                   "whose values are column keys of the input ``table``\n")
            raise HalotoolsError(msg)
    else:
        try:
            prim_haloprop = kwargs['prim_haloprop']
            sec_haloprop = kwargs['sec_haloprop']
        except KeyError:
            msg = ("\nIf not passing an input ``table`` to the ``compute_conditional_percentiles`` method,\n"
                   "you must pass a ``prim_haloprop`` and ``sec_haloprop`` arguments\n")
            raise HalotoolsError(msg)

    compute_prim_haloprop_bins_dict = {}
    compute_prim_haloprop_bins_dict['prim_haloprop'] = prim_haloprop
    try:
        compute_prim_haloprop_bins_dict['prim_haloprop_bin_boundaries'] = (
            kwargs['prim_haloprop_bin_boundaries'])
    except KeyError:
        pass
    try:
        compute_prim_haloprop_bins_dict['dlog10_prim_haloprop'] = kwargs['dlog10_prim_haloprop']
    except KeyError:
        pass
    prim_haloprop_bins = compute_prim_haloprop_bins(**compute_prim_haloprop_bins_dict)

    output = np.zeros_like(prim_haloprop)

    if type(p) is float:
        p = np.array([p for i in xrange(len(prim_haloprop))])

    # sort on secondary property only with each mass bin
    bins_in_halocat = set(prim_haloprop_bins)
    for ibin, pp in izip(bins_in_halocat, p):
        indices_of_prim_haloprop_bin = np.where(prim_haloprop_bins == ibin)[0]
        # Find the indices that sort by the secondary property
        perc = np.percentile(sec_haloprop[indices_of_prim_haloprop_bin], pp * 100)

        # place the percentiles into the catalog
        output[indices_of_prim_haloprop_bin] = perc

    return output


class ContinuousAssembias(HeavisideAssembias):
    """
    Class used to extend the behavior of `HeavisideAssembias` for continuous distributions.
    """
    # TODO where to put in max value of slope?
    def _disp_func(self,sec_haloprop, slope):
        """
        Function define the value of \delta N_max as a function of the sec_haloprop distribution and a slope param
        :param sec_haloprop:
            The table of secondary haloprops. In general, the median (or some other percentile) is subtracted
            from this value first.
        :param slope:
            The "slope" parameter, which controls the rate of change between negative and positive displacements.
        :return:
            The displacement as a function of the sec_halopropr
        """
        return np.reciprocal(1 + np.exp(-(10 ** slope) * (sec_haloprop)))

    def _initialize_assembias_param_dict(self, assembias_strength=0.5, assembias_slope=0.0, **kwargs):
        '''
        For full documentation, see the Heaviside Assembias Declaration in Halotools.

        This function calls the superclass's version. Then, it adds the parameters from disp_func to the dict as well.
        This is taken by inspecting the function signature and no input are needed.
        :param assembias_strength:
            Strength of assembias.
        :param kwargs:
            Other kwargs. Details in superclass.
        :return: None
        '''
        super(ContinuousAssembias, self)._initialize_assembias_param_dict(assembias_strength=assembias_strength,
                                                                      **kwargs)
        # Accept float or iterable
        slope = assembias_slope
        try:
            iterator = iter(slope)
            slope = list(slope)
        except TypeError:
            slope = [slope]

        # assert it has the proper length
        # TODO do we want to allow variable slopes but constant strengths,
        # or vice/versa? Maybe later...
        if custom_len(self._assembias_strength_abscissa) != custom_len(slope):
            raise HalotoolsError("``assembias_strength`` and ``assembias_slope`` "
                                 "must have the same length")

        for ipar, val in enumerate(slope):
            self.param_dict[self._get_continuous_assembias_param_dict_key(ipar)] = val

    # I'd like to add this to control the min/max
    # However, it is complex so ... we'll see
    #@model_helpers.bounds_enforcing_decorator_factory(-1, 1)
    def assembias_slope(self, prim_haloprop):
        """
        Method returns the slope of continuous assembly bias as a function of the primary halo property.

        Parameters
        ----------
        prim_haloprop : array_like
            Array storing the primary halo property.

        Returns
        -------
        slope : array_like
            Slope of continuous assembly bias as a function of the input halo property.
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

    def _get_continuous_assembias_param_dict_key(self,ipar):
        '''
        '''
        return self._method_name_to_decorate + '_' + self.gal_type + '_assembias_slope' + str(ipar + 1)

    def _galprop_perturbation(self, **kwargs):
        """
        Method determines hwo much to boost the baseline function
        according to the strength of assembly bias and the min/max
        boost allowable by the requirement that the all-halo baseline
        function be preserved. The returned perturbation applies to type-1 halos.

        Uses the disp_func passed in duing perturbation
        :param kwargs:
            Required kwargs are:
                baseline_result
                prim_haloprop
                sec_haloprop
        :return: result, np.arry with dimensions of prim_haloprop detailing the perturbation.
        """

        # TODO why don't I need this?
        # lower_bound_key = 'lower_bound_' + self._method_name_to_decorate + '_' + self.gal_type
        # baseline_lower_bound = getattr(self, lower_bound_key)
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

        # evaluate my continuous modification
        strength = self.assembias_strength(prim_haloprop)
        slope = self.assembias_slope(prim_haloprop)

        # the average displacement acts as a normalization we need.

        max_displacement = self._disp_func(sec_haloprop=sec_haloprop, slope=slope)
        disp_average = compute_conditional_averages(max_displacement,prim_haloprop=prim_haloprop)

        # TODO i should study the corresponding section more closely,
        # this is quite different
        bound1 = baseline_result / disp_average
        bound2 = (baseline_upper_bound - baseline_result) / (baseline_upper_bound - disp_average)
        # stop NaN broadcasting
        bound1[np.isnan(bound1)] = np.inf
        bound2[np.isnan(bound2)] = np.inf

        bound = np.minimum(bound1, bound2)

        result = strength * bound * (max_displacement- disp_average)
        # print 'Perturbations'
        # print result.mean(),result.std(), result.max(), result.min()

        return result

    def assembias_decorator(self, func):
        # TODO update this commetn
        # keep everything the smae but get rid of the perturbation flip cuz it's not necessary.
        # also the mask is not necessary


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

        @wraps(func)
        def wrapper(*args, **kwargs):

            #################################################################################
            # Retrieve the arrays storing prim_haloprop and sec_haloprop
            # The control flow below is what permits accepting an input
            # table or a directly inputting prim_haloprop and sec_haloprop arrays
            _HAS_table = False
            # TODO use the table
            # TODO sec_percentile is not used here, remove refs to it
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
            # t0 = time()
            result = func(*args, **kwargs)
            # t1 = time()
            # print 'Baseline time:',t1 - t0

            # We will only decorate values that are not edge cases,
            # so first compute the mask for non-edge cases
            no_edge_mask = (
                (split > 0) & (split < 1) &
                (result > baseline_lower_bound) & (result < baseline_upper_bound)
            )
            # Now create convenient references to the non-edge-case sub-arrays
            no_edge_result = result[no_edge_mask]
            no_edge_split = split[no_edge_mask]

            if _HAS_table is True:
                if self.sec_haloprop_key + '_percentile_values' in list(table.keys()):
                    no_edge_percentile_values = table[self.sec_haloprop_key + '_percentile_value'][no_edge_mask]
                else:
                    # the value of sec_haloprop_percentile will be computed from scratch
                    no_edge_percentile_values = compute_conditional_percentile_values( no_edge_split,
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
                    no_edge_percentile_values = compute_conditional_percentile_values(no_edge_split,
                        prim_haloprop=prim_haloprop[no_edge_mask],
                        sec_haloprop=sec_haloprop[no_edge_mask]
                    )
            # NOTE I've removed the type 1 mask as it is not necessary
            # this has all been rolled into the galprop_perturbation function
            # TODO for consistancy maybe I should change it back if possible.
            # It's not cuz the notion of type1/type2 is meaniningless in CAB

            if prim_haloprop[no_edge_mask].shape[0] == 0:
                perturbation = np.zeros_like(no_edge_result)
            else:
                perturbation = self._galprop_perturbation(
                    prim_haloprop=prim_haloprop[no_edge_mask],
                    sec_haloprop=sec_haloprop[no_edge_mask] - no_edge_percentile_values,
                    baseline_result=no_edge_result)

            no_edge_result += perturbation
            # print result.mean(), result.std(), result.max(), result.min()
            result[no_edge_mask] = no_edge_result

            # print 'End Wrapper'
            # print result.mean(), result.std(), result.max(), result.min()

            return result

        return wrapper
