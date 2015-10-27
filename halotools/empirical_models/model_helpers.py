# -*- coding: utf-8 -*-
"""

This module contains general purpose helper functions 
used by many of the hod model components.  

"""

__all__ = (
    ['GalPropModel', 'solve_for_polynomial_coefficients', 'polynomial_from_table', 
    'enforce_periodicity_of_box', 'custom_spline', 'create_composite_dtype', 'bind_default_kwarg_mixin_safe', 
    'custom_incomplete_gamma', 'bounds_enforcing_decorator_factory']
    )

__author__ = ['Andrew Hearin', 'Surhud More']
import numpy as np
from copy import copy
from astropy.extern import six
from abc import ABCMeta
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.special import gammaincc, gamma, expi
from warnings import warn 

from . import model_defaults

from ..utils.array_utils import custom_len, convert_to_ndarray
from ..custom_exceptions import HalotoolsError


@six.add_metaclass(ABCMeta)
class GalPropModel(object):
    """ Abstact container class for any model of any galaxy property. 
    """
    def __init__(self, galprop_key):
        self.galprop_key = galprop_key

def solve_for_polynomial_coefficients(abcissa, ordinates):
    """ Solves for coefficients of the unique, 
    minimum-degree polynomial that passes through 
    the input abcissa and attains values equal the input ordinates.  

    Parameters
    ----------
    abcissa : array 
        Elements are the abcissa at which the desired values of the polynomial 
        have been tabulated.

    ordinates : array 
        Elements are the desired values of the polynomial when evaluated at the abcissa.

    Returns
    -------
    polynomial_coefficients : array 
        Elements are the coefficients determining the polynomial. 
        Element i of polynomial_coefficients gives the degree i polynomial coefficient.

    Notes
    --------
    Input arrays abcissa and ordinates can in principle be of any dimension Ndim, 
    and there will be Ndim output coefficients.

    The input ordinates specify the desired values of the polynomial 
    when evaluated at the Ndim inputs specified by the input abcissa.
    There exists a unique, order Ndim polynomial that returns the input 
    ordinates when the polynomial is evaluated at the input abcissa.
    The coefficients of that unique polynomial are the output of the function. 

    As an example, suppose that a model in which the quenched fraction is 
    :math:`F_{q}(logM_{\\mathrm{halo}} = 12) = 0.25` and :math:`F_{q}(logM_{\\mathrm{halo}} = 15) = 0.9`. 
    Then this function takes [12, 15] as the input abcissa, 
    [0.25, 0.9] as the input ordinates, 
    and returns the array :math:`[c_{0}, c_{1}]`. 
    The unique polynomial linear in :math:`log_{10}M`  
    that passes through the input ordinates and abcissa is given by 
    :math:`F(logM) = c_{0} + c_{1}*log_{10}logM`.
    
    """

    columns = np.ones(len(abcissa))
    for i in np.arange(len(abcissa)-1):
        columns = np.append(columns,[abcissa**(i+1)])
    quenching_model_matrix = columns.reshape(
        len(abcissa),len(abcissa)).transpose()

    polynomial_coefficients = np.linalg.solve(
        quenching_model_matrix,ordinates)

    return np.array(polynomial_coefficients)

def polynomial_from_table(table_abcissa,table_ordinates,input_abcissa):
    """ Method to evaluate an input polynomial at the input_abcissa. 
    The input polynomial is determined by `solve_for_polynomial_coefficients` 
    from table_abcissa and table_ordinates. 

    Parameters
    ----------
    table_abcissa : array 
        Elements are the abcissa determining the input polynomial. 

    table_ordinates : array 
        Elements are the desired values of the input polynomial 
        when evaluated at table_abcissa

    input_abcissa : array 
        Points at which to evaluate the input polynomial. 

    Returns 
    -------
    output_ordinates : array 
        Values of the input polynomial when evaluated at input_abscissa. 

    """
    if not isinstance(input_abcissa, np.ndarray):
        input_abcissa = np.array(input_abcissa)
    coefficient_array = solve_for_polynomial_coefficients(
        table_abcissa,table_ordinates)
    output_ordinates = np.zeros(custom_len(input_abcissa))
    # Use coefficients to compute values of the inflection function polynomial
    for n,coeff in enumerate(coefficient_array):
        output_ordinates += coeff*input_abcissa**n

    return output_ordinates

def enforce_periodicity_of_box(coords, box_length):
    """ Function used to apply periodic boundary conditions 
    of the simulation, so that mock galaxies all lie in the range [0, Lbox].

    Parameters
    ----------
    coords : array_like
        float or ndarray containing a set of points with values ranging between 
        [-box_length, 2*box_length]
        
    box_length : float
        the size of simulation box (currently hard-coded to be Mpc/h units)

    Returns
    -------
    periodic_coords : array_like
        array with values and shape equal to input coords, 
        but with periodic boundary conditions enforced

    """    
    return coords % box_length


def piecewise_heaviside(bin_midpoints, bin_width, values_inside_bins, value_outside_bins, abcissa):
    """ Piecewise heaviside function. 

    The function returns values_inside_bins  
    when evaluated at points within bin_width/2 of bin_midpoints. 
    Otherwise, the output function returns value_outside_bins. 

    Parameters 
    ----------
    bin_midpoints : array_like 
        Length-Nbins array containing the midpoint of the abcissa bins. 
        Bin boundaries may touch, but overlapping bins will raise an exception. 

    bin_width : float  
        Width of the abcissa bins. 

    values_inside_bins : array_like 
        Length-Nbins array providing values of the desired function when evaluated 
        at a point inside one of the bins.

    value_outside_bins : float 
        value of the desired function when evaluated at any point outside the bins.

    abcissa : array_like 
        Points at which to evaluate binned_heaviside

    Returns 
    -------
    output : array_like  
        Values of the function when evaluated at the input abcissa

    """

    if custom_len(abcissa) > 1:
        abcissa = np.array(abcissa)
    if custom_len(values_inside_bins) > 1:
        values_inside_bins = np.array(values_inside_bins)
        bin_midpoints = np.array(bin_midpoints)

    # If there are multiple abcissa bins, make sure they do not overlap
    if custom_len(bin_midpoints)>1:
        midpoint_differences = np.diff(bin_midpoints)
        minimum_separation = midpoint_differences.min()
        if minimum_separation < bin_width:
            raise ValueError("Abcissa bins are not permitted to overlap")

    output = np.zeros(custom_len(abcissa)) + value_outside_bins

    if custom_len(bin_midpoints)==1:
        idx_abcissa_in_bin = np.where( 
            (abcissa >= bin_midpoints - bin_width/2.) & (abcissa < bin_midpoints + bin_width/2.) )[0]
        print(idx_abcissa_in_bin)
        output[idx_abcissa_in_bin] = values_inside_bins
    else:
        for ii, x in enumerate(bin_midpoints):
            idx_abcissa_in_binii = np.where(
                (abcissa >= bin_midpoints[ii] - bin_width/2.) & 
                (abcissa < bin_midpoints[ii] + bin_width/2.)
                )[0]
            output[idx_abcissa_in_binii] = values_inside_bins[ii]

    return output


def custom_spline(table_abcissa, table_ordinates, **kwargs):
    """ Convenience wrapper around scipy.InterpolatedUnivariateSpline, 
    written specifically to handle the edge case of a spline table being 
    built from a single point.  

    Parameters 
    ----------
    table_abcissa : array_like
        abcissa values defining the interpolation 

    table_ordinates : array_like
        ordinate values defining the interpolation 

    k : int, optional
        Degree of the desired spline interpolation

    Returns 
    -------
    output : object  
        Function object to use to evaluate the interpolation of 
        the input table_abcissa & table_ordinates 

    Notes 
    -----
    Only differs from the scipy.interpolate.UnivariateSpline for 
    the case where the input tables have a single element. The default behavior 
    of the scipy function is to raise an exception, which is silly: clearly 
    the desired behavior in this case is to simply return the input value 
    table_ordinates[0] for all values of the input abcissa. 

    """
    if custom_len(table_abcissa) != custom_len(table_ordinates):
        len_abcissa = custom_len(table_abcissa)
        len_ordinates = custom_len(table_ordinates)
        raise HalotoolsError("table_abcissa and table_ordinates must have the same length \n"
            " len(table_abcissa) = %i and len(table_ordinates) = %i" % (len_abcissa, len_ordinates))

    max_scipy_spline_degree = 5
    if 'k' in kwargs:
        k = np.min([custom_len(table_abcissa)-1, kwargs['k'], max_scipy_spline_degree])
    else:
        k = np.min([custom_len(table_abcissa)-1, max_scipy_spline_degree])

    if k<0:
        raise HalotoolsError("Spline degree must be non-negative")
    elif k==0:
        if custom_len(table_ordinates) != 1:
            raise HalotoolsError("In spline_degree=0 edge case, "
                "table_abcissa and table_abcissa must be 1-element arrays")
        return lambda x : np.zeros(custom_len(x)) + table_ordinates[0]
    else:
        spline_function = spline(table_abcissa, table_ordinates, k=k)
        return spline_function

def call_func_table(func_table, abcissa, func_indices):
    """ Returns the output of an array of functions evaluated at a set of input points 
    if the indices of required functions is known. 

    Parameters 
    ----------
    func_table : array_like 
        Length k array of function objects

    abcissa : array_like 
        Length Npts array of points at which to evaluate the functions. 

    func_indices : array_like 
        Length Npts array providing the indices to use to choose which function 
        operates on each abcissa element. Thus func_indices is an array of integers 
        ranging between 0 and k-1. 

    Returns 
    -------
    out : array_like 
        Length Npts array giving the evaluation of the appropriate function on each 
        abcissa element. 

    """
    func_table = convert_to_ndarray(func_table)
    abcissa = convert_to_ndarray(abcissa)
    func_indices = convert_to_ndarray(func_indices)
    
    func_argsort = func_indices.argsort()
    func_ranges = list(np.searchsorted(func_indices[func_argsort], range(len(func_table))))
    func_ranges.append(None)
    out = np.zeros_like(abcissa)
    for f, start, end in zip(func_table, func_ranges, func_ranges[1:]):
        ix = func_argsort[start:end]
        out[ix] = f(abcissa[ix])
    return out

def bind_required_kwargs(required_kwargs, obj, **kwargs):
    """ Method binds each element of ``required_kwargs`` to 
    the input object ``obj``, or raises and exception for cases 
    where a mandatory keyword argument was not passed to the 
    ``obj`` constructor.

    Used throughout the package when a required keyword argument 
    has no obvious default value. 

    Parameters 
    ----------
    required_kwargs : list 
        List of strings of the keyword arguments that are required 
        when instantiating the input ``obj``. 

    obj : object 
        The object being instantiated. 

    Notes 
    -----
    The `bind_required_kwargs` method assumes that each 
    required keyword argument should be bound to ``obj`` 
    as attribute with the same name as the keyword. 
    """
    for key in required_kwargs:
        if key in kwargs.keys():
            setattr(obj, key, kwargs[key])
        else:
            class_name = obj.__class__.__name__
            msg = (
                key + ' is a required keyword argument ' + 
                'to instantiate the '+class_name+' class'
                )
            raise KeyError(msg)

def create_composite_dtype(dtype_list):
    """ Find the union of the dtypes in the input list, and return a composite 
    dtype after verifying consistency of typing of possibly repeated fields. 

    Parameters 
    ----------
    dtype_list : list 
        List of dtypes with possibly repeated field names. 

    Returns 
    --------
    composite_dtype : dtype 
        Numpy dtype object composed of the union of the input dtypes. 

    Notes 
    -----
    Basically an awkward workaround to the fact 
    that numpy dtype objects are not iterable.
    """
    name_list = list(set([name for d in dtype_list for name in d.names]))

    composite_list = []
    for name in name_list:
        for dt in dtype_list:
            if name in dt.names:
                tmp = np.dtype(composite_list)
                if name in tmp.names:
                    if tmp[name].type == dt[name].type:
                        pass
                    else:
                        msg = ("Inconsistent dtypes for name = ``%s``.\n" 
                            "    dtype1 = %s\n    dtype2 = %s\n" % 
                            (name, tmp[name].type, dt[name].type))
                        raise HalotoolsError(msg)
                else:
                    composite_list.append((name, dt[name].type))
    composite_dtype = np.dtype(composite_list)
    return composite_dtype

def bind_default_kwarg_mixin_safe(obj, keyword_argument, constructor_kwargs, default_value):
    """ Function used to ensure that a keyword argument passed to the constructor 
    of an orthogonal mix-in class is not already an attribute bound to self.
    If it is safe to bind the keyword_argument to the object, 
    `bind_default_kwarg_mixin_safe` will do so.

    Parameters 
    ----------
    obj : class instance 
        Instance of the class to which we want to bind the input ``keyword_argument``.

    keyword_argument : string 
        name of the attribute that will be bound to the object if the action is deemed mix-in safe.

    constructor_kwargs : dict 
        keyword argument dictionary passed to the constructor of the input ``obj``.

    default_value : object 
        Whatever the default value for the attribute should be if ``keyword_argument`` does not 
        appear in kwargs nor is it already bound to the ``obj``.

    Notes 
    ------
    See the constructor of `~halotools.empirical_models.conc_mass_models.ConcMass` for a usage example.    
    """
    if hasattr(obj, keyword_argument):
        if keyword_argument in constructor_kwargs:
            clname = obj.__class__.__name__
            msg = ("Do not pass the  ``%s`` keyword argument "
                "to the constructor of the %s class when using the %s class "
                "as an orthogonal mix-in" % (keyword_argument, clname, clname))
            raise HalotoolsError(msg)
        else:
            pass
    else:
        if keyword_argument in constructor_kwargs:
            setattr(obj, keyword_argument, constructor_kwargs[keyword_argument])
        else:
            setattr(obj, keyword_argument, default_value)


def custom_incomplete_gamma(a, x):
    """ Incomplete gamma function. 
    
    For the case covered by scipy, a > 0, scipy is called. Otherwise the gamma function 
    recurrence relations are called, extending the scipy behavior. The only other difference from the 
    scipy function is that in `custom_incomplete_gamma` only supports the case where the input ``a`` is a scalar.
    
    Parameters
    -----------
    a : float 
    
    x : array_like 
    
    Returns 
    --------
    gamma : array_like 

    Examples 
    --------
    >>> a, x = 1, np.linspace(1, 10, 100)
    >>> g = custom_incomplete_gamma(a, x)
    >>> a = 0
    >>> g = custom_incomplete_gamma(a, x)
    >>> a = -1
    >>> g = custom_incomplete_gamma(a, x)
    """

    if a<0:
        return (custom_incomplete_gamma(a+1, x) - x**a * np.exp(-x))/a
    elif a==0:
        return -expi(-x)
    else:
        return gammaincc(a, x) * gamma(a)
custom_incomplete_gamma.__author__ = ['Surhud More']


def bounds_enforcing_decorator_factory(lower_bound, upper_bound, warning = True):
    """
    Function returns a decorator that can be used to clip the values 
    of an original function to produce a modified function whose 
    values are replaced by the input ``lower_bound`` and ``upper_bound`` whenever 
    the original function returns out of range values. 

    Parameters 
    -----------
    lower_bound : float or int 
        Lower bound defining the output decorator 

    upper_bound : float or int 
        Upper bound defining the output decorator 

    warning : bool, optional 
        If True, decorator will raise a warning for cases where the values of the 
        undecorated function fall outside the boundaries. Default is True. 

    Returns 
    --------
    decorator : object 
        Python decorator used to apply to any function for which you wish to 
        enforce that that the returned values of the original function are modified 
        to be bounded by ``lower_bound`` and ``upper_bound``. 

    Examples 
    --------
    >>> def original_function(x): return x + 4
    >>> lower_bound, upper_bound = 0, 5
    >>> decorator = bounds_enforcing_decorator_factory(lower_bound, upper_bound)
    >>> modified_function = decorator(original_function)
    >>> assert original_function(3) == 7
    >>> assert modified_function(3) == upper_bound
    >>> assert original_function(-10) == -6
    >>> assert modified_function(-10) == lower_bound
    >>> assert original_function(0) == modified_function(0) == 4

    """

    def decorator(input_func):

        def output_func(*args, **kwargs):

            unbounded_result = np.array(input_func(*args, **kwargs))
            lower_bounded_result = np.where(unbounded_result < lower_bound, lower_bound, unbounded_result)
            bounded_result = np.where(lower_bounded_result > upper_bound, upper_bound, lower_bounded_result)

            if warning is True:
                raise_warning = np.any(unbounded_result != bounded_result)
                if raise_warning is True:
                    func_name = input_func.__name__
                    msg = ("The " + func_name + " function \nreturned at least one value that was "
                        "outside the range (%.2f, %.2f)\n. The bounds_enforcing_decorator_factory "
                        "manually set all such values equal to \nthe appropriate boundary condition.\n")
                    warn(msg)

            return bounded_result

        return output_func

    return decorator






