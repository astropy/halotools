# -*- coding: utf-8 -*-
"""

This module contains general purpose helper functions 
used by many of the hod model components.  

"""

__all__=['solve_for_polynomial_coefficients','format_parameter_keys']

import numpy as np
from copy import copy
from ..utils.array_utils import array_like_length as custom_len

from scipy.interpolate import UnivariateSpline as spline

import model_defaults

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

def format_parameter_keys(input_param_dict,correct_initial_keys,
    gal_type, key_prefix=None):
    """ Simple method that tests whether the input keys are correct, 
    and if so, appends the key names with the galaxy type that they pertain to.

    Parameters 
    ----------
    input_param_dict : dictionary
        dictionary of parameters being used by the component model.

    correct_initial_keys : list
        list of strings providing the correct set of keys 
        that input_param_dict should have. 

    gal_type : string
        Galaxy type of the population being modeled by the component model. 
        This string will be appended to each key, with a leading underscore. 

    key_prefix : string, optional
        If not None, key_prefix will be prepended to 
        each dictionary key with a trailing underscore.


    Returns 
    -------
    output_param_dict : dictionary 
        Provided that the keys of input_param_dict are correct, 
        the output dictionary will be identical to the input, except 
        now each key has the gal_type string appended to it. 
    """

    initial_keys = input_param_dict.keys()

    # Check that the keys are correct
    # They should only be incorrect in cases where param_dict 
    # was passed to the initialization constructor
    test_correct_keys(initial_keys, correct_initial_keys)
#    if set(initial_keys) != set(correct_initial_keys):
#        raise KeyError("The param_dict passed to the initialization "
#            "constructor does not contain the expected keys")

    output_param_dict = copy(input_param_dict)

    key_suffix = '_'+gal_type
    for old_key in initial_keys:
        if key_prefix is not None:
            new_key = key_prefix+'_'+old_key+key_suffix
        else:
            new_key = old_key+key_suffix
        output_param_dict[new_key] = output_param_dict.pop(old_key)


    return output_param_dict

def test_correct_keys(input_keys,correct_keys):

    if set(input_keys) != set(correct_keys):
        raise KeyError("The param_dict passed to the initialization "
            "constructor does not contain the expected keys")

def test_repeated_keys(dict1, dict2):
    intersection = set(dict1) & set(dict2)
    assert intersection == set()


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
    # First correct negative coordinates
    periodic_coords = np.where(coords > box_length, 
        coords - box_length, coords)

    # Now correct coordinates that are too large
    periodic_coords = np.where(periodic_coords < 0, 
        periodic_coords + box_length, periodic_coords)
    
    return periodic_coords


def piecewise_heaviside(bin_midpoints, bin_width, values_inside_bins, value_outside_bins, abcissa):
    """ Piecewise heaviside function. 

    The output function values_inside_bins  
    when evaluated at points within bin_width/2 of bin_midpoints. Otherwise, 
    the output function returns value_outside_bins. 

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
        print("testing")
        for ii, x in enumerate(bin_midpoints):
            idx_abcissa_in_binii = np.where(
                (abcissa >= bin_midpoints[ii] - bin_width/2.) & 
                (abcissa < bin_midpoints[ii] + bin_width/2.)
                )[0]
            output[idx_abcissa_in_binii] = values_inside_bins[ii]

    return output


def custom_spline(table_abcissa, table_ordinates, k=0):
    """ Simple workaround to replace scipy's silly convention 
    for treating the spline_degree=0 edge case. 

    Parameters 
    ----------
    table_abcissa : array_like
        abcissa values defining the interpolation 

    table_ordinates : array_like
        ordinate values defining the interpolation 

    k : int 
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
        raise TypeError("table_abcissa and table_ordinates must have the same length \n"
            " len(table_abcissa) = %i and len(table_ordinates) = %i" % (len_abcissa, len_ordinates))

    if k >= custom_len(table_abcissa):
        len_abcissa = custom_len(table_abcissa)
        raise ValueError("Input spline degree k = %i "
            "must be less than len(abcissa) = %i" % (k, len_abcissa))

    max_scipy_spline_degree = 5
    k = np.min([k, max_scipy_spline_degree])

    if k<0:
        raise ValueError("Spline degree must be non-negative")
    elif k==0:
        if custom_len(table_ordinates) != 1:
            raise TypeError("In spline_degree=0 edge case, "
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
    func_argsort = func_indices.argsort()
    func_ranges = list(np.searchsorted(func_indices[func_argsort], range(len(func_table))))
    func_ranges.append(None)
    out = np.zeros_like(abcissa)
    for f, start, end in zip(func_table, func_ranges, func_ranges[1:]):
        ix = func_argsort[start:end]
        out[ix] = f(abcissa[ix])
    return out

def enforce_required_haloprops(haloprop_dict):
    required_prop_set = set(model_defaults.haloprop_key_dict)
    provided_prop_set = set(haloprop_dict)
    if not required_prop_set.issubset(provided_prop_set):
        raise KeyError("haloprop_key_dict must, at minimum, contain keys "
            "'prim_haloprop_key' and 'halo_boundary'")


def count_haloprops(haloprop_dict):
    trigger = 'haloprop_key'
    num_props = 0
    for key in haloprop_dict.keys():
        if key[-len(trigger):]==trigger:
            num_props += 1
    return num_props

def enforce_required_kwargs(required_kwargs, obj, **kwargs):
    for key in required_kwargs:
        if key in kwargs.keys():
            setattr(obj, key, kwargs[key])
        else:
            class_name = obj.__class__.__name__
            raise KeyError("``%s`` is a required keyword argument "
                "to instantiate the %s class" (key, class_name))











