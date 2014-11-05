# -*- coding: utf-8 -*-
"""

This module contains general purpose helper functions 
used by many of the hod model components.  

"""

__all__=['solve_for_polynomial_coefficients','format_parameter_keys']

import numpy as np
from copy import copy
from utils.array_utils import array_like_length as aph_len

from scipy.interpolate import UnivariateSpline as spline


def solve_for_polynomial_coefficients(abcissa,ordinates):
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
        Element i of polynomial_coefficients gives the degree i coefficient.

    Notes
    --------
    Input arrays abcissa and ordinates can in principle be of any dimension Ndim, 
    and there will be Ndim output coefficients.

    The input ordinates specify the desired values of the polynomial 
    when evaluated at the Ndim inputs specified by the input abcissa.
    There exists a unique, order Ndim polynomial that returns the input 
    ordinates when the polynomial is evaluated at the input abcissa.
    The coefficients of that unique polynomial are the output of the function. 

    This function is used by many of the methods in this module. For example, suppose 
    that a model in which the quenched fraction is 
    :math:`F_{q}(logM = 12) = 0.25` and :math:`F_{q}(logM = 15) = 0.9`. 
    Then this function takes [12, 15] as the input abcissa, 
    [0.25, 0.9] as the input ordinates, 
    and returns the array :math:`[c_{0}, c_{1}]`. 
    The unique polynomial linear in :math:`log_{10}M`  
    that passes through the input ordinates and abcissa is given by 
    :math:`F(logM) = c_{0} + c_{1}*log_{10}logM`.
    
    """

    ones = np.zeros(len(abcissa)) + 1
    columns = ones
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
    output_ordinates = np.zeros(aph_len(input_abcissa))
    # Use coefficients to compute values of the inflection function polynomial
    for n,coeff in enumerate(coefficient_array):
        output_ordinates += coeff*input_abcissa**n

    return output_ordinates



def format_parameter_keys(input_parameter_dict,correct_initial_keys,
    gal_type, key_prefix=None):
    """ Simple method that tests whether the input keys are correct, 
    and if so, appends the key names with the galaxy type that they pertain to.

    Parameters 
    ----------
    input_parameter_dict : dictionary
        dictionary of parameters being used by the component model.

    correct_initial_keys : list
        list of strings providing the correct set of keys 
        that input_parameter_dict should have. 

    gal_type : string
        Galaxy type of the population being modeled by the component model. 
        This string will be appended to each key, with a leading underscore. 

    key_prefix : string, optional
        If not None, key_prefix will be prepended to 
        each dictionary key with a trailing underscore.


    Returns 
    -------
    output_parameter_dict : dictionary 
        Provided that the keys of input_parameter_dict are correct, 
        the output dictionary will be identical to the input, except 
        now each key has the gal_type string appended to it. 
    """

    initial_keys = input_parameter_dict.keys()

    # Check that the keys are correct
    # They should only be incorrect in cases where parameter_dict 
    # was passed to the initialization constructor
    test_correct_keys(initial_keys, correct_initial_keys)
#    if set(initial_keys) != set(correct_initial_keys):
#        raise KeyError("The parameter_dict passed to the initialization "
#            "constructor does not contain the expected keys")

    output_parameter_dict = copy(input_parameter_dict)

    key_suffix = '_'+gal_type
    for old_key in initial_keys:
        if key_prefix is not None:
            new_key = key_prefix+'_'+old_key+key_suffix
        else:
            new_key = old_key+key_suffix
        output_parameter_dict[new_key] = output_parameter_dict.pop(old_key)


    return output_parameter_dict

def test_correct_keys(input_keys,correct_keys):

    if set(input_keys) != set(correct_keys):
        raise KeyError("The parameter_dict passed to the initialization "
            "constructor does not contain the expected keys")


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
    periodic_coords = np.where(coords > box_length, coords - box_length, coords)
    periodic_coords = np.where(coords < 0, coords + box_length, coords)
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

    if aph_len(abcissa) > 1:
        abcissa = np.array(abcissa)
    if aph_len(values_inside_bins) > 1:
        values_inside_bins = np.array(values_inside_bins)
        bin_midpoints = np.array(bin_midpoints)

    # If there are multiple abcissa bins, make sure they do not overlap
    if aph_len(bin_midpoints)>1:
        midpoint_differences = np.diff(bin_midpoints)
        minimum_separation = midpoint_differences.min()
        if minimum_separation < bin_width:
            raise ValueError("Abcissa bins are not permitted to overlap")

    output = np.zeros(aph_len(abcissa)) + value_outside_bins

    if aph_len(bin_midpoints)==1:
        idx_abcissa_in_bin = np.where( (abcissa >= bin_midpoints - bin_width/2.) & (abcissa < bin_midpoints + bin_width/2.) )[0]
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


def aph_spline(table_abcissa, table_ordinates, k=0):
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


    if aph_len(table_abcissa) != aph_len(table_ordinates):
        raise TypeError("table_abcissa and table_abcissa must have the same length")

    if k >= aph_len(table_abcissa):
        raise ValueError("Input spline degree must be less than len(abcissa)")

    max_scipy_spline_degree = 5
    k = np.min([k, max_scipy_spline_degree])

    if k<0:
        raise ValueError("Spline degree must be non-negative")
    elif k==0:
        if aph_len(table_ordinates) != 1:
            raise TypeError("In spline_degree=0 edge case, "
                "table_abcissa and table_abcissa must be 1-element arrays")
        return lambda x : table_ordinates[0]
    else:
        spline_function = spline(table_abcissa, table_ordinates, k=k)
        return spline_function


















