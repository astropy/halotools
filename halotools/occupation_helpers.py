# -*- coding: utf-8 -*-
"""

This module contains general purpose helper functions 
used by many of the hod model components.  

"""

__all__=['solve_for_polynomial_coefficients','format_parameter_keys']

import numpy as np
from copy import copy



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


def format_parameter_keys(input_parameter_dict,correct_initial_keys,gal_type):
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
    if set(initial_keys) != set(correct_initial_keys):
        raise KeyError("The parameter_dict passed to the initialization "
            "constructor does not contain the expected keys")

    output_parameter_dict = copy(input_parameter_dict)

    key_suffix = '_'+gal_type
    for old_key in initial_keys:
        new_key = old_key+key_suffix
        output_parameter_dict[new_key] = output_parameter_dict.pop(old_key)

    return output_parameter_dict


















