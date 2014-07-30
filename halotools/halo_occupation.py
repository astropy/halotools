# -*- coding: utf-8 -*-
"""

.. module : halo_occupation 
    :synopsis: Modules to and methods to generate HOD-type mocks  

.. moduleauthor: Andrew Hearin <andrew.hearin@yale.edu>


"""
#from __future__ import (absolute_import, division, print_function,
#                        unicode_literals)

import numpy as np
from scipy.special import erf
from scipy.stats import poisson
import defaults

from astropy.extern import six
from abc import ABCMeta, abstractmethod
import warnings


def num_ncen(logM,hod_dict):
    """ Returns Monte Carlo-generated array of 0 or 1 specifying whether there is a central in the halo.

    Parameters
    ----------
    logM : float or array
    hod_dict : dictionary

    Returns
    -------
    num_ncen_array : int or array

    
    """

    num_ncen_array = np.array(mean_ncen(logM,hod_dict) > np.random.random(len(logM)),dtype=int)
    return num_ncen_array

def num_nsat(logM,hod_dict):
    '''  Returns Monte Carlo-generated array of integers specifying the number of satellites in the halo.

    Parameters
    ----------
    logM : float or array
    hod_dict : dictionary

    Returns
    -------
    num_nsat_array : int or array
        Values of array specify the number of satellites hosted by each halo.


    '''
    Prob_sat = mean_nsat(logM,hod_dict)
	# NOTE: need to cut at zero, otherwise poisson bails
    # BUG IN SCIPY: poisson.rvs bails if there are zeroes in a numpy array
    test = Prob_sat <= 0
    Prob_sat[test] = defaults.default_tiny_poisson_fluctuation

    num_nsat_array = poisson.rvs(Prob_sat)

    return num_nsat_array





def anatoly_concentration(logM):
    ''' 

    Parameters
    ----------
    logM: array of log halo masses

    Returns
    -------
    Concentrations deriving from c-M relation from Anatoly Klypin's 2011 Bolshoi paper.

    '''
    
    masses = np.zeros(len(logM)) + 10.**logM
    c0 = 12.0
    Mpiv = 1.e12
    a = -0.075
    concentrations = c0*(masses/Mpiv)**a
    return concentrations

def cumulative_NFW_PDF(r,c):
    """

    Parameters
    ----------
    r : list of N floats in the range (0,1)
    c : list of N concentrations

    Returns
    -------
    List of N floats in the range (0,1). These values are given by 
    the cumulative probability distribution function for an NFW profile,
    where the input r is the halo-centric distance scaled by the halo Rvir. 

    Synopsis
    --------
    Currently being used by mock.HOD_mock to generate satellite profiles. 

    Warning
    -------
    Basic API likely to change.

    """
    norm=np.log(1.+c)-c/(1.+c)
    return (np.log(1.+r*c) - r*c/(1.+r*c))/norm


@six.add_metaclass(ABCMeta)
class HOD_Model(object):
    """ Base class for model parameters determining the HOD.
    
    
    """
    
    def __init__(self,model_nickname):

        self.hod_model_nickname = model_nickname
#        self.parameter_dict = {}

    @abstractmethod
    def mean_ncen(self,logM):
        raise NotImplementedError("mean_ncen is not implemented")

    @abstractmethod
    def mean_nsat(self,logM):
        raise NotImplementedError("mean_nsat is not implemented")

    @abstractmethod
    def mean_concentration(self,logM):
        raise NotImplementedError("mean_concentration is not implemented")


class Zheng07_HOD_Model(HOD_Model):
    """ HOD model taken from Zheng et al. 2007

    """

    def __init__(self,parameter_dict=None,threshold=None):
        model_nickname = 'Zheng07'
        HOD_Model.__init__(self,model_nickname)

        self.publication = 'arXiv:0703457'

        if parameter_dict is None:
            self.parameter_dict = self.published_parameters(threshold)
        else:
            #this should be more defensive. Fine for now.
            self.parameter_dict = parameter_dict


    def mean_ncen(self,logM):
        """
        Expected number of central galaxies in a halo of mass 10**logM.

        Parameters
        ----------        
        logM : float or array
        hod_dict : dictionary

        Returns
        -------
        mean_ncen : float or array
    
        Synopsis
        -------
        Mean number of central galaxies in a host halo of the specified mass. Values are restricted 0 <= mean_ncen <= 1.

        """

        if not isinstance(logM,np.ndarray):
            raise TypeError("Input logM to mean_ncen must be a numpy array")
        mean_ncen = 0.5*(1.0 + erf((logM - self.parameter_dict['logMmin_cen'])/self.parameter_dict['sigma_logM']))
        return mean_ncen

    def mean_nsat(self,logM):
        """Expected number of satellite galaxies in a halo of mass 10**logM.

        Parameters
        ----------
        logM : float or array
        hod_dict : dictionary

        Returns
        -------
        mean_nsat : float or array
            Mean number of satellite galaxies in a host halo of the specified mass. 

    
        """

        if not isinstance(logM,np.ndarray):
            raise TypeError("Input logM to mean_ncen must be a numpy array")
        halo_mass = 10.**logM
        M0 = 10.**self.parameter_dict['logM0_sat']
        M1 = 10.**self.parameter_dict['logM1_sat']
        mean_nsat = np.zeros(len(logM),dtype='f8')
        idx_nonzero_satellites = (halo_mass - M0) > 0
        mean_nsat[idx_nonzero_satellites] = self.mean_ncen(
            logM[idx_nonzero_satellites])*(
            ((halo_mass[idx_nonzero_satellites] - M0)/M1)
            **self.parameter_dict['alpha_sat'])
        return mean_nsat

    def mean_concentration(self,logM):
        return anatoly_concentration(logM)

    def published_parameters(self,threshold=None):

        # Check to see whether a luminosity threshold has been specified
        # If not, use Mr = -19.5 as the default choice, and alert the user
        if threshold is None:
            warnings.warn("HOD threshold unspecified: setting to -19.5")
            self.threshold = -19.5
        else:
            # If a threshold is specified, require that it is a sensible type
            if isinstance(threshold,int) or isinstance(threshold,float):
                self.threshold = float(threshold)                
            else:
                raise TypeError("Input luminosity threshold must be a scalar")


        #Load tabulated data from Zheng et al. 2007, Table 1
        logMmin_cen_array = [11.35,11.46,11.6,11.75,12.02,12.3,12.79,13.38,14.22]
        sigma_logM_array = [0.25,0.24,0.26,0.28,0.26,0.21,0.39,0.51,0.77]
        logM0_array = [11.2,10.59,11.49,11.69,11.38,11.84,11.92,13.94,14.0]
        logM1_array = [12.4,12.68,12.83,13.01,13.31,13.58,13.94,13.91,14.69]
        alpha_sat_array = [0.83,0.97,1.02,1.06,1.06,1.12,1.15,1.04,0.87]
        # define the luminosity thresholds corresponding to the above data
        threshold_array = np.arange(-22,-17.5,0.5)
        threshold_array = threshold_array[::-1]

        threshold_index = np.where(threshold_array==self.threshold)[0]
        if len(threshold_index)==1:
            parameter_dict = {
            'logMmin_cen' : logMmin_cen_array[threshold_index[0]],
            'sigma_logM' : sigma_logM_array[threshold_index[0]],
            'logM0_sat' : logM0_array[threshold_index[0]],
            'logM1_sat' : logM1_array[threshold_index[0]],
            'alpha_sat' : alpha_sat_array[threshold_index[0]],
            'fconc' : 1.0 # multiplicative factor used to scale satellite concentrations (not actually a parameter in Zheng+07)
            }
        else:
            raise TypeError("Input luminosity threshold does not match any of the Table 1 values of Zheng et al. 2007.")

        return parameter_dict


@six.add_metaclass(ABCMeta)
class Quenching_Model(object):
    """ Base class for model parameters determining mock galaxy quenching.
    
    
    """

    def __init__(self,model_nickname):
        self.quenching_model_nickname = model_nickname

    @abstractmethod
    def mean_quenched_fraction_centrals(self,logM):
        raise NotImplementedError(
            "quenched_fraction_centrals is not implemented")

    @abstractmethod
    def mean_quenched_fraction_satellites(self,logM):
        raise NotImplementedError(
            "quenched_fraction_satellites is not implemented")


class vdB03_Quenching_Model(Quenching_Model):
    """
    Traditional HOD model of galaxy quenching, similar to van den Bosch 2003

    """

    def __init__(self,parameter_dict=None):
        model_nickname = 'vdB03'
        Quenching_Model.__init__(self,model_nickname)
        self.publication = 'arXiv:0210495v3'

        if parameter_dict is None:
            self.parameter_dict = defaults.default_quenching_parameters
        else:
            #this should be more defensive. Fine for now.
            self.parameter_dict = parameter_dict

        self.central_quenching_polynomial_coefficients = (
            self.solve_for_quenching_polynomial_coefficients(
                self.parameter_dict['logM_abcissa'],
                self.parameter_dict['central_ordinates']))

        self.satellite_quenching_polynomial_coefficients = (
            self.solve_for_quenching_polynomial_coefficients(
                self.parameter_dict['logM_abcissa'],
                self.parameter_dict['satellite_ordinates']))


    def mean_quenched_fraction_centrals(self,logM):
        coefficients = self.central_quenching_polynomial_coefficients
        mean_quenched_fractions = self.quenching_polynomial(logM,coefficients)
        return mean_quenched_fractions

    def mean_quenched_fraction_satellites(self,logM):
        coefficients = self.satellite_quenching_polynomial_coefficients
        mean_quenched_fractions = self.quenching_polynomial(logM,coefficients)
        return mean_quenched_fractions

    def quenching_polynomial(self,logM,coefficients):
        mean_quenched_fractions = np.zeros(len(logM))
        polynomial_degree = len(self.parameter_dict['logM_abcissa'])
        for n,coeff in enumerate(coefficients):
            mean_quenched_fractions += coeff*logM**n

        test_negative = mean_quenched_fractions < 0
        mean_quenched_fractions[test_negative] = 0
        test_exceeds_unity = mean_quenched_fractions > 1
        mean_quenched_fractions[test_exceeds_unity] = 1

        return mean_quenched_fractions

    def solve_for_quenching_polynomial_coefficients(self,logM_abcissa,quenched_fractions):
        ''' Given the quenched fraction for some halo masses, 
        returns standard form polynomial coefficients specifying quenching function.

        Parameters
        ----------
        logM : array of log halo masses, treated as abcissa
        quenched_fractions : array of values of the quenched fraction at the abcissa

        Returns
        -------
        quenched_fraction_polynomial_coefficients : array of coefficients determining the quenched fraction polynomial 

        Synopsis
        --------
        Input arrays logM and quenched_fraction can in principle be of any dimension Ndim, and there will be Ndim output coefficients.

        The input quenched_fractions specify the desired quenched fraction evaluated at the Ndim inputs for logM.
        There exists a unique, order Ndim polynomial that produces those quenched fractions evaluated at the points logM.
        The coefficients of that output polynomial are the output of the function, such that the quenching function is given by:
        F_quenched(logM) = coeff[0] + coeff[1]*logM + coeff[2]*logM**2 + ... + coeff[len(logM)-1]*logM**(len(logM)-1)
    
        '''

        ones = np.zeros(len(logM_abcissa)) + 1
        columns = ones
        for i in np.arange(len(logM_abcissa)-1):
            columns = np.append(columns,[logM_abcissa**(i+1)])
        quenching_model_matrix = columns.reshape(
            len(logM_abcissa),len(logM_abcissa)).transpose()

        quenched_fraction_polynomial_coefficients = np.linalg.solve(
            quenching_model_matrix,quenched_fractions)

        return quenched_fraction_polynomial_coefficients







