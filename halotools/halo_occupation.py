# -*- coding: utf-8 -*-
"""

This module contains the classes and methods used to model the 
connection between galaxies and the halos they inhabit. 
Classes (will) include support for HODs, CLFs, CSMFs, and 
(conditional) abundance matching, including designations 
for whether a galaxy is quenched or star-forming.


"""

__all__ = ['anatoly_concentration','cumulative_NFW_PDF','HOD_Model',
'Zheng07_HOD_Model','HOD_Quenching_Model','vdB03_Quenching_Model']
#from __future__ import (absolute_import, division, print_function,
#                        unicode_literals)

import numpy as np
from scipy.special import erf
from scipy.stats import poisson
import defaults

from astropy.extern import six
from abc import ABCMeta, abstractmethod
import warnings



def anatoly_concentration(logM):
    """ Power law fitting formula for the concentration-mass relation of Bolshoi host halos at z=0
    Taken from Klypin et al. 2011, arXiv:1002.3660v4, Eqn. 12.


    Parameters
    ----------
    logM : array 
        array of log10(Mvir) of halos in catalog

    Returns
    -------
    concentrations : array

    """
    
    masses = np.zeros(len(logM)) + 10.**logM
    c0 = 9.6
    Mpiv = 1.e12
    a = -0.075
    concentrations = c0*(masses/Mpiv)**a
    return concentrations

def cumulative_NFW_PDF(x,c):
    """ Analytically calculated integral to the NFW profile.
    Unit-normalized so that the result is a cumulative PDF. 

    Parameters
    ----------
    x : array 
        Values are in the range (0,1).
        Elements x = r/Rvir specify host-centric distances in the range 0 < r/Rvir < 1.

    c : array
        Concentration of halo whose profile is being tabulated.

    Returns
    -------
    pdf : array 
        List of floats in the range (0,1). 
        Value gives the probability of randomly drawing a radial position x = r/Rvir 
        from an NFW profile of input concentration c.
        Function is used in Monte Carlo realization of satellite positions, using 
        standard method of transformation of variables. 

    Synopsis
    --------
    Currently being used by mock.HOD_mock to generate satellite profiles. 

    """
    norm=np.log(1.+c)-c/(1.+c)
    return (np.log(1.+x*c) - x*c/(1.+x*c))/norm


@six.add_metaclass(ABCMeta)
class HOD_Model(object):
    """ Abstract base class for model parameters determining the HOD.

    Any HOD-based model is a subclass of the HOD_Model object. 
    All such models must specify how the expectation value 
    of both central and satellite galaxy occupations vary with host mass.
    Additionally, any HOD-based mock must specify the assumed concentration-mass relation.
    
    """
    
    def __init__(self):
        self.publication = []
        self.parameter_dict = {}

    @abstractmethod
    def mean_ncen(self,logM):
        """
        Expected number of central galaxies in a halo of mass logM.
        """
        raise NotImplementedError("mean_ncen is not implemented")

    @abstractmethod
    def mean_nsat(self,logM):
        """
        Expected number of satellite galaxies in a halo of mass logM.
        """
        raise NotImplementedError("mean_nsat is not implemented")

    @abstractmethod
    def mean_concentration(self,logM):
        """
        Concentration-mass relation assumed by the model. 
        Used to assign positions to satellites.
        """
        raise NotImplementedError("mean_concentration is not implemented")


class Zheng07_HOD_Model(HOD_Model):
    """ Subclass of HOD_Model object, where functional forms for occupation statistics 
    are taken from Zheng et al. 2007, arXiv:0703457.


    Parameters 
    ----------
    parameter_dict : dictionary, optional.
        Contains values for the parameters specifying the model.
        Dictionary keys should be 'logMmin_cen', 'sigma_logM', 'logM0_sat','logM1_sat','alpha_sat'.

    threshold : float, optional.
        Luminosity threshold of the mock galaxy sample. If specified, input value must agree with 
        one of the thresholds used in Zheng07 to fit HODs: 
        [-18, -18.5, -19, -19.5, -20, -20.5, -21, -21.5, -22].

    Note
    ----

    Concentration-mass relation is current set to be Anatoly's, though 
    this is not the relation used in Zheng07.

    """

    def __init__(self,parameter_dict=None,threshold=None):
        HOD_Model.__init__(self)

        self.publication.extend(['arXiv:0703457'])

        if parameter_dict is None:
            self.parameter_dict = self.published_parameters(threshold)
        else:
            #this should be more defensive. Fine for now.
            self.parameter_dict = parameter_dict


    def mean_ncen(self,logM):
        """
        Expected number of central galaxies in a halo of mass logM.
        See Equation 2 of arXiv:0703457.

        Parameters
        ----------        
        logM : array 
            array of log10(Mvir) of halos in catalog

        Returns
        -------
        mean_ncen : array
    
        Synopsis
        -------
        Mean number of central galaxies in a host halo of the specified mass. Values are restricted 0 <= mean_ncen <= 1.

        """
        if not isinstance(logM,np.ndarray):
            raise TypeError("Input logM to mean_ncen must be a numpy array")
        mean_ncen = 0.5*(1.0 + erf((logM - self.parameter_dict['logMmin_cen'])/self.parameter_dict['sigma_logM']))
        return mean_ncen

    def mean_nsat(self,logM):
        """Expected number of satellite galaxies in a halo of mass logM.
        See Equation 5 of arXiv:0703457.

        Parameters
        ----------
        logM : array 
            array of log10(Mvir) of halos in catalog

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
        """
        Concentration-mass relation of the model.

        Parameters 
        ----------

        logM : array_like

        Returns 
        -------

        concentrations : array_like
            Mean concentration of logM halos, using anatoly_concentration model.



        """

        concentrations = anatoly_concentration(logM)
        return concentrations

    def published_parameters(self,threshold=None):
        """
        Best-fit HOD parameters from Table 1 of Zheng et al. 2007.

        Parameters 
        ----------

        threshold : float
            Luminosity threshold defining the SDSS sample to which Zheng et al. 
            fit their HOD model. Must be agree with one of the published values: 
            [-18, -18.5, -19, -19.5, -20, -20.5, -21, -21.5, -22].

        Returns 
        -------

        parameter_dict : dict
            Dictionary of model parameters whose values have been set to 
            agree with the values taken from Table 1 of Zheng et al. 2007.


        """

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
class HOD_Quenching_Model(HOD_Model):
    """ Abstract base class for model parameters determining mock galaxy quenching. 
    This is a subclass of HOD_Mock, which additionally requires methods specifying 
    the quenched fractions of centrals and satellites.  
    
    """

    def __init__(self):

        HOD_Model.__init__(self)
        self.hod_model = None


    @abstractmethod
    def mean_quenched_fraction_centrals(self,logM):
        """
        Expected fraction of centrals that are quenched as a function of host halo mass logM.
        A required method for any HOD_Quenching_Model object.
        """
        raise NotImplementedError(
            "quenched_fraction_centrals is not implemented")

    @abstractmethod
    def mean_quenched_fraction_satellites(self,logM):
        """
        Expected fraction of satellites that are quenched as a function of host halo mass logM.
        A required method for any HOD_Quenching_Model object.
        """
        raise NotImplementedError(
            "quenched_fraction_satellites is not implemented")



class vdB03_Quenching_Model(HOD_Quenching_Model):
    """
    Subclass of HOD_Quenching_Model, providing a traditional HOD model of galaxy quenching, 
    in which quenching designation is purely determined by host halo virial mass.
    Approach is adapted from van den Bosch 2003. All-galaxy central and satellite occupation 
    statistics are specified first; Zheng07_HOD_Model is the default choice, 
    but any supported HOD_Mock object could be chosen. A quenching designation is subsequently 
    applied to the galaxies. 
    Thus in this class of models, 
    the central galaxy SMHM has no dependence on quenched/active designation.

    """

    def __init__(self,hod_model=Zheng07_HOD_Model,
        hod_parameter_dict=None,threshold=None,
        quenching_parameter_dict=None):

        if not isinstance(hod_model(threshold=threshold),HOD_Model):
            raise TypeError("input hod_model must be one of the supported HOD_Model objects defined in this module")

        # Run initialization from super class. Currently not doing much.
        HOD_Quenching_Model.__init__(self)


        self.hod_model = hod_model(
            parameter_dict = hod_parameter_dict,threshold = threshold)


        self.publication.extend(self.hod_model.publication)
        self.publication.extend(['arXiv:0210495v3'])

        # The baseline HOD parameter dictionary is already an attribute 
        # of self.hod_model. That dictionary needs to be joined with 
        # the dictionary storing the quenching model parameters. 
        # If a quenching parameter dictionary is passed to the constructor,
        # concatenate that passed dictionary with the existing hod_model dictionary.
        # Otherwise, choose the default quenching model parameter set in defaults.py 
        # This should be more defensive. Fine for now.
        if quenching_parameter_dict is None:
            quenching_parameter_dict = defaults.default_quenching_parameters
        self.parameter_dict = dict(self.hod_model.parameter_dict.items() + 
            quenching_parameter_dict.items())

        # The quenching parameter dictionary specifies the quenched fraction 
        # at the input abcissa. Use this information to determine the unique polynomial
        # satisfying those conditions, specified by a set of coefficients.
        self.central_quenching_polynomial_coefficients = (
            self.solve_for_quenching_polynomial_coefficients(
                self.parameter_dict['logM_quenching_abcissa'],
                self.parameter_dict['central_quenching_ordinates']))

        self.satellite_quenching_polynomial_coefficients = (
            self.solve_for_quenching_polynomial_coefficients(
                self.parameter_dict['logM_quenching_abcissa'],
                self.parameter_dict['satellite_quenching_ordinates']))

    def mean_ncen(self,logM):
        """
        Expected number of central galaxies in a halo of mass logM.
        The appropriate method is already bound to the self.hod_model object.

        Parameters
        ----------
        logM : array 
            array of log10(Mvir) of halos in catalog

        Returns
        -------
        mean_ncen : float or array
            Mean number of central galaxies in a host halo of the specified mass. 


        """
        mean_ncen = self.hod_model.mean_ncen(logM)
        return mean_ncen

    def mean_nsat(self,logM):
        """
        Expected number of satellite galaxies in a halo of mass logM.
        The appropriate method is already bound to the self.hod_model object.

        Parameters
        ----------
        logM : array 
            array of log10(Mvir) of halos in catalog

        Returns
        -------
        mean_nsat : float or array
            Mean number of satellite galaxies in a host halo of the specified mass. 


        """
        mean_nsat = self.hod_model.mean_nsat(logM)
        return mean_nsat

    def mean_concentration(self,logM):
        """ Concentration-mass relation assumed by the underlying HOD_Model object.
        The appropriate method is already bound to the self.hod_model object.

        Parameters 
        ----------
        logM : array 
            array of log10(Mvir) of halos in catalog

        Returns 
        -------
        concentrations : array

        """

        concentrations = self.hod_model.mean_concentration(logM)
        return concentrations

    def mean_quenched_fraction_centrals(self,logM):
        """
        Expected fraction of centrals that are quenched as a function of host halo mass logM.
        A required method for any HOD_Quenching_Model object.

        Parameters 
        ----------
        logM : array_like
            array of log10(Mvir) of halos in catalog

        Returns 
        -------
        mean_quenched_fractions : array_like
            Values of the quenched fraction in halos of the input logM

        Notes 
        -----
        The model assumes the quenched fraction is a polynomial in logM.
        The degree N quenching polynomial is determined by solving for 
        the unique polynomial with values given by the central quenching ordinates 
        at the logM abcissa. The coefficients of this polynomial are 
        solved for by the solve_for_quenching_polynomial_coefficients method.
        This function assumes that these coefficients have already been solved for and 
        bound to the input object as an attribute.
         
        """
 
        coefficients = self.central_quenching_polynomial_coefficients
        mean_quenched_fractions = self.quenching_polynomial(logM,coefficients)
        return mean_quenched_fractions

    def mean_quenched_fraction_satellites(self,logM):
        """
        Expected fraction of satellites that are quenched as a function of host halo mass logM.
        A required method for any HOD_Quenching_Model object.

        Parameters 
        ----------
        logM : array_like
            array of log10(Mvir) of halos in catalog

        Returns 
        -------
        mean_quenched_fractions : array_like
            Values of the quenched fraction in halos of the input logM

        Notes 
        -----
        The model assumes the quenched fraction is a polynomial in logM.
        The degree N quenching polynomial is determined by solving for 
        the unique polynomial with values given by the central quenching ordinates 
        at the logM abcissa. The coefficients of this polynomial are 
        solved for by the solve_for_quenching_polynomial_coefficients method.
        This function assumes that these coefficients have already been solved for and 
        bound to the input object as an attribute.
        
        """

        coefficients = self.satellite_quenching_polynomial_coefficients
        mean_quenched_fractions = self.quenching_polynomial(logM,coefficients)
        return mean_quenched_fractions

    def quenching_polynomial(self,logM,coefficients):
        """ Polynomial function used to specify the quenched fraction of centrals and satellites.

        Parameters 
        ----------
        logM : array_like
            values of the halo mass at which the quenched fraction is to be computed.

        coefficients : array_like
            Length N array containing the coefficients of the degree N polynomial governing the 
            expected quenched fraction.

        Returns 
        -------
        mean_quenched_fractions : array_like
            numpy array giving the expected quenched fraction for galaxies in halos of mass logM 

        """

        mean_quenched_fractions = np.zeros(len(logM))
        polynomial_degree = len(self.parameter_dict['logM_quenching_abcissa'])
        for n,coeff in enumerate(coefficients):
            mean_quenched_fractions += coeff*logM**n

        test_negative = mean_quenched_fractions < 0
        mean_quenched_fractions[test_negative] = 0
        test_exceeds_unity = mean_quenched_fractions > 1
        mean_quenched_fractions[test_exceeds_unity] = 1

        return mean_quenched_fractions

    def solve_for_quenching_polynomial_coefficients(self,logM_abcissa,ordinates):
        """ Given the quenched fraction for some halo masses, 
        returns standard form polynomial coefficients specifying quenching function.

        Parameters
        ----------
        logM_abcissa : array 
            Values are halo masses, the as abcissa of the polynomial.

        ordinates : array 
            Elements are the desired values of the polynomial when evaluated at the abcissa.

        Returns
        -------
        polynomial_coefficients : array 
            Elements are the coefficients determining the polynomial. 
            Element N of polynomial_coefficients gives the degree N coefficient.

        Synopsis
        --------
        Input arrays logM_abcissa and ordinates can in principle be of any dimension Ndim, 
        and there will be Ndim output coefficients.

        The input ordinates specify the desired values of the polynomial 
        when evaluated at the Ndim inputs specified by the input logM_abcissa.
        There exists a unique, order Ndim polynomial that produces the input 
        ordinates when the polynomial is evaluated at the input logM_abcissa.
        The coefficients of that unique polynomial are the output of the function. 

        Example
        -------
        A traditional quenching model, such as the one suggested in van den Bosch et al. 2003, 
        is a polynomial determining the mean quenched fraction as a function of halo mass logM.
        If we denote the output of solve_for_quenching_polynomial_coefficients as the array coeff, 
        then the unique polynomial F_quenched determined by F_quenched(logM_abcissa) = ordinates 
        is given by: 
        F_quenched(logM) = coeff[0] + coeff[1]*logM + coeff[2]*logM**2 + ... + coeff[len(logM)-1]*logM**(len(logM)-1)
    
        """

        ones = np.zeros(len(logM_abcissa)) + 1
        columns = ones
        for i in np.arange(len(logM_abcissa)-1):
            columns = np.append(columns,[logM_abcissa**(i+1)])
        quenching_model_matrix = columns.reshape(
            len(logM_abcissa),len(logM_abcissa)).transpose()

        polynomial_coefficients = np.linalg.solve(
            quenching_model_matrix,ordinates)

        return polynomial_coefficients






















