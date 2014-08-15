# -*- coding: utf-8 -*-
"""

This module contains the classes and methods used to model the 
connection between galaxies and the halos they inhabit. 
Classes (will) include support for HODs, CLFs, CSMFs, and 
(conditional) abundance matching. Features will include designations 
for whether a galaxy is quenched or star-forming, or in the case 
of conditional abundance matching models, the full distributions 
of secondary galaxy properties such as SFR, color, morphology, etc.


"""

__all__ = ['anatoly_concentration','cumulative_NFW_PDF','HOD_Model',
'Zheng07_HOD_Model','HOD_Quenching_Model','vdB03_Quenching_Model',
'Assembly_Biased_HOD_Model','Assembly_Biased_HOD_Quenching_Model',
'Satcen_Correlation_Polynomial_HOD_Model']
#from __future__ import (absolute_import, division, print_function,
#                        unicode_literals)

import numpy as np
from scipy.special import erf
from scipy.stats import poisson
import defaults

from astropy.extern import six
from abc import ABCMeta, abstractmethod, abstractproperty
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
    """ Integral of the NFW profile.
    Unit-normalized so that the result is a cumulative PDF. 

    Parameters
    ----------
    x : array_like
        Values are in the range (0,1).
        Elements x = r/Rvir specify host-centric distances in the range 0 < r/Rvir < 1.

    c : array_like
        Concentration of halo whose profile is being tabulated.

    Returns
    -------
    pdf : array 
        List of floats in the range (0,1). 
        Value gives the probability of randomly drawing a radial position x = r/Rvir 
        from an NFW profile of input concentration c.
        Function is used in Monte Carlo realization of satellite positions, using 
        standard method of transformation of variables. 

    Notes
    --------
    Currently being used by mock.HOD_mock to generate 
    Monte Carlo realizations of satellite profiles. 

    """
    c = np.array(c)
    x = np.array(x)
    norm=np.log(1.+c)-c/(1.+c)
    return (np.log(1.+x*c) - x*c/(1.+x*c))/norm

def solve_for_polynomial_coefficients(abcissa,ordinates):
    """ Given the quenched fraction for some halo masses, 
    returns standard form polynomial coefficients specifying quenching function.

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
        Element N of polynomial_coefficients gives the degree N coefficient.

    Notes
    --------
    Input arrays abcissa and ordinates can in principle be of any dimension Ndim, 
    and there will be Ndim output coefficients.

    The input ordinates specify the desired values of the polynomial 
    when evaluated at the Ndim inputs specified by the input abcissa.
    There exists a unique, order Ndim polynomial that produces the input 
    ordinates when the polynomial is evaluated at the input abcissa.
    The coefficients of that unique polynomial are the output of the function. 

    This function is used by many of the methods below. For example, suppose 
    that a model in which the quenched fraction is 0.25 at logM = 12 and 0.9 at 
    logM = 15. Then this function takes [12, 15] and [0.25, 0.9] as input, and 
    returns the array [coeff0,coeff1]. The unique polynomial linear in logM 
    that smoothly varies between the desired quenched fraction values is given by 
    F(logM) = coeff0 + coeff1*logM.
    
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



@six.add_metaclass(ABCMeta)
class HOD_Model(object):
    """ Abstract base class for model parameters determining the HOD.

    Any HOD-based model is a subclass of the HOD_Model object. 
    All such models must provide their own specific functional forms 
    for how the expectation value of both central and satellite 
    galaxy occupations vary with host mass. Additionally, 
    any HOD-based mock must specify the assumed concentration-mass relation.
    
    """
    
    def __init__(self,parameter_dict=None,threshold=None):
        self.publication = []
        self.parameter_dict = parameter_dict
        self.threshold = threshold

    @abstractmethod
    def mean_ncen(self,primary_halo_property):
        """
        Expected number of central galaxies in a halo of mass logM.
        """
        raise NotImplementedError("mean_ncen is not implemented")

    @abstractmethod
    def mean_nsat(self,primary_halo_property):
        """
        Expected number of satellite galaxies in a halo of mass logM.
        """
        raise NotImplementedError("mean_nsat is not implemented")

    @abstractmethod
    def mean_concentration(self,primary_halo_property):
        """
        Concentration-mass relation assumed by the model. 
        Used to assign positions to satellites.
        """
        raise NotImplementedError("mean_concentration is not implemented")

    @abstractproperty
    def primary_halo_property_key(self):
        raise NotImplementedError("primary_halo_property_key "
            "needs to be implemented to ensure self-consistency "
            "of baseline HOD and assembly-biased HOD model features")



class Zheng07_HOD_Model(HOD_Model):
    """ Subclass of HOD_Model object, where functional forms for occupation statistics 
    are taken from Zheng et al. 2007, arXiv:0703457.


    Parameters 
    ----------
    parameter_dict : dictionary, optional.
        Contains values for the parameters specifying the model.
        Dictionary keys should be 
        'logMmin_cen', 'sigma_logM', 'logM0_sat','logM1_sat','alpha_sat'.

    threshold : float, optional.
        Luminosity threshold of the mock galaxy sample. 
        If specified, input value must agree with 
        one of the thresholds used in Zheng07 to fit HODs: 
        [-18, -18.5, -19, -19.5, -20, -20.5, -21, -21.5, -22].

    Notes
    -----

    Concentration-mass relation is current set to be Anatoly's, though 
    this is not the relation used in Zheng07.

    """

    def __init__(self,parameter_dict=None,threshold=None):
        HOD_Model.__init__(self)

        self.publication.extend(['arXiv:0703457'])

        if parameter_dict is None:
            self.parameter_dict = self.published_parameters(threshold)
        self.require_correct_keys()

    @property 
    def primary_halo_property_key(self):
        return 'MVIR'

    def mean_ncen(self,logM):
        """ Expected number of central galaxies in a halo of mass logM.
        See Equation 2 of arXiv:0703457.

        Parameters
        ----------        
        logM : array 
            array of log10(Mvir) of halos in catalog

        Returns
        -------
        mean_ncen : array
    
        Notes
        -------
        Mean number of central galaxies in a host halo of the specified mass. 
        Values are restricted 0 <= mean_ncen <= 1.

        """
        logM = np.array(logM)
        mean_ncen = 0.5*(1.0 + erf(
            (logM - self.parameter_dict['logMmin_cen'])/self.parameter_dict['sigma_logM']))
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

        logM = np.array(logM)
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

        logM = np.array(logM)

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

    def require_correct_keys(self):
        correct_set_of_keys = set(self.published_parameters(threshold = -20).keys())
        if set(self.parameter_dict.keys()) != correct_set_of_keys:
            raise TypeError("Set of keys of input parameter_dict do not match the set of keys required by the model")
        pass




@six.add_metaclass(ABCMeta)
class Assembly_Biased_HOD_Model(HOD_Model):
    """ Abstract base class for any HOD model with assembly bias. 

    In this class of models, central and/or satellite mean occupation depends on some primary  
    property (such as Mvir) and is modulated by some secondary property 
    (such as halo formation time). 

    """

    def __init__(self):

        # Executing the __init__ of the abstract base class HOD_Model 
        #sets self.parameter_dict to None, self.threshold to None, 
        # and self.publication to []
        HOD_Model.__init__(self)

    @abstractproperty
    def baseline_hod_model(self):
        pass

    @abstractproperty
    def primary_halo_property_key(self):
        raise NotImplementedError("primary_halo_property_key "
            "needs to be implemented to ensure self-consistency "
            "of baseline HOD and assembly-biased HOD model features")

    @abstractmethod
    def central_destruction(self,primary_halo_property,halo_type):
        """ Determines the excess probability that ``type 0`` 
        halos of logM host a central galaxy. """
        raise NotImplementedError(
            "central_destruction is not implemented")

    @abstractmethod
    def satellite_destruction(self,primary_halo_property,halo_type):
        """ Determines the excess probability that ``type 0`` 
        halos of logM host a satellite galaxy. """
        raise NotImplementedError(
            "satellite_destruction is not implemented")

    @abstractmethod
    def halotype_fraction_centrals(self,primary_halo_property,halo_type):
        """ Determines the fractional representation of host halo 
        type 1 as a function of primary_halo_property, as pertains to centrals. 

        Halo types can be either given by fixed-Mvir rank-orderings 
        of the host halos, or by the input occupation statistics functions.

         """
        raise NotImplementedError(
            "halo_type_fraction_centrals is not implemented")

    @abstractmethod
    def halotype_fraction_satellites(self,primary_halo_property,halo_type):
        """ Determines the fractional representation of host halo 
        type 1 as a function of primary_halo_property, as pertains to satellites.

        Halo types can be either given by fixed-Mvir rank-orderings 
        of the host halos, or by the input occupation statistics functions.

         """
        raise NotImplementedError(
            "halo_type_fraction_satellites is not implemented")

    def maximum_central_destruction(self,primary_halo_property,halo_type):
        """ The maximum possible boost of central galaxy abundance 
        in type 1 halos. 
        """

        halotype_fraction = self.halotype_fraction_centrals(primary_halo_property,halo_type)
        nonzero_fraction = halotype_fraction > 0
        maximum_destruction = np.zeros(len(primary_halo_property))
        maximum_destruction[nonzero_fraction] = 1./halotype_fraction[nonzero_fraction]
        return maximum_destruction

    def maximum_satellite_destruction(self,primary_halo_property,halo_type):
        """ The maximum possible boost of satellite galaxy abundance 
        in type 1 halos. 
        """
        halotype_fraction = self.halotype_fraction_satellites(primary_halo_property,halo_type)
        nonzero_fraction = halotype_fraction > 0
        maximum_destruction = np.zeros(len(primary_halo_property))
        maximum_destruction[nonzero_fraction] = 1./halotype_fraction[nonzero_fraction]
        return maximum_destruction

    def mean_ncen(self,primary_halo_property,halo_type):
        """ Override """
        return self.central_destruction(primary_halo_property,halo_type)*(
            self.baseline_hod_model.mean_ncen(primary_halo_property))

    def mean_nsat(self,primary_halo_property,halo_type):
        """ Override """
        return self.satellite_destruction(primary_halo_property,halo_type)*(
            self.baseline_hod_model.mean_nsat(primary_halo_property))

    def halo_type(self,primary_halo_property,secondary_halo_property):
        """ Determines the assembly bias type of the input halos.

        Bins input halos by primary_halo_property, splits each bin 
        according to the value of the model's halo_type_fraction in the bin, 
        and assigns halo type 0 (1) to the halos below (above) the split.

        """
        pass



class Satcen_Correlation_Polynomial_HOD_Model(Assembly_Biased_HOD_Model):
    """ HOD-style model in which satellite abundance 
    is correlated with the presence of a central galaxy.
    """

    def __init__(self,baseline_hod_model=Zheng07_HOD_Model,
            baseline_hod_parameter_dict=None,threshold=None,
            assembias_parameter_dict=None):


        baseline_hod_model_instance = baseline_hod_model(threshold=threshold)
        if not isinstance(baseline_hod_model_instance,HOD_Model):
            raise TypeError(
                "Input baseline_hod_model must be one of "
                "the supported HOD_Model objects defined in this module or by the user")
        # Temporarily store the baseline HOD model object
        # into a "private" attribute. This is a clunky workaround
        # to python's awkward conventions for required abstract properties
        self._baseline_hod_model = baseline_hod_model_instance

        # Executing the __init__ of the abstract base class Assembly_Biased_HOD_Model 
        # does nothing besides executing the __init__ of the abstract base class HOD_Model 
        # Executing the __init__ of the abstract base class HOD_Model 
        # sets self.parameter_dict to None, self.threshold to None, 
        # and self.publication to []        
        Assembly_Biased_HOD_Model.__init__(self)


        self.threshold = threshold

        self.publication.extend(self._baseline_hod_model.publication)
        self.baseline_hod_parameter_dict = self._baseline_hod_model.parameter_dict

        if assembias_parameter_dict == None:
            self.assembias_parameter_dict = defaults.default_satcen_parameters
        else:
            # If the user supplies a dictionary of assembly bias parameters, 
            # require that the set of keys is correct
            self.require_correct_keys(assembias_parameter_dict)
            # If correct, bind the input dictionary to the instance.
            self.assembias_parameter_dict = assembias_parameter_dict

        # combine baseline HOD parameters and assembly bias parameters
        # into the same dictionary
        self.parameter_dict = dict(self.baseline_hod_parameter_dict.items() + 
            self.assembias_parameter_dict.items())


    @property
    def baseline_hod_model(self):
        return self._baseline_hod_model

    @property 
    def primary_halo_property_key(self):
        return 'MVIR'

    def mean_concentration(self,primary_halo_property):
        """ Concentration-halo relation assumed by the underlying HOD_Model object.
        The appropriate method is already bound to the self.baseline_hod_model object.

        Parameters 
        ----------
        primary_halo_property : array_like
            array of primary halo property governing the occupation statistics 

        Returns 
        -------
        concentrations : numpy array

        """

        concentrations = self.baseline_hod_model.mean_concentration(primary_halo_property)
        return concentrations

    def central_destruction(self,primary_halo_property,halo_types):
        """ Determines the excess probability due to assembly bias that 
        halos host a central galaxy. """

        def central_destruction_type1(self,primary_halo_property):

            coefficient_array = solve_for_polynomial_coefficients(
                self.parameter_dict['assembias_abcissa'],
                self.parameter_dict['central_assembias_ordinates'])

            # Initialize array that the function will return
            output_destruction_function = np.zeros(len(primary_halo_property))

            # Use coefficients to compute values of the destruction function polynomial
            for n,coeff in enumerate(coefficient_array): 
                output_destruction_function += coeff*primary_halo_property**n

            # Apply baseline HOD constraint to central_destruction_type1
            array_of_ones = np.zeros(len(output_destruction_function)) + 1
            # First, apply the bound from above
            test_type1_exceeds_maximum = output_destruction_function > (
                self.maximum_central_destruction(primary_halo_property,array_of_ones))
            output_destruction_function[test_type1_exceeds_maximum] = (
                self.maximum_central_destruction(
                    primary_halo_property[test_type1_exceeds_maximum],array_of_ones))
            # Second, apply the bound from below
            test_type1_negative = output_destruction_function < 0 
            output_destruction_function[test_type1_negative] = 0

            return np.array(output_destruction_function)

        def central_destruction_type0(self,primary_halo_property):

            array_of_ones = np.zeros(len(primary_halo_property)) + 1
            type1_fraction = self.halotype_fraction_centrals(primary_halo_property,array_of_ones)
            destruction_type1 = central_destruction_type1(self,
                primary_halo_property)
            type0_fraction = 1 - type1_fraction

            output_destruction_function = np.zeros(len(primary_halo_property))

            # Ensure that we do not divide by zero
            test_positive = type0_fraction > np.zeros(len(type0_fraction))
            # type0 destruction is defined in terms of type1 destruction 
            # to ensure self-consistency of occupation statistics
            output_destruction_function[test_positive] = (
                (1-type1_fraction[test_positive]*
                destruction_type1[test_positive])/
                type0_fraction[test_positive])

            # Apply baseline HOD constraint to central_destruction_type1
            # First, apply the bound from above
            array_of_zeros = array_of_ones-1
            test_type0_exceeds_maximum = output_destruction_function > (
                self.maximum_central_destruction(primary_halo_property,array_of_zeros))
            output_destruction_function[test_type0_exceeds_maximum] = (
                self.maximum_central_destruction(
                    primary_halo_property[test_type0_exceeds_maximum],
                    array_of_zeros[test_type0_exceeds_maximum]))

            # Note that bound from below is automatically applied 
            # since we initialized an output array of zeros
            # and only filled entries that would be

            return np.array(output_destruction_function)


        # Run a blind search on values of halo_types
        # This could possibly be a source of speedup if the 
        # input arrays were pre-sorted
        idx_halo_type0 = np.where(halo_types==0)[0]
        idx_halo_type1 = np.where(halo_types==1)[0]

        central_destruction = np.zeros(len(primary_halo_property))

        central_destruction[idx_halo_type0] = (
            central_destruction_type0(self,primary_halo_property[idx_halo_type0]))
        central_destruction[idx_halo_type1] = (
            central_destruction_type1(self,primary_halo_property[idx_halo_type1]))

        return np.array(central_destruction)

    def satellite_destruction(self,primary_halo_property,halo_types):
        """ Determines the excess probability due to assembly bias that 
        halos host a satellite galaxy. """

        def satellite_destruction_type1(self,primary_halo_property):

            coefficient_array = solve_for_polynomial_coefficients(
                self.parameter_dict['assembias_abcissa'],
                self.parameter_dict['satellite_assembias_ordinates'])

            # Initialize array that the function will return
            output_destruction_function = np.zeros(len(primary_halo_property))

            # Use coefficients to compute values of the destruction function polynomial
            for n,coeff in enumerate(coefficient_array): 
                output_destruction_function += coeff*primary_halo_property**n

            # Apply baseline HOD constraint to satellite_destruction_type1
            array_of_ones = np.zeros(len(output_destruction_function)) + 1
            # First, apply the bound from above
            test_output_exceeds_maximum = output_destruction_function > (
                self.maximum_satellite_destruction(primary_halo_property,array_of_ones))
            output_destruction_function[test_output_exceeds_maximum] = (
                self.maximum_satellite_destruction(
                    primary_halo_property[test_output_exceeds_maximum],array_of_ones))
            # Second, apply the bound from below
            test_output_negative = output_destruction_function < 0 
            output_destruction_function[test_output_negative] = 0
            # Finally, for any range of the primary_halo_parameter 
            # for which the probability of halo_type1 is unity, 
            # set the destruction function equal to unity
            probability_type1 = self.halotype_fraction_satellites(
                primary_halo_property,array_of_ones)
            probability_type1_is_unity = np.where(probability_type1 == 1)[0]
            output_destruction_function[probability_type1_is_unity] = 1

            return np.array(output_destruction_function)

        def satellite_destruction_type0(self,primary_halo_property):

            array_of_ones = np.zeros(len(primary_halo_property)) + 1
            type1_fraction = self.halotype_fraction_satellites(primary_halo_property,array_of_ones)
            destruction_type1 = satellite_destruction_type1(self,
                primary_halo_property)
            type0_fraction = 1 - type1_fraction

            output_destruction_function = np.zeros(len(primary_halo_property))

            # type0 destruction is defined in terms of type1 destruction 
            # to ensure self-consistency of occupation statistics
            # Ensure that we do not divide by zero
            test_positive = type0_fraction > np.zeros(len(type0_fraction))
            output_destruction_function[test_positive] = (
                (1-type1_fraction[test_positive]*
                destruction_type1[test_positive])/
                type0_fraction[test_positive])
            # Apply baseline HOD constraint to central_destruction_type1
            # Apply the bound from above
            array_of_zeros = array_of_ones-1
            test_type0_exceeds_maximum = output_destruction_function > (
                self.maximum_satellite_destruction(primary_halo_property,array_of_zeros))
            output_destruction_function[test_type0_exceeds_maximum] = (
                self.maximum_satellite_destruction(
                    primary_halo_property[test_type0_exceeds_maximum],
                    array_of_zeros[test_type0_exceeds_maximum]))


            return np.array(output_destruction_function)

        # Run a blind search on values of halo_types
        # This could possibly be a source of speedup if the 
        # input arrays were pre-sorted
        idx_halo_type0 = np.where(halo_types==0)[0]
        idx_halo_type1 = np.where(halo_types==1)[0]

        satellite_destruction = np.zeros(len(primary_halo_property))

        satellite_destruction[idx_halo_type0] = (
            satellite_destruction_type0(self,primary_halo_property[idx_halo_type0]))
        satellite_destruction[idx_halo_type1] = (
            satellite_destruction_type1(self,primary_halo_property[idx_halo_type1]))

        return np.array(satellite_destruction)

    def halotype_fraction_centrals(self,primary_halo_property,halo_type):
        """ Determines the fractional representation of host halo 
        types as a function of the value of the primary halo property.

        Halo types can be either given by fixed-Mvir rank-orderings 
        of the host halos, or by the input occupation statistics functions.

        """
        output_array = np.zeros(len(primary_halo_property)) + 1
        idx0 = np.where(np.array(halo_type) == 0)[0]
        output_array[idx0] = 0

        return output_array

    def halotype_fraction_satellites(self,primary_halo_property,halo_type):
        """ Determines the fractional representation of host halo 
        types as a function of the value of the primary halo property.

        Halo types can be either given by fixed-Mvir rank-orderings 
        of the host halos, or by the input occupation statistics functions.

         """

        output_array = np.array(self.baseline_hod_model.mean_ncen(primary_halo_property))
        idx0 = np.where(halo_type == 0)[0]
        output_array[idx0] = 1.0 - output_array[idx0]

        return output_array

    def require_correct_keys(self,assembias_parameter_dict):
        correct_set_of_satcen_keys = set(defaults.default_satcen_parameters.keys())
        if set(assembias_parameter_dict.keys()) != correct_set_of_satcen_keys:
            raise TypeError("Set of keys of input assembias_parameter_dict"
            " does not match the set of keys required by the model." 
            " Correct set of keys is {'assembias_abcissa',"
            "'satellite_assembias_ordinates', 'central_assembias_ordinates'}. ")
        pass

@six.add_metaclass(ABCMeta)
class HOD_Quenching_Model(HOD_Model):
    """ Abstract base class for models determining mock galaxy quenching. 
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
    Approach is adapted from van den Bosch 2003. 

    All-galaxy central and satellite occupation 
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
                self.parameter_dict['quenching_abcissa'],
                self.parameter_dict['central_quenching_ordinates']))

        self.satellite_quenching_polynomial_coefficients = (
            self.solve_for_quenching_polynomial_coefficients(
                self.parameter_dict['quenching_abcissa'],
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
        polynomial_degree = len(self.parameter_dict['quenching_abcissa'])
        for n,coeff in enumerate(coefficients):
            mean_quenched_fractions += coeff*logM**n

        test_negative = mean_quenched_fractions < 0
        mean_quenched_fractions[test_negative] = 0
        test_exceeds_unity = mean_quenched_fractions > 1
        mean_quenched_fractions[test_exceeds_unity] = 1

        return mean_quenched_fractions



@six.add_metaclass(ABCMeta)
class Assembly_Biased_HOD_Quenching_Model(HOD_Quenching_Model):
    """ Abstract base class for any HOD model in which 
    central and/or satellite mean occupation depends on Mvir 
    plus an additional property.

    """

    def __init__(self):

        HOD_Quenching_Model.__init__(self)
        self.hod_model = None

    @abstractmethod
    def central_destruction(self,logM):
        """ Determines the excess probability that ``type 0`` 
        halos of logM host a central galaxy. """
        raise NotImplementedError(
            "central_destruction is not implemented")

    @abstractmethod
    def satellite_destruction(self,logM):
        """ Determines the excess probability that ``type 0`` 
        halos of logM host a satellite galaxy. """
        raise NotImplementedError(
            "satellite_destruction is not implemented")

    @abstractmethod
    def halo_type_fraction(self,logM):
        """ Determines the fractional representation of host halo 
        types as a function of logM.

        Halo types can be either given by fixed-Mvir rank-orderings 
        of the host halos, or by the input occupation statistics functions.

         """
        raise NotImplementedError(
            "halo_type_fraction is not implemented")

    @abstractmethod
    def central_conformity(self,logM):
        """ Determines the excess quenched fraction 
        of central galaxies residing in ``type 0`` halos of logM. """
        raise NotImplementedError(
            "central_conformity is not implemented")

    @abstractmethod
    def satellite_conformity(self,logM):
        """ Determines the excess quenched fraction 
        of satellite galaxies residing in``type 0`` halos of logM. """
        raise NotImplementedError(
            "satellite_conformity is not implemented")





















