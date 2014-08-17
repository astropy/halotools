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

__all__ = ['HOD_Model','Zheng07_HOD_Model','Assembly_Biased_HOD_Model',
'HOD_Quenching_Model','vdB03_Quenching_Model','Assembly_Biased_HOD_Quenching_Model',
'Satcen_Correlation_Polynomial_HOD_Model','Polynomial_Assembly_Biased_HOD_Model',
'cumulative_NFW_PDF','anatoly_concentration','solve_for_polynomial_coefficients']
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

    :math:`c(M) = c_{0}(M/M_{piv})^{\\alpha}`

    Parameters
    ----------
    logM : array 
        array of :math:`log_{10}M_{vir}` of halos in catalog

    Returns
    -------
    concentrations : array

    Notes 
    -----
    This is currently the only concentration-mass relation implemented. This will later be 
    bundled up into a class with a bunch of different radial profile methods, NFW and non-.

    Values are currently hard-coded to Anatoly's best-fit values:

    :math:`c_{0} = 9.6`

    :math:`\\alpha = -0.075`

    :math:`M_{piv} = 10^{12}M_{\odot}/h`

    """
    
    masses = np.zeros(len(logM)) + 10.**logM
    c0 = 9.6
    Mpiv = 1.e12
    a = -0.075
    concentrations = c0*(masses/Mpiv)**a
    return concentrations

def cumulative_NFW_PDF(x,c):
    """ Integral of an NFW profile with concentration c.
    Unit-normalized so that the result is a cumulative PDF. 

    :math:`F(x,c) = \\frac{ln(1+xc) - \\frac{xc}{1+xc}} 
    {ln(1+c) - \\frac{c}{1+c}}`

    Parameters
    ----------
    x : array_like
        Values are in the range (0,1).
        Elements x = r/Rvir specify host-centric distances in the range 0 < r/Rvir < 1.

    c : array_like
        Concentration of halo whose profile is being tabulated.

    Returns
    -------
    F : array 
        Array of floats in the range (0,1). 
        Value gives the probability of randomly drawing 
        a radial position :math:`x = \\frac{r}{R_{vir}}`  
        from an NFW profile of input concentration c.

    Notes
    --------
    Currently being used by mock.HOD_mock to generate 
    Monte Carlo realizations of satellite profiles, 
    using method of transformation of variables. 

    """
    c = np.array(c)
    x = np.array(x)
    norm=np.log(1.+c)-c/(1.+c)
    F = (np.log(1.+x*c) - x*c/(1.+x*c))/norm
    return F

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



@six.add_metaclass(ABCMeta)
class HOD_Model(object):
    """ Base class for any HOD-style model of the galaxy-halo connection.

    This is an abstract class, so you can't instantiate it. 
    Instead, you must work with one of its concrete subclasses, 
    such as `Zheng07_HOD_Model`. 

    All HOD-style models must provide their own specific functional forms 
    for how :math:`\langle N_{cen} \\rangle` and :math:`\langle N_{sat}\\rangle` 
    vary with the primary halo property. Additionally, 
    any HOD-based mock must specify the assumed concentration-halo relation. 

    Notes 
    -----
    Currently, the only implemented HOD-style model that is supported is 
    based on Zheng et al. 2007, which is specified in terms of virial halo mass. 
    But the HOD_Model class is sufficiently general that it will support 
    models for the galaxy-halo connection based on alternative host halo properties, 
    such as :math:`V_{max}` or :math:`M_{PE-corrected}`. 

    The current NFW profile requirement will eventually be relaxed, so that 
    arbitrary radial profiles are supported. 
    
    """
    
    def __init__(self,parameter_dict=None,threshold=None):
        self.publication = []
        self.parameter_dict = parameter_dict
        self.threshold = threshold

    @abstractmethod
    def mean_ncen(self,primary_halo_property):
        """
        Expected number of central galaxies in a halo 
        as a function of the primary property.

        Required method of any HOD_Model subclass.
        """
        raise NotImplementedError("mean_ncen is not implemented")

    @abstractmethod
    def mean_nsat(self,primary_halo_property):
        """
        Expected number of satellite galaxies in a halo 
        as a function of the primary property.

        Required method of any HOD_Model subclass.
        """
        raise NotImplementedError("mean_nsat is not implemented")

    @abstractmethod
    def mean_concentration(self,primary_halo_property):
        """
        Concentration-halo relation assumed by the model. 
        Used to assign positions to satellites.

        Required method of any HOD_Model subclass.
        """
        raise NotImplementedError("mean_concentration is not implemented")

    @abstractproperty
    def primary_halo_property_key(self):
        """ String providing the key to the halo catalog dictionary 
        where the primary halo property data is stored. 
        
        Required attribute of any HOD_Model subclass.
        """
        raise NotImplementedError("primary_halo_property_key "
            "needs to be explicitly stated to ensure self-consistency "
            "of baseline HOD and assembly-biased HOD model features")



class Zheng07_HOD_Model(HOD_Model):
    """ Subclass of `HOD_Model` object, where functional forms for occupation statistics 
    are taken from Zheng et al. 2007, arXiv:0703457.


    Parameters 
    ----------
    parameter_dict : dictionary, optional.
        Contains values for the parameters specifying the model.
        Dictionary keys are  
        'logMmin_cen', 'sigma_logM', 'logM0_sat','logM1_sat','alpha_sat'.
        Default values pertain to the best-fit values of their 
        :math:`M_{r} - 5log_{10}h< -19.5` threshold sample.

        All the best-fit parameter values provided in Table 1 of 
        Zheng et al. (2007) can be accessed via the 
        `published_parameters` method.

    threshold : float, optional.
        Luminosity threshold of the mock galaxy sample. 
        If specified, input value must agree with 
        one of the thresholds used in Zheng07 to fit HODs: 
        [-18, -18.5, -19, -19.5, -20, -20.5, -21, -21.5, -22].
        Default value is -19.5, specified in the `defaults` module.

    Notes
    -----
    :math:`c-M_{vir}` relation is current set to be Anatoly's, though 
    this is not the relation used in Zheng07. Their concentration-mass relation 
    is of the same form as the one implemented one, but with different 
    values for the hard-coded parameters. See Equation 1 of arXiv:0703457.

    """

    def __init__(self,parameter_dict=None,threshold=None):
        HOD_Model.__init__(self)

        self.publication.extend(['arXiv:0703457'])

        if parameter_dict is None:
            self.parameter_dict = self.published_parameters(threshold)
        self.require_correct_keys()

    @property 
    def primary_halo_property_key(self):
        """ Model is based on :math:`M = M_{vir}`.
        """
        return 'MVIR'

    def mean_ncen(self,logM):
        """ Expected number of central galaxies in a halo of mass logM.
        See Equation 2 of arXiv:0703457.

        Parameters
        ----------        
        logM : array 
            array of :math:`log_{10}(M)` of halos in catalog

        Returns
        -------
        mean_ncen : array
    
        Notes
        -------
        Mean number of central galaxies in a host halo of the specified mass. 

        :math:`\\langle N_{cen} \\rangle_{M} = 
        \\frac{1}{2}\\left( 1 + 
        erf\\left( \\frac{log_{10}M - 
        log_{10}M_{min}}{\\sigma_{log_{10}M}} \\right) \\right)`

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
            array of :math:`log_{10}(M)` of halos in catalog

        Returns
        -------
        mean_nsat : float or array
            Mean number of satellite galaxies in a host halo of the specified mass. 

        :math:`\\langle N_{sat} \\rangle_{M} = \left( \\frac{M - M_{0}}{M_{1}} \\right)^{\\alpha} \\langle N_{cen} \\rangle_{M}`


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
            Mean concentration of logM halos, using `anatoly_concentration` model.

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
        """ If a parameter dictionary is passed to the class upon instantiation, 
        this method is used to enforce that the set of keys is in accord 
        with the set of keys required by the model. 
        """
        correct_set_of_keys = set(self.published_parameters(threshold = -20).keys())
        if set(self.parameter_dict.keys()) != correct_set_of_keys:
            raise TypeError("Set of keys of input parameter_dict do not match the set of keys required by the model")
        pass




@six.add_metaclass(ABCMeta)
class Assembly_Biased_HOD_Model(HOD_Model):
    """ Abstract base class for any HOD model with assembly bias. 

    In this class of models, central and/or satellite mean occupation depends on some primary  
    property, such as :math:`M_{vir}`, and is modulated by some secondary property, 
    such as :math:`z_{form}`. 

    """

    def __init__(self):

        # Executing the __init__ of the abstract base class HOD_Model 
        #sets self.parameter_dict to None, self.threshold to None, 
        # and self.publication to []
        HOD_Model.__init__(self)

    @abstractproperty
    def baseline_hod_model(self):
        """ Underlying HOD model, about which assembly bias modulates 
        galaxy abundance. The baseline HOD model can in principle be driven 
        by any host halo property. 
        """
        pass

    @abstractproperty
    def primary_halo_property_key(self):
        """ String providing halo catalog dictionary key pointing 
        to primary halo property. Necessary to ensure self-consistency between 
        underlying halo model and assembly bias modulation. 

        """
        raise NotImplementedError("primary_halo_property_key "
            "needs to be implemented to ensure self-consistency "
            "of baseline HOD and assembly-biased HOD model features")
        pass

    @abstractmethod
    def unconstrained_central_destruction_halo_type1(self,primary_halo_property):
        """ Determines the excess probability that halos of type :math:`h_{1}` 
        host a central galaxy. Can be any arbitrary function. 
        Constraints on the value of this function resulting from the need to keep 
        the baseline HOD fixed 
        are automatically applied by the class. """
        raise NotImplementedError(
            "unconstrained_central_destruction_halo_type1 is not implemented")
        pass

    @abstractmethod
    def unconstrained_satellite_destruction_halo_type1(self,primary_halo_property):
        """ Determines the excess probability that ``type 1`` 
        halos host a satellite galaxy. """
        raise NotImplementedError(
            "unconstrained_satellite_destruction_halo_type1 is not implemented")
        pass

    @abstractmethod
    def halo_type1_fraction_centrals(self,primary_halo_property):
        """ Determines the fractional representation of host halo 
        type 1 as a function of primary_halo_property, as pertains to centrals. 

         """
        raise NotImplementedError(
            "halo_type_fraction_centrals is not implemented")
        pass

    @abstractmethod
    def halo_type1_fraction_satellites(self,primary_halo_property):
        """ Determines the fractional representation of host halo 
        type 1 as a function of primary_halo_property, as pertains to satellites.

         """
        raise NotImplementedError(
            "halo_type_fraction_satellites is not implemented")
        pass

    def halo_type_fraction_centrals(self,primary_halo_property,halo_type):
        """ Determines the fractional representation of host halos of input halo_type 
        as a function of primary_halo_property, as pertains to centrals.

        Parameters 
        ----------
        halo_type : array_like
            Array with elements equal to 0 or 1, specifying the type of the halo 
            whose fractional representation is being returned.

        primary_halo_property : array_like
            Array with elements equal to the primary_halo_property at which 
            the fractional representation of the halos of input halo_type is being returned.

        Returns 
        -------
        output_halo_type_fraction : array_like
            The ith element gives the probability that a halo with primary halo property = primary_halo_property[i] 
            has halo_type = halo_type[i]

         """

        output_halo_type_fraction = self.halo_type1_fraction_centrals(primary_halo_property)
        idx0 = np.where(halo_type == 0)[0]
        output_halo_type_fraction[idx0] = 1.0 - output_halo_type_fraction[idx0]

        return output_halo_type_fraction

    def halo_type_fraction_satellites(self,primary_halo_property,halo_type):
        """ Determines the fractional representation of host halos of input halo_type 
        as a function of primary_halo_property, as pertains to satellites.

        Parameters 
        ----------
        halo_type : array_like
            Array with elements equal to 0 or 1, specifying the type of the halo 
            whose fractional representation is being returned.

        primary_halo_property : array_like
            Array with elements equal to the primary_halo_property at which 
            the fractional representation of the halos of input halo_type is being returned.

        Returns 
        -------
        output_halo_type_fraction : array_like
            The ith element gives the probability that a halo with primary halo property = primary_halo_property[i] 
            has halo_type = halo_type[i]
        """
        output_halo_type_fraction = self.halo_type1_fraction_satellites(primary_halo_property)
        idx0 = np.where(halo_type == 0)[0]
        output_halo_type_fraction[idx0] = 1.0 - output_halo_type_fraction[idx0]

        return output_halo_type_fraction


    def maximum_destruction_centrals(self,primary_halo_property,halo_type):
        """ Maximum allowed value of the destruction function, as pertains to centrals.

        Parameters 
        ----------
        halo_type : array_like
            Array with elements equal to 0 or 1, specifying the type of the halo 
            whose fractional representation is being returned.

        primary_halo_property : array_like
            Array with elements equal to the primary_halo_property at which 
            the fractional representation of the halos of input halo_type is being returned.

        Returns 
        -------
        output_maximum_destruction : array_like
            Maximum allowed value of the destruction function, as pertains to centrals.

        """

        output_maximum_destruction = np.zeros(len(primary_halo_property))
        halo_type_fraction = self.halo_type_fraction_centrals(
            primary_halo_property,halo_type)
        idx_positive = halo_type_fraction > 0
        output_maximum_destruction[idx_positive] = 1./halo_type_fraction[idx_positive]
        return output_maximum_destruction

    def maximum_destruction_satellites(self,primary_halo_property,halo_type):
        """ Maximum allowed value of the destruction function, as pertains to centrals.

        Parameters 
        ----------
        halo_type : array_like
            Array with elements equal to 0 or 1, specifying the type of the halo 
            whose fractional representation is being returned.

        primary_halo_property : array_like
            Array with elements equal to the primary_halo_property at which 
            the fractional representation of the halos of input halo_type is being returned.

        Returns 
        -------
        output_maximum_destruction : array_like
            Maximum allowed value of the destruction function, as pertains to centrals.

        """

        output_maximum_destruction = np.zeros(len(primary_halo_property))
        halo_type_fraction = self.halo_type_fraction_satellites(
            primary_halo_property,halo_type)
        idx_positive = halo_type_fraction > 0
        output_maximum_destruction[idx_positive] = 1./halo_type_fraction[idx_positive]
        return output_maximum_destruction


    def destruction_satellites(self,primary_halo_property,halo_type):

        idx0 = np.where(halo_type == 0)[0]
        idx1 = np.where(halo_type == 1)[0]

        # Initialize array containing result to return
        output_destruction_allhalos = np.zeros(len(primary_halo_property))

        all_ones = np.zeros(len(primary_halo_property)) + 1

        # Start by ignoring the input halo_type, and  
        # assuming the halo_type = 1 for all inputs.
        # This is convenient and costs nothing, 
        # since the halo_type = 0 branch 
        # is defined in terms of the halo_type = 1 branch.
        output_destruction_allhalos = (
            self.unconstrained_satellite_destruction_halo_type1(
                primary_halo_property))
        ########################################
        # Now apply the baseline HOD constraints to output_destruction_allhalos, 
        # still behaving as if every input halo has halo_type=1
        # First, require that the destruction function is non-negative
        test_negative = output_destruction_allhalos < 0
        output_destruction_allhalos[test_negative] = 0
        # Second, require that the destruction function never exceed the 
        # maximum allowed value 
        maximum = self.maximum_destruction_satellites(primary_halo_property,all_ones)
        test_exceeds_maximum = output_destruction_allhalos > maximum
        output_destruction_allhalos[test_exceeds_maximum] = maximum[test_exceeds_maximum]
        # Finally, require that the destruction function is set to unity 
        # whenever the probability of halo_type=1 equals unity
        # This is requirement supercedes the previous two
        probability_type1 = self.halo_type_fraction_satellites(
            primary_halo_property,all_ones)
        test_unit_probability = (probability_type1 == 1)
        output_destruction_allhalos[test_unit_probability] = 1
        ########################################
        # At this point, output_destruction_allhalos has been properly conditioned. 
        # However, we have been assuming that all input halo_type = 1.
        # We now need to compute the correct output 
        # for cases where input halo_type = 0.
        # Define some shorthands (bookkeeping convenience)
        output_destruction_input_halo_type0 = output_destruction_allhalos[idx0]
        primary_halo_property_input_halo_type0 = primary_halo_property[idx0]
        probability_type1_input_halo_type0 = probability_type1[idx0]
        probability_type0_input_halo_type0 = 1.0 - probability_type1_input_halo_type0
        # Whenever the fraction of halos of type=0 is zero, the destruction function 
        # for type0 halos should be set to zero.
        test_positive = (probability_type0_input_halo_type0 > 0)

        output_destruction_input_halo_type0[test_positive] = (
            (1.0 - output_destruction_input_halo_type0[test_positive]*
                probability_type1_input_halo_type0[test_positive])/
            probability_type0_input_halo_type0[test_positive])

        test_zero = (probability_type0_input_halo_type0 == 0)
        output_destruction_input_halo_type0[test_zero] = 0

        # Now write the results back to the output (Why doesn't the above syntax
            # work without this step?)
        output_destruction_allhalos[idx0] = output_destruction_input_halo_type0

        return output_destruction_allhalos

    def destruction_centrals(self,primary_halo_property,halo_type):

        idx0 = np.where(halo_type == 0)[0]
        idx1 = np.where(halo_type == 1)[0]

        # Initialize array containing result to return
        output_destruction_allhalos = np.zeros(len(primary_halo_property))

        all_ones = np.zeros(len(primary_halo_property)) + 1

        # Start by ignoring the input halo_type, and  
        # assuming the halo_type = 1 for all inputs.
        # This is convenient and costs nothing, 
        # since the halo_type = 0 branch 
        # is defined in terms of the halo_type = 1 branch.
        output_destruction_allhalos = (
            self.unconstrained_central_destruction_halo_type1(
                primary_halo_property))
        ########################################
        # Now apply the baseline HOD constraints to output_destruction_allhalos, 
        # still behaving as if every input halo has halo_type=1
        # First, require that the destruction function is non-negative
        test_negative = output_destruction_allhalos < 0
        output_destruction_allhalos[test_negative] = 0
        # Second, require that the destruction function never exceed the 
        # maximum allowed value 
        maximum = self.maximum_destruction_centrals(primary_halo_property,all_ones)
        test_exceeds_maximum = output_destruction_allhalos > maximum
        output_destruction_allhalos[test_exceeds_maximum] = maximum[test_exceeds_maximum]
        # Finally, require that the destruction function is set to unity 
        # whenever the probability of halo_type=1 equals unity
        # This is requirement supercedes the previous two
        probability_type1 = self.halo_type_fraction_centrals(
            primary_halo_property,all_ones)
        test_unit_probability = (probability_type1 == 1)
        output_destruction_allhalos[test_unit_probability] = 1
        ########################################
        # At this point, output_destruction_allhalos has been properly conditioned. 
        # However, we have been assuming that all input halo_type = 1.
        # We now need to compute the correct output 
        # for cases where input halo_type = 0.
        # Define some shorthands (bookkeeping convenience)
        output_destruction_input_halo_type0 = output_destruction_allhalos[idx0]
        primary_halo_property_input_halo_type0 = primary_halo_property[idx0]
        probability_type1_input_halo_type0 = probability_type1[idx0]
        probability_type0_input_halo_type0 = 1.0 - probability_type1_input_halo_type0
        # Whenever the fraction of halos of type=0 is zero, the destruction function 
        # for type0 halos should be set to zero.
        test_positive = (probability_type0_input_halo_type0 > 0)

        output_destruction_input_halo_type0[test_positive] = (
            (1.0 - output_destruction_input_halo_type0[test_positive]*
                probability_type1_input_halo_type0[test_positive])/
            probability_type0_input_halo_type0[test_positive])

        test_zero = (probability_type0_input_halo_type0 == 0)
        output_destruction_input_halo_type0[test_zero] = 0

        # Now write the results back to the output (Why doesn't the above syntax
            # work without this step?)
        output_destruction_allhalos[idx0] = output_destruction_input_halo_type0

        return output_destruction_allhalos

    def mean_ncen(self,primary_halo_property,halo_type):
        """ Override the baseline HOD method so that mean_ncen is modulated by halo_type"""
        return self.destruction_centrals(primary_halo_property,halo_type)*(
            self.baseline_hod_model.mean_ncen(primary_halo_property))

    def mean_nsat(self,primary_halo_property,halo_type):
        """ Override the baseline HOD method so that mean_ncen is modulated by halo_type"""
        return self.destruction_satellites(primary_halo_property,halo_type)*(
            self.baseline_hod_model.mean_nsat(primary_halo_property))

    def host_halo_type_calculator(self,primary_halo_property,secondary_halo_property):
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


    def halo_type1_fraction_centrals(self,primary_halo_property):
        """ Determines the fractional representation of host halo 
        types as a function of the value of the primary halo property.

        Halo types can be either given by fixed-Mvir rank-orderings 
        of the host halos, or by the input occupation statistics functions.

        """
        # In this model, centrals exhibit no assembly bias
        # So simply set the halo type1 fraction to unity for centrals
        output_array = np.zeros(len(primary_halo_property)) + 1
        return output_array

    def halo_type1_fraction_satellites(self,primary_halo_property):
        """ Determines the fractional representation of host halo 
        types as a function of the value of the primary halo property.

        Halo types can be either given by fixed-Mvir rank-orderings 
        of the host halos, or by the input occupation statistics functions.

         """

        output_array = np.array(self.baseline_hod_model.mean_ncen(primary_halo_property))

        return output_array

    def unconstrained_polynomial_model(self,abcissa,ordinates,primary_halo_property):
        coefficient_array = solve_for_polynomial_coefficients(
            abcissa,ordinates)
        output_unconstrained_destruction_function = (
            np.zeros(len(primary_halo_property)))

        # Use coefficients to compute values of the destruction function polynomial
        for n,coeff in enumerate(coefficient_array):
            output_unconstrained_destruction_function += coeff*primary_halo_property**n

        return output_unconstrained_destruction_function

    def unconstrained_central_destruction_halo_type1(self,primary_halo_property):
        abcissa = self.parameter_dict['assembias_abcissa']
        ordinates = self.parameter_dict['central_assembias_ordinates']

        return self.unconstrained_polynomial_model(abcissa,ordinates,primary_halo_property)

    def unconstrained_satellite_destruction_halo_type1(self,primary_halo_property):
        abcissa = self.parameter_dict['assembias_abcissa']
        ordinates = self.parameter_dict['satellite_assembias_ordinates']

        return self.unconstrained_polynomial_model(abcissa,ordinates,primary_halo_property)

    def require_correct_keys(self,assembias_parameter_dict):
        correct_set_of_satcen_keys = set(defaults.default_satcen_parameters.keys())
        if set(assembias_parameter_dict.keys()) != correct_set_of_satcen_keys:
            raise TypeError("Set of keys of input assembias_parameter_dict"
            " does not match the set of keys required by the model." 
            " Correct set of keys is {'assembias_abcissa',"
            "'satellite_assembias_ordinates', 'central_assembias_ordinates'}. ")
        pass


class Polynomial_Assembly_Biased_HOD_Model(Assembly_Biased_HOD_Model):
    """ HOD-style model in which satellite abundance 
    is correlated with the presence of a central galaxy.
    """

    def __init__(self,baseline_hod_model=Zheng07_HOD_Model,
            baseline_hod_parameter_dict=None,threshold=None,
            assembias_parameter_dict=None,
            secondary_halo_property_key='Z04'):


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
            self.assembias_parameter_dict = defaults.default_assembias_parameters
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

    @property 
    def secondary_halo_property_key(self):
        return 'Z04'

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


    def halo_type1_fraction_centrals(self,primary_halo_property):
        """ Determines the fractional representation of host halo 
        types as a function of the value of the primary halo property.

        Halo types can be either given by fixed-Mvir rank-orderings 
        of the host halos, or by the input occupation statistics functions.

        """
        # In this model, centrals exhibit no assembly bias
        # So simply set the halo type1 fraction to unity for centrals
        abcissa = defaults.default_halo_type_split['halo_type_split_abcissa']
        ordinates = defaults.default_halo_type_split['halo_type_split_ordinates']
        output_fraction = self.unconstrained_polynomial_model(
            abcissa,ordinates,primary_halo_property)
        test_negative = (output_fraction < 0)
        output_fraction[test_negative] = 0
        test_exceeds_unity = (output_fraction > 1)
        output_fraction[test_exceeds_unity] = 1
        return output_fraction

    def halo_type1_fraction_satellites(self,primary_halo_property):
        """ Determines the fractional representation of host halo 
        types as a function of the value of the primary halo property.

        Halo types can be either given by fixed-Mvir rank-orderings 
        of the host halos, or by the input occupation statistics functions.

         """

        abcissa = defaults.default_halo_type_split['halo_type_split_abcissa']
        ordinates = defaults.default_halo_type_split['halo_type_split_ordinates']
        output_fraction = self.unconstrained_polynomial_model(
            abcissa,ordinates,primary_halo_property)
        test_negative = (output_fraction < 0)
        output_fraction[test_negative] = 0
        test_exceeds_unity = (output_fraction > 1)
        output_fraction[test_exceeds_unity] = 1
        return output_fraction

    def unconstrained_polynomial_model(self,abcissa,ordinates,primary_halo_property):
        coefficient_array = solve_for_polynomial_coefficients(
            abcissa,ordinates)
        output_unconstrained_destruction_function = (
            np.zeros(len(primary_halo_property)))

        # Use coefficients to compute values of the destruction function polynomial
        for n,coeff in enumerate(coefficient_array):
            output_unconstrained_destruction_function += coeff*primary_halo_property**n

        return output_unconstrained_destruction_function

    def unconstrained_central_destruction_halo_type1(self,primary_halo_property):
        abcissa = self.parameter_dict['assembias_abcissa']
        ordinates = self.parameter_dict['central_assembias_ordinates']

        return self.unconstrained_polynomial_model(abcissa,ordinates,primary_halo_property)

    def unconstrained_satellite_destruction_halo_type1(self,primary_halo_property):
        abcissa = self.parameter_dict['assembias_abcissa']
        ordinates = self.parameter_dict['satellite_assembias_ordinates']

        return self.unconstrained_polynomial_model(abcissa,ordinates,primary_halo_property)

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





















