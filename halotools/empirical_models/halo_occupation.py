# -*- coding: utf-8 -*-
"""

This module contains the classes and methods used to connect  
galaxies to halos using HOD-style models. 

"""

__all__ = ['HOD_Model','Zheng07_HOD_Model','Leauthaud11_SHMR_Model','Toy_HOD_Model','Assembias_HOD_Model',
'HOD_Quenching_Model','vdB03_Quenching_Model','Assembias_HOD_Quenching_Model',
'Satcen_Correlation_Polynomial_HOD_Model','Polynomial_Assembias_HOD_Model',
'Polynomial_Assembias_HOD_Quenching_Model',
'cumulative_NFW_PDF','anatoly_concentration','solve_for_polynomial_coefficients']
#from __future__ import (absolute_import, division, print_function,
#                        unicode_literals)

import numpy as np
from scipy.special import erf 
from scipy.stats import poisson
from scipy.optimize import brentq

import model_defaults
#from ..sim_manager import sim_model_defaults

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
    such as `Zheng07_HOD_Model` or `vdB03_Quenching_Model`. 

    All HOD-style models must provide their own specific functional forms 
    for how :math:`\langle N_{cen} \\rangle` and :math:`\langle N_{sat}\\rangle` 
    vary with the primary halo property. Additionally, 
    any HOD-based mock must specify the assumed concentration-halo relation. 

    Notes 
    -----
    Currently, the only baseline HOD model that has been implemented is 
    based on Zheng et al. 2007, which is specified in terms of virial halo mass. 
    But the HOD_Model class is sufficiently general that it will support 
    models for the galaxy-halo connection based on alternative host halo properties, 
    such as :math:`V_{max}` or :math:`M_{PE-corrected}`. 

    The only radial profile implemented is NFW, 
    but this requirement will eventually be relaxed, so that 
    arbitrary radial profiles are supported.

    Mocks instances constructed with the current form of this class 
    only exist in configuration space. Redshift-space features coming soon.
    
    """
    
    def __init__(self,parameter_dict=None,threshold=None):
        self.publication = []
        self.parameter_dict = parameter_dict
        self.threshold = threshold

    @abstractmethod
    def mean_ncen(self,primary_halo_property,halo_type):
        """
        Expected number of central galaxies in a halo 
        as a function of the primary property :math:`p` 
        and binary-valued halo type :math:`h_{i}`

        Required method of any HOD_Model subclass.
        """
        raise NotImplementedError("mean_ncen is not implemented")

    @abstractmethod
    def mean_nsat(self,primary_halo_property,halo_type):
        """
        Expected number of satellite galaxies in a halo 
        as a function of the primary property :math:`p` 
        and binary-valued halo type :math:`h_{i}`

        Required method of any HOD_Model subclass.
        """
        raise NotImplementedError("mean_nsat is not implemented")

    @abstractmethod
    def mean_concentration(self,primary_halo_property,halo_type):
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
        Default value is -20, specified in the `~halotools.model_defaults` module.

    Notes
    -----
    :math:`c-M_{vir}` relation is current set to be Anatoly's, though 
    this is not the relation used in Zheng07. Their concentration-mass relation 
    is of the same form as the one implemented one, but with different 
    values for the hard-coded parameters. See Equation 1 of arXiv:0703457.

    """

    def __init__(self,parameter_dict=None,threshold=None):
        HOD_Model.__init__(self)

        self.threshold = threshold
        if self.threshold==None:
            warnings.warn("HOD threshold unspecified: setting to value defined in model_defaults.py")
            self.threshold = model_defaults.default_luminosity_threshold

        self.publication.extend(['arXiv:0703457'])

        if parameter_dict is None:
            self.parameter_dict = self.published_parameters()
        else:
            self.parameter_dict = parameter_dict
        self.require_correct_keys()

    @property 
    def primary_halo_property_key(self):
        """ Model is based on :math:`M = M_{vir}`.
        """
        return 'MVIR'

    def mean_ncen(self,logM,halo_type):
        """ Expected number of central galaxies in a halo of mass logM.
        See Equation 2 of arXiv:0703457.

        Parameters
        ----------        
        logM : array 
            array of :math:`log_{10}(M)` of halos in catalog

        halo_type : array 
            array of halo types. Entirely ignored in this model. 
            Included as a passed variable purely for consistency 
            between the way this function is called by different models.

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

        #mean_ncen = np.zeros(len(logM)) + 0.01
        return mean_ncen

    def mean_nsat(self,logM,halo_type):
        """Expected number of satellite galaxies in a halo of mass logM.
        See Equation 5 of arXiv:0703457.

        Parameters
        ----------
        logM : array 
            array of :math:`log_{10}(M)` of halos in catalog

        halo_type : array 
            array of halo types. Entirely ignored in this model. 
            Included as a passed variable purely for consistency 
            between the way this function is called by different models.

        Returns
        -------
        mean_nsat : float or array
            Mean number of satellite galaxies in a host halo of the specified mass. 

        :math:`\\langle N_{sat} \\rangle_{M} = \left( \\frac{M - M_{0}}{M_{1}} \\right)^{\\alpha} \\langle N_{cen} \\rangle_{M}`


        """
        halo_type = np.array(halo_type)
        logM = np.array(logM)
        halo_mass = 10.**logM


        M0 = 10.**self.parameter_dict['logM0_sat']
        M1 = 10.**self.parameter_dict['logM1_sat']
        mean_nsat = np.zeros(len(logM),dtype='f8')
        idx_nonzero_satellites = (halo_mass - M0) > 0

        mean_nsat[idx_nonzero_satellites] = (
            self.mean_ncen(
            logM[idx_nonzero_satellites],halo_type[idx_nonzero_satellites])*
            (((halo_mass[idx_nonzero_satellites] - M0)/M1)
            **self.parameter_dict['alpha_sat']))

        return mean_nsat

    def mean_concentration(self,logM,halo_type):
        """
        Concentration-mass relation of the model.

        Parameters 
        ----------

        logM : array_like
            array of :math:`log_{10}(M)` of halos in catalog

        halo_type : array 
            array of halo types. Entirely ignored in this model. 
            Included as a passed variable purely for consistency 
            between the way this function is called by different models.

        Returns 
        -------

        concentrations : array_like
            Mean concentration of halos of the input mass, using `anatoly_concentration` model.

        """

        logM = np.array(logM)

        concentrations = anatoly_concentration(logM)
        return concentrations

    def published_parameters(self):
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
        correct_set_of_keys = set(self.published_parameters().keys())
        if set(self.parameter_dict.keys()) != correct_set_of_keys:
            raise TypeError("Set of keys of input parameter_dict do not match the set of keys required by the model")
        pass


class Leauthaud11_SHMR_Model(HOD_Model):
    """ Subclass of `HOD_Model` object, where functional forms for occupation statistics 
    are taken from Leauthaud et al. 2011 arXiv:1103.2077


    Parameters 
    ----------
    parameter_dict : dictionary, optional.
        Contains values for the parameters specifying the model.
        Dictionary keys are  
        'm1','ms0','beta','delta','gamma','siglogm','bcut','bsat','betacut','betasat','alphasat'.

        Default values pertain to the best-fit values for the z2 redshift bin in
        leauthaud et al. 2012. Note these these default values assume h=0.72, M200b, and a Chabrier IMF.

        threshold : float. Default value is default_stellar_mass_threshold
        Stellar Mass Threshold.
        
    Notes
    -----
    :math:`c-M_{vir}` relation is current set to be Anatoly's.
    """

    def __init__(self,parameter_dict=None,threshold=None):
        HOD_Model.__init__(self)

        self.publication.extend(['arXiv:1103.2077'])
        
        if parameter_dict is None:
            self.parameter_dict = self.published_parameters(threshold)
            
        if threshold is None:
            self.threshold = model_defaults.default_stellar_mass_threshold
            
        self.require_correct_keys()

    @property 
    def primary_halo_property_key(self):
        """ Model is based on :math:`M = M_{vir}`.
        """
        return 'MVIR'

    def mean_ncen(self,logM,halo_type):
        """ Expected number of central galaxies in a halo of mass logM.
        See Equation 8 of arXiv:1103.2077

        Parameters
        ----------        
        logM : array 
            array of :math:`log_{10}(M)` of halos in catalog

        halo_type : array 
            array of halo types. Entirely ignored in this model. 
            Included as a passed variable purely for consistency 
            between the way this function is called by different models.

        Returns
        -------
        mean_ncen : array
    
        Notes
        -------
        Mean number of central galaxies in a host halo of the specified mass. 

        """
        logM = np.array(logM) 

        ms_shmr = np.zeros(len(logM),dtype='f8')

        # Invert SHMR for this Mh value
        for i in np.arange(len(logM)):
            ms_shmr[i] = self.mh2ms(logM[i])
	
        # Eq 8 in Leauthaud 2011
        x=(self.threshold-ms_shmr)/(np.sqrt(2)*self.parameter_dict['siglogm']) 

        # Ncen
        return 0.5*(1-erf(x))   
        
        
    def ms2mh(self,ms):
        """
        Converts Stellar mass to Mhalo using SHMR
        Input is stellar mass in log10 units
        Output is Mh in log units

        Parameters 
        ----------
        ms : array_like
            array of stellar mass

        Returns 
        -------
        mh : array_like
            array of halo masses corresponding to the input stellar masses.

        """

        # In linear units
        x=10.**(ms-self.parameter_dict['ms0'])

        m1 = self.parameter_dict['m1']
        beta = self.parameter_dict['beta']
        delta = self.parameter_dict['delta']
        gamma = self.parameter_dict['gamma']
        
        # In log10
        mh=m1+(beta*np.log10(x))+(x**delta/(1+x**(-gamma)))-0.5
        
        return mh


    def mh2ms(self,mh):
        """
        Converts Halo mass to Stellar mass by inverting SHMR

        Parameters 
        ----------
        mh : array_like
            array of stellar mass

        Returns 
        -------
        ms : array_like
            array of halo masses corresponding to the input stellar masses.

        """

        out=brentq(self.mh2ms_funct,8,12,args=(mh))
	
        return out

    def mh2ms_funct(self,ms,mh):
        """
        Function for brentq
        Returns SHMR-Mh
        """
        out=self.ms2mh(ms)-mh 
    
        return out

    def mean_nsat(self,logM,halo_type):
        """Expected number of satellite galaxies in a halo of mass logM.
        See Equation 12 of arXiv:1103.2077

        Parameters
        ----------
        logM : array 
            array of :math:`log_{10}(M)` of halos in catalog

        halo_type : array 
            array of halo types. Entirely ignored in this model. 
            Included as a passed variable purely for consistency 
            between the way this function is called by different models.

        Returns
        -------
        mean_nsat : float or array
            Mean number of satellite galaxies in a host halo of the specified mass. 

        """
        halo_type = np.array(halo_type)
        logM = np.array(logM)
        halo_mass = 10.**logM

        mean_nsat = np.zeros(len(logM),dtype='f8')

        mh_shmr=10.**(self.ms2mh(self.threshold)-12.0)
        msat=self.parameter_dict['bsat']*(10.**12)*(mh_shmr)**self.parameter_dict['betasat']
        mcut=self.parameter_dict['bcut']*(10.**12)*(mh_shmr)**self.parameter_dict['betacut']

        htype = np.ones(len(logM))
        
        mean_nsat=self.mean_ncen(logM,htype)*((halo_mass/msat)**self.parameter_dict['alphasat'])*np.exp(-mcut/halo_mass)
    
        return mean_nsat

    def mean_concentration(self,logM,halo_type):
        """
        Concentration-mass relation of the model.

        Parameters 
        ----------

        logM : array_like
            array of :math:`log_{10}(M)` of halos in catalog

        halo_type : array 
            array of halo types. Entirely ignored in this model. 
            Included as a passed variable purely for consistency 
            between the way this function is called by different models.

        Returns 
        -------

        concentrations : array_like
            Mean concentration of halos of the input mass, using `anatoly_concentration` model.

        """

        logM = np.array(logM)

        concentrations = anatoly_concentration(logM)
        return concentrations

    def published_parameters(self,threshold=None):   
        """
        Best-fit HOD parameters from Table 5 of Leauthaud et al. 2012. For the second redshift bin.

        Parameters 
        ----------

        threshold : float
            Any stellar mass threshold

        Returns 
        -------

        parameter_dict : dict
            Dictionary of model parameters whose values have been set to 
            agree with the values taken from Table 5 of Leauthaud et al. 2012.

        """

        # Check to see whether a luminosity threshold has been specified
        if threshold is None:
            warnings.warn("stellar mass threshold unspecified: setting to default value")
            self.threshold = model_defaults.default_stellar_mass_threshold

        parameter_dict = {
            'm1':12.725,
            'ms0':11.038,
            'beta':0.466,
            'delta':0.61,
            'gamma':1.95,
            'siglogm':0.249,
            'bcut':1.65,
            'bsat':9.04,
            'betacut':0.59,
            'betasat':0.740,
            'alphasat':1.0}

        return parameter_dict

    def require_correct_keys(self):
        """ If a parameter dictionary is passed to the class upon instantiation, 
        this method is used to enforce that the set of keys is in accord 
        with the set of keys required by the model. 
        """
        correct_set_of_keys = set(self.published_parameters(threshold = model_defaults.default_stellar_mass_threshold).keys()) 
        if set(self.parameter_dict.keys()) != correct_set_of_keys:
            raise TypeError("Set of keys of input parameter_dict do not match the set of keys required by the model")
        pass

    
class Toy_HOD_Model(HOD_Model):
    """ Subclass of `HOD_Model` object, allowing for explicit specification of 
    the occupation statistics.


    Parameters 
    ----------
    mass_mask : array_like, optional.
        List the values of :math:`log_{10}M` of halos occupied by galaxies. 

    toy_ncen : array_like, optional.
        List the desired mean abundance of centrals at the input mass_mask

    toy_nsat : array_like, optional.
        List the desired mean abundance of satellites at the input mass_mask


    Notes
    -----
    Primarily useful for two reasons. First, constructing weird, unphysical HODs 
    is useful for developing a rigorous test suite, since tests that pass extreme 
    cases will pass reasonable cases. Second, useful to zero in on assembly bias effects 
    that are operative only over specific mass ranges. 

    """

    def __init__(self,mass_mask=[12,13,14],
        toy_ncen=[0.1,0.1,0.1],toy_nsat=[0.1,0.1,0.1],binsize=0.1,
        threshold=None):

        HOD_Model.__init__(self)
        self.parameter_dict = {}

        toy_ncen = np.array(toy_ncen)
        toy_nsat = np.array(toy_nsat)
        mass_mask = np.array(mass_mask)

        if len(mass_mask) != len(toy_ncen):
            raise TypeError("input toy_ncen array must be the same length as input mass_mask")
        if len(mass_mask) != len(toy_nsat):
            raise TypeError("input toy_nsat array must be the same length as input mass_mask")

        self.mass_mask = mass_mask
        self.binsize = self.impose_binsize_constraints(self.mass_mask,binsize)
        self.toy_ncen = toy_ncen
        self.toy_nsat = toy_nsat
        # Impose constrains on the input abundances
        idx_negative_ncen = np.where(self.toy_ncen < 0)[0]
        self.toy_ncen[idx_negative_ncen] = 0
        idx_negative_nsat = np.where(self.toy_nsat < 0)[0]
        self.toy_nsat[idx_negative_nsat] = 0
        idx_ncen_exceeds_unity = np.where(self.toy_ncen > 1)[0]
        self.toy_ncen[idx_ncen_exceeds_unity] = 1


    @property 
    def primary_halo_property_key(self):
        """ Model is based on :math:`M = M_{vir}`.
        """
        return 'MVIR'

    def impose_binsize_constraints(self,mass_mask,binsize):
        """ Make sure that the user-supplied mask_mask does not result 
        in overlapping mass bins.
        """
        output_binsize = binsize
        midpoint_differences = np.diff(mass_mask)
        minimum_separation = midpoint_differences.min()
        if minimum_separation < output_binsize:
            output_binsize = minimum_separation

        return output_binsize

    def mean_ncen(self,logM,halo_type):
        """ Expected number of central galaxies in a halo of mass logM.
        For any input logM in the range of 
        mass_mask[ii]-binsize/2 < logM < mass_mask[ii] + binsize/2 for some ii, 
        set :math:`\\langle N_{cen} \\rangle` equal to the input toy_ncen[ii].

        Parameters
        ----------        
        logM : array 
            array of :math:`log_{10}(M)` of halos in catalog

        halo_type : array 
            array of halo types. Entirely ignored in this model. 

        Returns
        -------
        mean_ncen : array
            array giving :math:`\\langle N_{cen} \\rangle`
    
        """
        logM = np.array(logM)
        mean_ncen = np.zeros(len(logM))

        for ii,logmass in enumerate(self.mass_mask):
            idx_logM_in_binii = (
                (logM > self.mass_mask[ii] - self.binsize/2.) & 
                (logM < self.mass_mask[ii] + self.binsize/2.))
            mean_ncen[idx_logM_in_binii] = self.toy_ncen[ii]

        return mean_ncen

    def mean_nsat(self,logM,halo_type):
        """ Expected number of satellite galaxies in a halo of mass logM.
        For any input logM in the range of 
        mass_mask[ii]-binsize/2 < logM < mass_mask[ii] + binsize/2 for some ii, 
        set :math:`\\langle N_{sat} \\rangle` equal to the input toy_nsat[ii].

        Parameters
        ----------        
        logM : array 
            array of :math:`log_{10}(M)` of halos in catalog

        halo_type : array 
            array of halo types. Entirely ignored in this model. 

        Returns
        -------
        mean_nsat : array
            array giving :math:`\\langle N_{sat} \\rangle`
    
        """
        logM = np.array(logM)
        mean_nsat = np.zeros(len(logM))

        for ii,logmass in enumerate(self.mass_mask):
            idx_logM_in_binii = (
                (logM > self.mass_mask[ii] - self.binsize/2.) & 
                (logM < self.mass_mask[ii] + self.binsize/2.))
            mean_nsat[idx_logM_in_binii] = self.toy_nsat[ii]

        return mean_nsat

    def mean_concentration(self,logM,halo_type):
        """
        Concentration-mass relation of the model.

        Parameters 
        ----------

        logM : array_like
            array of :math:`log_{10}(M)` of halos in catalog

        halo_type : array 
            array of halo types. Entirely ignored in this model. 
            Included as a passed variable purely for consistency 
            between the way this function is called by different models.

        Returns 
        -------

        concentrations : array_like
            Mean concentration of halos of the input mass, using `anatoly_concentration` model.

        """

        logM = np.array(logM)

        concentrations = anatoly_concentration(logM)
        return concentrations



@six.add_metaclass(ABCMeta)
class Assembias_HOD_Model(HOD_Model):
    """ Abstract base class for any HOD model with assembly bias. 

    In this class of models, central and/or satellite mean occupation depends on some primary  
    property, such as :math:`M_{vir}`, and the mean occupations are modulated by some secondary property, 
    such as :math:`z_{form}`. 

    Notes 
    -----
    All implementations of assembly bias are formulated in such a way that 
    the baseline occupation statistics are kept fixed. For example, suppose that a subclass 
    of this class introduces a dependence of :math:`\\langle N_{cen} \\rangle` on the 
    secondary halo property :math:`z_{form}`, and that the primary halo property is the traditional 
    :math:`M = M_{vir}`. Then in narrow bins of :math:`M`, host halos in the simulation 
    will be rank-ordered by :math:`z_{form}`, 
    and assigned a halo type :math:`h_{0}` or :math:`h_{1}`, 
    where the halos may be split by any fraction into the two halo types, 
    and this fractional split may vary with :math:`M` according to 
    any arbitrary function :math:`P^{cen}_{h_{1}}(M)`  supplied by the model 
    via the `halo_type1_fraction_centrals` method. 
    The :math:`z_{form}`-dependence of central occupation is governed by 
    the central inflection function :math:`\\mathcal{I}_{cen}(M | h_{i})` via 
    :math:`\\langle N_{cen} | h_{i} \\rangle_{M} = \\mathcal{I}_{cen}(M | h_{i})\\langle N_{cen} \\rangle_{M}`. 
    The subclass implementing assembly bias need only supply any arbitrary function 
    :math:`\\tilde{\\mathcal{I}}_{cen}(M | h_{i})` via the `unconstrained_central_inflection_halo_type1` 
    method, and a variety of the built-in methods of this class will automatically apply appropriate boundary 
    conditions to determine :math:`\\mathcal{I}_{cen}(M | h_{i})` from :math:`\\tilde{\\mathcal{I}}_{cen}(M | h_{i})`, 
    such that the following constraint is satisfied: 

    :math:`\\langle N_{cen} \\rangle_{M} = P^{cen}_{h_{0}}(M)\\langle N_{cen} | h_{0} \\rangle_{M} 
    + P^{cen}_{h_{1}}(M)\\langle N_{cen} | h_{1} \\rangle_{M}`. 

    Therefore, assembly bias of arbitrary strength can be introduced in a way that preserves the 
    baseline occupation statistics, allowing users of halotools to isolate the pure influence 
    of assembly bias on any observational statistic that can be computed in a mock. 

    There are entirely independent functions governing satellite galaxy assembly bias, 
    allowing the role of centrals and satellites to be parsed. The secondary property 
    used to modulate the occupation statistics of centrals can be distinct from the property 
    modulating satellite occupation.

    The secondary halo property can be any halo property computable from a halo catalog and/or merger tree. 
    The user need only change `secondary_halo_property_centrals_key` and/or 
    `secondary_halo_property_satellites_key` to create identical assembly-biased models based on 
    different secondary properties. Likewise, the primary halo property may also be varied 
    by changing `primary_halo_property_key`, provided that the user supplies a baseline HOD model 
    predicated upon the suppiled primary halo property.
    Thus this class allows one to create mock universes possessing 
    assembly bias of arbitrary strength and character, 
    subject to the caveat that the occupation statistics can be parameterized 
    by two halo properties. 

    Finally, the secondary property modulating the assembly bias 
    need not be a physical attribute of the host halo. 
    For example, the `Satcen_Correlation_Polynomial_HOD_Model` subclass 
    can be used to create mocks in which of satellite occupation statistics are 
    modulated by whether or not there is a central galaxy residing in the host halo.

    """

    def __init__(self):

        # Executing the __init__ of the abstract base class HOD_Model 
        #sets self.parameter_dict to None, self.threshold to None, 
        # and self.publication to []
        HOD_Model.__init__(self)

    @abstractproperty
    def baseline_hod_model(self):
        """ Underlying HOD model, about which assembly bias modulates 
        galaxy abundance and intra-halo spatial distribution. 
        Must be one of the supported subclasses of `HOD_Model`. 
        The baseline HOD model can in principle be driven 
        by any host halo property. 
        """
        pass

    @abstractproperty
    def primary_halo_property_key(self):
        """ String providing halo catalog dictionary key pointing 
        to primary halo property. Necessary to ensure self-consistency between 
        underlying halo model, occupation-dependence of assembly bias, 
        and color-dependence of assembly bias. 

        """
        raise NotImplementedError("primary_halo_property_key "
            "needs to be implemented to ensure self-consistency "
            "of baseline HOD and assembly-biased HOD model features")
        pass

    @abstractproperty
    def secondary_halo_property_centrals_key(self):
        """ String providing halo catalog dictionary key pointing 
        to primary halo property. Necessary to ensure self-consistency between 
        underlying halo model, occupation-dependence of assembly bias, 
        and color-dependence of assembly bias. 

        """
        raise NotImplementedError("secondary_halo_property_centrals_key "
            "needs to be implemented to ensure self-consistency "
            "of baseline HOD and assembly-biased HOD model features")
        pass

    @abstractproperty
    def secondary_halo_property_satellites_key(self):
        """ String providing halo catalog dictionary key pointing 
        to primary halo property. Necessary to ensure self-consistency between 
        underlying halo model, occupation-dependence of assembly bias, 
        and color-dependence of assembly bias. 

        """
        raise NotImplementedError("secondary_halo_property_centrals_key "
            "needs to be implemented to ensure self-consistency "
            "of baseline HOD and assembly-biased HOD model features")
        pass



    @abstractmethod
    def unconstrained_central_inflection_halo_type1(self,primary_halo_property):
        """ Method determining :math:`\\tilde{\\mathcal{I}}_{cen}(p | h_{1})`, 
        the unconstrained excess probability that halos of primary property :math:`p` and 
        secondary property type :math:`h_{1}` 
        host a central galaxy. 

        Can be any arbitrary function, 
        subject only to the requirement that it be bounded. 
        Constraints on the value of this function required in order to keep the baseline 
        :math:`\\langle N_{cen} \\rangle_{p}` fixed 
        are automatically applied by `inflection_centrals`. 

        Notes 
        -----
        If this function is set to be either identically unity or identically zero, 
        there will be no assembly bias effects for centrals.

        """
        raise NotImplementedError(
            "unconstrained_central_inflection_halo_type1 is not implemented")
        pass

    @abstractmethod
    def unconstrained_satellite_inflection_halo_type1(self,primary_halo_property):
        """ Method determining :math:`\\tilde{\\mathcal{I}}_{sat}(p | h_{1})`, 
        the unconstrained excess probability that halos of primary property :math:`p` and 
        secondary property type :math:`h_{1}` 
        host a satellite galaxy. 

        Can be any arbitrary function, 
        subject only to the requirement that it be bounded. 
        Constraints on the value of this function required in order to keep the baseline 
        :math:`\\langle N_{sat} \\rangle_{p}` fixed 
        are automatically applied by `inflection_satellites`. 

        Notes 
        -----
        If this function is set to be either identically unity or identically zero, 
        there will be no assembly bias effects for satellites.

        """
        raise NotImplementedError(
            "unconstrained_satellite_inflection_halo_type1 is not implemented")
        pass

    @abstractmethod
    def halo_type1_fraction_centrals(self,primary_halo_property):
        """ Determines :math:`F^{cen}_{h_{1}}(p)`, 
        the fractional representation of host halos of type :math:`h_{1}` 
        as a function of the primary halo property :math:`p`, as pertains to centrals. 

        Notes 
        -----
        If this function is set to be either identically unity or identically zero, 
        there will be no assembly bias effects for centrals, regardless of the 
        behavior of `unconstrained_central_inflection_halo_type1`.

        Code currently assumes that this function has already been guaranteed to 
        be bounded by zero and unity. This will need to be fixed to be more defensive, 
        so that any bounded function will automatically be converted to a proper PDF. 

         """
        raise NotImplementedError(
            "halo_type_fraction_centrals is not implemented")
        pass

    @abstractmethod
    def halo_type1_fraction_satellites(self,primary_halo_property):
        """ Determines :math:`F^{sat}_{h_{1}}(p)`, 
        the fractional representation of host halos of type :math:`h_{1}` 
        as a function of the primary halo property :math:`p`, as pertains to satellites. 


        Notes 
        -----
        If this function is set to be either identically unity or identically zero, 
        there will be no assembly bias effects for satellites, regardless of the 
        behavior of `unconstrained_satellite_inflection_halo_type1`.

        Code currently assumes that this function has already been guaranteed to 
        be bounded by zero and unity. This will need to be fixed to be more defensive, 
        so that any bounded function will automatically be converted to a proper PDF. 

         """
        raise NotImplementedError(
            "halo_type_fraction_satellites is not implemented")
        pass

    def halo_type_fraction_centrals(self,primary_halo_property,halo_type):
        """ Using the function `halo_type1_fraction_centrals` required by concrete subclasses,
        this method determines :math:`F^{cen}_{h_{i}}(p)`, the fractional representation 
        of host halos of input halo type :math:`h_{i}` 
        as a function of input primary halo property :math:`p`, as pertains to centrals.

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
            Each element gives the probability 
            that a halo with input primary halo property :math:`p` 
            has input halo type :math:`h_{i}`

         """

        output_halo_type_fraction = self.halo_type1_fraction_centrals(primary_halo_property)
        idx0 = np.where(halo_type == 0)[0]
        output_halo_type_fraction[idx0] = 1.0 - output_halo_type_fraction[idx0]

        return output_halo_type_fraction

    def halo_type_fraction_satellites(self,primary_halo_property,halo_type):
        """ Using the function `halo_type1_fraction_satellites` required by concrete subclasses,
        this method determines :math:`F^{sat}_{h_{i}}(p)`, the fractional representation 
        of host halos of input halo type :math:`h_{i}` 
        as a function of input primary halo property :math:`p`, as pertains to satellites.

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
            Each element gives the probability 
            that a halo with input primary halo property :math:`p` 
            has input halo type :math:`h_{i}`

         """
        output_halo_type_fraction = self.halo_type1_fraction_satellites(primary_halo_property)
        idx0 = np.where(halo_type == 0)[0]
        output_halo_type_fraction[idx0] = 1.0 - output_halo_type_fraction[idx0]

        return output_halo_type_fraction


    def maximum_inflection_centrals(self,primary_halo_property,halo_type):
        """ The maximum allowed value of the inflection function, as pertains to centrals.

        The combinatorics of assembly-biased HODs are such that 
        the inflection function :math:`\\mathcal{I}_{cen}(p | h_{i})` can exceed neither 
        :math:`1 / F_{h_{i}}^{cen}(p)`, nor :math:`1 / \\langle N_{cen} \\rangle_{p}`. 
        The first condition is necessary to keep fixed 
        the unconditioned mean central occupation :math:`\\langle N_{cen} \\rangle_{p}`; 
        the second condition is necessary to ensure that 
        :math:`\\langle N_{cen} | h_{i} \\rangle_{p} <= 1`.

        Additionally, :math:`F_{h_{i}}^{cen}(p) = 1 \Rightarrow \\mathcal{I}_{cen}(p | h_{i}) = 1`, 
        which is applied not by this function but within `inflection_centrals`. 

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
        output_maximum_inflection : array_like
            Maximum allowed value of the inflection function, as pertains to centrals.

        """
        # First initialize the output array to zero
        output_maximum_inflection_case1 = np.zeros(len(primary_halo_property))

        # Whenever there are some type 1 halos, 
        # set the maximum inflection function equal to 1/prob(type1 halos)
        halo_type_fraction = self.halo_type_fraction_centrals(
            primary_halo_property,halo_type)
        idx_positive = halo_type_fraction > 0
        output_maximum_inflection_case1[idx_positive] = 1./halo_type_fraction[idx_positive]

        # At this stage, maximum inflection still needs to be limited by <Ncen>
        # Initialize another array to test the second case
        output_maximum_inflection_case2 = np.zeros(len(primary_halo_property))
        # Compute <Ncen> in the baseline model
        mean_baseline_ncen = self.baseline_hod_model.mean_ncen(
            primary_halo_property,halo_type)
        # Where non-zero, set the case 2 condition to 1 / <Ncen>
        idx_nonzero_centrals = mean_baseline_ncen > 0
        output_maximum_inflection_case2[idx_nonzero_centrals] = (
            1./mean_baseline_ncen[idx_nonzero_centrals])

        # Now set the output array equal to the element-wise minimum of the above two arrays
        output_maximum_inflection = np.minimum(
            output_maximum_inflection_case1,output_maximum_inflection_case2)

        return output_maximum_inflection

    def minimum_inflection_centrals(self,primary_halo_property,halo_type):
        """ The minimum allowed value of the inflection function, as pertains to centrals.

        The combinatorics of assembly-biased HODs are such that 
        the inflection function :math:`\\mathcal{I}_{cen}(p | h_{0,1})` can neither be negative 
        (which would be uphysical) nor fall below 
        :math:`\\frac{1 - P_{h_{1,0}}(p) / \\langle N_{cen} \\rangle_{p}}{P_{h_{0,1}}(p)}`

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
        output_maximum_inflection : array_like
            Maximum allowed value of the inflection function, as pertains to centrals.


        """
        minimum_inflection_centrals = np.zeros(len(primary_halo_property))

        mean_ncen = self.baseline_hod_model.mean_ncen(
            primary_halo_property,halo_type)

        halo_type_fraction = self.halo_type_fraction_centrals(
            primary_halo_property,halo_type)
        complementary_halo_type_fraction = 1 - halo_type_fraction

        idx_both_positive = ((halo_type_fraction > 0) & (mean_ncen > 0))

        minimum_inflection_centrals[idx_both_positive] = (1 - 
            (complementary_halo_type_fraction[idx_both_positive]/
                mean_ncen[idx_both_positive]))/halo_type_fraction[idx_both_positive]

        idx_negative = (minimum_inflection_centrals < 0)
        minimum_inflection_centrals[idx_negative] = 0

        return minimum_inflection_centrals


    def maximum_inflection_satellites(self,primary_halo_property,halo_type):
        """ Maximum allowed value of the inflection function, as pertains to satellites.

        The combinatorics of assembly-biased HODs are such that 
        the inflection function :math:`\\mathcal{I}_{sat}(p | h_{i})` can exceed 
        :math:`1 / F_{h_{i}}^{sat}(p)`, or it would be impossible to keep fixed 
        the unconditioned mean satellite occupation :math:`\\langle N_{sat} \\rangle_{p}`. 

        Additionally, :math:`F_{h_{i}}^{sat}(p) = 1 \Rightarrow \\mathcal{I}_{sat}(p | h_{i}) = 1`, 
        which is applied not by this function but within `inflection_satellites`. 

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
        output_maximum_inflection : array_like
            Maximum allowed value of the inflection function, as pertains to satellites.

        """

        output_maximum_inflection = np.zeros(len(primary_halo_property))
        halo_type_fraction = self.halo_type_fraction_satellites(
            primary_halo_property,halo_type)
        idx_positive = halo_type_fraction > 0
        output_maximum_inflection[idx_positive] = 1./halo_type_fraction[idx_positive]
        return output_maximum_inflection


    def inflection_satellites(self,primary_halo_property,halo_type):
        """ Method determining :math:`\\mathcal{I}_{sat}(p | h_{i})`, 
        the true excess probability that halos of primary property :math:`p` and 
        secondary property type :math:`h_{i}` 
        host a satellite galaxy. 

        :math:`\\langle N_{sat} | h_{i} \\rangle_{p} \equiv \\mathcal{I}_{sat}(p | h_{i}) \\langle N_{sat} \\rangle_{p}`.

        All of the behavior of this function derives 
        from `unconstrained_satellite_inflection_halo_type1` and `halo_type1_fraction_satellites`, 
        both of which are required methods of the concrete subclass. The function 
        :math:`\\mathcal{I}_{sat}(p | h_{i})` only differs from :math:`\\tilde{\\mathcal{I}}_{sat}(p | h_{i})` 
        in regions of HOD parameter space where the provided values of 
        :math:`\\tilde{\\mathcal{I}}_{sat}(p | h_{i})` would violate the following constraint: 

        :math:`F^{sat}_{h_{0}}(p)\\langle N_{sat} | h_{0} \\rangle_{p} + 
        F^{sat}_{h_{1}}(p)\\langle N_{sat} | h_{1} \\rangle_{p} = 
        \\langle N_{sat} \\rangle_{p}^{baseline},`
        where the RHS is given by `baseline_hod_model`. 

        Defining :math:`\\mathcal{I}_{sat}(p | h_{i})` in this way 
        guarantees that the parameters modulating assembly bias have zero intrinsic covariance with parameters governing  
        the traditional HOD. Therefore, any degeneracy between the assembly bias parameters 
        and the traditional HOD parameters in the posterior likelihood is purely due to degenerate effects 
        of the parameters on the chosen observable. 

        """
 
        idx0 = np.where(halo_type == 0)[0]
        idx1 = np.where(halo_type == 1)[0]

        # Initialize array containing result to return
        output_inflection_allhalos = np.zeros(len(primary_halo_property))

        all_ones = np.zeros(len(primary_halo_property)) + 1

        # Start by ignoring the input halo_type, and  
        # assuming the halo_type = 1 for all inputs.
        # This is convenient and costs nothing, 
        # since the halo_type = 0 branch 
        # is defined in terms of the halo_type = 1 branch.
        output_inflection_allhalos = (
            self.unconstrained_satellite_inflection_halo_type1(
                primary_halo_property))
        ########################################
        # Now apply the baseline HOD constraints to output_inflection_allhalos, 
        # still behaving as if every input halo has halo_type=1
        # First, require that the inflection function never exceed the 
        # maximum allowed value. This guarantees that < Nsat | h0 > >= 0, 
        # and ensures that it will be possible to preserve the baseline HOD. 
        maximum = self.maximum_inflection_satellites(primary_halo_property,all_ones)
        test_exceeds_maximum = output_inflection_allhalos > maximum
        output_inflection_allhalos[test_exceeds_maximum] = maximum[test_exceeds_maximum]
        # Second, require that the satellite inflection function 
        # never exceed its minimum value of zero. This ensures < Nsat | h1 > >= 0
        test_negative = output_inflection_allhalos < 0
        output_inflection_allhalos[test_negative] = 0
        # Finally, require that the inflection function is set to unity 
        # whenever the probability of halo_type=1 equals unity
        # This requirement supercedes the previous two, and ensures that 
        # the central inflection in h1-halos will be ignored in cases 
        # where there are no h0-halos. This self-consistency condition is necessary because 
        # the unconstrained inflection function and the halo_type function 
        # are both independently specified by user-supplied subclasses.  
        probability_type1 = self.halo_type_fraction_satellites(
            primary_halo_property,all_ones)
        test_unit_probability = (probability_type1 == 1)
        output_inflection_allhalos[test_unit_probability] = 1
        ########################################
        # At this point, output_inflection_allhalos has been properly conditioned. 
        # However, we have been assuming that all input halo_type = 1.
        # We now need to compute the correct output 
        # for cases where input halo_type = 0.
        # Define some shorthands (bookkeeping convenience)
        output_inflection_input_halo_type0 = output_inflection_allhalos[idx0]
        primary_halo_property_input_halo_type0 = primary_halo_property[idx0]
        probability_type1_input_halo_type0 = probability_type1[idx0]
        probability_type0_input_halo_type0 = 1.0 - probability_type1_input_halo_type0
        # Whenever the fraction of halos of type=0 is zero, the inflection function 
        # for type0 halos should be set to zero.
        test_zero = (probability_type0_input_halo_type0 == 0)
        output_inflection_input_halo_type0[test_zero] = 0

        # For non-trivial cases, define the type0 inflection function 
        # in terms of the type1 inflection function in such a way that 
        # the baseline HOD will be unadulterated by assembly bias
        test_positive = (probability_type0_input_halo_type0 > 0)
        output_inflection_input_halo_type0[test_positive] = (
            (1.0 - output_inflection_input_halo_type0[test_positive]*
                probability_type1_input_halo_type0[test_positive])/
            probability_type0_input_halo_type0[test_positive])

        # Now write the results back to the output 
        output_inflection_allhalos[idx0] = output_inflection_input_halo_type0

        return output_inflection_allhalos

    def inflection_centrals(self,primary_halo_property,halo_type):
        """ Method determining :math:`\\mathcal{I}_{cen}(p | h_{i})`, 
        the true excess probability that halos of primary property :math:`p` and 
        secondary property type :math:`h_{i}` 
        host a central galaxy. 

        :math:`\\langle N_{cen} | h_{i} \\rangle_{p} \equiv \\mathcal{I}_{cen}(p | h_{i}) \\langle N_{cen} \\rangle_{p}`.

        All of the behavior of this function derives 
        from `unconstrained_central_inflection_halo_type1` and `halo_type1_fraction_centrals`, 
        both of which are required methods of the concrete subclass. The function 
        :math:`\\mathcal{I}_{cen}(p | h_{i})` only differs from :math:`\\tilde{\\mathcal{I}}_{cen}(p | h_{i})` 
        in regions of HOD parameter space where the provided values of 
        :math:`\\tilde{\\mathcal{I}}_{cen}(p | h_{i})` would violate the following constraint: 

        :math:`F^{cen}_{h_{0}}(p)\\langle N_{cen} | h_{0} \\rangle_{p} + 
        F^{cen}_{h_{1}}(p)\\langle N_{cen} | h_{1} \\rangle_{p} = 
        \\langle N_{cen} \\rangle_{p}^{baseline},`
        where the RHS is given by `baseline_hod_model`. 

        Defining :math:`\\mathcal{I}_{cen}(p | h_{i})` in this way 
        guarantees that the parameters modulating assembly bias have zero intrinsic covariance with parameters governing  
        the traditional HOD. Therefore, any degeneracy between the assembly bias parameters 
        and the traditional HOD parameters in the posterior likelihood is purely due to degenerate effects 
        of the parameters on the chosen observable. 

        """

        idx0 = np.where(halo_type == 0)[0]
        idx1 = np.where(halo_type == 1)[0]

        # Initialize array containing result to return
        output_inflection_allhalos = np.zeros(len(primary_halo_property))

        all_ones = np.zeros(len(primary_halo_property)) + 1

        # Start by ignoring the input halo_type, and  
        # assuming the halo_type = 1 for all inputs.
        # This is convenient and costs nothing, 
        # since the halo_type = 0 branch 
        # is defined in terms of the halo_type = 1 branch.
        output_inflection_allhalos = (
            self.unconstrained_central_inflection_halo_type1(
                primary_halo_property))
        ########################################
        # Now apply the baseline HOD constraints to output_inflection_allhalos, 
        # still behaving as if every input halo has halo_type=1
        #output_inflection_allhalos[test_negative] = 0
        # First, require that the inflection function never exceed the 
        # maximum allowed value. This guarantees that < Ncen | h0 > >= 0, 
        # that < Ncen | h1 > <= 1, and ensures that it will be possible to 
        # preserve the baseline HOD.
        maximum = self.maximum_inflection_centrals(primary_halo_property,all_ones)
        test_exceeds_maximum = output_inflection_allhalos > maximum
        output_inflection_allhalos[test_exceeds_maximum] = maximum[test_exceeds_maximum]
        # Next, require that the inflection function never falls below 
        # its minimum allowed value. This guarantees that < Ncen | h1 > >= 0 
        # that < Ncen | h0 > <= 1, and ensures that we will be able to preserve 
        # the baseline HOD. 
        minimum = self.minimum_inflection_centrals(primary_halo_property,all_ones)
        test_below_minimum = output_inflection_allhalos < minimum
        output_inflection_allhalos[test_below_minimum] = minimum[test_below_minimum]
        # Finally, require that the inflection function is set to unity 
        # whenever the probability of halo_type=1 equals unity
        # This requirement supercedes the previous two, and ensures that 
        # the central inflection in h1-halos will be ignored in cases 
        # where there are no h0-halos. This self-consistency condition is necessary because 
        # the unconstrained inflection function and the halo_type function 
        # are both independently specified by user-supplied subclasses.  
        probability_type1 = self.halo_type_fraction_centrals(
            primary_halo_property,all_ones)
        test_unit_probability = (probability_type1 == 1)
        output_inflection_allhalos[test_unit_probability] = 1
        ########################################
        # At this point, output_inflection_allhalos has been properly conditioned. 
        # However, we have been assuming that all input halo_type = 1.
        # We now need to compute the correct output 
        # for cases where input halo_type = 0.
        # Define some shorthands (bookkeeping convenience)
        output_inflection_input_halo_type0 = output_inflection_allhalos[idx0]
        primary_halo_property_input_halo_type0 = primary_halo_property[idx0]
        probability_type1_input_halo_type0 = probability_type1[idx0]
        probability_type0_input_halo_type0 = 1.0 - probability_type1_input_halo_type0
        # Whenever the fraction of halos of type=0 is zero, the inflection function 
        # for type0 halos should be set to zero.
        test_zero = (probability_type0_input_halo_type0 == 0)
        output_inflection_input_halo_type0[test_zero] = 0

        # For non-trivial cases, define the type0 inflection function 
        # in terms of the type1 inflection function in such a way that 
        # the baseline HOD will be unadulterated by assembly bias
        test_positive = (probability_type0_input_halo_type0 > 0)
        output_inflection_input_halo_type0[test_positive] = (
            (1.0 - output_inflection_input_halo_type0[test_positive]*
                probability_type1_input_halo_type0[test_positive])/
            probability_type0_input_halo_type0[test_positive])

        # Now write the results back to the output 
        output_inflection_allhalos[idx0] = output_inflection_input_halo_type0

        return output_inflection_allhalos

    def mean_ncen(self,primary_halo_property,halo_type):
        """ Override the baseline HOD method used to compute mean central occupation. 

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
        mean_ncen : array_like
            :math:`h_{i}`-conditioned mean central occupation as a function of the primary halo property :math:`p`.

        :math:`\\langle N_{cen} | h_{i} \\rangle_{p} = \\mathcal{I}_{cen}(p | h_{i})\\langle N_{cen} \\rangle_{p}`

        """
        return self.inflection_centrals(primary_halo_property,halo_type)*(
            self.baseline_hod_model.mean_ncen(primary_halo_property,halo_type))

    def mean_nsat(self,primary_halo_property,halo_type):
        """ Override the baseline HOD method used to compute mean satellite occupation. 

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
        mean_nsat : array_like
            :math:`h_{i}`-conditioned mean satellite occupation as a function of the primary halo property :math:`p`.

        :math:`\\langle N_{sat} | h_{i} \\rangle_{p} = \\mathcal{I}_{sat}(p | h_{i})\\langle N_{sat} \\rangle_{p}`

        """
        return self.inflection_satellites(primary_halo_property,halo_type)*(
            self.baseline_hod_model.mean_nsat(primary_halo_property,halo_type))

    def halo_type_calculator(self, 
        primary_halo_property, secondary_halo_property,
        halo_type_fraction_function,
        bin_spacing = model_defaults.default_halo_type_calculator_spacing):
        """ Determines the assembly bias type of the input halos, as pertains to centrals.

        Method bins the input halos by the primary halo property :math:`p`, splits each bin 
        according to the value of `halo_type_fraction_centrals` in the bin, 
        and assigns halo type :math:`h_{0} (h_{1})` to the halos below (above) the split.

        Parameters 
        ----------
        primary_halo_property : array_like
            Array with elements equal to the primary_halo_property 

        secondary_halo_property : array_like
            Array with elements equal to the value of the secondary_halo_property 

        Returns 
        -------
        halo_type : array_like
            Array with elements equal to 0 or 1, specifying the type of the halo 
            whose fractional representation is being returned.

        """
        halo_types = np.ones(len(primary_halo_property))

        # set up binning scheme
        # Uses numpy.linspace, so the primary halo property 
        # is presumed to be a logarithmic quantity
        # Therefore, be careful if not using logMvir
        minimum = primary_halo_property.min()
        maximum = primary_halo_property.max() + model_defaults.default_bin_max_epsilon
        Nbins = int(round((maximum-minimum)/bin_spacing))
        primary_halo_property_bins = np.linspace(minimum,maximum,num=Nbins)

        # Determine the fraction by which 
        # each bin in the primary halo property should be split
        bin_midpoints = (primary_halo_property_bins[0:-1] + 
            np.diff(primary_halo_property_bins)/2.)

        bin_splitting_fraction = (np.ones(len(bin_midpoints)) - 
            np.array(halo_type_fraction_function(bin_midpoints)))

        # Find the bin index of every halo
        array_of_bin_indices = np.digitize(primary_halo_property,primary_halo_property_bins)-1

        # Loop over bins of the primary halo property 
        # containing at least one member
        for bin_index_i in set(array_of_bin_indices):
            # For all halos in bin = bin_index_i, 
            # grab their secondary halo property
            secondary_property_of_halos_with_bin_index_i = (
                secondary_halo_property[(array_of_bin_indices==bin_index_i)])
            # grab the corresponding elements of the output array
            halo_types_with_bin_index_i = (
                halo_types[(array_of_bin_indices==bin_index_i)])
            # determine how the halos in this bin should be sorted
            array_of_indices_that_would_sort_bin_i = (
                np.argsort(secondary_property_of_halos_with_bin_index_i))
            # split the bin according to the input halo type fraction function
            bin_splitting_index = int(round(
                len(array_of_indices_that_would_sort_bin_i)*
                bin_splitting_fraction[bin_index_i]))
            # Now for all halos in bin i 
            # whose secondary property is below the splitting fraction of the bin, 
            # set the halo type of those halos equal to zero.
            # Since the halo type array was initialized to unity, 
            # the remaining halo types pertaining to secondary property values 
            # above the splitting fraction of the bin 
            # already have their halo type set correctly
            #print('BEFORE: maximum_halo_types_with_bin_index_i = ',halo_types_with_bin_index_i.max())
            #print('len(array_of_indices_that_would_sort_bin_i) = ',len(array_of_indices_that_would_sort_bin_i))
            #print('bin_splitting_index = ',bin_splitting_index)
            halo_types_with_bin_index_i[array_of_indices_that_would_sort_bin_i[0:bin_splitting_index]] = 0
            #print('AFTER: maximum_halo_types_with_bin_index_i = ',halo_types_with_bin_index_i.max())
            #print('')
            # Finally, write these values back to the output array 
            halo_types[(array_of_bin_indices==bin_index_i)] = halo_types_with_bin_index_i

        return halo_types


class Satcen_Correlation_Polynomial_HOD_Model(Assembias_HOD_Model):
    """ HOD-style model in which satellite abundance 
    is correlated with the presence of a central galaxy.

    Notes 
    -----
    This is special case of assembly biased occupation statistics. 
    It is implemented in `~make_mocks.HOD_Mock` 
    by setting the halo type of satellites after centrals have been 
    assigned to halos (satellite halo type = 1 if there is a central in the halo).

    """

    def __init__(self,baseline_hod_model=Zheng07_HOD_Model,
            baseline_hod_parameter_dict=None,threshold=None,
            assembias_parameter_dict=None):

        baseline_hod_model_instance = (
            baseline_hod_model(threshold=threshold,parameter_dict=baseline_hod_parameter_dict)
            )

        if not isinstance(baseline_hod_model_instance,HOD_Model):
            raise TypeError(
                "Input baseline_hod_model must be one of "
                "the supported HOD_Model objects defined in this module or by the user")
        # Temporarily store the baseline HOD model object
        # into a "private" attribute. This is a clunky workaround
        # to python's awkward conventions for required abstract properties
        self._baseline_hod_model = baseline_hod_model_instance

        # Executing the __init__ of the abstract base class Assembias_HOD_Model 
        # does nothing besides executing the __init__ of the abstract base class HOD_Model 
        # Executing the __init__ of the abstract base class HOD_Model 
        # sets self.parameter_dict to None, self.threshold to None, 
        # and self.publication to []        
        Assembias_HOD_Model.__init__(self)


        self.threshold = threshold

        self.publication.extend(self._baseline_hod_model.publication)
        self.baseline_hod_parameter_dict = self._baseline_hod_model.parameter_dict

        if assembias_parameter_dict == None:
            self.assembias_parameter_dict = model_defaults.default_satcen_parameters
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
        return self.baseline_hod_model.primary_halo_property_key

    @property 
    def secondary_halo_property_centrals_key(self):
        return None

    @property 
    def secondary_halo_property_satellites_key(self):
        return None

    def mean_concentration(self,primary_halo_property,halo_type):
        """ Concentration-halo relation assumed by the underlying HOD_Model object.
        The appropriate method is already bound to the self.baseline_hod_model object.

        Parameters 
        ----------
        primary_halo_property : array_like
            array of primary halo property governing the occupation statistics 

        halo_type : array 
            array of halo types. 

        Returns 
        -------
        concentrations : numpy array

        """

        concentrations = self.baseline_hod_model.mean_concentration(
            primary_halo_property,halo_type)
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
        all_ones = np.ones(len(primary_halo_property))

        output_array = np.array(
            self.baseline_hod_model.mean_ncen(
                primary_halo_property,all_ones))

        return output_array

    def unconstrained_polynomial_model(self,abcissa,ordinates,primary_halo_property):
        coefficient_array = solve_for_polynomial_coefficients(
            abcissa,ordinates)
        output_unconstrained_inflection_function = (
            np.zeros(len(primary_halo_property)))

        # Use coefficients to compute values of the inflection function polynomial
        for n,coeff in enumerate(coefficient_array):
            output_unconstrained_inflection_function += coeff*primary_halo_property**n

        return output_unconstrained_inflection_function

    def unconstrained_central_inflection_halo_type1(self,primary_halo_property):
        abcissa = self.parameter_dict['assembias_abcissa']
        ordinates = self.parameter_dict['central_assembias_ordinates']

        return self.unconstrained_polynomial_model(abcissa,ordinates,primary_halo_property)

    def unconstrained_satellite_inflection_halo_type1(self,primary_halo_property):
        abcissa = self.parameter_dict['assembias_abcissa']
        ordinates = self.parameter_dict['satellite_assembias_ordinates']

        return self.unconstrained_polynomial_model(abcissa,ordinates,primary_halo_property)

    def require_correct_keys(self,assembias_parameter_dict):
        # What is the purpose of using "set" here? The .keys() method never returns duplicates
        correct_set_of_satcen_keys = set(model_defaults.default_satcen_parameters.keys())
        if set(assembias_parameter_dict.keys()) != correct_set_of_satcen_keys:
            raise TypeError("Set of keys of input assembias_parameter_dict"
            " does not match the set of keys required by the model." 
            " Correct set of keys is {'assembias_abcissa',"
            "'satellite_assembias_ordinates', 'central_assembias_ordinates'}. ")
        pass


class Polynomial_Assembias_HOD_Model(Assembias_HOD_Model):
    """ Concrete subclass of `Assembias_HOD_Model`  
    in which occupation statistics exhibit assembly bias, 
    where some secondary host halo property modulates the mean galaxy abundance. 
    The strength of the assembly bias is set by explicitly specifing the strength 
    at specific values of the primary halo property. 
    """

    def __init__(self,baseline_hod_model=Zheng07_HOD_Model,
            baseline_hod_parameter_dict=None,
            threshold=model_defaults.default_luminosity_threshold,
            assembias_parameter_dict=None,
            secondary_halo_property_centrals_key=model_defaults.default_assembias_key,
            secondary_halo_property_satellites_key=model_defaults.default_assembias_key):


        baseline_hod_model_instance = (
            baseline_hod_model(threshold=threshold,parameter_dict=baseline_hod_parameter_dict)
            )
        if not isinstance(baseline_hod_model_instance,HOD_Model):
            raise TypeError(
                "Input baseline_hod_model must be one of "
                "the supported HOD_Model objects defined in this module or by the user")
        # Temporarily store a few "private" attributes. This is a clunky workaround
        # to python's awkward conventions for required abstract properties
        self._baseline_hod_model = baseline_hod_model_instance
        self._secondary_halo_property_centrals_key = secondary_halo_property_centrals_key
        self._secondary_halo_property_satellites_key = secondary_halo_property_satellites_key

        # Executing the __init__ of the abstract base class Assembias_HOD_Model 
        # does nothing besides executing the __init__ of the abstract base class HOD_Model 
        # Executing the __init__ of the abstract base class HOD_Model 
        # sets self.parameter_dict to None, self.threshold to None, 
        # and self.publication to []        
        Assembias_HOD_Model.__init__(self)


        self.threshold = threshold

        self.publication.extend(self._baseline_hod_model.publication)
        self.baseline_hod_parameter_dict = self._baseline_hod_model.parameter_dict

        if assembias_parameter_dict == None:
            self.assembias_parameter_dict = model_defaults.default_occupation_assembias_parameters
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
        return self.baseline_hod_model.primary_halo_property_key

    @property 
    def secondary_halo_property_centrals_key(self):
        return self._secondary_halo_property_centrals_key

    @property 
    def secondary_halo_property_satellites_key(self):
        return self._secondary_halo_property_satellites_key

    def mean_concentration(self,primary_halo_property,halo_type):
        """ Concentration-halo relation assumed by the underlying HOD_Model object.
        The appropriate method is already bound to the self.baseline_hod_model object.

        Parameters 
        ----------
        primary_halo_property : array_like
            array of primary halo property governing the occupation statistics 

        halo_type : array 
            array of halo types. 

        Returns 
        -------
        concentrations : numpy array

        """

        concentrations = (
            self.baseline_hod_model.mean_concentration(
                primary_halo_property,halo_type))
        return concentrations


    def halo_type1_fraction_centrals(self,primary_halo_property):
        """ Determines the fractional representation of host halo 
        types as a function of the value of the primary halo property.

        Halo types can be either given by fixed-Mvir rank-orderings 
        of the host halos, or by the input occupation statistics functions.

        """
        # In this model, centrals exhibit no assembly bias
        # So simply set the halo type1 fraction to unity for centrals
        abcissa = model_defaults.default_halo_type_split['halo_type_split_abcissa']
        ordinates = model_defaults.default_halo_type_split['halo_type_split_ordinates']
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

        abcissa = model_defaults.default_halo_type_split['halo_type_split_abcissa']
        ordinates = model_defaults.default_halo_type_split['halo_type_split_ordinates']
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
        output_unconstrained_inflection_function = (
            np.zeros(len(primary_halo_property)))

        # Use coefficients to compute values of the inflection function polynomial
        for n,coeff in enumerate(coefficient_array):
            output_unconstrained_inflection_function += coeff*primary_halo_property**n

        return output_unconstrained_inflection_function

    def unconstrained_central_inflection_halo_type1(self,primary_halo_property):
        abcissa = self.parameter_dict['assembias_abcissa']
        ordinates = self.parameter_dict['central_assembias_ordinates']

        return self.unconstrained_polynomial_model(abcissa,ordinates,primary_halo_property)

    def unconstrained_satellite_inflection_halo_type1(self,primary_halo_property):
        abcissa = self.parameter_dict['assembias_abcissa']
        ordinates = self.parameter_dict['satellite_assembias_ordinates']

        return self.unconstrained_polynomial_model(abcissa,ordinates,primary_halo_property)

    def require_correct_keys(self,assembias_parameter_dict):
        correct_set_of_assembias_keys = set(model_defaults.default_occupation_assembias_parameters.keys())
        if set(assembias_parameter_dict.keys()) != correct_set_of_assembias_keys:
            raise TypeError("Set of keys of input assembias_parameter_dict"
            " does not match the set of keys required by the model." 
            " Correct set of keys is {'assembias_abcissa',"
            "'satellite_assembias_ordinates', 'central_assembias_ordinates'}. ")
        pass



@six.add_metaclass(ABCMeta)
class HOD_Quenching_Model(HOD_Model):
    """ Abstract base class for models determining mock galaxy quenching. 
    A subclass of `HOD_Mock`, this class additionally requires methods specifying 
    the quenched fractions of centrals and satellites.  
    
    """

    def __init__(self):

        # Executing the __init__ of the abstract base class HOD_Model 
        #sets self.parameter_dict to None, self.threshold to None, 
        # and self.publication to []
        HOD_Model.__init__(self)

    @abstractproperty
    def baseline_hod_model(self):
        """ Underlying HOD model.

        Must be one of the supported subclasses of `HOD_Model`. 
        The baseline HOD model can in principle be driven 
        by any host halo property, and can have distinct 
        quenched/star-forming SMHM relations.
        """
        pass

    @abstractmethod
    def mean_quenched_fraction_centrals(self,primary_halo_property,halo_type):
        """
        Expected fraction of centrals that are quenched as a function of 
        the primary halo property.
        A required method for any halo occupation object with quenching designation.
        """
        raise NotImplementedError(
            "quenched_fraction_centrals is not implemented")

    @abstractmethod
    def mean_quenched_fraction_satellites(self,primary_halo_property,halo_type):
        """
        Expected fraction of satellites that are quenched as a function of 
        the primary halo property.
        A required method for any halo occupation object with quenching designation.
        """
        raise NotImplementedError(
            "quenched_fraction_satellites is not implemented")



class vdB03_Quenching_Model(HOD_Quenching_Model):
    """
    Subclass of `HOD_Quenching_Model`, providing a traditional HOD model of galaxy quenching, 
    in which quenching designation is purely determined by host halo virial mass.
    
    Approach is adapted from van den Bosch 2003. The desired quenched fraction is specified 
    at a particular set of masses, and the code then uses the unique, minimal-degree 
    polynomial passing through those points to determine the quenched fraction at any mass. 
    The desired quenched fraction must be independently specified for centrals and satellites. 

    Notes 
    -----
    All-galaxy central and satellite occupation 
    statistics are specified first; Zheng07_HOD_Model is the default choice, 
    but any supported HOD_Mock object could be chosen. A quenching designation is subsequently 
    applied to the galaxies. 
    Thus in this class of models, 
    the central galaxy SMHM has no dependence on quenched/active designation.

    """

    def __init__(self,baseline_hod_model=Zheng07_HOD_Model,
        baseline_hod_parameter_dict=None,threshold=None,
        quenching_parameter_dict=None):


        # Executing the __init__ of the abstract base class HOD_Quenching_Model 
        # does nothing besides executing the __init__ of the abstract base class HOD_Model 
        # Executing the __init__ of the abstract base class HOD_Model 
        # sets self.parameter_dict to None, self.threshold to None, 
        # and self.publication to []        
        HOD_Quenching_Model.__init__(self)

        baseline_hod_model_instance = (
            baseline_hod_model(
                threshold=threshold,
                parameter_dict=baseline_hod_parameter_dict)
            )
        if not isinstance(baseline_hod_model_instance,HOD_Model):
            raise TypeError(
                "Input baseline_hod_model must be one of "
                "the supported HOD_Model objects defined in this module or by the user")
        # Temporarily store the baseline HOD model object
        # into a "private" attribute. This is a clunky workaround
        # to python's awkward conventions for required abstract properties
        self._baseline_hod_model = baseline_hod_model_instance
        self.baseline_hod_parameter_dict = self._baseline_hod_model.parameter_dict

        self.threshold = self._baseline_hod_model.threshold


        self.publication.extend(self._baseline_hod_model.publication)
        self.publication.extend(['arXiv:0210495v3'])

        # The baseline HOD parameter dictionary is already an attribute 
        # of self.hod_model. That dictionary needs to be joined with 
        # the dictionary storing the quenching model parameters. 
        # If a quenching parameter dictionary is passed to the constructor,
        # concatenate that passed dictionary with the existing hod_model dictionary.
        # Otherwise, choose the default quenching model parameter set in model_defaults.py 
        # This should be more defensive. Fine for now.
        if quenching_parameter_dict is None:
            self.quenching_parameter_dict = model_defaults.default_quenching_parameters
        else:
            # If the user supplies a dictionary of quenching parameters, 
            # require that the set of keys is correct
            self.require_correct_keys(quenching_parameter_dict)
            # If correct, bind the input dictionary to the instance.
            self.quenching_parameter_dict = quenching_parameter_dict

        self.parameter_dict = dict(self.baseline_hod_parameter_dict.items() + 
            self.quenching_parameter_dict.items())

    @property
    def baseline_hod_model(self):
        return self._baseline_hod_model

    @property 
    def primary_halo_property_key(self):
        return 'MVIR'

    def mean_ncen(self,primary_halo_property,halo_type):
        """
        Expected number of central galaxies in a halo of mass logM.
        The appropriate method is already bound to the self.hod_model object.

        Parameters
        ----------
        logM : array 
            array of log10(Mvir) of halos in catalog

        halo_type : array 
            array of halo types. Entirely ignored in this model. 
            Included as a passed variable purely for consistency 
            between the way mean_ncen is called by different models.

        Returns
        -------
        mean_ncen : float or array
            Mean number of central galaxies in a host halo of the specified mass. 


        """
        mean_ncen = self.baseline_hod_model.mean_ncen(
            primary_halo_property,halo_type)
        return mean_ncen

    def mean_nsat(self,primary_halo_property,halo_type):
        """
        Expected number of satellite galaxies in a halo of mass logM.
        The appropriate method is already bound to the self.hod_model object.

        Parameters
        ----------
        logM : array 
            array of log10(Mvir) of halos in catalog

        halo_type : array 
            array of halo types. Entirely ignored in this model. 
            Included as a passed variable purely for consistency 
            between the way mean_ncen is called by different models.

        Returns
        -------
        mean_nsat : float or array
            Mean number of satellite galaxies in a host halo of the specified mass. 


        """
        mean_nsat = self.baseline_hod_model.mean_nsat(
            primary_halo_property,halo_type)
        return mean_nsat

    def mean_concentration(self,primary_halo_property,halo_type):
        """ Concentration-mass relation assumed by the underlying HOD_Model object.
        The appropriate method is already bound to the self.hod_model object.

        Parameters 
        ----------
        logM : array 
            array of log10(Mvir) of halos in catalog

        halo_type : array 
            array of halo types. Entirely ignored in this model. 
            Included as a passed variable purely for consistency 
            between the way mean_ncen is called by different models.

        Returns 
        -------
        concentrations : array

        """

        concentrations = self.baseline_hod_model.mean_concentration(
            primary_halo_property,halo_type)
        return concentrations

    def quenching_polynomial_model(self,abcissa,ordinates,primary_halo_property):
        coefficient_array = solve_for_polynomial_coefficients(
            abcissa,ordinates)
        output_quenched_fractions = (
            np.zeros(len(primary_halo_property)))

        # Use coefficients to compute values of the inflection function polynomial
        for n,coeff in enumerate(coefficient_array):
            output_quenched_fractions += coeff*primary_halo_property**n

        test_negative = output_quenched_fractions < 0
        output_quenched_fractions[test_negative] = 0
        test_exceeds_unity = output_quenched_fractions > 1
        output_quenched_fractions[test_exceeds_unity] = 1

        return output_quenched_fractions

    def mean_quenched_fraction_centrals(self,primary_halo_property,halo_type):
        """
        Expected fraction of centrals that are quenched as a function of host halo mass logM.
        A required method for any HOD_Quenching_Model object.

        Parameters 
        ----------
        logM : array_like
            array of log10(Mvir) of halos in catalog

        halo_type : array 
            array of halo types. Entirely ignored in this model. 
            Included as a passed variable purely for consistency 
            between the way mean_ncen is called by different models.

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
        abcissa = self.quenching_parameter_dict['quenching_abcissa']
        ordinates = self.quenching_parameter_dict['central_quenching_ordinates']

        mean_quenched_fractions = self.quenching_polynomial_model(
            abcissa,ordinates,primary_halo_property)
 
        return mean_quenched_fractions

    def mean_quenched_fraction_satellites(self,primary_halo_property,halo_type):
        """
        Expected fraction of satellites that are quenched as a function of host halo mass logM.
        A required method for any HOD_Quenching_Model object.

        Parameters 
        ----------
        logM : array_like
            array of log10(Mvir) of halos in catalog

        halo_type : array 
            array of halo types. Entirely ignored in this model. 
            Included as a passed variable purely for consistency 
            between the way mean_ncen is called by different models.

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

        abcissa = self.quenching_parameter_dict['quenching_abcissa']
        ordinates = self.quenching_parameter_dict['satellite_quenching_ordinates']

        mean_quenched_fractions = self.quenching_polynomial_model(
            abcissa,ordinates,primary_halo_property)
 
        return mean_quenched_fractions

    def require_correct_keys(self,quenching_parameter_dict):
        correct_set_of_quenching_keys = set(model_defaults.default_quenching_parameters.keys())
        if set(quenching_parameter_dict.keys()) != correct_set_of_quenching_keys:
            raise TypeError("Set of keys of input quenching_parameter_dict"
            " does not match the set of keys required by the model." 
            " Correct set of keys is {'quenching_abcissa',"
            "'central_quenching_ordinates', 'satellite_quenching_ordinates'}. ")
        pass



@six.add_metaclass(ABCMeta)
class Assembias_HOD_Quenching_Model(Assembias_HOD_Model):
    """ Abstract base class for any HOD model in which 
    both galaxy abundance and galaxy quenching on Mvir 
    plus an additional property.

    """

    def __init__(self):

        # Executing the init constructor of the abstract base class Assembias_HOD_Model 
        # only executes the initialization constructor of HOD_Model.
        # This just sets self.parameter_dict to None, self.threshold to None, 
        # and self.publication to []
        Assembias_HOD_Model.__init__(self)


    @abstractmethod
    def unconstrained_central_conformity_halo_type1(self,primary_halo_property):
        """ Method determining :math:`\\tilde{\\mathcal{C}}_{cen_{Q}}(p | h_{1})`, 
        the unconstrained excess quenched fraction of centrals 
        in halos of primary property :math:`p` and 
        secondary property type :math:`h_{1}`.

        Can be any arbitrary function, 
        subject only to the requirement that it be bounded. 
        Constraints on the value of this function required 
        in order to keep the unconditioned quenched fraction  
        :math:`F_{cen_{Q}}(p)` fixed 
        are automatically applied by `conformity_centrals`. 

        Notes 
        -----
        If this function is set to be either identically unity or identically zero, 
        there will be no assembly bias effects for centrals.

        """
        raise NotImplementedError(
            "unconstrained_central_conformity_halo_type1 is not implemented")


    @abstractmethod
    def unconstrained_satellite_conformity_halo_type1(self,primary_halo_property):
        """ Method determining :math:`\\tilde{\\mathcal{C}}_{sat_{Q}}(p | h_{1})`, 
        the unconstrained excess quenched fraction of satellites 
        in halos of primary property :math:`p` and 
        secondary property type :math:`h_{1}`.

        Can be any arbitrary function, 
        subject only to the requirement that it be bounded. 
        Constraints on the value of this function required 
        in order to keep the unconditioned quenched fraction  
        :math:`F_{sat_{Q}}(p)` fixed 
        are automatically applied by `conformity_satellites`. 

        Notes 
        -----
        If this function is set to be either identically unity or identically zero, 
        there will be no assembly bias effects for centrals.

        """
        raise NotImplementedError(
            "unconstrained_central_conformity_halo_type1 is not implemented")

    def conformity_case_ratio_centrals(self,primary_halo_property,halo_type):
        """
        The bounds on the conformity function depend on the other HOD model parameters.
        This function determines which case should be used in computing the conformity bounds.

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
        conformity_case_ratio : array_like 
            Array giving the ratio that determines how maximum conformity is computed.

        """

        conformity_case_ratio = np.ones(len(primary_halo_property))

        inflection = self.inflection_centrals(primary_halo_property,halo_type)
        type_fraction = self.halo_type_fraction_centrals(primary_halo_property,halo_type)
        baseline_quenched_fraction = (
            self.baseline_hod_model.mean_quenched_fraction_centrals(
                primary_halo_property,halo_type)
            )

        idx_both_positive = ( (type_fraction > 0) & (inflection > 0) )

        conformity_case_ratio[idx_both_positive] = (
            baseline_quenched_fraction[idx_both_positive] / 
                (inflection[idx_both_positive]*type_fraction[idx_both_positive])
            )

        return conformity_case_ratio 

    def maximum_conformity_centrals(self,primary_halo_property,halo_type):
        """ The maximum allowed value of the conformity function, as pertains to centrals.

        The combinatorics of assembly-biased HODs are such that 
        the conformity function :math:`\\mathcal{C}_{cen_{Q}}(p | h_{i})` can exceed neither 
        :math:`1 / \\mathcal{I}_{cen}(p | h_{i})P_{h_{i}}(p)`, 
        nor :math:`1 / F_{cen_{Q}}(p | h_{i})`. 

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
        output_maximum_conformity : array_like
            Maximum allowed value of the conformity function, as pertains to centrals.

        """

        output_maximum_conformity = np.zeros(len(primary_halo_property))

        conformity_case_ratio = self.conformity_case_ratio_centrals(primary_halo_property,halo_type)

        inflection = self.inflection_centrals(primary_halo_property,halo_type)
        halo_type_fraction = self.halo_type_fraction_centrals(primary_halo_property,halo_type)
        baseline_quenched_fraction = (
            self.baseline_hod_model.mean_quenched_fraction_centrals(
                primary_halo_property,halo_type))

        idx_nontrivial_case1 = (
            (conformity_case_ratio < 1) & 
            (inflection > 0) & (halo_type_fraction > 0) )

        output_maximum_conformity[idx_nontrivial_case1] = (1. / 
            (inflection[idx_nontrivial_case1]*halo_type_fraction[idx_nontrivial_case1])
            )

        idx_nontrivial_case2 = (
            (conformity_case_ratio >= 1) & (baseline_quenched_fraction > 0) )

        output_maximum_conformity[idx_nontrivial_case2] = 1. / baseline_quenched_fraction[idx_nontrivial_case2]

        return output_maximum_conformity


    def minimum_conformity_centrals(self,primary_halo_property,halo_type):
        """ The minimum allowed value of the inflection function, as pertains to centrals.

        The combinatorics of assembly-biased HODs are such that 
        the conformity function :math:`\\mathcal{C}_{cen_{Q}}(p | h_{0,1})` 
        must exceed both a and b.

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
        output_minimum_conformity : array_like
            Minimum allowed value of the conformity function, as pertains to centrals.


        """
        output_minimum_conformity = np.zeros(len(primary_halo_property))

        opposite_halo_type = np.zeros(len(halo_type))
        opposite_halo_type[halo_type==0] = 1

        inflection = self.inflection_centrals(primary_halo_property,halo_type)
        halo_type_fraction = self.halo_type_fraction_centrals(primary_halo_property,halo_type)

        opposite_inflection = self.inflection_centrals(primary_halo_property,opposite_halo_type)
        opposite_halo_type_fraction = self.halo_type_fraction_centrals(primary_halo_property,opposite_halo_type)
        opposite_maximum_conformity = self.maximum_conformity_centrals(primary_halo_property,opposite_halo_type)

        idx_nontrivial_case = ( (inflection > 0) & (halo_type_fraction > 0) )

        output_minimum_conformity[idx_nontrivial_case] = (
            (1. - (opposite_halo_type_fraction[idx_nontrivial_case]*
                opposite_maximum_conformity[idx_nontrivial_case])) / (
            inflection[idx_nontrivial_case]*halo_type_fraction[idx_nontrivial_case])
                )

        return output_minimum_conformity


    def conformity_centrals(self,primary_halo_property,halo_type):
        """
        Conformity function as pertains to centrals

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
        output_conformity : array_like 
            Array giving the multiple by which the mean quenched fraction is boosted.
        """

        idx0 = np.where(halo_type == 0)[0]
        idx1 = np.where(halo_type == 1)[0]

        # Initialize array containing result to return
        output_conformity = np.zeros(len(primary_halo_property))

        all_ones = np.ones(len(primary_halo_property))

        # Start by ignoring the input halo_type, and  
        # assuming the halo_type = 1 for all inputs.
        # This is convenient and costs nothing, 
        # since the halo_type = 0 branch 
        # is defined in terms of the halo_type = 1 branch.
        output_conformity = (
            self.unconstrained_central_conformity_halo_type1(
                primary_halo_property))
        ########################################
        # Now apply the baseline HOD constraints to output_conformity, 
        # still behaving as if every input halo has halo_type=1
        maximum = self.maximum_conformity_centrals(primary_halo_property,all_ones)
        test_exceeds_maximum = output_conformity > maximum
        output_conformity[test_exceeds_maximum] = maximum[test_exceeds_maximum]
        # Next, require that the conformity function never falls below 
        # its minimum allowed value. This guarantees that ... 
        minimum = self.minimum_conformity_centrals(primary_halo_property,all_ones)
        test_below_minimum = output_conformity < minimum
        output_conformity[test_below_minimum] = minimum[test_below_minimum]
        # Finally, require that the conformity function is set to unity 
        # whenever the probability of halo_type=1 equals unity
        # This requirement supercedes the previous two, and ensures that 
        # the conformity inflection in h1-halos will be ignored in cases 
        # where there are no h0-halos. This self-consistency condition is necessary because 
        # the unconstrained conformity function and the halo_type function 
        # are both independently specified by user-supplied subclasses.  
        ###
        probability_type1 = self.halo_type_fraction_centrals(
            primary_halo_property,all_ones)
        idx_trivial_probability = (probability_type1 == 0) | (probability_type1 == 1)

        baseline_quenched_fraction = (
            self.baseline_hod_model.mean_quenched_fraction_centrals(
            primary_halo_property,all_ones)
            )
        idx_trivial_quenching = (baseline_quenched_fraction == 1)

        output_conformity[idx_trivial_probability] = 1
        output_conformity[idx_trivial_quenching] = 1

        ########################################
        # At this point, output_conformity has been properly conditioned. 
        # However, we have been assuming that all input halo_type = 1.
        # We now need to compute the correct output 
        # for cases where input halo_type = 0.
        # Define some shorthands (bookkeeping convenience)
        output_conformity_input_halo_type0 = output_conformity[idx0]
        primary_halo_property_input_halo_type0 = primary_halo_property[idx0]
        probability_type1_input_halo_type0 = probability_type1[idx0]
        probability_type0_input_halo_type0 = 1.0 - probability_type1_input_halo_type0
        # Whenever the fraction of halos of type=0 is zero, the conformity function 
        # for type0 halos should be set to unity.
        test_zero = (probability_type0_input_halo_type0 == 0)
        output_conformity_input_halo_type0[test_zero] = 1

        # For non-trivial cases, define the type0 conformity function 
        # in terms of the type1 conformity function in such a way that 
        # the baseline HOD will be unadulterated by assembly bias
        test_positive = (probability_type0_input_halo_type0 > 0)
        output_conformity_input_halo_type0[test_positive] = (
            (1.0 - output_conformity_input_halo_type0[test_positive]*
                probability_type1_input_halo_type0[test_positive])/
            probability_type0_input_halo_type0[test_positive])

        # Now write the results back to the output 
        output_conformity[idx0] = output_conformity_input_halo_type0

        return output_conformity


    def conformity_case_ratio_satellites(self,primary_halo_property,halo_type):
        """
        The bounds on the conformity function depend on the other HOD model parameters.
        This function determines which case should be used in computing the conformity bounds.

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
        conformity_case_ratio : array_like 
            Array giving the ratio that determines how maximum conformity is computed.

        """

        conformity_case_ratio = np.ones(len(primary_halo_property))

        inflection = self.inflection_satellites(primary_halo_property,halo_type)
        type_fraction = self.halo_type_fraction_satellites(primary_halo_property,halo_type)
        baseline_quenched_fraction = (
            self.baseline_hod_model.mean_quenched_fraction_satellites(primary_halo_property,halo_type))

        idx_both_positive = ( (type_fraction > 0) & (inflection > 0) )

        conformity_case_ratio[idx_both_positive] = (
            baseline_quenched_fraction[idx_both_positive] / 
                (inflection[idx_both_positive]*type_fraction[idx_both_positive])
            )

        return conformity_case_ratio 

    def maximum_conformity_satellites(self,primary_halo_property,halo_type):
        """ The maximum allowed value of the conformity function, as pertains to satellites.

        The combinatorics of assembly-biased HODs are such that 
        the conformity function :math:`\\mathcal{C}_{cen_{Q}}(p | h_{i})` can exceed neither 
        :math:`1 / \\mathcal{I}_{cen}(p | h_{i})P_{h_{i}}(p)`, 
        nor :math:`1 / F_{cen_{Q}}(p | h_{i})`. 

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
        output_maximum_conformity : array_like
            Maximum allowed value of the conformity function, as pertains to satellites.

        """

        output_maximum_conformity = np.zeros(len(primary_halo_property))

        conformity_case_ratio = self.conformity_case_ratio_satellites(primary_halo_property,halo_type)

        inflection = self.inflection_satellites(primary_halo_property,halo_type)
        halo_type_fraction = self.halo_type_fraction_satellites(primary_halo_property,halo_type)
        baseline_quenched_fraction = (
            self.baseline_hod_model.mean_quenched_fraction_satellites(primary_halo_property,halo_type))

        idx_nontrivial_case1 = (
            (conformity_case_ratio < 1) & 
            (inflection > 0) & (halo_type_fraction > 0) )

        output_maximum_conformity[idx_nontrivial_case1] = (1. / 
            (inflection[idx_nontrivial_case1]*halo_type_fraction[idx_nontrivial_case1])
            )

        idx_nontrivial_case2 = (
            (conformity_case_ratio >= 1) & (baseline_quenched_fraction > 0) )

        output_maximum_conformity[idx_nontrivial_case2] = 1. / baseline_quenched_fraction[idx_nontrivial_case2]

        return output_maximum_conformity


    def minimum_conformity_satellites(self,primary_halo_property,halo_type):
        """ The minimum allowed value of the inflection function, as pertains to satellites.

        The combinatorics of assembly-biased HODs are such that 
        the conformity function :math:`\\mathcal{C}_{cen_{Q}}(p | h_{0,1})` 
        must exceed both a and b.

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
        output_minimum_conformity : array_like
            Minimum allowed value of the conformity function, as pertains to satellites.


        """
        output_minimum_conformity = np.zeros(len(primary_halo_property))

        opposite_halo_type = np.zeros(len(halo_type))
        opposite_halo_type[halo_type==0] = 1

        inflection = self.inflection_satellites(primary_halo_property,halo_type)
        halo_type_fraction = self.halo_type_fraction_satellites(primary_halo_property,halo_type)

        opposite_inflection = self.inflection_satellites(primary_halo_property,opposite_halo_type)
        opposite_halo_type_fraction = self.halo_type_fraction_satellites(primary_halo_property,opposite_halo_type)
        opposite_maximum_conformity = self.maximum_conformity_satellites(primary_halo_property,opposite_halo_type)

        idx_nontrivial_case = ( (inflection > 0) & (halo_type_fraction > 0) )

        output_minimum_conformity[idx_nontrivial_case] = (
            (1. - (opposite_halo_type_fraction[idx_nontrivial_case]*
                opposite_maximum_conformity[idx_nontrivial_case])) / (
            inflection[idx_nontrivial_case]*halo_type_fraction[idx_nontrivial_case])
                )

        return output_minimum_conformity


    def conformity_satellites(self,primary_halo_property,halo_type):
        """
        Conformity function as pertains to satellites.

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
        output_conformity : array_like 
            Array giving the multiple by which the mean quenched fraction is boosted.

        """

        idx0 = np.where(halo_type == 0)[0]
        idx1 = np.where(halo_type == 1)[0]

        # Initialize array containing result to return
        output_conformity = np.zeros(len(primary_halo_property))

        all_ones = np.ones(len(primary_halo_property))

        # Start by ignoring the input halo_type, and  
        # assuming the halo_type = 1 for all inputs.
        # This is convenient and costs nothing, 
        # since the halo_type = 0 branch 
        # is defined in terms of the halo_type = 1 branch.
        output_conformity = (
            self.unconstrained_satellite_conformity_halo_type1(
                primary_halo_property))
        ########################################
        # Now apply the baseline HOD constraints to output_conformity, 
        # still behaving as if every input halo has halo_type=1
        maximum = self.maximum_conformity_satellites(primary_halo_property,all_ones)
        test_exceeds_maximum = output_conformity > maximum
        output_conformity[test_exceeds_maximum] = maximum[test_exceeds_maximum]
        # Next, require that the conformity function never falls below 
        # its minimum allowed value. This guarantees that ... 
        minimum = self.minimum_conformity_satellites(primary_halo_property,all_ones)
        test_below_minimum = output_conformity < minimum
        output_conformity[test_below_minimum] = minimum[test_below_minimum]
        # Finally, require that the conformity function is set to unity 
        # whenever the probability of halo_type=1 equals unity
        # This requirement supercedes the previous two, and ensures that 
        # the conformity inflection in h1-halos will be ignored in cases 
        # where there are no h0-halos. This self-consistency condition is necessary because 
        # the unconstrained conformity function and the halo_type function 
        # are both independently specified by user-supplied subclasses.  
        ###
        probability_type1 = self.halo_type_fraction_satellites(
            primary_halo_property,all_ones)
        idx_trivial_probability = (probability_type1 == 0) | (probability_type1 == 1)

        baseline_quenched_fraction = (
            self.baseline_hod_model.mean_quenched_fraction_satellites(
            primary_halo_property,all_ones)
            )
        idx_trivial_quenching = (baseline_quenched_fraction == 1)

        output_conformity[idx_trivial_probability] = 1
        output_conformity[idx_trivial_quenching] = 1

        ########################################
        # At this point, output_conformity has been properly conditioned. 
        # However, we have been assuming that all input halo_type = 1.
        # We now need to compute the correct output 
        # for cases where input halo_type = 0.
        # Define some shorthands (bookkeeping convenience)
        output_conformity_input_halo_type0 = output_conformity[idx0]
        primary_halo_property_input_halo_type0 = primary_halo_property[idx0]
        probability_type1_input_halo_type0 = probability_type1[idx0]
        probability_type0_input_halo_type0 = 1.0 - probability_type1_input_halo_type0
        # Whenever the fraction of halos of type=0 is zero, the conformity function 
        # for type0 halos should be set to unity.
        test_zero = (probability_type0_input_halo_type0 == 0)
        output_conformity_input_halo_type0[test_zero] = 1

        # For non-trivial cases, define the type0 conformity function 
        # in terms of the type1 conformity function in such a way that 
        # the baseline HOD will be unadulterated by assembly bias
        test_positive = (probability_type0_input_halo_type0 > 0)
        output_conformity_input_halo_type0[test_positive] = (
            (1.0 - output_conformity_input_halo_type0[test_positive]*
                probability_type1_input_halo_type0[test_positive])/
            probability_type0_input_halo_type0[test_positive])

        # Now write the results back to the output 
        output_conformity[idx0] = output_conformity_input_halo_type0

        return output_conformity


    def mean_quenched_fraction_centrals(self,primary_halo_property,halo_type):
        """ Override the baseline HOD method used to compute central quenched fraction. 

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
        quenched_fraction : array_like
            :math:`h_{i}`-conditioned central quenched fraction as a function of the primary halo property :math:`p`.

        :math:`F_{Q}^{cen}(p | h_{i}) = \\mathcal{C}_{cen}(p | h_{i})F_{Q}^{cen}(p)`

        """
        return self.conformity_centrals(primary_halo_property,halo_type)*(
            self.baseline_hod_model.mean_quenched_fraction_centrals(primary_halo_property,halo_type))

    def mean_quenched_fraction_satellites(self,primary_halo_property,halo_type):
        """ Override the baseline HOD method used to compute satellite quenched fraction. 

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
        quenched_fraction : array_like
            :math:`h_{i}`-conditioned central quenched fraction as a function of the primary halo property :math:`p`.

        :math:`F_{Q}^{sat}(p | h_{i}) = \\mathcal{C}_{sat}(p | h_{i})F_{Q}^{sat}(p)`

        """
        return self.conformity_satellites(primary_halo_property,halo_type)*(
            self.baseline_hod_model.mean_quenched_fraction_satellites(primary_halo_property,halo_type))


class Polynomial_Assembias_HOD_Quenching_Model(Assembias_HOD_Quenching_Model):
    """ Concrete subclass of `Assembias_HOD_Model`. 

    In this model, both galaxy abundances and quenched fractions exhibit assembly bias, 
    where some secondary host halo property modulates the mean galaxy abundance. 
    The assembly bias effect is set by explicitly specifing its strength 
    at specific values of the primary halo property. 

    """

    def __init__(self,
        baseline_hod_quenching_model=vdB03_Quenching_Model,
        baseline_hod_quenching_parameter_dict=None,
        baseline_hod_model=Zheng07_HOD_Model,
        baseline_hod_parameter_dict=None,
        threshold=model_defaults.default_luminosity_threshold,
        occupation_assembias_parameter_dict=None,quenching_assembias_parameter_dict=None,
        secondary_halo_property_centrals_key=model_defaults.default_assembias_key,
        secondary_halo_property_satellites_key=model_defaults.default_assembias_key):

        baseline_hod_quenching_model_instance = (
            baseline_hod_quenching_model(baseline_hod_model=baseline_hod_model,
                threshold=threshold,baseline_hod_parameter_dict=baseline_hod_parameter_dict)
            )
        if not isinstance(baseline_hod_quenching_model_instance,HOD_Quenching_Model):
            raise TypeError(
                "Input baseline_hod_quenching_model must be one of "
                "the supported HOD_Quenching_Model objects defined in this module or by the user")
        # Temporarily store the baseline HOD quenching model object
        # into a "private" attribute. This is a clunky workaround
        # to python's awkward conventions for required abstract properties
        self._baseline_hod_model = baseline_hod_quenching_model_instance
        self._secondary_halo_property_centrals_key = secondary_halo_property_centrals_key
        self._secondary_halo_property_satellites_key = secondary_halo_property_satellites_key

        # Executing the __init__ of the abstract base class Assembias_HOD_Quenching_Model  
        # does nothing besides executing the __init__ of the abstract base class Assembias_HOD_Model, 
        # which in turn does nothing besides execute the __init__ of the abstract base class HOD_Model. 
        # Executing the __init__ of the abstract base class HOD_Model 
        # sets self.parameter_dict to None, self.threshold to None, 
        # and self.publication to []        
        Assembias_HOD_Quenching_Model.__init__(self)


        # The following line should probably instead be self.threshold = self._baseline_hod_model.threshold
        # Leave as is for now, but when I fix this, fix it everywhere
        self.threshold = threshold 
        self.publication.extend(self._baseline_hod_model.publication)
        self.baseline_hod_parameter_dict = self._baseline_hod_model.parameter_dict


        if occupation_assembias_parameter_dict == None:
            self.occupation_assembias_parameter_dict = model_defaults.default_occupation_assembias_parameters
        else:
            self.occupation_assembias_parameter_dict = occupation_assembias_parameter_dict

        if quenching_assembias_parameter_dict == None:
            self.quenching_assembias_parameter_dict = model_defaults.default_quenching_assembias_parameters
        else:
            self.quenching_assembias_parameter_dict = quenching_assembias_parameter_dict

        # combine occupation assembias parameters and quenching assembias parameters
        # into the same dictionary
        self.assembias_parameter_dict = dict(
            self.occupation_assembias_parameter_dict.items() + 
            self.quenching_assembias_parameter_dict.items()
            )
            # require that the set of keys is correct
        self.require_correct_keys(self.assembias_parameter_dict)

        self.parameter_dict = dict(
            self.baseline_hod_parameter_dict.items() + 
            self.assembias_parameter_dict.items()
            )


# Note that this is only checking correct keys for the assembly bias parameter dictionary
    def require_correct_keys(self,assembias_parameter_dict):
        """ If the init constructor is passed an input parameter dictionary, 
        verify that the keys are correct."""

        correct_set_of_occupation_keys = model_defaults.default_occupation_assembias_parameters.keys()
        correct_set_of_quenching_keys = model_defaults.default_quenching_assembias_parameters.keys()
        correct_set_of_keys = correct_set_of_occupation_keys + correct_set_of_quenching_keys
        if set(assembias_parameter_dict.keys()) != set(correct_set_of_keys):
            raise TypeError("Set of keys of input assembias_parameter_dict"
            " does not match the set of keys required by the model." 
            " Correct set of keys is {'assembias_abcissa', "
            "'satellite_assembias_ordinates', 'central_assembias_ordinates',"
            "'quenching_assembias_abcissa', "
            "'satellite_assembias_ordinates', 'central_assembias_ordinates'} ")
        


    @property
    def baseline_hod_model(self):
        return self._baseline_hod_model

    @property 
    def primary_halo_property_key(self):
        return self.baseline_hod_model.primary_halo_property_key

    @property 
    def secondary_halo_property_centrals_key(self):
        return self._secondary_halo_property_centrals_key

    @property 
    def secondary_halo_property_satellites_key(self):
        return self._secondary_halo_property_satellites_key

    def mean_concentration(self,primary_halo_property,halo_type):
        """ Concentration-halo relation assumed by the underlying HOD_Model object.
        The appropriate method is already bound to the self.baseline_hod_model object.

        Parameters 
        ----------
        primary_halo_property : array_like
            array of primary halo property governing the occupation statistics 

        halo_type : array 
            array of halo types. 

        Returns 
        -------
        concentrations : numpy array

        """

        concentrations = (
            self.baseline_hod_model.mean_concentration(
                primary_halo_property,halo_type))
        return concentrations


##################################################
########## The following lines of code are copied-and-pasted from 
########## the Polynomial_Assembias_HOD_Model. This is bad practice.
########## Figure out some new design so that both classes 
########## can call the same methods, to avoid duplication.
    def halo_type1_fraction_centrals(self,primary_halo_property):
        """ Determines the fractional representation of host halo 
        types as a function of the value of the primary halo property.

        Halo types can be either given by fixed-Mvir rank-orderings 
        of the host halos, or by the input occupation statistics functions.

        """
        # In this model, centrals exhibit no assembly bias
        # So simply set the halo type1 fraction to unity for centrals
        abcissa = model_defaults.default_halo_type_split['halo_type_split_abcissa']
        ordinates = model_defaults.default_halo_type_split['halo_type_split_ordinates']
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

        abcissa = model_defaults.default_halo_type_split['halo_type_split_abcissa']
        ordinates = model_defaults.default_halo_type_split['halo_type_split_ordinates']
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
        output_unconstrained_inflection_function = (
            np.zeros(len(primary_halo_property)))

        # Use coefficients to compute values of the inflection function polynomial
        for n,coeff in enumerate(coefficient_array):
            output_unconstrained_inflection_function += coeff*primary_halo_property**n

        return output_unconstrained_inflection_function

    def unconstrained_central_inflection_halo_type1(self,primary_halo_property):
        abcissa = self.parameter_dict['assembias_abcissa']
        ordinates = self.parameter_dict['central_assembias_ordinates']

        return self.unconstrained_polynomial_model(abcissa,ordinates,primary_halo_property)

    def unconstrained_satellite_inflection_halo_type1(self,primary_halo_property):
        abcissa = self.parameter_dict['assembias_abcissa']
        ordinates = self.parameter_dict['satellite_assembias_ordinates']

        return self.unconstrained_polynomial_model(abcissa,ordinates,primary_halo_property)

##################################################


    def unconstrained_central_conformity_halo_type1(self,primary_halo_property):
        """ Method determining :math:`\\tilde{\\mathcal{C}}_{cen_{Q}}(p | h_{1})`, 
        the unconstrained excess quenched fraction of centrals 
        in halos of primary property :math:`p` and 
        secondary property type :math:`h_{1}`.
        """
        abcissa = self.parameter_dict['quenching_assembias_abcissa']
        ordinates = self.parameter_dict['central_quenching_assembias_ordinates']

        return self.unconstrained_polynomial_model(abcissa,ordinates,primary_halo_property)

    def unconstrained_satellite_conformity_halo_type1(self,primary_halo_property):
        """ Method determining :math:`\\tilde{\\mathcal{C}}_{cen_{Q}}(p | h_{1})`, 
        the unconstrained excess quenched fraction of satellites 
        in halos of primary property :math:`p` and 
        secondary property type :math:`h_{1}`.
        """
        abcissa = self.parameter_dict['quenching_assembias_abcissa']
        ordinates = self.parameter_dict['satellite_quenching_assembias_ordinates']

        return self.unconstrained_polynomial_model(abcissa,ordinates,primary_halo_property)











