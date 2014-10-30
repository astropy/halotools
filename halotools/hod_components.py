# -*- coding: utf-8 -*-
"""

This module contains the components for occupation statistics 
used by `halotools.hod_designer` to build composite HOD models 
by composing the behavior of the components. 

"""

__all__ = ['Zheng07_Centrals','Zheng07_Satellites','vdB03_Quiescence']


import numpy as np
from scipy.special import erf 
from scipy.stats import poisson
from scipy.optimize import brentq
from scipy.interpolate import UnivariateSpline as spline

import defaults
from utils.array_utils import array_like_length as aph_len
import occupation_helpers as occuhelp

from astropy.extern import six
from abc import ABCMeta, abstractmethod, abstractproperty
import warnings



class Zheng07_Centrals(object):
    """ Model for the occupation statistics of central galaxies, 
    taken from Zheng et al. 2007, arXiv:0703457.


    Parameters 
    ----------
    parameter_dict : dictionary, optional.
        Contains values for the parameters specifying the model.
        Dictionary keys are 'logMmin_cen' and 'sigma_logM'

        Their best-fit parameter values provided in Table 1 of 
        Zheng et al. (2007) are pre-loaded into this class, and 
        can be accessed via the `published_parameters` method.

    threshold : float, optional.
        Luminosity threshold of the mock galaxy sample. 
        If specified, input value must agree with 
        one of the thresholds used in Zheng07 to fit HODs: 
        [-18, -18.5, -19, -19.5, -20, -20.5, -21, -21.5, -22].
        Default value is specified in the `~halotools.defaults` module.

    gal_type : string, optional
        Sets the key value used by `~halotools.hod_designer` and 
        `~halotools.hod_factory` to access the behavior of the methods 
        of this class. 

    """

    def __init__(self,parameter_dict=None,
        threshold=defaults.default_luminosity_threshold,
        gal_type='centrals'):

        self.gal_type = gal_type

        self.threshold = threshold
        if parameter_dict is None:
            self.parameter_dict = self.published_parameters(self.threshold)
        else:
            self.parameter_dict = parameter_dict
        # Put parameter_dict keys in standard form
        correct_keys = self.published_parameters(self.threshold).keys()
        self.parameter_dict = occuhelp.format_parameter_keys(
            self.parameter_dict,correct_keys,self.gal_type)
        # get the new keys so that the methods know 
        # how to evaluate their functions
        self.logMmin_key = 'logMmin_'+self.gal_type
        self.sigma_logM_key = 'sigma_logM_'+self.gal_type


    def mean_occupation(self,logM):
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
            (logM - self.parameter_dict[self.logMmin_key])
            /self.parameter_dict[self.sigma_logM_key]))

        return mean_ncen


    def published_parameters(self,threshold):
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
        logMmin_array = [11.35,11.46,11.6,11.75,12.02,12.3,12.79,13.38,14.22]
        sigma_logM_array = [0.25,0.24,0.26,0.28,0.26,0.21,0.39,0.51,0.77]
        # define the luminosity thresholds corresponding to the above data
        threshold_array = np.arange(-22,-17.5,0.5)
        threshold_array = threshold_array[::-1]

        threshold_index = np.where(threshold_array==threshold)[0]
        if len(threshold_index)==1:
            parameter_dict = {
            'logMmin' : logMmin_array[threshold_index[0]],
            'sigma_logM' : sigma_logM_array[threshold_index[0]]
            }
        else:
            raise ValueError("Input luminosity threshold "
                "does not match any of the Table 1 values of Zheng et al. 2007 (arXiv:0703457).")

        return parameter_dict


class Zheng07_Satellites(object):
    """ Model for the occupation statistics of central galaxies, 
    taken from Zheng et al. 2007, arXiv:0703457.


    Parameters 
    ----------
    parameter_dict : dictionary, optional.
        Contains values for the parameters specifying the model.
        Dictionary keys are 'logMmin_cen' and 'sigma_logM'

        Their best-fit parameter values provided in Table 1 of 
        Zheng et al. (2007) are pre-loaded into this class, and 
        can be accessed via the `published_parameters` method.

    threshold : float, optional.
        Luminosity threshold of the mock galaxy sample. 
        If specified, input value must agree with 
        one of the thresholds used in Zheng07 to fit HODs: 
        [-18, -18.5, -19, -19.5, -20, -20.5, -21, -21.5, -22].
        Default value is specified in the `~halotools.defaults` module.

    gal_type : string, optional
        Sets the key value used by `~halotools.hod_designer` and 
        `~halotools.hod_factory` to access the behavior of the methods 
        of this class. 

    central_occupation_model : occupation model instance, optional
        If present, the mean occupation method of this model will 
        be multiplied by the value of central_occupation_model at each mass, 
        as in Zheng et al. 2007.
        Default is None.

    """

    def __init__(self,parameter_dict=None,
        threshold=defaults.default_luminosity_threshold,
        gal_type='satellites',
        central_occupation_model=None):

        self.gal_type = gal_type
        self.central_occupation_model = central_occupation_model
        if self.central_occupation_model is not None:
            if threshold != self.central_occupation_model.threshold:
                warnings.warn("Satellite and Central luminosity tresholds do not match")

        self.threshold = threshold
        if parameter_dict is None:
            self.parameter_dict = self.published_parameters(self.threshold)
        else:
            self.parameter_dict = parameter_dict
        # Put parameter_dict keys in standard form
        correct_keys = self.published_parameters(self.threshold).keys()
        self.parameter_dict = occuhelp.format_parameter_keys(
            self.parameter_dict,correct_keys,self.gal_type)
        # get the new keys so that the methods know 
        # how to evaluate their functions
        self.logM0_key = 'logM0_'+self.gal_type
        self.logM1_key = 'logM1_'+self.gal_type
        self.alpha_key = 'alpha_'+self.gal_type


    def mean_occupation(self,logM):
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
        logM = np.array(logM)
        halo_mass = 10.**logM


        M0 = 10.**self.parameter_dict[self.logM0_key]
        M1 = 10.**self.parameter_dict[self.logM1_key]
        mean_nsat = np.zeros(len(logM),dtype='f8')
        idx_nonzero_satellites = (halo_mass - M0) > 0


        mean_nsat[idx_nonzero_satellites] = (
            (((halo_mass[idx_nonzero_satellites] - M0)/M1)
            **self.parameter_dict[self.alpha_key]))

        # If a central occupation model was passed to the constructor, 
        # multiply the mean satellite occupation by the mean central occupation
        if self.central_occupation_model is not None:
            mean_ncen = self.central_occupation_model.mean_occupation(
                logM[idx_nonzero_satellites])
            mean_nsat[idx_nonzero_satellites] = mean_ncen*mean_nsat[idx_nonzero_satellites]

        return mean_nsat

    def published_parameters(self,threshold):
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
        logM0_array = [11.2,10.59,11.49,11.69,11.38,11.84,11.92,13.94,14.0]
        logM1_array = [12.4,12.68,12.83,13.01,13.31,13.58,13.94,13.91,14.69]
        alpha_array = [0.83,0.97,1.02,1.06,1.06,1.12,1.15,1.04,0.87]
        # define the luminosity thresholds corresponding to the above data
        threshold_array = np.arange(-22,-17.5,0.5)
        threshold_array = threshold_array[::-1]

        threshold_index = np.where(threshold_array==threshold)[0]
        if len(threshold_index)==1:
            parameter_dict = {
            'logM0' : logM0_array[threshold_index[0]],
            'logM1' : logM1_array[threshold_index[0]],
            'alpha' : alpha_array[threshold_index[0]]
            }
        else:
            raise ValueError("Input luminosity threshold "
                "does not match any of the Table 1 values of Zheng et al. 2007 (arXiv:0703457).")

        return parameter_dict



class vdB03_Quiescence(object):
    """
    Traditional HOD-style model of galaxy quenching 
    in which the expectation value for a binary SFR designation of the galaxy 
    is purely determined by the primary halo property.
    
    Approach is adapted from van den Bosch et al. 2003. 
    The parameters of this component model govern the value of the quiescent fraction 
    at a particular set of masses. 
    The class then uses an input `halotools.occupation_helpers` function 
    to infer the quiescent fraction at values other than the input abcissa.

    Notes 
    -----

    In the construction sequence of a composite HOD model, 
    if `halotools.hod_designer` uses this component model *after* 
    using a central occupation component, then  
    the resulting central galaxy stellar-to-halo mass relation 
    will have no dependence on quenched/active designation. 
    Employing this component *before* the occupation component allows 
    for an explicit halo mass dependence in the central galaxy SMHM. 
    Thus the sequence of the composition of the quiescence and occupation models 
    determines whether the resulting composite model satisfies the following 
    classical separability condition between stellar mass and star formation rate: 

    :math:`P( M_{*}, \dot{M_{*}} | M_{h}) = P( M_{*} | M_{h})\\times P( \dot{M_{*}} | M_{h})`

    """

    def __init__(self, gal_type, parameter_dict=defaults.default_quiescence_dict, 
        interpol_method='spline',input_spline_degree=3):
        """ 
        Parameters 
        ----------
        gal_type : string, optional
            Sets the key value used by `halotools.hod_designer` and 
            `~halotools.hod_factory` to access the behavior of the methods 
            of this class. 

        parameter_dict : dictionary, optional 
            Dictionary specifying what the quiescent fraction should be 
            at a set of input values of the primary halo property. 
            Default values are set in `halotools.defaults`. 

        interpol_method : string, optional 
            Keyword specifying how `mean_quiescence_fraction` 
            evaluates input value of the primary halo property 
             that differ from the small number of values 
            in self.parameter_dict. 
            The default spline option interpolates the 
            model's abcissa and ordinates. 
            The polynomial option uses the unique, degree N polynomial 
            passing through the ordinates, where N is the number of supplied ordinates. 

        input_spline_degree : int, optional
            Degree of the spline interpolation for the case of interpol_method='spline'. 
            If there are k abcissa values specifying the model, input_spline_degree 
            is ensured to never exceed k-1, nor exceed 5. 
        """

        self.gal_type = gal_type

        self.parameter_dict = parameter_dict
        # Put parameter_dict keys in standard form
        correct_keys = defaults.default_quiescence_dict.keys()
        self.parameter_dict = occuhelp.format_parameter_keys(
            self.parameter_dict,correct_keys,self.gal_type)
        self.abcissa_key = 'quiescence_abcissa_'+self.gal_type
        self.ordinates_key = 'quiescence_ordinates_'+self.gal_type

        # Set the interpolation scheme 
        if interpol_method not in ['spline', 'polynomial']:
            raise IOError("Input interpol_method must be 'polynomial' or 'spline'.")
        self.interpol_method = interpol_method

        if self.interpol_method=='spline':
            scipy_maxdegree = 5
            self.spline_degree = np.min(
                [scipy_maxdegree, input_spline_degree, 
                aph_len(self.parameter_dict[self.abcissa_key])-1])
            self.spline_function = spline(
                self.parameter_dict[self.abcissa_key],
                self.parameter_dict[self.ordinates_key],
                k=self.spline_degree)


    def mean_quiescence_fraction(self,input_abcissa):
        """
        Expected fraction of gal_type galaxies that are quiescent 
        as a function of the primary halo property.

        Parameters 
        ----------
        input_abcissa : array_like
            array of primary halo property at which the quiescent fraction 
            is being computed. 

        Returns 
        -------
        mean_quiescence_fraction : array_like
            Values of the quiescent fraction evaluated at input_abcissa. 

        Notes 
        -----

        Either assumes the quiescent fraction is a polynomial function 
        of the primary halo property, or is interpolated from a grid. 
        Either way, the behavior of this method is fully determined by 
        its values at the model abcissa, as specified in parameter_dict. 
        """

        model_abcissa = self.parameter_dict[self.abcissa_key]
        model_ordinates = self.parameter_dict[self.ordinates_key]

        if self.interpol_method=='polynomial':
            mean_quiescence_fraction = occuhelp.polynomial_from_table(
                model_abcissa,model_ordinates,input_abcissa)
        elif self.interpol_method=='spline':
            mean_quiescence_fraction = self.spline_function(input_abcissa)
        else:
            raise IOError("Input interpol_method must be 'polynomial' or 'spline'.")

        # Enforce boundary conditions of any Fraction function
        test_negative = np.array(mean_quiescence_fraction<0)
        test_exceeds_unity = np.array(mean_quiescence_fraction>1)
        mean_quiescence_fraction[test_negative]=0
        mean_quiescence_fraction[test_exceeds_unity]=1

        return mean_quiescence_fraction
























