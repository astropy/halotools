# -*- coding: utf-8 -*-
"""

This module contains functions and classes 
providing mappings between halos and the abundance and properties of 
galaxies residing in those halos. The classes serve primarily 
as components used by `halotools.hod_factory` and 
`halotools.hod_designer`, which act together to compose 
the behavior of the components into composite models. 
"""

__all__ = ['Kravtsov04Cens','Kravtsov04Sats','vdB03Quiescence']


import numpy as np
from scipy.special import erf 
from scipy.stats import poisson
from scipy.optimize import brentq
from scipy.interpolate import UnivariateSpline as spline

import model_defaults
from ..utils.array_utils import array_like_length as aph_len
import occupation_helpers as occuhelp

from astropy.extern import six
from abc import ABCMeta, abstractmethod, abstractproperty
import warnings

@six.add_metaclass(ABCMeta)
class OccupationComponent(object):
    """ Abstract super class of any occupation model. 
    Functionality is mostly trivial. 
    The sole function of the super class is to 
    standardize the attributes and methods 
    required of any occupation model component. 
    """
    def __init__(self, gal_type, haloprop_key_dict, 
        threshold, occupation_bound):
        self.gal_type = gal_type

        occuhelp.enforce_required_haloprops(haloprop_key_dict)
        self.haloprop_key_dict = haloprop_key_dict

        self.num_haloprops = occuhelp.count_haloprops(self.haloprop_key_dict)
        if self.num_haloprops > 2:
            raise SyntaxError("An OccupationComponent class instance can "
                "use only one or two halo properties, "
                "received %i" % self.num_haloprops)
        
        self.threshold = threshold
        self.occupation_bound = occupation_bound

        self._set_primary_function_dict()
        
    @abstractmethod
    def _get_param_dict(self):
        pass

    @abstractmethod
    def mc_occupation(self):
        pass

    def _set_primary_function_dict(self):
        self.prim_func_dict = {None : self.mc_occupation}
        self.additional_methods_to_inherit = [self.mean_occupation]

    def retrieve_haloprops(self, *args, **kwargs):

        if 'halos' in kwargs.keys():
            if self.num_haloprops==1:
                return kwargs['halos'][self.haloprop_key_dict['prim_haloprop_key']]
            else:
                return (
                    kwargs['halos'][self.haloprop_key_dict['prim_haloprop_key']],
                    kwargs['halos'][self.haloprop_key_dict['sec_haloprop_key']] 
                    )
        else:
            if self.num_haloprops==1:
                return args[0]
            else:
                return args[0], args[1]




class Kravtsov04Cens(OccupationComponent):
    """ Erf function model for the occupation statistics of central galaxies, 
    introduced in Kravtsov et al. 2004, arXiv:0308519.

    """

    def __init__(self,input_param_dict=None,
        haloprop_key_dict=model_defaults.haloprop_key_dict,
        threshold=model_defaults.default_luminosity_threshold,
        gal_type='centrals'):
        """
        Parameters 
        ----------
        input_param_dict : dictionary, optional.
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
            Default value is specified in the `~halotools.model_defaults` module.

        gal_type : string, optional
            Sets the key value used by `~halotools.hod_designer` and 
            `~halotools.hod_factory` to access the behavior of the methods 
            of this class. 

        """

        occupation_bound = 1.0
        # Call the super class constructor, which binds all the 
        # arguments to the instance.  
        OccupationComponent.__init__(self, gal_type, haloprop_key_dict, 
            threshold, occupation_bound)

        self.param_dict = self._get_param_dict(input_param_dict)

        self.publications = []


    def _get_param_dict(self, input_param_dict):

        self.logMmin_key = 'logMmin_'+self.gal_type
        self.sigma_logM_key = 'sigma_logM_'+self.gal_type

        correct_keys = [self.logMmin_key, self.sigma_logM_key]
        if input_param_dict != None:
            occuhelp.test_correct_keys(input_param_dict, correct_keys)
            output_param_dict = input_param_dict
        else:
            output_param_dict = self.get_published_parameters(self.threshold)

        return output_param_dict


    def mean_occupation(self, *args, **kwargs):
        """ Expected number of central galaxies in a halo of mass logM.
        See Equation 2 of arXiv:0703457.

        Parameters
        ----------        
        logM : array, optional
            array of :math:`log_{10}(M)` of halos in catalog

        halos : table, optional

        input_param_dict : dict, optional

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
        if 'input_param_dict' not in kwargs.keys():
            param_dict = self.param_dict 
        else:
            param_dict = kwargs['input_param_dict']

        logM = np.log10(self.retrieve_haloprops(*args, **kwargs))

        mean_ncen = 0.5*(1.0 + erf(
            (logM - param_dict[self.logMmin_key])
            /param_dict[self.sigma_logM_key]))

        return mean_ncen

    def mc_occupation(self, *args, **kwargs):
        """ Method to generate Monte Carlo realizations of the abundance of galaxies. 

        Parameters
        ----------        
        logM : array 
            array of :math:`log_{10}(M)` of halos in catalog

        Returns
        -------
        mc_abundance : array
            array of length len(logM) giving the number of self.gal_type galaxies in the halos. 
    
        """
        if 'input_param_dict' not in kwargs.keys():
            param_dict = self.param_dict 
        else:
            param_dict = kwargs['input_param_dict']

        halo_mass = self.retrieve_haloprops(*args, **kwargs)

        mc_generator = np.random.random(aph_len(halo_mass))
        mc_abundance = np.where(mc_generator < self.mean_occupation(halo_mass, 
            input_param_dict = param_dict), 1, 0)

        return mc_abundance


    def get_published_parameters(self,threshold):
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
        param_dict : dict
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
            param_dict = {
            self.logMmin_key : logMmin_array[threshold_index[0]],
            self.sigma_logM_key : sigma_logM_array[threshold_index[0]]
            }
        else:
            raise ValueError("Input luminosity threshold "
                "does not match any of the Table 1 values of "
                "Zheng et al. 2007 (arXiv:0703457)")

        return param_dict


class Kravtsov04Sats(OccupationComponent):
    """ Power law model for the occupation statistics of satellite galaxies, 
    introduced in Kravtsov et al. 2004, arXiv:0308519.

    """

    def __init__(self,input_param_dict=None,
        haloprop_key_dict=model_defaults.haloprop_key_dict,
        threshold=model_defaults.default_luminosity_threshold,
        gal_type='satellites',
        central_occupation_model=None, 
        input_central_param_dict=None):
        """
        Parameters 
        ----------
        param_dict : dictionary, optional.
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
            Default value is specified in the `~halotools.model_defaults` module.

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

        occupation_bound = float("inf")
        # Call the super class constructor, which binds all the 
        # arguments to the instance.  
        OccupationComponent.__init__(self, gal_type, haloprop_key_dict, 
            threshold, occupation_bound)

        self.param_dict = self._get_param_dict(input_param_dict)

        self._set_central_behavior(
            central_occupation_model, input_central_param_dict)

        self.publications = []

    def _get_param_dict(self, input_param_dict):

        # set attribute names for the keys so that the methods know 
        # how to evaluate their functions
        self.logM0_key = 'logM0_'+self.gal_type
        self.logM1_key = 'logM1_'+self.gal_type
        self.alpha_key = 'alpha_'+self.gal_type

        correct_keys = [self.logM0_key, self.logM1_key, self.alpha_key]
        if input_param_dict != None:
            occuhelp.test_correct_keys(input_param_dict, correct_keys)
            output_param_dict = input_param_dict
        else:
            output_param_dict = self.get_published_parameters(self.threshold)

        return output_param_dict


    def _set_central_behavior(self, 
        central_occupation_model, input_central_param_dict):

        self.central_occupation_model = central_occupation_model
        self.central_param_dict = input_central_param_dict
        
        if self.central_occupation_model is not None:
            # Test thresholds of centrals and satellites are equal
            if self.threshold != self.central_occupation_model.threshold:
                warnings.warn("Satellite and Central luminosity tresholds do not match")
            #
            self.central_param_dict = (
                self.central_occupation_model._get_param_dict(
                    input_central_param_dict)
                )

    def mean_occupation(self, *args, **kwargs):
        """Expected number of satellite galaxies in a halo of mass logM.
        See Equation 5 of arXiv:0703457.

        Parameters
        ----------
        logM : array 
            array of :math:`log_{10}(M)` of halos in catalog

        input_param_dict : dict, optional

        input_central_param_dict : dict, optional

        Returns
        -------
        mean_nsat : float or array
            Mean number of satellite galaxies in a host halo of the specified mass. 

        :math:`\\langle N_{sat} \\rangle_{M} = \left( \\frac{M - M_{0}}{M_{1}} \\right)^{\\alpha} \\langle N_{cen} \\rangle_{M}`


        """
        if 'input_param_dict' not in kwargs.keys():
            param_dict = self.param_dict 
        else:
            param_dict = kwargs['input_param_dict']

        if 'input_central_param_dict' not in kwargs.keys():
            central_param_dict = self.central_param_dict 
        else:
            central_param_dict = kwargs['input_central_param_dict']

        halo_mass = self.retrieve_haloprops(*args, **kwargs)
        logM = np.log10(halo_mass)

        M0 = 10.**param_dict[self.logM0_key]
        M1 = 10.**param_dict[self.logM1_key]

        # Call to np.where raises a harmless RuntimeWarning exception if 
        # there are entries of input logM for which mean_nsat = 0
        # Evaluating mean_nsat using the catch_warnings context manager 
        # suppresses this warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # Simultaneously evaluate mean_nsat and impose the usual cutoff
            mean_nsat = np.where(halo_mass - M0 > 0, 
                ((halo_mass - M0)/M1)**param_dict[self.alpha_key], 0)

        # If a central occupation model was passed to the constructor, 
        # multiply mean_nsat by an overall factor of mean_ncen
        if self.central_occupation_model is not None:
            mean_ncen = self.central_occupation_model.mean_occupation(
                logM, input_param_dict=central_param_dict)
            mean_nsat = np.where(mean_nsat > 0, mean_nsat*mean_ncen, mean_nsat)

        return mean_nsat


    def mc_occupation(self, *args, **kwargs):
        """ Method to generate Monte Carlo realizations of the abundance of galaxies. 
        Assumes gal_type galaxies obey Poisson statistics. 

        Parameters
        ----------        
        logM : array 
            array of :math:`log_{10}(M)` of halos in catalog

        input_param_dict : dict, optional

        input_central_param_dict : dict, optional

        Returns
        -------
        mc_abundance : array
            array of length len(logM) giving the number of self.gal_type galaxies in the halos. 
    
        """

        if 'input_param_dict' not in kwargs.keys():
            param_dict = self.param_dict 
        else:
            param_dict = kwargs['input_param_dict']

        if 'input_central_param_dict' not in kwargs.keys():
            central_param_dict = self.central_param_dict 
        else:
            central_param_dict = kwargs['input_central_param_dict']

        halo_mass = self.retrieve_haloprops(*args, **kwargs)
        logM = np.log10(halo_mass)

        expectation_values = self.mean_occupation(halo_mass, 
            input_param_dict=param_dict, 
            input_central_param_dict=central_param_dict)

        # The scipy built-in Poisson number generator raises an exception 
        # if its input is zero, so here we impose a simple workaround
        expectation_values = np.where(expectation_values <=0, 
            model_defaults.default_tiny_poisson_fluctuation, expectation_values)

        mc_abundance = poisson.rvs(expectation_values)

        return mc_abundance

    def get_published_parameters(self,threshold):
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

        param_dict : dict
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
            param_dict = {
            self.logM0_key : logM0_array[threshold_index[0]],
            self.logM1_key : logM1_array[threshold_index[0]],
            self.alpha_key : alpha_array[threshold_index[0]]
            }
        else:
            raise ValueError("Input luminosity threshold "
                "does not match any of the Table 1 values of Zheng et al. 2007 (arXiv:0703457).")

        return param_dict


class vdB03Quiescence(object):
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

    def __init__(self, gal_type, param_dict=model_defaults.default_quiescence_dict, 
        interpol_method='spline',input_spline_degree=3):
        """ 
        Parameters 
        ----------
        gal_type : string, optional
            Sets the key value used by `halotools.hod_designer` and 
            `~halotools.hod_factory` to access the behavior of the methods 
            of this class. 

        param_dict : dictionary, optional 
            Dictionary specifying what the quiescent fraction should be 
            at a set of input values of the primary halo property. 
            Default values are set in `halotools.model_defaults`. 

        interpol_method : string, optional 
            Keyword specifying how `mean_quiescence_fraction` 
            evaluates input value of the primary halo property 
            that differ from the small number of values 
            in self.param_dict. 
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

        self.param_dict = param_dict
        # Put param_dict keys in standard form
        correct_keys = model_defaults.default_quiescence_dict.keys()
        self.param_dict = occuhelp.format_parameter_keys(
            self.param_dict,correct_keys,self.gal_type)
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
                aph_len(self.param_dict[self.abcissa_key])-1])
            self.spline_function = occuhelp.aph_spline(
                self.param_dict[self.abcissa_key],
                self.param_dict[self.ordinates_key],
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
        its values at the model abcissa, as specified in param_dict. 
        """

        model_abcissa = self.param_dict[self.abcissa_key]
        model_ordinates = self.param_dict[self.ordinates_key]

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
























