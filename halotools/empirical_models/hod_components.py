# -*- coding: utf-8 -*-
"""
This module contains various component features used by 
HOD-style models of the galaxy-halo connection. For example, 
the `~halotools.empirical_models.Kravtsov04Cens` class 
governs the occupation statistics of a centrals-like population, 
and so has a ``mean_occupation`` method. 

A common use for these objects is to bundle them together to make a 
composite galaxy model, with multiple populations having their 
own occupation statistics and profiles. Instances of classes in this module 
can be passed to the `~halotools.empirical_models.hod_factory`, 
and you will be returned a model object that can directly populate 
simulations with mock galaxies. See the tutorials on these models 
for further details on their use. 
"""

__all__ = ['OccupationComponent','Kravtsov04Cens','Kravtsov04Sats']

from copy import copy
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
    The sole purpose of the super class is to 
    standardize the attributes and methods 
    required of any HOD-style occupation model component. 
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

    @abstractmethod
    def _set_param_dict(self):
        """ Builds the parameter dictionary, whose keys are names of MCMC parameters, 
        and values are used in `mc_occupation` and `mean_occupation`. 
        Dictionary qarameter names will have the `gal_type` string with a leading underscore. 
        This protects against the case where multiple populations might share some 
        component behavior. 
        """
        pass

    @abstractmethod
    def mc_occupation(self):
        """ Primary method used to generate Monte Carlo realizations 
        of an occupation model. 
        """
        pass

    @abstractmethod
    def mean_occupation(self):
        """ Method giving the first moment of the occupation distribution. 
        """
        pass

    def retrieve_haloprops(self, *args, **kwargs):
        """ Interface used to pass the correct numpy array to `mc_occupation`. 

        Many methods need to behave properly whether they are passed a numpy array, 
        or a data table. This method identifies what has been passed, and returns 
        the correct numpy array. 
        """

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
    """ ``Erf`` function model for the occupation statistics of central galaxies, 
    introduced in Kravtsov et al. 2004, arXiv:0308519.

    Parameters 
    ----------
    input_param_dict : dict, optional.
        Contains values for the parameters specifying the model.
        Dictionary keys should have names like 
        ``logMmin_centrals`` and ``sigma_logM_centrals``.

        If ``input_param_dict`` is not passed, 
        the best-fit parameter values provided in Table 1 of 
        Zheng et al. (2007) are chosen. 
        See the `get_published_parameters` method for details. 

    threshold : float, optional.
        Luminosity threshold of the mock galaxy sample. 
        If specified, input value must agree with 
        one of the thresholds used in Zheng07 to fit HODs: 
        [-18, -18.5, -19, -19.5, -20, -20.5, -21, -21.5, -22].
        Default value is specified in the `~halotools.empirical_models.model_defaults` module.

    gal_type : string, optional
        Name of the galaxy population being modeled, e.g., ``cens`. 
        Default setting is ``centrals``.  

    Notes 
    -----
    There can be one and only one central galaxy per halo, 
    so to compute :math:`\\langle N_{\mathrm{cen}}(M_{\mathrm{halo}}) \\rangle_{>L}` , 
    the mean number of centrals brighter than some luminosity residing 
    in a halo of some virial mass, we just need to integrate :math:`P( L | M_{\\mathrm{halo}})` , 
    the probability that a halo of a given mass hosts a central brighter than L

    :math:`\\langle N_{\\mathrm{cen}}( M_{\\rm halo} )\\rangle_{>L} = 
    \\int_{L}^{\\infty}\\mathrm{d}L'P( L' | M_{\mathrm{halo}})`

    The `Kravtsov04Cens` model assumes the stellar-to-halo-mass 
    PDF is log-normal, 
    in which case the mean occupation function is just an ``erf`` function, 
    as in the `mean_occupation` method. 

    The test suite for this model is documented at 
    `~halotools.empirical_models.test_empirical_models.test_Kravtsov04Cens`
    """

    def __init__(self,input_param_dict=None,
        haloprop_key_dict=model_defaults.haloprop_key_dict,
        threshold=model_defaults.default_luminosity_threshold,
        gal_type='centrals'):
        """

        """

        occupation_bound = 1.0
        # Call the super class constructor, which binds all the 
        # arguments to the instance.  
        OccupationComponent.__init__(self, gal_type, haloprop_key_dict, 
            threshold, occupation_bound)

        self._set_param_dict(input_param_dict)

        self.publications = []


    def _set_param_dict(self, input_param_dict):
        """ Private method used to retrieve the 
        dictionary governing the parameters of the model. 
        """

        self.logMmin_key = 'logMmin_'+self.gal_type
        self.sigma_logM_key = 'sigma_logM_'+self.gal_type

        correct_keys = [self.logMmin_key, self.sigma_logM_key]
        if input_param_dict is not None:
            occuhelp.test_correct_keys(input_param_dict, correct_keys)
            output_param_dict = input_param_dict
        else:
            output_param_dict = self.get_published_parameters(self.threshold)

        self.param_dict = output_param_dict


    def mean_occupation(self, *args, **kwargs):
        """ Expected number of central galaxies in a halo of mass halo_mass.
        See Equation 2 of arXiv:0703457.

        Parameters
        ----------        
        halo_mass : array, optional positional argument
            array of :math:`M_{\\mathrm{vir}}` of halos in catalog

        halos : object, optional keyword argument 
            Data table storing halo catalog. 

        input_param_dict : dict, optional
            dictionary of parameters governing the model. If not passed, 
            values bound to ``self`` will be chosen. 

        Returns
        -------
        mean_ncen : array
            Mean number of central galaxies in the halo of the input mass. 

        Notes 
        -----

        The `mean_occupation` method computes the following function: 

        :math:`\\langle N_{\\mathrm{cen}} \\rangle_{M} = 
        \\frac{1}{2}\\left( 1 + 
        \\mathrm{erf}\\left( \\frac{\\log_{10}M - 
        \\log_{10}M_{min}}{\\sigma_{\\log_{10}M}} \\right) \\right)`

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

        Assumes a nearest integer distribution for the central occupation function, 
        where the first moment is governed by `mean_occupation`, 
        and the per-halo occupations are bounded by unity. 

        Parameters
        ----------        
        halo_mass : array, optional positional argument
            array of :math:`M_{\\mathrm{vir}}` of halos in catalog

        halos : object, optional keyword argument 
            Data table storing halo catalog. 

        Returns
        -------
        mc_abundance : array
            array with same length as input *halo_mass*,  
            returning the number of central galaxies in each input halo. 
            Values will be either 0 or 1. 
    
        """
        if 'input_param_dict' not in kwargs.keys():
            param_dict = self.param_dict 
        else:
            param_dict = kwargs['input_param_dict']

        halo_mass = self.retrieve_haloprops(*args, **kwargs)

        if 'seed' in kwargs.keys():
            np.random.seed(seed=kwargs['seed'])
        else:
            np.random.seed(seed=None)

        mc_generator = np.random.random(aph_len(halo_mass))
        mc_abundance = np.where(mc_generator < self.mean_occupation(halo_mass, 
            input_param_dict = param_dict), 1, 0)

        return mc_abundance


    def get_published_parameters(self, threshold, publication='Zheng07'):
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

        def get_zheng07_params(threshold):
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

        if publication in ['zheng07', 'Zheng07', 'Zheng_etal07', 'zheng_etal07','zheng2007','Zheng2007']:
            param_dict = get_zheng07_params(threshold)
            return param_dict
        else:
            raise KeyError("For Kravtsov04Cens, only supported best-fit models are currently Zheng et al. 2007")


class Kravtsov04Sats(OccupationComponent):
    """ Power law model for the occupation statistics of satellite galaxies, 
    introduced in Kravtsov et al. 2004, arXiv:0308519.

    :math:`\\langle N_{sat} \\rangle_{M} = \left( \\frac{M - M_{0}}{M_{1}} \\right)^{\\alpha}`

    """

    def __init__(self,input_param_dict=None,
        haloprop_key_dict=model_defaults.haloprop_key_dict,
        threshold=model_defaults.default_luminosity_threshold,
        gal_type='satellites',
        central_occupation_model=None):
        """
        Parameters 
        ----------
        input_param_dict : dictionary, optional.
            Contains values for the parameters specifying the model.
            Dictionary keys are ``logM0_satellites``, ``logM1_satellites``
            and ``alpha_satellites``. 

            If no input_param_dict is passed, 
            the best-fit parameter values provided in Table 1 of 
            Zheng et al. (2007) are chosen.

        threshold : float, optional.
            Luminosity threshold of the mock galaxy sample. 
            If specified, input value must agree with 
            one of the thresholds used in Zheng07 to fit HODs: 
            [-18, -18.5, -19, -19.5, -20, -20.5, -21, -21.5, -22].
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        gal_type : string, optional
            Name of the galaxy population being modeled, e.g., ``sats``. 
            Default setting is ``satellites``. 

        central_occupation_model : occupation model instance, optional
            If using, must be an instance of a sub-class of `~halotools.empirical_models.OccupationComponent`. 
            If using, the mean occupation method of this model will 
            be multiplied by the value of central_occupation_model at each mass, 
            as in Zheng et al. 2007, so that 
            :math:`\\langle N_{\mathrm{sat}}|M\\rangle\\Rightarrow\\langle N_{\mathrm{sat}}|M\\rangle\\times\\langle N_{\mathrm{cen}}|M\\rangle`
        """

        occupation_bound = float("inf")
        # Call the super class constructor, which binds all the 
        # arguments to the instance.  
        OccupationComponent.__init__(self, gal_type, haloprop_key_dict, 
            threshold, occupation_bound)

        self._set_param_dict(input_param_dict)

        self._set_central_behavior(central_occupation_model)

        self.publications = []

    def _set_param_dict(self, input_param_dict):

        # set attribute names for the keys so that the methods know 
        # how to evaluate their functions
        self.logM0_key = 'logM0_'+self.gal_type
        self.logM1_key = 'logM1_'+self.gal_type
        self.alpha_key = 'alpha_'+self.gal_type

        correct_keys = [self.logM0_key, self.logM1_key, self.alpha_key]
        if input_param_dict is not None:
            occuhelp.test_correct_keys(input_param_dict, correct_keys)
            output_param_dict = input_param_dict
        else:
            output_param_dict = self.get_published_parameters(self.threshold)

        self.param_dict = output_param_dict

    def _set_central_behavior(self, central_occupation_model):

        self.central_occupation_model = central_occupation_model
        
        if self.central_occupation_model is not None:
            if not isinstance(self.central_occupation_model, OccupationComponent):
                msg = ("When passing a central_occupation_model to " + 
                    "the Kravtsov04Sats constructor, \n you must pass an instance of " + 
                    "an OccupationComponent.")
                if issubclass(self.central_occupation_model, OccupationComponent):
                    msg = (msg + 
                        "\n Instead, the Kravtsov04Sats received the actual class " + 
                        self.central_occupation_model.__name__+", " + 
                    "rather than an instance of that class. ")
                raise SyntaxError(msg)

            # Test thresholds of centrals and satellites are equal
            if self.threshold != self.central_occupation_model.threshold:
                warnings.warn("Satellite and Central luminosity tresholds do not match")
            #

    def mean_occupation(self, *args, **kwargs):
        """Expected number of satellite galaxies in a halo of mass logM.
        See Equation 5 of arXiv:0703457.

        Parameters
        ----------
        halo_mass : array, optional
            array of :math:`M_{\\mathrm{vir}}`-like variable of halos in catalog

        halos : object, optional keyword argument 
            Data table storing halo catalog. 

        input_param_dict : dict, optional
            dictionary of parameters governing the model. If not passed, 
            values bound to ``self`` will be chosen. 

        Returns
        -------
        mean_nsat : float or array
            Mean number of satellite galaxies in a host halo of the specified mass. 

        :math:`\\langle N_{\\mathrm{sat}} \\rangle_{M} = \left( \\frac{M - M_{0}}{M_{1}} \\right)^{\\alpha} \\langle N_{\\mathrm{cen}} \\rangle_{M}`

        or 

        :math:`\\langle N_{\\mathrm{sat}} \\rangle_{M} = \left( \\frac{M - M_{0}}{M_{1}} \\right)^{\\alpha}`, 

        depending on whether a central model was passed to the constructor. 

        """
        if 'input_param_dict' not in kwargs.keys():
            param_dict = self.param_dict 
        else:
            param_dict = kwargs['input_param_dict']

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
                *args, **kwargs)
            mean_nsat = np.where(mean_nsat > 0, mean_nsat*mean_ncen, mean_nsat)

        return mean_nsat


    def mc_occupation(self, *args, **kwargs):
        """ Method to generate Monte Carlo realizations of the abundance of galaxies. 
        Assumes gal_type galaxies obey Poisson statistics. 

        Parameters
        ----------        
        halo_mass : array, optional
            array of :math:`M_{\\mathrm{vir}}`-like variable of halos in catalog

        halos : object, optional keyword argument 
            Data table storing halo catalog. 

        input_param_dict : dict, optional
            dictionary of parameters governing the model. If not passed, 
            values bound to ``self`` will be chosen. 

        Returns
        -------
        mc_abundance : array
            array giving the number of satellite-type galaxies per input halo. 
    
        """

        if 'input_param_dict' not in kwargs.keys():
            param_dict = self.param_dict 
        else:
            param_dict = kwargs['input_param_dict']

        halo_mass = self.retrieve_haloprops(*args, **kwargs)
        logM = np.log10(halo_mass)

        expectation_values = self.mean_occupation(halo_mass, 
            input_param_dict=param_dict)

        # The scipy built-in Poisson number generator raises an exception 
        # if its input is zero, so here we impose a simple workaround
        expectation_values = np.where(expectation_values <=0, 
            model_defaults.default_tiny_poisson_fluctuation, expectation_values)

        if 'seed' in kwargs.keys():
            np.random.seed(seed=kwargs['seed'])
        else:
            np.random.seed(seed=None)

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
























