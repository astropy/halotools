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
can be passed to the `~halotools.empirical_models.model_factories.HodModelFactory`, 
and you will be returned a model object that can directly populate 
simulations with mock galaxies. See the tutorials on model-building 
for further details on their use. 
"""

__all__ = ['OccupationComponent','Kravtsov04Cens','Kravtsov04Sats', 'BinaryGalpropInterpolModel']

from functools import partial
from copy import copy
import numpy as np
from scipy.special import erf 
from scipy.stats import poisson
from scipy.optimize import brentq
from scipy.interpolate import UnivariateSpline as spline

import model_defaults
from ..utils.array_utils import array_like_length as custom_len
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
    def __init__(self, gal_type, threshold, occupation_bound, **kwargs):

        self.gal_type = gal_type
        self.threshold = threshold
        self.occupation_bound = occupation_bound

        if 'prim_haloprop_key' in kwargs.keys():
            self.prim_haloprop_key = kwargs['prim_haloprop_key']
        else:
            raise KeyError("All OccupationComponent sub-classes "
                "must pass a prim_haloprop_key to the constructor \n"
                "so that the mc_occupation and mean_occupation methods "
                "know how to interpret a halo catalog input")
        if 'sec_haloprop_key' in kwargs.keys():
            self.sec_haloprop_key = kwargs['sec_haloprop_key']

        if 'param_dict' in kwargs.keys():
            self.param_dict = kwargs['param_dict']
        else:
            self.param_dict = {}

    def mc_occupation(self, **kwargs):
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

        if 'galaxy_table' in kwargs.keys():
            mass = kwargs['galaxy_table'][self.prim_haloprop_key]
        elif 'halos' in kwargs.keys():
            mass = kwargs['halos'][self.prim_haloprop_key]
        elif 'mass' in kwargs.keys():
            mass = kwargs['mass']
        elif 'prim_haloprop' in kwargs.keys():
            mass = kwargs['prim_haloprop']
        else:
            raise KeyError("Must pass one of the following keyword arguments to mc_occupation:\n"
                "``halos``, ``mass``, ``prim_haloprop``, or ``galaxy_table``")
 
        if 'seed' in kwargs.keys():
            np.random.seed(seed=kwargs['seed'])
        else:
            np.random.seed(seed=None)

        if self.occupation_bound == 1:
            mc_generator = np.random.random(custom_len(mass))
            mc_abundance = np.where(mc_generator < self.mean_occupation(**kwargs), 1, 0)
            return mc_abundance

        elif self.occupation_bound == float("inf"):
            expectation_values = self.mean_occupation(**kwargs)
            # The scipy built-in Poisson number generator raises an exception 
            # if its input is zero, so here we impose a simple workaround
            expectation_values = np.where(expectation_values <=0, 
                model_defaults.default_tiny_poisson_fluctuation, expectation_values)

            mc_abundance = poisson.rvs(expectation_values)
            return mc_abundance
        else:
            raise KeyError("The only permissible values of occupation_bound for instances "
                "of OccupationComponent are unity and infinity")


    @abstractmethod
    def mean_occupation(self):
        """ Method giving the first moment of the occupation distribution. 
        """
        raise NotImplementedError("All subclasses of OccupationComponent " 
            "must implement a mean_occupation method. ")


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

    def __init__(self, **kwargs):
        """
        """
        occupation_bound = 1.0

        if 'gal_type' in kwargs.keys():
            gal_type = kwargs['gal_type']
        else:
            gal_type = 'centrals'

        if 'threshold' in kwargs.keys():
            threshold = kwargs['threshold']
        else:
            threshold = model_defaults.default_luminosity_threshold

        if 'prim_haloprop_key' in kwargs.keys():
            prim_haloprop_key = kwargs['prim_haloprop_key']
        else:
            prim_haloprop_key = model_defaults.prim_haloprop_key

        # Call the super class constructor, which binds all the 
        # arguments to the instance.  
        super(Kravtsov04Cens, self).__init__(
            gal_type, threshold, occupation_bound, 
            prim_haloprop_key = prim_haloprop_key)

        if 'input_param_dict' in kwargs.keys():
            input_param_dict = kwargs['input_param_dict']
        else:
            input_param_dict = None
        self._initialize_param_dict(input_param_dict)

        self.publications = []


    def _initialize_param_dict(self, input_param_dict):
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


    def mean_occupation(self, **kwargs):
        """ Expected number of central galaxies in a halo of mass halo_mass.
        See Equation 2 of arXiv:0703457.

        Parameters
        ----------        
        mass : array, optional keyword argument
            array of :math:`M_{\\mathrm{vir}}` of halos in catalog

        halos : object, optional keyword argument 
            Data table storing halo catalog. 

        galaxy_table : object, optional keyword argument 
            Data table storing mock galaxy catalog. 

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

        if 'galaxy_table' in kwargs.keys():
            mass = kwargs['galaxy_table'][self.prim_haloprop_key]
        elif 'halos' in kwargs.keys():
            mass = kwargs['halos'][self.prim_haloprop_key]
        elif 'mass' in kwargs.keys():
            mass = kwargs['mass']
        else:
            raise KeyError("Must pass one of the following keyword arguments to mean_occupation:\n"
                "``halos``, ``mass``, or ``galaxy_table``")

        logM = np.log10(mass)

        mean_ncen = 0.5*(1.0 + erf(
            (logM - param_dict[self.logMmin_key])
            /param_dict[self.sigma_logM_key]))

        return mean_ncen


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

    def __init__(self, **kwargs):
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

        if 'gal_type' in kwargs.keys():
            gal_type = kwargs['gal_type']
        else:
            gal_type = 'satellites'

        if 'threshold' in kwargs.keys():
            threshold = kwargs['threshold']
        else:
            threshold = model_defaults.default_luminosity_threshold

        if 'prim_haloprop_key' in kwargs.keys():
            prim_haloprop_key = kwargs['prim_haloprop_key']
        else:
            prim_haloprop_key = model_defaults.prim_haloprop_key

        # Call the super class constructor, which binds all the 
        # arguments to the instance.  
        super(Kravtsov04Sats, self).__init__(
            gal_type, threshold, occupation_bound, 
            prim_haloprop_key = prim_haloprop_key)

        if 'input_param_dict' in kwargs.keys():
            input_param_dict = kwargs['input_param_dict']
        else:
            input_param_dict = None
        self._initialize_param_dict(input_param_dict)

        if 'central_occupation_model' in kwargs.keys():
            central_occupation_model = kwargs['central_occupation_model']
        else:
            central_occupation_model = None
        self._set_central_behavior(central_occupation_model)

        self.publications = []

    def _initialize_param_dict(self, input_param_dict):

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
        """ Method ensures that the input central_occupation_model is sensible, 
        and then binds the result to the class instance. 
        """
        self.central_occupation_model = central_occupation_model
        
        if self.central_occupation_model is not None:
            # Test that we were given a sensible input central_occupation_model 
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

            # Test if centrals and satellites thresholds are equal
            if self.threshold != self.central_occupation_model.threshold:
                warnings.warn("Satellite and Central luminosity tresholds do not match")
            #

    def mean_occupation(self, **kwargs):
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

        if 'galaxy_table' in kwargs.keys():
            mass = kwargs['galaxy_table'][self.prim_haloprop_key]
        elif 'halos' in kwargs.keys():
            mass = kwargs['halos'][self.prim_haloprop_key]
        elif 'mass' in kwargs.keys():
            mass = kwargs['mass']
        else:
            raise KeyError("Must pass one of the following keyword arguments to mean_occupation:\n"
                "``halos``, ``mass``, or ``galaxy_table``")

        M0 = 10.**param_dict[self.logM0_key]
        M1 = 10.**param_dict[self.logM1_key]

        # Call to np.where raises a harmless RuntimeWarning exception if 
        # there are entries of input logM for which mean_nsat = 0
        # Evaluating mean_nsat using the catch_warnings context manager 
        # suppresses this warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # Simultaneously evaluate mean_nsat and impose the usual cutoff
            mean_nsat = np.where(mass - M0 > 0, 
                ((mass - M0)/M1)**param_dict[self.alpha_key], 0)

        # If a central occupation model was passed to the constructor, 
        # multiply mean_nsat by an overall factor of mean_ncen
        if self.central_occupation_model is not None:
            mean_ncen = self.central_occupation_model.mean_occupation(**kwargs)
            #mean_nsat = np.where(mean_nsat > 0, mean_nsat*mean_ncen, mean_nsat)
            mean_nsat *= mean_ncen

        return mean_nsat



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

        if publication in ['zheng07', 'Zheng07', 'Zheng_etal07', 'zheng_etal07','zheng2007','Zheng2007']:
            param_dict = get_zheng07_params(threshold)
            return param_dict
        else:
            raise KeyError("For Kravtsov04Sats, only supported best-fit models are currently Zheng et al. 2007")


class BinaryGalpropInterpolModel(object):
    """
    Component model for any binary-valued galaxy property 
    whose assignment is determined by interpolating between points on a grid. 

    One example of a model of this class could be used to help build is 
    Tinker et al. (2013), arXiv:1308.2974. 
    In this model, a galaxy is either active or quiescent, 
    and the quiescent fraction is purely a function of halo mass, with
    separate parameters for centrals and satellites. The value of the quiescent 
    fraction is computed by interpolating between a grid of values of mass. 
    `BinaryGalpropInterpolModel` is quite flexible, and can be used as a template for 
    any binary-valued galaxy property whatsoever. See the examples below for usage instructions. 

    """

    def __init__(self,  logparam=True, interpol_method='spline', 
        abcissa = [12, 15], ordinates = [0.25, 0.75], 
        prim_haloprop_key = 'mvir', galprop_key = 'quiescent', **kwargs):
        """ 
        Parameters 
        ----------
        galprop_key : array, optional keyword argument 
            String giving the name of galaxy property being assigned a binary value. 
            Default is 'quiescent'. 

        abcissa : array, optional keyword argument 
            Values of the primary halo property at which the galprop fraction is specified. 
            Default is [12, 15], in accord with the default True value for ``logparam``. 

        ordinates : array, optional keyword argument 
            Values of the galprop fraction when evaluated at the input abcissa. 
            Default is [0.25, 0.75]

        logparam : bool, optional keyword argument
            If set to True, the interpolation will be done 
            in the base-10 logarithm of the primary halo property, 
            rather than linearly. Default is True. 

        prim_haloprop_key : string, optional keyword argument 
            String giving the key name used to access the primary halo property 
            from an input halo or galaxy catalog. Default is 'mvir'. 

        gal_type : string, optional keyword argument
            Name of the galaxy population being modeled, e.g., 'centrals'. 
            This is only necessary to specify in cases where 
            the `BinaryGalpropInterpolModel` instance is part of a composite model, 
            with multiple population types. Default is None. 

        interpol_method : string, optional keyword argument 
            Keyword specifying how `mean_galprop_fraction` 
            evaluates input values of the primary halo property. 
            The default spline option interpolates the 
            model's abcissa and ordinates. 
            The polynomial option uses the unique, degree N polynomial 
            passing through the ordinates, where N is the number of supplied ordinates. 

        input_spline_degree : int, optional keyword argument
            Degree of the spline interpolation for the case of interpol_method='spline'. 
            If there are k abcissa values specifying the model, input_spline_degree 
            is ensured to never exceed k-1, nor exceed 5. Default is 3. 

        Examples
        -----------       
        Suppose we wish to construct a model for whether a central galaxy is 
        star-forming or quiescent. We want to set the quiescent fraction to 1/3 
        for Milky Way-type centrals (:math:`M_{\\mathrm{vir}}=10^{12}M_{\odot}`), 
        and 90% for massive cluster centrals (:math:`M_{\\mathrm{vir}}=10^{15}M_{\odot}`).

        >>> abcissa, ordinates = [12, 15], [1/3., 0.9]
        >>> cen_quiescent_model = BinaryGalpropInterpolModel(galprop_key='quiescent', abcissa=abcissa, ordinates=ordinates, prim_haloprop_key='mvir', gal_type='cens')

        The ``cen_quiescent_model`` has a built-in method that computes the quiescent fraction 
        as a function of mass:

        >>> quiescent_frac = cen_quiescent_model.mean_quiescent_fraction(prim_haloprop=1e12)

        There is also a built-in method to return a Monte Carlo realization of quiescent/star-forming galaxies:

        >>> masses = np.logspace(10, 15, num=100)
        >>> quiescent_realization = cen_quiescent_model.mc_quiescent(prim_haloprop=masses)

        Now ``quiescent_realization`` is a boolean-valued array of the same length as ``masses``. 
        Entries of ``quiescent_realization`` that are ``True`` correspond to central galaxies that are quiescent. 

        Here is another example of how you could use `BinaryGalpropInterpolModel` 
        to construct a simple model for satellite morphology, where the early- vs. late-type 
        of the satellite depends on :math:`V_{\\mathrm{peak}}` value of the host halo

        >>> sat_morphology_model = BinaryGalpropInterpolModel(galprop_key='late_type', abcissa=abcissa, ordinates=ordinates, prim_haloprop_key='vpeak_host', gal_type='sats')
        >>> vmax_array = np.logspace(2, 3, num=100)
        >>> morphology_realization = sat_morphology_model.mc_late_type(prim_haloprop=vmax_array)

        .. automethod:: _mean_galprop_fraction
        .. automethod:: _mc_galprop
        """

        self._interpol_method = interpol_method
        self._logparam = logparam
        self._abcissa = abcissa
        self._ordinates = ordinates
        self.prim_haloprop_key = prim_haloprop_key
        self.galprop_key = galprop_key
        setattr(self, self.galprop_key+'_abcissa', self._abcissa)

        if 'gal_type' in kwargs.keys():
            self.gal_type = kwargs['gal_type']
            self._abcissa_key = self.galprop_key+'_abcissa_'+self.gal_type
            self._ordinates_key_prefix = self.galprop_key+'_ordinates_'+self.gal_type
        else:
            self._abcissa_key = self.galprop_key+'_abcissa'
            self._ordinates_key_prefix = self.galprop_key+'_ordinates'

        self._build_param_dict()

        if self._interpol_method=='spline':
            if 'input_spline_degree' in kwargs.keys():
                self._input_spine_degree = kwargs['input_spline_degree']
            else:
                self._input_spline_degree = 3
            scipy_maxdegree = 5
            self._spline_degree = np.min(
                [scipy_maxdegree, self._input_spline_degree, 
                custom_len(self._abcissa)-1])

        setattr(self, 'mean_'+self.galprop_key+'_fraction', self._mean_galprop_fraction)
        setattr(self, 'mc_'+self.galprop_key, self._mc_galprop)

    def _build_param_dict(self):

        self._ordinates_keys = [self._ordinates_key_prefix + '_param' + str(i+1) for i in range(custom_len(self._abcissa))]
        self.param_dict = {key:value for key, value in zip(self._ordinates_keys, self._ordinates)}

    def _mean_galprop_fraction(self, **kwargs):
        """
        Expectation value of the galprop for galaxies living in the input halos.  

        Parameters 
        ----------
        mass_like : array_like, optional keyword argument
            Array of primary halo property, e.g., `mvir`, 
            at which the galprop fraction is being computed. 

        prim_haloprop : array_like, optional keyword argument
            Array of primary halo property, e.g., `mvir`, 
            at which the galprop fraction is being computed. 
            Functionality is equivalent to using the mass_like keyword argument. 

        halos : table, optional keyword argument
            Astropy Table containing a halo catalog. 
            If the ``halos`` keyword argument is passed, 
            ``self.prim_haloprop_key`` must be a column of the halo catalog. 

        galaxy_table : table, optional keyword argument
            Astropy Table containing a galaxy catalog. 
            If the ``galaxy_table`` keyword argument is passed, 
            ``self.prim_haloprop_key`` must be a column of the galaxy catalog. 

        input_param_dict : dict, optional keyword argument 
            If passed, the model will first update 
            its param_dict with input_param_dict, altering the behavior of the model. 

        Returns 
        -------
        mean_galprop_fraction : array_like
            Values of the galprop fraction evaluated at the input primary halo properties. 

        """

        # If requested, update the parameter dictionary defining the behavior of the model
        if 'input_param_dict' in kwargs.keys():
            input_param_dict = kwargs['input_param_dict']
            for key in self.param_dict.keys():
                if key in input_param_dict.keys():
                    self.param_dict[key] = input_param_dict[key]

        if 'mass_like' in kwargs.keys():
            mass_like = kwargs['mass_like']
        elif 'prim_haloprop' in kwargs.keys():
            mass_like = kwargs['prim_haloprop']
        elif 'halos' in kwargs.keys():
            mass_like = kwargs['halos'][self.prim_haloprop_key]
        elif 'galaxy_table' in kwargs.keys():
            mass_like = kwargs['galaxy_table'][self.prim_haloprop_key]            
        else:
            raise KeyError("Must pass mean_galprop_fraction one of the "
                "following keyword arguments:\n'mass_like', 'prim_haloprop', 'halos', or 'galaxy_table'\n"
                "Received none of these.")

        if self._logparam is True:
            mass_like = np.log10(mass_like)

        model_ordinates = [self.param_dict[ordinate_key] for ordinate_key in self._ordinates_keys]
        if self._interpol_method=='polynomial':
            mean_galprop_fraction = occuhelp.polynomial_from_table(
                self._abcissa,model_ordinates,mass_like)
        elif self._interpol_method=='spline':
            spline_function = occuhelp.custom_spline(
                self._abcissa,model_ordinates,
                    k=self._spline_degree)
            mean_galprop_fraction = spline_function(mass_like)
        else:
            raise IOError("Input interpol_method must be 'polynomial' or 'spline'.")

        # Enforce boundary conditions 
        mean_galprop_fraction[mean_galprop_fraction<0]=0
        mean_galprop_fraction[mean_galprop_fraction>1]=1

        return mean_galprop_fraction

    def _mc_galprop(self, **kwargs):
        """
        Monte Carlo realization of the binary-valued galprop. 

        Parameters 
        ----------
        mass_like : array_like, optional keyword argument
            Array of primary halo property, e.g., `mvir`, 
            at which the galprop fraction is being computed. 

        prim_haloprop : array_like, optional keyword argument
            Array of primary halo property, e.g., `mvir`, 
            at which the galprop fraction is being computed. 
            Functionality is equivalent to using the mass_like keyword argument. 

        halos : table, optional keyword argument
            Astropy Table containing a halo catalog. 
            If the ``halos`` keyword argument is passed, 
            ``self.prim_haloprop_key`` must be a column of the halo catalog. 

        galaxy_table : table, optional keyword argument
            Astropy Table containing a galaxy catalog. 
            If the ``galaxy_table`` keyword argument is passed, 
            ``self.prim_haloprop_key`` must be a column of the galaxy catalog. 

        input_param_dict : dict, optional keyword argument 
            If passed, the model will first update 
            its param_dict with input_param_dict, altering the behavior of the model. 

        Returns 
        -------
        mc_galprop_fraction : array_like
            Boolean value of whether or not the mock galaxy is posses the galprop, 
            where the Monte Carlo realization is drawn from a nearest-integer 
            distribution determined by `_mean_galprop_fraction`. 

        """
        mean_galprop_fraction = self._mean_galprop_fraction(**kwargs)
        mc_generator = np.random.random(custom_len(mean_galprop_fraction))
        mc_galprop = np.zeros_like(mean_galprop_fraction, dtype=bool)
        mc_galprop[mc_generator<mean_galprop_fraction] = True
        return mc_galprop





















