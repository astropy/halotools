# -*- coding: utf-8 -*-
"""
This module contains various component features used by 
HOD-style models of the galaxy-halo connection. 

"""

__all__ = (['OccupationComponent','Zheng07Cens','Kravtsov04Sats', 
    'Leauthaud11Cens', 'Leauthaud11Sats']
    )

from functools import partial
from copy import copy
import numpy as np
import math
from scipy.special import erf 
from scipy.stats import poisson
from scipy.optimize import brentq
from scipy.interpolate import InterpolatedUnivariateSpline as spline

from . import model_defaults
from . import occupation_helpers as occuhelp
from . import smhm_components

from ..utils.array_utils import array_like_length as custom_len
from ..  import sim_manager

from astropy.extern import six
from abc import ABCMeta, abstractmethod, abstractproperty
import warnings

@six.add_metaclass(ABCMeta)
class OccupationComponent(object):
    """ Abstract base class of any occupation model. 
    Functionality is mostly trivial. 
    The sole purpose of the base class is to 
    standardize the attributes and methods 
    required of any HOD-style model for halo occupation statistics. 
    """
    def __init__(self, **kwargs):
        """ 
        Parameters 
        ----------
        gal_type : string, keyword argument 
            Name of the galaxy population whose occupation statistics is being modeled. 

        threshold : float, keyword argument
            Threshold value defining the selection function of the galaxy population 
            being modeled. Typically refers to absolute magnitude or stellar mass. 

        occupation_bound : float, keyword argument
            Upper bound on the number of gal_type galaxies per halo. 
            The only currently supported values are unity or infinity. 

        prim_haloprop_key : string, keyword argument 
            String giving the column name of the primary halo property governing 
            the occupation statistics of gal_type galaxies. 

        sec_haloprop_key : string, optional keyword argument
            String giving the column name of the secondary halo property governing 
            the occupation statistics of gal_type galaxies. 
            Only pertains to galaxy populations with assembly-biased occupations. 
            Default is None. 

        input_param_dict : dict, optional keyword argument
            Dictionary containing values for the parameters specifying the model.
            All dict keys must conclude with the substring 
            ``_gal_type`` that matches ``self.gal_type``, 
            e.g., ``logMmin_centrals`` or ``alpha_satellites``.
            If no ``input_param_dict`` is passed, ``self.param_dict`` will be 
            initialized with an empty dictionary. 
            The parameters stored in ``self.param_dict`` are the only ones that 
            will be varied in MCMC-type likelihood analyses. 
        """
        required_kwargs = ['gal_type',  'threshold', 'occupation_bound', 'prim_haloprop_key']
        occuhelp.bind_required_kwargs(required_kwargs, self, **kwargs)

        if 'sec_haloprop_key' in kwargs.keys():
            self.sec_haloprop_key = kwargs['sec_haloprop_key']

        if 'input_param_dict' in kwargs.keys():
            self.param_dict = kwargs['input_param_dict']
        else:
            self.param_dict = {}

        # Enforce the requirement that sub-classes have been configured properly
        required_method_name = 'mean_occupation'
        if not hasattr(self, required_method_name):
            raise SyntaxError("Any sub-class of OccupationComponent must "
                "implement a method named %s " % required_method_name)


    def mc_occupation(self, seed=None, **kwargs):
        """ Method to generate Monte Carlo realizations of the abundance of galaxies. 

        Parameters
        ----------        
        prim_haloprop : array, optional keyword argument 
            Array of mass-like variable upon which occupation statistics are based. 
            If ``prim_haloprop`` is not passed, then either ``halos`` or ``galaxy_table`` 
            keyword arguments must be passed. 

        halos : object, optional keyword argument 
            Data table storing halo catalog. 
            If ``halos`` is not passed, then either ``prim_haloprop`` or ``galaxy_table`` 
            keyword arguments must be passed. 

        galaxy_table : object, optional keyword argument 
            Data table storing galaxy catalog. 
            If ``galaxy_table`` is not passed, then either ``prim_haloprop`` or ``halos`` 
            keyword arguments must be passed. 

        input_param_dict : dict, optional keyword argument 
            Dictionary of parameters governing the model. 
            If not passed, values bound to ``self`` will be chosen. 

        seed : int, optional keyword argument 
            Random number seed used to generate the Monte Carlo realization. 
            Default is None. 

        Returns
        -------
        mc_abundance : array
            Integer array giving the number of galaxies in each of the input halos.     
        """ 
        first_occupation_moment = self.mean_occupation(**kwargs)
        if self.occupation_bound == 1:
            return self._nearest_integer_distribution(first_occupation_moment, seed=seed, **kwargs)
        elif self.occupation_bound == float("inf"):
            return self._poisson_distribution(first_occupation_moment, seed=seed, **kwargs)
        else:
            raise KeyError("The only permissible values of occupation_bound for instances "
                "of OccupationComponent are unity and infinity.")

    def _nearest_integer_distribution(self, first_occupation_moment, seed=None, **kwargs):
        """ Nearest-integer distribution used to draw Monte Carlo occupation statistics 
        for central-like populations with only permissible galaxy per halo.

        Parameters 
        ----------
        first_occupation_moment : array
            Array giving the first moment of the occupation distribution function. 

        seed : int, optional keyword argument 
            Random number seed used to generate the Monte Carlo realization. 
            Default is None. 

        Returns
        -------
        mc_abundance : array
            Integer array giving the number of galaxies in each of the input halos. 
        """
        np.random.seed(seed=seed)
        mc_generator = np.random.random(custom_len(first_occupation_moment))
        return np.where(mc_generator < first_occupation_moment, 1, 0)

    def _poisson_distribution(self, first_occupation_moment, seed=None, **kwargs):
        """ Poisson distribution used to draw Monte Carlo occupation statistics 
        for satellite-like populations in which per-halo abundances are unbounded. 

        Parameters 
        ----------
        first_occupation_moment : array
            Array giving the first moment of the occupation distribution function. 

        seed : int, optional keyword argument 
            Random number seed used to generate the Monte Carlo realization. 
            Default is None. 

        Returns
        -------
        mc_abundance : array
            Integer array giving the number of galaxies in each of the input halos. 
        """
        np.random.seed(seed=seed)
        # The scipy built-in Poisson number generator raises an exception 
        # if its input is zero, so here we impose a simple workaround
        first_occupation_moment = np.where(first_occupation_moment <=0, 
            model_defaults.default_tiny_poisson_fluctuation, first_occupation_moment)
        return poisson.rvs(first_occupation_moment)

    @abstractmethod
    def mean_occupation(self):
        """ Method giving the first moment of the occupation distribution. 

        Parameters
        ----------        
        prim_haloprop : array, optional keyword argument 
            Array of mass-like variable upon which occupation statistics are based. 
            If ``prim_haloprop`` is not passed, then either ``halos`` or ``galaxy_table`` 
            keyword arguments must be passed. 

        halos : object, optional keyword argument 
            Data table storing halo catalog. 
            If ``halos`` is not passed, then either ``prim_haloprop`` or ``galaxy_table`` 
            keyword arguments must be passed. 

        galaxy_table : object, optional keyword argument 
            Data table storing galaxy catalog. 
            If ``galaxy_table`` is not passed, then either ``prim_haloprop`` or ``halos`` 
            keyword arguments must be passed. 

        input_param_dict : dict, optional keyword argument 
            Dictionary of parameters governing the model. 
            If not passed, values bound to ``self`` will be chosen. 

        Returns 
        --------
        first_occupation_moment : array
            Array giving the first moment of the occupation distribution function. 
        """
        raise NotImplementedError("All subclasses of OccupationComponent " 
            "must implement a mean_occupation method. ")

class Zheng07Cens(OccupationComponent):
    """ ``Erf`` function model for the occupation statistics of central galaxies, 
    introduced in Zheng et al. 2005, arXiv:0408564. This implementation uses 
    Zheng et al. 2007, arXiv:0703457, to assign fiducial parameter values. 

    """

    def __init__(self, 
        gal_type='centrals', 
        threshold=model_defaults.default_luminosity_threshold,
        prim_haloprop_key=model_defaults.prim_haloprop_key,
        **kwargs):
        """
        Parameters 
        ----------
        gal_type : string, optional keyword argument
            Name of the galaxy population being modeled. Default is ``centrals``.  

        threshold : float, optional keyword argument
            Luminosity threshold of the mock galaxy sample. If specified, 
            input value must agree with one of the thresholds used in Zheng07 to fit HODs: 
            [-18, -18.5, -19, -19.5, -20, -20.5, -21, -21.5, -22].
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        prim_haloprop_key : string, optional keyword argument 
            String giving the column name of the primary halo property governing 
            the occupation statistics of gal_type galaxies. 
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        input_param_dict : dict, optional keyword argument. 
            If ``input_param_dict`` is not passed, the best-fit parameter values 
            provided in Table 1 of Zheng et al. (2007) are chosen. 
            See the `get_published_parameters` method for details. 

        Examples 
        --------
        >>> cen_model = Zheng07Cens()
        >>> cen_model = Zheng07Cens(gal_type='cens', threshold=-19.5)
        >>> cen_model = Zheng07Cens(prim_haloprop_key='m200b')

        Notes 
        -----
        The test suite for this model is documented at 
        `~halotools.empirical_models.test_empirical_models.test_Zheng07Cens`
        """
        occupation_bound = 1.0

        # Call the super class constructor, which binds all the 
        # arguments to the instance.  
        super(Zheng07Cens, self).__init__(gal_type=gal_type, 
            threshold=threshold, occupation_bound=occupation_bound, 
            prim_haloprop_key=prim_haloprop_key, 
            **kwargs)

        self._initialize_param_dict(**kwargs)

        self.publications = ['arXiv:0408564', 'arXiv:0703457']


    def _initialize_param_dict(self, input_param_dict={}, **kwargs):
        """ Private method used to retrieve the 
        dictionary governing the parameters of the model. 
        """
        input_param_dict = input_param_dict

        self.logMmin_key = 'logMmin_'+self.gal_type
        self.sigma_logM_key = 'sigma_logM_'+self.gal_type
        published_param_dict = self.get_published_parameters(self.threshold)

        model_keys = [self.logMmin_key, self.sigma_logM_key]
        for key in model_keys:
            if key in input_param_dict.keys():
                self.param_dict[key] = input_param_dict[key]
            else:
                self.param_dict[key] = published_param_dict[key]

    def mean_occupation(self, **kwargs):
        """ Expected number of central galaxies in a halo of mass halo_mass.
        See Equation 2 of arXiv:0703457.

        Parameters
        ----------        
        prim_haloprop : array, optional keyword argument
            Array storing a mass-like variable that governs the occupation statistics. 
            If ``prim_haloprop`` is not passed, then either ``halos`` or ``galaxy_table`` 
            keyword arguments must be passed. 

        halos : object, optional keyword argument 
            Data table storing halo catalog. 
            If ``halos`` is not passed, then either ``prim_haloprop`` or ``galaxy_table`` 
            keyword arguments must be passed. 

        galaxy_table : object, optional keyword argument 
            Data table storing mock galaxy catalog. 
            If ``galaxy_table`` is not passed, then either ``prim_haloprop`` or ``halos`` 
            keyword arguments must be passed. 

        input_param_dict : dict, optional keyword argument 
            Dictionary of parameters governing the model. 
            If not passed, the values already bound to ``self`` will be used. 

        Returns
        -------
        mean_ncen : array
            Mean number of central galaxies in the input halos. 

        Examples 
        --------
        >>> cen_model = Zheng07Cens()

        The `mean_occupation` method of all OccupationComponent instances supports 
        three different options for arguments. The first option is to directly 
        pass the array of the primary halo property: 

        >>> testmass = np.logspace(10, 15, num=50)
        >>> mean_ncen = cen_model.mean_occupation(prim_haloprop=testmass)

        The second option is to pass `mean_occupation` a full halo catalog. 
        In this case, the array storing the primary halo property will be selected 
        by accessing the ``cen_model.prim_haloprop_key`` column of the input halo catalog. 
        For illustration purposes, we'll use a fake halo catalog rather than a 
        (much larger) full one:

        >>> fake_sim = sim_manager.FakeSim()
        >>> mean_ncen = cen_model.mean_occupation(halos=fake_sim.halos)

        The third option is to pass a table storing a mock galaxy catalog. In this case, 
        the syntax is the same as it is when passing a halo catalog:

        >>> fake_mock = sim_manager.FakeMock()
        >>> mean_ncen = cen_model.mean_occupation(galaxy_table=fake_mock.galaxy_table)

        Notes 
        -----
        The `mean_occupation` method computes the following function: 

        :math:`\\langle N_{\\mathrm{cen}} \\rangle_{M} = 
        \\frac{1}{2}\\left( 1 + 
        \\mathrm{erf}\\left( \\frac{\\log_{10}M - 
        \\log_{10}M_{min}}{\\sigma_{\\log_{10}M}} \\right) \\right)`

        """
        # Retrieve the array storing the mass-like variable
        if 'galaxy_table' in kwargs.keys():
            key = model_defaults.host_haloprop_prefix+self.prim_haloprop_key
            mass = kwargs['galaxy_table'][key]
        elif 'halos' in kwargs.keys():
            mass = kwargs['halos'][self.prim_haloprop_key]
        elif 'prim_haloprop' in kwargs.keys():
            mass = kwargs['prim_haloprop']
        else:
            raise KeyError("Must pass one of the following keyword arguments to mean_occupation:\n"
                "``halos``, ``prim_haloprop``, or ``galaxy_table``")

        occuhelp.update_param_dict(self, **kwargs)

        logM = np.log10(mass)
        mean_ncen = 0.5*(1.0 + erf(
            (logM - self.param_dict[self.logMmin_key])
            /self.param_dict[self.sigma_logM_key]))

        return mean_ncen


    def get_published_parameters(self, threshold, publication='Zheng07'):
        """
        Best-fit HOD parameters from Table 1 of Zheng et al. 2007.

        Parameters 
        ----------

        threshold : float
            Luminosity threshold defining the SDSS sample 
            to which Zheng et al. fit their HOD model. 
            If the ``publication`` keyword argument is set to ``Zheng07``, 
            then ``threshold`` must be agree with one of the published values: 
            [-18, -18.5, -19, -19.5, -20, -20.5, -21, -21.5, -22].

        publication : string, optional keyword argument 
            String specifying the publication that will be used to set  
            the values of ``param_dict``. Default is Zheng et al. (2007). 

        Returns 
        -------
        param_dict : dict
            Dictionary of model parameters whose values have been set to 
            agree with the values taken from Table 1 of Zheng et al. 2007.

        Examples 
        --------
        >>> cen_model = Zheng07Cens()
        >>> cen_model.param_dict = cen_model.get_published_parameters(cen_model.threshold)

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
            raise KeyError("For Zheng07Cens, only supported best-fit models are currently Zheng et al. 2007")


class Leauthaud11Cens(OccupationComponent):
    """ HOD-style model for any central galaxy occupation that derives from 
    a stellar-to-halo-mass relation. 
    """
    def __init__(self, smhm_model=smhm_components.Moster13SmHm, 
        gal_type = 'centrals', 
        threshold = model_defaults.default_stellar_mass_threshold, 
        prim_haloprop_key=model_defaults.prim_haloprop_key,
        **kwargs):
        """
        Parameters 
        ----------
        gal_type : string, optional keyword argument
            Name of the galaxy population being modeled. Default is ``centrals``.  

        threshold : float, optional keyword argument
            Stellar mass threshold of the mock galaxy sample. 
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        prim_haloprop_key : string, optional keyword argument 
            String giving the column name of the primary halo property governing 
            the occupation statistics of gal_type galaxies. 
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        smhm_model : object, optional keyword argument 
            Sub-class of `~halotools.empirical_models.smhm_components.SmHmModel` governing 
            the stellar-to-halo-mass relation 

        redshift : float, optional keyword argument 
            Redshift of the stellar-to-halo-mass relation. Default is 0. 

        scatter_model : object, optional keyword argument 
            Class governing stochasticity of stellar mass. Default scatter is log-normal, 
            implemented by the `~halotools.empirical_models.LogNormalScatterModel` class. 

        scatter_abcissa : array_like, optional keyword argument 
            Array of values giving the abcissa at which
            the level of scatter will be specified by the input ordinates.
            Default behavior will result in constant scatter at a level set in the 
            `~halotools.empirical_models.model_defaults` module. 

        scatter_ordinates : array_like, optional keyword argument 
            Array of values defining the level of scatter at the input abcissa.
            Default behavior will result in constant scatter at a level set in the 
            `~halotools.empirical_models.model_defaults` module. 

        input_param_dict : dict, optional keyword argument. 
            If ``input_param_dict`` is not passed, the best-fit parameter values 
            will be taken from the input ``smhm_model``. 
        """
        occupation_bound = 1.0

        # Call the super class constructor, which binds all the 
        # arguments to the instance.  
        super(Leauthaud11Cens, self).__init__(
            gal_type=gal_type, threshold=threshold, 
            occupation_bound=occupation_bound, 
            prim_haloprop_key = prim_haloprop_key, 
            **kwargs)

        self.smhm_model = smhm_model(
            gal_type=gal_type, prim_haloprop_key = prim_haloprop_key, 
            **kwargs)
        self.param_dict = self.smhm_model.param_dict

        self.publications = ['arXiv:1103.2077', 'arXiv:1104.0928']

    def mean_occupation(self, **kwargs):
        """ Expected number of central galaxies in a halo of mass halo_mass.
        See Equation 8 of arXiv:1103.2077.

        Parameters
        ----------        
        prim_haloprop : array, optional keyword argument
            array of masses of halos in the catalog

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
        Assumes constant scatter in the stellar-to-halo-mass relation. 
        """
        occuhelp.update_param_dict(self, **kwargs)
        if 'input_param_dict' in kwargs.keys():
            del kwargs['input_param_dict']

        logmstar = np.log10(self.smhm_model.mean_stellar_mass(
            input_param_dict = self.param_dict, **kwargs))
        logscatter = math.sqrt(2)*self.smhm_model.mean_scatter(
            input_param_dict = self.param_dict, **kwargs)

        mean_ncen = 0.5*(1.0 - 
            erf((self.threshold - logmstar)/logscatter))

        return mean_ncen


class Kravtsov04Sats(OccupationComponent):
    """ Power law model for the occupation statistics of satellite galaxies, 
    introduced in Kravtsov et al. 2004, arXiv:0308519.

    :math:`\\langle N_{sat} \\rangle_{M} = \left( \\frac{M - M_{0}}{M_{1}} \\right)^{\\alpha}`

    """

    def __init__(self,
        gal_type='satellites', 
        threshold=model_defaults.default_luminosity_threshold,
        prim_haloprop_key=model_defaults.prim_haloprop_key,
        central_occupation_model = None, 
        **kwargs):
        """
        Parameters 
        ----------
        gal_type : string, optional keyword argument
            Name of the galaxy population being modeled. Default is ``satellites``.  

        threshold : float, optional keyword argument
            Luminosity threshold of the mock galaxy sample. If specified, 
            input value must agree with one of the thresholds used in Zheng07 to fit HODs: 
            [-18, -18.5, -19, -19.5, -20, -20.5, -21, -21.5, -22].
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        prim_haloprop_key : string, optional keyword argument 
            String giving the column name of the primary halo property governing 
            the occupation statistics of gal_type galaxies. 
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        input_param_dict : dict, optional keyword argument. 
            If ``input_param_dict`` is not passed, the best-fit parameter values 
            provided in Table 1 of Zheng et al. (2007) are chosen. 
            See the `get_published_parameters` method for details. 

        central_occupation_model : occupation model instance, optional
            Must be an instance of a sub-class of `~halotools.empirical_models.OccupationComponent`. 
            If a ``central_occupation_model`` is being used, 
            the mean occupation method of `Kravtsov04Sats` will 
            be multiplied by the value of central_occupation_model at each mass, 
            as in Zheng et al. 2007, so that 
            :math:`\\langle N_{\mathrm{sat}}\\rangle_{M}\\Rightarrow\\langle N_{\mathrm{sat}}\\rangle_{M}\\times\\langle N_{\mathrm{cen}}\\rangle_{M}`

        Examples 
        --------
        >>> sat_model = Kravtsov04Sats()
        >>> sat_model = Kravtsov04Sats(gal_type='sats')
        >>> sat_model = Kravtsov04Sats(threshold = -21)

        The ``input_param_dict`` keyword argument can be used to build an alternate 
        model from an existing instance. This feature has a variety of uses. For example, 
        suppose you wish to study how the choice of halo mass definition impacts HOD predictions:

        >>> sat_model1 = Kravtsov04Sats(threshold = -19.5, prim_haloprop_key='m200b')
        >>> sat_model1.param_dict['alpha_satellites'] = 1.05
        >>> sat_model2 = Kravtsov04Sats(threshold = sat_model1.threshold, input_param_dict = sat_model1.param_dict, prim_haloprop_key='m500c')

        A common convention in HOD modeling of satellite populations is for the first 
        occupation moment of the satellites to be multiplied by the first central occupation 
        moment. Coupling the ``input_param_dict`` and ``central_occupation_model`` 
        keyword arguments allows you to study the impact of this choice:

        >>> sat_model1 = Kravtsov04Sats(threshold=-18)
        >>> cen_model_instance = Zheng07Cens(threshold = sat_model1.threshold)
        >>> sat_model2 = Kravtsov04Sats(threshold = sat_model1.threshold, input_param_dict=sat_model1.param_dict, central_occupation_model=cen_model_instance)

        Now ``sat_model1`` and ``sat_model2`` are identical in every respect, 
        excepting only the following difference:

        :math:`\\langle N_{\mathrm{sat}}\\rangle^{\mathrm{model 2}} = \\langle N_{\mathrm{cen}}\\rangle\\times\\langle N_{\mathrm{sat}}\\rangle^{\mathrm{model 1}}`


        Notes 
        -----
        The test suite for this model is documented at 
        `~halotools.empirical_models.test_empirical_models.test_Kravtsov04Sats`

        """
        occupation_bound = float("inf")

        # Call the super class constructor, which binds all the 
        # arguments to the instance.  
        super(Kravtsov04Sats, self).__init__(
            gal_type=gal_type, threshold=threshold, 
            occupation_bound=occupation_bound, 
            prim_haloprop_key = prim_haloprop_key, 
            **kwargs)

        self._initialize_param_dict(**kwargs)

        self._check_consistent_central_behavior(central_occupation_model)
        self.central_occupation_model = central_occupation_model

        self.publications = ['arXiv:0308519', 'arXiv:0703457']

    def _initialize_param_dict(self, input_param_dict = {}, **kwargs):
        """ Method stores default values into ``self.param_dict``. 
        """

        input_param_dict = input_param_dict

        self.logM0_key = 'logM0_'+self.gal_type
        self.logM1_key = 'logM1_'+self.gal_type
        self.alpha_key = 'alpha_'+self.gal_type
        published_param_dict = self.get_published_parameters(self.threshold)

        model_keys = [self.logM0_key, self.logM1_key, self.alpha_key]
        for key in model_keys:
            if key in input_param_dict.keys():
                self.param_dict[key] = input_param_dict[key]
            else:
                self.param_dict[key] = published_param_dict[key]

    def _check_consistent_central_behavior(self, central_occupation_model):
        """ Method ensures that the input central_occupation_model is sensible.
        """        
        if central_occupation_model is not None:
            # Test that we were given a sensible input central_occupation_model 
            if not isinstance(central_occupation_model, OccupationComponent):
                msg = ("When passing a central_occupation_model to " + 
                    "the Kravtsov04Sats constructor, \n you must pass an instance of " + 
                    "an OccupationComponent.")
                if issubclass(central_occupation_model, OccupationComponent):
                    msg = (msg + 
                        "\n Instead, the Kravtsov04Sats received the actual class " + 
                        central_occupation_model.__name__+", " + 
                    "rather than an instance of that class. ")
                raise SyntaxError(msg)

            # Test if centrals and satellites thresholds are equal
            if self.threshold != central_occupation_model.threshold:
                warnings.warn("Satellite and Central luminosity tresholds do not match")
            #

    def mean_occupation(self, **kwargs):
        """Expected number of satellite galaxies in a halo of mass logM.
        See Equation 5 of arXiv:0703457.

        Parameters
        ----------        
        prim_haloprop : array, optional keyword argument
            Array storing a mass-like variable that governs the occupation statistics. 
            If ``prim_haloprop`` is not passed, then either ``halos`` or ``galaxy_table`` 
            keyword arguments must be passed. 

        halos : object, optional keyword argument 
            Data table storing halo catalog. 
            If ``halos`` is not passed, then either ``prim_haloprop`` or ``galaxy_table`` 
            keyword arguments must be passed. 

        galaxy_table : object, optional keyword argument 
            Data table storing mock galaxy catalog. 
            If ``galaxy_table`` is not passed, then either ``prim_haloprop`` or ``halos`` 
            keyword arguments must be passed. 

        input_param_dict : dict, optional keyword argument 
            Dictionary of parameters governing the model. 
            If not passed, the values already bound to ``self`` will be used. 

        Returns
        -------
        mean_nsat : float or array
            Mean number of satellite galaxies in a host halo of the specified mass. 

        :math:`\\langle N_{\\mathrm{sat}} \\rangle_{M} = \left( \\frac{M - M_{0}}{M_{1}} \\right)^{\\alpha} \\langle N_{\\mathrm{cen}} \\rangle_{M}`

        or 

        :math:`\\langle N_{\\mathrm{sat}} \\rangle_{M} = \left( \\frac{M - M_{0}}{M_{1}} \\right)^{\\alpha}`, 

        depending on whether a central model was passed to the constructor. 

        Examples 
        --------
        The `mean_occupation` method of all OccupationComponent instances supports 
        three different options for arguments. The first option is to directly 
        pass the array of the primary halo property: 

        >>> sat_model = Kravtsov04Sats()
        >>> testmass = np.logspace(10, 15, num=50)
        >>> mean_nsat = sat_model.mean_occupation(prim_haloprop=testmass)

        The second option is to pass `mean_occupation` a full halo catalog. 
        In this case, the array storing the primary halo property will be selected 
        by accessing the ``sat_model.prim_haloprop_key`` column of the input halo catalog. 
        For illustration purposes, we'll use a fake halo catalog rather than a 
        (much larger) full one:

        >>> fake_sim = sim_manager.FakeSim()
        >>> mean_nsat = sat_model.mean_occupation(halos=fake_sim.halos)

        The third option is to pass a table storing a mock galaxy catalog. In this case, 
        the syntax is the same as it is when passing a halo catalog:

        >>> fake_mock = sim_manager.FakeMock()
        >>> mean_nsat = sat_model.mean_occupation(galaxy_table=fake_mock.galaxy_table)

        """
        # Retrieve the array storing the mass-like variable
        if 'galaxy_table' in kwargs.keys():
            key = model_defaults.host_haloprop_prefix+self.prim_haloprop_key
            mass = kwargs['galaxy_table'][key]
        elif 'halos' in kwargs.keys():
            mass = kwargs['halos'][self.prim_haloprop_key]
        elif 'prim_haloprop' in kwargs.keys():
            mass = kwargs['prim_haloprop']
        else:
            raise KeyError("Must pass one of the following keyword arguments to mean_occupation:\n"
                "``halos``, ``prim_haloprop``, or ``galaxy_table``")

        occuhelp.update_param_dict(self, **kwargs)

        M0 = 10.**self.param_dict[self.logM0_key]
        M1 = 10.**self.param_dict[self.logM1_key]

        # Call to np.where raises a harmless RuntimeWarning exception if 
        # there are entries of input logM for which mean_nsat = 0
        # Evaluating mean_nsat using the catch_warnings context manager 
        # suppresses this warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # Simultaneously evaluate mean_nsat and impose the usual cutoff
            mean_nsat = np.where(mass - M0 > 0, 
                ((mass - M0)/M1)**self.param_dict[self.alpha_key], 0)

        # If a central occupation model was passed to the constructor, 
        # multiply mean_nsat by an overall factor of mean_ncen
        if self.central_occupation_model is not None:
            mean_ncen = self.central_occupation_model.mean_occupation(**kwargs)
            mean_nsat *= mean_ncen

        return mean_nsat

    def get_published_parameters(self, threshold, publication='Zheng07'):
        """
        Best-fit HOD parameters from Table 1 of Zheng et al. 2007.

        Parameters 
        ----------
        threshold : float
            Luminosity threshold of the mock galaxy sample. 
            Input value must agree with one of the thresholds used in Zheng07 to fit HODs: 
            [-18, -18.5, -19, -19.5, -20, -20.5, -21, -21.5, -22].

        publication : string, optional keyword argument 
            String specifying the publication that will be used to set  
            the values of ``param_dict``. Default is Zheng et al. (2007). 

        Returns 
        -------
        param_dict : dict
            Dictionary of model parameters whose values have been set to 
            the values taken from Table 1 of Zheng et al. 2007.

        Examples 
        --------
        >>> sat_model = Kravtsov04Sats()
        >>> sat_model.param_dict = sat_model.get_published_parameters(sat_model.threshold)
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


class Leauthaud11Sats(OccupationComponent):
    """ HOD-style model for any satellite galaxy occupation that derives from 
    a stellar-to-halo-mass relation. 
    """
    def __init__(self, smhm_model=smhm_components.Moster13SmHm, 
        gal_type = 'satellites', 
        threshold = model_defaults.default_stellar_mass_threshold, 
        prim_haloprop_key=model_defaults.prim_haloprop_key,
        modulate_with_cenocc = True, 
        **kwargs):
        """
        Parameters 
        ----------
        gal_type : string, optional keyword argument
            Name of the galaxy population being modeled. Default is ``satellites``.  

        threshold : float, optional keyword argument
            Stellar mass threshold of the mock galaxy sample. 
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        prim_haloprop_key : string, optional keyword argument 
            String giving the column name of the primary halo property governing 
            the occupation statistics of gal_type galaxies. 
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        smhm_model : object, optional keyword argument 
            Sub-class of `~halotools.empirical_models.smhm_components.SmHmModel` governing 
            the stellar-to-halo-mass relation 

        redshift : float, optional keyword argument 
            Redshift of the stellar-to-halo-mass relation. Default is 0. 

        modulate_with_cenocc : bool, optional keyword argument 
            If True, the first satellite moment will be multiplied by the 
            the first central moment. Default is True. 

        scatter_model : object, optional keyword argument 
            Class governing stochasticity of stellar mass. Default scatter is log-normal, 
            implemented by the `~halotools.empirical_models.LogNormalScatterModel` class. 

        scatter_abcissa : array_like, optional keyword argument 
            Array of values giving the abcissa at which
            the level of scatter will be specified by the input ordinates.
            Default behavior will result in constant scatter at a level set in the 
            `~halotools.empirical_models.model_defaults` module. 

        scatter_ordinates : array_like, optional keyword argument 
            Array of values defining the level of scatter at the input abcissa.
            Default behavior will result in constant scatter at a level set in the 
            `~halotools.empirical_models.model_defaults` module. 

        input_param_dict : dict, optional keyword argument. 
            If ``input_param_dict`` is not passed, the best-fit parameter values 
            will be taken from the input ``smhm_model``. 

        Examples 
        --------
        >>> sat_model = Leauthaud11Sats()
        """

        self.central_occupation_model = Leauthaud11Cens(
            gal_type='centrals', threshold=threshold,
            prim_haloprop_key = prim_haloprop_key, smhm_model = smhm_model, 
            **kwargs)

        super(Leauthaud11Sats, self).__init__(
            gal_type=gal_type, threshold=threshold, 
            occupation_bound=float("inf"), 
            prim_haloprop_key = prim_haloprop_key, 
            **kwargs)

        self._initialize_param_dict(**kwargs)

        self.modulate_with_cenocc = modulate_with_cenocc

        self.publications = self.central_occupation_model.publications

    def mean_occupation(self, **kwargs):
        """ Expected number of central galaxies in a halo of mass halo_mass.
        See Equation 12-14 of arXiv:1103.2077.

        Parameters
        ----------        
        prim_haloprop : array, optional keyword argument
            array of masses of halos in the catalog

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

        Examples 
        --------
        >>> sat_model = Leauthaud11Sats()
        >>> mean_nsat = sat_model.mean_occupation(prim_haloprop = 1.e13)

        Notes 
        -----
        Assumes constant scatter in the stellar-to-halo-mass relation. 
        """
        self._update_satellite_params(**kwargs)

        # Retrieve the array storing the mass-like variable
        if 'galaxy_table' in kwargs.keys():
            key = model_defaults.host_haloprop_prefix+self.prim_haloprop_key
            mass = kwargs['galaxy_table'][key]
        elif 'halos' in kwargs.keys():
            mass = kwargs['halos'][self.prim_haloprop_key]
        elif 'prim_haloprop' in kwargs.keys():
            mass = kwargs['prim_haloprop']
        else:
            raise KeyError("Must pass one of the following keyword arguments to mean_occupation:\n"
                "``halos``, ``prim_haloprop``, or ``galaxy_table``")

        mean_nsat = (
            np.exp(-self._mcut/mass)*
            (mass/self._msat)**self.param_dict[self._alphasat_key]
            )

        if self.modulate_with_cenocc is True:
            mean_nsat *= self.central_occupation_model.mean_occupation(**kwargs)

        return mean_nsat

    def _initialize_param_dict(self, **kwargs):
        """ Set the initial values of ``self.param_dict`` according to 
        the SIG_MOD1 values of Table 5 of arXiv:1104.0928 for the 
        lowest redshift bin. 

        Parameters 
        ----------
        input_param_dict : dict, optional
            dictionary of parameters governing the model. If not passed, 
            values bound to ``self`` will be chosen. 

        Notes 
        -----
        These values are only for ballpark purposes, and are 
        not self-consistent with arXiv:1104.0928, 
        because a different stellar-to-halo-mass relation is used here. 
        """
        occuhelp.update_param_dict(self, **kwargs)

        self._msat_mcut_abcissa = np.logspace(9, 15, num=500)

        self._bsat_key = 'bsat_'+self.gal_type
        self._bcut_key = 'bcut_'+self.gal_type
        self._betasat_key = 'betasat_'+self.gal_type
        self._betacut_key = 'betacut_'+self.gal_type
        self._alphasat_key = 'alphasat_'+self.gal_type

        self.param_dict[self._alphasat_key] = 1.0
        self.param_dict[self._bsat_key] = 10.62
        self.param_dict[self._bcut_key] = 1.47
        self.param_dict[self._betacut_key] = -0.13
        self.param_dict[self._betasat_key] = 0.859

        self._update_satellite_params(**kwargs)


    def _update_satellite_params(self, **kwargs):
        """ Private method to update the model parameters. 

        Parameters 
        ----------
        input_param_dict : dict, optional
            dictionary of parameters governing the model. If not passed, 
            values bound to ``self`` will be chosen. 
        """
        occuhelp.update_param_dict(self, **kwargs)

        if 'input_param_dict' in kwargs.keys():
            input_param_dict = kwargs['input_param_dict']
        else:
            input_param_dict = {}

        # Tabulate the inverse stellar-to-halo-mass relation
        ordinates = self.central_occupation_model.smhm_model.mean_stellar_mass(
            prim_haloprop=self._msat_mcut_abcissa, 
            input_param_dict = input_param_dict)
        spline_function = spline(ordinates, self._msat_mcut_abcissa)

        # Call the interpolater to compute the knee
        knee = spline_function(10.**self.threshold)

        self._msat = (
            1.e12*self.param_dict[self._bsat_key]*
            (knee / 1.e12)**self.param_dict[self._betasat_key])

        self._mcut = (
            1.e12*self.param_dict[self._bcut_key]*
            (knee / 1.e12)**self.param_dict[self._betacut_key])

























