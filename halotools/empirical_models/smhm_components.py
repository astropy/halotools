# -*- coding: utf-8 -*-
"""
Module containing classes used to model the mapping between 
stellar mass and subhalo_table. 
"""
import numpy as np

from scipy.interpolate import UnivariateSpline

from astropy.extern import six
from abc import ABCMeta, abstractmethod, abstractproperty

from . import model_defaults
from . import model_helpers as model_helpers

from ..utils.array_utils import custom_len
from ..sim_manager import sim_defaults 

from astropy import cosmology

from warnings import warn
from functools import partial

__all__ = ['PrimGalpropModel', 'Moster13SmHm', 'Behroozi10SmHm', 'LogNormalScatterModel']

class LogNormalScatterModel(object):
    """ Simple model used to generate log-normal scatter 
    in a stellar-to-halo-mass type relation. 

    """

    def __init__(self, 
        prim_haloprop_key=model_defaults.default_smhm_haloprop, 
        **kwargs):
        """
        Parameters 
        ----------
        prim_haloprop_key : string, optional  
            String giving the column name of the primary halo property governing 
            the level of scatter. 
            Default is set in the `~halotools.empirical_models.model_defaults` module. 

        scatter_abcissa : array_like, optional  
            Array of values giving the abcissa at which
            the level of scatter will be specified by the input ordinates.
            Default behavior will result in constant scatter at a level set in the 
            `~halotools.empirical_models.model_defaults` module. 

        scatter_ordinates : array_like, optional  
            Array of values defining the level of scatter at the input abcissa.
            Default behavior will result in constant scatter at a level set in the 
            `~halotools.empirical_models.model_defaults` module. 

        Examples 
        ---------
        >>> scatter_model = LogNormalScatterModel()
        >>> scatter_model = LogNormalScatterModel(prim_haloprop_key='halo_mvir')

        To implement variable scatter, we need to define the level 
        of log-normal scatter at a set of control values 
        of the primary halo property. Here we give an example of a model 
        in which the scatter is 0.3 dex for Milky Way halo_table and 0.1 dex in cluster halo_table:

        >>> scatter_abcissa = [12, 15]
        >>> scatter_ordinates = [0.3, 0.1]
        >>> scatter_model = LogNormalScatterModel(scatter_abcissa=scatter_abcissa, scatter_ordinates=scatter_ordinates)

        """
        default_scatter = model_defaults.default_smhm_scatter
        self.prim_haloprop_key = prim_haloprop_key

        if ('scatter_abcissa' in kwargs.keys()) and ('scatter_ordinates' in kwargs.keys()):
            self.abcissa = kwargs['scatter_abcissa']
            self.ordinates = kwargs['scatter_ordinates']
        else:
            self.abcissa = [12]
            self.ordinates = [default_scatter]

        self._initialize_param_dict()

        self._update_interpol()

    def mean_scatter(self, **kwargs):
        """ Return the amount of log-normal scatter that should be added 
        to the galaxy property as a function of the input halo_table. 

        Parameters 
        ----------
        prim_haloprop : array, optional  
            Array of mass-like variable upon which occupation statistics are based. 
            If ``prim_haloprop`` is not passed, then ``halo_table`` keyword argument must be passed. 

        halo_table : object, optional  
            Data table storing halo catalog. 
            If ``halo_table`` is not passed, then ``prim_haloprop`` keyword argument must be passed. 

        Returns 
        -------
        scatter : array_like 
            Array containing the amount of log-normal scatter evaluated 
            at the input halo_table. 
        """
        # Retrieve the array storing the mass-like variable
        if 'halo_table' in kwargs.keys():
            mass = kwargs['halo_table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in kwargs.keys():
            mass = kwargs['prim_haloprop']
        else:
            raise KeyError("Must pass one of the following keyword arguments to mean_occupation:\n"
                "``halo_table`` or ``prim_haloprop``")

        self._update_interpol()

        return self.spline_function(np.log10(mass))

    def scatter_realization(self, seed=None, **kwargs):
        """ Return the amount of log-normal scatter that should be added 
        to the galaxy property as a function of the input halo_table. 

        Parameters 
        ----------
        prim_haloprop : array, optional  
            Array of mass-like variable upon which occupation statistics are based. 
            If ``prim_haloprop`` is not passed, then ``halo_table`` keyword argument must be passed. 

        halo_table : object, optional  
            Data table storing halo catalog. 
            If ``halo_table`` is not passed, then ``prim_haloprop`` keyword argument must be passed. 

        seed : int, optional  
            Random number seed. Default is None. 

        Returns 
        -------
        scatter : array_like 
            Array containing a random variable realization that should be summed 
            with the galaxy property to add scatter.  
        """

        scatter_scale = self.mean_scatter(**kwargs)

        np.random.seed(seed=seed)
            
        return np.random.normal(loc=0, scale=scatter_scale)

    def _update_interpol(self):
        """ Private method that updates the interpolating functon used to 
        define the level of scatter as a function of the input halo_table. 
        If this method is not called after updating ``self.param_dict``, 
        changes in ``self.param_dict`` will not alter the model behavior. 
        """

        scipy_maxdegree = 5
        degree_list = [scipy_maxdegree, custom_len(self.abcissa)-1]
        self.spline_degree = np.min(degree_list)

        self.ordinates = [self.param_dict[self._get_param_key(i)] for i in range(len(self.abcissa))]

        self.spline_function = model_helpers.custom_spline(
            self.abcissa, self.ordinates, k=self.spline_degree)

    def _initialize_param_dict(self):
        """ Private method used to initialize ``self.param_dict``. 
        """
        self.param_dict={}
        for ipar, val in enumerate(self.ordinates):
            key = self._get_param_key(ipar)
            self.param_dict[key] = val

    def _get_param_key(self, ipar):
        """ Private method used to retrieve the key of self.param_dict 
        that corresponds to the appropriately selected i^th ordinate 
        defining the behavior of the scatter model. 
        """
        return 'scatter_model_param'+str(ipar+1)

@six.add_metaclass(ABCMeta)
class PrimGalpropModel(model_helpers.GalPropModel):
    """ Abstract container class for models connecting halo_table to their primary
    galaxy property, e.g., stellar mass or luminosity. 
    """

    def __init__(self, galprop_key = 'stellar_mass', 
        prim_haloprop_key = model_defaults.default_smhm_haloprop, 
        scatter_model = LogNormalScatterModel, 
        **kwargs):
        """
        Parameters 
        ----------
        galprop_key : string, optional  
            Name of the galaxy property being assigned. Default is ``stellar mass``, 
            though another common case may be ``luminosity``. 

        prim_haloprop_key : string, optional  
            String giving the column name of the primary halo property governing 
            stellar mass.  
            Default is set in the `~halotools.empirical_models.model_defaults` module. 

        scatter_model : object, optional  
            Class governing stochasticity of stellar mass. Default scatter is log-normal, 
            implemented by the `LogNormalScatterModel` class. 

        redshift : float, optional  
            Redshift of the stellar-to-halo-mass relation. Default is 0. 

        scatter_abcissa : array_like, optional  
            Array of values giving the abcissa at which
            the level of scatter will be specified by the input ordinates.
            Default behavior will result in constant scatter at a level set in the 
            `~halotools.empirical_models.model_defaults` module. 

        scatter_ordinates : array_like, optional  
            Array of values defining the level of scatter at the input abcissa.
            Default behavior will result in constant scatter at a level set in the 
            `~halotools.empirical_models.model_defaults` module. 

        new_haloprop_func_dict : function object, optional  
            Dictionary of function objects used to create additional halo properties 
            that may be needed by the model component. 
            Used strictly by the `MockFactory` during call to the `process_halo_catalog` method. 
            Each dict key of ``new_haloprop_func_dict`` will 
            be the name of a new column of the halo catalog; each dict value is a function 
            object that returns a length-N numpy array when passed a length-N Astropy table 
            via the ``halo_table`` keyword argument. 
            The input ``model`` model object has its own new_haloprop_func_dict; 
            if the keyword argument ``new_haloprop_func_dict`` passed to `MockFactory` 
            contains a key that already appears in the ``new_haloprop_func_dict`` bound to 
            ``model``, and exception will be raised. 
        """
        self.galprop_key = galprop_key
        self.prim_haloprop_key = prim_haloprop_key

        if 'redshift' in kwargs.keys():
            self.redshift = kwargs['redshift']

        if 'new_haloprop_func_dict' in kwargs.keys():
            self.new_haloprop_func_dict = kwargs['new_haloprop_func_dict']

        self.scatter_model = scatter_model(
            prim_haloprop_key=self.prim_haloprop_key, **kwargs)

        self._build_param_dict(**kwargs)

        # Enforce the requirement that sub-classes have been configured properly
        required_method_name = 'mean_'+self.galprop_key
        if not hasattr(self, required_method_name):
            raise SyntaxError("Any sub-class of PrimGalpropModel must "
                "implement a method named %s " % required_method_name)

        # If the sub-class did not implement their own Monte Carlo method mc_galprop, 
        # then use _mc_galprop and give it the usual name
        if not hasattr(self, 'mc_'+self.galprop_key):
            setattr(self, 'mc_'+self.galprop_key, self._mc_galprop)

        super(PrimGalpropModel, self).__init__(galprop_key=self.galprop_key)

    def mean_scatter(self, **kwargs):
        """ Use the ``param_dict`` of `PrimGalpropModel` to update the ``param_dict`` 
        of the scatter model, and then call the `mean_scatter` method of 
        the scatter model. 
        """
        for key in self.scatter_model.param_dict.keys():
            self.scatter_model.param_dict[key] = self.param_dict[key]

        return self.scatter_model.mean_scatter(**kwargs)

    def scatter_realization(self, **kwargs):
        """ Use the ``param_dict`` of `PrimGalpropModel` to update the ``param_dict`` 
        of the scatter model, and then call the `scatter_realization` method of 
        the scatter model. 
        """
        for key in self.scatter_model.param_dict.keys():
            self.scatter_model.param_dict[key] = self.param_dict[key]

        return self.scatter_model.scatter_realization(**kwargs)

    def _build_param_dict(self, **kwargs):
        """ Method combines the parameter dictionaries of the 
        smhm model and the scatter model. 
        """

        if hasattr(self, 'retrieve_default_param_dict'):
            self.param_dict = self.retrieve_default_param_dict()
        else:
            self.param_dict = {}

        scatter_param_dict = self.scatter_model.param_dict

        for key, value in scatter_param_dict.iteritems():
            self.param_dict[key] = value

    def _mc_galprop(self, include_scatter = True, **kwargs):
        """ Return the prim_galprop of the galaxies living in the input halo_table. 

        Parameters 
        ----------
        prim_haloprop : array, optional  
            Array of mass-like variable upon which occupation statistics are based. 
            If ``prim_haloprop`` is not passed, then ``halo_table`` keyword argument must be passed. 

        halo_table : object, optional  
            Data table storing halo catalog. 
            If ``halo_table`` is not passed, then ``prim_haloprop`` keyword argument must be passed. 

        redshift : float, optional 
            Redshift of the halo hosting the galaxy. 

        include_scatter : boolean, optional  
            Determines whether or not the scatter model is applied to add stochasticity 
            to the galaxy property assignment. Default is True. 
            If False, model is purely deterministic, and the behavior is determined 
            by the ``mean_galprop`` method of the sub-class. 

        Returns 
        -------
        prim_galprop : array_like 
            Array storing the values of the primary galaxy property 
            of the galaxies living in the input halo_table. 
        """

        # Interpret the inputs to determine the appropriate redshift
        if 'redshift' not in kwargs.keys():
            if hasattr(self, 'redshift'):
                kwargs['redshift'] = self.redshift
            else:
                warn("\nThe PrimGalpropModel class was not instantiated with a redshift,\n"
                "nor was a redshift passed to the primary function call.\n"
                "Choosing the default redshift z = %.2f\n" % sim_defaults.default_redshift)
                kwargs['redshift'] = sim_defaults.default_redshift

        prim_galprop_func = getattr(self, 'mean_'+self.galprop_key)
        galprop_first_moment = prim_galprop_func(**kwargs)

        if include_scatter is False:
            return galprop_first_moment
        else:
            log10_galprop_with_scatter = (
                np.log10(galprop_first_moment) + 
                self.scatter_realization(**kwargs)
                )
            return 10.**log10_galprop_with_scatter


class Moster13SmHm(PrimGalpropModel):
    """ Stellar-to-halo-mass relation based on 
    Moster et al. (2013), arXiv:1205.5807. 
    """

    def __init__(self, **kwargs):
        """
        Parameters 
        ----------
        prim_haloprop_key : string, optional  
            String giving the column name of the primary halo property governing stellar mass. 
            Default is set in the `~halotools.empirical_models.model_defaults` module. 

        scatter_model : object, optional  
            Class governing stochasticity of stellar mass. Default scatter is log-normal, 
            implemented by the `LogNormalScatterModel` class. 

        scatter_abcissa : array_like, optional  
            Array of values giving the abcissa at which
            the level of scatter will be specified by the input ordinates.
            Default behavior will result in constant scatter at a level set in the 
            `~halotools.empirical_models.model_defaults` module. 

        scatter_ordinates : array_like, optional  
            Array of values defining the level of scatter at the input abcissa.
            Default behavior will result in constant scatter at a level set in the 
            `~halotools.empirical_models.model_defaults` module. 
        """

        super(Moster13SmHm, self).__init__(
            galprop_key='stellar_mass', **kwargs)

        self.publications = ['arXiv:0903.4682', 'arXiv:1205.5807']

    def mean_stellar_mass(self, **kwargs):
        """ Return the stellar mass of a central galaxy as a function 
        of the input halo_table.  

        Parameters 
        ----------
        prim_haloprop : array, optional  
            Array of mass-like variable upon which occupation statistics are based. 
            If ``prim_haloprop`` is not passed, then ``halo_table`` keyword argument must be passed. 

        halo_table : object, optional  
            Data table storing halo catalog. 
            If ``halo_table`` is not passed, then ``prim_haloprop`` keyword argument must be passed. 

        redshift : float or array, optional 
            Redshift of the halo hosting the galaxy. 
            Default is set in `~halotools.sim_manager.sim_defaults`. 
            If passing an array, must be of the same length as 
            the ``prim_haloprop`` or ``halo_table`` argument. 

        Returns 
        -------
        mstar : array_like 
            Array containing stellar masses living in the input halo_table. 
        """

        # Retrieve the array storing the mass-like variable
        if 'halo_table' in kwargs.keys():
            mass = kwargs['halo_table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in kwargs.keys():
            mass = kwargs['prim_haloprop']
        else:
            raise KeyError("Must pass one of the following keyword arguments to mean_occupation:\n"
                "``halo_table`` or ``prim_haloprop``")

        if 'redshift' in kwargs.keys():
            redshift = kwargs['redshift']
        elif hasattr(self, 'redshift'):
            redshift = self.redshift
        else:
            redshift = sim_defaults.default_redshift

        # compute the parameter values that apply to the input redshift
        a = 1./(1+redshift)

        m1 = self.param_dict['m10'] + self.param_dict['m11']*(1-a)
        n = self.param_dict['n10'] + self.param_dict['n11']*(1-a)
        beta = self.param_dict['beta10'] + self.param_dict['beta11']*(1-a)
        gamma = self.param_dict['gamma10'] + self.param_dict['gamma11']*(1-a)

        # Calculate each term contributing to Eqn 2
        norm = 2.*n*mass
        m_by_m1 = mass/(10.**m1)
        denom_term1 = m_by_m1**(-beta)
        denom_term2 = m_by_m1**gamma

        mstar = norm / (denom_term1 + denom_term2)
        return mstar

    def retrieve_default_param_dict(self):
        """ Method returns a dictionary of all model parameters 
        set to the values in Table 1 of Moster et al. (2013). 

        Returns 
        -------
        d : dict 
            Dictionary containing parameter values. 
        """

        d = {
        'm10': 11.590, 
        'm11': 1.195, 
        'n10': 0.0351, 
        'n11': -0.0247, 
        'beta10': 1.376, 
        'beta11': -0.826, 
        'gamma10': 0.608, 
        'gamma11': 0.329
        }

        return d


class Behroozi10SmHm(PrimGalpropModel):
    """ Stellar-to-halo-mass relation based on 
    Behroozi et al. (2010), arXiv:1205.5807. 
    """

    def __init__(self, **kwargs):
        """
        Parameters 
        ----------
        prim_haloprop_key : string, optional  
            String giving the column name of the primary halo property governing stellar mass. 
            Default is set in the `~halotools.empirical_models.model_defaults` module. 

        scatter_model : object, optional  
            Class governing stochasticity of stellar mass. Default scatter is log-normal, 
            implemented by the `LogNormalScatterModel` class. 

        scatter_abcissa : array_like, optional  
            Array of values giving the abcissa at which
            the level of scatter will be specified by the input ordinates.
            Default behavior will result in constant scatter at a level set in the 
            `~halotools.empirical_models.model_defaults` module. 

        scatter_ordinates : array_like, optional  
            Array of values defining the level of scatter at the input abcissa.
            Default behavior will result in constant scatter at a level set in the 
            `~halotools.empirical_models.model_defaults` module. 
        """
        self.littleh = 0.7

        super(Behroozi10SmHm, self).__init__(
            galprop_key='stellar_mass', **kwargs)

        self.publications = ['arXiv:1001.0015']

    def retrieve_default_param_dict(self):
        """ Method returns a dictionary of all model parameters 
        set to the column 2 values in Table 2 of Behroozi et al. (2010). 

        Returns 
        -------
        d : dict 
            Dictionary containing parameter values. 
        """
        # All calculations are done internally using the same h=0.7 units 
        # as in Behroozi et al. (2010), so the parameter values here are 
        # the same as in Table 2, even though the mean_log_halo_mass and 
        # mean_stellar_mass methods use accept and return arguments in h=1 units. 

        d = {
        'm0_0': 10.72, 
        'm0_a': 0.59, 
        'm1_0': 12.35, 
        'm1_a': 0.3,
        'beta_0': 0.43,
        'beta_a': 0.18, 
        'delta_0': 0.56, 
        'delta_a': 0.18, 
        'gamma_0': 1.54,  
        'gamma_a': 2.52}

        return d

    def mean_log_halo_mass(self, log_stellar_mass, redshift=sim_defaults.default_redshift):
        """ Return the halo mass of a central galaxy as a function 
        of the stellar mass.  

        Parameters 
        ----------
        log_stellar_mass : array
            Array of base-10 logarithm of stellar masses in h=1 solar mass units. 

        redshift : float or array, optional 
            Redshift of the halo hosting the galaxy. If passing an array, 
            must be of the same length as the input ``log_stellar_mass``. 
            Default is set in `~halotools.sim_manager.sim_defaults`. 

        Returns 
        -------
        log_halo_mass : array_like 
            Array containing 10-base logarithm of halo mass in h=1 solar mass units. 
        """
        stellar_mass = (10.**log_stellar_mass)*(self.littleh**2)
        a = 1./(1. + redshift)

        logm0 = self.param_dict['m0_0'] + self.param_dict['m0_a']*(a - 1)
        m0 = 10.**logm0
        logm1 = self.param_dict['m1_0'] + self.param_dict['m1_a']*(a - 1)
        beta = self.param_dict['beta_0'] + self.param_dict['beta_a']*(a - 1)
        delta = self.param_dict['delta_0'] + self.param_dict['delta_a']*(a - 1)
        gamma = self.param_dict['gamma_0'] + self.param_dict['gamma_a']*(a - 1)

        stellar_mass_by_m0 = stellar_mass/m0
        term3_numerator = (stellar_mass_by_m0)**delta
        term3_denominator = 1 + (stellar_mass_by_m0)**(-gamma)

        log_halo_mass = logm1 + beta*np.log10(stellar_mass_by_m0) + (term3_numerator/term3_denominator) - 0.5

        return np.log10((10.**log_halo_mass)/self.littleh)

    def mean_stellar_mass(self, **kwargs):
        """ Return the stellar mass of a central galaxy as a function 
        of the input halo_table.  

        Parameters 
        ----------
        prim_haloprop : array, optional 
            Array of mass-like variable upon which occupation statistics are based. 
            If ``prim_haloprop`` is not passed, then ``halo_table`` keyword argument must be passed. 

        halo_table : object, optional 
            Data table storing halo catalog. 
            If ``halo_table`` is not passed, then ``prim_haloprop`` keyword argument must be passed. 

        redshift : float or array
            Redshift of the halo hosting the galaxy. If passing an array, 
            must be of the same length as the input ``stellar_mass``. 
            Default is set in `~halotools.sim_manager.sim_defaults`. 

        Returns 
        -------
        mstar : array_like 
            Array containing stellar masses living in the input halo_table, 
            in solar mass units assuming h = 1.
        """
        # Retrieve the array storing the mass-like variable
        if 'halo_table' in kwargs.keys():
            halo_mass = kwargs['halo_table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in kwargs.keys():
            halo_mass = kwargs['prim_haloprop']
        else:
            raise KeyError("Must pass one of the following keyword arguments to mean_occupation:\n"
                "``halo_table`` or ``prim_haloprop``")

        if 'redshift' in kwargs:
            redshift = kwargs['redshift']
        else:
            redshift = sim_defaults.default_redshift

        log_stellar_mass_table = np.linspace(8.5, 12.5, 100)
        log_halo_mass_table = self.mean_log_halo_mass(log_stellar_mass_table, redshift=redshift)

        interpol_func = model_helpers.custom_spline(log_halo_mass_table, log_stellar_mass_table)

        log_stellar_mass = interpol_func(np.log10(halo_mass))

        stellar_mass = 10.**log_stellar_mass

        return stellar_mass









