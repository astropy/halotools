# -*- coding: utf-8 -*-
"""
Module containing classes used to model the mapping between 
stellar mass and subhalos. 
"""
import numpy as np

from astropy.extern import six
from abc import ABCMeta, abstractmethod, abstractproperty

from . import model_defaults
from . import model_helpers as model_helpers

from ..utils.array_utils import array_like_length as custom_len
from ..sim_manager import sim_defaults 

from warnings import warn
from functools import partial

__all__ = ['PrimGalpropModel', 'Moster13SmHm', 'LogNormalScatterModel']

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
        prim_haloprop_key : string, optional keyword argument 
            String giving the column name of the primary halo property governing 
            the level of scatter. 
            Default is set in the `~halotools.empirical_models.model_defaults` module. 

        gal_type : string, optional keyword argument 
            Name of the galaxy population being modeled. Default is None. 

        scatter_abcissa : array_like, optional keyword argument 
            Array of values giving the abcissa at which
            the level of scatter will be specified by the input ordinates.
            Default behavior will result in constant scatter at a level set in the 
            `~halotools.empirical_models.model_defaults` module. 

        scatter_ordinates : array_like, optional keyword argument 
            Array of values defining the level of scatter at the input abcissa.
            Default behavior will result in constant scatter at a level set in the 
            `~halotools.empirical_models.model_defaults` module. 

        Examples 
        ---------
        >>> scatter_model = LogNormalScatterModel()
        >>> scatter_model = LogNormalScatterModel(prim_haloprop_key='mvir')

        To implement variable scatter, we need to define the level 
        of log-normal scatter at a set of control values 
        of the primary halo property. Here we give an example of a model 
        in which the scatter is 0.3 dex for Milky Way halos and 0.1 dex in cluster halos:

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

        if 'gal_type' in kwargs.keys():
            self.gal_type = kwargs['gal_type']
        self._initialize_param_dict()

        self._update_interpol()

    def mean_scatter(self, **kwargs):
        """ Return the amount of log-normal scatter that should be added 
        to the galaxy property as a function of the input halos. 

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

        Returns 
        -------
        scatter : array_like 
            Array containing the amount of log-normal scatter evaluated 
            at the input halos. 
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

        self._update_interpol()

        return self.spline_function(np.log10(mass))

    def scatter_realization(self, seed=None, **kwargs):
        """ Return the amount of log-normal scatter that should be added 
        to the galaxy property as a function of the input halos. 

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

        seed : int, optional keyword argument 
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
        define the level of scatter as a function of the input halos. 
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
    """ Abstract container class for models connecting halos to their primary
    galaxy property, e.g., stellar mass or luminosity. 
    """

    def __init__(self, galprop_key = 'stellar_mass', 
        prim_haloprop_key = model_defaults.default_smhm_haloprop, 
        scatter_model = LogNormalScatterModel, 
        **kwargs):
        """
        Parameters 
        ----------
        galprop_key : string, optional keyword argument 
            Name of the galaxy property being assigned. Default is ``stellar mass``, 
            though another common case may be ``luminosity``. 

        prim_haloprop_key : string, optional keyword argument 
            String giving the column name of the primary halo property governing 
            stellar mass.  
            Default is set in the `~halotools.empirical_models.model_defaults` module. 

        gal_type : string, optional keyword argument 
            Name of the galaxy population being modeled. Default is None. 

        scatter_model : object, optional keyword argument 
            Class governing stochasticity of stellar mass. Default scatter is log-normal, 
            implemented by the `LogNormalScatterModel` class. 

        redshift : float, optional keyword argument 
            Redshift of the stellar-to-halo-mass relation. Default is 0. 

        scatter_abcissa : array_like, optional keyword argument 
            Array of values giving the abcissa at which
            the level of scatter will be specified by the input ordinates.
            Default behavior will result in constant scatter at a level set in the 
            `~halotools.empirical_models.model_defaults` module. 

        scatter_ordinates : array_like, optional keyword argument 
            Array of values defining the level of scatter at the input abcissa.
            Default behavior will result in constant scatter at a level set in the 
            `~halotools.empirical_models.model_defaults` module. 

        new_haloprop_func_dict : function object, optional keyword argument 
            Dictionary of function objects used to create additional halo properties 
            that may be needed by the model component. 
            Used strictly by the `MockFactory` during call to the `process_halo_catalog` method. 
            Each dict key of ``new_haloprop_func_dict`` will 
            be the name of a new column of the halo catalog; each dict value is a function 
            object that returns a length-N numpy array when passed a length-N Astropy table 
            via the ``halos`` keyword argument. 
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
        if hasattr(self.scatter_model, 'gal_type'):
            self.gal_type = self.scatter_model.gal_type

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
            smhm_param_dict = self.retrieve_default_param_dict()
        else:
            smhm_param_dict = {}

        scatter_param_dict = self.scatter_model.param_dict

        self.param_dict = dict(
            smhm_param_dict.items() + 
            scatter_param_dict.items()
            )

    def _mc_galprop(self, include_scatter = True, **kwargs):
        """ Return the prim_galprop of the galaxies living in the input halos. 

        Parameters 
        ----------
        prim_haloprop : array, optional keyword argument 
            Array of mass-like variable governing the primary galaxy property. 
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

        redshift : float, optional keyword argument
            Redshift of the halo hosting the galaxy. 

        include_scatter : boolean, optional keyword argument 
            Determines whether or not the scatter model is applied to add stochasticity 
            to the galaxy property assignment. Default is True. 
            If False, model is purely deterministic, and the behavior is determined 
            by the ``mean_galprop`` method of the sub-class. 

        Returns 
        -------
        prim_galprop : array_like 
            Array storing the values of the primary galaxy property 
            of the galaxies living in the input halos. 
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
                self.scatter_model.scatter_realization(**kwargs)
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
        prim_haloprop_key : string, optional keyword argument 
            String giving the column name of the primary halo property governing 
            stellar mass. 
            Default is set in the `~halotools.empirical_models.model_defaults` module. 

        scatter_model : object, optional keyword argument 
            Class governing stochasticity of stellar mass. Default scatter is log-normal, 
            implemented by the `LogNormalScatterModel` class. 

        scatter_abcissa : array_like, optional keyword argument 
            Array of values giving the abcissa at which
            the level of scatter will be specified by the input ordinates.
            Default behavior will result in constant scatter at a level set in the 
            `~halotools.empirical_models.model_defaults` module. 

        scatter_ordinates : array_like, optional keyword argument 
            Array of values defining the level of scatter at the input abcissa.
            Default behavior will result in constant scatter at a level set in the 
            `~halotools.empirical_models.model_defaults` module. 

        scatter_spline_degree : int, optional keyword argument
            Degree of the spline interpolation for the case of interpol_method='spline'. 
            If there are k abcissa values specifying the model, input_spline_degree 
            is ensured to never exceed k-1, nor exceed 5. 

        input_param_dict : dict, optional keyword argument
            Dictionary containing values for the parameters specifying the model.
            If none is passed, the `Moster13SmHm` instance will be initialized to 
            the best-fit values taken from Moster et al. (2013). 
        """

        super(Moster13SmHm, self).__init__(
            galprop_key='stellar_mass', **kwargs)

        self.publications = ['arXiv:0903.4682', 'arXiv:1205.5807']

    def mean_stellar_mass(self, **kwargs):
        """ Return the stellar mass of a central galaxy as a function 
        of the input halos.  

        Parameters 
        ----------
        prim_haloprop : array, optional keyword argument 
            Array of mass-like variable governing stellar mass. 
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

        redshift : float, keyword argument
            Redshift of the halo hosting the galaxy

        input_param_dict : dict, optional
            dictionary of parameters governing the model. If not passed, 
            values bound to ``self`` will be chosen. 

        Returns 
        -------
        mstar : array_like 
            Array containing stellar masses living in the input halos. 
        """
        model_helpers.update_param_dict(self, **kwargs)

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

        if 'redshift' in kwargs.keys():
            redshift = kwargs['redshift']
        elif hasattr(self, 'redshift'):
            redshift = self.redshift
        else:
            redshift = sim_defaults.default_redshift

        # compute the parameter values that apply to the input redshift
        a = 1./(1+redshift)

        m1 = self.param_dict[self._m10_key] + self.param_dict[self._m11_key]*(1-a)
        n = self.param_dict[self._n10_key] + self.param_dict[self._n11_key]*(1-a)
        beta = self.param_dict[self._beta10_key] + self.param_dict[self._beta11_key]*(1-a)
        gamma = self.param_dict[self._gamma10_key] + self.param_dict[self._gamma11_key]*(1-a)

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

        If ``self`` has a ``gal_type``, then each dict key will 
        be appended with the ``gal_type`` string. This is useful 
        to protect composite models with multiple ``gal_types`` 
        from having repeated keys. 
        
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
        # If the Moster13SmHm model instance is part of a composite model 
        # with multiple galaxy types, then the instance has a gal_type 
        # attribute. Calling the _set_param_dict_key_attrs method 
        # creates a set of (private) attributes that allow us to 
        # access our param_dict in the same way regardless of whether the 
        # composite model discriminates between gal_type.  
        self._set_param_dict_key_attrs(d)

        # If the Moster13SmHm model instance is part of a composite model 
        # with multiple galaxy types, then we rename the param_dict keys 
        # to protect against possibly repeated 
        # keys in the param_dict of our composite model. 
        if hasattr(self, 'gal_type'):
            for oldkey in d.keys():
                newkey = oldkey + '_'+self.gal_type
                d[newkey] = d.pop(oldkey)

        return d

    def _set_param_dict_key_attrs(self, uncorrected_dict):
        """ Method binds to ``self`` one new private attribute 
        for each key of ``param_dict``. 
        For cases where ``self`` has no ``gal_type``, each private attribute 
        has the same name as its param_dict key. 
        For cases where ``self`` does have a ``gal_type``, 
        each private attribute has the same name as its param_dict key except 
        for the ``gal_type`` substring has been stripped from param_dict key. 
        This allows the `mean_stellar_mass` method to access the keys 
        of ``param_dict`` keys in the same fashion regardless of whether 
        ``self`` has a ``gal_type`` attribute, even though ``param_dict`` has
        different key names in these two cases. 
        """
        for key in uncorrected_dict.keys():
            attr_name = '_'+key+'_key'
            if hasattr(self, 'gal_type'):
                keyname = key + '_' + self.gal_type
            else:
                keyname = key
            setattr(self, attr_name, keyname)

