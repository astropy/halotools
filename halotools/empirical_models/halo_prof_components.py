# -*- coding: utf-8 -*-
"""

halotools.halo_prof_components contains the classes and functions 
used by galaxy occupation models to control the intra-halo position 
of mock galaxies. 

"""

__all__ = ['HaloProfileModel','TrivialProfile','NFWProfile']

from astropy.extern import six
from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np
from scipy.interpolate import UnivariateSpline as spline

from functools import partial
from itertools import product

from ..utils.array_utils import array_like_length as aph_len
import occupation_helpers as occuhelp 

from ..sim_manager import sim_defaults

import astropy.cosmology as cosmology
from astropy import units as u

import model_defaults
import halo_prof_param_components


##################################################################################


@six.add_metaclass(ABCMeta)
class HaloProfileModel(object):
    """ Container class for any halo profile model. 

    This is an abstract class, and cannot itself be instantiated. 
    Rather, `HaloProfileModel` provides a 
    template for any radial profile component model used by the 
    empirical model factories such as `~halotools.empirical_models.hod_factory`. 

    Parameters 
    ----------
    cosmology : object 
        astropy cosmology object

    redshift : float 

    prof_param_keys : string, or list of strings
        Provides the names of the halo profile parameters of the model. 
        String entries are typically an underscore-concatenation 
        of the model nickname and parameter nickname, e.g., ``NFWmodel_conc``. 

    haloprop_key_dict : dict, optional
        Dictionary determining the halo properties used by the model. 
        Dictionary keys are, e.g., ``prim_haloprop_key``; 
        dictionary values are strings providing the column name 
        used to extract the relevant data from a halo catalog, e.g., ``mvir``. 
        Used by the method `_set_prof_param_table_dict`. 
        Default is an empty dict. 

    Notes 
    -----
    For development purposes, `HaloProfileModel` is temporarily 
    hard-coded to only use z=0 Bryan & Norman (1998)
    virial mass definition fitting function 
    for standard LCDM cosmological parameter values.

    """

    def __init__(self, cosmology, redshift, prof_param_keys, 
        haloprop_key_dict={}):

        self.redshift = redshift
        self.cosmology = cosmology
        self.haloprop_key_dict = haloprop_key_dict

        self.prof_param_keys = list(prof_param_keys)

    @abstractmethod
    def density_profile(self, x, *args):
        """ Required method giving the density profile evaluated at the input radius. 

        Parameters 
        ----------
        x : array_like 
            Value of the radius at which density profile is to be evaluated. 
            Should be scaled by the halo boundary, 
            so that :math:`x \equiv r / R_{\\mathrm{halo}}`, and 
            :math:`0 < x < 1`

        input_params : array_like 
            Parameters specifying the halo profile. 
            Should have the same length as the input x. 

        Returns 
        -------
        rho : array_like 
            Dark matter density evaluated at each value of the input array x. 
        """
        raise NotImplementedError("All subclasses of HaloProfileModel"
        " must include a density_profile method")

    @abstractmethod
    def cumulative_mass_PDF(self, x, *args):
        """ Required method specifying the cumulative PDF of the halo mass profile. 

        Parameters 
        ----------
        x : array_like 
            Value of the radius at which density profile is to be evaluated. 
            Should be scaled by the halo boundary, as in `density_profile`. 

        input_params : array_like 
            Parameters specifying the halo profile. 
            Should have the same length as the input x. 

        Returns 
        -------
        cumu_mass_PDF : array_like 
            Cumulative fraction of dark matter mass interior to input x. 
        """
        raise NotImplementedError("All sub-classes of HaloProfileModel "
            "must include a cumulative_mass_PDF method")

    @abstractproperty
    def halo_prof_func_dict(self):
        """ Required dictionary attribute containing function objects 
        that specify the mapping between halos and their profile parameters. 

        `halo_prof_func_dict` is a dictionary. 
        Each key of this dictionary is a profile parameter name, e.g., ``NFWmodel_conc``. 
        Each value of this dictionary is a function object; 
        these function objects take 
        halos as input, and return the profile parameter value, 
        e.g., `~halotools.empirical_models.halo_prof_param_components.ConcMass.conc_mass`.  
        All such mappings are found in the 
        `~halotools.empirical_models.halo_prof_param_components` module. 

        Notes 
        -----
        The ``halo_prof_func_dict`` dictionary can be empty, as is the case for `TrivialProfile`. 

        This dictionary standardizes the way composite models access 
        the profile parameter mappings. This standardization is necessary 
        so that composite models can use a uniform syntax to handle cases of 
        user-defined :math:`c(M)`-type relations whose method names 
        are not known in advance. 

        When instances of `HaloProfileModel` are called by the mock factories 
        such as `~halotools.empirical_models.HodMockFactory`, 
        each dictionary key of `halo_prof_func_dict` will correspond 
        to the name of a new column for the halo catalog 
        that will be created by the mock factory 
        during the pre-processing of the halo catalog.

        """
        raise NotImplementedError("All subclasses of HaloProfileModel must"
            " provide a halo_prof_func_dict attribute. \n"
            "This attribute is a dictionary with dict keys"
            " giving the names of the halo profile parameters, "
            " and dict values being the functions used to map "
            "profile parameter values onto halos")

    @abstractmethod
    def _set_prof_param_table_dict(self,input_dict):
        """ Required method providing instructions for how to discretize 
        halo profile parameter values. 

        After calling this method, the class instance has a 
        ``prof_param_table_dict`` attribute that is a dictionary. 
        Each dict key is a profile parameter name, e.g., ``NFWmodel_conc``. 
        Each dict value is a 3-element tuple; 
        the tuple entries provide, respectively, the min, max, and linear 
        spacing used to discretize the profile parameter. 
        This discretization is used by `build_inv_cumu_lookup_table` to 
        create a lookup table associated with `HaloProfileModel`. 

        Parameters 
        ----------
        input_dict : dict, optional
            Passed dictionary used to manually determine how to discretize 
            a halo profile parameter. 
            All keys of ``input_dict`` other than those already in 
            ``self.prof_param_table_dict`` will be ignored. 
            The value bound to each key must be a 3-element tuple,
            which will be used to govern how the halo profile parameter is discretized. 
            The entries of the tuple give the minimum parameter 
            value of the table to be built, the 
            maximum value, and the linear parameter spacing, respectively. 
            If ``input_dict`` is an empty dict, 
            the default discretization will be chosen, 
            which is set in the `halotools.empirical_models.model_defaults` module. 
            The same applies for the keys of ``self.prof_param_table_dict`` 
            with no match in ``input_dict``.   

        Notes 
        -----
        The ``prof_param_table_dict`` dictionary can be empty, 
        as is the case for `TrivialProfile`. 

        """ 
        raise NotImplementedError("All subclasses of HaloProfileModel must"
            " provide a _set_prof_param_table_dict method used to create a "
            "(possibly trivial) dictionary prof_param_table_dict.")

    def build_inv_cumu_lookup_table(self, prof_param_table_dict={}, 
        profile_table_radius_array_dict = model_defaults.profile_table_radius_array_dict):
        """ Method used to create a lookup table of inverse cumulative mass 
        profile functions. 

        `build_inv_cumu_lookup_table` does not return anything. 
        Instead, when called, the class instance will have two new attributes: 

            * ``cumu_inv_param_table``, an array (or arrays) of discretized profile parameter values.

            * ``cumu_inv_func_table``, an array of inverse cumulative density profile function objects, :math:`P( <x | p)`, associated with each point in the grid of profile parameters. 

        The function objects in the ``cumu_inv_func_table`` lookup table are computed 
        by `cumulative_mass_PDF`. 

        Parameters 
        ----------
        prof_param_table_dict : dict, optional
            Dictionary created by `_set_prof_param_table_dict` 
            providing instructions for how to generate a grid of 
            values for each halo profile parameter. 
            Default is an empty dict. 

        Notes 
        ----- 
        Used by mock factories such as `~halotools.empirical_models.HodMockFactory` 
        to rapidly generate Monte Carlo realizations of intra-halo positions. 

        """
        
        self._set_prof_param_table_dict(prof_param_table_dict)

        npts_radius = profile_table_radius_array_dict['npts']
        rmin = profile_table_radius_array_dict['rmin']
        rmax = profile_table_radius_array_dict['rmax']
        radius_array = np.linspace(rmin,rmax,npts_radius)

        self.cumu_inv_param_table_dict = {}
        param_array_list = []
        for prof_param_key in self.prof_param_keys:
            parmin, parmax, dpar = self.prof_param_table_dict[prof_param_key]
            npts_par = int(np.round((parmax-parmin)/dpar))
            param_array = np.linspace(parmin,parmax,npts_par)
            param_array_list.append(param_array)
            self.cumu_inv_param_table_dict[prof_param_key] = param_array

        
        # Using the itertools product method requires 
        # special handling of the length-zero edge case
        if len(param_array_list) == 0:
            self.cumu_inv_func_table = np.array([])
        else:
            func_table = []
            for items in product(*param_array_list):
                funcobj = spline(self.cumulative_mass_PDF(radius_array,*items),radius_array)
                func_table.append(funcobj)

            param_array_dimensions = [len(param_array) for param_array in param_array_list]
            func_table = np.array(func_table).reshape(param_array_dimensions)
            self.cumu_inv_func_table = func_table
            func_table_indices = (
                np.arange(np.prod(param_array_dimensions)).reshape(param_array_dimensions)
                )
            self.func_table_indices = func_table_indices



    def _get_param_key(self, model_nickname, param_nickname):
        """ Trivial function providing standardized names for halo profile parameters. 

        Parameters 
        ----------
        model_nickname : string 
            Each sub-class of `HaloProfileModel` has a nickname, e.g., ``NFWmodel``. 

        param_nickname : string 
            Each profile parameter has a nickname, e.g., ``conc``. 

        Returns 
        -------
        param_key : string 
            Underscore-concatenation of the two inputs, e.g., ``NFWmodel_conc``. 
        """
        param_key = model_nickname+'_'+param_nickname
        return param_key


class TrivialProfile(HaloProfileModel):
    """ Profile of dark matter halos with all their mass 
    concentrated at exactly the halo center. 

    Primarily used as a dummy class to assign 
    positions to central-type galaxies. 

    Parameters 
    ----------
    cosmology : object 
        Astropy cosmology object. Default cosmology is WMAP5. 

    redshift : float 

    Notes 
    -----
    Testing done by `~halotools.empirical_models.test_empirical_models.test_TrivialProfile`

    """
    def __init__(self, 
        cosmology=sim_defaults.default_cosmology, 
        redshift=sim_defaults.default_redshift):

        self.model_nickname = 'TrivialProfile'

        haloprop_key_dict = {}
        prof_param_keys = []

        # Call the init constructor of the super-class, 
        # whose only purpose is to bind cosmology, redshift, haloprop_key_dict, 
        # and a list of prof_param_keys to the NFWProfile instance. 

        super(TrivialProfile, self).__init__(
            cosmology, redshift, prof_param_keys, 
            haloprop_key_dict=haloprop_key_dict)

        empty_dict = {}
        #self._set_prof_param_table_dict(empty_dict)
        self.build_inv_cumu_lookup_table(empty_dict)

        self.publication = []

    def density_profile(self, x):
        """ Trivial density profile function. 

        :math:`\\rho(x=0) = 1`

        :math:`\\rho(x>0) = 0`

        Parameters 
        ----------
        x : array_like 
            Value of the radius at which density profile is to be evaluated. 
            Should be scaled by the halo boundary, 
            so that :math:`x \equiv r / R_{\\mathrm{halo}}`, and 
            :math:`0 < x < 1`

        Returns 
        -------
        rho : array_like 
            Dark matter density evaluated at each value of the input array x. 

        """
        return np.where(r == 0, 1, 0)

    @property
    def halo_prof_func_dict(self):
        """ Trivial method binding the empty dictionary ``halo_prof_func_dict`` 
        to the class instance. 

        Method is required of any `HaloProfileModel` sub-class. 
        For `TrivialProfile`, the ``halo_prof_func_dict`` is empty because in this case 
        there are no profile parameters that need to be mapped onto halos. 
        """
        return {}

    def _set_prof_param_table_dict(self,input_dict):
        """ Trivial method binding the empty dictionary 
        ``prof_param_table_dict`` to the class instance. 

        Method is required of any `HaloProfileModel` sub-class. 
        For `TrivialProfile`, the ``prof_param_table_dict`` is empty 
        because in this case there are no profile parameters 
        for which a lookup table needs to be built. 
        """
        self.prof_param_table_dict = input_dict

    def cumulative_mass_PDF(self, x):
        """ Trivial function returning unity for any input. 

        Method is required of any `HaloProfileModel` sub-class.         
        """
        return 1


class NFWProfile(HaloProfileModel):
    """ NFW halo profile, based on Navarro, Frenk, and White (1999).

    Parameters 
    ----------
    cosmology : object, optional
        Astropy cosmology object. Default cosmology is WMAP5. 

    redshift : float, optional
        Default redshift is 0.

    prof_param_table_dict : dictionary, optional 
        Dictionary governing how the `build_inv_cumu_lookup_table` method 
        discretizes the concentration parameter when building the profile lookup table. 

        The ``prof_param_table_dict`` dictionary has a single key giving the 
        concentration parameter name, ``NFWmodel_conc``. 
        The value bound to this key is a 3-element tuple; 
        the tuple entries provide, respectively, the min, max, and linear 
        spacing used to discretize halo concentration. 

        The `_set_prof_param_table_dict` method binds this dictionary to 
        the class instance; if no ``prof_param_table_dict`` argument 
        is passed to the constructor, 
        the discretization will be determined by the default settings in 
        the `~halotools.empirical_models.model_defaults` module.  

    haloprop_key_dict : dict, optional
        Dictionary determining the halo properties used by the model. 
        Dictionary keys must include ``prim_haloprop_key`` 
        and ``halo_boundary``. 
        Dictionary values are strings providing the column name 
        used to extract the relevant data from a halo catalog, 
        e.g., ``mvir`` and ``rvir``. 
        ``haloprop_key_dict`` is used by the method 
        `_set_prof_param_table_dict`. 
        Default values are set in `~halotools.empirical_models.model_defaults`. 

    conc_mass_relation_key : string, optional 
        String specifying which concentration-mass relation is used to paint model 
        concentrations onto simulated halos. Passed to the 
        `~halotools.empirical_models.halo_prof_param_components.ConcMass.conc_mass` 
        method of the `~halotools.empirical_models.halo_prof_param_components.ConcMass` 
        container class that stores the :math:`c(M)` supported relations. 
        Default string/model is set in `~halotools.empirical_models.model_defaults`.


    Notes 
    -----
    For development purposes, object is temporarily hard-coded to only use  
    the Dutton & Maccio 2014 concentration-mass relation pertaining to 
    a virial mass definition of a dark matter halo. This should eventually be 
    generalized to allow for cosmology-dependent concentration-mass relations, 
    such as Bhattacharya et al. (2013), and Diemer & Kravtsov (2014). 

    For a review of basic properties of the NFW profile, 
    see for example Lokas & Mamon (2000), arXiv:0002395. 
    """

    def __init__(self, 
        cosmology=sim_defaults.default_cosmology, 
        redshift=sim_defaults.default_redshift,
        prof_param_table_dict={},
        haloprop_key_dict=model_defaults.haloprop_key_dict,
        conc_mass_relation_key = model_defaults.conc_mass_relation_key):

        self.model_nickname = 'NFWmodel'

        # Call a method inherited from the super-class 
        # to assign a string that will be used to 
        # name the concentration parameter assigned to the halo catalog
        self._conc_parname = self._get_param_key(self.model_nickname, 'conc')

        # Call the init constructor of the super-class, 
        # whose only purpose is to bind cosmology, redshift, prim_haloprop_key, 
        # and a list of prof_param_keys to the NFWProfile instance. 
        super(NFWProfile, self).__init__(
            cosmology, redshift, [self._conc_parname], haloprop_key_dict=haloprop_key_dict)

        self._conc_mass_func = self._get_conc_mass_model(conc_mass_relation_key)

        self.build_inv_cumu_lookup_table(
            prof_param_table_dict=prof_param_table_dict)

        self.publication = ['arXiv:9611107','arXiv:1402.7073']

    def g(self, x):
        """ Convenience function used to evaluate the profile. 

        Parameters 
        ----------
        x : array_like 

        Returns 
        -------
        g : array_like 
            :math:`1 / g(x) = \\log(1+x) - x / (1+x)`
        """
        denominator = np.log(1.0+x) - (x/(1.0+x))
        return 1./denominator

    def rho_s(self, c):
        """ Normalization of the NFW profile. 

        Parameters 
        ----------
        c : array_like
            concentration of the profile

        Returns 
        -------
        rho_s : array_like 
            Profile normalization 
            :math:`\\rho_{s}^{NFW} = \\frac{1}{3}\\Delta_{vir}c^{3}g(c)\\bar{\\rho}_{m}`

        """
        return (self.delta_vir/3.)*c*c*c*self.g(c)*self.cosmic_matter_density

    def density_profile(self, r, c):
        """ Mass density profile given by 
        :math:`\\rho^{NFW}(r | c) = \\rho_{s}^{NFW} / cr(1+cr)^{2}`

        Parameters 
        ----------
        r : array_like 
            Value of the radius at which density profile is to be evaluated. 
            Should be scaled by the halo boundary, so that :math: `0 < r < 1`

        c : array_like 
            Concentration specifying the halo profile. 
            If an array, should be of the same length 
            as the input r. 

        Returns 
        -------
        result : array_like 
            NFW density profile :math:`\\rho^{NFW}(r | c)`.
        """
        numerator = self.rho_s(c)
        denominator = (c*r)*(1.0 + c*r)*(1.0 + c*r)
        return numerator / denominator

    def cumulative_mass_PDF(self, r, *args):
        """ Cumulative probability distribution of the NFW profile, 
        :math:`P^{NFW}( <r | c)`. 

        Parameters 
        ----------
        r : array_like 
            Value of the radius at which density profile is to be evaluated. 
            Should be scaled by the halo boundary, so that :math:`0 < r < 1`

        c : array_like 
            Concentration specifying the halo profile. 
            If an array, should be of the same length 
            as the input r. 

        Returns 
        -------
        cumulative_PDF : array_like
            :math:`P^{NFW}(<r | c) = g(c) / g(c*r)`. 

        """
        if len(args)==0:
            raise SyntaxError("Must pass array of concentrations to cumulative_mass_PDF. \n"
                "Only received array of radii.")
        else:
            return self.g(args[0]) / self.g(r*args[0])


    @property 
    def halo_prof_func_dict(self):
        """ Dictionary used as a container for 
        the functions that map profile parameter values onto dark matter halos. 

        Each dict key of ``halo_prof_func_dict`` corresponds to 
        the name of a halo profile parameter, e.g., 'NFWmodel_conc'. 
        The dict value attached to each dict key is a function object
        providing the mapping between halos and the halo profile parameter, 
        such as a concentration-mass function. 

        Notes 
        ----- 
        Implemented as a read-only getter method via the ``@property`` decorator syntax. 
        """
        return {self._conc_parname : self._conc_mass_func}


    def _set_prof_param_table_dict(self,input_dict={}):
        """ Method sets the value of the prof_param_table_dict attribute. 

        The prof_param_table_dict attribute is a dictionary 
        used in the set up of a gridded correspondence between 
        halo profile properties and inverse cumulative function objects. 
        This grid is used by mock factories such as `halotools.empirical_models.mock_factory` 
        to rapidly generate Monte Carlo realizations of satellite profiles. 

        Parameters 
        ----------
        input_dict : dict, optional
            Passed dictionary used to manually determine how to discretize 
            a halo profile parameter. 
            All dict keys of ``input_dict`` besides ``NFWmodel_conc`` will be ignored. 
            The dict value bound to the ``NFWmodel_conc`` key must be a 3-element tuple,
            which will be used to govern how halo concentration is discretized. 
            The entries of the tuple give the minimum parameter 
            value of the table to be built, the 
            maximum value, and the linear spacing, respectively. 
            If an empty dict, or if no ``NFWmodel_conc`` key is present, 
            default behavior is set in the `halotools.empirical_models.model_defaults` module. 

        Notes 
        ----- 
        Method does not return anything. Instead, input_dict is bound to 
        the NFWProfile instance with the attribute name prof_param_table_dict. 

        """

        # First declare the what prof_param_table_dict will be 
        # in the absence of any input_dict
        cmin = model_defaults.min_permitted_conc
        cmax = model_defaults.max_permitted_conc
        dconc = model_defaults.default_dconc
        self.prof_param_table_dict = {self._conc_parname:(cmin, cmax, dconc)}

        for key in self.prof_param_table_dict.keys():

            if key in input_dict.keys():
                if not isinstance(input_dict[key], tuple):
                    raise TypeError("Values of input_dict passed to "
                        "_set_prof_param_table_dict must be a tuple")

                if len(input_dict[key]) != 3:
                    raise TypeError("Tuple value of "
                        "input_dict passed to _set_prof_param_table_dict" 
                        "must have exactly 3 elements")

                self.prof_param_table_dict[key] = input_dict[key]


    def _get_conc_mass_model(self, conc_mass_relation_key):

        # Instantiate the container class for concentration-mass relations, 
        # defined in the external module halo_prof_param_components
        conc_mass_model_instance = halo_prof_param_components.ConcMass(
            cosmology = self.cosmology, redshift = self.redshift)

        # We want to call the specific function where the 'model' keyword argument 
        # is fixed to the conc-mass relation we want. 
        # For this, we use Python's functools
        conc_mass_func = partial(
            conc_mass_model_instance.conc_mass, model=conc_mass_relation_key)

        return conc_mass_func

##################################################################################
















