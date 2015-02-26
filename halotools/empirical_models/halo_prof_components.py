# -*- coding: utf-8 -*-
"""

halotools.halo_prof_components contains the classes and functions 
used by galaxy occupation models to control the intra-halo position 
of mock galaxies. 

"""

__all__ = ['HaloProfileModel','NFWProfile']

from astropy.extern import six
from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np
from scipy.interpolate import UnivariateSpline as spline

import functools

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
    """ Container class for any halo profile model. This is an abstract class, 
    and cannot itself be instantiated. Rather, HaloProfileModel provides a 
    blueprint for any radial profile component model used by the 
    empirical model factories such as `halotools.hod_factory`. 

    Parameters 
    ----------
    cosmology : object 
        astropy cosmology object

    redshift : float 

    Notes 
    -----
    For development purposes, object is temporarily hard-coded to only use z=0 Bryan & Norman 
    virial mass definition for standard LCDM cosmological parameter values.

    The first characters of any string used as a key 
    for a halo profile parameter must be host_haloprop_prefix, set in `~halotools.model_defaults`, 
    or else the resulting class will not correctly interface with the mock factory. 
    The two dictionaries using these keys are 
    prof_param_table_dict and halo_prof_func_dict, which are set by set_prof_param_table_dict and 
    set_halo_prof_func_dict, respectively. 
    """

    def __init__(self, cosmology, redshift, prof_param_keys, 
        prim_haloprop_key=model_defaults.haloprop_key_dict['prim_haloprop']):
        """
        Parameters 
        ----------
        cosmology : object 
            astropy cosmology object

        redshift : float 

        prim_haloprop_key : string, optional
            This string controls which column of the halo_table 
            is used as the primary halo property governing the 
            radial profile. Default is set in `halotools.model_defaults`. 
        """

        self.redshift = redshift
        self.cosmology = cosmology
        self.prim_haloprop_key = prim_haloprop_key

        self.prof_param_keys = list(prof_param_keys)

    @abstractmethod
    def density_profile(self, r, *args):
        """ Required method giving the density profile evaluated at the input radius. 

        Parameters 
        ----------
        r : array_like 
            Value of the radius at which density profile is to be evaluated. 
            Should be scaled by the halo boundary, so that :math:`0 < r < 1`

        args : array_like 
            Parameters specifying the halo profile. 
            If an array, should be of the same length 
            as the input r. 

        Returns 
        -------
        rho : array_like 
            Dark matter density evaluated at the input r. 
        """
        raise NotImplementedError("All halo profile models must include a density_profile method")

    @abstractmethod
    def cumulative_mass_PDF(self, r, *args):
        """ Required method specifying the cumulative PDF of the halo mass profile. 

        Parameters 
        ----------
        r : array_like 
            Value of the radius at which density profile is to be evaluated. 
            Should be scaled by the halo boundary, so that :math:`0 < r < 1`

        args : array_like 
            Parameters specifying the halo profile. 
            If an array, should be of the same length 
            as the input r. 

        Returns 
        -------
        cumu_mass_PDF : array_like 
            Cumulative fraction of dark matter mass interior to input r. 
        """
        raise NotImplementedError("All halo profile models must include a cumulative_mass_PDF method")

    @abstractmethod
    def set_halo_prof_func_dict(self,input_dict):
        """ Required method specifying the mapping between halo profile parameters 
        and some halo property (or properties). 
        The most common example halo profile parameter 
        is NFW concentration, and the simplest mapping is a power-law type
        concentration-mass relation. 

        The sole function of this method is to bind a dictionary to the 
        HaloProfileModel instance. 
        This dictionary standardizes the way composite models access 
        the profile parameter mappings, regardless of what the user names the methods. 
        The key(s) of the dictionary created by this method gives the name(s) of the 
        halo profile parameter(s) of the model; the value(s) of the dictionary are 
        function object(s) providing the mapping between halos and profile parameter(s).  
        When HaloProfileModel is called by the mock factories in `halotools.mock_factory`, 
        each dictionary key will correspond to the name of a new column for the halo catalog 
        that will be created by the mock factory during the pre-processing of the halo catalog.

        """
        raise NotImplementedError("All halo profile models must"
            " provide a dictionary with keys giving the names of the halo profile parameters, "
            " and values being the functions used to map parameter values onto halos")

    def get_param_key(self, model_nickname, param_nickname):
        param_key = model_nickname+'_'+param_nickname
        return param_key

    def cumulative_mass_PDF(self, r, *args):
        return 1



class TrivialProfile(HaloProfileModel):
    """ Profile of central galaxies residing at exactly the halo center. 
    """
    def __init__(self, 
        cosmology=sim_defaults.default_cosmology, redshift=sim_defaults.default_redshift,
        build_inv_cumu_table=True, prof_param_table_dict=None,
        prim_haloprop_key=model_defaults.haloprop_key_dict['prim_haloprop']):

        self.model_nickname = 'TrivialProfile'

        # Call the init constructor of the super-class, 
        # whose only purpose is to bind cosmology, redshift, prim_haloprop_key, 
        # and a list of prof_param_keys to the NFWProfile instance. 
        empty_list = []
        HaloProfileModel.__init__(self, 
            cosmology, redshift, empty_list, prim_haloprop_key)

        empty_dict = {}
        self.set_halo_prof_func_dict(empty_dict)
        self.set_prof_param_table_dict(empty_dict)

        self.publication = empty_list

    def density_profile(self, r, *args):
        return np.where(r == 0, 1, 0)

    def set_halo_prof_func_dict(self,input_dict):
        self.halo_prof_func_dict = input_dict

    def set_prof_param_table_dict(self,input_dict):
        self.prof_param_table_dict = input_dict



class NFWProfile(HaloProfileModel):
    """ NFW halo profile, based on Navarro, Frenk, and White (1999).

    Notes 
    -----
    For development purposes, object is temporarily hard-coded to only use  
    the Dutton & Maccio 2014 concentration-mass relation pertaining to 
    a virial mass definition of a dark matter halo. This should eventually be 
    generalized to allow for other concentration-mass relations, including those 
    that are dependent on cosmology, such as Diemer & Kravtsov 2014. 
    """

    def __init__(self, 
        cosmology=sim_defaults.default_cosmology, redshift=sim_defaults.default_redshift,
        build_inv_cumu_table=True, prof_param_table_dict=None,
        prim_haloprop_key=model_defaults.haloprop_key_dict['prim_haloprop'],
        conc_mass_relation_key = model_defaults.conc_mass_relation_key):
        """
        Parameters 
        ----------
        cosmology : object, optional
            astropy cosmology object. Default cosmology is WMAP5. 

        redshift : float, optional
            Default redshift is 0.

        build_inv_cumu_table : bool, optional
            If True, upon instantiation the __init__ constructor 
            will build a sequence of interpolation lookup tables 
            providing a mapping between :math:`x = r / R_{\\mathrm{vir}}` 
            and the unit-normalized cumulative mass profile function 
            :math:`\\rho_{NFW}(x | c)`. 
        """

        self.model_nickname = 'NFWmodel'
        # Call a method inherited from the super-class 
        # to assign a string that will be used to 
        # name the concentration parameter assigned to the halo catalog
        self._conc_parname = self.get_param_key(self.model_nickname, 'conc')

        # Call the init constructor of the super-class, 
        # whose only purpose is to bind cosmology, redshift, prim_haloprop_key, 
        # and a list of prof_param_keys to the NFWProfile instance. 
        HaloProfileModel.__init__(self, 
            cosmology, redshift, [self._conc_parname], prim_haloprop_key)

        conc_mass_func = self.get_conc_mass_model(conc_mass_relation_key)
        # Now bundle this function into self.halo_prof_func_dict
        self.set_halo_prof_func_dict({self._conc_parname:conc_mass_func})

        # Build a table stored in the dictionary prof_param_table_dict 
        # that dictates how to discretize the profile parameters
        self.set_prof_param_table_dict(input_dict=prof_param_table_dict)

        self.publication = ['arXiv:9611107','arXiv:1402.7073']

        if build_inv_cumu_table is True:
            self.build_inv_cumu_lookup_table(
                prof_param_table_dict=self.prof_param_table_dict)


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

    def cumulative_mass_PDF(self, r, c):
        """ Cumulative probability distribution of the NFW profile, 
        :math:`P^{NFW}( <r | c)`. 

        Parameters 
        ----------
        r : array_like 
            Value of the radius at which density profile is to be evaluated. 
            Should be scaled by the halo boundary, so that :math:`0 < r < 1`

        c : array_like 
            Concentration specifying the halo profile. If an array, should be of the same length 
            as the input r. 

        Returns 
        -------
        cumulative_PDF : array_like
            :math:`P^{NFW}(<r | c) = g(c) / g(c*r)`. 

        """
        return self.g(c) / self.g(r*c)

    def build_inv_cumu_lookup_table(self, prof_param_table_dict=None):
        """ Method used to create a lookup table of inverse cumulative mass 
        profile functions. Used by mock factories such as `~halotools.mock_factory` 
        to rapidly generate Monte Carlo realizations of satellite profiles. 

        Parameters 
        ----------
        prof_param_table_dict : dict, optional
            Dictionary providing instructions for how to generate a grid of 
            values for each halo profile parameter. Keys of this dictionary 
            are the profile parameter names; each value is a 3-element tuple 
            giving the minimum parameter value of the table to be built, the 
            maximum value, and the linear spacing. Default is None, 
            in which case the `NFWProfile.set_prof_param_table_dict` method 
            will control the grid values. 

            This method does not return anything. Instead, when called 
            the NFWProfile model instance will have two new attributes: 
            cumu_inv_param_table and cumu_inv_func_table. 
            The former is an array of NFW concentration parameter values, 
            the latter is an array of inverse cumulative density profile 
            function objects :math:`P^{NFW}( <r | c)` associated with 
            each concentration in the table. 
        """

        #Set up the grid used to tabulate inverse cumulative NFW mass profiles
        #This will be used to assign halo-centric distances to the satellites
        self.set_prof_param_table_dict(prof_param_table_dict)

        cmin, cmax, dconc = self.prof_param_table_dict[self._conc_parname]

        Npts_radius = model_defaults.default_Npts_radius_array  
        minrad = model_defaults.default_min_rad 
        maxrad = model_defaults.default_max_rad 
        radius_array = np.linspace(minrad,maxrad,Npts_radius)

        Npts_concen = int(np.round((cmax-cmin)/dconc))
        conc_array = np.linspace(cmin,cmax,Npts_concen)

        # After executing the following lines, 
        # self.cumu_inv_func_table will be an array of functions 
        # bound to the NFW profile instance.
        # The elements of this array are functions giving spline interpolations of the 
        # inverse cumulative mass of halos with different NFW concentrations.
        # Each function takes a scalar y in [0,1] as input, 
        # and outputs the x = r/Rvir corresponding to Prob_NFW( x < r/Rvir ) = y. 
        # Thus each array element is a function object. 
        cumu_inv_funcs = []
        for c in conc_array:
            cumu_inv_funcs.append(
                spline(self.cumulative_mass_PDF(radius_array,c),radius_array))
        self.cumu_inv_func_table = np.array(cumu_inv_funcs)
        self.cumu_inv_param_table = conc_array

    def set_halo_prof_func_dict(self, input_dict):
        """ Trivial required method whose sole design purpose is to 
        standardize the interface of halo profile models. 

        Parameters 
        ----------
        input_dict : dict 
            Each key corresponds to the name of a halo profile parameter, 
            e.g., 'halo_NFW_conc', which are set by the get_param_key 
            method if the super-class. The value attached to each key is a function object 
            providing the mapping between halos and the halo profile parameter, 
            such as a concentration-mass function. 

        Notes 
        ----- 
        Method does not return anything. Instead, input_dict is bound to 
        the NFWProfile instance with the attribute name halo_prof_func_dict. 
        """

        self.halo_prof_func_dict = input_dict

    def set_prof_param_table_dict(self,input_dict=None):
        """ Method sets the value of the prof_param_table_dict attribute. 
        The prof_param_table_dict attribute is a dictionary 
        used in the set up of a gridded correspondence between 
        halo profile properties and inverse cumulative function objects. 
        This grid is used by mock factories such as `halotools.mock_factory` 
        to rapidly generate Monte Carlo realizations of satellite profiles. 

        Parameters 
        ----------
        input_dict : dict, optional
            Each key corresponds to the name of a halo profile parameter, 
            e.g., 'halo_NFW_conc'. Each value is a 3-element tuple used 
            to govern how that parameter is gridded up by 
            `NFWProfile.build_inv_cumu_lookup_table`. 
            The entries of each tuple give the minimum parameter 
            value of the table to be built, the 
            maximum value, and the linear spacing.
            If None, default behavior is set in `halotools.model_defaults` module. 

        Notes 
        ----- 
        Method does not return anything. Instead, input_dict is bound to 
        the NFWProfile instance with the attribute name prof_param_table_dict. 

        """

        if input_dict is None:
            cmin = model_defaults.min_permitted_conc
            cmax = model_defaults.max_permitted_conc
            dconc = model_defaults.default_dconc
            self.prof_param_table_dict = (
                {self._conc_parname:(cmin, cmax, dconc)}
                )
        else:
            # Run some consistency checks on  
            # input_dict before binding it to the model instance
            if set(input_dict.keys()) != {self._conc_parname}:
                raise KeyError("The only permitted key of prof_param_table_dict "
                    " in the NFWProfile model is %s" % self._conc_parname)
            if not isinstance(input_dict[self._conc_parname], tuple):
                raise TypeError("Values of prof_param_table_dict must be a tuple")
            if len(input_dict[self._conc_parname]) != 3:
                raise TypeError("Tuple value of prof_param_table_dict " 
                    "must have exactly 3 elements")
            self.prof_param_table_dict = input_dict

    def get_conc_mass_model(self, conc_mass_relation_key):

        # Instantiate the container class for concentration-mass relations, 
        # defined in the external module halo_prof_param_components
        conc_mass_model_instance = halo_prof_param_components.ConcMass(
            cosmology = self.cosmology, redshift = self.redshift)

        # We want to call the specific function where the 'model' keyword argument 
        # is fixed to the conc-mass relation we want. 
        # For this, we use Python's functools
        conc_mass_func = functools.partial(
            conc_mass_model_instance.conc_mass, model=conc_mass_relation_key)

        return conc_mass_func

##################################################################################
















