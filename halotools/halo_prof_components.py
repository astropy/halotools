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

from utils.array_utils import array_like_length as aph_len
import occupation_helpers as occuhelp 
import defaults

import astropy.cosmology as cosmology
from astropy import units as u

import defaults

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

    """

    def __init__(self, cosmology, redshift, prim_haloprop_key='MVIR'):
        """
        Parameters 
        ----------
        cosmology : object 
            astropy cosmology object

        redshift : float 

        prim_haloprop_key : string, optional
            This string controls which column of the halo_table 
            is used as the primary halo property governing the 
            radial profile. Default is 'MVIR'. 
        """

        self.redshift = redshift
        self.cosmology = cosmology
        self.prim_haloprop_key = prim_haloprop_key

        #littleh = self.cosmology.H0/100.0
        #crit_density = (self.cosmology.critical_density(0).to(u.Msun/u.Mpc**3)/littleh**2)
        #self.cosmic_matter_density = crit_density*self.cosmology.Om0

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
        """
        raise NotImplementedError("All halo profile models must include a mass_density method")

    @abstractmethod
    def cumulative_mass_PDF(self, r, *args):
        """ Required method specifying the cumulative PDF of the halo mass profile. 

        Parameters 
        ----------
        r : array_like 
            Value of the radius at which the 
            cumulative profile is to be evaluated. 

        args : array_like 
            Parameters specifying the halo profile. 
            If an array, should be of the same length 
            as the input r. 

        """
        raise NotImplementedError("All halo profile models must include a cumulative_mass_PDF method")

    @abstractmethod
    def set_param_func_dict(self,input_dict):
        """ Required method specifying the mapping between halo profile parameters 
        and some halo property (or properties). 
        The most common example halo profile parameter 
        is NFW concentration, and the most common mapping is a power-law type
        concentration-mass relation. 

        The sole function of this method is to bind a dictionary to the 
        HaloProfileModel instance. The purpose of this dictionary 
        is to provide a standardized way that composite models can access 
        the halo-parameter mappings, regardless of what the user names the methods. 
        The key(s) of the dictionary created by this method gives the name(s) of the 
        halo profile parameter(s) of the model; the value(s) of the dictionary are 
        function object(s) providing the mapping between halos and profile parameter(s).  
        When HaloProfileModel is called by mock factories such as `halotools.mock_factory`, 
        each dictionary key will correspond to the name of a new column for halo_table
        that will be created by the mock factory during the pre-processing of the halo catalog.

        By strongly suggested convention, the dictionary keys giving the 
        names of halo profile parameters should begin with 'halo_prof_model_'. 
        This will make clear the interpretation of the new columns of the halo_table 
        created by the mock factories, and helps ensure that the no existing 
        columns of a halo_table will be over-written. 

        """
        raise NotImplementedError("All halo profile models must"
            " provide a dictionary with keys giving the names of the halo profile parameters, "
            " and values being the functions used to map parameter values onto halos")


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
        cosmology=cosmology.WMAP5, redshift=0.0,
        build_inv_cumu_table=True, prof_param_table_dict=None,
        prim_haloprop_key='MVIR'):
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
        # Call the init constructor of the super-class, 
        # whose only purpose is to bind cosmology, redshift, and prim_haloprop_key
        # to the NFWProfile instance. 
        HaloProfileModel.__init__(self, cosmology, redshift, prim_haloprop_key)


        self.set_param_func_dict({'halo_prof_model_conc':self.conc_mass})
        self.set_prof_param_table_dict(input_dict=prof_param_table_dict)

        self.publication = ['arXiv:9611107','arXiv:1402.7073']

        if build_inv_cumu_table is True:
            self.build_inv_cumu_lookup_table(
                prof_param_table_dict=self.prof_param_table_dict)

    def conc_mass(self, mass):
        """ Power-law fit to the concentration-mass relation from 
        Dutton & Maccio 2014, MNRAS 441, 3359, arXiv:1402.7073.

        Parameters 
        ----------
        mass : array_like 
            Input array of halo masses. 

        Returns 
        -------
        c : array_like
            Concentrations of the input halos. 

        Notes 
        -----
        This model was only calibrated for the Planck 1-year cosmology.

        Model assumes that halo mass definition is Mvir.
        """

        a = 0.537 + (1.025 - 0.537) * np.exp(-0.718 * self.redshift**1.08)
        b = -0.097 + 0.024 * self.redshift

        logc = a + b * np.log10(mass / 1.E12)
        c = 10**logc

        return c


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
            the NFWProfile instance will have two new attributes: 
            cumu_inv_conc_table and cumu_inv_func_table. 
            The former is an array of NFW concentration parameter values, 
            the latter is an array of inverse cumulative density profile 
            function objects :math:`P^{NFW}( <r | c)` associated with 
            each concentration in the table. 
        """

        #Set up the grid used to tabulate inverse cumulative NFW mass profiles
        #This will be used to assign halo-centric distances to the satellites
        self.set_prof_param_table_dict(prof_param_table_dict)

        cmin, cmax, dconc = self.prof_param_table_dict['halo_prof_model_conc']

        Npts_radius = defaults.default_Npts_radius_array  
        minrad = defaults.default_min_rad 
        maxrad = defaults.default_max_rad 
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
        self.cumu_inv_conc_table = conc_array

    def set_param_func_dict(self, input_dict):
        """ Trivial required method whose sole design purpose is to 
        standardize the interface of halo profile models. 

        Parameters 
        ----------
        input_dict : dict 
            Each key corresponds to the name of a halo profile parameter, 
            e.g., 'halo_prof_model_conc'. Each value is a function object 
            providing the mapping between halos and the halo profile parameter, 
            such as a concentration-mass function. 

        Notes 
        ----- 
        Method does not return anything. Instead, input_dict is bound to 
        the NFWProfile instance with the attribute name param_func_dict. 
        """

        self.param_func_dict = input_dict

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
            e.g., 'halo_prof_model_conc'. Each value is a 3-element tuple used 
            to govern how that parameter is gridded up by 
            `NFWProfile.build_inv_cumu_lookup_table`. 
            The entries of each tuple give the minimum parameter 
            value of the table to be built, the 
            maximum value, and the linear spacing.
            If None, default behavior is set in `halotools.defaults` module. 

        Notes 
        ----- 
        Method does not return anything. Instead, input_dict is bound to 
        the NFWProfile instance with the attribute name prof_param_table_dict. 

        """

        if input_dict is None:
            cmin = defaults.min_permitted_conc
            cmax = defaults.max_permitted_conc
            dconc = defaults.default_dconc
            self.prof_param_table_dict = (
                {'halo_prof_model_conc':(cmin, cmax, dconc)})
        else:
            # Run some tests on the input_dict before binding it to the model instance
            if set(input_dict.keys()) != {'halo_prof_model_conc'}:
                raise KeyError("The only prof_param_table_dict key of "
                    " the NFWProfile model is 'halo_prof_model_conc'")
            if not isinstance(input_dict['halo_prof_model_conc'], tuple):
                raise TypeError("Value of prof_param_table_dict['halo_prof_model_conc'] must be a tuple instance")
            if len(input_dict['halo_prof_model_conc']) != 3:
                raise TypeError("Tuple value of prof_param_table_dict must have exactly 3 elements")
            self.prof_param_table_dict = input_dict



##################################################################################
















