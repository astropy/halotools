# -*- coding: utf-8 -*-
"""
This module contains the classes and functions 
used by galaxy occupation models to control the 
intra-halo position of mock galaxies. 
"""

__all__ = ['HaloProfileModel','TrivialProfile','NFWProfile']

from abc import ABCMeta, abstractmethod, abstractproperty
from functools import partial
from itertools import product

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline

from astropy.extern import six
from astropy import cosmology
from astropy import units as u

from . import model_helpers 
from . import model_defaults
from . import halo_prof_param_components

from ..utils.array_utils import array_like_length as custom_len
from ..sim_manager import sim_defaults


##################################################################################

@six.add_metaclass(ABCMeta)
class HaloProfileModel(object):
    """ Container class for any halo profile model. 

    This is an abstract class, and cannot itself be instantiated. 
    Rather, `HaloProfileModel` provides a template for any model of 
    the radial profile of dark matter particles within their halos. 

    """

    def __init__(self, halo_boundary = model_defaults.halo_boundary, 
        prim_haloprop_key = model_defaults.prim_haloprop_key, **kwargs):
        """
        Parameters 
        ----------
        prof_param_keys : string, or list of strings
            Provides the names of the halo profile parameters of the model. 
            String entries are typically an underscore-concatenation 
            of the model nickname and parameter nickname, e.g., ``NFWmodel_conc``. 

        halo_boundary : string, optional keyword argument 
            String giving the column name of the halo catalog that stores the boundary of the halo. 
            Default is set in the `~halotools.empirical_models.model_defaults` module. 

        prim_haloprop_key : string, optional keyword argument 
            String giving the column name of the primary halo property governing 
            the occupation statistics of gal_type galaxies. 
            Default is set in `~halotools.empirical_models.sim_defaults`.

        cosmology : object, optional keyword argument
            Astropy cosmology object. Default is None.

        redshift : float, optional keyword argument 
            Default is None. 

        """

        self.halo_boundary = halo_boundary
        self.prim_haloprop_key = prim_haloprop_key

        required_kwargs = ['prof_param_keys']
        model_helpers.bind_required_kwargs(required_kwargs, self, **kwargs)

        if 'redshift' in kwargs.keys():
            self.redshift = kwargs['redshift']

        if 'cosmology' in kwargs.keys():
            self.cosmology = kwargs['cosmology']

    def build_inv_cumu_lookup_table(self, 
        logrmin = model_defaults.default_lograd_min, 
        logrmax = model_defaults.default_lograd_max, 
        Npts_radius_table=model_defaults.Npts_radius_table):
        """ Method used to create a lookup table of inverse cumulative mass 
        profile functions. 

        Parameters 
        ----------
        logrmin : float, optional 
            Minimum radius used to build the spline table. 
            Default is set in `~halotools.empirical_models.model_defaults`. 

        logrmax : float, optional 
            Maximum radius used to build the spline table
            Default is set in `~halotools.empirical_models.model_defaults`. 

        Npts_radius_table : int, optional 
            Number of control points used in the spline. 
            Default is set in `~halotools.empirical_models.model_defaults`. 

        Notes 
        ----- 

            * Used by mock factories such as `~halotools.empirical_models.HodMockFactory` to rapidly generate Monte Carlo realizations of intra-halo positions. 

            * As tested in `~halotools.empirical_models.test_empirical_models.test_halo_prof_components`, for the case of a `~halotools.empirical_models.NFWProfile`, errors due to interpolation from the lookup table are below 0.1 percent at all relevant radii and concentration. 

            * The interpolation is done in log-space. Thus each function object stored in ``cumu_inv_func_table`` operates on :math:`\\log_{10}\\mathrm{P}`, and returns :math:`\\log_{10}r`, where :math:`\\mathrm{P} = \\mathrm{P}_{\\mathrm{NFW}}( < r | c )`, computed by the `cumulative_mass_PDF` method. 

        """
        
        radius_array = np.logspace(logrmin,logrmax,Npts_radius_table)
        logradius_array = np.log10(radius_array)

        param_array_list = []
        for prof_param_key in self.prof_param_keys:
            parmin = getattr(self, prof_param_key + '_lookup_table_min')
            parmax = getattr(self, prof_param_key + '_lookup_table_max')
            dpar = getattr(self, prof_param_key + '_lookup_table_spacing')
            npts_par = int(np.round((parmax-parmin)/dpar))
            param_array = np.linspace(parmin,parmax,npts_par)
            param_array_list.append(param_array)
            setattr(self, prof_param_key + '_lookup_table_bins', param_array)
        
        # Using the itertools product method requires 
        # special handling of the length-zero edge case
        if len(param_array_list) == 0:
            self.cumu_inv_func_table = np.array([])
            self.func_table_indices = np.array([])
        else:
            func_table = []
            for items in product(*param_array_list):
                table_ordinates = self.cumulative_mass_PDF(radius_array,*items)
                log_table_ordinates = np.log10(table_ordinates)
                funcobj = spline(log_table_ordinates, logradius_array, k=4)
                func_table.append(funcobj)

            param_array_dimensions = [len(param_array) for param_array in param_array_list]
            self.cumu_inv_func_table = np.array(func_table).reshape(param_array_dimensions)
            self.func_table_indices = (
                np.arange(np.prod(param_array_dimensions)).reshape(param_array_dimensions)
                )


class TrivialProfile(HaloProfileModel):
    """ Profile of dark matter halos with all their mass concentrated at exactly the halo center. 

    This class has virtually no functionality on its own. It is primarily used 
    as a dummy class to assign positions to central-type galaxies. 

    """
    def __init__(self, **kwargs):
        """
        Notes 
        -----
        Testing done by `~halotools.empirical_models.test_empirical_models.test_TrivialProfile`

        Examples 
        --------
        You can load a trivial profile model with the default settings simply by calling 
        the class constructor with no arguments:

        >>> trivial_halo_prof_model = TrivialProfile()

        Use the keyword arguments for ``cosmology`` and ``redshift`` to load profiles 
        with alternative settings:

        >>> from astropy.cosmology import Planck13
        >>> trivial_halo_prof_model = TrivialProfile(cosmology = Planck13, redshift = 0.5)
        """

        super(TrivialProfile, self).__init__(prof_param_keys=[], **kwargs)

        self.build_inv_cumu_lookup_table()

        self.publications = []

class NFWProfile(HaloProfileModel):
    """ NFW halo profile, based on Navarro, Frenk, and White (1999).

    """

    def __init__(self, 
        cosmology=sim_defaults.default_cosmology, 
        redshift=sim_defaults.default_redshift,
        halo_boundary=model_defaults.halo_boundary,
        conc_mass_model = model_defaults.conc_mass_model, **kwargs):
        """
        Parameters 
        ----------
        prim_haloprop_key : string, optional keyword argument 
            String giving the column name of the primary halo property governing 
            the occupation statistics of gal_type galaxies. 
            Default is set in `~halotools.empirical_models.sim_defaults`.

        conc_mass_model : string, optional keyword argument 
            Specifies the calibrated fitting function used to model the concentration-mass relation. 
             Default is set in `~halotools.empirical_models.sim_defaults`.

        cosmology : object, optional keyword argument
            Astropy cosmology object. Default is set in `~halotools.empirical_models.sim_defaults`.

        redshift : float, optional keyword argument 
            Default is set in `~halotools.empirical_models.sim_defaults`.

        halo_boundary : string, optional keyword argument 
            String giving the column name of the halo catalog that stores the boundary of the halo. 
            Default is set in the `~halotools.empirical_models.model_defaults` module. 

        Examples 
        --------
        You can load a NFW profile model with the default settings simply by calling 
        the class constructor with no arguments:

        >>> nfw_halo_prof_model = NFWProfile()

        For an NFW profile with an alternative cosmology and redshift:

        >>> from astropy.cosmology import WMAP9
        >>> nfw_halo_prof_model = NFWProfile(cosmology = WMAP9, redshift = 2)
        """

        super(NFWProfile, self).__init__(
            cosmology=cosmology, redshift=redshift, halo_boundary=halo_boundary, 
            prof_param_keys=['NFWmodel_conc'], **kwargs)

        self.NFWmodel_conc_lookup_table_min = model_defaults.min_permitted_conc
        self.NFWmodel_conc_lookup_table_max = model_defaults.max_permitted_conc
        self.NFWmodel_conc_lookup_table_spacing = model_defaults.default_dconc

        conc_mass_model = halo_prof_param_components.ConcMass(
            cosmology=self.cosmology, redshift = self.redshift, 
            conc_mass_model=conc_mass_model, **kwargs)

        self.NFWmodel_conc = conc_mass_model.__call__

        self.build_inv_cumu_lookup_table()

        self.publications = ['arXiv:9611107', 'arXiv:0002395', 'arXiv:1402.7073']

    def g(self, x):
        """ Convenience function used to evaluate the profile. 

        Parameters 
        ----------
        x : array_like 

        Returns 
        -------
        g : array_like 
            :math:`1 / g(x) = \\log(1+x) - x / (1+x)`

        Examples 
        --------
        >>> model = NFWProfile()
        >>> g = model.g(1)
        >>> Npts = 25
        >>> g = model.g(np.logspace(-1, 1, Npts))
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
            :math:`\\rho_{\\mathrm{s}} = \\frac{1}{3}\\Delta_{\\mathrm{vir}}c^{3}g(c)\\bar{\\rho}_{\\mathrm{m}}`

        """
        return (self.delta_vir/3.)*c*c*c*self.g(c)*self.cosmic_matter_density

    def density_profile(self, r, c):
        """ NFW profile density. 

        :math:`\\rho_{\\mathrm{NFW}}(r | c) = \\rho_{\\mathrm{s}} / cr(1+cr)^{2}`

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
            NFW density profile :math:`\\rho_{\\mathrm{NFW}}(r | c)`.
        """
        numerator = self.rho_s(c)
        denominator = (c*r)*(1.0 + c*r)*(1.0 + c*r)
        return numerator / denominator

    def cumulative_mass_PDF(self, r, *args):
        """ Cumulative probability distribution of the NFW profile. 

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
            :math:`P_{\\mathrm{NFW}}(<r | c) = g(c) / g(c*r)`. 

        Examples 
        --------
        To evaluate the cumulative PDF for a single profile: 

        >>> nfw_halo_prof_model = NFWProfile()
        >>> Npts = 100
        >>> radius = np.logspace(-2, 0, Npts)
        >>> conc = 8
        >>> cumulative_prob = nfw_halo_prof_model.cumulative_mass_PDF(radius, conc)

        Or, to evaluate the cumulative PDF for profiles with a range of concentrations:

        >>> conc_array = np.linspace(1, 25, Npts)
        >>> cumulative_prob = nfw_halo_prof_model.cumulative_mass_PDF(radius, conc_array)
        """

        if len(args)==0:
            raise SyntaxError("Must pass array of concentrations to cumulative_mass_PDF. \n"
                "Only received array of radii.")
        else:
            if custom_len(args[0]) == 1:
                c = np.ones(len(r))*args[0]
                return self.g(c) / self.g(r*c)
            elif custom_len(args[0]) != custom_len(r):
                raise ValueError("If passing an array of concentrations to "
                    "cumulative_mass_PDF, the array must have the same length "
                    "as the array of radial positions")
            else:
                c = args[0]
                return self.g(c) / self.g(r*c)

##################################################################################
















