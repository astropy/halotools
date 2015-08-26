# -*- coding: utf-8 -*-
"""
This module contains the components for 
the radial profiles of galaxies 
inside their halos. 
"""

import numpy as np 
from phase_space_metaclasses import AnalyticDensityProf
from ..sim_manager import sim_defaults 
from . import model_defaults 

__author__ = ['Andrew Hearin']

class TrivialProfile(AnalyticDensityProf):
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

        >>> trivial_halo_prof_model = TrivialProfile() # doctest: +SKIP 

        """

        super(TrivialProfile, self).__init__(prof_param_keys=[], **kwargs)

        self.build_inv_cumu_lookup_table()

        self.publications = []


class NFWProfile(AnalyticDensityProf):
    """ NFW halo profile, based on Navarro, Frenk and White (1999).

    """

    def __init__(self, 
        cosmology=sim_defaults.default_cosmology, 
        redshift=sim_defaults.default_redshift,
        halo_boundary=model_defaults.halo_boundary,
        conc_mass_model = model_defaults.conc_mass_model, **kwargs):
        """
        Parameters 
        ----------
        prim_haloprop_key : string, optional  
            String giving the column name of the primary halo property governing 
            the occupation statistics of gal_type galaxies. 
            Default is set in `~halotools.empirical_models.sim_defaults`.

        conc_mass_model : string, optional  
            Specifies the calibrated fitting function used to model the concentration-mass relation. 
             Default is set in `~halotools.empirical_models.sim_defaults`.

        cosmology : object, optional 
            Astropy cosmology object. Default is set in `~halotools.empirical_models.sim_defaults`.

        redshift : float, optional  
            Default is set in `~halotools.empirical_models.sim_defaults`.

        halo_boundary : string, optional  
            String giving the column name of the halo catalog that stores the boundary of the halo. 
            Default is set in the `~halotools.empirical_models.model_defaults` module. 

        Examples 
        --------
        You can load a NFW profile model with the default settings simply by calling 
        the class constructor with no arguments:

        >>> nfw_halo_prof_model = NFWProfile() # doctest: +SKIP 

        For an NFW profile with an alternative cosmology and redshift:

        >>> from astropy.cosmology import WMAP9
        >>> nfw_halo_prof_model = NFWProfile(cosmology = WMAP9, redshift = 2) # doctest: +SKIP 
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
        >>> model = NFWProfile() # doctest: +SKIP 
        >>> g = model.g(1) # doctest: +SKIP 
        >>> Npts = 25 # doctest: +SKIP 
        >>> g = model.g(np.logspace(-1, 1, Npts)) # doctest: +SKIP 
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

        >>> nfw_halo_prof_model = NFWProfile() # doctest: +SKIP 
        >>> Npts = 100 # doctest: +SKIP 
        >>> radius = np.logspace(-2, 0, Npts) # doctest: +SKIP 
        >>> conc = 8 # doctest: +SKIP 
        >>> cumulative_prob = nfw_halo_prof_model.cumulative_mass_PDF(radius, conc) # doctest: +SKIP 

        Or, to evaluate the cumulative PDF for profiles with a range of concentrations:

        >>> conc_array = np.linspace(1, 25, Npts) # doctest: +SKIP 
        >>> cumulative_prob = nfw_halo_prof_model.cumulative_mass_PDF(radius, conc_array) # doctest: +SKIP 
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













