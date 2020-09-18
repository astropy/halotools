"""
This module contains the `NFWProfile` class,
which is used to model the spatial distribution of mass and/or galaxies
inside dark matter halos according to the fitting function introduced in
Navarry, Frenk and White (1995), `arXiv:9508025 <http://arxiv.org/abs/astro-ph/9508025/>`_.
a sub-class of `~halotools.empirical_models.AnalyticDensityProf`.
"""
from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np

from .conc_mass import direct_from_halo_catalog, dutton_maccio14
from .kernels import nfw_dimensionless_mass_density, nfw_cumulative_mass_PDF
from .kernels import standalone_mc_generate_nfw_radial_positions

from ...profile_model_template import AnalyticDensityProf

from ..... import model_defaults

from ......sim_manager import sim_defaults


__all__ = ("NFWProfile",)
__author__ = ("Andrew Hearin", "Benedikt Diemer")


class NFWProfile(AnalyticDensityProf):
    r""" Model for the spatial distribution of mass
    and/or galaxies residing in an NFW halo profile,
    based on Navarro, Frenk and White (1995),
    `arXiv:9508025 <http://arxiv.org/abs/astro-ph/9508025/>`_.

    For a review of the mathematics underlying the NFW profile,
    including descriptions of how the relevant equations are
    implemented in the Halotools code base, see :ref:`nfw_profile_tutorial`.

    """

    def __init__(
        self,
        cosmology=sim_defaults.default_cosmology,
        redshift=sim_defaults.default_redshift,
        mdef=model_defaults.halo_mass_definition,
        conc_mass_model=model_defaults.conc_mass_model,
        concentration_key=model_defaults.concentration_key,
        halo_boundary_key=None,
        **kwargs
    ):
        r"""
        Parameters
        ----------
        cosmology : object, optional
            Instance of an astropy `~astropy.cosmology`.
            Default cosmology is set in
            `~halotools.sim_manager.sim_defaults`.

        redshift : float, optional
            Default is set in `~halotools.sim_manager.sim_defaults`.

        mdef: str, optional
            String specifying the halo mass definition, e.g., 'vir' or '200m'.
            Default is set in `~halotools.empirical_models.model_defaults`.

        halo_boundary_key : str, optional
            Default behavior is to use the column associated with the input mdef.

        conc_mass_model : string or callable, optional
            Specifies the function used to model the relation between
            NFW concentration and halo mass.
            Can either be a custom-built callable function,
            or one of the following strings:
            ``dutton_maccio14``, ``direct_from_halo_catalog``.

        concentration_key : string, optional
            Column name of the halo catalog storing NFW concentration.

            This argument is only relevant when ``conc_mass_model``
            is set to ``direct_from_halo_catalog``. In such a case,
            the default value is ``halo_nfw_conc``,
            which is consistent with all halo catalogs provided by Halotools
            but may differ from the convention adopted in custom halo catalogs.

        Examples
        --------
        >>> nfw = NFWProfile()
        """
        AnalyticDensityProf.__init__(
            self, cosmology, redshift, mdef, halo_boundary_key=halo_boundary_key
        )

        self.gal_prof_param_keys = ["conc_NFWmodel"]
        self.halo_prof_param_keys = ["conc_NFWmodel"]

        self.publications = ["arXiv:9611107", "arXiv:0002395"]

        self._initialize_conc_mass_behavior(
            conc_mass_model, concentration_key=concentration_key
        )

    def _initialize_conc_mass_behavior(self, conc_mass_model, **kwargs):
        if conc_mass_model == "direct_from_halo_catalog":
            self.concentration_key = kwargs.get(
                "concentration_key", model_defaults.concentration_key
            )

        self.conc_mass_model = conc_mass_model

    def conc_NFWmodel(self, *args, **kwargs):
        r""" NFW concentration as a function of halo mass.

        Parameters
        ----------
        prim_haloprop : array, optional
            Array storing the mass-like variable, e.g., ``halo_mvir``.

            If ``prim_haloprop`` is not passed,
            then ``table`` keyword argument must be passed.

        table : object, optional
            `~astropy.table.Table` storing the halo catalog.

            If your NFW model is based on the virial definition,
            then ``halo_mvir`` must appear in the input table,
            and likewise for other halo mass definitions.

            If ``table`` is not passed,
            then ``prim_haloprop`` keyword argument must be passed.

        Returns
        -------
        conc : array_like
            Concentrations of the input halos.

            Note that concentrations will be clipped to their min/max permitted
            values set in the `~halotools.empirical_models.model_defaults` module.
            The purpose of this clipping is to ensure stable results during
            mock galaxy population. Due to this clipping,
            the behavior of the `conc_NFWmodel` function
            is different from the concentration-mass relation that underlies it.

        Examples
        ---------
        In the examples below, we'll demonstrate the various ways to use the
        `~halotools.empirical_models.NFWProfile.conc_NFWmodel` function, depending
        on the initial choice for the ``conc_mass_model``.

        >>> fake_masses = np.logspace(12, 15, 10)

        If you use the ``direct_from_halo_catalog`` option, you must pass a
        ``table`` argument storing a `~astropy.table.Table` with a column name
        for the halo mass that is consistent with your chosen halo mass definition:

        >>> from astropy.table import Table
        >>> nfw = NFWProfile(conc_mass_model='direct_from_halo_catalog', mdef='vir')
        >>> fake_conc = np.zeros_like(fake_masses) + 5.
        >>> fake_halo_table = Table({'halo_mvir': fake_masses, 'halo_nfw_conc': fake_conc})
        >>> model_conc = nfw.conc_NFWmodel(table=fake_halo_table)

        In case your halo catalog uses a different keyname from the Halotools
        default ``halo_nfw_conc``:

        >>> nfw = NFWProfile(conc_mass_model='direct_from_halo_catalog', mdef='vir', concentration_key='my_conc_keyname')
        >>> fake_halo_table = Table({'halo_mvir': fake_masses, 'my_conc_keyname': fake_conc})
        >>> model_conc = nfw.conc_NFWmodel(table=fake_halo_table)

        One of the available options provided by Halotools is ``dutton_maccio14``.
        With this option, you can either pass in a ``table`` argument, or alternatively
        an array of masses via the ``prim_haloprop`` argument:

        >>> nfw = NFWProfile(conc_mass_model='dutton_maccio14')
        >>> fake_halo_table = Table({'halo_mvir': fake_masses, 'halo_nfw_conc': fake_conc})
        >>> model_conc = nfw.conc_NFWmodel(table=fake_halo_table)
        >>> model_conc = nfw.conc_NFWmodel(prim_haloprop=fake_masses)

        Finally, you may also have chosen to define your own concentration-mass relation.
        If so, your function must at a minimum accept a ``table`` keyword argument.
        Below we give a trivial example of using the identity function:

        >>> def identity_func(*args, **kwargs): return kwargs['table']['halo_mvir']
        >>> nfw = NFWProfile(conc_mass_model=identity_func, mdef='vir')
        >>> fake_halo_table = Table({'halo_mvir': fake_masses})
        >>> model_conc = nfw.conc_NFWmodel(table=fake_halo_table)
        """
        if self.conc_mass_model == "direct_from_halo_catalog":
            try:
                table = kwargs["table"]
            except KeyError:
                msg = (
                    "Must pass ``table`` argument to the ``conc_NFWmodel`` function\n"
                    "when ``conc_mass_model`` is set to ``direct_from_halo_catalog``\n"
                )
                raise KeyError(msg)
            result = direct_from_halo_catalog(
                table=table, concentration_key=self.concentration_key
            )

        elif self.conc_mass_model == "dutton_maccio14":
            msg = (
                "Must either pass a ``prim_haloprop`` argument, \n"
                "or a ``table`` argument with an astropy Table that has the ``{0}`` key"
            )
            try:
                mass = kwargs["table"][self.prim_haloprop_key]
            except:
                try:
                    mass = kwargs["prim_haloprop"]
                except:
                    raise KeyError(msg.format(self.prim_haloprop_key))
            result = dutton_maccio14(mass, self.redshift)

        else:
            result = self.conc_mass_model(*args, **kwargs)

        cmin = model_defaults.min_permitted_conc
        cmax = model_defaults.max_permitted_conc
        result = np.where(result < cmin, cmin, result)
        result = np.where(result > cmax, cmax, result)
        return result

    def dimensionless_mass_density(self, scaled_radius, conc):
        r"""
        Physical density of the NFW halo scaled by the density threshold of the mass definition.

        The `dimensionless_mass_density` is defined as
        :math:`\tilde{\rho}_{\rm prof}(\tilde{r}) \equiv \rho_{\rm prof}(\tilde{r}) / \rho_{\rm thresh}`,
        where :math:`\tilde{r}\equiv r/R_{\Delta}`.

        For an NFW halo,
        :math:`\tilde{\rho}_{\rm NFW}(\tilde{r}, c) = \frac{c^{3}/3g(c)}{c\tilde{r}(1 + c\tilde{r})^{2}},`

        where :math:`g(x) \equiv \log(1+x) - x / (1+x)` is computed using the `g` function.

        The quantity :math:`\rho_{\rm thresh}` is a function of
        the halo mass definition, cosmology and redshift,
        and is computed via the
        `~halotools.empirical_models.profile_helpers.density_threshold` function.
        The quantity :math:`\rho_{\rm prof}` is the physical mass density of the
        halo profile and is computed via the `mass_density` function.
        See :ref:`nfw_spatial_profile_derivations` for a derivation of this expression.

        Parameters
        -----------
        scaled_radius : array_like
            Halo-centric distance *r* scaled by the halo boundary :math:`R_{\Delta}`, so that
            :math:`0 <= \tilde{r} \equiv r/R_{\Delta} <= 1`. Can be a scalar or numpy array.

        conc : array_like
            Value of the halo concentration. Can either be a scalar, or a numpy array
            of the same dimension as the input ``scaled_radius``.

        Returns
        -------
        dimensionless_density: array_like
            Dimensionless density of a dark matter halo
            at the input ``scaled_radius``, normalized by the
            `~halotools.empirical_models.profile_helpers.density_threshold`
            :math:`\rho_{\rm thresh}` for the
            halo mass definition, cosmology, and redshift.
            Result is an array of the dimension as the input ``scaled_radius``.
        """
        return nfw_dimensionless_mass_density(scaled_radius, conc)

    def mass_density(self, radius, mass, conc):
        r"""
        Physical density of the halo at the input radius,
        given in units of :math:`h^{3}/{\rm Mpc}^{3}`.

        Parameters
        -----------
        radius : array_like
            Halo-centric distance in Mpc/h units; can be a scalar or numpy array

        mass : array_like
            Total mass of the halo; can be a scalar or numpy array of the same
            dimension as the input ``radius``.

        conc : array_like
            Value of the halo concentration. Can either be a scalar, or a numpy array
            of the same dimension as the input ``radius``.

        Returns
        -------
        density: array_like
            Physical density of a dark matter halo of the input ``mass``
            at the input ``radius``. Result is an array of the
            dimension as the input ``radius``, reported in units of :math:`h^{3}/Mpc^{3}`.

        Examples
        --------
        >>> model = NFWProfile()
        >>> Npts = 100
        >>> radius = np.logspace(-2, -1, Npts)
        >>> mass = np.zeros(Npts) + 1e12
        >>> conc = 5
        >>> result = model.mass_density(radius, mass, conc)
        >>> concarr = np.linspace(1, 100, Npts)
        >>> result = model.mass_density(radius, mass, concarr)

        Notes
        ------
        See :ref:`halo_profile_definitions` for derivations and implementation details.
        """
        return AnalyticDensityProf.mass_density(self, radius, mass, conc)

    def cumulative_mass_PDF(self, scaled_radius, conc):
        r"""
        Analytical result for the fraction of the total mass
        enclosed within dimensionless radius of an NFW halo,

        :math:`P_{\rm NFW}(<\tilde{r}) \equiv M_{\Delta}(<\tilde{r}) / M_{\Delta} = g(c\tilde{r})/g(\tilde{r}),`

        where :math:`g(x) \equiv \int_{0}^{x}dy\frac{y}{(1+y)^{2}} = \log(1+x) - x / (1+x)` is computed
        using `g`, and where :math:`\tilde{r} \equiv r / R_{\Delta}`.

        See :ref:`nfw_cumulative_mass_pdf_derivation` for a derivation of this expression.

        Parameters
        -------------
        scaled_radius : array_like
            Halo-centric distance *r* scaled by the halo boundary :math:`R_{\Delta}`, so that
            :math:`0 <= \tilde{r} \equiv r/R_{\Delta} <= 1`. Can be a scalar or numpy array.

        conc : array_like
            Value of the halo concentration. Can either be a scalar, or a numpy array
            of the same dimension as the input ``scaled_radius``.

        Returns
        -------------
        p: array_like
            The fraction of the total mass enclosed
            within the input ``scaled_radius``, in :math:`M_{\odot}/h`;
            has the same dimensions as the input ``scaled_radius``.

        Examples
        --------
        >>> model = NFWProfile()
        >>> Npts = 100
        >>> scaled_radius = np.logspace(-2, 0, Npts)
        >>> conc = 5
        >>> result = model.cumulative_mass_PDF(scaled_radius, conc)
        >>> concarr = np.linspace(1, 100, Npts)
        >>> result = model.cumulative_mass_PDF(scaled_radius, concarr)
        """
        scaled_radius = np.where(scaled_radius > 1, 1, scaled_radius)
        scaled_radius = np.where(scaled_radius < 0, 0, scaled_radius)
        return nfw_cumulative_mass_PDF(scaled_radius, conc)

    def enclosed_mass(self, radius, total_mass, conc):
        r"""
        The mass enclosed within the input radius,
        :math:`M(<r) = 4\pi\int_{0}^{r}dr'r'^{2}\rho(r)`.

        Parameters
        -----------
        radius : array_like
            Halo-centric distance in Mpc/h units; can be a scalar or numpy array

        total_mass : array_like
            Total mass of the halo; can be a scalar or numpy array of the same
            dimension as the input ``radius``.

        conc : array_like
            Value of the halo concentration. Can either be a scalar, or a numpy array
            of the same dimension as the input ``radius``.

        Returns
        ----------
        enclosed_mass: array_like
            The mass enclosed within radius r, in :math:`M_{\odot}/h`;
            has the same dimensions as the input ``radius``.

        Examples
        --------
        >>> model = NFWProfile()
        >>> Npts = 100
        >>> radius = np.logspace(-2, -1, Npts)
        >>> total_mass = np.zeros(Npts) + 1e12
        >>> conc = 5
        >>> result = model.enclosed_mass(radius, total_mass, conc)
        >>> concarr = np.linspace(1, 100, Npts)
        >>> result = model.enclosed_mass(radius, total_mass, concarr)

        Notes
        ------
        See :ref:`halo_profile_definitions` for derivations and implementation details.
        """
        return AnalyticDensityProf.enclosed_mass(self, radius, total_mass, conc)

    def virial_velocity(self, total_mass):
        r""" The circular velocity evaluated at the halo boundary,
        :math:`V_{\rm vir} \equiv \sqrt{GM_{\rm halo}/R_{\rm halo}}`.

        Parameters
        --------------
        total_mass : array_like
            Total mass of the halo; can be a scalar or numpy array.

        Returns
        --------
        vvir : array_like
            Virial velocity in km/s.

        Examples
        --------
        >>> model = NFWProfile()
        >>> Npts = 100
        >>> mass_array = np.logspace(11, 15, Npts)
        >>> vvir_array = model.virial_velocity(mass_array)

        Notes
        ------
        See :ref:`halo_profile_definitions` for derivations and implementation details.

        """
        return AnalyticDensityProf.virial_velocity(self, total_mass)

    def circular_velocity(self, radius, total_mass, conc):
        r"""
        The circular velocity, :math:`V_{\rm cir} \equiv \sqrt{GM(<r)/r}`,
        as a function of halo-centric distance r.

        Parameters
        --------------
        radius : array_like
            Halo-centric distance in Mpc/h units; can be a scalar or numpy array

        total_mass : array_like
            Total mass of the halo; can be a scalar or numpy array of the same
            dimension as the input ``radius``.

        conc : array_like
            Value of the halo concentration. Can either be a scalar, or a numpy array
            of the same dimension as the input ``radius``.

        Returns
        ----------
        vc: array_like
            The circular velocity in km/s; has the same dimensions as the input ``radius``.

        Examples
        --------
        >>> model = NFWProfile()
        >>> Npts = 100
        >>> radius = np.logspace(-2, -1, Npts)
        >>> total_mass = np.zeros(Npts) + 1e12
        >>> conc = 5
        >>> result = model.circular_velocity(radius, total_mass, conc)
        >>> concarr = np.linspace(1, 100, Npts)
        >>> result = model.circular_velocity(radius, total_mass, concarr)

        Notes
        ------
        See :ref:`halo_profile_definitions` for derivations and implementation details.
        """
        return AnalyticDensityProf.circular_velocity(self, radius, total_mass, conc)

    def rmax(self, total_mass, conc):
        r""" Radius at which the halo attains its maximum circular velocity,
        :math:`R_{\rm max}^{\rm NFW} = 2.16258R_{\Delta}/c`.

        Parameters
        ----------
        total_mass: array_like
            Total halo mass in :math:`M_{\odot}/h`; can be a number or a numpy array.

        conc : array_like
            Value of the halo concentration. Can either be a scalar, or a numpy array
            of the same dimension as the input ``total_mass``.

        Returns
        --------
        rmax : array_like
            :math:`R_{\rm max}` in Mpc/h.

        Examples
        --------
        >>> model = NFWProfile()
        >>> Npts = 100
        >>> total_mass = np.zeros(Npts) + 1e12
        >>> conc = 5
        >>> result = model.rmax(total_mass, conc)
        >>> concarr = np.linspace(1, 100, Npts)
        >>> result = model.rmax(total_mass, concarr)

        Notes
        ------
        See :ref:`halo_profile_definitions` for derivations and implementation details.

        """
        halo_radius = self.halo_mass_to_halo_radius(total_mass)
        scale_radius = halo_radius / conc
        return 2.16258 * scale_radius

    def vmax(self, total_mass, conc):
        r""" Maximum circular velocity of the halo profile,
        :math:`V_{\rm max}^{\rm NFW} = V_{\rm cir}^{\rm NFW}(r = 2.16258R_{\Delta}/c)`.

        Parameters
        ----------
        total_mass: array_like
            Total halo mass in :math:`M_{\odot}/h`; can be a number or a numpy array.

        conc : array_like
            Value of the halo concentration. Can either be a scalar, or a numpy array
            of the same dimension as the input ``total_mass``.

        Returns
        --------
        vmax : array_like
            :math:`V_{\rm max}` in km/s.

        Examples
        --------
        >>> model = NFWProfile()
        >>> Npts = 100
        >>> total_mass = np.zeros(Npts) + 1e12
        >>> conc = 5
        >>> result = model.vmax(total_mass, conc)
        >>> concarr = np.linspace(1, 100, Npts)
        >>> result = model.vmax(total_mass, concarr)

        Notes
        ------
        See :ref:`halo_profile_definitions` for derivations and implementation details.
        """
        Rmax = self.rmax(total_mass, conc)
        vmax = self.circular_velocity(Rmax, total_mass, conc)
        return vmax

    def halo_mass_to_halo_radius(self, total_mass):
        r"""
        Spherical overdensity radius as a function of the input mass.

        Note that this function is independent of the form of the density profile.

        Parameters
        ----------
        total_mass: array_like
            Total halo mass in :math:`M_{\odot}/h`; can be a number or a numpy array.

        Returns
        -------
        radius : array_like
            Radius of the halo in Mpc/h units.
            Will have the same dimension as the input ``total_mass``.

        Examples
        --------
        >>> model = NFWProfile()
        >>> halo_radius = model.halo_mass_to_halo_radius(1e13)

        """
        return AnalyticDensityProf.halo_mass_to_halo_radius(self, total_mass)

    def halo_radius_to_halo_mass(self, radius):
        r"""
        Spherical overdensity mass as a function of the input radius.

        Note that this function is independent of the form of the density profile.

        Parameters
        ------------
        radius : array_like
            Radius of the halo in Mpc/h units; can be a number or a numpy array.

        Returns
        ----------
        total_mass: array_like
            Total halo mass in :math:`M_{\odot}/h`.
            Will have the same dimension as the input ``radius``.

        Examples
        --------
        >>> model = NFWProfile()
        >>> halo_mass = model.halo_mass_to_halo_radius(500.)
        """
        return AnalyticDensityProf.halo_radius_to_halo_mass(self, radius)

    def mc_generate_nfw_radial_positions(self, **kwargs):
        r""" Return a Monte Carlo realization of points in an NFW profile.

        See :ref:`monte_carlo_nfw_spatial_profile` for a discussion of this technique.

        Parameters
        -----------
        num_pts : int, optional
            Number of points in the Monte Carlo realization of the profile.
            Default is 1e4.

        conc : float, optional
            Concentration of the NFW profile being realized.
            Default is 5.

        halo_mass : float, optional
            Total mass of the halo whose profile is being realized,
            used to define the halo boundary for the mass definition
            bound to the NFWProfile instance as ``mdef``.

            If ``halo_mass`` is unspecified,
            keyword argument ``halo_radius`` must be specified.

        halo_radius : float, optional
            Physical boundary of the halo whose profile is being realized
            in units of Mpc/h.

            If ``halo_radius`` is unspecified,
            keyword argument ``halo_mass`` must be specified, in which case the
            outer boundary of the halo will be determined according to the mass definition
            bound to the NFWProfile instance as ``mdef``.

        seed : int, optional
            Random number seed used in the Monte Carlo realization.
            Default is None, which will produce stochastic results.

        Returns
        --------
        radial_positions : array_like
            Numpy array storing a Monte Carlo realization of the halo profile.
            All values will lie strictly between 0 and the halo boundary.

        Examples
        ---------
        >>> nfw = NFWProfile()
        >>> radial_positions = nfw.mc_generate_nfw_radial_positions(halo_mass = 1e12, conc = 10)
        >>> radial_positions = nfw.mc_generate_nfw_radial_positions(halo_radius = 0.25)
        """
        kwargs["mdef"] = self.mdef
        kwargs["cosmology"] = self.cosmology
        kwargs["redshift"] = self.redshift
        return standalone_mc_generate_nfw_radial_positions(**kwargs)
