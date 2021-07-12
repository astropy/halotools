"""
"""
from __future__ import division, print_function, absolute_import

import numpy as np
from astropy.table import Table

from .nfw_profile import NFWProfile
from .kernels import unbiased_dimless_vrad_disp as unbiased_dimless_vrad_disp_kernel

from ...monte_carlo_helpers import MonteCarloGalProf

from ..... import model_defaults


__author__ = ["Andrew Hearin"]
__all__ = ["NFWPhaseSpace"]


class NFWPhaseSpace(NFWProfile, MonteCarloGalProf):
    r""" Model for the phase space distribution of mass and/or galaxies
    in isotropic Jeans equilibrium in an NFW halo profile,
    based on Navarro, Frenk and White (1995),
    where the concentration of the galaxies is the same
    as the concentration of the parent halo

    For a review of the mathematics underlying the NFW profile,
    including descriptions of how the relevant equations are
    implemented in the Halotools code base, see :ref:`nfw_profile_tutorial`.
    """

    def __init__(self, **kwargs):
        r"""
        Parameters
        ----------
        conc_mass_model : string or callable, optional
            Specifies the function used to model the relation between
            NFW concentration and halo mass.
            Can either be a custom-built callable function,
            or one of the following strings:
            ``dutton_maccio14``, ``direct_from_halo_catalog``.

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

        concentration_key : string, optional
            Column name of the halo catalog storing NFW concentration.

            This argument is only relevant when ``conc_mass_model``
            is set to ``direct_from_halo_catalog``. In such a case,
            the default value is ``halo_nfw_conc``,
            which is consistent with all halo catalogs provided by Halotools
            but may differ from the convention adopted in custom halo catalogs.

        concentration_bins : ndarray, optional
            Array storing how halo concentrations will be digitized when building
            a lookup table for mock-population purposes.
            The spacing of this array sets a limit on how accurately the
            concentration parameter can be recovered in a likelihood analysis.

        Examples
        --------
        >>> model = NFWPhaseSpace()
        """
        NFWProfile.__init__(self, **kwargs)
        MonteCarloGalProf.__init__(self)

        prof_lookup_args = self._retrieve_prof_lookup_info(**kwargs)
        self.setup_prof_lookup_tables(*prof_lookup_args)

        self._mock_generation_calling_sequence = ["assign_phase_space"]

    def _retrieve_prof_lookup_info(self, **kwargs):
        r""" Retrieve the arrays defining the lookup table control points
        """
        cmin, cmax = (
            model_defaults.min_permitted_conc,
            model_defaults.max_permitted_conc,
        )
        dc = 1.0
        npts_conc = int(np.round((cmax - cmin) / float(dc)))
        default_conc_arr = np.linspace(cmin, cmax, npts_conc)
        conc_arr = kwargs.get("concentration_bins", default_conc_arr)
        return [conc_arr]

    def assign_phase_space(self, table, seed=None):
        r""" Primary method of the `NFWPhaseSpace` class
        called during the mock-population sequence.

        Parameters
        -----------
        table : object
            `~astropy.table.Table` storing halo catalog.

            After calling the `assign_phase_space` method,
            the `x`, `y`, `z`, `vx`, `vy`, and `vz`
            columns of the input ``table`` will be over-written
            with their host-centric values.

        seed : int, optional
            Random number seed used in the Monte Carlo realization.
            Default is None, which will produce stochastic results.

        """
        MonteCarloGalProf.mc_pos(self, table=table, seed=seed)
        if seed is not None:
            seed += 1
        MonteCarloGalProf.mc_vel(self, table=table, seed=seed)

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
        `~halotools.empirical_models.NFWPhaseSpace.conc_NFWmodel` function, depending
        on the initial choice for the ``conc_mass_model``.

        >>> fake_masses = np.logspace(12, 15, 10)

        If you use the ``direct_from_halo_catalog`` option, you must pass a
        ``table`` argument storing a `~astropy.table.Table` with a column name
        for the halo mass that is consistent with your chosen halo mass definition:

        >>> from astropy.table import Table
        >>> nfw = NFWPhaseSpace(conc_mass_model='direct_from_halo_catalog', mdef='vir')
        >>> fake_conc = np.zeros_like(fake_masses) + 5.
        >>> fake_halo_table = Table({'halo_mvir': fake_masses, 'halo_nfw_conc': fake_conc})
        >>> model_conc = nfw.conc_NFWmodel(table=fake_halo_table)

        In case your halo catalog uses a different keyname from the Halotools
        default ``halo_nfw_conc``:

        >>> nfw = NFWPhaseSpace(conc_mass_model='direct_from_halo_catalog', mdef='vir', concentration_key='my_conc_keyname')
        >>> fake_halo_table = Table({'halo_mvir': fake_masses, 'my_conc_keyname': fake_conc})
        >>> model_conc = nfw.conc_NFWmodel(table=fake_halo_table)

        One of the available options provided by Halotools is ``dutton_maccio14``.
        With this option, you can either pass in a ``table`` argument, or alternatively
        an array of masses via the ``prim_haloprop`` argument:

        >>> nfw = NFWPhaseSpace(conc_mass_model='dutton_maccio14')
        >>> fake_halo_table = Table({'halo_mvir': fake_masses, 'halo_nfw_conc': fake_conc})
        >>> model_conc = nfw.conc_NFWmodel(table=fake_halo_table)
        >>> model_conc = nfw.conc_NFWmodel(prim_haloprop=fake_masses)

        Finally, you may also have chosen to define your own concentration-mass relation.
        If so, your function must at a minimum accept a ``table`` keyword argument.
        Below we give a trivial example of using the identity function:

        >>> def identity_func(*args, **kwargs): return kwargs['table']['halo_mvir']
        >>> nfw = NFWPhaseSpace(conc_mass_model=identity_func, mdef='vir')
        >>> fake_halo_table = Table({'halo_mvir': fake_masses})
        >>> model_conc = nfw.conc_NFWmodel(table=fake_halo_table)
        """
        return NFWProfile.conc_NFWmodel(self, **kwargs)

    def dimensionless_mass_density(self, scaled_radius, conc):
        r"""
        Physical density of the NFW halo scaled by the density threshold of the mass definition:

        The `dimensionless_mass_density` is defined as
        :math:`\tilde{\rho}_{\rm prof}(\tilde{r}) \equiv \rho_{\rm prof}(\tilde{r}) / \rho_{\rm thresh}`,
        where :math:`\tilde{r}\equiv r/R_{\Delta}`.

        For an NFW halo,
        :math:`\tilde{\rho}_{\rm NFW}(\tilde{r}, c) = \frac{c^{3}}{3g(c)}\times\frac{1}{c\tilde{r}(1 + c\tilde{r})^{2}},`

        where :math:`g(x) \equiv \int_{0}^{x}dy\frac{y}{(1+y)^{2}} = \log(1+x) - x / (1+x)` is computed using the `g` function.

        The quantity :math:`\rho_{\rm thresh}` is a function of
        the halo mass definition, cosmology and redshift,
        and is computed via the
        `~halotools.empirical_models.profile_helpers.density_threshold` function.
        The quantity :math:`\rho_{\rm prof}` is the physical mass density of the
        halo profile and is computed via the `mass_density` function.

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
        return NFWProfile.dimensionless_mass_density(self, scaled_radius, conc)

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
        >>> model = NFWPhaseSpace()
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
        return NFWProfile.mass_density(self, radius, mass, conc)

    def cumulative_gal_PDF(self, scaled_radius, conc):
        r""" Analogous to `cumulative_mass_PDF`, but for the satellite galaxy distribution
        instead of the host halo mass distribution.

        In `~halotools.empirical_models.NFWPhaseSpace` there is no distinction between the
        two methods, but in `~halotools.empirical_models.BiasedNFWPhaseSpace` these two
        function are different.

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
        >>> model = NFWPhaseSpace()
        >>> Npts = 100
        >>> scaled_radius = np.logspace(-2, 0, Npts)
        >>> conc = 5
        >>> result = model.cumulative_gal_PDF(scaled_radius, conc)
        >>> concarr = np.linspace(1, 100, Npts)
        >>> result = model.cumulative_gal_PDF(scaled_radius, concarr)

        """
        return NFWProfile.cumulative_mass_PDF(self, scaled_radius, conc)

    def cumulative_mass_PDF(self, scaled_radius, conc):
        r"""
        Analytical result for the fraction of the total mass
        enclosed within r/Rvir of an NFW halo,

        :math:`P_{\rm NFW}(<\tilde{r}) \equiv M_{\Delta}(<\tilde{r}) / M_{\Delta} = g(c\tilde{r})/g(\tilde{r}),`

        where :math:`g(x) \equiv \int_{0}^{x}dy\frac{y}{(1+y)^{2}} = \log(1+x) - x / (1+x)` is computed
        using `g`, and where :math:`\tilde{r} \equiv r / R_{\Delta}`.

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
        >>> model = NFWPhaseSpace()
        >>> Npts = 100
        >>> scaled_radius = np.logspace(-2, 0, Npts)
        >>> conc = 5
        >>> result = model.cumulative_mass_PDF(scaled_radius, conc)
        >>> concarr = np.linspace(1, 100, Npts)
        >>> result = model.cumulative_mass_PDF(scaled_radius, concarr)

        Notes
        ------
        See :ref:`halo_profile_definitions` for derivations and implementation details.
        """
        return NFWProfile.cumulative_mass_PDF(self, scaled_radius, conc)

    def enclosed_mass(self, radius, total_mass, conc):
        r"""
        The mass enclosed within the input radius, :math:`M(<r) = 4\pi\int_{0}^{r}dr'r'^{2}\rho(r)`.

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
        return NFWProfile.enclosed_mass(self, radius, total_mass, conc)

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
        return NFWProfile.virial_velocity(self, total_mass)

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
        >>> model = NFWPhaseSpace()
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
        return NFWProfile.circular_velocity(self, radius, total_mass, conc)

    def vmax(self, total_mass, conc):
        r""" Maximum circular velocity of the halo profile.

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
        >>> model = NFWPhaseSpace()
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
        return NFWProfile.vmax(self, total_mass, conc)

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
        >>> model = NFWPhaseSpace()
        >>> halo_radius = model.halo_mass_to_halo_radius(1e13)
        """
        return NFWProfile.halo_mass_to_halo_radius(self, total_mass)

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
        >>> model = NFWPhaseSpace()
        >>> halo_mass = model.halo_mass_to_halo_radius(500.)

        """
        return NFWProfile.halo_radius_to_halo_mass(self, radius)

    def dimensionless_radial_velocity_dispersion(self, scaled_radius, *conc):
        r"""
        Analytical solution to the isotropic jeans equation for an NFW potential,
        rendered dimensionless via scaling by the virial velocity.

        :math:`\tilde{\sigma}^{2}_{r}(\tilde{r})\equiv\sigma^{2}_{r}(\tilde{r})/V_{\rm vir}^{2} = \frac{c^{2}\tilde{r}(1 + c\tilde{r})^{2}}{g(c)}\int_{c\tilde{r}}^{\infty}{\rm d}y\frac{g(y)}{y^{3}(1 + y)^{2}}`

        See :ref:`nfw_jeans_velocity_profile_derivations` for derivations and implementation details.

        Parameters
        -----------
        scaled_radius : array_like
            Length-Ngals numpy array storing the halo-centric distance
            *r* scaled by the halo boundary :math:`R_{\Delta}`, so that
            :math:`0 <= \tilde{r} \equiv r/R_{\Delta} <= 1`.

        conc : float
            Concentration of the halo.

        Returns
        -------
        result : array_like
            Radial velocity dispersion profile scaled by the virial velocity.
            The returned result has the same dimension as the input ``scaled_radius``.
        """
        return unbiased_dimless_vrad_disp_kernel(scaled_radius, *conc)

    def radial_velocity_dispersion(self, radius, total_mass, halo_conc):
        r"""
        Method returns the radial velocity dispersion scaled by
        the virial velocity as a function of the halo-centric distance.

        Parameters
        ----------
        radius : array_like
            Radius of the halo in Mpc/h units; can be a float or
            ndarray of shape (num_radii, )

        total_mass : array_like
            Float or ndarray of shape (num_radii, ) storing the host halo mass

        halo_conc : array_like
            Float or ndarray of shape (num_radii, ) storing the host halo concentration

        Returns
        -------
        result : array_like
            Radial velocity dispersion profile as a function of the input ``radius``,
            in units of km/s.

        """
        virial_velocities = self.virial_velocity(total_mass)
        halo_radius = self.halo_mass_to_halo_radius(total_mass)
        scaled_radius = radius / halo_radius

        dimensionless_velocities = self.dimensionless_radial_velocity_dispersion(
            scaled_radius, halo_conc
        )
        return dimensionless_velocities * virial_velocities

    def setup_prof_lookup_tables(self, *concentration_bins):
        r"""
        This method sets up how we will digitize halo concentrations during mock population.

        Parameters
        ----------
        concentration_bins : ndarray
            Array storing how concentrations will be digitized during mock-population
        """
        MonteCarloGalProf.setup_prof_lookup_tables(self, *concentration_bins)

    def build_lookup_tables(
        self,
        logrmin=model_defaults.default_lograd_min,
        logrmax=model_defaults.default_lograd_max,
        Npts_radius_table=model_defaults.Npts_radius_table,
    ):
        r""" Method used to create a lookup table of the spatial and velocity radial profiles.

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

        """
        MonteCarloGalProf.build_lookup_tables(self, logrmin, logrmax, Npts_radius_table)

    def mc_unit_sphere(self, Npts, **kwargs):
        r""" Returns Npts random points on the unit sphere.

        Parameters
        ----------
        Npts : int
            Number of 3d points to generate

        seed : int, optional
            Random number seed used in the Monte Carlo realization.
            Default is None, which will produce stochastic results.

        Returns
        -------
        x, y, z : array_like
            Length-Npts arrays of the coordinate positions.

        """
        return MonteCarloGalProf.mc_unit_sphere(self, Npts, **kwargs)

    def mc_solid_sphere(self, *concentration_array, **kwargs):
        r""" Method to generate random, three-dimensional, halo-centric positions of galaxies.

        Parameters
        ----------
        concentration_array : array_like, optional
            Length-Ngals numpy array storing the concentrations of the mock galaxies.

        table : data table, optional
            Astropy Table storing a length-Ngals galaxy catalog.
            If ``table`` is not passed, ``concentration_array`` must be passed.

        seed : int, optional
            Random number seed used in the Monte Carlo realization.
            Default is None, which will produce stochastic results.

        Returns
        -------
        x, y, z : arrays
            Length-Ngals array storing a Monte Carlo realization of the galaxy positions.

        """
        return MonteCarloGalProf.mc_solid_sphere(self, *concentration_array, **kwargs)

    def mc_halo_centric_pos(self, *concentration_array, **kwargs):
        r""" Method to generate random, three-dimensional
        halo-centric positions of galaxies.

        Parameters
        ----------
        table : data table, optional
            Astropy Table storing a length-Ngals galaxy catalog.
            If ``table`` is not passed, ``concentration_array`` and
            keyword argument ``halo_radius`` must be passed.

        concentration_array : array_like, optional
            Length-Ngals numpy array storing the concentrations of the mock galaxies.
            If ``table`` is not passed, ``concentration_array`` and
            keyword argument ``halo_radius`` must be passed.

        halo_radius : array_like, optional
            Length-Ngals array storing the radial boundary of the halo
            hosting each galaxy. Units assumed to be in Mpc/h.
            If ``concentration_array`` and ``halo_radius`` are not passed,
            ``table`` must be passed.

        seed : int, optional
            Random number seed used in the Monte Carlo realization.
            Default is None, which will produce stochastic results.

        Returns
        -------
        x, y, z : arrays
            Length-Ngals array storing a Monte Carlo realization of the galaxy positions.

        """
        return MonteCarloGalProf.mc_halo_centric_pos(
            self, *concentration_array, **kwargs
        )

    def mc_pos(self, *concentration_array, **kwargs):
        r""" Method to generate random, three-dimensional positions of galaxies.

        Parameters
        ----------
        table : data table, optional
            Astropy Table storing a length-Ngals galaxy catalog.
            If ``table`` is not passed, ``concentration_array`` and ``halo_radius`` must be passed.

        concentration_array : array_like, optional
            Length-Ngals numpy array storing the concentrations of the mock galaxies.
            If ``table`` is not passed, ``concentration_array`` and
            keyword argument ``halo_radius`` must be passed.
            If ``concentration_array`` is passed, ``halo_radius`` must be passed as a keyword argument.
            The sequence must have the same order as ``self.gal_prof_param_keys``.

        halo_radius : array_like, optional
            Length-Ngals array storing the radial boundary of the halo
            hosting each galaxy. Units assumed to be in Mpc/h.
            If ``concentration_array`` and ``halo_radius`` are not passed,
            ``table`` must be passed.

        seed : int, optional
            Random number seed used in Monte Carlo realization. Default is None.

        Returns
        -------
        x, y, z : arrays, optional
            For the case where no ``table`` is passed as an argument,
            method will return x, y and z points distributed about the
            origin according to the profile model.

            For the case where ``table`` is passed as an argument
            (this is the use case of populating halos with mock galaxies),
            the ``x``, ``y``, and ``z`` columns of the table will be over-written.
            When ``table`` is passed as an argument, the method
            assumes that the ``x``, ``y``, and ``z`` columns already store
            the position of the host halo center.

        seed : int, optional
            Random number seed used in the Monte Carlo realization.
            Default is None, which will produce stochastic results.

        """
        return MonteCarloGalProf.mc_pos(self, *concentration_array, **kwargs)

    def _vrad_disp_from_lookup(self, scaled_radius, *concentration_array, **kwargs):
        r""" Method to generate Monte Carlo realizations of the profile model.

        Parameters
        ----------
        scaled_radius : array_like
            Halo-centric distance *r* scaled by the halo boundary :math:`R_{\Delta}`, so that
            :math:`0 <= \tilde{r} \equiv r/R_{\Delta} <= 1`. Can be a scalar or numpy array.

        concentration_array : array_like
            Length-Ngals numpy array storing the concentrations of the mock galaxies.

        Returns
        -------
        sigma_vr : array
            Length-Ngals array containing the radial velocity dispersion
            of galaxies within their halos,
            scaled by the size of the halo's virial velocity.

        """
        return MonteCarloGalProf._vrad_disp_from_lookup(
            self, scaled_radius, *concentration_array, **kwargs
        )

    def mc_radial_velocity(
        self, scaled_radius, total_mass, *concentration_array, **kwargs
    ):
        r"""
        Method returns a Monte Carlo realization of radial velocities drawn from Gaussians
        with a width determined by the solution to the isotropic Jeans equation.

        Parameters
        ----------
        scaled_radius : array_like
            Halo-centric distance *r* scaled by the halo boundary :math:`R_{\Delta}`, so that
            :math:`0 <= \tilde{r} \equiv r/R_{\Delta} <= 1`. Can be a scalar or numpy array.

        total_mass: array_like
            Length-Ngals numpy array storing the halo mass in :math:`M_{\odot}/h`.

        concentration_array : array_like
            Length-Ngals numpy array storing the concentrations of the mock galaxies.

        seed : int, optional
            Random number seed used in the Monte Carlo realization.
            Default is None, which will produce stochastic results.

        Returns
        -------
        radial_velocities : array_like
            Array of radial velocities drawn from Gaussians with a width determined by the
            solution to the Jeans equation.
        """
        return MonteCarloGalProf.mc_radial_velocity(
            self, scaled_radius, total_mass, *concentration_array, **kwargs
        )

    def mc_vel(self, table, seed=None):
        r""" Method assigns a Monte Carlo realization of the Jeans velocity
        solution to the halos in the input ``table``.

        Parameters
        -----------
        table : Astropy Table
            `astropy.table.Table` object storing the halo catalog.
            Calling the `mc_vel` method will over-write the existing values of
            the ``vx``, ``vy`` and ``vz`` columns.

        seed : int, optional
            Random number seed used in the Monte Carlo realization.
            Default is None, which will produce stochastic results.

        """
        MonteCarloGalProf.mc_vel(self, table, seed=seed)

    def mc_generate_nfw_phase_space_points(
        self, Ngals=int(1e4), conc=5, mass=1e12, verbose=True, seed=None
    ):
        r""" Return a Monte Carlo realization of points
        in the phase space of an NFW halo in isotropic Jeans equilibrium.

        Parameters
        -----------
        Ngals : int, optional
            Number of galaxies in the Monte Carlo realization of the
            phase space distribution. Default is 1e4.

        conc : float, optional
            Concentration of the NFW profile being realized.
            Default is 5.

        mass : float, optional
            Mass of the halo whose phase space distribution is being realized
            in units of Msun/h. Default is 1e12.

        verbose : bool, optional
            If True, a message prints with an estimate of the build time.
            Default is True.

        seed : int, optional
            Random number seed used in the Monte Carlo realization.
            Default is None, which will produce stochastic results.

        Returns
        --------
        t : table
            `~astropy.table.Table` containing the Monte Carlo realization of the
            phase space distribution.
            Keys are 'x', 'y', 'z', 'vx', 'vy', 'vz', 'radial_position', 'radial_velocity'.
            Length units in Mpc/h, velocity units in km/s.

        Examples
        ---------
        >>> nfw = NFWPhaseSpace()
        >>> mass, conc = 1e13, 8.
        >>> data = nfw.mc_generate_nfw_phase_space_points(Ngals=100, mass=mass, conc=conc, verbose=False)

        Now suppose you wish to compute the radial velocity dispersion of all the returned points:

        >>> vrad_disp = np.std(data['radial_velocity'])

        If you wish to do the same calculation but for points in a specific range of radius:

        >>> mask = data['radial_position'] < 0.1
        >>> vrad_disp_inner_points = np.std(data['radial_velocity'][mask])

        You may also wish to select points according to their distance to the halo center
        in units of the virial radius. In such as case, you can use the
        `~halotools.empirical_models.NFWPhaseSpace.halo_mass_to_halo_radius`
        method to scale the halo-centric distances. Here is an example
        of how to compute the velocity dispersion in the z-dimension of all points
        residing within :math:`R_{\rm vir}/2`:

        >>> halo_radius = nfw.halo_mass_to_halo_radius(mass)
        >>> scaled_radial_positions = data['radial_position']/halo_radius
        >>> mask = scaled_radial_positions < 0.5
        >>> vz_disp_inner_half = np.std(data['vz'][mask])

        """
        m = np.atleast_1d(mass)
        c = np.atleast_1d(conc)
        if (len(m) > 1) & (len(c) > 1):
            assert len(m) == len(c), "Input ``mass`` and ``conc`` must have same length"
        elif len(m) > 1:
            Ngals = len(m)
            c = np.zeros(Ngals) + conc
        elif len(c) > 1:
            Ngals = len(c)
            m = np.zeros(Ngals) + mass
        else:
            c = np.zeros(Ngals) + conc
            m = np.zeros(Ngals) + mass

        rvir = NFWProfile.halo_mass_to_halo_radius(self, total_mass=m)

        x, y, z = MonteCarloGalProf.mc_halo_centric_pos(
            self, c, halo_radius=rvir, seed=seed
        )
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        scaled_radius = r / rvir

        if seed is not None:
            seed += 1
        vx = MonteCarloGalProf.mc_radial_velocity(self, scaled_radius, m, c, seed=seed)
        if seed is not None:
            seed += 1
        vy = MonteCarloGalProf.mc_radial_velocity(self, scaled_radius, m, c, seed=seed)
        if seed is not None:
            seed += 1
        vz = MonteCarloGalProf.mc_radial_velocity(self, scaled_radius, m, c, seed=seed)

        xrel, vxrel = _relative_positions_and_velocities(x, 0, v1=vx, v2=0)
        yrel, vyrel = _relative_positions_and_velocities(y, 0, v1=vy, v2=0)
        zrel, vzrel = _relative_positions_and_velocities(z, 0, v1=vz, v2=0)

        vrad = (xrel * vxrel + yrel * vyrel + zrel * vzrel) / r

        t = Table(
            {
                "x": x,
                "y": y,
                "z": z,
                "vx": vx,
                "vy": vy,
                "vz": vz,
                "radial_position": r,
                "radial_velocity": vrad,
            }
        )

        return t


def _sign_pbc(x1, x2, period=None, equality_fill_val=0.0, return_pbc_correction=False):
    x1 = np.atleast_1d(x1)
    x2 = np.atleast_1d(x2)
    result = np.sign(x1 - x2)

    if period is not None:
        try:
            assert np.all(x1 >= 0)
            assert np.all(x2 >= 0)
            assert np.all(x1 < period)
            assert np.all(x2 < period)
        except AssertionError:
            msg = "If period is not None, all values of x and y must be between [0, period)"
            raise ValueError(msg)

        d = np.abs(x1 - x2)
        pbc_correction = np.sign(period / 2.0 - d)
        result = pbc_correction * result

    if equality_fill_val != 0:
        result = np.where(result == 0, equality_fill_val, result)

    if return_pbc_correction:
        return result, pbc_correction
    else:
        return result


def _relative_positions_and_velocities(x1, x2, period=None, **kwargs):
    s = _sign_pbc(x1, x2, period=period, equality_fill_val=1.0)
    absd = np.abs(x1 - x2)
    if period is None:
        xrel = s * absd
    else:
        xrel = s * np.where(absd > period / 2.0, period - absd, absd)

    try:
        v1 = kwargs["v1"]
        v2 = kwargs["v2"]
        return xrel, s * (v1 - v2)
    except KeyError:
        return xrel
