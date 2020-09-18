"""
Module defining the `~halotools.empirical_models.BiasedNFWPhaseSpace` class
governing the phase space distribution of massless tracers of an NFW potential,
where the concentration of the tracers is permitted to differ from the
host halo concentration.
"""
from __future__ import division, print_function, absolute_import

import numpy as np
from warnings import warn

from .nfw_phase_space import NFWPhaseSpace
from .kernels import biased_dimless_vrad_disp

from ..... import model_defaults

__author__ = ("Andrew Hearin",)
__all__ = ("BiasedNFWPhaseSpace",)


lookup_table_performance_warning = (
    "You have selected {0} bins to digitize host halo concentration \n"
    "and {1} bins to digitize the galaxy bias parameter.\n"
    "To populate mocks, the BiasedNFWPhaseSpace class builds a lookup table with shape ({0}, {1}, {2}),\n"
    "one entry for every numerical solution to the Jeans equation.\n"
    "Using this fine of a binning requires a long pre-computation of {3} integrals\n."
    "Make sure you actually need to use so many bins"
)


class BiasedNFWPhaseSpace(NFWPhaseSpace):
    r""" Model for the phase space distribution of galaxies
    in isotropic Jeans equilibrium in an NFW halo profile,
    based on Navarro, Frenk and White (1995),
    where the concentration of the tracers is permitted to differ from the
    host halo concentration.

    For a review of the mathematics underlying the NFW profile,
    including descriptions of how the relevant equations are
    implemented in the Halotools code base, see :ref:`nfw_profile_tutorial`.
    """

    def __init__(self, profile_integration_tol=1e-5, **kwargs):
        r"""
        Parameters
        ----------
        conc_gal_bias_logM_abscissa : array_like, optional
            Numpy array of shape (num_gal_bias_bins, ) storing the values of
            log10(Mhalo). For each entry of ``conc_gal_bias_logM_abscissa``,
            there will be a corresponding parameter in the ``param_dict`` allowing
            you to vary the strength of the galaxy concentration bias,
            using log-linear interpolation for the intermediate values
            between these control points.

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

        conc_gal_bias_bins : ndarray, optional
            Array storing how biases in galaxy concentrations will be digitized
            when building a lookup table for mock-population purposes.
            The spacing of this array sets a limit on how accurately the
            galaxy bias parameter can be recovered in a likelihood analysis.

        profile_integration_tol : float, optional
            Default is 1e-5

        Examples
        ---------
        >>> biased_nfw = BiasedNFWPhaseSpace()

        The behavior of the `~halotools.empirical_models.BiasedNFWPhaseSpace`
        model is controlled by the values
        stored in its ``param_dict``. In the above default model, we have one parameter
        that controls the concentration of the satellites,
        ``conc_gal_bias``.
        The value of :math:`F_{\rm gal}` = ``conc_gal_bias`` sets the value for
        the concentration of satellites according to
        :math:`F_{\rm gal} * c_{\rm halo},` where :math:`c_{\rm halo}` is the
        value of the halo concentration specified by the concentration-mass model.


        By default, :math:`F_{\rm gal} = 1`. Satellites with *larger* values of
        :math:`F_{\rm gal}` will be *more* radially concentrated and have *smaller*
        radial velocity dispersions.

        The `BiasedNFWPhaseSpace` gives you the option to allow :math:`F_{\rm gal}`
        to vary with halo mass. You can accomplish this via the
        ``conc_gal_bias_logM_abscissa`` keyword:

        >>> biased_nfw_mass_dep = BiasedNFWPhaseSpace(conc_gal_bias_logM_abscissa=[12., 15.])

        In the above model, the values of :math:`F_{\rm gal} = F_{\rm gal}(M_{\rm halo})`
        can be independently specified at either of the two control points,
        :math:`10^{12}M_{\odot}` or :math:`10^{15}M_{\odot}`.
        For every element in the input ``conc_gal_bias_logM_abscissa``, there is a
        correposponding value in the ``param_dict`` controlling the value of
        :math:`F_{\rm gal}` at that mass.

        For example, in the ``biased_nfw_mass_dep`` defined above,
        to set the value of :math:`F_{\rm gal}(M_{\rm halo} = 10^{15})`:

        >>> biased_nfw_mass_dep.param_dict['conc_gal_bias_param1'] = 2

        To triple the value of :math:`F_{\rm gal}(M_{\rm halo} = 10^{12})`:

        >>> biased_nfw_mass_dep.param_dict['conc_gal_bias_param0'] *= 3

        Values of :math:`F_{\rm gal}` at masses
        between the control points will be determined by log-linear interpolation.
        When extrapolating :math:`F_{\rm gal}` beyond
        the specified range, the values will be kept constant at the end point values.
        """

        NFWPhaseSpace.__init__(self, **kwargs)
        self._profile_integration_tol = profile_integration_tol

        self.gal_prof_param_keys = ["conc_NFWmodel", "conc_gal_bias"]

        prof_lookup_bins = self._retrieve_prof_lookup_info(**kwargs)
        self.setup_prof_lookup_tables(*prof_lookup_bins)

        self._mock_generation_calling_sequence = [
            "calculate_conc_gal_bias",
            "assign_phase_space",
        ]
        self._methods_to_inherit = ["calculate_conc_gal_bias"]

        self._galprop_dtypes_to_allocate = np.dtype(
            [
                ("host_centric_distance", "f8"),
                ("x", "f8"),
                ("y", "f8"),
                ("z", "f8"),
                ("vx", "f8"),
                ("vy", "f8"),
                ("vz", "f8"),
                ("conc_gal_bias", "f8"),
                ("conc_galaxy", "f8"),
            ]
        )

        self.param_dict = self._initialize_conc_bias_param_dict(**kwargs)

    def _initialize_conc_bias_param_dict(self, **kwargs):
        r""" Set up the appropriate number of keys in the parameter dictionary
        and give the keys standardized names.
        """
        if "conc_gal_bias_logM_abscissa" in list(kwargs.keys()):
            _conc_bias_logM_abscissa = np.atleast_1d(
                kwargs.get("conc_gal_bias_logM_abscissa")
            ).astype("f4")
            d = {
                "conc_gal_bias_param" + str(i): 1.0
                for i in range(len(_conc_bias_logM_abscissa))
            }
            d2 = {
                "conc_gal_bias_logM_abscissa_param" + str(i): float(logM)
                for i, logM in enumerate(_conc_bias_logM_abscissa)
            }
            self._num_conc_bias_params = len(_conc_bias_logM_abscissa)
            for key, value in d2.items():
                d[key] = value
            return d

        else:
            return {"conc_gal_bias": 1.0}

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

        conc_gal_bias_arr = kwargs.get(
            "conc_gal_bias_bins", model_defaults.default_conc_gal_bias_bins
        )

        npts_conc, npts_gal_bias = len(conc_arr), len(conc_gal_bias_arr)
        if npts_conc * npts_gal_bias > 300:
            args = (
                npts_conc,
                npts_gal_bias,
                model_defaults.Npts_radius_table,
                npts_conc * npts_gal_bias * model_defaults.Npts_radius_table,
            )
            warn(lookup_table_performance_warning.format(*args))

        return [conc_arr, conc_gal_bias_arr]

    def conc_NFWmodel(self, **kwargs):
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
        halo_conc : array_like
            Concentrations of the halos.

            Note that concentrations will be clipped to their min/max permitted
            values set in the `~halotools.empirical_models.model_defaults module`.
            For unclipped concentrations, set these variables to -np.inf, +np.inf.

        """
        return NFWPhaseSpace.conc_NFWmodel(self, **kwargs)

    def _clipped_galaxy_concentration(self, halo_conc, conc_gal_bias):
        r""" Private method used to ensure that biased concentrations are still
        bound by the min/max bounds permitted by the lookup tables.
        """
        gal_conc = conc_gal_bias * halo_conc

        try:
            cmin = self._conc_NFWmodel_lookup_table_min
            cmax = self._conc_NFWmodel_lookup_table_max
        except AttributeError:
            cmin = model_defaults.min_permitted_conc
            cmax = model_defaults.max_permitted_conc

        # Now clip the galaxy concentration as necessary
        gal_conc = np.where(gal_conc < cmin, cmin, gal_conc)
        gal_conc = np.where(gal_conc > cmax, cmax, gal_conc)
        return gal_conc

    def cumulative_gal_PDF(self, scaled_radius, halo_conc, conc_gal_bias):
        r""" Analogous to `cumulative_mass_PDF`, but for the satellite galaxy distribution
        instead of the host halo mass distribution.

        Parameters
        -------------
        scaled_radius : array_like
            Halo-centric distance *r* scaled by the halo boundary :math:`R_{\Delta}`, so that
            :math:`0 <= \tilde{r} \equiv r/R_{\Delta} <= 1`. Can be a scalar or numpy array.

        halo_conc : array_like
            Value of the halo concentration. Can either be a scalar, or a numpy array
            of the same dimension as the input ``scaled_radius``.

        conc_gal_bias : array_like
            Ratio of the galaxy concentration to the halo concentration,
            :math:`c_{\rm gal}/c_{\rm halo}`.

        Returns
        -------------
        p: array_like
            The fraction of the total mass enclosed
            within the input ``scaled_radius``, in :math:`M_{\odot}/h`;
            has the same dimensions as the input ``scaled_radius``.

        Examples
        --------
        >>> model = BiasedNFWPhaseSpace()

        >>> scaled_radius = 0.5  # units of Rvir
        >>> halo_conc = 5
        >>> conc_gal_bias = 3
        >>> result1 = model.cumulative_gal_PDF(scaled_radius, halo_conc, conc_gal_bias)

        >>> num_halos = 50
        >>> scaled_radius = np.logspace(-2, 0, num_halos)
        >>> halo_conc = np.linspace(1, 25, num_halos)
        >>> conc_gal_bias = np.zeros(num_halos) + 2.
        >>> result2 = model.cumulative_gal_PDF(scaled_radius, halo_conc, conc_gal_bias)
        """
        gal_conc = self._clipped_galaxy_concentration(halo_conc, conc_gal_bias)
        return NFWPhaseSpace.cumulative_mass_PDF(self, scaled_radius, gal_conc)

    def cumulative_mass_PDF(self, scaled_radius, halo_conc):
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

        halo_conc : array_like
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
        >>> halo_conc = 5
        >>> result = model.cumulative_mass_PDF(scaled_radius, halo_conc)
        >>> halo_conc_arr = np.linspace(1, 100, Npts)
        >>> result = model.cumulative_mass_PDF(scaled_radius, halo_conc_arr)

        Notes
        ------
        See :ref:`halo_profile_definitions` for derivations and implementation details.
        """
        return NFWPhaseSpace.cumulative_mass_PDF(self, scaled_radius, halo_conc)

    def dimensionless_radial_velocity_dispersion(
        self, scaled_radius, halo_conc, conc_gal_bias
    ):
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

        halo_conc : float
            Concentration of the halo.

        Returns
        -------
        result : array_like
            Radial velocity dispersion profile scaled by the virial velocity.
            The returned result has the same dimension as the input ``scaled_radius``.
        """
        gal_conc = self._clipped_galaxy_concentration(halo_conc, conc_gal_bias)
        return biased_dimless_vrad_disp(
            scaled_radius,
            halo_conc,
            gal_conc,
            profile_integration_tol=self._profile_integration_tol,
        )

    def radial_velocity_dispersion(self, radius, total_mass, halo_conc, conc_gal_bias):
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

        conc_gal_bias : array_like
            Ratio of the galaxy concentration to the halo concentration,
            :math:`c_{\rm gal}/c_{\rm halo}`.

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
            scaled_radius, halo_conc, conc_gal_bias
        )
        return dimensionless_velocities * virial_velocities

    def calculate_conc_gal_bias(self, seed=None, **kwargs):
        r""" Calculate the ratio of the galaxy concentration to the halo concentration,
        :math:`c_{\rm gal}/c_{\rm halo}`.

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
        conc_gal_bias : array_like
            Ratio of the galaxy concentration to the halo concentration,
            :math:`c_{\rm gal}/c_{\rm halo}`.
        """
        if "table" in list(kwargs.keys()):
            table = kwargs["table"]
            mass = table[self.prim_haloprop_key]
        elif "prim_haloprop" in list(kwargs.keys()):
            mass = np.atleast_1d(kwargs["prim_haloprop"]).astype("f4")
        else:
            msg = (
                "\nYou must pass either a ``table`` or ``prim_haloprop`` argument \n"
                "to the ``assign_conc_gal_bias`` function of the ``BiasedNFWPhaseSpace`` class.\n"
            )
            raise KeyError(msg)

        if "conc_gal_bias_logM_abscissa_param0" in self.param_dict.keys():
            abscissa_keys = list(
                "conc_gal_bias_logM_abscissa_param" + str(i)
                for i in range(self._num_conc_bias_params)
            )
            abscissa = [self.param_dict[key] for key in abscissa_keys]

            ordinates_keys = list(
                "conc_gal_bias_param" + str(i)
                for i in range(self._num_conc_bias_params)
            )
            ordinates = [self.param_dict[key] for key in ordinates_keys]

            result = np.interp(np.log10(mass), abscissa, ordinates)
        else:
            result = np.zeros_like(mass) + self.param_dict["conc_gal_bias"]

        if "table" in list(kwargs.keys()):
            table["conc_gal_bias"][:] = result
            halo_conc = table["conc_NFWmodel"]
            gal_conc = self._clipped_galaxy_concentration(halo_conc, result)
            table["conc_galaxy"][:] = gal_conc
        else:
            return result
