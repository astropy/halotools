r"""
This module contains occupation components used by the Leauthaud11 composite model.
"""

import numpy as np
import math
from scipy.special import erf
import warnings

from .occupation_model_template import OccupationComponent

from .. import model_defaults
from ..assembias_models import HeavisideAssembias

from ... import sim_manager
from ...custom_exceptions import HalotoolsError

from ..component_model_templates import PrimGalpropModel
from ..smhm_models.smhm_helpers import safely_retrieve_redshift
from .. import model_helpers


L11_LITTLEH = 0.72
L11_LGH = np.log10(L11_LITTLEH)

__all__ = (
    "Leauthaud11Cens",
    "Leauthaud11Sats",
    "AssembiasLeauthaud11Cens",
    "AssembiasLeauthaud11Sats",
)


class Leauthaud11Cens(OccupationComponent):
    r"""HOD-style model for any central galaxy occupation that derives from
    a stellar-to-halo-mass relation.

    .. note::

        The `Leauthaud11Cens` model is part of the ``leauthaud11``
        prebuilt composite HOD-style model. For a tutorial on the ``leauthaud11``
        composite model, see :ref:`leauthaud11_composite_model`.

    """

    def __init__(
        self,
        threshold=model_defaults.default_stellar_mass_threshold,
        prim_haloprop_key=model_defaults.prim_haloprop_key,
        redshift=sim_manager.sim_defaults.default_redshift,
        **kwargs
    ):
        r"""
        Parameters
        ----------
        threshold : float, optional
            Stellar mass threshold of the mock galaxy sample in h=1 solar mass units.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

            Values in the Leauthaud11 parameter dictionary are quoted assuming h=0.72,
            so that a direct comparison can be made to the best-fitting values quoted in
            Leauthaud+11. However, the threshold of the sample in halotools
            is defined assuming h=1. This means that in order to compare your
            parameter dictionary to the best-fitting parameters in Leauthaud+11,
            you will need to compare to the appropriately scaled threshold.
            For example, in Figure 2 of arXiv:1103.2077, the most massive sample
            is labeled logsm>11.4. In Halotools, this corresponds to threshold=11.115.

        prim_haloprop_key : string, optional
            String giving the column name of the primary halo property governing
            the occupation statistics of gal_type galaxies.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        redshift : float, optional
            Redshift of the stellar-to-halo-mass relation.
            Default is set in `~halotools.sim_manager.sim_defaults`.

        Examples
        --------
        >>> cen_model = Leauthaud11Cens()
        >>> cen_model = Leauthaud11Cens(threshold = 11.25)
        >>> cen_model = Leauthaud11Cens(prim_haloprop_key = 'halo_m200b')

        """
        upper_occupation_bound = 1.0

        # Call the super class constructor, which binds all the
        # arguments to the instance.
        super(Leauthaud11Cens, self).__init__(
            gal_type="centrals",
            threshold=threshold,
            upper_occupation_bound=upper_occupation_bound,
            prim_haloprop_key=prim_haloprop_key,
            **kwargs
        )
        self.redshift = redshift

        self.smhm_model = Leauthaud11SmHm(prim_haloprop_key=prim_haloprop_key, **kwargs)

        for key, value in self.smhm_model.param_dict.items():
            self.param_dict[key] = value

        self._methods_to_inherit = [
            "mc_occupation",
            "mean_occupation",
            "mean_stellar_mass",
            "mean_log_halo_mass",
        ]

        self.publications = ["arXiv:1103.2077", "arXiv:1104.0928"]
        self.publications.extend(self.smhm_model.publications)
        self.publications = list(set(self.publications))

    def get_published_parameters(self):
        r"""Return the values of ``self.param_dict`` according to
        the SIG_MOD1 values of Table 5 of arXiv:1104.0928 for the
        lowest redshift bin.

        """
        d = {}
        d["smhm_m1_0"] = 12.52
        d["smhm_m0_0"] = 10.916
        d["smhm_beta_0"] = 0.457
        d["smhm_delta_0"] = 0.566
        d["smhm_gamma_0"] = 1.54
        d["scatter_model_param1"] = 0.206
        return d

    def mean_occupation(self, **kwargs):
        r"""Expected number of central galaxies in a halo.
        See Equation 8 of arXiv:1103.2077.

        Parameters
        ----------
        prim_haloprop : array, optional
            Array of mass-like variable upon which occupation statistics are based.
            If ``prim_haloprop`` is not passed, then ``table`` keyword argument must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop`` keyword argument must be passed.

        Returns
        -------
        mean_ncen : array
            Mean number of central galaxies in the halo of the input mass.

        Notes
        -----
        Assumes constant scatter in the stellar-to-halo-mass relation.
        """
        for key, value in self.param_dict.items():
            if key in list(self.smhm_model.param_dict.keys()):
                self.smhm_model.param_dict[key] = value

        logmstar_h1p0 = np.log10(
            self.smhm_model.mean_stellar_mass(redshift=self.redshift, **kwargs)
        )
        logmstar_h0p72 = logmstar_h1p0 - 2 * L11_LGH

        logscatter = math.sqrt(2) * self.smhm_model.mean_scatter(**kwargs)

        threshold_h0p72 = self.threshold - 2 * L11_LGH
        mean_ncen = 0.5 * (1.0 - erf((threshold_h0p72 - logmstar_h0p72) / logscatter))

        return mean_ncen

    def mean_stellar_mass(self, **kwargs):
        r"""Return the stellar mass of a central galaxy as a function
        of the input table.

        Parameters
        ----------
        prim_haloprop : array, optional
            Array of mass-like variable upon which occupation statistics are based.
            If ``prim_haloprop`` is not passed, then ``table`` keyword argument must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop`` keyword argument must be passed.

        Returns
        -------
        mstar_h1p0 : array_like
            Array containing stellar masses in units of h=1

            Note that throughout Leauthaud+11 it is assumed that h=0.72.
            As a sanity check on your conversion:
            mstar_h0p72 = mstar_h1p0/0.5184
            So that mstar_h0p72 is larger than mstar_h1p0

        """

        for key, value in self.param_dict.items():
            if key in self.smhm_model.param_dict:
                self.smhm_model.param_dict[key] = value
        mstar_h1p0 = self.smhm_model.mean_stellar_mass(redshift=self.redshift, **kwargs)
        return mstar_h1p0

    def mean_log_halo_mass(self, log_stellar_mass_h1p0):
        r"""Return the base-10 logarithm of the halo mass of a central galaxy as a function
        of the base-10 logarithm of the input stellar mass.

        Parameters
        ----------
        log_stellar_mass_h1p0 : array
            Array of base-10 logarithm of stellar masses in h=1 solar mass units.

            Note that throughout Leauthaud+11 it is assumed that h=0.72.
            As a sanity check on your conversion:
            logsm_h0p72 = logsm_h1p0 - 2*log10(0.72)
            So that logsm_h0p72 is larger than logsm_h1p0

        Returns
        -------
        log_halo_mass_h1p0 : array_like
            Array containing 10-base logarithm of halo mass in h=1 solar mass units.

            Note that throughout Leauthaud+11 it is assumed that h=0.72.
            As a sanity check on your conversion:
            log_halo_mass_h0p72 = log_halo_mass_h1p0 - log10(0.72)
            So that log_halo_mass_h0p72 is larger than log_halo_mass_h1p0

        """
        for key, value in self.param_dict.items():
            if key in self.smhm_model.param_dict:
                self.smhm_model.param_dict[key] = value
        log_halo_mass_h1p0 = self.smhm_model.mean_log_halo_mass(
            log_stellar_mass_h1p0, redshift=self.redshift
        )
        return log_halo_mass_h1p0


class Leauthaud11Sats(OccupationComponent):
    r"""HOD-style model for any satellite galaxy occupation that derives from
    a stellar-to-halo-mass relation.

    .. note::

        The `Leauthaud11Sats` model is part of the ``leauthaud11``
        prebuilt composite HOD-style model. For a tutorial on the ``leauthaud11``
        composite model, see :ref:`leauthaud11_composite_model`.
    """

    def __init__(
        self,
        threshold=model_defaults.default_stellar_mass_threshold,
        prim_haloprop_key=model_defaults.prim_haloprop_key,
        redshift=sim_manager.sim_defaults.default_redshift,
        modulate_with_cenocc=True,
        cenocc_model=None,
        **kwargs
    ):
        r"""
        Parameters
        ----------
        threshold : float, optional
            Stellar mass threshold of the mock galaxy sample in h=1 solar mass units.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

            Values in the Leauthaud11 parameter dictionary are quoted assuming h=0.72,
            so that a direct comparison can be made to the best-fitting values quoted in
            Leauthaud+11. However, the threshold of the sample in halotools
            is defined assuming h=1. This means that in order to compare your
            parameter dictionary to the best-fitting parameters in Leauthaud+11,
            you will need to compare to the appropriately scaled threshold.
            For example, in Figure 2 of arXiv:1103.2077, the most massive sample
            is labeled logsm>11.4. In Halotools, this corresponds to threshold=11.115.


        prim_haloprop_key : string, optional
            String giving the column name of the primary halo property governing
            the occupation statistics of gal_type galaxies.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        redshift : float, optional
            Redshift of the stellar-to-halo-mass relation.
            Default is set in `~halotools.sim_manager.sim_defaults`.

        modulate_with_cenocc : bool, optional
            If True, the first satellite moment will be multiplied by the
            the first central moment. Default is True.

        cenocc_model : `OccupationComponent`, optional
            If the ``cenocc_model`` keyword argument is set to its default value
            of None, then the :math:`\langle N_{\mathrm{cen}}\rangle_{M}` prefactor will be
            calculated according to `leauthaud11Cens.mean_occupation`.
            However, if an instance of the `OccupationComponent` class is instead
            passed in via the ``cenocc_model`` keyword,
            then the first satellite moment will be multiplied by
            the ``mean_occupation`` function of the ``cenocc_model``.
            The ``modulate_with_cenocc`` keyword must be set to True in order
            for the ``cenocc_model`` to be operative.
            See :ref:`zheng07_using_cenocc_model_tutorial` for further details.

        Examples
        --------
        >>> sat_model = Leauthaud11Sats()
        """

        if cenocc_model is None:
            cenocc_model = Leauthaud11Cens(
                prim_haloprop_key=prim_haloprop_key, threshold=threshold
            )
        else:
            if modulate_with_cenocc is False:
                msg = (
                    "You chose to input a ``cenocc_model``, but you set the \n"
                    "``modulate_with_cenocc`` keyword to False, so your "
                    "``cenocc_model`` will have no impact on the model's behavior.\n"
                    "Be sure this is what you intend before proceeding.\n"
                    "Refer to the Leauthand et al. (2011) composite model tutorial for details.\n"
                )
                warnings.warn(msg)

        self.modulate_with_cenocc = modulate_with_cenocc

        if self.modulate_with_cenocc:
            try:
                assert isinstance(cenocc_model, OccupationComponent)
            except AssertionError:
                msg = (
                    "The input ``cenocc_model`` must be an instance of \n"
                    "``OccupationComponent`` or one of its sub-classes.\n"
                )
                raise HalotoolsError(msg)

        self.central_occupation_model = cenocc_model

        super(Leauthaud11Sats, self).__init__(
            gal_type="satellites",
            threshold=threshold,
            upper_occupation_bound=float("inf"),
            prim_haloprop_key=prim_haloprop_key,
            **kwargs
        )

        self.redshift = redshift

        self._initialize_param_dict()

        self.param_dict.update(self.central_occupation_model.param_dict)

        self.publications = self.central_occupation_model.publications

    def mean_occupation(self, **kwargs):
        r"""Expected number of satellite galaxies in a halo of mass halo_mass.
        See Equation 12-14 of arXiv:1103.2077.

        Parameters
        ----------
        prim_haloprop : array, optional
            array of masses of table in the catalog

        table : object, optional
            Data table storing halo catalog.

        Returns
        -------
        mean_nsat : array
            Mean number of satellite galaxies in the halo of the input mass.

        Examples
        --------
        >>> sat_model = Leauthaud11Sats()
        >>> mean_nsat = sat_model.mean_occupation(prim_haloprop = 1.e13)

        Notes
        -----
        Assumes constant scatter in the stellar-to-halo-mass relation.
        """
        # Retrieve the array storing the mass-like variable
        # Halotools assumes your input values of mass are supplied assuming h=1
        if "table" in list(kwargs.keys()):
            halo_mass_h1p0 = kwargs["table"][self.prim_haloprop_key]
        elif "prim_haloprop" in list(kwargs.keys()):
            halo_mass_h1p0 = np.atleast_1d(kwargs["prim_haloprop"])
        else:
            raise KeyError(
                "Must pass one of the following keyword arguments "
                "to mean_occupation:\n``table`` or ``prim_haloprop``"
            )

        # Convert halo mass to a numerical value that assumes h=0.72
        halo_mass_h0p72 = halo_mass_h1p0 / L11_LITTLEH
        # Henceforth, the implemented formulas have an identical form
        # to the formulas that appear in Leauthaud+11

        self._update_satellite_params()

        mean_nsat = (
            np.exp(-self._mcut / halo_mass_h0p72)
            * (halo_mass_h0p72 / self._msat) ** self.param_dict["alphasat"]
        )

        if self.modulate_with_cenocc is True:
            mean_nsat *= self.central_occupation_model.mean_occupation(**kwargs)

        return mean_nsat

    def _initialize_param_dict(self):
        """Set the initial values of ``self.param_dict`` according to
        the SIG_MOD1 values of Table 5 of arXiv:1104.0928 for the
        lowest redshift bin.

        """

        self.param_dict["alphasat"] = 1.0
        self.param_dict["bsat"] = 10.62
        self.param_dict["bcut"] = 1.47
        self.param_dict["betacut"] = -0.13
        self.param_dict["betasat"] = 0.859

        for key, value in self.central_occupation_model.param_dict.items():
            self.param_dict[key] = value

        self._update_satellite_params()

    def _update_satellite_params(self):
        """Private method to update the model parameters."""
        for key, value in self.param_dict.items():
            if key in self.central_occupation_model.param_dict:
                self.central_occupation_model.param_dict[key] = value

        log_halo_mass_threshold_h1p0 = self.central_occupation_model.mean_log_halo_mass(
            log_stellar_mass_h1p0=self.threshold
        )
        log_halo_mass_threshold_h0p72 = log_halo_mass_threshold_h1p0 - L11_LGH
        knee_threshold_h0p72 = 10.0**log_halo_mass_threshold_h0p72

        # 1e12 is the numerical value used in Leauthaud+11 and so assumes h=0.72
        knee_mass_h0p72 = 1.0e12

        self._msat = (
            knee_mass_h0p72
            * self.param_dict["bsat"]
            * (knee_threshold_h0p72 / knee_mass_h0p72) ** self.param_dict["betasat"]
        )

        self._mcut = (
            knee_mass_h0p72
            * self.param_dict["bcut"]
            * (knee_threshold_h0p72 / knee_mass_h0p72) ** self.param_dict["betacut"]
        )


class AssembiasLeauthaud11Cens(Leauthaud11Cens, HeavisideAssembias):
    """Assembly-biased modulation of `Leauthaud11Cens`."""

    def __init__(self, **kwargs):
        r"""
        Parameters
        ----------
        threshold : float, optional
            Stellar mass threshold of the mock galaxy sample in h=1 solar mass units.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

            Values in the Leauthaud11 parameter dictionary are quoted assuming h=0.72,
            so that a direct comparison can be made to the best-fitting values quoted in
            Leauthaud+11. However, the threshold of the sample in halotools
            is defined assuming h=1. This means that in order to compare your
            parameter dictionary to the best-fitting parameters in Leauthaud+11,
            you will need to compare to the appropriately scaled threshold.
            For example, in Figure 2 of arXiv:1103.2077, the most massive sample
            is labeled logsm>11.4. In Halotools, this corresponds to threshold=11.115.

        prim_haloprop_key : string, optional
            String giving the column name of the primary halo property governing
            the occupation statistics of gal_type galaxies.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        sec_haloprop_key : string, optional
            String giving the column name of the secondary halo property
            governing the assembly bias. Must be a key in the table
            passed to the methods of `HeavisideAssembiasComponent`.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        redshift : float, optional
            Redshift of the stellar-to-halo-mass relation.
            Default is set in the `~halotools.sim_manager.sim_defaults` module.

        split : float or list, optional
            Fraction or list of fractions between 0 and 1 defining how
            we split halos into two groupings based on
            their conditional secondary percentiles.
            Default is 0.5 for a constant 50/50 split.

        split_abscissa : list, optional
            Values of the primary halo property at which the halos are split as described above in
            the ``split`` argument. If ``loginterp`` is set to True (the default behavior),
            the interpolation will be done in the logarithm of the primary halo property.
            Default is to assume a constant 50/50 split.

        assembias_strength : float or list, optional
            Fraction or sequence of fractions between -1 and 1
            defining the assembly bias correlation strength.
            Default is 0.5.

        assembias_strength_abscissa : list, optional
            Values of the primary halo property at which the assembly bias strength is specified.
            Default is to assume a constant strength of 0.5. If passing a list, the strength
            will interpreted at the input ``assembias_strength_abscissa``.
            Default is to assume a constant strength of 0.5.

        """
        Leauthaud11Cens.__init__(self, **kwargs)
        HeavisideAssembias.__init__(
            self,
            lower_assembias_bound=self._lower_occupation_bound,
            upper_assembias_bound=self._upper_occupation_bound,
            method_name_to_decorate="mean_occupation",
            **kwargs
        )


class AssembiasLeauthaud11Sats(Leauthaud11Sats, HeavisideAssembias):
    """Assembly-biased modulation of `Leauthaud11Sats`."""

    def __init__(self, **kwargs):
        r"""
        Parameters
        ----------
        threshold : float, optional
            Stellar mass threshold of the mock galaxy sample in h=1 solar mass units.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

            Values in the Leauthaud11 parameter dictionary are quoted assuming h=0.72,
            so that a direct comparison can be made to the best-fitting values quoted in
            Leauthaud+11. However, the threshold of the sample in halotools
            is defined assuming h=1. This means that in order to compare your
            parameter dictionary to the best-fitting parameters in Leauthaud+11,
            you will need to compare to the appropriately scaled threshold.
            For example, in Figure 2 of arXiv:1103.2077, the most massive sample
            is labeled logsm>11.4. In Halotools, this corresponds to threshold=11.115.

        prim_haloprop_key : string, optional
            String giving the column name of the primary halo property governing
            the occupation statistics of gal_type galaxies.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        sec_haloprop_key : string, optional
            String giving the column name of the secondary halo property
            governing the assembly bias. Must be a key in the table
            passed to the methods of `HeavisideAssembiasComponent`.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        redshift : float, optional
            Redshift of the stellar-to-halo-mass relation.
            Default is set in `~halotools.sim_manager.sim_defaults`.

        split : float or list, optional
            Fraction or list of fractions between 0 and 1 defining how
            we split halos into two groupings based on
            their conditional secondary percentiles.
            Default is 0.5 for a constant 50/50 split.

        split_abscissa : list, optional
            Values of the primary halo property at which the halos are split as described above in
            the ``split`` argument. If ``loginterp`` is set to True (the default behavior),
            the interpolation will be done in the logarithm of the primary halo property.
            Default is to assume a constant 50/50 split.

        assembias_strength : float or list, optional
            Fraction or sequence of fractions between -1 and 1
            defining the assembly bias correlation strength.
            Default is 0.5.

        assembias_strength_abscissa : list, optional
            Values of the primary halo property at which the assembly bias strength is specified.
            Default is to assume a constant strength of 0.5. If passing a list, the strength
            will interpreted at the input ``assembias_strength_abscissa``.
            Default is to assume a constant strength of 0.5.

        """
        Leauthaud11Sats.__init__(self, **kwargs)
        HeavisideAssembias.__init__(
            self,
            lower_assembias_bound=self._lower_occupation_bound,
            upper_assembias_bound=self._upper_occupation_bound,
            method_name_to_decorate="mean_occupation",
            **kwargs
        )


class Leauthaud11SmHm(PrimGalpropModel):
    """Stellar-to-halo-mass relation based on
    `Behroozi et al 2010 <http://arxiv.org/abs/astro-ph/1001.0015/>`_
    and adapted in Leauthaud+11.

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
            implemented by the `~halotools.empirical_models.LogNormalScatterModel` class.

        scatter_abscissa : array_like, optional
            Array of values giving the abscissa at which
            the level of scatter will be specified by the input ordinates.
            Default behavior will result in constant scatter at a level set in the
            `~halotools.empirical_models.model_defaults` module.

        scatter_ordinates : array_like, optional
            Array of values defining the level of scatter at the input abscissa.
            Default behavior will result in constant scatter at a level set in the
            `~halotools.empirical_models.model_defaults` module.

        redshift : float, optional
            Redshift of the stellar-to-halo-mass relation. Recommended default behavior
            is to leave this argument unspecified.

            If no ``redshift`` argument is given to the constructor, you will be free to use the
            analytical relations bound to `Leauthaud11SmHm` to study the redshift-dependence
            of the SMHM by passing in a ``redshift`` argument to the `mean_log_halo_mass`
            and `mean_stellar_mass` methods.

            If you do pass a ``redshift`` argument to the constructor, the instance of the
            `Leauthaud11SmHm` will only return results for this redshift, and will raise an
            exception if you attempt to pass in a redshift to these methods.
            See the Notes below to understand the motivation for this behavior.

        Notes
        ------
        Note that the `Leauthaud11SmHm` class is a nearly identical copy of the
        distinct from the `Behroozi10` model, except for the treatment of little h.

        """

        super(Leauthaud11SmHm, self).__init__(galprop_name="stellar_mass", **kwargs)

        self._methods_to_inherit.extend(["mean_log_halo_mass"])

        self.publications = ["arXiv:1001.0015"]

    def retrieve_default_param_dict(self):
        """Method returns a dictionary of all model parameters
        set to the column 2 values in Table 2 of Behroozi et al. (2010).

        Returns
        -------
        d : dict
            Dictionary containing parameter values.
        """
        # All calculations are done internally using the same h=0.72 units
        # as in Leauthaud et al. (2011), so the parameter values here are
        # the same as in Table 2, even though the mean_log_halo_mass and
        # mean_stellar_mass methods use accept and return arguments in h=1 units.

        d = {
            "smhm_m0_0": 10.72,
            "smhm_m0_a": 0.59,
            "smhm_m1_0": 12.35,
            "smhm_m1_a": 0.3,
            "smhm_beta_0": 0.43,
            "smhm_beta_a": 0.18,
            "smhm_delta_0": 0.56,
            "smhm_delta_a": 0.18,
            "smhm_gamma_0": 1.54,
            "smhm_gamma_a": 2.52,
        }

        return d

    def mean_log_halo_mass(self, log_stellar_mass_h1p0, **kwargs):
        """Return the halo mass of a central galaxy as a function
        of the stellar mass.

        Parameters
        ----------
        log_stellar_mass_h1p0 : array
            Array of base-10 logarithm of stellar masses in h=1 solar mass units.

            Note that throughout Leauthaud+11 it is assumed that h=0.72.
            As a sanity check on your conversion:
            logsm_h0p72 = logsm_h1p0 - 2*log10(0.72)
            So that logsm_h0p72 is larger than logsm_h1p0

        redshift : float or array, optional
            Redshift of the halo hosting the galaxy. If passing an array,
            must be of the same length as the input ``log_stellar_mass``.
            Default is set in `~halotools.sim_manager.sim_defaults`.

        Returns
        -------
        log_halo_mass_h1p0 : array_like
            Array containing 10-base logarithm of halo mass in h=1 solar mass units.

            Note that throughout Leauthaud+11 it is assumed that h=0.72.
            As a sanity check on your conversion:
            log_halo_mass_h0p72 = log_halo_mass_h1p0 - log10(0.72)
            So that log_halo_mass_h0p72 is larger than log_halo_mass_h1p0

        Notes
        ------
        The parameter values in Behroozi+10 were fit to data assuming h=0.7,
        but all halotools inputs are in h=1 units. Thus we will transform our
        input stellar mass to h=0.7 units, evaluate using the behroozi parameters,
        and then transform back to h=1 units before returning the result.
        """
        redshift = safely_retrieve_redshift(self, "mean_log_halo_mass", **kwargs)

        # convert mass from h=1 to h=0.72
        stellar_mass_h1p0 = 10.0**log_stellar_mass_h1p0
        stellar_mass_h0p72 = stellar_mass_h1p0 / (L11_LITTLEH**2)
        a = 1.0 / (1.0 + redshift)

        logm0 = self.param_dict["smhm_m0_0"] + self.param_dict["smhm_m0_a"] * (a - 1)
        m0 = 10.0**logm0
        logm1 = self.param_dict["smhm_m1_0"] + self.param_dict["smhm_m1_a"] * (a - 1)
        beta = self.param_dict["smhm_beta_0"] + self.param_dict["smhm_beta_a"] * (a - 1)
        delta = self.param_dict["smhm_delta_0"] + self.param_dict["smhm_delta_a"] * (
            a - 1
        )
        gamma = self.param_dict["smhm_gamma_0"] + self.param_dict["smhm_gamma_a"] * (
            a - 1
        )

        stellar_mass_by_m0 = stellar_mass_h0p72 / m0
        term3_numerator = (stellar_mass_by_m0) ** delta
        term3_denominator = 1 + (stellar_mass_by_m0) ** (-gamma)

        log_halo_mass_h0p72 = (
            logm1
            + beta * np.log10(stellar_mass_by_m0)
            + (term3_numerator / term3_denominator)
            - 0.5
        )

        # convert from h=0.72 back to h=1 and return the result
        log_halo_mass_h1p0 = log_halo_mass_h0p72 + L11_LGH
        return log_halo_mass_h1p0

    def mean_stellar_mass(self, **kwargs):
        """Return the stellar mass of a central galaxy as a function
        of the input table.

        Parameters
        ----------
        prim_haloprop : array, optional
            Array of mass-like variable upon which occupation statistics are based.
            If ``prim_haloprop`` is not passed, then ``table`` keyword argument must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop`` keyword argument must be passed.

        redshift : float or array, optional
            Redshift of the halo hosting the galaxy. If passing an array,
            must be of the same length as the input ``stellar_mass``.
            Default is set in `~halotools.sim_manager.sim_defaults`.

        Returns
        -------
        mstar_h1p0 : array_like
            Array containing stellar masses in units of h=1

            Note that throughout Leauthaud+11 it is assumed that h=0.72.
            As a sanity check on your conversion:
            mstar_h0p72 = mstar_h1p0/0.5184
            So that mstar_h0p72 is larger than mstar_h1p0

        """
        redshift = safely_retrieve_redshift(self, "mean_stellar_mass", **kwargs)

        # Retrieve the array storing the mass-like variable
        if "table" in list(kwargs.keys()):
            halo_mass_h1p0 = kwargs["table"][self.prim_haloprop_key]
        elif "prim_haloprop" in list(kwargs.keys()):
            halo_mass_h1p0 = kwargs["prim_haloprop"]
        else:
            raise KeyError(
                "Must pass one of the following keyword arguments "
                "to mean_occupation:\n``table`` or ``prim_haloprop``"
            )

        log_stellar_mass_table_h1p0 = np.linspace(8.5, 12.5, 100)
        log_halo_mass_table_h1p0 = self.mean_log_halo_mass(
            log_stellar_mass_table_h1p0, redshift=redshift
        )

        if not np.all(np.isfinite(log_halo_mass_table_h1p0)):
            msg = (
                "The value of the mean_stellar_mass function in the Leauthuad+11 model \n"
                "is calculated by numerically inverting results "
                "from the mean_log_halo_mass function.\nThese lookup tables "
                "have infinite-valued entries, which may lead to incorrect results.\n"
                "This is likely caused by the values of one or more of the model parameters "
                "being set to unphysically large/small values."
            )
            warnings.warn(msg)

        interpol_func_h1p0 = model_helpers.custom_spline(
            log_halo_mass_table_h1p0, log_stellar_mass_table_h1p0
        )

        log_stellar_mass_h1p0 = interpol_func_h1p0(np.log10(halo_mass_h1p0))
        mstar_h1p0 = 10.0**log_stellar_mass_h1p0

        return mstar_h1p0
