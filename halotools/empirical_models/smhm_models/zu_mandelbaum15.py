"""
Module containing classes used to model the mapping between
stellar mass and halo mass based on Zu & Mandelbaum et al. (2015).
"""
from __future__ import division, print_function, absolute_import, unicode_literals
import numpy as np
from warnings import warn
from astropy.utils.misc import NumpyRNGContext

from ..component_model_templates import PrimGalpropModel

__all__ = ("ZuMandelbaum15SmHm",)


class ZuMandelbaum15SmHm(PrimGalpropModel):
    """Stellar-to-halo-mass relation based on
    `Zu and Mandelbaum 2015 <http://arxiv.org/abs/astro-ph/1505.02781/>`_.

    .. note::

        The `~halotools.empirical_models.ZuMandelbaum15SmHm` model is part of
        the ``zu_mandelbaum15`` prebuilt composite HOD-style model.
        For a tutorial on the ``zu_mandelbaum15``
        composite model, see :ref:`zu_mandelbaum15_composite_model`.

    """

    def __init__(self, prim_haloprop_key="halo_m200m", **kwargs):
        """
        Parameters
        ----------
        prim_haloprop_key : string, optional
            String giving the column name of the primary halo property
            governing stellar mass.

        Notes
        -----
        Note that the best-fit parameters of this model are based on the
        ``halo_m200m`` halo mass definition.
        Using alternative choices of mass definition will require altering the
        model parameters in order to mock up the same model published in Zu & Mandelbaum 2015.
        The `Colossus python package <https://bitbucket.org/bdiemer/colossus/>`_
        written by Benedikt Diemer can be used to
        convert between different halo mass definitions. This may be useful if you wish to use an
        existing halo catalog for which the halo mass definition you need is unavailable.
        """

        super(ZuMandelbaum15SmHm, self).__init__(
            galprop_name="stellar_mass", prim_haloprop_key=prim_haloprop_key, **kwargs
        )

        self.param_dict = self.retrieve_default_param_dict()

        self._methods_to_inherit.extend(["mean_halo_mass", "mean_stellar_mass"])
        self.publications = ["arXiv:1505.02781"]

    def retrieve_default_param_dict(self):
        """
        Returns
        -------
        d : dict
            Dictionary containing parameter values.
        """
        lgmh1 = 12.10
        lgmh0 = 10.31
        beta = 0.33
        delta = 0.42
        gamma = 1.21
        sigma = 0.5
        eta = -0.04

        d = {
            "smhm_m0": 10 ** lgmh0,
            "smhm_m1": 10 ** lgmh1,
            "smhm_beta": beta,
            "smhm_delta": delta,
            "smhm_gamma": gamma,
            "smhm_sigma": sigma,
            "smhm_sigma_slope": eta,
        }

        return d

    def mean_halo_mass(self, stellar_mass, **kwargs):
        r"""Return the halo mass of a central galaxy as a function
        of the stellar mass.

        Parameters
        ----------
        stellar_mass : array
            Array of stellar masses in h=1 solar mass units.

        Returns
        -------
        halo_mass : array_like
            Array of halo mass in h=1 solar mass units.

        Examples
        --------
        >>> from halotools.empirical_models import ZuMandelbaum15SmHm
        >>> model = ZuMandelbaum15SmHm()
        >>> halo_mass = model.mean_halo_mass(10**11)
        """
        m0 = self.param_dict["smhm_m0"]
        m1 = self.param_dict["smhm_m1"]
        beta = self.param_dict["smhm_beta"]
        delta = self.param_dict["smhm_delta"]
        gamma = self.param_dict["smhm_gamma"]

        mass_ratio = stellar_mass / m0
        exparg = ((mass_ratio ** delta) / (1.0 + mass_ratio ** (-gamma))) - 0.5
        halo_mass = m1 * (mass_ratio ** beta) * 10 ** (exparg)
        return halo_mass

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
        stellar_mass : array_like
            Array containing stellar masses living in the input table,
            in solar mass units assuming h = 1.

        Examples
        --------
        >>> from halotools.empirical_models import ZuMandelbaum15SmHm
        >>> model = ZuMandelbaum15SmHm()
        >>> stellar_mass = model.mean_stellar_mass(prim_haloprop=10**12)
        """

        if "table" in list(kwargs.keys()):
            halo_mass = np.atleast_1d(kwargs["table"][self.prim_haloprop_key])
        elif "prim_haloprop" in list(kwargs.keys()):
            halo_mass = np.atleast_1d(kwargs["prim_haloprop"])
        else:
            raise KeyError(
                "Must pass one of the following keyword arguments "
                "to mean_stellar_mass:\n``table`` or ``prim_haloprop``"
            )

        stellar_mass_table = np.logspace(8.5, 12.5, 500)
        halo_mass_table = self.mean_halo_mass(stellar_mass_table, **kwargs)
        log_stellar_mass = np.interp(
            np.log10(halo_mass), np.log10(halo_mass_table), np.log10(stellar_mass_table)
        )
        return 10.0 ** log_stellar_mass

    def scatter_ln_mstar(self, halo_mass):
        r"""Scatter in :math:`{\rm ln M}_{\ast}` as a function of halo mass.

        Parameters
        -----------
        halo_mass : array_like
            Halo mass in units of Msun with h=1.

        Returns
        --------
        scatter : array_like

        Examples
        --------
        >>> from halotools.empirical_models import ZuMandelbaum15SmHm
        >>> model = ZuMandelbaum15SmHm()
        >>> sigma = model.scatter_ln_mstar(1e12)
        """
        m1 = self.param_dict["smhm_m1"]
        sigma = self.param_dict["smhm_sigma"]
        eta = self.param_dict["smhm_sigma_slope"]

        return np.where(halo_mass < m1, sigma, sigma + eta * np.log10(halo_mass / m1))

    def mean_scatter(self, **kwargs):
        if "table" in kwargs.keys():
            halo_mass = kwargs["table"][self.prim_haloprop_key]
        else:
            halo_mass = np.atleast_1d(kwargs["prim_haloprop"])
        return np.log10(np.e) * self.scatter_ln_mstar(halo_mass)

    def scatter_realization(self, **kwargs):
        """Monte Carlo realization of stellar mass stochasticity"""
        seed = kwargs.get("seed", None)

        scatter_scale = np.atleast_1d(self.mean_scatter(**kwargs))

        # initialize result with zero scatter result
        result = np.zeros(len(scatter_scale))

        # only draw from a normal distribution for non-zero values of scatter
        mask = scatter_scale > 0.0
        with NumpyRNGContext(seed):
            result[mask] = np.random.normal(loc=0, scale=scatter_scale[mask])

        return result
