"""
This module contains occupation components used by the ZuMandelbaum15 composite model.
"""
import numpy as np
from scipy.special import erf

from .occupation_model_template import OccupationComponent

from .. import model_defaults
from ..smhm_models import ZuMandelbaum15SmHm

__all__ = ("ZuMandelbaum15Cens", "ZuMandelbaum15Sats")


class ZuMandelbaum15Cens(OccupationComponent):
    """HOD-style model for any central galaxy occupation that derives from
    a stellar-to-halo-mass relation.

    .. note::

        The `~halotools.empirical_models.ZuMandelbaum15Cens` model is part of
        the ``zu_mandelbaum15`` prebuilt composite HOD-style model.
        For a tutorial on the ``zu_mandelbaum15``
        composite model, see :ref:`zu_mandelbaum15_composite_model`.

    """

    def __init__(
        self,
        threshold=model_defaults.default_stellar_mass_threshold,
        prim_haloprop_key="halo_m200m",
        **kwargs
    ):
        """
        Parameters
        ----------
        threshold : float, optional
            Stellar mass threshold of the mock galaxy sample in h=1 solar mass units.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        prim_haloprop_key : string, optional
            String giving the column name of the primary halo property governing
            the occupation statistics of gal_type galaxies.
            Default value is ``halo_m200m``, as in Zu & Mandelbaum (2015)

        redshift : float, optional
            Redshift of the stellar-to-halo-mass relation. Default is z=0.

        Examples
        --------
        >>> cen_model = ZuMandelbaum15Cens()
        >>> cen_model = ZuMandelbaum15Cens(threshold=11.25)
        >>> cen_model = ZuMandelbaum15Cens(prim_haloprop_key='halo_m200b')

        Notes
        -----
        Note also that the best-fit parameters of this model are based on the
        ``halo_m200m`` halo mass definition.
        Using alternative choices of mass definition will require altering the
        model parameters in order to mock up the same model published in Zu & Mandelbaum 2015.
        The `Colossus python package <https://bitbucket.org/bdiemer/colossus/>`_
        written by Benedikt Diemer can be used to
        convert between different halo mass definitions. This may be useful if you wish to use an
        existing halo catalog for which the halo mass definition you need is unavailable.

        """
        upper_occupation_bound = 1.0

        # Call the super class constructor, which binds all the
        # arguments to the instance.
        OccupationComponent.__init__(
            self,
            gal_type="centrals",
            threshold=threshold,
            upper_occupation_bound=upper_occupation_bound,
            prim_haloprop_key=prim_haloprop_key,
        )

        self.smhm_model = ZuMandelbaum15SmHm(prim_haloprop_key=prim_haloprop_key)

        for key, value in self.smhm_model.param_dict.items():
            self.param_dict[key] = value

        self._methods_to_inherit = [
            "mc_occupation",
            "mean_occupation",
            "mean_stellar_mass",
            "mean_halo_mass",
        ]

        self.publications = ["arXiv:1103.2077", "arXiv:1104.0928", "1505.02781"]
        self.publications.extend(self.smhm_model.publications)
        self.publications = list(set(self.publications))

    def get_published_parameters(self):
        """"""
        return ZuMandelbaum15SmHm.get_published_parameters(self)

    def mean_occupation(self, **kwargs):
        """Expected number of central galaxies in a halo.

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

        Examples
        --------
        >>> cen_model = ZuMandelbaum15Cens(threshold=10.75)
        >>> halo_masses = np.logspace(11, 15, 25)
        >>> mean_ncen = cen_model.mean_occupation(prim_haloprop=halo_masses)

        """
        for key, value in self.param_dict.items():
            if key in list(self.smhm_model.param_dict.keys()):
                self.smhm_model.param_dict[key] = value

        if "table" in list(kwargs.keys()):
            halo_mass = np.atleast_1d(kwargs["table"][self.prim_haloprop_key])
        elif "prim_haloprop" in list(kwargs.keys()):
            halo_mass = np.atleast_1d(kwargs["prim_haloprop"])
        else:
            raise KeyError(
                "Must pass one of the following keyword arguments "
                "to mean_occupation:\n``table`` or ``prim_haloprop``"
            )

        sigma = self.smhm_model.scatter_ln_mstar(halo_mass)
        mean = self.smhm_model.mean_stellar_mass(prim_haloprop=halo_mass)
        erfarg = (np.log(10 ** self.threshold) - np.log(mean)) / (sigma * np.sqrt(2))
        return 0.5 * (1 - erf(erfarg))

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

        Returns
        -------
        mstar : array_like
            Array containing stellar masses living in the input table.

        Examples
        --------
        >>> cen_model = ZuMandelbaum15Cens(threshold=10.75)
        >>> halo_masses = np.logspace(11, 15, 25)
        >>> mstar = cen_model.mean_stellar_mass(prim_haloprop=halo_masses)
        """

        for key, value in self.param_dict.items():
            if key in self.smhm_model.param_dict:
                self.smhm_model.param_dict[key] = value
        return self.smhm_model.mean_stellar_mass(**kwargs)

    def mean_halo_mass(self, stellar_mass):
        """Return the halo mass of a central galaxy as a function
        of the input stellar mass.

        Parameters
        ----------
        stellar_mass : array
            Array of stellar masses in h=1 solar mass units.

        Returns
        -------
        halo_mass : array_like
            Array containing halo mass in h=1 solar mass units.

        Examples
        --------
        >>> cen_model = ZuMandelbaum15Cens(threshold=10.75)
        >>> stellar_mass = np.logspace(9, 12, 25)
        >>> halo_mass = cen_model.mean_halo_mass(stellar_mass)

        """
        for key, value in self.param_dict.items():
            if key in self.smhm_model.param_dict:
                self.smhm_model.param_dict[key] = value
        return self.smhm_model.mean_halo_mass(stellar_mass)


class ZuMandelbaum15Sats(OccupationComponent):
    r"""HOD-style model for a satellite galaxy occupation
    based on Zu & Mandelbaum 2015.

    .. note::

        The `~halotools.empirical_models.ZuMandelbaum15Sats` model is part of
        the ``zu_mandelbaum15`` prebuilt composite HOD-style model.
        For a tutorial on the ``zu_mandelbaum15``
        composite model, see :ref:`zu_mandelbaum15_composite_model`.
    """

    def __init__(
        self,
        threshold=model_defaults.default_stellar_mass_threshold,
        prim_haloprop_key="halo_m200m",
        **kwargs
    ):
        r"""
        Parameters
        ----------
        threshold : float, optional
            Stellar mass threshold of the mock galaxy sample in h=1 solar mass units.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        prim_haloprop_key : string, optional
            String giving the column name of the primary halo property governing
            the occupation statistics of gal_type galaxies.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

        Examples
        --------
        >>> sat_model = ZuMandelbaum15Sats()
        >>> sat_model = ZuMandelbaum15Sats(threshold=11)
        >>> sat_model = ZuMandelbaum15Sats(prim_haloprop_key='halo_mvir')

        Notes
        -----
        Note also that the best-fit parameters of this model are based on the
        ``halo_m200m`` halo mass definition.
        Using alternative choices of mass definition will require altering the
        model parameters in order to mock up the same model published in Zu & Mandelbaum 2015.
        The `Colossus python package <https://bitbucket.org/bdiemer/colossus/>`_
        written by Benedikt Diemer can be used to
        convert between different halo mass definitions. This may be useful if you wish to use an
        existing halo catalog for which the halo mass definition you need is unavailable.
        """
        self.central_occupation_model = ZuMandelbaum15Cens(
            prim_haloprop_key=prim_haloprop_key, threshold=threshold
        )

        OccupationComponent.__init__(
            self,
            gal_type="satellites",
            threshold=threshold,
            upper_occupation_bound=float("inf"),
            prim_haloprop_key=prim_haloprop_key,
        )

        self._initialize_param_dict()

        self.param_dict.update(self.central_occupation_model.param_dict)

        self.publications = self.central_occupation_model.publications

    def mean_occupation(self, **kwargs):
        """Expected number of satellite galaxies in a halo of mass halo_mass.

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
        >>> sat_model = ZuMandelbaum15Sats()
        >>> halo_masses = np.logspace(11, 15, 25)
        >>> mean_nsat = sat_model.mean_occupation(prim_haloprop=halo_masses)

        """
        # Retrieve the array storing the mass-like variable
        if "table" in list(kwargs.keys()):
            halo_mass = kwargs["table"][self.prim_haloprop_key]
        elif "prim_haloprop" in list(kwargs.keys()):
            halo_mass = np.atleast_1d(kwargs["prim_haloprop"])
        else:
            raise KeyError(
                "Must pass one of the following keyword arguments "
                "to mean_occupation:\n``table`` or ``prim_haloprop``"
            )

        self._update_satellite_params()

        mean_ncen = self.central_occupation_model.mean_occupation(**kwargs)
        mean_nsat = (
            mean_ncen
            * np.exp(-self._mcut / halo_mass)
            * (halo_mass / self._msat) ** self.param_dict["alphasat"]
        )
        return mean_nsat

    def _initialize_param_dict(self):
        """Set the initial values of ``self.param_dict`` according to
        the SIG_MOD1 values of Table 5 of arXiv:1104.0928 for the
        lowest redshift bin.

        """

        self.param_dict["alphasat"] = 1.0
        self.param_dict["bsat"] = 8.98
        self.param_dict["betasat"] = 0.9
        self.param_dict["bcut"] = 0.86
        self.param_dict["betacut"] = 0.41
        self.param_dict.update(self.central_occupation_model.param_dict)

        self._update_satellite_params()

    def _update_satellite_params(self):
        """Private method to update the model parameters."""
        for key, value in self.param_dict.items():
            if key in self.central_occupation_model.param_dict:
                self.central_occupation_model.param_dict[key] = value

        knee_threshold = self.central_occupation_model.mean_halo_mass(
            10 ** self.threshold
        )
        knee_mass = 1.0e12

        self._msat = (
            knee_mass
            * self.param_dict["bsat"]
            * (knee_threshold / knee_mass) ** self.param_dict["betasat"]
        )

        self._mcut = (
            knee_mass
            * self.param_dict["bcut"]
            * (knee_threshold / knee_mass) ** self.param_dict["betacut"]
        )
