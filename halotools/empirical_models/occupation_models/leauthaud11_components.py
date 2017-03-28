r"""
This module contains occupation components used by the Leauthaud11 composite model.
"""

import numpy as np
import math
from scipy.special import erf
import warnings

from .occupation_model_template import OccupationComponent

from .. import model_defaults
from ..smhm_models import Behroozi10SmHm
from ..assembias_models import HeavisideAssembias

from ... import sim_manager
from ...custom_exceptions import HalotoolsError

__all__ = ('Leauthaud11Cens', 'Leauthaud11Sats',
           'AssembiasLeauthaud11Cens', 'AssembiasLeauthaud11Sats')


class Leauthaud11Cens(OccupationComponent):
    r""" HOD-style model for any central galaxy occupation that derives from
    a stellar-to-halo-mass relation.

    .. note::

        The `Leauthaud11Cens` model is part of the ``leauthaud11``
        prebuilt composite HOD-style model. For a tutorial on the ``leauthaud11``
        composite model, see :ref:`leauthaud11_composite_model`.

    """

    def __init__(self, threshold=model_defaults.default_stellar_mass_threshold,
            prim_haloprop_key=model_defaults.prim_haloprop_key,
            redshift=sim_manager.sim_defaults.default_redshift, **kwargs):
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
            gal_type='centrals', threshold=threshold,
            upper_occupation_bound=upper_occupation_bound,
            prim_haloprop_key=prim_haloprop_key,
            **kwargs)
        self.redshift = redshift

        self.smhm_model = Behroozi10SmHm(
            prim_haloprop_key=prim_haloprop_key, **kwargs)

        for key, value in self.smhm_model.param_dict.items():
            self.param_dict[key] = value

        self._methods_to_inherit = (
            ['mc_occupation', 'mean_occupation',
            'mean_stellar_mass', 'mean_log_halo_mass']
            )

        self.publications = ['arXiv:1103.2077', 'arXiv:1104.0928']
        self.publications.extend(self.smhm_model.publications)
        self.publications = list(set(self.publications))

    def get_published_parameters(self):
        r""" Return the values of ``self.param_dict`` according to
        the SIG_MOD1 values of Table 5 of arXiv:1104.0928 for the
        lowest redshift bin.

        """
        d = {}
        d['smhm_m1_0'] = 12.52
        d['smhm_m0_0'] = 10.916
        d['smhm_beta_0'] = 0.457
        d['smhm_delta_0'] = 0.566
        d['smhm_gamma_0'] = 1.54
        d['scatter_model_param1'] = 0.206
        return d

    def mean_occupation(self, **kwargs):
        r""" Expected number of central galaxies in a halo.
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

        logmstar = np.log10(self.smhm_model.mean_stellar_mass(
            redshift=self.redshift, **kwargs))
        logscatter = math.sqrt(2)*self.smhm_model.mean_scatter(**kwargs)

        mean_ncen = 0.5*(1.0 -
            erf((self.threshold - logmstar)/logscatter))

        return mean_ncen

    def mean_stellar_mass(self, **kwargs):
        r""" Return the stellar mass of a central galaxy as a function
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
        """

        for key, value in self.param_dict.items():
            if key in self.smhm_model.param_dict:
                self.smhm_model.param_dict[key] = value
        return self.smhm_model.mean_stellar_mass(redshift=self.redshift, **kwargs)

    def mean_log_halo_mass(self, log_stellar_mass):
        r""" Return the base-10 logarithm of the halo mass of a central galaxy as a function
        of the base-10 logarithm of the input stellar mass.

        Parameters
        ----------
        log_stellar_mass : array
            Array of base-10 logarithm of stellar masses in h=1 solar mass units.

        Returns
        -------
        log_halo_mass : array_like
            Array containing 10-base logarithm of halo mass in h=1 solar mass units.
        """
        for key, value in self.param_dict.items():
            if key in self.smhm_model.param_dict:
                self.smhm_model.param_dict[key] = value
        return self.smhm_model.mean_log_halo_mass(log_stellar_mass,
            redshift=self.redshift)


class Leauthaud11Sats(OccupationComponent):
    r""" HOD-style model for any satellite galaxy occupation that derives from
    a stellar-to-halo-mass relation.

    .. note::

        The `Leauthaud11Sats` model is part of the ``leauthaud11``
        prebuilt composite HOD-style model. For a tutorial on the ``leauthaud11``
        composite model, see :ref:`leauthaud11_composite_model`.
    """

    def __init__(self, threshold=model_defaults.default_stellar_mass_threshold,
            prim_haloprop_key=model_defaults.prim_haloprop_key,
            redshift=sim_manager.sim_defaults.default_redshift,
            modulate_with_cenocc=True, cenocc_model=None,
            **kwargs):
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

        self.littleh = 0.72

        if cenocc_model is None:
            cenocc_model = Leauthaud11Cens(
                prim_haloprop_key=prim_haloprop_key, threshold=threshold
            )
        else:
            if modulate_with_cenocc is False:
                msg = ("You chose to input a ``cenocc_model``, but you set the \n"
                       "``modulate_with_cenocc`` keyword to False, so your "
                       "``cenocc_model`` will have no impact on the model's behavior.\n"
                       "Be sure this is what you intend before proceeding.\n"
                       "Refer to the Leauthand et al. (2011) composite model tutorial for details.\n")
                warnings.warn(msg)

        self.modulate_with_cenocc = modulate_with_cenocc

        if self.modulate_with_cenocc:
            try:
                assert isinstance(cenocc_model, OccupationComponent)
            except AssertionError:
                msg = ("The input ``cenocc_model`` must be an instance of \n"
                       "``OccupationComponent`` or one of its sub-classes.\n")
                raise HalotoolsError(msg)

        self.central_occupation_model = cenocc_model

        super(Leauthaud11Sats, self).__init__(
            gal_type='satellites', threshold=threshold,
            upper_occupation_bound=float("inf"),
            prim_haloprop_key=prim_haloprop_key,
            **kwargs)

        self.redshift = redshift

        self._initialize_param_dict()

        self.param_dict.update(self.central_occupation_model.param_dict)

        self.publications = self.central_occupation_model.publications

    def mean_occupation(self, **kwargs):
        r""" Expected number of satellite galaxies in a halo of mass halo_mass.
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
        if 'table' in list(kwargs.keys()):
            mass = kwargs['table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in list(kwargs.keys()):
            mass = np.atleast_1d(kwargs['prim_haloprop'])
        else:
            raise KeyError("Must pass one of the following keyword arguments "
                "to mean_occupation:\n``table`` or ``prim_haloprop``")

        self._update_satellite_params()

        mean_nsat = (
            np.exp(-self._mcut/(mass*self.littleh)) *
            (mass*self.littleh/self._msat)**self.param_dict['alphasat']
            )

        if self.modulate_with_cenocc is True:
            mean_nsat *= self.central_occupation_model.mean_occupation(**kwargs)

        return mean_nsat

    def _initialize_param_dict(self):
        """ Set the initial values of ``self.param_dict`` according to
        the SIG_MOD1 values of Table 5 of arXiv:1104.0928 for the
        lowest redshift bin.

        """

        self.param_dict['alphasat'] = 1.0
        self.param_dict['bsat'] = 10.62
        self.param_dict['bcut'] = 1.47
        self.param_dict['betacut'] = -0.13
        self.param_dict['betasat'] = 0.859

        for key, value in self.central_occupation_model.param_dict.items():
            self.param_dict[key] = value

        self._update_satellite_params()

    def _update_satellite_params(self):
        """ Private method to update the model parameters.

        """
        for key, value in self.param_dict.items():
            if key in self.central_occupation_model.param_dict:
                self.central_occupation_model.param_dict[key] = value

        log_halo_mass_threshold = self.central_occupation_model.mean_log_halo_mass(
            log_stellar_mass=self.threshold)
        knee_threshold = (10.**log_halo_mass_threshold)*self.littleh

        knee_mass = 1.e12

        self._msat = (
            knee_mass*self.param_dict['bsat'] *
            (knee_threshold / knee_mass)**self.param_dict['betasat'])

        self._mcut = (
            knee_mass*self.param_dict['bcut'] *
            (knee_threshold / knee_mass)**self.param_dict['betacut'])


class AssembiasLeauthaud11Cens(Leauthaud11Cens, HeavisideAssembias):
    """ Assembly-biased modulation of `Leauthaud11Cens`.
    """

    def __init__(self, **kwargs):
        r"""
        Parameters
        ----------
        threshold : float, optional
            Stellar mass threshold of the mock galaxy sample.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

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
        HeavisideAssembias.__init__(self,
            lower_assembias_bound=self._lower_occupation_bound,
            upper_assembias_bound=self._upper_occupation_bound,
            method_name_to_decorate='mean_occupation', **kwargs)


class AssembiasLeauthaud11Sats(Leauthaud11Sats, HeavisideAssembias):
    """ Assembly-biased modulation of `Leauthaud11Sats`.
    """

    def __init__(self, **kwargs):
        r"""
        Parameters
        ----------
        threshold : float, optional
            Stellar mass threshold of the mock galaxy sample.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

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
        HeavisideAssembias.__init__(self,
            lower_assembias_bound=self._lower_occupation_bound,
            upper_assembias_bound=self._upper_occupation_bound,
            method_name_to_decorate='mean_occupation', **kwargs)
