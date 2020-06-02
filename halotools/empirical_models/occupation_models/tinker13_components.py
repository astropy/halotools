"""
This module contains occupation components used by the Tinker13 composite model.
"""

import numpy as np
import math
from scipy.special import erf
from astropy.utils.misc import NumpyRNGContext

from .occupation_model_template import OccupationComponent

from .. import model_defaults, model_helpers
from ..smhm_models import Behroozi10SmHm
from ..assembias_models import HeavisideAssembias

from ...utils.array_utils import custom_len
from ... import sim_manager
from ...custom_exceptions import HalotoolsError

__all__ = ('Tinker13Cens', 'Tinker13QuiescentSats',
           'Tinker13ActiveSats', 'AssembiasTinker13Cens')

# The following 4 lines of copde maintain python 2 and 3 compatability.
# See Tinker13Cens.mean_occupation() method for the use the unicode type
try:
    unicode  # Python 2: type "unicode" is built-in
except NameError:
    unicode = str  # Python 3


class Tinker13Cens(OccupationComponent):
    """ HOD-style model for a central galaxy occupation that derives from
    two distinct active/quiescent stellar-to-halo-mass relations.

    .. note::

        The `Tinker13Cens` model is part of the ``tinker13``
        prebuilt composite HOD-style model. For a tutorial on the ``tinker13``
        composite model, see :ref:`tinker13_composite_model`.

    """

    def __init__(self, threshold=model_defaults.default_stellar_mass_threshold,
            prim_haloprop_key=model_defaults.prim_haloprop_key,
            redshift=sim_manager.sim_defaults.default_redshift,
            **kwargs):
        """
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

        quiescent_fraction_abscissa : array, optional
            Values of the primary halo property at which the quiescent fraction is specified.
            Default is [10**12, 10**13.5, 10**15].

        quiescent_fraction_ordinates : array, optional
            Values of the quiescent fraction when evaluated at the input abscissa.
            Default is [0.25, 0.7, 0.95]

        """
        upper_occupation_bound = 1.0

        self.littleh = 0.72

        # Call the super class constructor, which binds all the
        # arguments to the instance.
        super(Tinker13Cens, self).__init__(
            gal_type='centrals', threshold=threshold,
            upper_occupation_bound=upper_occupation_bound,
            prim_haloprop_key=prim_haloprop_key,
            **kwargs)
        self.redshift = redshift

        self.smhm_model = Behroozi10SmHm(
            prim_haloprop_key=prim_haloprop_key, **kwargs)

        self._initialize_param_dict(**kwargs)

        self.sfr_designation_key = 'central_sfr_designation'

        self.publications = ['arXiv:1308.2974', 'arXiv:1103.2077', 'arXiv:1104.0928']

        # The _methods_to_inherit determines which methods will be directly callable
        # by the composite model built by the HodModelFactory
        # Here we are overriding this attribute that is normally defined
        # in the OccupationComponent super class
        self._methods_to_inherit = (
            ['mc_occupation', 'mean_occupation', 'mean_occupation_active', 'mean_occupation_quiescent',
            'mean_stellar_mass_active', 'mean_stellar_mass_quiescent',
            'mean_log_halo_mass_active', 'mean_log_halo_mass_quiescent']
            )

        # The _mock_generation_calling_sequence determines which methods
        # will be called during mock population, as well as in what order they will be called
        self._mock_generation_calling_sequence = ['mc_sfr_designation', 'mc_occupation']
        self._galprop_dtypes_to_allocate = np.dtype([
            ('halo_num_' + self.gal_type, 'i4'),
            (self.sfr_designation_key, object),
            ('sfr_designation', object),
            ])

    def _initialize_param_dict(self,
            quiescent_fraction_abscissa=[6.31e10, 3.98e11, 2.51e12, 1.58e13, 1.e14],
            quiescent_fraction_ordinates=[0.052, 0.14, 0.54, 0.63, 0.77], **kwargs):
        """
        """
        self.param_dict = {}
        for key, value in self.smhm_model.param_dict.items():
            active_key = key + '_active'
            quiescent_key = key + '_quiescent'
            self.param_dict[active_key] = value
            self.param_dict[quiescent_key] = value

        # From Table 2 of Tinker+13
        self.param_dict['smhm_m1_0_active'] = 12.56
        self.param_dict['smhm_m1_0_quiescent'] = 12.08
        self.param_dict['smhm_m0_0_active'] = 10.96
        self.param_dict['smhm_m0_0_quiescent'] = 10.7
        self.param_dict['smhm_beta_0_active'] = 0.44
        self.param_dict['smhm_beta_0_quiescent'] = 0.32
        self.param_dict['smhm_delta_0_active'] = 0.52
        self.param_dict['smhm_delta_0_quiescent'] = 0.93
        self.param_dict['smhm_gamma_0_active'] = 1.48
        self.param_dict['smhm_gamma_0_quiescent'] = 0.81
        self.param_dict['scatter_model_param1_active'] = 0.21
        self.param_dict['scatter_model_param1_quiescent'] = 0.28

        self._quiescent_fraction_abscissa = np.array(quiescent_fraction_abscissa)/self.littleh
        ordinates_key_prefix = 'quiescent_fraction_ordinates'
        self._ordinates_keys = (
            [ordinates_key_prefix + '_param' + str(i+1)
            for i in range(custom_len(self._quiescent_fraction_abscissa))]
            )
        for key, value in zip(self._ordinates_keys, quiescent_fraction_ordinates):
            self.param_dict[key] = value

    def mean_quiescent_fraction(self, **kwargs):
        """
        """
        model_ordinates = [self.param_dict[ordinate_key] for ordinate_key in self._ordinates_keys]
        spline_function = model_helpers.custom_spline(
            np.log10(self._quiescent_fraction_abscissa), model_ordinates)

        if 'prim_haloprop' in kwargs:
            prim_haloprop = np.atleast_1d(kwargs['prim_haloprop'])
        elif 'table' in kwargs:
            table = kwargs['table']
            try:
                prim_haloprop = table[self.prim_haloprop_key]
            except KeyError:
                msg = ("The ``table`` passed as a keyword argument to the mean_quiescent_fraction method\n"
                    "does not have the requested ``%s`` key")
                raise HalotoolsError(msg % self.prim_haloprop_key)

        fraction = spline_function(np.log10(prim_haloprop))

        fraction = np.where(fraction < 0, 0., fraction)
        fraction = np.where(fraction > 1, 1., fraction)

        return fraction

    def mc_sfr_designation(self, seed=None, **kwargs):
        """
        """
        quiescent_fraction = self.mean_quiescent_fraction(**kwargs)

        with NumpyRNGContext(seed):
            mc_generator = np.random.random(custom_len(quiescent_fraction))

        result = np.where(mc_generator < quiescent_fraction, 'quiescent', 'active')
        if 'table' in kwargs:
            kwargs['table'][self.sfr_designation_key] = result
            kwargs['table']['sfr_designation'] = result

        return result

    def mean_occupation(self, **kwargs):
        """
        """
        if 'table' in kwargs:
            table = kwargs['table']
            try:
                prim_haloprop = table[self.prim_haloprop_key]
            except KeyError:
                msg = ("The ``table`` passed as a keyword argument to the ``mean_occupation`` method\n"
                    "does not have the requested ``%s`` key")
                raise HalotoolsError(msg % self.prim_haloprop_key)
            try:
                sfr_designation = table[self.sfr_designation_key]
            except KeyError:
                msg = ("The ``table`` passed as a keyword argument to the ``mean_occupation`` method\n"
                    "does not have the requested ``%s`` key used for SFR designation")
                raise HalotoolsError(msg % self.sfr_designation_key)
        else:
            try:
                prim_haloprop = np.atleast_1d(kwargs['prim_haloprop'])
                sfr_designation = np.atleast_1d(kwargs['sfr_designation'])
            except KeyError:
                msg = ("If not passing a ``table`` keyword argument to the ``mean_occupation`` method,\n"
                    "you must pass both ``prim_haloprop`` and ``sfr_designation`` keyword arguments")
                raise HalotoolsError(msg)
            if type(sfr_designation[0]) in (str, unicode, np.string_, np.unicode_):
                if sfr_designation[0] not in ['active', 'quiescent']:
                    msg = ("The only acceptable values of "
                        "``sfr_designation`` are ``active`` or ``quiescent``")
                    raise HalotoolsError(msg)

        if 'table' in kwargs:
            quiescent_result = self.mean_occupation_quiescent(table=table)
            active_result = self.mean_occupation_active(table=table)
        else:
            quiescent_result = self.mean_occupation_quiescent(prim_haloprop=prim_haloprop)
            active_result = self.mean_occupation_active(prim_haloprop=prim_haloprop)

        result = np.where(sfr_designation == 'quiescent', quiescent_result, active_result)

        return result

    def mean_occupation_active(self, **kwargs):
        """
        """
        self._update_smhm_param_dict('active')

        logmstar = np.log10(self.smhm_model.mean_stellar_mass(
            redshift=self.redshift, **kwargs))
        logscatter = math.sqrt(2)*self.smhm_model.mean_scatter(**kwargs)

        mean_ncen = 0.5*(1.0 -
            erf((self.threshold - logmstar)/logscatter))
        mean_ncen *= (1. - self.mean_quiescent_fraction(**kwargs))

        return mean_ncen

    def mean_occupation_quiescent(self, **kwargs):
        """
        """
        self._update_smhm_param_dict('quiescent')

        logmstar = np.log10(self.smhm_model.mean_stellar_mass(
            redshift=self.redshift, **kwargs))
        logscatter = math.sqrt(2)*self.smhm_model.mean_scatter(**kwargs)

        mean_ncen = 0.5*(1.0 -
            erf((self.threshold - logmstar)/logscatter))
        mean_ncen *= self.mean_quiescent_fraction(**kwargs)

        return mean_ncen

    def mean_stellar_mass_active(self, **kwargs):
        """
        """
        self._update_smhm_param_dict('active')
        return self.smhm_model.mean_stellar_mass(redshift=self.redshift, **kwargs)

    def mean_stellar_mass_quiescent(self, **kwargs):
        """
        """
        self._update_smhm_param_dict('quiescent')
        return self.smhm_model.mean_stellar_mass(redshift=self.redshift, **kwargs)

    def mean_log_halo_mass_active(self, log_stellar_mass):
        """
        """
        self._update_smhm_param_dict('active')
        return self.smhm_model.mean_log_halo_mass(log_stellar_mass,
            redshift=self.redshift)

    def mean_log_halo_mass_quiescent(self, log_stellar_mass):
        """
        """
        self._update_smhm_param_dict('quiescent')
        return self.smhm_model.mean_log_halo_mass(log_stellar_mass,
            redshift=self.redshift)

    def _update_smhm_param_dict(self, sfr_key):

        for key, value in self.param_dict.items():
            if sfr_key in key:
                stripped_key = key[:-len(sfr_key)-1]
            else:
                stripped_key = key
            if stripped_key in self.smhm_model.param_dict:
                self.smhm_model.param_dict[stripped_key] = value


class AssembiasTinker13Cens(Tinker13Cens, HeavisideAssembias):
    """ HOD-style model for a central galaxy occupation that derives from
    two distinct active/quiescent stellar-to-halo-mass relations.
    """

    def __init__(self, threshold=model_defaults.default_stellar_mass_threshold,
            prim_haloprop_key=model_defaults.prim_haloprop_key,
            redshift=sim_manager.sim_defaults.default_redshift,
            **kwargs):
        """
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

        quiescent_fraction_abscissa : array, optional
            Values of the primary halo property at which the quiescent fraction is specified.
            Default is [10**12, 10**13.5, 10**15].

        quiescent_fraction_ordinates : array, optional
            Values of the quiescent fraction when evaluated at the input abscissa.
            Default is [0.25, 0.7, 0.95]

        sec_haloprop_key : string, optional
            String giving the column name of the secondary halo property
            governing the assembly bias. Must be a key in the table
            passed to the methods of `HeavisideAssembiasComponent`.
            Default value is specified in the `~halotools.empirical_models.model_defaults` module.

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
        Tinker13Cens.__init__(self, **kwargs)
        HeavisideAssembias.__init__(self,
            method_name_to_decorate='mean_quiescent_fraction',
            lower_assembias_bound=0.,
            upper_assembias_bound=1.,
            **kwargs)


class Tinker13QuiescentSats(OccupationComponent):
    """ HOD-style model for a central galaxy occupation that derives from
    two distinct active/quiescent stellar-to-halo-mass relations.

    .. note::

        The `Tinker13QuiescentSats` model is part of the ``tinker13``
        prebuilt composite HOD-style model. For a tutorial on the ``tinker13``
        composite model, see :ref:`tinker13_composite_model`.
    """

    def __init__(self, threshold=model_defaults.default_stellar_mass_threshold,
            prim_haloprop_key=model_defaults.prim_haloprop_key,
            redshift=sim_manager.sim_defaults.default_redshift, **kwargs):
        """
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

        """
        upper_occupation_bound = float("inf")

        self.littleh = 0.72

        # Call the super class constructor, which binds all the
        # arguments to the instance.
        super(Tinker13QuiescentSats, self).__init__(
            gal_type='quiescent_satellites', threshold=threshold,
            upper_occupation_bound=upper_occupation_bound,
            prim_haloprop_key=prim_haloprop_key, **kwargs)
        self.redshift = redshift

        self.smhm_model = Behroozi10SmHm(
            prim_haloprop_key=prim_haloprop_key, **kwargs)

        self._initialize_param_dict()

        self.sfr_designation_key = 'sfr_designation'

        self.publications = ['arXiv:1308.2974', 'arXiv:1103.2077', 'arXiv:1104.0928']

        # The _methods_to_inherit determines which methods will be directly callable
        # by the composite model built by the HodModelFactory
        # Here we are overriding this attribute that is normally defined
        # in the OccupationComponent super class
        self._methods_to_inherit = ['mc_occupation', 'mean_occupation']

        # The _mock_generation_calling_sequence determines which methods
        # will be called during mock population, as well as in what order they will be called
        self._mock_generation_calling_sequence = ['mc_occupation', 'mc_sfr_designation']
        self._galprop_dtypes_to_allocate = np.dtype([
            ('halo_num_' + self.gal_type, 'i4'),
            (self.sfr_designation_key, object),
            ])

    def mean_occupation(self, **kwargs):
        """ Expected number of central galaxies in a halo of mass halo_mass.
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
            Mean number of central galaxies in the halo of the input mass.

        Examples
        --------
        >>> sat_model = Tinker13QuiescentSats()
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
            function_name = "Tinker13QuiescentSats.mean_occupation"
            raise HalotoolsError(function_name)

        self._update_satellite_params()

        power_law_factor = (mass*self.littleh/self._msat)**self.param_dict['alphasat_quiescent']

        exp_arg_numerator = self._mcut + 10.**self.smhm_model.mean_log_halo_mass(
            log_stellar_mass=self.threshold, redshift=self.redshift)
        exp_factor = np.exp(-exp_arg_numerator/(mass*self.littleh))

        mean_nsat = exp_factor*power_law_factor

        return mean_nsat

    def mc_sfr_designation(self, table, **kwargs):
        """
        """
        table[self.sfr_designation_key][:] = 'quiescent'

    def _initialize_param_dict(self):
        """ Set the initial values of ``self.param_dict`` according to
        the SIG_MOD1 values of Table 5 of arXiv:1104.0928 for the
        lowest redshift bin.

        """

        self.param_dict['bcut_quiescent'] = 21.42
        self.param_dict['bsat_quiescent'] = 17.9
        self.param_dict['betacut_quiescent'] = -0.12
        self.param_dict['betasat_quiescent'] = 0.62
        self.param_dict['alphasat_quiescent'] = 1.08

        for key, value in self.smhm_model.param_dict.items():
            quiescent_key = key + '_quiescent'
            self.param_dict[quiescent_key] = value

        self.param_dict['smhm_m1_0_quiescent'] = 12.08
        self.param_dict['smhm_m0_0_quiescent'] = 10.7
        self.param_dict['smhm_beta_0_quiescent'] = 0.32
        self.param_dict['smhm_delta_0_quiescent'] = 0.93
        self.param_dict['smhm_gamma_0_quiescent'] = 0.81

        self._update_satellite_params()

    def _update_satellite_params(self):
        """ Private method to update the model parameters.

        """
        for key, value in self.param_dict.items():
            stripped_key = key[:-len('_quiescent')]
            if stripped_key in self.smhm_model.param_dict:
                self.smhm_model.param_dict[stripped_key] = value

        log_halo_mass_threshold = self.smhm_model.mean_log_halo_mass(
            log_stellar_mass=self.threshold, redshift=self.redshift)
        knee_threshold = (10.**log_halo_mass_threshold)*self.littleh

        knee_mass = 1.e12

        self._msat = (
            knee_mass*self.param_dict['bsat_quiescent'] *
            (knee_threshold / knee_mass)**self.param_dict['betasat_quiescent'])

        self._mcut = (
            knee_mass*self.param_dict['bcut_quiescent'] *
            (knee_threshold / knee_mass)**self.param_dict['betacut_quiescent'])


class Tinker13ActiveSats(OccupationComponent):
    """ HOD-style model for a central galaxy occupation that derives from
    two distinct active/active stellar-to-halo-mass relations.

    .. note::

        The `Tinker13ActiveSats` model is part of the ``tinker13``
        prebuilt composite HOD-style model. For a tutorial on the ``tinker13``
        composite model, see :ref:`tinker13_composite_model`.
    """

    def __init__(self, threshold=model_defaults.default_stellar_mass_threshold,
            prim_haloprop_key=model_defaults.prim_haloprop_key,
            redshift=sim_manager.sim_defaults.default_redshift, **kwargs):
        """
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
        """
        upper_occupation_bound = float("inf")

        self.littleh = 0.72

        # Call the super class constructor, which binds all the
        # arguments to the instance.
        super(Tinker13ActiveSats, self).__init__(
            gal_type='active_satellites', threshold=threshold,
            upper_occupation_bound=upper_occupation_bound,
            prim_haloprop_key=prim_haloprop_key, **kwargs)
        self.redshift = redshift

        self.smhm_model = Behroozi10SmHm(
            prim_haloprop_key=prim_haloprop_key, **kwargs)

        self._initialize_param_dict()

        self.sfr_designation_key = 'sfr_designation'

        self.publications = ['arXiv:1308.2974', 'arXiv:1103.2077', 'arXiv:1104.0928']

        # The _methods_to_inherit determines which methods will be directly callable
        # by the composite model built by the HodModelFactory
        # Here we are overriding this attribute that is normally defined
        # in the OccupationComponent super class
        self._methods_to_inherit = ['mc_occupation', 'mean_occupation']

        # The _mock_generation_calling_sequence determines which methods
        # will be called during mock population, as well as in what order they will be called
        self._mock_generation_calling_sequence = ['mc_occupation', 'mc_sfr_designation']
        self._galprop_dtypes_to_allocate = np.dtype([
            ('halo_num_' + self.gal_type, 'i4'),
            (self.sfr_designation_key, object),
            ])

    def mean_occupation(self, **kwargs):
        """ Expected number of central galaxies in a halo of mass halo_mass.
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
            Mean number of central galaxies in the halo of the input mass.

        Examples
        --------
        >>> sat_model = Tinker13ActiveSats()
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
            function_name = "Tinker13ActiveSats.mean_occupation"
            raise HalotoolsError(function_name)

        self._update_satellite_params()

        power_law_factor = (mass*self.littleh/self._msat)**self.param_dict['alphasat_active']

        exp_arg_numerator = self._mcut + 10.**self.smhm_model.mean_log_halo_mass(
            log_stellar_mass=self.threshold, redshift=self.redshift)
        exp_factor = np.exp(-exp_arg_numerator/(mass*self.littleh))

        mean_nsat = exp_factor*power_law_factor

        return mean_nsat

    def mc_sfr_designation(self, table, **kwargs):
        """
        """
        table[self.sfr_designation_key][:] = 'active'

    def _initialize_param_dict(self):
        """ Set the initial values of ``self.param_dict`` according to
        the z1 values of Table 2 of arXiv:1308.2974.
        """

        self.param_dict['bcut_active'] = 0.28
        self.param_dict['bsat_active'] = 33.96
        self.param_dict['betacut_active'] = 0.77
        self.param_dict['betasat_active'] = 1.05
        self.param_dict['alphasat_active'] = 0.99

        for key, value in self.smhm_model.param_dict.items():
            active_key = key + '_active'
            self.param_dict[active_key] = value

        self.param_dict['smhm_m1_0_active'] = 12.56
        self.param_dict['smhm_m0_0_active'] = 10.96
        self.param_dict['smhm_beta_0_active'] = 0.44
        self.param_dict['smhm_delta_0_active'] = 0.52
        self.param_dict['smhm_gamma_0_active'] = 1.48

        self._update_satellite_params()

    def _update_satellite_params(self):
        """ Private method to update the model parameters.

        """
        for key, value in self.param_dict.items():
            stripped_key = key[:-len('_active')]
            if stripped_key in self.smhm_model.param_dict:
                self.smhm_model.param_dict[stripped_key] = value

        log_halo_mass_threshold = self.smhm_model.mean_log_halo_mass(
            log_stellar_mass=self.threshold, redshift=self.redshift)
        knee_threshold = (10.**log_halo_mass_threshold)*self.littleh

        knee_mass = 1.e12

        self._msat = (
            knee_mass*self.param_dict['bsat_active'] *
            (knee_threshold / knee_mass)**self.param_dict['betasat_active'])

        self._mcut = (
            knee_mass*self.param_dict['bcut_active'] *
            (knee_threshold / knee_mass)**self.param_dict['betacut_active'])
