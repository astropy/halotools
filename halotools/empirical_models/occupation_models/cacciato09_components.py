r"""
This module contains occupation components used by the cacciato09 composite
model.
"""

import numpy as np
from scipy.special import erfc, erfcinv
from astropy.utils.misc import NumpyRNGContext

from .occupation_model_template import OccupationComponent
from .engines import cacciato09_sats_mc_prim_galprop_engine

from .. import custom_incomplete_gamma

from ...custom_exceptions import HalotoolsError

__all__ = ('Cacciato09Cens', 'Cacciato09Sats')


class Cacciato09Cens(OccupationComponent):
    r""" CLF-style model for the central galaxy occupation. Since it is a CLF
    model, it also assigns a primary galaxy property to galaxies in addition to
    effectively being an HOD model. The primary galaxy property can for example
    be luminosity or stellar mass.

    See :ref:`cacciato09_composite_model` for a tutorial on this model.

    """

    def __init__(self, threshold=10.0, prim_haloprop_key='halo_m180b',
                 prim_galprop_key='luminosity', **kwargs):
        r"""
        Parameters
        ----------
        threshold : float, optional
            Logarithm of the primary galaxy property threshold. If the primary
            galaxy property is luminosity, it is given in h=1 solar luminosity
            units.

        prim_haloprop_key : string, optional
            String giving the column name of the primary halo property governing
            the occupation statistics of central galaxies.

        prim_galprop_key : string, optional
            String giving the column name of the primary galaxy property that
            is assigned.

        Examples
        --------
        >>> cen_model = Cacciato09Cens()
        >>> cen_model = Cacciato09Cens(threshold = 11.25)
        >>> cen_model = Cacciato09Cens(prim_haloprop_key = 'halo_mvir')

        """

        super(Cacciato09Cens, self).__init__(
            gal_type='centrals', threshold=threshold,
            upper_occupation_bound=1.0,
            prim_haloprop_key=prim_haloprop_key,
            **kwargs)

        self._mock_generation_calling_sequence = ['mc_occupation',
                                                  'mc_prim_galprop']
        self.prim_galprop_key = prim_galprop_key
        self._galprop_dtypes_to_allocate = np.dtype([(prim_galprop_key, 'f8')])
        self.param_dict = self.get_published_parameters()
        self._methods_to_inherit = (['mc_occupation', 'median_prim_galprop',
                                     'mean_occupation', 'mc_prim_galprop',
                                     'clf'])
        self.publications = ['arXiv:0807.4932']

    def get_published_parameters(self):
        r""" Return the values of ``self.param_dict`` according to
        the best-fit values of the WMAP3 model in Table 3 of arXiv:0807.4932.
        In this analysis, halo masses have been defined using an overdensity of
        180 times the background density of the Universe.
        """
        param_dict = (
            {'log_L_0': 9.935,
             'log_M_1': 11.07,
             'gamma_1': 3.273,
             'gamma_2': 0.255,
             'sigma': 0.143}
            )

        return param_dict

    def median_prim_galprop(self, **kwargs):
        r""" Return the median primary galaxy property of a central galaxy as a
        function of the input table.

        Parameters
        ----------
        prim_haloprop : array, optional
            Array of mass-like variable upon which the calculation is based.
            If ``prim_haloprop`` is not passed, then ``table`` keyword argument
            must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop`` keyword argument
            must be passed.

        Returns
        -------
        prim_galprop : array_like
            Array containing the median primary galaxy property of the halos
            specified.
        """

        # Retrieve the array storing the mass-like variable.
        if 'table' in list(kwargs.keys()):
            mass = kwargs['table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in list(kwargs.keys()):
            mass = kwargs['prim_haloprop']
        else:
            msg = ("\nYou must pass either a ``table`` or ``prim_haloprop`` "
                   "argument to the ``median_prim_galprop`` function of the "
                   "``Cacciato09Cens`` class.\n")
            raise HalotoolsError(msg)

        gamma_1 = self.param_dict['gamma_1']
        gamma_2 = self.param_dict['gamma_2']
        mass_c = 10**self.param_dict['log_M_1']
        prim_galprop_c = 10**self.param_dict['log_L_0']

        r = mass / mass_c

        return prim_galprop_c * (r / (1 + r))**gamma_1 * (1 + r)**gamma_2

    def clf(self, prim_galprop=1e10, prim_haloprop=1e12):
        r""" Return the CLF in units of dn/dlogL for the primary halo property
        and galaxy property L.

        Parameters
        ----------
        prim_haloprop : array_like, optional
            Array of mass-like variable upon which the calculation is based.

        prim_galprop : array_like, optional
            Array of luminosity-like variable of the galaxy upon which the
            calculation is based.

        Returns
        -------
        clf : array_like
            Array containing the CLF in units of dN/dlogL. If ``prim_haloprop``
            has only one element or is a scalar, the same primary halo property
            is assumed for all CLF values. Similarly, if ``prim_galprop`` has
            only one element or is a scalar, the same primary galaxy property is
            assumed throughout.
        """

        prim_galprop = np.atleast_1d(prim_galprop)
        prim_haloprop = np.atleast_1d(prim_haloprop)

        if (len(prim_haloprop) > 1) & (len(prim_galprop) > 1):
            msg = ("If both ``prim_galprop`` and ``prim_haloprop`` are arrays"
                   "with multiple elements, they must have the same length.\n")
            assert len(prim_galprop) == len(prim_haloprop), msg

        med_prim_galprop = self.median_prim_galprop(prim_haloprop=prim_haloprop)

        return ((1.0 / (np.sqrt(2.0 * np.pi) * self.param_dict['sigma'])) *
                np.exp(-(np.log10(prim_galprop / med_prim_galprop))**2 / (2.0 *
                       self.param_dict['sigma']**2)))

    def mean_occupation(self, prim_galprop_min=None, prim_galprop_max=None, **kwargs):
        r""" Expected number of central galaxies in a halo. Derived from
        integrating the CLF from the primary galaxy property threshold to
        infinity.

        Parameters
        ----------
        prim_galprop_min : float, optional
            Lower limit of the CLF integration used to calculate the expected
            number of central galaxies. If not specified, the lower limit is the
            threshold.
       
       prim_galprop_max : float, optional
            Upper limit of the CLF integration used to calculate the expected
            number of central galaxies. If not specified, the upper limit is
            infinity.

        prim_haloprop : array, optional
            Array of mass-like variable upon which occupation statistics are
            based.
            If ``prim_haloprop`` is not passed, then ``table`` keyword argument
            must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop`` keyword argument
            must be passed.

        Returns
        -------
        mean_ncen : array
            Mean number of central galaxies in halos of the input mass.
        """

        if prim_galprop_min is not None:
            prim_galprop_min = prim_galprop_min
        else:
            prim_galprop_min = 10**self.threshold
        
        if prim_galprop_max is not None:
            if prim_galprop_max <= prim_galprop_min:
                msg = ("\nFor the ``mean_occupation`` function of the "
                       "``Cacciato09Cens`` class the ``prim_galprop_max`` "
                       "keyword must be bigger than 10^threshold or "
                       "``prim_galprop_min`` if provided.\n")
                raise HalotoolsError(msg)

        log_prim_galprop = np.log10(self.median_prim_galprop(**kwargs))
        
        
        result = (0.5 * erfc((np.log10(prim_galprop_min) - log_prim_galprop) /
                           (np.sqrt(2.0) * self.param_dict['sigma'])))
       
        if prim_galprop_max is not None:
            result = (result - 0.5 * erfc((np.log10(prim_galprop_max) - log_prim_galprop) /
                           (np.sqrt(2.0) * self.param_dict['sigma'])))
  
        return result

    def mc_prim_galprop(self, **kwargs):
        r""" Method to generate Monte Carlo realizations of the primary galaxy
        property of galaxies under the constraint that the primary galaxy
        property is above the primary galaxy property threshold.

        Parameters
        ----------
        prim_haloprop : array, optional
            Array of mass-like variable upon which the primary galaxy properties
            are based.
            If ``prim_haloprop`` is not passed, then ``table`` keyword argument
            must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop`` keyword argument
            must be passed.

        seed : int, optional
            Random number seed used to generate the Monte Carlo realization.
            Default is None.

        Returns
        -------
        mc_prim_galprop : array
            Float array giving the Monte Carlo realization of primary galaxy
            properties of centrals in halos of the given mass. All primary
            galaxy properties returned are above the threshold.
        """

        # Retrieve the array storing the mass-like variable.
        if 'table' in list(kwargs.keys()):
            mass = kwargs['table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in list(kwargs.keys()):
            mass = kwargs['prim_haloprop']
        else:
            msg = ("\nYou must pass either a ``table`` or ``prim_haloprop``"
                   "argument to the ``mc_prim_galprop`` function of the "
                   "``Cacciato09Cens`` class.\n")
            raise HalotoolsError(msg)

        log_median_prim_galprop = np.log10(self.median_prim_galprop(**kwargs))
        mean_occupation = self.mean_occupation(**kwargs)

        if np.any(mean_occupation <= 0):
            msg = (
                "\nOne of the input halos to the ``mc_prim_galprop``  function "
                "of the ``Cacciato09Cens`` class has (virtually) no expected "
                "galaxy.\n")
            raise HalotoolsError(msg)

        seed = kwargs.get('seed', None)

        prim_galprop = np.zeros(len(mean_occupation))

        with NumpyRNGContext(seed):

            # Draw cumulative distribution function (CDF) values for the
            # primary galaxy properties in [0, 1).
            x = np.random.random(len(mass))

            # Take into account that the occupation with one central sets a
            # lower limit on the CDF values. We also compute 1 - CDF because
            # for low expected occupations CDF ~ 1 which can lead to numerical
            # problems.
            cdf = mean_occupation * x + (1 - mean_occupation)
            cdfc = mean_occupation * (1 - x)  # 1 - cdf

            # Draw primary galaxy properties.
            mask = cdf <= 0.5
            prim_galprop[mask] = 10**(-erfcinv(2 * cdf[mask]) *
                                      np.sqrt(2 * self.param_dict['sigma']**2) +
                                      log_median_prim_galprop[mask])
            mask = np.logical_not(mask)  # cdf > 0.5
            prim_galprop[mask] = 10**(erfcinv(2 * cdfc[mask]) *
                                      np.sqrt(2 * self.param_dict['sigma']**2) +
                                      log_median_prim_galprop[mask])

        if 'table' in list(kwargs.keys()):
            kwargs['table'][self.prim_galprop_key][:] = prim_galprop

        return prim_galprop


class Cacciato09Sats(OccupationComponent):
    r""" CLF-style model for the satellite galaxy occupation. Since it is a CLF
    model, it also assigns a primary galaxy property to galaxies in addition to
    effectively being an HOD model. The primary galaxy property can for example
    be luminosity or stellar mass.

    See :ref:`cacciato09_composite_model` for a tutorial on this model.
    """

    def __init__(self, threshold=10.0, prim_haloprop_key='halo_m180b',
                 prim_galprop_key='luminosity', **kwargs):
        r"""
        Parameters
        ----------
            Logarithm of the primary galaxy property threshold. If the primary
            galaxy property is luminosity, it is given in h=1 solar luminosity
            units.

        prim_haloprop_key : string, optional
            String giving the column name of the primary halo property governing
            the occupation statistics of satellite galaxies.

        prim_galprop_key : string, optional
            String giving the column name of the primary galaxy property that
            is assigned.

        Examples
        --------
        >>> sat_model = Cacciato09Sats()
        >>> sat_model = Cacciato09Sats(threshold = 11.25)
        >>> sat_model = Cacciato09Sats(prim_haloprop_key = 'halo_mvir')

        """
        super(Cacciato09Sats, self).__init__(
            gal_type='satellites', threshold=threshold,
            upper_occupation_bound=float("inf"),
            prim_haloprop_key=prim_haloprop_key,
            **kwargs)

        self._mock_generation_calling_sequence = ['mc_occupation',
                                                  'mc_prim_galprop']
        self._galprop_dtypes_to_allocate = np.dtype([(prim_galprop_key, 'f8')])
        self.prim_galprop_key = prim_galprop_key
        self.param_dict = self.get_default_parameters()
        self.central_occupation_model = Cacciato09Cens(
            threshold=threshold, prim_haloprop_key=prim_haloprop_key,
            prim_galprop_key=prim_galprop_key, **kwargs)
        self._methods_to_inherit = (
            ['mc_occupation', 'mean_occupation', 'mc_prim_galprop', 'clf',
             'phi_sat', 'alpha_sat', 'prim_galprop_cut']
            )
        self.publications = ['arXiv:0807.4932']

    def get_default_parameters(self):
        r""" Return the values of ``self.param_dict`` according to
        the best-fit values of the WMAP3 model in Table 3 of arXiv:0807.4932.
        In this analysis, halo masses have been defined using an overdensity of
        180 times the background density of the Universe.
        """

        param_dict = (
            {'a_1': 0.501,
             'a_2': 2.106,
             'log_M_2': 14.28,
             'b_0': -0.766,
             'b_1': 1.008,
             'b_2': -0.094,
             'delta_1': 0.0,
             'delta_2': 0.0,
             'log_L_0': 9.935,
             'log_M_1': 11.07,
             'gamma_1': 3.273,
             'gamma_2': 0.255}
            )

        return param_dict

    def _update_central_params(self):
        r"""
        Private method to update the model parameters.
        """

        for key, value in self.param_dict.items():
            if key in self.central_occupation_model.param_dict:
                self.central_occupation_model.param_dict[key] = value

    def phi_sat(self, **kwargs):
        r""" Return the normalization for the CLF as a function of the input
        table. See equation (36) in Cacciato et al. (2009) for details.
        The normalization refers to $\phi_s^\star$.

        Parameters
        ----------
        prim_haloprop : array, optional
            Array of mass-like variable upon which the calculation is based.
            If ``prim_haloprop`` is not passed, then ``table`` keyword argument
            must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop`` keyword argument
            must be passed.

        Returns
        -------
        phi_sat : array_like
            Array containing the CLF normalization values of the halos
            specified.
        """

        # Retrieve the array storing the primary galaxy property.
        if 'table' in list(kwargs.keys()):
            prim_haloprop = kwargs['table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in list(kwargs.keys()):
            prim_haloprop = kwargs['prim_haloprop']
        else:
            msg = ("\nYou must pass either a ``table`` or ``prim_haloprop`` "
                   "argument to the ``alpha_sat`` function of the "
                   "``Cacciato09Sats`` class.\n")
            raise HalotoolsError(msg)

        b_0 = self.param_dict['b_0']
        b_1 = self.param_dict['b_1']
        b_2 = self.param_dict['b_2']
        log_prim_haloprop = np.log10(prim_haloprop)

        return 10**(b_0 + b_1 * (log_prim_haloprop - 12.0) + b_2 *
                    (log_prim_haloprop - 12.0)**2)

    def alpha_sat(self, **kwargs):
        r""" Return the power-law slope of the CLF as a function of the input
        table.

        Parameters
        ----------
        prim_haloprop : array, optional
            Array of mass-like variable upon which the calculation is based.
            If ``prim_haloprop`` is not passed, then ``table`` keyword argument
            must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop`` keyword argument
            must be passed.

        Returns
        -------
        alpha_sat : array_like
            Array containing the CLF power-law slopes of the halos specified.
        """

        # Retrieve the array storing the mass-like variable.
        if 'table' in list(kwargs.keys()):
            mass = kwargs['table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in list(kwargs.keys()):
            mass = kwargs['prim_haloprop']
        else:
            msg = ("\nYou must pass either a ``table`` or ``prim_haloprop`` "
                   "argument to the ``alpha_sat`` function of the "
                   "``Cacciato09Sats`` class.\n")
            raise HalotoolsError(msg)

        a_1 = self.param_dict['a_1']
        a_2 = self.param_dict['a_2']
        log_m_2 = self.param_dict['log_M_2']

        return -2.0 + a_1 * (1.0 - 2.0 / np.pi * np.arctan(a_2 * (np.log10(
            mass) - log_m_2)))

    def prim_galprop_cut(self, **kwargs):
        r""" Return the cut-off primary galaxy properties of the CLF as a
        function of the input table. See equation (38) in Cacciato et al. (2009)
        for details. The cut-off primary galaxy property refers to $\L_s^\star$.

        Parameters
        ----------
        prim_haloprop : array, optional
            Array of mass-like variable upon which the calculation is based.
            If ``prim_haloprop`` is not passed, then ``table`` keyword argument
            must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop`` keyword argument
            must be passed.

        Returns
        -------
        prim_galprop_cut : array_like
            Array containing the cut-off primary galaxy property of the halos
            specified.
        """

        if not ('table' in list(kwargs.keys()) or 'prim_haloprop'
                in list(kwargs.keys())):
            msg = ("\nYou must pass either a ``table`` or ``prim_haloprop`` "
                   "argument to the ``prim_galprop_cut`` function of the "
                   "``Cacciato09Sats`` class.\n")
            raise HalotoolsError(msg)

        self._update_central_params()

        med_prim_galprop = self.central_occupation_model.median_prim_galprop(
            **kwargs)

        return med_prim_galprop * 0.562

    def mean_occupation(self, prim_galprop_min=None, prim_galprop_max=None,
                        **kwargs):
        r""" Expected number of satellite galaxies in a halo. Derived from
        integrating the CLF from the luminosity threshold to infinity.

        Parameters
        ----------
        prim_galprop_min : float, optional
            Lower limit of the CLF integration used to calculate the expected
            number of satellite galaxies. If not specified, the lower limit is
            the threshold.

        prim_galprop_max : float, optional
            Upper limit of the CLF integration used to calculate the expected
            number of satellite galaxies. If not specified, the upper limit is
            infinity.

        prim_haloprop : array, optional
            Array of mass-like variable upon which occupation statistics are
            based.
            If ``prim_haloprop`` is not passed, then ``table`` keyword argument
            must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop`` keyword argument
            must be passed.

        Returns
        -------
        mean_nsat : array
            Mean number of satellite galaxies in the halo of the input mass.
        """

        # Retrieve the array storing the primary halo property.
        if 'table' in list(kwargs.keys()):
            prim_haloprop = kwargs['table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in list(kwargs.keys()):
            prim_haloprop = kwargs['prim_haloprop']
        else:
            msg = ("\nYou must pass either a ``table`` or ``prim_haloprop`` "
                   "argument to the ``mean_occupation`` function of the "
                   "``Cacciato09Sats`` class.\n")
            raise HalotoolsError(msg)

        if prim_galprop_min is not None:
            prim_galprop_min = prim_galprop_min
        else:
            prim_galprop_min = 10**self.threshold

        if prim_galprop_max is not None:
            if prim_galprop_max <= prim_galprop_min:
                msg = ("\nFor the ``mean_occupation`` function of the "
                       "``Cacciato09Sats`` class the ``prim_galprop_max`` "
                       "keyword must be bigger than 10^threshold or "
                       "``prim_galprop_min`` if provided.\n")
                raise HalotoolsError(msg)

        alpha_sat = self.alpha_sat(prim_haloprop=prim_haloprop)
        prim_galprop_cut = self.prim_galprop_cut(prim_haloprop=prim_haloprop)
        phi_sat = self.phi_sat(prim_haloprop=prim_haloprop)
        delta = 10**(self.param_dict['delta_1'] + self.param_dict['delta_2'] *
                     (np.log10(prim_haloprop) - 12))

        result = custom_incomplete_gamma(alpha_sat / 2.0 + 0.5,
                                         delta * (prim_galprop_min /
                                                  prim_galprop_cut)**2)
        if prim_galprop_max is not None:
            result = result - custom_incomplete_gamma(
                alpha_sat / 2.0 + 0.5,
                delta * (prim_galprop_max / prim_galprop_cut)**2)

        result = result * (phi_sat / 2.0) * delta**(- (alpha_sat + 1.0) / 2.0)
        return result

    def clf(self, prim_galprop=1e10, prim_haloprop=1e12):
        r""" Return the CLF in units of dn/dlogL for the primary halo property
        and galaxy property L.

        Parameters
        ----------
        prim_haloprop : array_like, optional
            Array of mass-like variable upon which the calculation is based.

        prim_galprop : array_like, optional
            Array of luminosity-like variable of the galaxy upon which the
            calculation is based.

        Returns
        -------
        clf : array_like
            Array containing the CLF in units of dN/dlogL. If ``prim_haloprop``
            has only one element or is a scalar, the same primary halo property
            is assumed for all CLF values. Similarly, if ``prim_galprop`` has
            only one element or is a scalar, the same primary galaxy property is
            assumed throughout.
        """

        prim_galprop = np.atleast_1d(prim_galprop)
        prim_haloprop = np.atleast_1d(prim_haloprop)

        try:
            assert ((len(prim_haloprop) == 1) or (len(prim_galprop) == 1) or
                    (len(prim_haloprop) == (len(prim_galprop))))
        except AssertionError:
            msg = ("If both ``prim_galprop`` and ``prim_haloprop`` are arrays"
                   " with multiple elements, they must have the same length.\n")
            raise HalotoolsError(msg)

        phi_sat = self.phi_sat(prim_haloprop=prim_haloprop)
        alpha_sat = self.alpha_sat(prim_haloprop=prim_haloprop)
        prim_galprop_cut = self.prim_galprop_cut(prim_haloprop=prim_haloprop)
        delta = 10**(self.param_dict['delta_1'] + self.param_dict['delta_2'] *
                     (np.log10(prim_haloprop) - 12))

        return (phi_sat * (prim_galprop / prim_galprop_cut)**(alpha_sat + 1) *
                np.exp(-delta * (prim_galprop / prim_galprop_cut)**2) *
                np.log(10))

    def mc_prim_galprop(self, **kwargs):
        r""" Method to generate Monte Carlo realizations of the primary galaxy
        property of galaxies under the constraint that the primary galaxy
        property is above the primary galaxy property threshold.

        Parameters
        ----------
        prim_haloprop : array, optional
            Array of mass-like variable upon which the primary galaxy properties
            are based.
            If ``prim_haloprop`` is not passed, then ``table`` keyword argument
            must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop`` keyword argument
            must be passed.

        seed : int, optional
            Random number seed used to generate the Monte Carlo realization.
            Default is None.

        Returns
        -------
        mc_prim_galprop : array
            Float array giving the Monte Carlo realization of primary galaxy
            properties of satellites in halos of the given mass. All primary
            galaxy properties returned are above the threshold.
        """

        # Retrieve the array storing the mass-like variable.
        if 'table' in list(kwargs.keys()):
            mass = kwargs['table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in list(kwargs.keys()):
            mass = kwargs['prim_haloprop']
        else:
            msg = ("\nYou must pass either a ``table`` or ``prim_haloprop`` "
                   "argument to the ``mc_prim_galprop`` function of the "
                   "``Cacciato09Sats`` class.\n")
            raise HalotoolsError(msg)

        alpha_sat = self.alpha_sat(**kwargs)
        prim_galprop_cut = self.prim_galprop_cut(**kwargs)
        prim_galprop = np.zeros(len(mass))

        if np.any(alpha_sat > 10):
            msg = ("\nThe ``mc_prim_galprop`` function of the "
                   "``Cacciato09Sats`` class does not support models where "
                   "alpha_sat is bigger than 10.\n")
            raise HalotoolsError(msg)

        seed = kwargs.get('seed', None)
        with NumpyRNGContext(seed):
            while np.any(prim_galprop == 0):
                randoms = np.random.random(size=len(mass) * 2)
                prim_galprop = cacciato09_sats_mc_prim_galprop_engine(
                    prim_galprop, randoms, alpha_sat, prim_galprop_cut,
                    10**self.threshold)

        if 'table' in list(kwargs.keys()):
            kwargs['table'][self.prim_galprop_key][:] = prim_galprop

        return prim_galprop
