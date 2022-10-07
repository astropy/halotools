r"""
This module contains occupation components used by the Zheng07 composite model.
"""

import numpy as np
from scipy.stats import nbinom

from .zheng07_components import Zheng07Sats, AssembiasZheng07Sats
from ...custom_exceptions import HalotoolsError

__all__ = ("MBK10Sats", "AssembiasMBK10Sats")


class MBK10Sats(Zheng07Sats):
    r"""Power law model for the occupation statistics of satellite galaxies,
    introduced in Kravtsov et al. 2004, arXiv:0308519. This implementation uses
    Zheng et al. 2007, arXiv:0703457, to assign fiducial parameter values,
    with a second moment defined by a negative binomial distribution, as in
    Boylan-Kolchin et al. 2010, arXiv:0911.4484.

    :math:`\langle N_{sat} \rangle_{M} = \left( \frac{M - M_{0}}{M_{1}} \right)^{\alpha}`.

    The behavior of a negative binomial distribution is controlled
    by two parameters, n and p. The relationship between these parameters
    and the mean and variance is as follows:

    :math:`p = \frac{\mu}{\sigma^2}`

    :math:`n = \frac{\mu^2}{\sigma^2 - \mu}`

    In MBK10Sats, the behavior of the deviations from Poisson
    behavior is controlled by the parameter ``nsat_up0``,
    which can take on any value on the real line. The
    parameter ``nsat_up0`` is related to the negative binomial
    distribution parameter p by a simple sigmoid transformation,
    which enforces the mathematical constraint :math:`0<p<1`

    For any Poisson distribution, the following relationship
    between the first and second moments holds:

    :math:`\sigma^2 = \mu`

    In MBK10Sats, changing the parameter ``nsat_up0`` has no effect
    on the first moment, but the second moment is altered as shown
    in the figure below.

    .. image:: /_static/mbk10_nsat_up0_behavior.png

    In terms of two-point correlation functions, deviations from
    Poisson fluctuations only influence satellite-satellite pair counts,
    and so the parameter ``nsat_up0`` will strictly influence
    clustering in the 1-halo term, and the effects will be
    larger in galaxy samples with larger satellite fractions.

    .. image:: /_static/non_poisson_mc_realization.png

    """

    def __init__(self, **kwargs):
        r"""
        Parameters
        ----------
        threshold : float, optional
            Luminosity threshold of the mock galaxy sample. If specified,
            input value must agree with one of the thresholds
            used in Zheng07 to fit HODs:
            [-18, -18.5, -19, -19.5, -20, -20.5, -21, -21.5, -22].
            Default value is specified in the
            `~halotools.empirical_models.model_defaults` module.

        prim_haloprop_key : string, optional
            String giving the column name of the primary halo property governing
            the occupation statistics of gal_type galaxies.
            Default value is specified in the
            `~halotools.empirical_models.model_defaults` module.

        """
        Zheng07Sats.__init__(self, **kwargs)
        self._second_moment = "negative_binomial"
        self.param_dict["nsat_up0"] = 10.0

    def non_poissonian_p(self, **kwargs):
        """Parameter controlling the variance of the negative binomial distribution

        Parameters
        ----------
        prim_haloprop : array, optional
            Array storing a mass-like variable that governs the occupation statistics.
            If ``prim_haloprop`` is not passed, then ``table``
            keyword arguments must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop``
            keyword arguments must be passed.

        Returns
        -------
        p : float or array

        """
        # Retrieve the array storing the mass-like variable
        if "table" in list(kwargs.keys()):
            mass = kwargs["table"][self.prim_haloprop_key]
        elif "prim_haloprop" in list(kwargs.keys()):
            mass = np.atleast_1d(kwargs["prim_haloprop"])
        else:
            msg = (
                "\nYou must pass either a ``table`` or ``prim_haloprop`` argument \n"
                "to the ``mean_occupation`` function of the ``Zheng07Sats`` class.\n"
            )
            raise HalotoolsError(msg)

        up0 = np.zeros_like(mass) + self.param_dict["nsat_up0"]
        p = self._sigmoid(up0)
        return p

    def std_occupation(self, **kwargs):
        """Standard deviation of the occupation statistics
        (square root of the second moment)

        Parameters
        ----------
        prim_haloprop : array, optional
            Array of mass-like variable upon which occupation statistics are based.
            If ``prim_haloprop`` is not passed,
            then ``table`` keyword argument must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed,
            then ``prim_haloprop`` keyword argument must be passed.

        Returns
        -------
        std_nsat : array
            Standard deviation of the number of satellites in the input halos.

        Notes
        -----
        At fixed value of the non_poissonian_p parameter ``nsat_up0``,
        the quantity x = std_occupation**2 / mean_occupation is a constant.
        Increasing up0 will decrease x with no change to mean_occupation.
        In the limit of infinite up0, x approaches the Poisson value of unity

        """
        mu = self.mean_occupation(**kwargs)
        p = self.non_poissonian_p(**kwargs)
        var = mu / p
        return np.sqrt(var)

    def mc_occupation(self, seed=None, **kwargs):
        """Method to generate Monte Carlo realizations of satellite abundance

        Parameters
        ----------
        prim_haloprop : array, optional
            Array of mass-like variable
            upon which occupation statistics are based.
            If ``prim_haloprop`` is not passed,
            then ``table`` keyword argument must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed,
            then ``prim_haloprop`` keyword argument must be passed.

        seed : int, optional
            Random number seed used to generate the Monte Carlo realization.
            Default is None.

        Returns
        -------
        mc_abundance : array
            Integer array giving the number of galaxies in the input halos

        """
        first_occupation_moment = self.mean_occupation(**kwargs)
        p = self.non_poissonian_p(**kwargs)
        n = first_occupation_moment * p / (1 - p)
        rng = np.random.RandomState(seed)
        result = nbinom.rvs(n, p, random_state=rng)

        if "table" in kwargs:
            kwargs["table"]["halo_num_" + self.gal_type] = result
        return result

    def _sigmoid(self, x):
        x0, k, ymin, ymax = 0.5, 0.1, 0, 1
        height_diff = ymax - ymin
        return ymin + height_diff / (1 + np.exp(-k * (x - x0)))


class AssembiasMBK10Sats(AssembiasZheng07Sats):
    r"""Power law model for the occupation statistics of satellite galaxies,
    introduced in Kravtsov et al. 2004, arXiv:0308519. This implementation uses
    Zheng et al. 2007, arXiv:0703457, to assign fiducial parameter values,
    with a second moment defined by a negative binomial distribution, as in
    Boylan-Kolchin et al. 2010, arXiv:0911.4484, and uses the Decorated HOD
    to incorporate galaxy assembly bias.

    In contrast to MBK10Sats, here the non-Poissonian fluctuations are
    implemented at fixed primary and secondary halo properties.
    Note that even in the parent class AssembiasZheng07Sats, fluctuations
    in satellite occupation statistics are non-Poissonian at fixed primary
    halo property, which is a generic feature of assembly bias.

    The behavior of a negative binomial distribution is controlled
    by two parameters, n and p. The relationship between these parameters
    and the mean and variance is as follows:

    :math:`p = \frac{\mu}{\sigma^2}`

    :math:`n = \frac{\mu^2}{\sigma^2 - \mu}`

    In MBK10Sats, the behavior of the deviations from Poisson
    behavior is controlled by the parameter ``nsat_up0``,
    which can take on any value on the real line. The
    parameter ``nsat_up0`` is related to the negative binomial
    distribution parameter p by a simple sigmoid transformation,
    which enforces the mathematical constraint :math:`0<p<1`

    For any Poisson distribution, the following relationship
    between the first and second moments holds:

    :math:`\sigma^2 = \mu`

    In MBK10Sats, changing the parameter ``nsat_up0`` has no effect
    on the first moment, but the second moment is altered as shown
    in the figure below.

    .. image:: /_static/mbk10_nsat_up0_behavior.png

    In terms of two-point correlation functions, deviations from
    Poisson fluctuations only influence satellite-satellite pair counts,
    and so the parameter ``nsat_up0`` will strictly influence
    clustering in the 1-halo term, and the effects will be
    larger in galaxy samples with larger satellite fractions.

    """

    def __init__(self, **kwargs):
        r"""
        Parameters
        ----------
        threshold : float, optional
            Luminosity threshold of the mock galaxy sample. If specified,
            input value must agree with one of the thresholds
            used in Zheng07 to fit HODs:
            [-18, -18.5, -19, -19.5, -20, -20.5, -21, -21.5, -22].
            Default value is specified in the
            `~halotools.empirical_models.model_defaults` module.

        prim_haloprop_key : string, optional
            String giving the column name of the primary halo property governing
            the occupation statistics of gal_type galaxies.
            Default value is specified in the
            `~halotools.empirical_models.model_defaults` module.

        """
        AssembiasZheng07Sats.__init__(self, **kwargs)
        self._second_moment = "negative_binomial"
        self.param_dict["nsat_up0"] = 10.0

    def non_poissonian_p(self, **kwargs):
        """Parameter controlling the variance of the negative binomial distribution

        Parameters
        ----------
        prim_haloprop : array, optional
            Array storing a mass-like variable that governs the occupation statistics.
            If ``prim_haloprop`` is not passed, then ``table``
            keyword arguments must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop``
            keyword arguments must be passed.

        Returns
        -------
        p : float or array

        """
        # Retrieve the array storing the mass-like variable
        if "table" in list(kwargs.keys()):
            mass = kwargs["table"][self.prim_haloprop_key]
        elif "prim_haloprop" in list(kwargs.keys()):
            mass = np.atleast_1d(kwargs["prim_haloprop"])
        else:
            msg = (
                "\nYou must pass either a ``table`` or ``prim_haloprop`` argument \n"
                "to the ``mean_occupation`` function of the ``Zheng07Sats`` class.\n"
            )
            raise HalotoolsError(msg)

        up0 = np.zeros_like(mass) + self.param_dict["nsat_up0"]
        p = self._sigmoid(up0)
        return p

    def std_occupation(self, **kwargs):
        """Standard deviation of the occupation statistics
        (square root of the second moment)

        Parameters
        ----------
        prim_haloprop : array, optional
            Array of mass-like variable upon which occupation statistics are based.
            If ``prim_haloprop`` is not passed,
            then ``table`` keyword argument must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed,
            then ``prim_haloprop`` keyword argument must be passed.

        Returns
        -------
        std_nsat : array
            Standard deviation of the number of satellites in the input halos.

        Notes
        -----
        At fixed value of the non_poissonian_p parameter ``nsat_up0``,
        the quantity x = std_occupation**2 / mean_occupation is a constant.
        Increasing up0 will decrease x with no change to mean_occupation.
        In the limit of infinite up0, x approaches the Poisson value of unity

        """
        mu = self.mean_occupation(**kwargs)
        p = self.non_poissonian_p(**kwargs)
        var = mu / p
        return np.sqrt(var)

    def mc_occupation(self, seed=None, **kwargs):
        """Method to generate Monte Carlo realizations of satellite abundance

        Parameters
        ----------
        prim_haloprop : array, optional
            Array of mass-like variable
            upon which occupation statistics are based.
            If ``prim_haloprop`` is not passed,
            then ``table`` keyword argument must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed,
            then ``prim_haloprop`` keyword argument must be passed.

        seed : int, optional
            Random number seed used to generate the Monte Carlo realization.
            Default is None.

        Returns
        -------
        mc_abundance : array
            Integer array giving the number of galaxies in the input halos

        """
        first_occupation_moment = self.mean_occupation(**kwargs)
        p = self.non_poissonian_p(**kwargs)
        n = first_occupation_moment * p / (1 - p)
        rng = np.random.RandomState(seed)
        result = nbinom.rvs(n, p, random_state=rng)

        if "table" in kwargs:
            kwargs["table"]["halo_num_" + self.gal_type] = result
        return result

    def _sigmoid(self, x):
        x0, k, ymin, ymax = 0.5, 0.1, 0, 1
        height_diff = ymax - ymin
        return ymin + height_diff / (1 + np.exp(-k * (x - x0)))
