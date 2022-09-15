"""
This module contains the template class `~halotools.empirical_models.OccupationComponent`,
which standardizes the form of the classes responsible for governing galaxy abundance
in all HOD-style models of the galaxy-halo connection.
"""

import numpy as np
from scipy.special import pdtrik

from astropy.utils.misc import NumpyRNGContext

from .. import model_defaults, model_helpers

from ...utils.array_utils import custom_len
from ...custom_exceptions import HalotoolsError

__all__ = ("OccupationComponent",)


class OccupationComponent(object):
    """Abstract base class of any occupation model.
    Functionality is mostly trivial.
    The sole purpose of the base class is to
    standardize the attributes and methods
    required of any HOD-style model for halo occupation statistics.
    """

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        gal_type : string, keyword argument
            Name of the galaxy population whose occupation statistics is being modeled.

        threshold : float, keyword argument
            Threshold value defining the selection function of the galaxy population
            being modeled. Typically refers to absolute magnitude or stellar mass.

        upper_occupation_bound : float, keyword argument
            Upper bound on the number of gal_type galaxies per halo.
            The only currently supported values are unity or infinity.

        second_moment : string, optional
            Method for computing the second occupation moment.
            For centrals, only Bernoulli is supported.
            For satellites, options are "poisson" and "weighted_nearest_integer".
            Satellite default is "poisson".

        prim_haloprop_key : string, optional
            String giving the column name of the primary halo property governing
            the occupation statistics of gal_type galaxies, e.g., ``halo_mvir``.

        See also
        ---------
        :ref:`writing_your_own_hod_occupation_component`
        """
        required_kwargs = ["gal_type", "threshold"]
        model_helpers.bind_required_kwargs(required_kwargs, self, **kwargs)

        try:
            self.prim_haloprop_key = kwargs["prim_haloprop_key"]
        except:
            pass

        try:
            self._upper_occupation_bound = kwargs["upper_occupation_bound"]
        except KeyError:
            msg = "\n``upper_occupation_bound`` is a required keyword argument of OccupationComponent\n"
            raise KeyError(msg)

        self._lower_occupation_bound = 0.0

        self._second_moment = kwargs.get("second_moment", "poisson")

        if not hasattr(self, "param_dict"):
            self.param_dict = {}

        # Enforce the requirement that sub-classes have been configured properly
        required_method_name = "mean_occupation"
        if not hasattr(self, required_method_name):
            raise SyntaxError(
                "Any sub-class of OccupationComponent must "
                "implement a method named %s " % required_method_name
            )

        try:
            self.redshift = kwargs["redshift"]
        except KeyError:
            pass

        # The _methods_to_inherit determines which methods will be directly callable
        # by the composite model built by the HodModelFactory
        try:
            self._methods_to_inherit.extend(["mc_occupation", "mean_occupation"])
        except AttributeError:
            self._methods_to_inherit = ["mc_occupation", "mean_occupation"]

        # The _attrs_to_inherit determines which methods will be directly bound
        # to the composite model built by the HodModelFactory
        try:
            self._attrs_to_inherit.append("threshold")
        except AttributeError:
            self._attrs_to_inherit = ["threshold"]

        if not hasattr(self, "publications"):
            self.publications = []

        # The _mock_generation_calling_sequence determines which methods
        # will be called during mock population, as well as in what order they will be called
        self._mock_generation_calling_sequence = ["mc_occupation"]
        self._galprop_dtypes_to_allocate = np.dtype(
            [("halo_num_" + self.gal_type, "i4")]
        )

    def mc_occupation(self, seed=None, **kwargs):
        """Method to generate Monte Carlo realizations of the abundance of galaxies.

        Parameters
        ----------
        prim_haloprop : array, optional
            Array of mass-like variable upon which occupation statistics are based.
            If ``prim_haloprop`` is not passed, then ``table`` keyword argument must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop`` keyword argument must be passed.

        seed : int, optional
            Random number seed used to generate the Monte Carlo realization.
            Default is None.

        Returns
        -------
        mc_abundance : array
            Integer array giving the number of galaxies in each of the input table.
        """
        first_occupation_moment = self.mean_occupation(**kwargs)
        if self._upper_occupation_bound == 1:
            return self._nearest_integer_distribution(
                first_occupation_moment, seed=seed, **kwargs
            )
        elif self._upper_occupation_bound == float("inf"):
            if self._second_moment == "poisson":
                return self._poisson_distribution(
                    first_occupation_moment, seed=seed, **kwargs
                )
            elif self._second_moment == "weighted_nearest_integer":
                return self._weighted_nearest_integer(
                    first_occupation_moment, seed=seed, **kwargs
                )
            else:
                raise ValueError("Unrecognized second moment")
        else:
            msg = (
                "\nYou have chosen to set ``_upper_occupation_bound`` to some value \n"
                "besides 1 or infinity. In such cases, you must also \n"
                "write your own ``mc_occupation`` method that overrides the method in the \n"
                "OccupationComponent super-class\n"
            )
            raise HalotoolsError(msg)

    def _nearest_integer_distribution(
        self, first_occupation_moment, seed=None, **kwargs
    ):
        """Nearest-integer distribution used to draw Monte Carlo occupation statistics
        for central-like populations with only permissible galaxy per halo.

        Parameters
        ----------
        first_occupation_moment : array
            Array giving the first moment of the occupation distribution function.

        seed : int, optional
            Random number seed used to generate the Monte Carlo realization.
            Default is None.

        Returns
        -------
        mc_abundance : array
            Integer array giving the number of galaxies in each of the input table.
        """
        with NumpyRNGContext(seed):
            mc_generator = np.random.random(custom_len(first_occupation_moment))

        result = np.where(mc_generator < first_occupation_moment, 1, 0)
        if "table" in kwargs:
            kwargs["table"]["halo_num_" + self.gal_type] = result
        return result

    def _poisson_distribution(self, first_occupation_moment, seed=None, **kwargs):
        """Poisson distribution used to draw Monte Carlo occupation statistics
        for satellite-like populations in which per-halo abundances are unbounded.

        Parameters
        ----------
        first_occupation_moment : array
            Array giving the first moment of the occupation distribution function.

        seed : int, optional
            Random number seed used to generate the Monte Carlo realization.
            Default is None.

        Returns
        -------
        mc_abundance : array
            Integer array giving the number of galaxies in each of the input table.
        """
        # We don't use the built-in Poisson number generator so that when a seed
        # is specified, it preserves the ranks among rvs even when mean is changed.
        with NumpyRNGContext(seed):
            result = np.ceil(
                pdtrik(
                    np.random.rand(*first_occupation_moment.shape),
                    first_occupation_moment,
                )
            ).astype(int)
        if "table" in kwargs:
            kwargs["table"]["halo_num_" + self.gal_type] = result
        return result

    def _weighted_nearest_integer(self, first_occupation_moment, seed=None, **kwargs):
        """Non-Poisson distribution for satellite occupation statistics.
        If <Nsat> = i + r where r is the remainder, then the Monte Carlo realization
        will produce Nsat = i with probability r, and Nsat = i + 1 with probability 1-r.

        Parameters
        ----------
        first_occupation_moment : array
            Array giving the first moment of the occupation distribution function.

        seed : int, optional
            Random number seed used to generate the Monte Carlo realization.
            Default is None.

        Returns
        -------
        mc_abundance : array
            Integer array giving the number of galaxies in each of the input table.
        """
        nsat_lo = np.floor(first_occupation_moment).astype(int)
        with NumpyRNGContext(seed):
            uran = np.random.uniform(nsat_lo, nsat_lo + 1, first_occupation_moment.size)
        result = np.where(uran > first_occupation_moment, nsat_lo, nsat_lo + 1)
        if "table" in kwargs:
            kwargs["table"]["halo_num_" + self.gal_type] = result
        return result
