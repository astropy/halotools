r"""
Module containing the `~halotools.empirical_models.BinaryGalpropModel` class
used to map a binary-valued galaxy property to a halo catalog.
"""
from __future__ import division, print_function, absolute_import

import numpy as np
from astropy.utils.misc import NumpyRNGContext

from .. import model_defaults
from .. import model_helpers

from ...utils.array_utils import custom_len
from ...custom_exceptions import HalotoolsError

__all__ = ("BinaryGalpropModel", "BinaryGalpropInterpolModel")
__author__ = ("Andrew Hearin",)


class BinaryGalpropModel(object):
    r"""
    Container class for any component model of a binary-valued galaxy property.

    """

    def __init__(
        self, prim_haloprop_key=model_defaults.default_binary_galprop_haloprop, **kwargs
    ):
        r"""
        Parameters
        ----------
        galprop_name : string, keyword argument
            Name of the galaxy property being assigned.

        prim_haloprop_key : string, optional
            String giving the column name of the primary halo property governing
            the galaxy propery being modeled.
            Default is set in the `~halotools.empirical_models.model_defaults` module.

        """
        required_kwargs = ["galprop_name"]
        model_helpers.bind_required_kwargs(required_kwargs, self, **kwargs)

        self.prim_haloprop_key = prim_haloprop_key

        if "sec_haloprop_key" in list(kwargs.keys()):
            self.sec_haloprop_key = kwargs["sec_haloprop_key"]

        # Enforce the requirement that sub-classes have been configured properly
        required_method_name = "mean_" + self.galprop_name + "_fraction"
        if not hasattr(self, required_method_name):
            raise HalotoolsError(
                "Any sub-class of BinaryGalpropModel must "
                "implement a method named %s " % required_method_name
            )

        setattr(self, "mc_" + self.galprop_name, self._mc_galprop)

        self._mock_generation_calling_sequence = ["mc_" + self.galprop_name]
        self._methods_to_inherit = [
            "mean_" + self.galprop_name + "_fraction",
            "mc_" + self.galprop_name,
        ]

        self._galprop_dtypes_to_allocate = np.dtype([(self.galprop_name, bool)])

    def _mc_galprop(self, seed=None, **kwargs):
        r"""Return a Monte Carlo realization of the galaxy property
        based on draws from a nearest-integer distribution.

        Parameters
        ----------
        prim_haloprop : array, optional
            Array of mass-like variable governing the galaxy property.
            If ``prim_haloprop`` is not passed, then either ``halos`` or ``galaxy_table``
            keyword arguments must be passed.

        halos : object, optional
            Data table storing halo catalog.
            If ``halos`` is not passed, then either ``prim_haloprop`` or ``galaxy_table``
            keyword arguments must be passed.

        galaxy_table : object, optional
            Data table storing galaxy catalog.
            If ``galaxy_table`` is not passed, then either ``prim_haloprop`` or ``halos``
            keyword arguments must be passed.

        seed : int, optional
            Random number seed used to generate the Monte Carlo realization.
            Default is None.

        Returns
        -------
        mc_galprop : array_like
            Array storing the values of the primary galaxy property
            of the galaxies living in the input halos.
        """

        mean_func = getattr(self, "mean_" + self.galprop_name + "_fraction")
        mean_galprop_fraction = mean_func(**kwargs)
        with NumpyRNGContext(seed):
            mc_generator = np.random.random(custom_len(mean_galprop_fraction))
        result = np.where(mc_generator < mean_galprop_fraction, True, False)
        if "table" in kwargs:
            kwargs["table"][self.galprop_name][:] = result
        return result


class BinaryGalpropInterpolModel(BinaryGalpropModel):
    r"""
    Component model for any binary-valued galaxy property
    whose assignment is determined by interpolating between points on a grid.

    One example of a model of this class could be used to help build is
    Tinker et al. (2013), arXiv:1308.2974.
    In this model, a galaxy is either active or quiescent,
    and the quiescent fraction is purely a function of halo mass, with
    separate parameters for centrals and satellites. The value of the quiescent
    fraction is computed by interpolating between a grid of values of mass.
    `BinaryGalpropInterpolModel` is quite flexible, and can be used as a template for
    any binary-valued galaxy property whatsoever. See the examples below for usage instructions.

    """

    def __init__(
        self,
        galprop_abscissa,
        galprop_ordinates,
        logparam=True,
        interpol_method="spline",
        **kwargs
    ):
        r"""
        Parameters
        ----------
        galprop_name : array, keyword argument
            String giving the name of galaxy property being assigned a binary value.

        gal_type : string, optional
            Name of the galaxy population.
            Default is None, in which case the model instance will not have
            the ``gal_type`` attribute.

        prim_haloprop_key : string, optional
            String giving the column name of the primary halo property governing
            stellar mass.
            Default is set in the `~halotools.empirical_models.model_defaults` module.

        galprop_abscissa : array, optional
            Values of the primary halo property at which the galprop fraction is specified.

        galprop_ordinates : array, optional
            Values of the galprop fraction when evaluated at the input abscissa.

        logparam : bool, optional
            If set to True, the interpolation will be done
            in the base-10 logarithm of the primary halo property,
            rather than linearly. Default is True.

        interpol_method : string, optional
            Keyword specifying how `mean_galprop_fraction`
            evaluates input values of the primary halo property.
            The default spline option interpolates the
            model's abscissa and ordinates.
            The polynomial option uses the unique, degree N polynomial
            passing through the ordinates, where N is the number of supplied ordinates.

        input_spline_degree : int, optional
            Degree of the spline interpolation for the case of interpol_method='spline'.
            If there are k abscissa values specifying the model, input_spline_degree
            is ensured to never exceed k-1, nor exceed 5. Default is 3.

        Examples
        -----------
        Suppose we wish to construct a model for whether a central galaxy is
        star-forming or quiescent. We want to set the quiescent fraction to 1/3
        for Milky Way-type centrals (:math:`M_{\mathrm{vir}}=10^{12}M_{\odot}`),
        and 90% for massive cluster centrals (:math:`M_{\mathrm{vir}}=10^{15}M_{\odot}`).
        We can use the `BinaryGalpropInterpolModel` to implement this as follows:

        >>> abscissa, ordinates = [12, 15], [1/3., 0.9]
        >>> cen_quiescent_model = BinaryGalpropInterpolModel(galprop_name='quiescent', galprop_abscissa=abscissa, galprop_ordinates=ordinates, prim_haloprop_key='mvir')

        The ``cen_quiescent_model`` has a built-in method that computes the quiescent fraction
        as a function of mass:

        >>> quiescent_frac = cen_quiescent_model.mean_quiescent_fraction(prim_haloprop =1e12)

        There is also a built-in method to return a Monte Carlo realization of quiescent/star-forming galaxies:

        >>> masses = np.logspace(10, 15, num=100)
        >>> quiescent_realization = cen_quiescent_model.mc_quiescent(prim_haloprop = masses)

        Now ``quiescent_realization`` is a boolean-valued array of the same length as ``masses``.
        Entries of ``quiescent_realization`` that are ``True`` correspond to central galaxies that are quiescent.

        Here is another example of how you could use `BinaryGalpropInterpolModel`
        to construct a simple model for satellite morphology, where the early- vs. late-type
        of the satellite depends on :math:`V_{\mathrm{peak}}` value of the host halo

        >>> sat_morphology_model = BinaryGalpropInterpolModel(galprop_name='late_type', galprop_abscissa=abscissa, galprop_ordinates=ordinates, prim_haloprop_key='vpeak_host')
        >>> vmax_array = np.logspace(2, 3, num=100)
        >>> morphology_realization = sat_morphology_model.mc_late_type(prim_haloprop =vmax_array)

        .. automethod:: _mean_galprop_fraction
        """
        try:
            galprop_name = kwargs["galprop_name"]
        except KeyError:
            raise HalotoolsError(
                "\nAll sub-classes of BinaryGalpropInterpolModel must pass "
                "a ``galprop_name`` keyword argument to the constructor\n"
            )

        setattr(self, "mean_" + galprop_name + "_fraction", self._mean_galprop_fraction)
        super(BinaryGalpropInterpolModel, self).__init__(**kwargs)

        self._interpol_method = interpol_method
        self._logparam = logparam

        galprop_abscissa = np.atleast_1d(galprop_abscissa)
        galprop_ordinates = np.atleast_1d(galprop_ordinates)
        self._test_abscissa_ordinates(galprop_abscissa, galprop_ordinates)
        self._abscissa = galprop_abscissa
        self._ordinates = galprop_ordinates

        try:
            self.gal_type = kwargs["gal_type"]
        except KeyError:
            pass

        if self._interpol_method == "spline":
            if "input_spline_degree" in list(kwargs.keys()):
                self._input_spine_degree = kwargs["input_spline_degree"]
            else:
                self._input_spline_degree = 3
            scipy_maxdegree = 5
            self._spline_degree = np.min(
                [
                    scipy_maxdegree,
                    self._input_spline_degree,
                    custom_len(self._abscissa) - 1,
                ]
            )

        self._abscissa_key = self.galprop_name + "_abscissa"
        try:
            self._ordinates_key_prefix = (
                self.gal_type + "_" + self.galprop_name + "_ordinates"
            )
        except AttributeError:
            self._ordinates_key_prefix = self.galprop_name + "_ordinates"
        self._build_param_dict()

        setattr(self, self.galprop_name + "_abscissa", self._abscissa)

    def _test_abscissa_ordinates(self, galprop_abscissa, galprop_ordinates):
        try:
            assert len(galprop_abscissa) == len(galprop_ordinates)
        except AssertionError:
            msg = "\nInput ``galprop_abscissa`` and ``galprop_ordinates`` must have the same length\n"
            raise HalotoolsError(msg)

        try:
            assert len(set(galprop_abscissa)) == len(galprop_abscissa)
        except AssertionError:
            msg = "\nYour input ``galprop_abscissa`` cannot have any repeated values\n"
            raise HalotoolsError(msg)

        try:
            assert np.all(galprop_abscissa >= 0)
            assert np.all(galprop_ordinates <= 1)
        except AssertionError:
            msg = "\nAll values of the input ``galprop_ordinates`` must be between 0 and 1, inclusive."
            raise HalotoolsError(msg)

    def _build_param_dict(self):

        self._ordinates_keys = [
            self._ordinates_key_prefix + "_param" + str(i + 1)
            for i in range(custom_len(self._abscissa))
        ]
        self.param_dict = {
            key: value for key, value in zip(self._ordinates_keys, self._ordinates)
        }

    def _mean_galprop_fraction(self, **kwargs):
        r"""
        Expectation value of the galprop for galaxies living in the input halos.

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
        mean_galprop_fraction : array_like
            Values of the galprop fraction evaluated at the input primary halo properties.

        """
        # Retrieve the array storing the mass-like variable
        if "table" in list(kwargs.keys()):
            prim_haloprop = kwargs["table"][self.prim_haloprop_key]
        elif "prim_haloprop" in list(kwargs.keys()):
            prim_haloprop = kwargs["prim_haloprop"]
        else:
            raise KeyError(
                "Must pass one of the following keyword arguments to mean_occupation:\n"
                "``table`` or  ``prim_haloprop``"
            )

        if self._logparam is True:
            prim_haloprop = np.log10(prim_haloprop)

        # Update self._abscissa, in case the user has changed it
        self._abscissa = getattr(self, self.galprop_name + "_abscissa")

        model_ordinates = [
            self.param_dict[ordinate_key] for ordinate_key in self._ordinates_keys
        ]
        if self._interpol_method == "polynomial":
            mean_galprop_fraction = model_helpers.polynomial_from_table(
                self._abscissa, model_ordinates, prim_haloprop
            )
        elif self._interpol_method == "spline":
            spline_function = model_helpers.custom_spline(
                self._abscissa, model_ordinates, k=self._spline_degree
            )
            mean_galprop_fraction = spline_function(prim_haloprop)
        else:
            raise HalotoolsError(
                "Input interpol_method must be 'polynomial' or 'spline'."
            )

        # Enforce boundary conditions
        mean_galprop_fraction[mean_galprop_fraction < 0] = 0
        mean_galprop_fraction[mean_galprop_fraction > 1] = 1

        return mean_galprop_fraction
