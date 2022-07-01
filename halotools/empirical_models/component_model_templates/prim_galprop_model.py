"""
Module containing the `~halotools.empirical_models.PrimGalpropModel` class
used to model the mapping between the "primary galaxy property" (usually stellar mass)
and some underlying halo property (such as halo mass).
"""

from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np
from warnings import warn

from . import LogNormalScatterModel

from .. import model_defaults

from ...sim_manager import sim_defaults


__all__ = ("PrimGalpropModel",)
__author__ = ("Andrew Hearin",)


class PrimGalpropModel(object):
    """Abstract container class for models connecting table to their primary
    galaxy property, e.g., stellar mass or luminosity.

    """

    def __init__(
        self,
        galprop_name,
        prim_haloprop_key=model_defaults.default_smhm_haloprop,
        scatter_model=LogNormalScatterModel,
        **kwargs
    ):
        """
        Parameters
        ----------
        galprop_name : string
            Name of the galaxy property being assigned. Most likely,
            this is either ``stellar mass`` or ``luminosity``,
            but any name is permissible, e.g. ``baryonic_mass``.
            Whatever you choose, this will be name of the
            column assigned to your mock galaxy catalog,
            and your model will have methods with the following two names
            deriving from your choice:
            ``mean_galprop_name``, ``mc_galprop_name``.

        prim_haloprop_key : string, optional
            String giving the column name of the primary halo property governing
            stellar mass.
            Default is set in the `~halotools.empirical_models.model_defaults` module.

        scatter_model : object, optional
            Class governing stochasticity of stellar mass. Default scatter is log-normal,
            implemented by the `LogNormalScatterModel` class.

        redshift : float, optional
            Redshift of the stellar-to-halo-mass relation. Default is 0.

        scatter_abscissa : array_like, optional
            Array of values giving the abscissa at which
            the level of scatter will be specified by the input ordinates.
            Default behavior will result in constant scatter at a level set in the
            `~halotools.empirical_models.model_defaults` module.

        scatter_ordinates : array_like, optional
            Array of values defining the level of scatter at the input abscissa.
            Default behavior will result in constant scatter at a level set in the
            `~halotools.empirical_models.model_defaults` module.

        """

        self.galprop_name = galprop_name
        self.prim_haloprop_key = prim_haloprop_key

        if "redshift" in kwargs:
            self.redshift = float(max(0, kwargs["redshift"]))

        self.scatter_model = scatter_model(
            prim_haloprop_key=self.prim_haloprop_key, **kwargs
        )

        self._build_param_dict(**kwargs)

        # Enforce the requirement that sub-classes have been configured properly
        required_method_name = "mean_" + self.galprop_name
        if not hasattr(self, required_method_name):
            raise SyntaxError(
                "Any sub-class of PrimGalpropModel must "
                "implement a method named %s " % required_method_name
            )

        # If the sub-class did not implement their own Monte Carlo method mc_galprop,
        # then use _mc_galprop and give it the usual name
        if not hasattr(self, "mc_" + self.galprop_name):
            setattr(self, "mc_" + self.galprop_name, self._mc_galprop)

        # The _mock_generation_calling_sequence determines which methods
        # will be called during mock population, as well as in what order they will be called
        self._mock_generation_calling_sequence = ["mc_" + self.galprop_name]
        self._galprop_dtypes_to_allocate = np.dtype([(str(self.galprop_name), "f4")])

        # The _methods_to_inherit determines which methods will be directly callable
        # by the composite model built by the HodModelFactory
        method_names_to_inherit = [
            "mc_" + self.galprop_name,
            "mean_" + self.galprop_name,
        ]
        try:
            self._methods_to_inherit.extend(method_names_to_inherit)
        except AttributeError:
            self._methods_to_inherit = method_names_to_inherit

    def mean_scatter(self, **kwargs):
        """Use the ``param_dict`` of `PrimGalpropModel` to update the ``param_dict``
        of the scatter model, and then call the `mean_scatter` method of
        the scatter model.
        """
        for key in list(self.scatter_model.param_dict.keys()):
            self.scatter_model.param_dict[key] = self.param_dict[key]

        return self.scatter_model.mean_scatter(**kwargs)

    def scatter_realization(self, **kwargs):
        """Use the ``param_dict`` of `PrimGalpropModel` to update the ``param_dict``
        of the scatter model, and then call the `scatter_realization` method of
        the scatter model.
        """
        for key in list(self.scatter_model.param_dict.keys()):
            self.scatter_model.param_dict[key] = self.param_dict[key]

        return self.scatter_model.scatter_realization(**kwargs)

    def _build_param_dict(self, **kwargs):
        """Method combines the parameter dictionaries of the
        smhm model and the scatter model.
        """

        if hasattr(self, "retrieve_default_param_dict"):
            self.param_dict = self.retrieve_default_param_dict()
        else:
            self.param_dict = {}

        scatter_param_dict = self.scatter_model.param_dict

        for key, value in scatter_param_dict.items():
            self.param_dict[key] = value

    def _mc_galprop(self, include_scatter=True, **kwargs):
        """Return the prim_galprop of the galaxies living in the input table.

        Parameters
        ----------
        prim_haloprop : array, optional
            Array of mass-like variable upon which occupation statistics are based.
            If ``prim_haloprop`` is not passed, then ``table`` keyword argument must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop`` keyword argument must be passed.

        redshift : float, optional
            Redshift of the halo hosting the galaxy.

        include_scatter : boolean, optional
            Determines whether or not the scatter model is applied to add stochasticity
            to the galaxy property assignment. Default is True.
            If False, model is purely deterministic, and the behavior is determined
            by the ``mean_galprop`` method of the sub-class.

        Returns
        -------
        prim_galprop : array_like
            Array storing the values of the primary galaxy property
            of the galaxies living in the input table.
        """

        # Interpret the inputs to determine the appropriate redshift
        if "redshift" not in list(kwargs.keys()):
            if hasattr(self, "redshift"):
                kwargs["redshift"] = self.redshift
            else:
                warn(
                    "\nThe PrimGalpropModel class was not instantiated with a redshift,\n"
                    "nor was a redshift passed to the primary function call.\n"
                    "Choosing the default redshift z = %.2f\n"
                    % sim_defaults.default_redshift
                )
                kwargs["redshift"] = sim_defaults.default_redshift

        prim_galprop_func = getattr(self, "mean_" + self.galprop_name)
        galprop_first_moment = prim_galprop_func(**kwargs)

        if include_scatter is False:
            result = galprop_first_moment
        else:
            log10_galprop_with_scatter = np.log10(
                galprop_first_moment
            ) + self.scatter_realization(**kwargs)
            result = 10.0**log10_galprop_with_scatter

        if "table" in kwargs:
            kwargs["table"][self.galprop_name][:] = result

        return result
