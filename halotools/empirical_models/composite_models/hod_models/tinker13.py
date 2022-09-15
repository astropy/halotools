"""
Module containing some commonly used composite HOD models.
"""
from __future__ import division, print_function, absolute_import, unicode_literals

from ... import model_defaults
from ...occupation_models import tinker13_components
from ...phase_space_models import NFWPhaseSpace, TrivialPhaseSpace

__all__ = ["tinker13_model_dictionary"]


def tinker13_model_dictionary(
    threshold=model_defaults.default_stellar_mass_threshold, **kwargs
):
    """Dictionary to build an HOD-style based on Tinker et al. (2013), arXiv:1308.2974.

    See :ref:`tinker13_composite_model` for a tutorial on this model.

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

    Examples
    ----------
    The simplest way to instantiate the tinker13 model is
    using the `~halotools.empirical_models.PrebuiltHodModelFactory` class:

    >>> from halotools.empirical_models import PrebuiltHodModelFactory
    >>> model_instance = PrebuiltHodModelFactory('tinker13')

    Alternatively, you can pass the returned values of `tinker13_model_dictionary` to
    `~halotools.empirical_models.HodModelFactory`. The calling signature is slightly
    more complicated relative to, for example,
    `~halotools.empirical_models.leauthaud11_model_dictionary` because
    `tinker13_model_dictionary` also returns a supplementary_dictionary specifying
    the `model_feature_calling_sequence` (see :ref:`model_feature_calling_sequence_mechanism`).

    >>> model_dictionary, supplementary_dictionary = tinker13_model_dictionary()
    >>> constructor_kwargs = model_dictionary
    >>> for key in supplementary_dictionary: constructor_kwargs[key] = supplementary_dictionary[key]
    >>> from halotools.empirical_models import HodModelFactory
    >>> model_instance = HodModelFactory(**constructor_kwargs)

    To load a model based on a different threshold and redshift:

    >>> model_instance = PrebuiltHodModelFactory('tinker13', threshold = 11, redshift = 2)

    Or, equivalently,

    >>> model_dictionary, supplementary_dictionary = tinker13_model_dictionary(threshold = 11, redshift = 2)
    >>> constructor_kwargs = model_dictionary
    >>> for key in supplementary_dictionary: constructor_kwargs[key] = supplementary_dictionary[key]
    >>> model_instance = HodModelFactory(**constructor_kwargs)

    As with all instances of the `~halotools.empirical_models.PrebuiltHodModelFactory`,
    you can populate a mock by passing the model a halo catalog:

    >>> from halotools.sim_manager import FakeSim
    >>> halocat = FakeSim(redshift = model_instance.redshift)
    >>> model_instance.populate_mock(halocat)

    """

    # Build the occupation model
    centrals_occupation = tinker13_components.Tinker13Cens(
        threshold=threshold, **kwargs
    )
    centrals_occupation._suppress_repeated_param_warning = True
    # Build the profile model

    centrals_profile = TrivialPhaseSpace(**kwargs)

    # Build the occupation model
    quiescent_satellites_occupation = tinker13_components.Tinker13QuiescentSats(
        threshold=threshold, **kwargs
    )
    # Build the profile model
    quiescent_satellites_profile = NFWPhaseSpace(
        concentration_binning=(1, 35, 1), **kwargs
    )

    # Build the occupation model
    active_satellites_occupation = tinker13_components.Tinker13ActiveSats(
        threshold=threshold, **kwargs
    )
    # Build the profile model
    active_satellites_profile = NFWPhaseSpace(
        concentration_binning=(1, 35, 1), **kwargs
    )
    active_satellites_profile.new_haloprop_func_dict.pop("conc_NFWmodel")

    model_dictionary = {
        "centrals_occupation": centrals_occupation,
        "centrals_profile": centrals_profile,
        "quiescent_satellites_profile": quiescent_satellites_profile,
        "quiescent_satellites_occupation": quiescent_satellites_occupation,
        "active_satellites_profile": active_satellites_profile,
        "active_satellites_occupation": active_satellites_occupation,
    }

    gal_type_list = ["centrals", "active_satellites", "quiescent_satellites"]
    model_feature_calling_sequence = (
        "centrals_occupation",
        "quiescent_satellites_occupation",
        "active_satellites_occupation",
        "centrals_profile",
        "quiescent_satellites_profile",
        "active_satellites_profile",
    )
    supplementary_dictionary = {
        "gal_type_list": gal_type_list,
        "model_feature_calling_sequence": model_feature_calling_sequence,
    }

    return model_dictionary, supplementary_dictionary
