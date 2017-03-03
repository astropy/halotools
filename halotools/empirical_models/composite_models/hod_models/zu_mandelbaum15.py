"""
Module containing the HOD-style composite model based on Zu & Mandelbaum et al. (2015).
"""
from __future__ import division, print_function, absolute_import, unicode_literals

from ... import model_defaults
from ...occupation_models import ZuMandelbaum15Cens, ZuMandelbaum15Sats
from ...phase_space_models import NFWPhaseSpace, TrivialPhaseSpace

__all__ = ['zu_mandelbaum15_model_dictionary']


def zu_mandelbaum15_model_dictionary(threshold=model_defaults.default_stellar_mass_threshold,
        prim_haloprop_key=model_defaults.prim_haloprop_key, **kwargs):
    """
    Dictionary to build an HOD-style based on Zu & Mandelbaum et al. (2015), arXiv:1505.02781.
    The behavior of this model is governed by an assumed underlying stellar-to-halo-mass relation.

    See :ref:`zu_mandelbaum15_composite_model` for a tutorial on this model.

    There are two populations, centrals and satellites.
    Central occupation statistics are given by a nearest integer distribution
    with first moment given by an ``erf`` function; the class governing this
    behavior is `~halotools.empirical_models.occupation_components.ZuMandelbaum15Cens`.
    Central galaxies are assumed to reside at the exact center of the host halo;
    the class governing this behavior is `~halotools.empirical_models.TrivialPhaseSpace`.

    Satellite occupation statistics are given by a Poisson distribution
    with first moment given by a power law that has been truncated at the low-mass end;
    the class governing this behavior is
    `~halotools.empirical_models.occupation_components.ZuMandelbaum15Sats`;
    satellites in this model follow an (unbiased) NFW profile, as governed by the
    `~halotools.empirical_models.NFWPhaseSpace` class.

    This composite model is built by the `~halotools.empirical_models.HodModelFactory`.

    Parameters
    ----------
    threshold : float, optional
        Stellar mass threshold of the mock galaxy sample.
        Default value is specified in the `~halotools.empirical_models.model_defaults` module.

    concentration_binning : tuple, optional
        Three-element tuple. The first entry will be the minimum
        value of the concentration in the lookup table for the satellite NFW profile,
        the second entry the maximum, the third entry
        the linear spacing of the grid.
        Default is set in `~halotools.empirical_models.model_defaults`.
        If high-precision is not required, the lookup tables will build much faster if
        ``concentration_binning`` is set to (1, 25, 0.5).

    Returns
    -------
    model_dictionary : dict
        Dictionary passed to `~halotools.empirical_models.HodModelFactory`

    Notes
    -----
    Note that in the original Zu & Mandelbaum publication,
    satellite concentrations were slightly lower than their host dark matter halos.
    This is not implemented here but will be changed in a future release.

    Note also that the best-fit parameters of this model are based on the
    ``halo_m200m`` halo mass definition.
    Using alternative choices of mass definition will require altering the
    model parameters in order to mock up the same model published in Zu & Mandelbaum 2015.
    The `Colossus python package <https://bitbucket.org/bdiemer/colossus/>`_
    written by Benedikt Diemer can be used to
    convert between different halo mass definitions. This may be useful if you wish to use an
    existing halo catalog for which the halo mass definition you need is unavailable.


    Examples
    --------
    >>> from halotools.empirical_models import HodModelFactory
    >>> model_dictionary = zu_mandelbaum15_model_dictionary()
    >>> model_instance = HodModelFactory(**model_dictionary)

    The default settings are set in the `~halotools.empirical_models.model_defaults` module.
    To load a model based on a different stellar mass threshold:

    >>> model_dictionary = zu_mandelbaum15_model_dictionary(threshold=11, prim_haloprop_key='halo_mvir')
    >>> model_instance = HodModelFactory(**model_dictionary)

    For this model, you can also use the following syntax candy,
    which accomplishes the same task as the above:

    >>> from halotools.empirical_models import PrebuiltHodModelFactory
    >>> model_instance = PrebuiltHodModelFactory('zu_mandelbaum15', threshold=11, prim_haloprop_key='halo_mvir')

    As with all instances of the `~halotools.empirical_models.PrebuiltHodModelFactory`,
    you can populate a mock by passing the model a halo catalog:

    >>> from halotools.sim_manager import FakeSim
    >>> halocat = FakeSim()
    >>> model_instance.populate_mock(halocat)

    """
    # Build model for centrals
    # Build the occupation model
    centrals_occupation = ZuMandelbaum15Cens(threshold=threshold,
        prim_haloprop_key=prim_haloprop_key, **kwargs)
    centrals_occupation._suppress_repeated_param_warning = True
    # Build the profile model

    centrals_profile = TrivialPhaseSpace(**kwargs)

    # Build model for satellites
    # Build the occupation model
    satellites_occupation = ZuMandelbaum15Sats(threshold=threshold,
        prim_haloprop_key=prim_haloprop_key, **kwargs)
    # Build the profile model
    satellites_profile = NFWPhaseSpace(**kwargs)

    return ({'centrals_occupation': centrals_occupation,
        'centrals_profile': centrals_profile,
        'satellites_occupation': satellites_occupation,
        'satellites_profile': satellites_profile})
