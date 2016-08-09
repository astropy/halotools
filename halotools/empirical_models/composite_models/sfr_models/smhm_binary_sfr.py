"""
Module containing a pre-built subhalo-based model
that has a binary quenching feature.
"""
from __future__ import division, print_function, absolute_import

from ... import model_defaults
from ...smhm_models import Behroozi10SmHm
from ...component_model_templates import BinaryGalpropInterpolModel

from ....sim_manager import sim_defaults


__all__ = ['smhm_binary_sfr_model_dictionary']


def smhm_binary_sfr_model_dictionary(
        prim_haloprop_key=model_defaults.default_smhm_haloprop,
        smhm_model=Behroozi10SmHm,
        scatter_level=0.2,
        redshift=sim_defaults.default_redshift,
        sfr_abscissa=[12, 15], sfr_ordinates=[0.25, 0.75], logparam=True,
        **kwargs):
    """ Dictionary to build a subhalo-based model for both stellar mass
    and star-formation rate.

    Stellar masses are based on `~halotools.empirical_models.Behroozi10SmHm` by default.
    SFR-designation is binary and is based on an input quenched fraction as a function
    of some halo property such as virial mass. In particular, SFR designation is determined by
    interpolating between a set of input control points, with default
    behavior being a 25% quiescent fraction for galaxies
    residing in Milky Way halos, and 75% for cluster galaxies.

    Since `smhm_binary_sfr_model_dictionary` does not discriminate
    between centrals and satellites in the SFR assignment,
    this model is physically unrealistic and is
    included here primarily for demonstration purposes.

    Parameters
    ----------
    prim_haloprop_key : string, optional
        String giving the column name of the primary halo property governing
        the galaxy propery being modeled.
        Default is set in the `~halotools.empirical_models.model_defaults` module.

    smhm_model : object, optional
        Sub-class of `~halotools.empirical_models.PrimGalpropModel` governing
        the stellar-to-halo-mass relation. Default is `Behroozi10SmHm`.

    scatter_level : float, optional
        Constant amount of scatter in stellar mass, in dex. Default is 0.2.

    redshift : float, optional
        Redshift of the halo hosting the galaxy. Used to evaluate the
        stellar-to-halo-mass relation. Default is set in `~halotools.sim_manager.sim_defaults`.

    sfr_abscissa : array, optional
        Values of the primary halo property at which the quiescent fraction is specified.
        Default is [12, 15], in accord with the default True value for ``logparam``.

    sfr_ordinates : array, optional
        Values of the quiescent fraction when evaluated at the input abscissa.
        Default is [0.25, 0.75]

    logparam : bool, optional
        If set to True, the interpolation will be done
        in the base-10 logarithm of the primary halo property,
        rather than linearly. Default is True.

    threshold : float, optional
        Stellar mass threshold of mock galaxy catalog. Default is None,
        in which case the lower bound on stellar mass will be entirely determined
        by the resolution of the N-body simulation and the model parameters.

    Returns
    -------
    model : object
        Instance of `~halotools.empirical_models.SubhaloModelFactory`

    Examples
    --------
    >>> from halotools.empirical_models import SubhaloModelFactory
    >>> model_dictionary = smhm_binary_sfr_model_dictionary()
    >>> model_instance = SubhaloModelFactory(**model_dictionary)


    """

    sfr_model = BinaryGalpropInterpolModel(sfr_abscissa, sfr_ordinates,
        galprop_name='quiescent', prim_haloprop_key=prim_haloprop_key,
        logparam=logparam, **kwargs)

    sm_model = smhm_model(
        prim_haloprop_key=prim_haloprop_key, redshift=redshift,
        scatter_abscissa=[12], scatter_ordinates=[scatter_level], **kwargs)

    model_dictionary = {'stellar_mass': sm_model, 'quiescent': sfr_model}

    return model_dictionary
