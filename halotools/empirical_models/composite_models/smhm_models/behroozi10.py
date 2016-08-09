"""
Module containing the subhalo-based composite model
based on the Behroozi et al. (2010) stellar-to-halo-mass relation.
"""
from __future__ import division, print_function, absolute_import, unicode_literals

from ...smhm_models import Behroozi10SmHm

from ....sim_manager import sim_defaults

__all__ = ['behroozi10_model_dictionary']


def behroozi10_model_dictionary(redshift=sim_defaults.default_redshift, **kwargs):
    """
    Dictionary that can be passed to the `~halotools.empirical_models.SubhaloModelFactory`
    to build a subhalo-based composite model using the stellar-to-halo-mass relation
    published in Behroozi et al. (2010),
    `arXiv:1205.5807 <http://arxiv.org/abs/astro-ph/1205.5807/>`_.

    For a tutorial on this composite model, see :ref:`behroozi10_composite_model`.

    Parameters
    ----------
    redshift : float, optional
        Redshift of the stellar-to-halo-mass relation of the model. Must be consistent
        with the redshift of the halo catalog you populate. Default value is set in the
        `~halotools.sim_manager.sim_defaults` by the ``default_redshift`` variable.

    prim_haloprop_key : string, optional
        String giving the column name of the primary halo property governing stellar mass.
        Default is set in the `~halotools.empirical_models.model_defaults` module by the
        ``default_smhm_haloprop`` variable.

    scatter_model : object, optional
        Class governing stochasticity of stellar mass. Default scatter is log-normal,
        implemented by the `~halotools.empirical_models.LogNormalScatterModel` class.

    scatter_abscissa : array_like, optional
        Array of values giving the abscissa at which
        the level of scatter will be specified by the input ordinates.
        Default behavior will result in constant scatter at a level set in the
        `~halotools.empirical_models.model_defaults` module by the
        ``default_smhm_scatter`` variable.

    scatter_ordinates : array_like, optional
        Array of values defining the level of scatter at the input abscissa.
        Default behavior will result in constant scatter at a level set in the
        `~halotools.empirical_models.model_defaults` module by the
        ``default_smhm_scatter`` variable.

    Examples
    --------
    Calling the `behroozi10_model_dictionary` with no arguments retrieves a python dictionary
    that can be passed to the `~halotools.empirical_models.SubhaloModelFactory` to build the
    `behroozi10` composite model with default settings:

    >>> from halotools.empirical_models import SubhaloModelFactory
    >>> model_dictionary = behroozi10_model_dictionary()
    >>> model_instance = SubhaloModelFactory(**model_dictionary)

    As with all instances of the `~halotools.empirical_models.PrebuiltHodModelFactory`,
    you can populate a mock by passing the model a halo catalog:

    >>> from halotools.sim_manager import FakeSim
    >>> halocat = FakeSim(redshift = model_instance.redshift)
    >>> model_instance.populate_mock(halocat)

    """

    stellar_mass_model = Behroozi10SmHm(redshift=redshift, **kwargs)
    return {'stellar_mass': stellar_mass_model}
