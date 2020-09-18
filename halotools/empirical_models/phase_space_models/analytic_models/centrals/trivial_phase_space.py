"""
This module contains the `~halotools.empirical_models.TrivialPhaseSpace` class
used to place central galaxies at the center of, and at rest with respect to, their host halo.
"""
from __future__ import division, print_function, absolute_import

import numpy as np

from .... import model_defaults

from .....sim_manager import sim_defaults


__author__ = ["Andrew Hearin"]
__all__ = ["TrivialPhaseSpace"]


class TrivialPhaseSpace(object):
    r""" Profile of central galaxies residing at the exact center of their
    host halo with the exact same velocity as the halo velocity.

    :math:`P(\vec{x}_{\rm cen}, \vec{v}_{\rm cen}) = \delta^{(6)}(\vec{x}_{\rm halo}, \vec{v}_{\rm halo})`.
    """

    def __init__(
        self,
        cosmology=sim_defaults.default_cosmology,
        redshift=sim_defaults.default_redshift,
        mdef=model_defaults.halo_mass_definition,
        halo_boundary_key=None,
        **kwargs
    ):
        """
        Parameters
        ----------
        cosmology : object, optional
            Astropy cosmology object. Default is set in `~halotools.sim_manager.sim_defaults`.

        redshift : float, optional
            Default is set in `~halotools.sim_manager.sim_defaults`.

        mdef: str
            String specifying the halo mass definition, e.g., 'vir' or '200m'.
            Default is set in `~halotools.empirical_models.model_defaults`.
        """
        self._mock_generation_calling_sequence = ["assign_phase_space"]
        self._galprop_dtypes_to_allocate = np.dtype(
            [
                ("x", "f8"),
                ("y", "f8"),
                ("z", "f8"),
                ("vx", "f8"),
                ("vy", "f8"),
                ("vz", "f8"),
            ]
        )

        self.param_dict = {}

        self.cosmology = cosmology
        self.redshift = redshift
        self.mdef = mdef
        if halo_boundary_key is None:
            self.halo_boundary_key = model_defaults.get_halo_boundary_key(self.mdef)
        else:
            self.halo_boundary_key = halo_boundary_key

    def assign_phase_space(self, table, **kwargs):
        r"""
        """
        phase_space_keys = ["x", "y", "z", "vx", "vy", "vz"]
        for key in phase_space_keys:
            table[key][:] = table["halo_" + key][:]
