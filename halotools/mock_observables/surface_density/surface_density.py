"""
Module containing the `~halotools.mock_observables.surface_density_in_annulus`
and `~halotools.mock_observables.surface_density_in_cylinder` functions used to
calculate galaxy-galaxy lensing.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from .mass_in_cylinders import total_mass_enclosed_in_stack_of_cylinders


__all__ = ('surface_density_in_annulus', 'surface_density_in_cylinder')
__author__ = ('Andrew Hearin', )


def surface_density_in_annulus(centers, particles, particle_masses,
        downsampling_factor, rp_bins, period,
        num_threads=1, approx_cell1_size=None, approx_cell2_size=None):
    """ Calculate the average surface mass density of particles in a stack of annuli
    """

    total_mass_in_stack_of_cylinders = total_mass_enclosed_in_stack_of_cylinders(
            centers, particles, particle_masses, downsampling_factor, rp_bins, period,
            num_threads=num_threads, approx_cell1_size=approx_cell1_size,
            approx_cell2_size=approx_cell2_size)

    total_mass_in_stack_of_annuli = np.diff(total_mass_in_stack_of_cylinders)

    rp_sq = rp_bins * rp_bins
    area_annuli = np.pi * np.diff(rp_sq)
    num_annuli = float(centers.shape[0])
    surface_densities = total_mass_in_stack_of_annuli / (area_annuli * num_annuli)

    return surface_densities


def surface_density_in_cylinder(centers, particles, particle_masses,
        downsampling_factor, rp_bins, period,
        num_threads=1, approx_cell1_size=None, approx_cell2_size=None):
    """ Calculate the average surface mass density of particles in a stack of annuli
    """

    total_mass_in_stack_of_cylinders = total_mass_enclosed_in_stack_of_cylinders(
            centers, particles, particle_masses, downsampling_factor, rp_bins, period,
            num_threads=num_threads, approx_cell1_size=approx_cell1_size,
            approx_cell2_size=approx_cell2_size)

    area_cylinders = np.pi * rp_bins * rp_bins
    num_cylinders = float(centers.shape[0])
    surface_densities = total_mass_in_stack_of_cylinders / (area_cylinders * num_cylinders)

    return surface_densities
