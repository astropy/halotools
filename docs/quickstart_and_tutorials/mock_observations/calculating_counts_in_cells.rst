:orphan:

.. _calculating_counts_in_cells:

**************************************************************************
Calculating counts-in-cells on a realistic galaxy catalog
**************************************************************************

.. currentmodule:: halotools.mock_observables

In this tutorial we will how to use the `counts_in_cylinders` function
on a realistic galaxy catalog, making full use of the variable search length
feature of this function.

Calculation overview
=====================

First we'll generate a mock galaxy catalog using a fake simulation,
and then select two populations of galaxies, the first with relatively
high stellar mass, the second with lower stellar mass. We'll get an estimate
on the virial radius of the high-mass galaxies using the same methods you might
use on a real galaxy catalog in which you don't know the true halo mass of the
galaxies. Then we'll use the `counts_in_cylinders` function
to calculate the number of low-mass galaxies
inside a cylinder centered at each high-mass galaxy, where each
cylinder has a radius equal to twice the estimated virial radius,
and length equal to three times the estimated virial velocity.


Generate fake data
===================

>>> from halotools.sim_manager import FakeSim
>>> halocat = FakeSim()
>>> from halotools.empirical_models import PrebuiltSubhaloModelFactory
>>> model = PrebuiltSubhaloModelFactory('behroozi10', redshift=0)
>>> model.populate_mock(halocat)

>>> gals = model.mock.galaxy_table
>>> high_mass_mask = (gals['stellar_mass'] >= 1e11) & (gals['stellar_mass'] <= 2e11)
>>> high_mass_gals = gals[high_mass_mask]
>>> low_mass_mask = (gals['stellar_mass'] >= 1e10) & (gals['stellar_mass'] <= 5e10)
>>> low_mass_gals = gals[low_mass_mask]

Estimating the virial radius and virial velocity
=================================================

In an analysis of any real galaxy catalog, the true virial radius of mock galaxies
is unknown. However, under the assumption that all ``sample1`` galaxies
are centrals, we can get an estimate of the virial radius of each ``sample1`` galaxy
using a stellar-to-halo mass relation. This assumption will not be perfect
as any real galaxy sample will have satellite interlopers,
but we'll stick with this assumption to get a rough estimate of the appropriate search radius.

To get the estimate, we'll exploit the fact that the central galaxy
stellar-to-halo mass relation is invertible:

>>> import numpy as np
>>> from halotools.empirical_models import Behroozi10SmHm
>>> model = Behroozi10SmHm(redshift = 0)
>>> log_sm_sample1 = np.log10(high_mass_gals['stellar_mass'])
>>> approx_log_mhalo_sample1 = model.mean_log_halo_mass(log_stellar_mass=log_sm_sample1)

We now have a rough estimate of the host halo mass of our high-mass galaxy sample.
From this information, we can use the `~halotools.empirical_models.halo_mass_to_halo_radius`
function to estimate :math:`R_{\rm vir}`:

>>> from halotools.empirical_models import halo_mass_to_halo_radius
>>> cosmology = halocat.cosmology
>>> redshift = 0
>>> mdef = 'vir'
>>> approx_rvir_sample1 = halo_mass_to_halo_radius(10**approx_log_mhalo_sample1, cosmology, redshift, mdef)

To estimate :math:`V_{\rm vir}` we can use the
`~halotools.empirical_models.halo_mass_to_virial_velocity` function:

>>> from halotools.empirical_models import halo_mass_to_virial_velocity
>>> approx_vvir_sample1 = halo_mass_to_virial_velocity(10**approx_log_mhalo_sample1, cosmology, redshift, mdef)

Using the variable search radius feature of counts-in-cylinders
=================================================================

From the previous section, we now have two arrays, ``approx_rvir_sample1``
and ``approx_vvir_sample1``. We want to use these arrays in our call to the
`counts_in_cylinders` function to place a cylinder of radius :math:`2R_{\rm vir}`
and half-length defined by :math:`3V_{\rm vir}`.
Both inputs must be in length units of Mpc/h.
Our ``approx_rvir_sample1`` array already is in Mpc/h, so we have:

>>> proj_search_radius = 2*approx_rvir_sample1

However, our ``approx_vvir_sample1`` array is in km/s.
Since all units in Halotools assume *h=1*, then in Halotools units we have
:math:`H_{0} = 100h` km/s/Mpc, and our virial velocity criteria
gets transformed into a z-dimension length criteria as:

>>> H0 = 100.0
>>> cylinder_half_length = 3*approx_vvir_sample1/H0

Now we just need to place our galaxy positions into redshift-space,
formatting the result into the
appropriately shaped array using the `return_xyz_formatted_array` function:

>>> from halotools.mock_observables import return_xyz_formatted_array
>>> x1, y1, z1 = high_mass_gals['x'], high_mass_gals['y'], high_mass_gals['z']
>>> x2, y2, z2 = low_mass_gals['x'], low_mass_gals['y'], low_mass_gals['z']
>>> vz1, vz2 = high_mass_gals['vz'], low_mass_gals['vz']
>>> sample1 = return_xyz_formatted_array(x1, y1, z1, period=halocat.Lbox, velocity=vz1, velocity_distortion_dimension='z')
>>> sample2 = return_xyz_formatted_array(x2, y2, z2, period=halocat.Lbox, velocity=vz2, velocity_distortion_dimension='z')

We now have all inputs to the `counts_in_cylinders` function put into Mpc/h units
and in the appropriately shaped arrays, so we can call the function:

>>> from halotools.mock_observables import counts_in_cylinders
>>> result = counts_in_cylinders(sample1, sample2, proj_search_radius, cylinder_half_length, period=halocat.Lbox)
