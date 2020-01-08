:orphan:

.. _mock_obs_pos_formatting:

**************************************************************************
Formatting your xyz coordinates for Mock Observables calculations
**************************************************************************

The `~halotools.mock_observables` package adopts a specific convention for
how its functions accept spatial coordinate inputs.
If you have a collection of *Npts* coordinates for either *Ndim=2* or *Ndim=3*,
the convention is that you will pass a multi-dimensional Numpy array
of shape *(Npts, Ndim)* storing the coordinates.
All the `~halotools.mock_observables` functions that operate on multi-dimensional data
follow this convention. For example,
`~halotools.mock_observables.tpcf`, `~halotools.mock_observables.void_prob_func`
and `~halotools.mock_observables.mean_delta_sigma` all accept data formatted as
`~numpy.ndarray` of shape *(Npts, 3)*, while `~halotools.mock_observables.angular_tpcf` accepts
a `~numpy.ndarray` of shape *(Npts, 2)*.

Example of how to transform your coordinates
===============================================
Suppose you have a collection of *x, y, z* arrays
storing the spatial positions of halos or galaxies.

>>> Npts = int(1e5)
>>> Lbox = 250
>>> import numpy as np
>>> x = np.random.uniform(0, Lbox, Npts)
>>> y = np.random.uniform(0, Lbox, Npts)
>>> z = np.random.uniform(0, Lbox, Npts)

In order to bundle these arrays into the shape of the multi-dimensional array
used by the `~halotools.mock_observables` package:

>>> pos = np.vstack((x, y, z)).T

The ``pos`` array is now formatted in a form that can be directly passed, for example,
to the `~halotools.mock_observables.tpcf` function as the first positional argument.

If you had two-dimensional data instead:

>>> ra = np.random.uniform(0, 2*np.pi, Npts)
>>> dec = np.random.uniform(-np.pi/2., np.pi/2, Npts)
>>> angular_coords = np.vstack((ra, dec)).T

The ``angular_coords`` array is now formatted in a form that can be directly passed, for example,
to the `~halotools.mock_observables.angular_tpcf` function as the first positional argument.

Using the `~halotools.mock_observables.return_xyz_formatted_array` convenience function
=========================================================================================

When using the `~halotools.mock_observables` package,
the above transformation is so commonly encountered that there is a convenience function
dedicated to handling it:

>>> from halotools.mock_observables import return_xyz_formatted_array
>>> pos = return_xyz_formatted_array(x, y, z)

There is no difference between using
`~halotools.mock_observables.return_xyz_formatted_array` or `numpy.vstack`.
However, the `~halotools.mock_observables.return_xyz_formatted_array` function comes
with two additional features that are worthy of special mention.

Applying redshift-space distortions
---------------------------------------
For some science targets, you may wish to apply redshift-space distortions to your
coordinates before computing the observable statistic.
For example, RSD has a very significant impact on galaxy group identification,
and so most applications using the `~halotools.mock_observables.FoFGroups` feature
will want to account for this effect.
To do, you can use the ``velocity_distortion_dimension`` keyword argument together
with the ``velocity`` keyword storing an array with
the peculiar velocity in whatever dimension you want to distort. In the code below,
we'll apply redshift-space distortions assuming the default cosmology and redshift:

>>> velz = np.random.normal(loc=0, scale=100, size=Npts)
>>> pos_zdist = return_xyz_formatted_array(x, y, z, velocity=velz, velocity_distortion_dimension='z')

Under the distant-observer approximation,
the ``pos_zdist`` array includes the effect of redshift-space distortions,
so that pos_zdist[:, 0] and pos_zdist[:,1] slices
can serve as the directions perpendicular to the line-of-sight,
and pos_zdist[:, 2] the direction parallel to the line-of-sight.

You may wish to use the `return_xyz_formatted_array` function to apply realistic z-space
distortions for mock galaxy samples "observed" at higher redshift, and/or assuming a different cosmology.
This can be handled using the ``redshift`` and/or ``cosmology`` keyword arguments:

>>> from astropy.cosmology import Planck15
>>> redshift = 0.45
>>> velz = np.random.normal(loc=0, scale=100, size=Npts)
>>> pos_zdist = return_xyz_formatted_array(x, y, z, velocity=velz, velocity_distortion_dimension='z', cosmology=Planck15, redshift=redshift)


Selecting subsamples
-----------------------
There is an additional feature of the
`~halotools.mock_observables.return_xyz_formatted_array` function
that allows you to retrieve a specific subsample of your coordinates.
Let's see how this works in a realistic example:
retrieving the spatial positions of quiescent and star-forming samples
in a mock galaxy catalog.

>>> from halotools.empirical_models import PrebuiltSubhaloModelFactory
>>> model = PrebuiltSubhaloModelFactory('smhm_binary_sfr')
>>> from halotools.sim_manager import FakeSim
>>> halocat = FakeSim()
>>> model.populate_mock(halocat)

Our ``model`` now has a ``mock`` object attached to it with a ``galaxy_table``
storing the mock galaxies in the form of an Astropy `~astropy.table.Table`.

>>> x = model.mock.galaxy_table['x']
>>> y = model.mock.galaxy_table['y']
>>> z = model.mock.galaxy_table['z']

>>> red_sample_mask = model.mock.galaxy_table['quiescent'] == True
>>> red_pos = return_xyz_formatted_array(x, y, z, mask = red_sample_mask)
>>> blue_pos = return_xyz_formatted_array(x, y, z, mask = ~red_sample_mask)



























