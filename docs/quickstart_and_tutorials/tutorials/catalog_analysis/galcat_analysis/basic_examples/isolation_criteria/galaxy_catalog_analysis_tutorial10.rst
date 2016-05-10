.. _galaxy_catalog_analysis_tutorial10:

Galaxy Catalog Analysis Example: Identifying isolated galaxies
==============================================================

In this tutorial, we'll start from a mock galaxy catalog and show how to
determine which galaxies are "isolated" according to a variety of
criteria.

There is also an IPython Notebook in the following location that can be 
used as a companion to the material in this section of the tutorial:


    **halotools/docs/notebooks/galcat_analysis/basic_examples/galaxy_catalog_analysis_tutorial10.ipynb**

By following this tutorial together with this notebook, 
you can play around with your own variations of the calculation 
as you learn the basic syntax. 

This tutorial only displays basic usage of the isolation functions. For more sophisticated 
analyses, such as those in which the definition of isolation varies with stellar mass 
on a galaxy-by-galaxy basis, see :ref:`galaxy_catalog_intermediate_analysis_tutorial1`. 

Generate a mock galaxy catalog
------------------------------

Let's start out by generating a mock galaxy catalog into an N-body
simulation in the usual way. Here we'll assume you have the
:math:``z=0`` rockstar halos for the bolshoi simulation, as this is the
default halo catalog.

.. code:: python

    from halotools.empirical_models import PrebuiltSubhaloModelFactory
    model = PrebuiltSubhaloModelFactory('smhm_binary_sfr')

    from halotools.sim_manager import CachedHaloCatalog
    halocat = CachedHaloCatalog(simname = 'bolshoi', redshift = 0, halo_finder = 'rockstar')
    
    model.populate_mock(halocat)

Now suppose the data we are interested in is complete for
:math:`M_{\ast} > 10^{10}M_{\odot},` so we will make a cut on the mock.
Our mock galaxies are stored in the ``galaxy_table`` of ``model.mock``
in the form of an Astropy ``Table``.

.. code:: python

    sample_mask = model.mock.galaxy_table['stellar_mass'] > 1e10
    gals = model.mock.galaxy_table[sample_mask]

Example 1: Isolation in 3d
--------------------------

For the first example we'll find "isolated" galaxies using simple
spherical isolation: a galaxy will be said to be isolated if no other
galaxy resides within a sphere of size ``r_max``.

.. code:: python

    from halotools.mock_observables import spherical_isolation

The calling signature of `~halotools.mock_observables.spherical_isolation` accepts a
multi-dimensional array storing the x, y, z positions of each point. You
can place your points into the appropriate form using
`numpy.vstack([x, y, z]).T`, but below we'll demo how to use the
`~halotools.mock_observables.return_xyz_formatted_array` function for
this purpose, as this function provides additional convenient behavior
that we'll use later in the tutorial.

.. code:: python

    from halotools.mock_observables import return_xyz_formatted_array
    sample1 = return_xyz_formatted_array(gals['x'], gals['y'], gals['z'])

The `~halotools.mock_observables.spherical_isolation` function accepts distinct inputs for
``sample1`` and ``sample2``. The calculation is designed to treat
``sample2`` points as the tracer field, and ``sample1`` the points for
which you would like to apply the isolation criteria, so that for each
point in ``sample1``, the points in ``sample2`` will be searched for
neighbors. Thus if you pass in ``sample1`` as both arguments, you will
search for points in ``sample1`` that are isolated from other points in
``sample1``.

.. code:: python

    r_max = 0.5 # Note that all lengths are in Mpc/h units throughout Halotools
    is_isolated = spherical_isolation(sample1, sample1, r_max, period = model.mock.Lbox)

The returned result ``is_isolated`` is a boolean array; for any galaxy
for which the corresponding entry is ``True``, there are no other
galaxies within 500 kpc/h in the ``gals`` table.

Example 2: Isolation in redshift-space
--------------------------------------

In this next example we'll show how to apply isolation criteria in a
more observationally realistic manner: first we place galaxies into
redshift-space, and then we apply separate conditions for the
perpendicular and line-of-sight directions.

.. code:: python

    from halotools.mock_observables import cylindrical_isolation

    sample1 = return_xyz_formatted_array(gals['x'], gals['y'], gals['z'], 
                velocity=gals['vz'], velocity_distortion_dimension = 'z', period = model.mock.Lbox)

Now let's define the notion of isolation to mean that no other galaxies
lies within a projected distance of 300 kpc/h and a line-of-sight
distance of 500 km/s. All units in Halotools assume *h=1*, with lengths
always in Mpc/h, so we have:

.. code:: python

    rp_max = 0.3

Since *h=1* implies :math:`H_{0} = 100` km/s/Mpc, our 500 km/s velocity
criteria gets transformed into a z-dimension length criteria as:

.. code:: python

    H0 = 100.0
    pi_max = 500./H0

    is_isolated = cylindrical_isolation(sample1, sample1, rp_max, pi_max, period = model.mock.Lbox)

Example 3: Determining isolation from massive galaxies
------------------------------------------------------

In this final example, we'll show how to formulate a different variation
of isolation: let's determine which galaxies in our sample are isolated
from massive galaxies with :math:`M_{\ast} > 3\times10^{11}M_{\odot}.`

This variation can be handled simply: we just use two different samples
of galaxies. We'll demonstrate this using the ``mask`` feature of
``return_xyz_formatted_array``, but you can of course apply your own
mask manually.

.. code:: python

    sample1 = return_xyz_formatted_array(gals['x'], gals['y'], gals['z'], 
                velocity=gals['vz'], velocity_distortion_dimension = 'z', period = model.mock.Lbox)

    sm_cut = 3e11
    sample2 = return_xyz_formatted_array(gals['x'], gals['y'], gals['z'], 
                velocity=gals['vz'], velocity_distortion_dimension = 'z', period = model.mock.Lbox, 
                mask = gals['stellar_mass'] > sm_cut)

    rp_max = 5 # projected separation cut of 5 Mpc/h
    pi_max = 3000./H0 # line-of-sight velocity cut of 3000 km/s
    
    is_isolated = cylindrical_isolation(sample1, sample2, rp_max, pi_max, period = model.mock.Lbox)

Next steps 
------------

This tutorial only displays basic usage of the isolation functions. For more sophisticated 
analyses, such as those in which the definition of isolation varies with stellar mass 
on a galaxy-by-galaxy basis, see :ref:`galaxy_catalog_intermediate_analysis_tutorial1`. 



