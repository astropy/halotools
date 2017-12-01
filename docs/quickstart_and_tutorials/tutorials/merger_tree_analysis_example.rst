:orphan:

.. _example_merger_tree_analysis:

.. currentmodule:: halotools.utils

**************************************************************
Calculating the Sum of Halo Progenitor Masses
**************************************************************

This section of the documentation describes how to use the `crossmatch`
and `group_member_generator` utility functions to analyze subhalo merger trees.
Many more complicated analyses of merger trees can be built upon by
matching the basic patterns shown here, which speed up naive algorithms by
several orders of magnitude.

Halos gain mass by a combination of merges and smooth accretion. If you have two
catalogs of subhalos at successive snapshots, `subhalos_z0` and `subhalos_z1`,
and if the catalog at the earlier timestep `subhalos_z1`
contains a column specifying the halo ID that each subhalo descends into, then
there is sufficient information to compute the sum of the progenitor masses
for every object in `subhalos_z0`. The naive algorithm for this calculation is just a
double for loop with a blind lookup at every step, which quickly becomes prohibitively slow
for subhalo catalogs of modern simulations. The `crossmatch`
and `group_member_generator` utility functions speed up this calculation considerably,
as demonstrated below.

First we create some fake data for demonstration purposes. In the setup below, the `subhalos_z1` catalog is from the snapshot immediately prior to the `subhalos_z0` catalog. The *desc_id* column stores the *halo_id* that each `subhalos_z1` descends into; the same *desc_id* can appear multiple times in the `subhalos_z1` catalog, and there need not be a matching *halo_id* in the `subhalos_z0` catalog.

>>> from astropy.table import Table
>>> import numpy as np

>>> subhalos_z0 = Table()
>>> num_subhalos_z0 = 47893
>>> subhalos_z0['halo_id'] = np.arange(num_subhalos_z0).astype('i8')

>>> subhalos_z1 = Table()
>>> num_subhalos_z1 = 58105
>>> subhalos_z1['halo_id'] = np.arange(num_subhalos_z0, num_subhalos_z0+num_subhalos_z1).astype('i8')
>>> subhalos_z1['desc_id'] = np.random.randint(0, 2*num_subhalos_z0, num_subhalos_z1)
>>> subhalos_z1['halo_mass'] = np.random.uniform(1e10, 1e15, num_subhalos_z1)

Now sort the subhalos in the earlier snapshot so that `subhalos_z1` with a common descendant are grouped together, and build the `group_member_generator` so that it yields the mass of the progenitor halos with each iteration.

>>> from halotools.utils import group_member_generator
>>> subhalos_z1.sort('desc_id')
>>> grouping_key = 'desc_id'
>>> requested_columns = ['halo_mass']
>>> group_gen = group_member_generator(subhalos_z1, grouping_key, requested_columns)

Now we iterate over the newly created generator:

>>> sum_of_coprogenitor_masses = np.zeros(num_subhalos_z1)
>>> for first, last, member_props in group_gen:
...    masses = member_props[0]
...    sum_of_coprogenitor_masses[first:last] = np.sum(masses)
>>> subhalos_z1['coprogenitor_mass_sum'] = sum_of_coprogenitor_masses

In the above loop, there is one step of the loop for each unique *desc_id* that appears in `subhalos_z1`,
and at each new step, all `subhalos_z1` subhalos associated with that descendant are yielded (including the main progenitor mass).
The array *sum_of_coprogenitor_masses* now stores the total mass of the descendant grouping
associated with each subhalo in the earlier timestep. Now we use the `crossmatch` function
to broadcast these results down into the descendant halos.

>>> from halotools.utils import crossmatch
>>> idxA, idxB = crossmatch(subhalos_z1['desc_id'], subhalos_z0['halo_id'])
>>> subhalos_z0['sum_of_progenitor_masses'] = 0.
>>> subhalos_z0['sum_of_progenitor_masses'][idxB] = subhalos_z1['coprogenitor_mass_sum'][idxA]

In the above calculation, the way we set up the fake data, the descendant of every `subhalos_z1` halo did not necessarily appear in the `subhalos_z0` catalog. We can verify this using the `crossmatch` function as follows:

>>> subhalos_z1['has_match'] = False
>>> subhalos_z1['has_match'][idxA] = True
>>> assert not np.all(subhalos_z1['has_match'] == True)

That did not impact our final calculation because of the way `crossmatch` works: the indexing array `idxA` has no entries corresponding to `subhalos_z1` with no matching descendant.

Now let's ask a slightly more complicated question, and exclude the main progenitor mass from the sum. This will tell us how much mass each `subhalos_z0` gained as a result of merging from distinct subhalos. We'll do this by first sorting each *desc_id*-grouping by mass, and excluding the final row corresponding to the most massive progenitor.


>>> subhalos_z1.sort(['desc_id', 'halo_mass'])
>>> grouping_key = 'desc_id'
>>> requested_columns = ['halo_mass']
>>> group_gen = group_member_generator(subhalos_z1, grouping_key, requested_columns)

Because of the two-variable sort, within each grouping the most-massive progenitor will appear last, which makes it easy to iterate over the generator and exclude the mmp from the sum:

>>> sum_of_merging_masses_no_mmp = np.zeros(num_subhalos_z1) - 1.
>>> for first, last, member_props in group_gen:
...    masses = member_props[0]
...    sum_of_merging_masses_no_mmp[first:last] = np.sum(masses[:-1])
>>> subhalos_z1['non_mmp_coprogenitor_mass_sum'] = sum_of_merging_masses_no_mmp

Just as before, we broadcast the newly-added column down to the descendant halos:

>>> idxA, idxB = crossmatch(subhalos_z1['desc_id'], subhalos_z0['halo_id'])
>>> subhalos_z0['mass_gain_from_mergers'] = 0.
>>> subhalos_z0['mass_gain_from_mergers'][idxB] = subhalos_z1['non_mmp_coprogenitor_mass_sum'][idxA]

For further demonstrations of how to use `group_member_generator`, see :ref:`galaxy_catalog_analysis_tutorial1` and :ref:`halo_catalog_analysis_tutorial1`.


