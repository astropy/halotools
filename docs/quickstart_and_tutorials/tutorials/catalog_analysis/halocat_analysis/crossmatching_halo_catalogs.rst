:orphan:

.. _crossmatching_halo_catalogs:

****************************************************
Cross-matching subhalo catalogs
****************************************************

All halo catalogs come with an integer ID column providing a unique 
identifier of the (sub)halo in the catalog. This tutorial demonstrates 
two different examples of how you can use the 
`~halotools.utils.crossmatch` function to exploit this column to create 
"value-added" versions of your halo catalogs. 

Example 1: Combining information from different halo catalogs 
=================================================================
When analyzing halo catalogs, it's a common situation for you to have 
two different versions of a halo catalog, 
one with halo properties that you wish to transfer to the other. 
In general, the two versions may only partially overlap, 
different cuts have have been applied to the catalogs. 
We'll demonstrate this scenario using the `~halotools.sim_manager.FakeSim` 
halo catalog that is randomly generated on-the-fly, but the 
same calculation applies equally well to real halo catalogs, 
or generally any structured data table with an object ID. 

>>> from halotools.sim_manager import FakeSim
>>> halocat1 = FakeSim()
>>> halo_table1 = halocat1.halo_table

>>> halocat2 = FakeSim()
>>> mask = halocat2.halo_table['halo_mvir'] > 1e11
>>> halo_table2 = halocat2.halo_table[mask]

Now let's add some new column information to ``halo_table2`` 
and use the `~halotools.utils.crossmatch` function to transfer 
this information to ``halo_table1``. 

>>> import numpy as np
>>> halo_table2['some_new_column'] = np.random.random(len(halo_table2))

The halo catalog column ``halo_id`` is a Long giving a unique identifier 
to every halo and subhalo in the halo catalog, so we can use that column 
to match one object to the other. 

>>> halo_table1['transferred_column'] = np.zeros(len(halo_table1), dtype = halo_table2['some_new_column'].dtype)
>>> from halotools.utils import crossmatch
>>> idx_table1, idx_table2 = crossmatch(halo_table1['halo_id'], halo_table2['halo_id'])
>>> halo_table1['transferred_column'][idx_table1] = halo_table2['some_new_column'][idx_table2]

Now for those objects in ``halo_table1`` that are also in ``halo_table2``, 
the values from the ``some_new_column`` column will be stored in the 
``transferred_column``; rows without a matching entry will still be set to their 
initial value of zero. 

Example 2: Transferring host halo properties to their subhalos  
=================================================================
When analyzing catalogs that include subhalos, one very common kind of calculation 
that is done over and over is to group subhalos according to some property of the 
host halo, such as host halo mass. Such calculations become easy when there is a 
column in your data table storing the associated host halo property, 
and in order to create such a column, you need to cross-match the 
``halo_id`` column against the ``halo_hostid`` column. 
As described in :ref:`rockstar_subhalo_nomenclature`, for the case of subhalos, 
either of these two columns points to the ``halo_id`` of the host halo. 
So we use the `~halotools.utils.crossmatch` function to add new columns to 
the halo catalog such that some property of the host halo is transferred onto 
all of its subhalos. (Note that many subhalos share a common 
host, and in some cases there may be no matching host halo, 
so this calculation is not a simple table `~astropy.table.join`). 

>>> halocat = FakeSim()
>>> idx_table1, idx_table2 = crossmatch(halocat.halo_table['halo_hostid'], halocat.halo_table['halo_id']) 
>>> halocat.halo_table['host_halo_mvir'] = halocat.halo_table['halo_mvir'] # initialize the array
>>> halocat.halo_table['host_halo_mvir'][idx_table1] = halocat.halo_table[idx_table2]['halo_mvir'] 







