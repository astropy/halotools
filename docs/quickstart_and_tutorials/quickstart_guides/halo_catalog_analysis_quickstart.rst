
.. _halo_catalog_analysis_quickstart:

*********************************************
Quickstart guide to analyzing halo catalogs
*********************************************

In this section of the documentation we'll give a quick demonstration 
of how information in Halotools-formatted halo catalogs is organized. 
In particular, you'll see how to access both halo catalog metadata 
as well as the Astropy `~astropy.table.Table` storing the tabular halo data. 

For more in-depth information about how to analyze halo catalogs, 
see the :ref:`halo_catalog_analysis_tutorial` section of the documentation. 
This quickstart guide assumes you have followed the 
:ref:`getting_started` section of the documentation, so that you 
already have the default halo catalog stored on your machine. 

Loading cached halo catalogs into memory
=========================================

To load the default halo catalog into memory, just instantiate 
the `~halotools.sim_manager.CachedHaloCatalog` class with no arguments:

.. code:: python

    from halotools.sim_manager import CachedHaloCatalog
    halocat = CachedHaloCatalog()

You may find it useful to read the documentation of the 
`~halotools.sim_manager.CachedHaloCatalog` class together with this quickstart guide. 

The default halo catalog in Halotools is the redshift-zero Bolshoi simulation 
with halos identified using Rockstar. This is reflected in the metadata of the 
halo catalog:

.. code:: python 

    print(halocat.simname, halocat.halo_finder, halocat.redshift)

.. parsed-literal::

    ('bolshoi', 'rockstar', -0.0003)

Loading alternative catalogs 
-----------------------------

As described in the documentation on the `~halotools.sim_manager.CachedHaloCatalog` class, 
you can access any cached halo catalog using the same syntax as above, but using 
keyword arguments to specify which cached catalog you'd like. For example, if you 
have used the ``halotools/scripts/download_additional_halocat.py`` script to 
download the Bolshoi-Planck *z = 0.5* snapshot, then you can load that catalog 
into memory as follows:

.. code:: python

    halocat = CachedHaloCatalog(simname = 'bolplanck', redshift = 0.5)

Note that the `~halotools.sim_manager.CachedHaloCatalog` class 
works with *any* Halotools-formatted halo catalog stored in any disk location, 
not just Halotools-provided snapshots stored in the default cache location. 
This includes your own reductions of 
`the publicly available Rockstar catalogs <http://hipacc.ucsc.edu/Bolshoi/MergerTrees.html>`_  
and/or your own proprietary simulation 
with halos identified by whatever method you prefer.  

Organization of halo information 
----------------------------------------------------------

A Halotools-formatted halo catalog comes equipped with both the tabular
data associated with the halos, and metadata about the simulation
snapshot. In this quickstart guide, we'll demonstrate how to access both
kinds of information in the two sections below. 

.. _accessing_halo_table_data: 

Accessing the tabular data storing the halo catalog 
=====================================================

The catalog of halos itself is stored as the ``halo_table`` attribute in
the form of an Astropy `~astropy.table.Table` object:

.. code:: python

    halos = halocat.halo_table

To see what halo properties are available, you can use the ``keys`` method, just like a python dictionary

.. code:: python

    print(halos.keys())

.. parsed-literal::

    ['halo_vmax_firstacc', 'halo_dmvir_dt_tdyn', 'halo_macc', 'halo_scale_factor', 'halo_vmax_mpeak', 'halo_m_pe_behroozi', 'halo_xoff', 'halo_spin', 'halo_scale_factor_firstacc', 'halo_c_to_a', 'halo_mvir_firstacc', 'halo_scale_factor_last_mm', 'halo_scale_factor_mpeak', 'halo_pid', 'halo_m500c', 'halo_id', 'halo_halfmass_scale_factor', 'halo_upid', 'halo_t_by_u', 'halo_rvir', 'halo_vpeak', 'halo_dmvir_dt_100myr', 'halo_mpeak', 'halo_m_pe_diemer', 'halo_jx', 'halo_jy', 'halo_jz', 'halo_m2500c', 'halo_mvir', 'halo_voff', 'halo_axisA_z', 'halo_axisA_x', 'halo_axisA_y', 'halo_y', 'halo_b_to_a', 'halo_x', 'halo_z', 'halo_m200b', 'halo_vacc', 'halo_scale_factor_lastacc', 'halo_vmax', 'halo_m200c', 'halo_vx', 'halo_vy', 'halo_vz', 'halo_dmvir_dt_inst', 'halo_rs', 'halo_nfw_conc', 'halo_hostid', 'halo_mvir_host_halo']

You can read about the conventions used to define subhalos vs. host halos in 
the :ref:`rockstar_subhalo_nomenclature` section of the documentation. 
For a thorough discussion of the meaning of each column in these halo catalogs, 
see the appendix of `Rodriguez Puebla et al 2016 <http://arxiv.org/abs/1602.04813>`_.

You can select a particular sample of halos using a Numpy boolean mask:

.. code:: python

    mask = (halos['halo_mvir'] > 1e12) & (halos['halo_mvir'] < 2e12) & (halos['halo_upid'] == -1)
    milky_way_halos = halos[mask]

.. _accessing_snapshot_metadata: 

Accessing the snapshot metadata
=================================

All metadata associated with a Halotools-formatted halo catalog is
accessible via attributes of the `~halotools.sim_manager.CachedHaloCatalog` object.

.. code:: python

    print(halocat.redshift, halocat.Lbox)

.. parsed-literal::

    (0.4966, 250.0)


The ``Lbox`` attribute can be useful in performing calculations, for
example in accounting for the periodic boundary conditions of the
simulation. There are also many attributes dedicated to rigorously
keeping track of how a halo catalog was processed.

For example, during the initial processing of the halo catalog, cuts may
have been placed on certain columns of the halo catalog. If you
processed your halo catalog using the
`halotools.sim_manager.RockstarHlistReader`, every cut you used to
reduce the halo catalog will have a corresponding attribute reminding
you of the choice you made during the data reduction. In the
Halotools-provided snapshots, any (sub)halo that never had more than 300
particles at any point in its assembly history was discarded. The
``halo_mpeak`` column of the halo table stores the largest value of the
virial mass ever attained by the halo throughout its assembly history,
and so this 300-particle cut is reflected by the
``halo_mpeak_row_cut_min`` attribute of the halo catalog:

.. code:: python

    print("Minimum halo_mpeak = %.2e" % halocat.halo_mpeak_row_cut_min)

.. parsed-literal::

    Minimum halo_mpeak = 4.05e+10


As simple bookkeeping errors are so common in simulation analysis, you
may find Halotools useful to help avoid buggy results even if the
`~halotools.sim_manager.CachedHaloCatalog` is the only feature of the package that you use.

