
.. _using_halocat_binaries:

*************************************
Managing pre-processed halo catalogs
*************************************

.. code:: python

    from halotools import sim_manager

.. code:: python

    default_snapshot = sim_manager.ProcessedSnapshot()


.. parsed-literal::

    Loading z = -0.00 halo catalog with the following absolute path: 
    /Users/aphearin/.astropy/cache/halotools/halo_catalogs/bolshoi/rockstar/hlist_1.00030.list.halotools.official.version.hdf5
    


The ``~halotools.sim_manager.ProcessedSnapshot`` is the primary class
you will use when working with halo catalogs. When you instantiate this
class, as in the second line of code above, Halotools searches for the
relevant halo catalog and attaches the the halos to the snapshot object
in the form of the ``halos`` attribute:

.. code:: python

    print(default_snapshot.halos[0:4])


.. parsed-literal::

    scale    haloid   scale_desc ... mvir_firstacc vmax_firstacc vmax_mpeak
    ------ ---------- ---------- ... ------------- ------------- ----------
    1.0003 3060299107        0.0 ...     1.643e+14        952.39     952.39
    1.0003 3060312953        0.0 ...     1.589e+14        823.11     823.11
    1.0003 3058440575        0.0 ...     1.144e+14        799.42     799.42
    1.0003 3058441456        0.0 ...     9.709e+13        679.37     679.37


The data structure behind the scenes of the ``halos`` attribute of
``default_snapshot`` is an Astropy Table. We'll give a few simple
examples illustrating how to manipulate Astropy Tables below, but for
more detailed information about this data structure, see
``astropy.table``.

The ``default_snapshot`` also has the halo catalog and simulation
metadata bound to it. Here are a few examples:

.. code:: python

    print("Simulation name = %s " % default_snapshot.simname)
    print("Halo-finder = %s " % default_snapshot.halo_finder)
    print("Snapshot redshift = %.1f " % default_snapshot.redshift)
    print("Simulation spatial resolution = %.1f kpc/h" % default_snapshot.softening_length)


.. parsed-literal::

    Simulation name = bolshoi 
    Halo-finder = rockstar 
    Snapshot redshift = -0.0 
    Simulation spatial resolution = 1.0 kpc/h


Downloading other pre-processed snapshots
=========================================

Up until now, you have been working with the default snapshot downloaded
by the startup script ``download_initial_halocat``.

