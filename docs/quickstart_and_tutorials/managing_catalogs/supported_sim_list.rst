:orphan:

.. _supported_sim_list:

*****************************************************
Simulations and Halo Catalogs Provided by Halotools
*****************************************************

In this section of the documentation we give a brief description of each of the simulations provided by the package. If you would like to use your own catalog, rather than one of the simulations listed below, see the :ref:`working_with_alternative_catalogs`.

Halotools is configured to download, process and manage a range of
simulations and halo catalogs.
For each supported simulation, there are pre-processed binaries available
for download for redshifts z = 0, 0.5, 1, and 2.
Halo catalogs based on the Rockstar halo-finder are available for all
of the simulations below; for Bolshoi, BDM-based catalogs are also available.
To see simple examples of how to manipulate the data stored in halo catalogs,
see the Examples section of the `~halotools.sim_manager.CachedHaloCatalog` documentation.
For more detailed information, refer to the :ref:`halo_catalog_analysis_quickstart` tutorial.

Below appears a summary table of the provided snapshots. See the sections below for further information

============  ===========================  ======================
simname          available halo-finders     available redshifts
============  ===========================  ======================
bolshoi       rockstar, bdm                0.0, 0.5, 0.84, 2.0
bolplanck     rockstar                     0.0, 0.5, 1.00, 2.0
multidark     rockstar                     0.0, 0.5, 1.00, 2.15
consuelo      rockstar                     0.0, 0.5, 1.00, 2.15
============  ===========================  ======================



Bolshoi (simname = `bolshoi`)
==================================
WMAP5 cosmology with Lbox = 250 Mpc/h and particle mass of ~1e8 Msun/h.

For a detailed description of the simulation specs, see
http://www.cosmosim.org/cms/simulations/multidark-project/bolshoi.

To download the halos upon which the Halotools-provided catalogs are based, see
`Source of Bolshoi Rockstar halos <http://www.slac.stanford.edu/~behroozi/Bolshoi_Catalogs/>`_
and `Source of Bolshoi BDM halos <http://www.slac.stanford.edu/~behroozi/Bolshoi_Catalogs_BDM/>`_.

Note that the above web source of halo data is frequently updated and is not maintained
by Halotools, and so it is not possible to guarantee that a halo catalog you download
from this location will be the same as the one used to produce the Halotools-provided catalogs.

Bolshoi-Planck (simname = `bolplanck`)
====================================================================
Planck 2013 cosmology with Lbox = 250 Mpc/h and particle mass of ~1e8 Msun/h.

For a detailed description of the simulation specs, see
http://www.cosmosim.org/cms/simulations/bolshoip-project/bolshoip/.

To download the halos upon which the Halotools-provided catalogs are based, see
`Source of Bolshoi Planck Rockstar halos <http://www.slac.stanford.edu/~behroozi/BPlanck_Hlists/>`_.

Note that the above web source of halo data is frequently updated and is not maintained
by Halotools, and so it is not possible to guarantee that a halo catalog you download
from this location will be the same as the one used to produce the Halotools-provided catalogs.

Multidark (simname = `multidark`)
====================================================================
WMAP5 cosmology with Lbox = 1Gpc/h and particle mass of ~1e10 Msun/h.

For a detailed description of the simulation specs, see
http://www.cosmosim.org/cms/simulations/multidark-project/mdr1.

To download the halos upon which the Halotools-provided catalogs are based, see
`Source of Multidark Rockstar halos <http://slac.stanford.edu/~behroozi/MultiDark_Hlists_Rockstar/>`_.

Note that the above web source of halo data is frequently updated and is not maintained
by Halotools, and so it is not possible to guarantee that a halo catalog you download
from this location will be the same as the one used to produce the Halotools-provided catalogs.

Consuelo (simname = `consuelo`)
====================================================================
WMAP5-like cosmology with Lbox = 420 Mpc/h and particle mass of ~1e9 Msun/h.

For a detailed description of the simulation specs, see
http://lss.phy.vanderbilt.edu/lasdamas/simulations.html.

To download the halos upon which the Halotools-provided catalogs are based, see
`Source of Consuelo Rockstar halos <http://www.slac.stanford.edu/~behroozi/Consuelo_Catalogs/>`_.

Note that the above web source of halo data is frequently updated and is not maintained
by Halotools, and so it is not possible to guarantee that a halo catalog you download
from this location will be the same as the one used to produce the Halotools-provided catalogs.



