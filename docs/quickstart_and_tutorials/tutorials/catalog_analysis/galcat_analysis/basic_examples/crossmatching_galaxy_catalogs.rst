:orphan:

.. _crossmatching_galaxy_catalogs:

****************************************************
Cross-matching galaxy and halo catalogs
****************************************************

This tutorial demonstrates how to use the 
`~halotools.utils.crossmatch` function to create "value-added" versions of your 
galaxy catalog that are supplemented by additional properties 
only stored in the halo catalog. 

For a closely related tutorial, see :ref:`crossmatching_halo_catalogs`. 

Let's start out by generating a mock galaxy catalog from a (fake) halo catalog:

>>> from halotools.sim_manager import FakeSim
>>> halocat = FakeSim()
>>> from halotools.empirical_models import PrebuiltHodModelFactory
>>> model = PrebuiltHodModelFactory('leauthaud11')
>>> model.populate_mock(halocat)

The tabular data of mock galaxies is stored in ``model.mock.galaxy_table``, 
an Astropy `~astropy.table.Table`. All mock galaxy catalogs come with 
a ``halo_id`` that lets you cross-match the galaxies against the halos they live in 
using the `~halotools.utils.crossmatch` function. This function returns the indices 
providing the correspondence between the rows in the ``galaxy_table`` that have 
matches in the ``halo_table``. You can use this function to transfer any 
property stored in the halo catalog onto the galaxies in your mock. 
If you are working with a model you have built yourself, and you know in advance that 
you would like some particular halo property to always be a part of your ``galaxy_table``, 
then you can exploit :ref:`list_of_haloprops_needed_mechanism` when building your model. 
But you can always use the `~halotools.utils.crossmatch` function to perform this task 
in a post-processing phase, as shown here for the case of adding the ``halo_vmax`` 
column to the ``galaxy_table``. 

>>> import numpy as np
>>> from halotools.utils import crossmatch 
>>> idx_galaxies, idx_halos = crossmatch(model.mock.galaxy_table['halo_id'], halocat.halo_table['halo_id'])
>>> model.mock.galaxy_table['halo_vmax'] = np.zeros(len(model.mock.galaxy_table), dtype = halocat.halo_table['halo_vmax'].dtype)
>>> model.mock.galaxy_table['halo_vmax'][idx_galaxies] = halocat.halo_table['halo_vmax'][idx_halos]




