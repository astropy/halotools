:orphan:

.. _reducing_and_caching_a_new_rockstar_catalog:

**************************************************************
Instructions for Reducing and Caching a Rockstar Catalog 
**************************************************************

This section of the documentation describes how to reduce 
a Rockstar hlist ASCII file into an hdf5 file stored in Halotools cache. 
If you just want to read an hlist file without caching the data, 
you should instead use the `~halotools.sim_manager.TabularAsciiReader` class. 

You are responsible for acquiring or generating your own catalog of Rockstar halos. 
The :ref:`supported_sim_list` section of the documentation provides links 
to the web locations of the original ASCII data upon which the Halotools-provided 
catalogs are based. Additional catalogs can also be found at 
`The CosmoSim database <https://www.cosmosim.org/>`_. 

Before reducing and caching your catalog with 
please carefully read *both* this tutorial *and* the entire docstring of 
the `~halotools.sim_manager.RockstarHlistReader` class. 


Initializing the reader
===========================

To instantiate the `~halotools.sim_manager.RockstarHlistReader` class, 
in addition to the path to the ASCII you must provide the following information: 

1. the columns of data you want

2. metadata used to keep track of the simulation in cache

3. an output filename 

We will comment on each of these three inputs in turn. 
For a description of how to additionally make on-the-fly row-cuts, 
see the section on :ref:`making_on_the_fly_row_cuts` below. 

Specifying the columns you want with the *columns_to_keep_dict*. 
---------------------------------------------------------------------

In order to use the `~halotools.sim_manager.RockstarHlistReader` class, 
you must manually inspect the hlist file to determine what information you want, 
and in what column the information is stored. This information can be determined 
by inspecting the header of the ASCII file. See the docstring of the 
`~halotools.sim_manager.RockstarHlistReader` class for instructions on exactly 
how this dictionary is formatted. 

This step of the reduction cannot be robustly automated because there is no 
universal standard form for Rockstar headers. Even if the header became standard, 
existing hlist files that are currently publicly available and 
in wide use would not conform to the new standard. With your labor in this step, 
you are providing the necessary standardization. 


Specifing the simulation metadata
---------------------------------------------

There are six required pieces of metadata that you must specify: 
the ``simname``, ``halo_finder``, ``version_name``, ``redshift``, 
``Lbox`` and ``particle_mass``. 
(In most cases, specifying the ``halo_finder`` is redundant, 
though the `~halotools.sim_manager.RockstarHlistReader` class 
can also be used to reduce and cache any halo catalog that is formatted in the 
same way as a typical hlist file. In fact, that is how the Bolshoi-BDM catalogs 
provided by Halotools were generated). 

The first four of these pieces of metadata govern how the cache will 
be used to keep track of this catalog. After caching the halos with the 
`~halotools.sim_manager.RockstarHlistReader.read_halocat` method, 
you can load the cached catalog into memory as follows:

>>> from halotools.sim_manager import CachedHaloCatalog 
>>> halocat = CachedHaloCatalog(simname = simname, halo_finder = halo_finder, version_name = version_name, redshift = redshift) # doctest: +SKIP

Each time you process a new halo catalog, we recommend that you choose a different ``version_name``, 
*especially* if you make different cuts. 
If you use one of the same simulations as those provided by Halotools, 
it is recommended that you follow the ``simname`` conventions laid out on the 
:ref:`supported_sim_list` page. 
Although not strictly necessary, make an effort to specify the redshift accurately to four decimals 
as this is the string format used to store the ``redshift`` metadata. 

``Lbox`` should be given in Mpc/h and ``particle_mass`` in Msun/h. 

Although optional, it is strongly recommended that you also set the 
``processing_notes`` argument to be some string giving a plain-language description of 
the row-cuts you placed on the catalog (see :ref:`making_on_the_fly_row_cuts` below). 

Choosing your *output_fname* 
-----------------------------------

By setting the ``output_fname`` to be the absolute path to an hdf5 file, 
you are free to store the halos in any location on disk that you like. 
By setting ``output_fname`` to the string ``std_cache_loc``, 
Halotools will place the reduced catalog in the following location on disk:

$HOME/.astropy/cache/halotools/halo_catalogs/simname/halo_finder/input_fname.version_name.hdf5

Wherever you store the hdf5 file of halos, 
you should try to choose a reasonably permanent location to keep them. 
Moving halo catalogs around on disk is a common way to 
introduce buggy behavior into simulation analysis. 
If you decide you want to change the disk location of the hdf5 file you produced 
after storing it in cache, you will need to update the cache log with the new location. 
In that event, see the :ref:`relocating_simulation_data` section of the documentation. 


.. _making_on_the_fly_row_cuts: 

Making on-the-fly row-cuts (optional)
---------------------------------------

Halo catalogs typically occupy many Gb of disk space. Because of the 
shape of the CDM mass function, most halos in any catalog are right at (or beyond) 
the resolution limits of the simulation. Thus for many science targets 
most of the halos in the catalog are irrelevant and so you should not waste 
disk space storing them. For example, the Halotools-provided catalogs 
only include halos and subhalos with a few hundred particles (as described 
in the ``processing_notes`` metadata bound to these catalogs). 

The `~halotools.sim_manager.RockstarHlistReader` class allows you to 
apply cuts on the rows of ASCII data as the file is being read, so that only 
halos passing your desired cuts will be stored in the cached catalog. 
This not only saves disk space, but because the cuts are applied on-the-fly, 
this also allows you to reduce a halo catalog that is too large to fit into RAM. 
With the `~halotools.sim_manager.RockstarHlistReader` class, only the final, reduced 
catalog need fit into memory. 

By default, no row-cuts are made, but the following four optional keyword arguments 
allow you to construct a highly customizable on-the-fly cut on the ASCII rows:

*row_cut_min_dict, row_cut_max_dict, row_cut_eq_dict* and *row_cut_neq_dict*. 

See the notes in the `~halotools.sim_manager.RockstarHlistReader` docstring  
for how to construct a cut of your liking with these arguments. 

Running the reader
======================

Once you have instantiated the `~halotools.sim_manager.RockstarHlistReader` class, 
you can read the ASCII data by calling the 
`~halotools.sim_manager.RockstarHlistReader.read_halocat` method. 
As described in the the `~halotools.sim_manager.RockstarHlistReader.read_halocat` docstring, 
this method does not return anything but instead binds the halo catalog to the 
``halo_table`` attribute of the reader instance. If you call the 
`~halotools.sim_manager.RockstarHlistReader.read_halocat` method with no arguments, 
that is all that will happen: by default, Halotools will not write large amounts of 
data to your disk. 
However, in the majority of use-cases you should set both of these arguments to True, 
in which case your reduced catalog will be saved on disk and stored in cache. 


The end result 
================

After calling the `~halotools.sim_manager.RockstarHlistReader.read_halocat` method, 
your catalog is now stored in cache and you can load it into memory using 
the `~halotools.sim_manager.CachedHaloCatalog` class as follows:

>>> from halotools.sim_manager import CachedHaloCatalog 
>>> halocat = CachedHaloCatalog(simname = simname, halo_finder = halo_finder, version_name = version_name, redshift = redshift) # doctest: +SKIP

When you load an instance of the `~halotools.sim_manager.CachedHaloCatalog` class, 
the metadata of the hdf5 file you created is inspected and all its metadata gets bound to the 
`~halotools.sim_manager.CachedHaloCatalog` as convenience-attributes. For example, 
you can remind yourself of the cuts you placed on the catalog:

>>> print(halocat.processing_notes) # doctest: +SKIP

The `~halotools.sim_manager.RockstarHlistReader` automatically creates 
some additional metadata to help with your bookkeeping. For example:

>>> print(halocat.orig_ascii_fname) # doctest: +SKIP
>>> print(halocat.time_of_catalog_production) # doctest: +SKIP

See the docstring of the `~halotools.sim_manager.CachedHaloCatalog` class for more information. 



