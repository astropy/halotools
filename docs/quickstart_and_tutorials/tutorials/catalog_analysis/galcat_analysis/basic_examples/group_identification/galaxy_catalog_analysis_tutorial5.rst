.. _galaxy_catalog_analysis_tutorial5:

Galaxy Catalog Analysis Example: Galaxy group identification
====================================================================

In this example, we'll show how to start from a sample of mock galaxies
and identify galaxy groups in an observationally realistic manner.

There is also an IPython Notebook in the following location that can be 
used as a companion to the material in this section of the tutorial:


    **halotools/docs/notebooks/galcat_analysis/basic_examples/galaxy_catalog_analysis_tutorial5.ipynb**

By following this tutorial together with this notebook, 
you can play around with your own variations of the calculation 
as you learn the basic syntax. 

Generate a mock galaxy catalog
------------------------------

Let's start out by generating a mock galaxy catalog into an N-body
simulation in the usual way. Here we'll assume you have the z=0 rockstar
halos for the bolshoi simulation, as this is the default halo catalog.

.. code:: python

    from halotools.empirical_models import PrebuiltSubhaloModelFactory
    model = PrebuiltSubhaloModelFactory('behroozi10')
    from halotools.sim_manager import CachedHaloCatalog
    halocat = CachedHaloCatalog(simname = 'bolshoi', redshift = 0, halo_finder = 'rockstar')
    model.populate_mock(halocat)

Our mock galaxies are stored in the ``galaxy_table`` of ``model.mock``
in the form of an Astropy Table.

Extract subsamples of galaxy positions
--------------------------------------
Galaxy group identification is conducted by the
`~halotools.mock_observables.FoFGroups` class. 
As described in :ref:`mock_obs_pos_formatting`, 
functions in the `~halotools.mock_observables` package 
such `~halotools.mock_observables.FoFGroups` take array inputs in a 
specific form: a (*Npts, 3)*-shape Numpy array. You can use the 
`~halotools.mock_observables.return_xyz_formatted_array` convenience 
function for this purpose, which we will do after first 
throwing out galaxies below a stellar mass completeness cut 
of :math:`M_{\ast} > 10^{10.75}M_{\odot}` for illustration purposes. 

.. code:: python

    sample_mask = model.mock.galaxy_table['stellar_mass'] > 10**10.75
    galaxy_sample = model.mock.galaxy_table[sample_mask]
    x = galaxy_sample['x']
    y = galaxy_sample['y']
    z = galaxy_sample['z']
    vz = galaxy_sample['vz']

    from halotools.mock_observables import return_xyz_formatted_array
    
    pos = return_xyz_formatted_array(x, y, z, 
                velocity=vz, velocity_distortion_dimension = 'z')

Note that in the above code we have also applied redshift-space 
distortions to the positions, as this has an important effect on 
galaxy group identification. 

Identify FoF galaxy groups
--------------------------
In order to identify FoF groups, it is necessary 
to choose a linking length in both the line-of-sight and 
transverse directions. These linking lengths are specified by 
:math:`b_{\rm para}` and :math:`b_{\rm perp}`, respectively. 
We will choose values based on 
`Berlind et al 2006 <http://arxiv.org/abs/astro-ph/0601346>`_, 
which were optimized to minimize bias in the group multiplicity 
function, but you can set these to any values you like. 

.. code:: python

    from halotools.mock_observables import FoFGroups

    b_para, b_perp = 0.7, 0.15 
    groups = FoFGroups(pos, b_perp, b_para, 
                          Lbox = model.mock.Lbox, num_threads='max')

    galaxy_sample['fof_group_id'] = groups.group_ids
    
The ``galaxy_sample`` storing the mock now has a column storing the
group IDs as they would have been found in real observational data,
including redshift-space distortion effects.

Determine group centrals and satellites
---------------------------------------
In the following calculation, we'll use the 
`~halotools.utils.group_member_generator` in order to group our 
galaxy table together according to the FoF-determined group, and then 
calculate a few quantities based on group membership. You can read about 
the `~halotools.utils.group_member_generator` in its docstring. 
You may also find it useful to review :ref:`galaxy_catalog_analysis_tutorial1` 
for usage similar to what follows, only with more exposition. 

Here we perform another two-column sort. First, the galaxies 
are sorted by their FoF group ID, and then within each grouping, 
they are sorted by :math:`-M_{\ast}`, which will place the most massive 
galaxy first within each FoF group. 

.. code:: python

    from halotools.utils import group_member_generator

    galaxy_sample['negative_stellar_mass'] = -1*galaxy_sample['stellar_mass']
    galaxy_sample.sort(['fof_group_id', 'negative_stellar_mass'])
    grouping_key = 'fof_group_id'
    requested_columns = []

    group_gen = group_member_generator(galaxy_sample, grouping_key, requested_columns)

    group_central = np.zeros(len(galaxy_sample), dtype=bool)
    for first, last, member_props in group_gen:
        temp_result = [False for member in xrange(first, last)]
        temp_result[0] = True
        group_central[first:last] = temp_result
        
    galaxy_sample['group_central'] = group_central

Let's inspect our results

.. code:: python

    print(galaxy_sample[0:15])

.. parsed-literal::

    halo_upid  halo_mpeak  halo_x ... stellar_mass fof_group_id group_central
    ---------- ---------- ------- ... ------------ ------------ -------------
            -1  2.549e+12 20.8524 ...  1.47289e+11            0          True
            -1  8.513e+11 20.6768 ...  8.08962e+10            0         False
            -1  1.237e+12 22.3349 ...  2.40317e+11            1          True
    3058440575  3.237e+12 21.9039 ...  1.56945e+11            1         False
            -1  1.144e+14  21.812 ...  1.41576e+11            1         False
    3058440575   1.39e+12 22.3782 ...  8.50839e+10            1         False
    3058440575  1.163e+12 21.7744 ...  8.11016e+10            1         False
    3058440575   2.56e+12 21.9585 ...  7.28933e+10            1         False
            -1  9.709e+13 26.1803 ...  4.35889e+11            2          True
            -1  1.869e+12 25.4072 ...   9.4756e+10            2         False
            -1  8.876e+11 25.4922 ...  9.16859e+10            2         False
            -1  1.373e+12  25.946 ...  9.06732e+10            2         False
    3058441456  2.926e+13 25.6703 ...  1.56814e+11            3          True
    3058441456  8.404e+11 26.3213 ...  8.19847e+10            4          True
            -1  4.076e+13 23.7934 ...  1.71193e+11            5          True


Calculating group richness :math:`N_{\rm group}`
------------------------------------------------
Now we'll use the same generator to calculate the total number of members in each FoF group. 

.. code:: python

    grouping_key = 'fof_group_id'
    requested_columns = []

    group_gen = group_member_generator(galaxy_sample, grouping_key, requested_columns)

    group_richness = np.zeros(len(galaxy_sample), dtype=int)
    for first, last, member_props in group_gen:
        group_richness[first:last] = last-first
    galaxy_sample['group_richness'] = group_richness

.. code:: python

    print(galaxy_sample[0:15])

.. parsed-literal::

    halo_upid  halo_mpeak  halo_x ... fof_group_id group_central group_richness
    ---------- ---------- ------- ... ------------ ------------- --------------
            -1  2.549e+12 20.8524 ...            0          True              2
            -1  8.513e+11 20.6768 ...            0         False              2
            -1  1.237e+12 22.3349 ...            1          True              6
    3058440575  3.237e+12 21.9039 ...            1         False              6
            -1  1.144e+14  21.812 ...            1         False              6
    3058440575   1.39e+12 22.3782 ...            1         False              6
    3058440575  1.163e+12 21.7744 ...            1         False              6
    3058440575   2.56e+12 21.9585 ...            1         False              6
            -1  9.709e+13 26.1803 ...            2          True              4
            -1  1.869e+12 25.4072 ...            2         False              4
            -1  8.876e+11 25.4922 ...            2         False              4
            -1  1.373e+12  25.946 ...            2         False              4
    3058441456  2.926e+13 25.6703 ...            3          True              1
    3058441456  8.404e+11 26.3213 ...            4          True              1
            -1  4.076e+13 23.7934 ...            5          True              1


Calculate true halo mass of group central :math:`M_{\rm cen}^{\rm true}`
------------------------------------------------------------------------

.. code:: python

    galaxy_sample.sort(['fof_group_id', 'negative_stellar_mass'])
    grouping_key = 'fof_group_id'
    requested_columns = ['halo_mvir_host_halo']
    group_gen = group_member_generator(galaxy_sample, grouping_key, requested_columns)

    group_central_true_mvir = np.zeros(len(galaxy_sample))
    for first, last, member_props in group_gen:
        member_masses = member_props[0]
        true_mass = member_masses[0]
        group_central_true_mvir[first:last] = true_mass

    galaxy_sample['group_central_true_mvir'] = group_central_true_mvir
    
