.. _galaxy_catalog_analysis_tutorial5:

Basic Galaxy Catalog Analysis Example 5: Galaxy group identification
====================================================================

In this example, we'll show how to start from a sample of mock galaxies
and identify galaxy groups in an observationally realistic manner.

Generate a mock galaxy catalog
------------------------------

Let's start out by generating a mock galaxy catalog into an N-body
simulation in the usual way. Here we'll assume you have the z=0 rockstar
halos for the bolshoi simulation, as this is the default halo catalog.

.. code:: python

    from halotools.empirical_models import PrebuiltSubhaloModelFactory
    model = PrebuiltSubhaloModelFactory('behroozi10')
    model.populate_mock(simname = 'bolshoi', redshift = 0, halo_finder = 'rockstar')

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
`~halotools.utils.add_new_table_column` function in order to group our 
galaxy table together according to the FoF-determined group, and then 
calculate a few quantities based on group membership. You can read about 
the `~halotools.utils.add_new_table_column` function in its docstring, 
and see usage similar to what follows in :ref:`galaxy_catalog_analysis_tutorial1`. 

.. code:: python

    from halotools.utils import add_new_table_column

    galaxy_sample['negative_stellar_mass'] = -1*galaxy_sample['stellar_mass']

    grouping_key = 'fof_group_id'
    new_colname, new_coltype = 'group_central', bool
    
    sorting_keys = ['fof_group_id', 'negative_stellar_mass']
    # In sorting by -M*, within each fof group the most 
    # massive galaxy will appear first. The most massive 
    # galaxy in a group is typically defined as the group central
    
    # Define the function that assigns the first 
    # element of each group to be True, and all remaining 
    # elements to be False
    def assign_first_group_member_true(x):
        result = [False for elt in x]
        result[0] = True
        return result
    aggregation_function = assign_first_group_member_true 
    colnames_needed_by_function = ['fof_group_id'] # the value is never used, so any column will do
    
    add_new_table_column(galaxy_sample, 
            new_colname, new_coltype, grouping_key, 
            aggregation_function, colnames_needed_by_function, 
            sorting_keys = sorting_keys)
    
    # we can now dispense with the negative_stellar_mass column
    del galaxy_sample['negative_stellar_mass']

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

.. code:: python

    grouping_key = 'fof_group_id'
    new_colname, new_coltype = 'group_richness', 'i4'
    
    sorting_keys = ['fof_group_id']
    
    def richness(x): return len(x)
    aggregation_function = richness 
    colnames_needed_by_function = ['fof_group_id'] # the value is never used, so any column will do
    
    add_new_table_column(galaxy_sample, 
            new_colname, new_coltype, grouping_key, 
            aggregation_function, colnames_needed_by_function, 
            table_is_already_sorted = True)

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

    grouping_key = 'fof_group_id'
    new_colname, new_coltype = 'group_central_true_mvir', 'f4'
    
    sorting_keys = ['fof_group_id']
    
    # Define the function that returns whatever value 
    # is stored in the first group member
    def return_first_element_in_sequence(x):
        return x[0]
    aggregation_function = return_first_element_in_sequence 
    colnames_needed_by_function = ['halo_mvir_host_halo'] # the value is never used, so any column will do
    
    add_new_table_column(galaxy_sample, 
            new_colname, new_coltype, grouping_key, 
            aggregation_function, colnames_needed_by_function, 
            table_is_already_sorted = True)

Calculate :math:`\langle N_{\rm group}\rangle` as a function of :math:`M_{\rm cen}^{\rm true}`
----------------------------------------------------------------------------------------------

For this calculation, we'll use `~halotools.mock_observables.mean_y_vs_x` to 
compute the mean group richness as a function of true central halo mass. 
Note that we only loop over group centrals, otherwise we would incorrectly fold each 
group's satellites into the Poisson error estimate. 

.. code:: python

    from halotools.mock_observables import mean_y_vs_x

    group_cenmask = galaxy_sample['group_central'] == True
    group_cens = galaxy_sample[group_cenmask]
    
    log10_mvir_array, avg_richness, err_richness = mean_y_vs_x(np.log10(group_cens['group_central_true_mvir']), 
                                                         group_cens['group_richness'], 
                                                         error_estimator = 'error_on_mean')

Plot the result
~~~~~~~~~~~~~~~

.. code:: python

    from seaborn import plt

    plt.errorbar(10**log10_mvir_array, avg_richness, yerr=err_richness, 
                 color='red', fmt = "none")
    plt.plot(10**log10_mvir_array, avg_richness, 'D', color='seagreen')
    
    plt.xscale('log')
    plt.xticks(size=22)
    plt.yticks(size=18)
    plt.xlabel(r'$M_{\rm cen}^{\rm true}$  $[M_{\odot}]$', fontsize=25)
    plt.ylabel(r'$\langle N_{\rm group}\rangle$', fontsize=20)
    plt.xlim(xmin = 1e12, xmax = 1e15)


.. image:: group_richness_vs_group_cenmass.png

This tutorial continues with :ref:`galaxy_catalog_analysis_tutorial6`. 


