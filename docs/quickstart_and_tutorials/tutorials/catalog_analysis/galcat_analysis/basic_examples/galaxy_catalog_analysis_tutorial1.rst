
.. _galaxy_catalog_analysis_tutorial1:

Galaxy Catalog Analysis Example 1: Galaxy properties as a function of halo mass
===================================================================================

In this example, we'll show how to start from a sample of mock galaxies 
and calculate how various galaxies properties scale with halo mass. 
In particular, we'll calculate the average total stellar mass, 
:math:`\langle M_{\ast}^{\rm tot}\vert M_{\rm halo}\rangle`, and also the average quiescent fraction 
for centrals and satellites, :math:`\langle F_{\rm cen}^{\rm q}\vert M_{\rm halo}\rangle` 
and :math:`\langle F_{\rm sat}^{\rm q}\vert M_{\rm halo}\rangle`. 

There is also an IPython Notebook in the following location that can be 
used as a companion to the material in this section of the tutorial:


    halotools/docs/notebooks/galcat_analysis/galaxy_catalog_analysis_tutorial1.ipynb

By following this tutorial together with this notebook, 
you can play around with your own variations of the calculation 
as you learn the basic syntax. 

Generate a mock galaxy catalog 
---------------------------------
Let's start out by generating a mock galaxy catalog into an N-body
simulation in the usual way. Here we'll assume you have the *z=0*
rockstar halos for the bolshoi simulation, as this is the
default halo catalog. 

.. code:: python

    from halotools.empirical_models import PrebuiltSubhaloModelFactory
    model = PrebuiltSubhaloModelFactory('smhm_binary_sfr')
    model.populate_mock(simname = 'bolshoi', redshift = 0, halo_finder = 'rockstar')

Now suppose the data we are interested in is complete for
:math:`M_{\ast} > 10^{10}M_{\odot},` so we will make a cut on the mock.
Our mock galaxies are stored in the ``galaxy_table`` of ``model.mock``
in the form of an Astropy `~astropy.table.Table`.

.. code:: python

    sample_mask = model.mock.galaxy_table['stellar_mass'] > 1e10
    gals = model.mock.galaxy_table[sample_mask]

Calculate total stellar mass :math:`M_{\ast}^{\rm tot}` in each halo
------------------------------------------------------------------------------

To calculate the total stellar mass of galaxies in each halo, we'll use
the `halotools.utils.add_new_table_column` function. You can read more about the 
details of that function in its documentation, here we'll just demo some basic usage. 

The ``halo_id`` is a natural grouping key for a galaxy catalog whose
host halos are known. Let's use this grouping key to calculate the total
stellar mass of galaxies in each halo, :math:`M_{\ast}^{\rm tot},` and
broadcast the result to the members of the halo.

.. code:: python

    from halotools.utils import add_new_table_column

    grouping_key = 'halo_id'
    new_colname, new_coltype = 'halo_total_stellar_mass', 'f4'
    
    # The aggregation function operates on the members of each halo, 
    # in this case returning the sum of whatever column it is passed
    aggregation_function = np.sum
    colnames_needed_by_function = ['stellar_mass'] 
    
    add_new_table_column(gals, 
            new_colname, new_coltype, grouping_key, 
            aggregation_function, colnames_needed_by_function)

Our ``gals`` table now has a ``halo_total_stellar_mass`` column.

Calculate host halo mass :math:`M_{\rm host}` of each galaxy
------------------------------------------------------------

Now we'll do a very similar calculation, but instead broadcasting the
host halo mass to each halo's members

.. code:: python

    new_colname, new_coltype = 'halo_mhost', 'f4'
    grouping_key = 'halo_id'
    
    sorting_keys = ['halo_id', 'halo_upid']
    # upid = -1 for the the host halo, 
    # so this choice for ``sorting_keys`` will place 
    # host halos in the first element of each grouped array
    
    # Define the function that returns whatever value 
    # is stored in the first group member
    def return_first_element_in_sequence(x):
        return x[0]
    aggregation_function = return_first_element_in_sequence 
    colnames_needed_by_function = ['halo_mvir'] 
    
    add_new_table_column(gals, 
            new_colname, new_coltype, grouping_key, 
            aggregation_function, colnames_needed_by_function, 
            sorting_keys=sorting_keys)

Our ``gals`` table now has a ``halo_mhost`` column.

Calculate :math:`\langle M_{\ast}^{\rm tot}\rangle` vs. :math:`M_{\rm halo}`
-------------------------------------------------------------------------------------------------
    
.. code:: python

    from halotools.mock_observables import mean_y_vs_x
    import numpy as np 
    
    bins = np.logspace(12, 15, 25)
    result = mean_y_vs_x(gals['halo_mhost'].data, 
                         gals['halo_total_stellar_mass'].data, 
                         bins = bins, 
                         error_estimator = 'variance') 
    
    host_mass, mean_stellar_mass, mean_stellar_mass_err = result

Plot the result 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from seaborn import plt
    
    plt.errorbar(host_mass, mean_stellar_mass, yerr=mean_stellar_mass_err, 
                 fmt = "none", ecolor='gray')
    plt.plot(host_mass, mean_stellar_mass, 'D', color='k')

    plt.loglog()
    plt.xticks(size=18)
    plt.yticks(size=18)
    plt.xlabel(r'$M_{\rm halo}/M_{\odot}$', fontsize=25)
    plt.ylabel(r'$\langle M_{\ast}^{\rm tot}/M_{\odot}\rangle$', fontsize=25)
    plt.ylim(ymax=2e12)

.. image:: output_18_1.png


Quiescent fraction of centrals and satellites
----------------------------------------------

In this section we'll perform a very similar calculation to the above, only here we'll compute the average quiescent fraction of centrals and satellites. 

Calculate :math:`\langle F_{\rm q}^{\rm cen}\vert M_{\rm halo} \rangle` and :math:`\langle F_{\rm q}^{\rm sat} \vert M_{\rm halo}\rangle`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the above calculation, we needed to create new columns for our galaxy catalog, :math:`M_{\rm host}` and :math:`M_{\ast}^{\rm tot}`. Here we'll reuse the :math:`M_{\rm host}` column, and our model already created a boolean-valued ``quiescent`` column for our galaxies. So all we need to do is calculate the average trends as a function of halo mass. 

.. code:: python

    cens_mask = gals['halo_upid'] == -1
    cens = gals[cens_mask]
    sats = gals[~cens_mask]
    
    bins = np.logspace(12, 14.5, 15)
    
    # centrals 
    result = mean_y_vs_x(cens['halo_mhost'].data, cens['quiescent'].data, 
                bins = bins)
    host_mass, fq_cens, fq_cens_err_on_mean = result 
    
    # satellites 
    result = mean_y_vs_x(sats['halo_mhost'].data, sats['quiescent'].data, 
                bins = bins)
    host_mass, fq_sats, fq_sats_err_on_mean = result 

Plot the result and compare it to the underlying analytical relation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    plt.errorbar(host_mass, fq_cens, yerr=fq_cens_err_on_mean, 
                 color='seagreen', fmt = "none")
    plt.plot(host_mass, fq_cens, 'D', color='seagreen', 
                 label = 'galaxy population')

    analytic_result_mhost_bins = np.logspace(10, 15.5, 100)
    analytic_result_mean_quiescent_fraction = model.mean_quiescent_fraction(prim_haloprop = analytic_result_mhost_bins)
    plt.plot(analytic_result_mhost_bins,
             analytic_result_mean_quiescent_fraction, 
             color='blue', label = 'analytical model')
    
    plt.xscale('log')
    plt.xticks(size=22)
    plt.yticks(size=18)
    plt.xlabel(r'$M_{\rm halo}/M_{\odot}$', fontsize=25)
    plt.ylabel('quiescent fraction', fontsize=20)
    plt.xlim(xmin = 1e12, xmax = 1e15)
    plt.ylim(ymin = 0.2, ymax=0.8)
    plt.legend(frameon=False, loc='best', fontsize=20)
    plt.title('Central galaxy quenching: model vs. mock', fontsize=17)


.. image:: output_23_1.png

This tutorial continues with :ref:`galaxy_catalog_analysis_tutorial2`. 
