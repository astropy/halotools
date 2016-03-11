.. _subhalo_modeling_tutorial1:

****************************************************************
Example 1: Building a simple subhalo-based model
****************************************************************

.. currentmodule:: halotools.empirical_models

This section of the documentation describes how you can use the 
`SubhaloModelFactory` to build one of the simplest and most widely 
used galaxy-halo models in the literature: the stellar-to-halo 
mass relation based on 
`Behroozi et al 2010 <http://arxiv.org/abs/1001.0015>`_. 
By following this example, 
you will learn the basic calling sequence for building subhalo-based models. 
We will build an exact replica of the ``behroozi10`` pre-built model, 
so you will also learn how the `PrebuiltSubhaloModelFactory` is really just 
"syntax candy" for the `SubhaloModelFactory`. 

There is also an IPython Notebook in the following location that can be 
used as a companion to the material in this section of the tutorial:


    **halotools/docs/notebooks/subhalo_modeling/subhalo_modeling_tutorial1.ipynb**

By following this tutorial together with this notebook, 
you can play around with your own variations of the models we'll build 
as you learn the basic syntax. 
The notebook also covers supplementary material that you may find clarifying, 
so we recommend that you read the notebook side by side with this documentation. 

Preliminary comments
=====================================================

As sketched in :ref:`factory_design_diagram`, 
to build any subhalo-based composite model you pass a 
set of component model instances to the `SubhaloModelFactory`. 
These instances are all passed in the form of keyword arguments, 
where the keyword you choose will be used as a nickname for the feature. 
While you can in principle use any keywords you like, try to use 
keywords that are expressive for the feature you are modeling. 
For example, a component controlling stellar mass :math:`M_{\ast}` should be 
attached to a keyword such as ``stellar_mass``, 
while a component controlling :math:`{\rm d}M_{\ast}/{\rm d}t` should 
be attached to a keyword such as ``sfr``. 

For the sake of concreteness, let's now dive in and 
write down the source code for how to call the `SubhaloModelFactory`. 

.. _subhalo_factory_tutorial_behroozi10_source_code:

Source code for the ``behroozi10`` model
----------------------------------------

>>> from halotools.empirical_models import SubhaloModelFactory
>>> from halotools.empirical_models import Behroozi10SmHm
>>> sm_model =  Behroozi10SmHm(redshift = 0)
>>> model_instance = SubhaloModelFactory(stellar_mass = sm_model)

To use the model instance populate a fake simulation that is generated on-the-fly:

>>> from halotools.sim_manager import FakeSim
>>> halocat = FakeSim()
>>> model_instance.populate_mock(halocat)

Setting simname to 'fake' populates a mock into a fake halo catalog 
that is generated on-the-fly, but you can use the 
`SubhaloModelFactory.populate_mock` method with 
any Halotools-formatted catalog. 

Now that you have called the `SubhaloModelFactory.populate_mock` method, 
your ``model_instance`` has a ``mock`` attribute containing a
``galaxy_table`` where your synthetic galaxy population is stored in the
form of an Astropy `~astropy.table.Table` object:

.. code:: python

    print(model_instance.mock.galaxy_table[0:5])

.. parsed-literal::

    halo_upid halo_mpeak  halo_x  halo_y ...   vy      vz   galid stellar_mass
    --------- ---------- ------- ------- ... ------ ------- ----- ------------
           -1  4.443e+10 21.4241 12.9027 ... 330.42    80.6     0   6.8085e+07
           -1  9.159e+10 21.2689 12.9744 ... 399.15   73.33     1   7.6912e+08
           -1  9.909e+10 19.6521 14.0854 ... 216.81  -315.0     2   4.0504e+08
           -1  7.469e+10 20.4365 14.4506 ... 285.96 -263.34     3  1.35992e+08
           -1  6.024e+10 20.3154 14.4435 ... 429.55 -326.15     4  1.09347e+08

We will not cover any of the details about the component model 
features collected together by the ``behroozi10`` composite model. 
You can read about that in the :ref:`behroozi10_composite_model` 
section of the documentation, and also by reading the docstring 
of the `Behroozi10SmHm` class. 

As we will see later in this tutorial, the above syntax applies to *all*
Halotools composite models, no matter what their features are. Once you
have built a composite model with one of the factories, you can always
use the model to populate *any* Halotools-formatted halo catalog with
the same syntax. As you change the features of the composite model, this
simply changes what columns will be created for the ``galaxy_table``
storing the mock.

Unpacking the ``behroozi10`` source code
=====================================================

Notice how the `SubhaloModelFactory` was instantiated 
with a single keyword argument, ``stellar_mass``. 
This simple model has only a single feature, :math:`M_{\ast}`, 
and so we only pass a single keyword argument, and the resulting 
mock galaxy population stored in the ``galaxy_table`` will only 
have one attribute column for the physical properties of the galaxies, ``stellar_mass``. 

It is important to realize that the name *stellar_mass* that appears as a column 
in the ``galaxy_table`` is *not* determined by the 
string chosen for the keyword argument passed to the `SubhaloModelFactory`. 
In all cases, galaxy property names are determined by the 
component model instances. 
In this case, the name ``stellar_mass`` was set in the source code 
for the `Behroozi10SmHm` class via the ``_galprop_dtypes_to_allocate`` mechanism. 
We will cover this mechanism in detail later on in this tutorial. For now, 
we'll just provide an explicit demonstration that the keyword passed to the 
factory is merely a nickname used as an internal bookkeeping device, and does not 
influence the properties of your mock galaxy population. The only change 
in the source code below from the above is that we will use ``mstar`` as the 
keyword attached to the `Behroozi10SmHm` controlling :math:`M_{\ast}`. 

.. code:: python

    sm_model =  Behroozi10SmHm(redshift = 0)
    model_instance = SubhaloModelFactory(mstar = sm_model)
    from halotools.sim_manager import CachedHaloCatalog
    halocat = CachedHaloCatalog(simname = 'bolshoi')
    model_instance.populate_mock(halocat)
    print(model_instance.mock.galaxy_table[0:5])

.. parsed-literal::

    halo_upid   halo_mpeak      halo_x    ...       vz       galid stellar_mass
    --------- ------------- ------------- ... -------------- ----- ------------
           -1 10000000000.0 238.356419525 ...  128.746372256     0  2.20813e+06
           -1 10000000000.0 114.297583082 ... -457.454694597     1  4.18238e+06
           -1 10000000000.0 169.758190609 ...   332.83620235     2  1.47292e+06
           -1 10000000000.0 154.016143562 ...  204.050095347     3  1.20527e+06
       101526 10000000000.0 197.266328297 ... -6.90451725995     4  2.32464e+06

As you can see, the ``galaxy_table`` has a *stellar_mass* column, 
just as it did before. 


Concluding comments 
======================
In subhalo-based models, there is a one-to-one correspondence between 
galaxies and (sub)halos. For numerous reasons, 
this makes the implementation of the `SubhaloModelFactory`  
quite different from the `HodModelFactory` 
(most notably in how memory for the galaxy table is allocated). 
If you have previously followed the  
:ref:`hod_modeling_tutorial0`, you will notice that here there is no need 
to format the input keyword arguments in the special way that is necessary 
for HOD-style models. This is because for subhalo-based models, it is not 
required that you provide separate models for satellites and centrals, 
and so here there is no need to use the keywords to explicitly declare 
the type of galaxy to which the component model applies. 
If you have not already read the :ref:`hod_modeling_tutorial0`, 
you can safely ignore this comment and continue following this tutorial, 
which continues with :ref:`subhalo_modeling_tutorial2`. 

This tutorial continues with :ref:`subhalo_modeling_tutorial2`. 



