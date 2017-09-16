:orphan:

.. _zspace_distortion_derivation:

****************************************************
Derivation of z−space Distortions in Simulation Data
****************************************************

In this section of the documentation we derive a simple expression for how to apply z-space distortions to the usual comoving Cartesian coordinates of a simulation. Suppose you have a set of :math:`x_{\rm com}, y_{\rm com}, z_{\rm com}` points in co-moving coordinates in a cosmological simulation. Let the redshift of the snapshot be :math:`z_{\rm true-redshift}`, and the peculiar velocity in the z-direction be :math:`v_{\rm z}`.  Using the distant-observer approximation and treating the z-dimension as the line-of-sight, these notes derive the following expression for the z-coordinate in redshift-space:

.. math::

    z^{z-{\rm space}}_{\rm com} = z_{\rm com} + \frac{(1+z_{\rm true-redshift})}{H(z_{\rm true-redshift})}v_{\rm z}

The first part of the derivation is repeated from the textbook Mo, van den Bosch and White (2010), section 3.1.4 on peculiar velocities. For an observer in the cosmic rest frame at :math:`z_{\rm true-redshift},` a galaxy at :math:`z_{\rm true-redshift}` moving with peculiar velocity :math:`v_{\rm z}` will be Doppler shifted according to :math:`1 + z_{\rm redshift}^{\rm pec} = \sqrt{\frac{1 + v_{\rm z}/c}{1 - v_{\rm z}/c}}.` On Earth we are observing that galaxy from the vantage of z=0, then we have:

.. math::
    1 + z_{\rm redshift}^{\rm obs} = (1 + z_{\rm redshift}^{\rm pec})(1 + z_{\rm true-redshift}).

Provided that the peculiar velocity of the galaxy is non-relativistic (which should be true even in cluster environments), then :math:`z_{\rm redshift}^{\rm pec}=v_{\rm z}/c`, and we have

.. math::
    z_{\rm redshift}^{\rm obs} = z_{\rm redshift}^{\rm true} + \frac{v_{\rm z}}{c}(1 + z_{\rm true-redshift}).

Now what we want to do is calculate how we should shift :math:`z_{\rm com}` in accord with

.. math::
    \delta z_{\rm redshift} \equiv z_{\rm redshift}^{\rm obs}-z_{\rm redshift}^{\rm true}.

To do that, we just take the difference in their comoving distances,

.. math::
    \delta D_{\rm com} = D_{\rm com}(z_{\rm redshift}^{\rm obs}) - D_{\rm com}(z_{\rm redshift}^{\rm true}),

where

.. math::
    D_{\rm com}(z) \equiv \frac{c}{H_0} \int_{0}^{z}\frac{dz'}{E(z')},

with :math:`H(z) \equiv H_0 E(z),` where :math:`H_0 = 100h{\rm km/s/Mpc}` is the Hubble constant and the function :math:`E(z)` is implemented by the `astropy.cosmology.FLRW.efunc` function.

Taking the difference gives

.. math::
    \delta D_{\rm com} = \frac{c}{H_0}\int_{0}^{z_{\rm redshift}^{\rm obs}}\frac{dz'}{H_0 E(z')} - \frac{c}{H_0}\int_{0}^{z_{\rm redshift}^{\rm true}}\frac{dz'}{H_0 E(z')}

.. math::
    \delta D_{\rm com} = \frac{c}{H_0}\int_{z_{\rm redshift}^{\rm true}}^{z_{\rm redshift}^{\rm obs}}\frac{dz'}{E(z')}

Since :math:`\delta z_{\rm redshift} = \frac{v_{\rm z}}{c}(1+z_{\rm redshift}^{\rm true})` is typically a very small interval for peculiar velocities relevant to galaxies at observable redshifts, then due to the smoothness of the Hubble rate :math:`E(z)` we can approximate the above integral by treating :math:`E(z)` as a constant-valued :math:`E(z_{\rm redshift})`:

.. math::
    \delta D_{\rm com} \approx c\delta z_{\rm redshift}/H(z_{\rm redshift}^{\rm true})

.. math::
    \delta D_{\rm com} \approx \frac{(1+z_{\rm redshift}^{\rm true})}{H(z_{\rm redshift}^{\rm true})}v_{\rm z}

Now that we have seen the analytical derivation, let's see how this maps onto Halotools source code. The line of code in the `~halotools.mock_observables.apply_zspace_distortion` function implementing this behavior appears as follows:

.. code-block:: python

    pos_err = peculiar_velocity/100./cosmology.efunc(redshift)/scale_factor

The ``pos_err`` variable stores the value of :math:`\delta D_{\rm com}`, and ``1/scale_factor`` is equal to :math:`1+z_{\rm redshift}^{\rm true}`. The ``cosmology.efunc(redshift)`` variable stores the `astropy.cosmology.FLRW.efunc` implementation of :math:`E(z).` Since Halotools adopts the convention that :math:`h=1,` then :math:`H(z_{\rm redshift}^{\rm true}) = 100hE(z) = 100E(z).`

The `~halotools.mock_observables.return_xyz_formatted_array` function implements the result of the same expression stored in the ``spatial_distortion`` variable.

