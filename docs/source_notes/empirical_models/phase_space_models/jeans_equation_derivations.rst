:orphan:

.. _jeans_equation_derivations:

****************************************************
Jeans Equation Derivations 
****************************************************

In this section of the documentation we derive a useful and commonly-used simplification of the collisionless Boltzmann equations known as the Jeans equations. We will take the following expression as our starting point, and refer the reader to Section 5.4.2 of *Galaxy Formation and Evolution* by Mo, van den Bosch and White (2010) for a first-principles derivation of this equation:

.. math::

	\frac{1}{\rho_{\rm sat}(r)}\frac{\rm d}{{\rm d}r}\left[\rho_{\rm sat}(r)\sigma^{2}_{r}(r)\right] + 2\beta(r)\frac{\sigma^{2}_{r}(r)}{r} = -\frac{{\rm d}\Phi(r)}{{\rm d}r}

In the above equation, :math:`\rho_{\rm sat}` is the number density profile of an ensemble of massless tracer particles (in this case the satellite galaxies); :math:`\sigma^{2}_{r}` is the second moment of the distribution of radial velocities of the satellites; :math:`\Phi` is the gravitational potential; and :math:`\beta\equiv 1 - \frac{\sigma_{\theta}^{2}}{\sigma_{r}^{2}}` is the so-called anisotropy profile, with :math:`\beta=0` corresponding to isotropic orbits and :math:`\beta=1` purely radial orbits. 

With two independent unknowns, :math:`\beta(r)` and :math:`\sigma_{r}^{2}(r)`, this differential cannot be solved without making additional assumptions. 

.. _isotropic_jeans_derivation:

Derivation of the isotropic Jeans equations 
==============================================

If we further assume that satellite orbits are isotropic, then the second term on the LHS of the above form of the Jeans equation drops out and we have:

.. math::

	\frac{\rm d}{{\rm d}r}\left[\rho_{\rm sat}(r)\sigma^{2}_{r}(r)\right] = -\rho_{\rm sat}(r)\frac{{\rm d}\Phi(r)}{{\rm d}r} \\ 

	\Rightarrow \left[\rho_{\rm sat}(r)\sigma^{2}_{r}(r)\right]_{r}^{\infty} = -\int_{r}^{\infty}\rho_{\rm sat}(r)\frac{{\rm d}\Phi(r)}{{\rm d}r}

	\Rightarrow \sigma^{2}_{r}(r) = \frac{1}{\rho_{\rm sat}(r)}\int_{r}^{\infty}\rho_{\rm sat}(r)\frac{{\rm d}\Phi(r)}{{\rm d}r}, 

where in the last equation we have applied the boundary condition that :math:`\rho_{\rm sat}(r\rightarrow\infty)\rightarrow 0`. This is the form of the equation used by the `~halotools.empirical_models.IsotropicJeansVelocity` component model and the `~halotools.empirical_models.NFWPhaseSpace` composite model. 

Further Reading 
=================
There are many interesting papers on the Jeans equations. We refer the interested reader to the following papers for details: Lokas & Mamon (2000), arXiv:0002395; van den Bosch et al. (2004), arXiv:0404033; More et al. (2008), arXiv:0807.4532; Wojtak et al. (2008), arXiv:0802.0429, and references therein. 




