"""
Module containing the template class `~halotools.empirical_models.MockFactory`
used to construct mock galaxy populations.
"""
from __future__ import absolute_import

import numpy as np
from multiprocessing import cpu_count
from copy import copy
from astropy.table import Table

from .mock_helpers import three_dim_pos_bundle, infer_mask_from_kwargs

from .. import model_helpers, model_defaults
import weakref

try:
    from ... import mock_observables

    HAS_MOCKOBS = True
except ImportError:
    HAS_MOCKOBS = False

from ...utils.array_utils import randomly_downsample_data
from ...custom_exceptions import HalotoolsError


__all__ = ["MockFactory"]
__author__ = ["Andrew Hearin"]


class MockFactory(object):
    """Abstract base class responsible for populating a simulation
    with a synthetic galaxy population.

    `MockFactory` is an abstract base class, and cannot be instantiated.
    Concrete sub-classes of `MockFactory` such as `HodMockFactory` and
    `SubhaloMockFactory` are the objects used
    to populate simulations with galaxies.
    """

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        halocat : object
            Object containing the halo catalog and other associated data.
            Produced by `~halotools.sim_manager.CachedHaloCatalog`

        model : object
            A model built by a sub-class of `~halotools.empirical_models.ModelFactory`.

        """

        required_kwargs = ["model"]
        model_helpers.bind_required_kwargs(required_kwargs, self, **kwargs)

        # Make any cuts on the halo catalog requested by the model
        try:
            halocat = kwargs["halocat"]
            self.model = kwargs["model"]
        except KeyError:
            msg = "\n``halocat`` and ``model`` are required ``MockFactory`` arguments\n"
            raise HalotoolsError(msg)
        for key in list(halocat.__dict__.keys()):
            setattr(self, key, halocat.__dict__[key])

        try:
            self.ptcl_table = (
                halocat.ptcl_table
            )  # pre-retrieve the particles from disk, if available
        except:
            pass

        try:
            self.gal_types = self.model.gal_types
        except:
            pass

        # Create a list of halo properties that will be inherited by the mock galaxies
        self.additional_haloprops = copy(
            model_defaults.default_haloprop_list_inherited_by_mock
        )

        if hasattr(self.model, "_haloprop_list"):
            self.additional_haloprops.extend(self.model._haloprop_list)
        # Eliminate any possible redundancies
        self.additional_haloprops = list(set(self.additional_haloprops))

        self.galaxy_table = Table()

    @property
    def model(self):
        """model, a weak reference to the model because mock is always a member of model"""
        return self._model()

    @model.setter
    def model(self, value):
        self._model = weakref.ref(value)

    def populate(self, **kwargs):
        """
        Method populating halos with mock galaxies.

        By calling the `populate` method of your mock, you will repopulate
        the halo catalog with a new realization of the model based on
        whatever values of the model parameters are currently stored in the
        ``param_dict`` of the model.

        For documentation specific to the `populate` method of subhalo-based
        models, see `halotools.empirical_models.SubhaloMockFactory.populate`;
        for HOD-style models
        see `halotools.empirical_models.HodMockFactory.populate`.

        Notes
        -----
        Note the difference between the
        `halotools.empirical_models.MockFactory.populate` method and the
        closely related method
        `halotools.empirical_models.ModelFactory.populate_mock`.
        The `~halotools.empirical_models.ModelFactory.populate_mock` method
        is bound to a composite model instance and is called the *first* time
        a composite model is used to generate a mock. Calling the
        `~halotools.empirical_models.ModelFactory.populate_mock` method creates
        the `~halotools.empirical_models.MockFactory` instance and binds it to
        composite model. From then on, if you want to *repopulate* a new Universe
        with the same composite model, you should instead call the
        `~halotools.empirical_models.MockFactory.populate` method
        bound to ``model.mock``. The reason for this distinction is that
        calling `~halotools.empirical_models.ModelFactory.populate_mock`
        triggers a large number of relatively expensive pre-processing steps
        and self-consistency checks that need only be carried out once.
        See the Examples section below for an explicit demonstration.

        In particular, if you are running an MCMC type analysis,
        you will choose your halo catalog and completeness cuts, and call
        `~halotools.empirical_models.ModelFactory.populate_mock`
        with the appropriate arguments. Thereafter, you can
        explore parameter space by changing the values stored in the
        ``param_dict`` dictionary attached to the model, and then calling the
        `~halotools.empirical_models.MockFactory.populate` method
        bound to ``model.mock``. Any changes to the ``param_dict`` of the
        model will automatically propagate into the behavior of
        the `~halotools.empirical_models.MockFactory.populate` method.

        Normally, repeated calls to the
        `~halotools.empirical_models.MockFactory.populate` method
        should not increase the RAM usage of halotools because a new mock
        catalog is created and the old one
        deleted. However, on certain machines the memory usage was found to
        increase over time. If this is the case and memory usage is critical you
        can try calling gc.collect() immediately following the call to
        ``mock.populate`` to manually invoke python's garbage collection.

        Examples
        ----------
        We'll use a pre-built HOD-style model to demonstrate basic usage.
        The same syntax applies to subhalo-based models.

        >>> from halotools.empirical_models import PrebuiltHodModelFactory
        >>> model_instance = PrebuiltHodModelFactory('zheng07')

        Here we will use a fake simulation, but you can populate mocks
        using any instance of `~halotools.sim_manager.CachedHaloCatalog` or
        `~halotools.sim_manager.UserSuppliedHaloCatalog`.

        >>> from halotools.sim_manager import FakeSim
        >>> halocat = FakeSim()
        >>> model_instance.populate_mock(halocat)

        Your ``model_instance`` now has a ``mock`` attribute bound to it.
        You can call the `populate` method bound to the ``mock``,
        which will repopulate the halo catalog with a new Monte Carlo
        realization of the model.

        >>> model_instance.mock.populate()

        If you want to change the behavior of your model, just change the
        values stored in the ``param_dict``. Differences in the parameter values
        will change the behavior of the mock-population.

        >>> model_instance.param_dict['logMmin'] = 12.1
        >>> model_instance.mock.populate()

        """
        raise NotImplementedError(
            "All subclasses of MockFactory" " must include a populate method"
        )

    @property
    def number_density(self):
        """Comoving number density of the mock galaxy catalog.

        Returns
        --------
        number density : float
            Comoving number density in units of :math:`(h/Mpc)^{3}`.

        """
        ngals = len(self.galaxy_table)
        comoving_volume = np.prod(self.Lbox)
        return ngals / float(comoving_volume)

    def compute_galaxy_clustering(self, include_crosscorr=False, **kwargs):
        """
        Built-in method for all mock catalogs to compute the galaxy clustering signal.

        Parameters
        ----------
        variable_galaxy_mask : scalar, optional
            Any value used to construct a mask to select a sub-population
            of mock galaxies. See examples below.

        include_crosscorr : bool, optional
            Only for simultaneous use with a ``variable_galaxy_mask``-determined mask.
            If ``include_crosscorr`` is set to False (the default option), method will return
            the auto-correlation function of the subsample of galaxies determined by
            the input ``variable_galaxy_mask``. If ``include_crosscorr`` is True,
            method will return the auto-correlation of the subsample,
            the cross-correlation of the subsample and the complementary subsample,
            and the the auto-correlation of the complementary subsample, in that order.
            See examples below.

        mask_function : array, optional
            Function object returning a masking array when operating on the galaxy_table.
            More flexible than the simpler ``variable_galaxy_mask`` option because ``mask_function``
            allows for the possibility of multiple simultaneous cuts. See examples below.

        rbins : array, optional
            Bins in which the correlation function will be calculated.
            Default is set in `~halotools.empirical_models.model_defaults` module.

        num_threads : int, optional
            Number of CPU cores to use in the calculation.
            Default is maximum number available.

        Returns
        --------
        rbin_centers : array
            Midpoint of the bins used in the correlation function calculation

        correlation_func : array
            If not using any mask (the default option), method returns the
            correlation function of the full mock galaxy catalog.

            If using a mask, and if ``include_crosscorr`` is False (the default option),
            method returns the correlation function of the subsample of galaxies determined by
            the input mask.

            If using a mask, and if ``include_crosscorr`` is True,
            method will return the auto-correlation of the subsample,
            the cross-correlation of the subsample and the complementary subsample,
            and the the auto-correlation of the complementary subsample, in that order.
            See the example below.

        Examples
        --------
        Compute two-point clustering of all galaxies in the mock:

        >>> r, clustering = mock.compute_galaxy_clustering() # doctest: +SKIP

        Compute two-point clustering of central galaxies only:

        >>> r, clustering = mock.compute_galaxy_clustering(gal_type = 'centrals') # doctest: +SKIP

        Compute two-point clustering of quiescent galaxies, star-forming galaxies,
        as well as the cross-correlation:

        >>> r, quiescent_clustering, q_sf_cross_clustering, star_forming_clustering = mock.compute_galaxy_clustering(quiescent = True, include_crosscorr = True) # doctest: +SKIP

        Notes
        -----
        The `compute_galaxy_clustering` method bound to mock instances is just a convenience wrapper
        around the `~halotools.mock_observables.tpcf` function. If you wish for greater
        control over how your galaxy clustering signal is estimated,
        see the `~halotools.mock_observables.tpcf` documentation.
        """
        if HAS_MOCKOBS is False:
            msg = (
                "\nThe compute_galaxy_clustering method is only available "
                " if the mock_observables sub-package has been compiled.\n"
                "You are likely encountering this error because you are using \nyour Halotools repository "
                "as your working directory."
            )
            raise HalotoolsError(msg)

        try:
            num_threads = kwargs["num_threads"]
        except KeyError:
            num_threads = cpu_count()

        if "rbins" in kwargs:
            rbins = kwargs["rbins"]
        else:
            rbins = model_defaults.default_rbins
        rbin_centers = (rbins[1:] + rbins[0:-1]) / 2.0
        rmax = max(rbins)

        mask = infer_mask_from_kwargs(self.galaxy_table, **kwargs)
        # Verify that the mask is non-trivial
        if len(self.galaxy_table["x"][mask]) == 0:
            msg = "Zero mock galaxies pass your cuts"
            raise HalotoolsError(msg)

        if include_crosscorr is False:
            pos = three_dim_pos_bundle(
                table=self.galaxy_table,
                key1="x",
                key2="y",
                key3="z",
                mask=mask,
                return_complement=False,
            )
            clustering = mock_observables.tpcf(
                pos,
                rbins,
                period=self.Lbox,
                num_threads=num_threads,
                approx_cell1_size=[rmax, rmax, rmax],
            )
            return rbin_centers, clustering
        else:
            # Verify that the complementary mask is non-trivial
            if len(self.galaxy_table["x"][mask]) == len(self.galaxy_table["x"]):
                msg = (
                    "All mock galaxies pass your cut.\n"
                    "If this result is expected, you should not call the compute_galaxy_clustering"
                    "method with the include_complement keyword"
                )
                raise HalotoolsError(msg)
            pos, pos2 = three_dim_pos_bundle(
                table=self.galaxy_table,
                key1="x",
                key2="y",
                key3="z",
                mask=mask,
                return_complement=True,
            )
            xi11, xi12, xi22 = mock_observables.tpcf(
                sample1=pos,
                rbins=rbins,
                sample2=pos2,
                period=self.Lbox,
                num_threads=num_threads,
                approx_cell1_size=[rmax, rmax, rmax],
            )
            return rbin_centers, xi11, xi12, xi22

    def compute_galaxy_matter_cross_clustering(
        self, include_complement=False, seed=None, **kwargs
    ):
        """
        Built-in method for all mock catalogs to compute the galaxy-matter cross-correlation function.

        Parameters
        ----------
        variable_galaxy_mask : scalar, optional
            Any value used to construct a mask to select a sub-population
            of mock galaxies. See examples below.

        include_complement : bool, optional
            Only for simultaneous use with a ``variable_galaxy_mask``-determined mask.
            If ``include_complement`` is set to False (the default option), method will return
            the cross-correlation function between a random downsampling of dark matter particles
            and the subsample of galaxies determined by
            the input ``variable_galaxy_mask``. If ``include_complement`` is True,
            method will also return the cross-correlation between the dark matter particles
            and the complementary subsample. See examples below.

        mask_function : array, optional
            Function object returning a masking array when operating on the galaxy_table.
            More flexible than the simpler ``variable_galaxy_mask`` option because ``mask_function``
            allows for the possibility of multiple simultaneous cuts. See examples below.

        rbins : array, optional
            Bins in which the correlation function will be calculated.
            Default is set in `~halotools.empirical_models.model_defaults` module.

        num_threads : int, optional
            Number of CPU cores to use in the calculation.
            Default is maximum number available.

        seed : integer, optional
            Random number seed used when drawing random numbers with `numpy.random`.
            Useful when deterministic results are desired, such as during unit-testing.
            Default is None, producing stochastic results.

        Returns
        --------
        rbin_centers : array
            Midpoint of the bins used in the correlation function calculation

        correlation_func : array
            If not using a mask (the default option), method returns the
            correlation function of the full mock galaxy catalog.

            If using a mask, and if ``include_complement`` is False (the default option),
            method returns the cross-correlation function between a random downsampling
            of dark matter particles and the subsample of galaxies determined by
            the input mask.

            If using a mask, and if ``include_complement`` is True,
            method will also return the cross-correlation between the dark matter particles
            and the complementary subsample. See examples below.

        Examples
        --------
        Compute two-point clustering between all mock galaxies and dark matter particles:

        >>> r, galaxy_matter_clustering = mock.compute_galaxy_matter_cross_clustering() # doctest: +SKIP

        Compute the same quantity but for central galaxies only:

        >>> r, central_galaxy_matter_clusteringclustering = mock.compute_galaxy_matter_cross_clustering(gal_type = 'centrals') # doctest: +SKIP

        Compute the galaxy-matter cross-clustering for quiescent galaxies and for star-forming galaxies:

        >>> r, quiescent_matter_clustering, star_forming_matter_clustering = mock.compute_galaxy_matter_cross_clustering(quiescent = True, include_complement = True) # doctest: +SKIP

        Notes
        -----
        The `compute_galaxy_matter_cross_clustering` method bound to mock instances is just a convenience wrapper
        around the `~halotools.mock_observables.tpcf` function. If you wish for greater
        control over how your galaxy clustering signal is estimated,
        see the `~halotools.mock_observables.tpcf` documentation.
        """
        if HAS_MOCKOBS is False:
            msg = (
                "\nThe compute_galaxy_matter_cross_clustering method is only available "
                " if the mock_observables sub-package has been compiled\n"
                "You are likely encountering this error because you are using \nyour Halotools repository "
                "as your working directory."
            )
            raise HalotoolsError(msg)

        nptcl = np.max([model_defaults.default_nptcls, len(self.galaxy_table)])
        if nptcl < len(self.ptcl_table):
            ptcl_table = randomly_downsample_data(self.ptcl_table, nptcl, seed=seed)
        else:
            ptcl_table = self.ptcl_table

        ptcl_pos = three_dim_pos_bundle(table=ptcl_table, key1="x", key2="y", key3="z")

        try:
            num_threads = kwargs["num_threads"]
        except KeyError:
            num_threads = cpu_count()

        if "rbins" in kwargs:
            rbins = kwargs["rbins"]
        else:
            rbins = model_defaults.default_rbins
        rbin_centers = (rbins[1:] + rbins[0:-1]) / 2.0
        rmax = max(rbins)

        try:
            f = kwargs["mask_function"]
            mask = f(self.galaxy_table)
        except KeyError:
            mask = infer_mask_from_kwargs(self.galaxy_table, **kwargs)
        # Verify that the mask is non-trivial
        if len(self.galaxy_table["x"][mask]) == 0:
            msg = "Zero mock galaxies pass your cut"
            raise HalotoolsError(msg)

        if include_complement is False:
            pos = three_dim_pos_bundle(
                table=self.galaxy_table,
                key1="x",
                key2="y",
                key3="z",
                mask=mask,
                return_complement=False,
            )
            clustering = mock_observables.tpcf(
                sample1=pos,
                rbins=rbins,
                sample2=ptcl_pos,
                period=self.Lbox,
                num_threads=num_threads,
                do_auto=False,
                approx_cell1_size=[rmax, rmax, rmax],
            )
            return rbin_centers, clustering
        else:
            # Verify that the complementary mask is non-trivial
            if len(self.galaxy_table["x"][mask]) == len(self.galaxy_table["x"]):
                msg = (
                    "All mock galaxies pass your cut.\n"
                    "If this result is expected, you should not call the compute_galaxy_clustering"
                    "method with the include_complement keyword"
                )
                raise HalotoolsError(msg)
            pos, pos2 = three_dim_pos_bundle(
                table=self.galaxy_table,
                key1="x",
                key2="y",
                key3="z",
                mask=mask,
                return_complement=True,
            )
            clustering = mock_observables.tpcf(
                sample1=pos,
                rbins=rbins,
                sample2=ptcl_pos,
                period=self.Lbox,
                num_threads=num_threads,
                do_auto=False,
                approx_cell1_size=[rmax, rmax, rmax],
            )
            clustering2 = mock_observables.tpcf(
                sample1=pos2,
                rbins=rbins,
                sample2=ptcl_pos,
                period=self.Lbox,
                num_threads=num_threads,
                do_auto=False,
                approx_cell1_size=[rmax, rmax, rmax],
            )
            return rbin_centers, clustering, clustering2

    def compute_fof_group_ids(
        self,
        zspace=True,
        b_perp=model_defaults.default_b_perp,
        b_para=model_defaults.default_b_para,
        **kwargs
    ):
        """
        Method computes the friends-of-friends group IDs of the
        mock galaxy catalog after (optionally) placing the mock into redshift space.

        Parameters
        ----------
        zspace : bool, optional
            Boolean determining whether we apply redshift-space distortions to the
            positions of galaxies using the distant-observer approximation.
            Default is True.

        b_perp : float, optional
            Maximum linking length in the perpendicular direction,
            normalized by the mean separation between galaxies.
            Default is set in `~halotools.empirical_models.model_defaults` module.

        b_para : float, optional
            Maximum linking length in the line-of-sight direction,
            normalized by the mean separation between galaxies.
            Default is set in `~halotools.empirical_models.model_defaults` module.

        num_threads : int, optional
            Number of CPU cores to use in the calculation.
            Default is maximum number available.

        Returns
        --------
        ids : array
            Integer array containing the group ID of each mock galaxy.

        Notes
        -----
        The `compute_fof_group_ids` method bound to mock instances is just a convenience wrapper
        around the `~halotools.mock_observables.groups.FoFGroups` class.
        If you wish for greater control over how your galaxy clustering signal is estimated,
        see the `~halotools.mock_observables.groups.FoFGroups.group_ids` documentation.

        """
        if HAS_MOCKOBS is False:
            msg = (
                "\nThe compute_fof_group_ids method is only available "
                " if the mock_observables sub-package has been compiled\n"
                "You are likely encountering this error because you are using \nyour Halotools repository "
                "as your working directory."
            )
            raise HalotoolsError(msg)

        try:
            num_threads = kwargs["num_threads"]
        except KeyError:
            num_threads = cpu_count()

        x = self.galaxy_table["x"]
        y = self.galaxy_table["y"]
        z = self.galaxy_table["z"]
        if zspace is True:
            z += self.galaxy_table["vz"] / 100.0
            z = model_helpers.enforce_periodicity_of_box(z, self.Lbox[2])
        pos = np.vstack((x, y, z)).T

        group_finder = mock_observables.FoFGroups(
            positions=pos,
            b_perp=b_perp,
            b_para=b_para,
            Lbox=self.Lbox,
            num_threads=num_threads,
        )

        return group_finder.group_ids

    @property
    def satellite_fraction(self):
        """Fraction of mock galaxies that are satellites."""
        satmask = self.galaxy_table["gal_type"] != "centrals"
        return len(self.galaxy_table[satmask]) / float(len(self.galaxy_table))
