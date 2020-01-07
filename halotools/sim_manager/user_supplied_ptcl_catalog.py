""" Module containing the UserSuppliedPtclCatalog class.
"""
import numpy as np
import os
from warnings import warn
import datetime

from astropy.table import Table

try:
    import h5py
except ImportError:
    warn("Most of the functionality of the sim_manager "
         "sub-package requires h5py to be installed,\n"
         "which can be accomplished either with pip or conda")

from .ptcl_table_cache import PtclTableCache
from .ptcl_table_cache_log_entry import PtclTableCacheLogEntry
from .halo_table_cache_log_entry import get_redshift_string

from ..utils.array_utils import custom_len
from ..custom_exceptions import HalotoolsError

__all__ = ('UserSuppliedPtclCatalog', )


class UserSuppliedPtclCatalog(object):
    """ Class used to transform a user-provided particle catalog
    into the standard form recognized by Halotools.

    Random downsamplings of dark matter particles are not especially useful
    catalogs in their own right. So primary purpose of this class
    is the `add_ptclcat_to_cache` method,
    which sets you up to use the dark matter particle collection
    together with the associated halo catalog.

    See :ref:`working_with_alternative_particle_data` for a tutorial on this class.

    """

    def __init__(self, **kwargs):
        """

        Parameters
        ------------
        **metadata : float or string
            Keyword arguments storing catalog metadata.
            The quantities `Lbox` and `particle_mass`
            are required and must be in Mpc/h and Msun/h units, respectively.
            `redshift` is also required metadata.
            See Examples section for further notes.

        **ptcl_catalog_columns : sequence of arrays
            Sequence of length-*Nptcls* arrays passed in as keyword arguments.

            Each key will be the column name attached to the input array.
            At a minimum, there must be columns ``x``, ``y`` and ``z``.
            See Examples section for further notes.

        Examples
        ----------
        Here is an example using dummy data to show how to create a new `UserSuppliedPtclCatalog`
        and store it in cache for future use with the associated halo catalog.
        First the setup:

        >>> redshift = 0.0
        >>> Lbox = 250.
        >>> particle_mass = 1e9
        >>> num_ptcls = int(1e4)
        >>> x = np.random.uniform(0, Lbox, num_ptcls)
        >>> y = np.random.uniform(0, Lbox, num_ptcls)
        >>> z = np.random.uniform(0, Lbox, num_ptcls)
        >>> ptcl_ids = np.arange(0, num_ptcls)
        >>> vx = np.random.uniform(-100, 100, num_ptcls)
        >>> vy = np.random.uniform(-100, 100, num_ptcls)
        >>> vz = np.random.uniform(-100, 100, num_ptcls)

        Now we simply pass in both the metadata and the particle catalog columns as keyword arguments:

        >>> ptcl_catalog = UserSuppliedPtclCatalog(redshift=redshift, Lbox=Lbox, particle_mass=particle_mass, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, ptcl_ids=ptcl_ids)

        Take note: it is important that the value of the input ``redshift`` matches
        whatever the redshift is of the associated halo catalog. Your ``redshift``
        should be accurate to four decimal places.

        Now that we have built a Halotools-formatted particle catalog, we can add it to the cache as follows.

        First choose a relatively permanent location on disk where you will be storing the particle data:

        >>> my_fname = 'some_fname.hdf5'

        Next choose the ``simname`` that matches the ``simname`` of the associated halo catalog, for example:

        >>> my_simname = 'bolplanck'

        Now choose any version name that will help you keep track of
        potentially different version of the same catalog of particles.

        >>> my_version_name = 'any version name'

        Finally, give a short, plain-language descriptions of how
        you obtained your collection of particles:

        >>> my_processing_notes = 'This particle catalog was obtained through the following means: ...'

        Now we add the particle catalog to cache using the following syntax:

        >>> ptcl_catalog.add_ptclcat_to_cache(my_fname, my_simname, my_version_name, my_processing_notes) # doctest: +SKIP

        Your particle catalog has now been cached and is accessible whenever
        you load the associated halo catalog into memory. For example:

        >>> from halotools.sim_manager import CachedHaloCatalog
        >>> halocat = CachedHaloCatalog(simname=my_simname, halo_finder='some halo-finder', version_name='some version-name', redshift=redshift, ptcl_version_name=my_version_name) # doctest: +SKIP

        Note the arguments passed to the `~halotools.sim_manager.CachedHaloCatalog` class.
        The ``version_name`` here refers to the *halos*, not the particles.
        When loading the `~halotools.sim_manager.CachedHaloCatalog`,
        you specify the version name of the particles
        with the ``ptcl_version_name`` keyword argument.
        The ``ptcl_version_name`` need not agree with the ``version_name`` of the associated halos.
        This allows halo and particle catalogs to evolve independently over time.
        In fact, for cases where you have supplied your own particles, it is *strongly* recommended
        that you choose a version name for your particles that differs from the version name
        that Halotools uses for its catalogs. This will help avoid future confusion over the
        where the cached particle catalog came from.

        The particle catalog itself is stored in the ``ptcl_table`` attribute,
        with columns accessed as follows:

        >>> array_of_x_positions = halocat.ptcl_table['x'] # doctest: +SKIP

        If you do not wish to store your particle catalog in cache,
        see the :ref:`using_user_supplied_ptcl_catalog_without_the_cache` section
        of the :ref:`working_with_alternative_particle_data` tutorial.

        """

        ptcl_table_dict, metadata_dict = self._parse_constructor_kwargs(**kwargs)
        self.ptcl_table = Table(ptcl_table_dict)

        self._test_metadata_dict(**metadata_dict)

        # make Lbox a 3-vector
        _Lbox = metadata_dict.pop('Lbox')
        metadata_dict['Lbox'] = np.empty(3)
        metadata_dict['Lbox'][:] = _Lbox

        for key, value in metadata_dict.items():
            setattr(self, key, value)

    def _parse_constructor_kwargs(self, **kwargs):
        """
        """
        try:
            x = kwargs['x']
            assert type(x) is np.ndarray
            y = kwargs['y']
            assert type(y) is np.ndarray
            z = kwargs['z']
            assert type(z) is np.ndarray

            Nptcls = custom_len(x)
            assert Nptcls >= 1e4
            assert Nptcls == len(y)
            assert Nptcls == len(z)
        except KeyError:
            msg = ("\nThe UserSuppliedPtclCatalog requires ``x``, ``y`` and "
                   "``z`` keyword arguments,\n each of which must store an "
                   "ndarray of the same length Nptcls >= 1e4.\n")

            raise HalotoolsError(msg)

        ptcl_table_dict = (
            {key: kwargs[key] for key in kwargs
             if (type(kwargs[key]) is np.ndarray) and
             (custom_len(kwargs[key]) == Nptcls)})

        metadata_dict = (
            {key: kwargs[key] for key in kwargs if key not in ptcl_table_dict})

        return ptcl_table_dict, metadata_dict

    def _test_metadata_dict(self, **metadata_dict):

        try:
            assert 'Lbox' in metadata_dict
            assert custom_len(metadata_dict['Lbox']) in [1,3]
        except AssertionError:
            msg = ("\nThe UserSuppliedPtclCatalog requires keyword argument "
                   "``Lbox``, storing either a scalar or 3-vector.\n")
            raise HalotoolsError(msg)

        try:
            assert 'particle_mass' in metadata_dict
            assert custom_len(metadata_dict['particle_mass']) == 1
            assert 'redshift' in metadata_dict
        except AssertionError:
            msg = ("\nThe UserSuppliedPtclCatalog requires keyword arguments "
                   "``particle_mass`` and ``redshift``\n"
                   "storing scalars that will be interpreted as metadata "
                   "about the particle catalog.\n")
            raise HalotoolsError(msg)

        Lbox = np.empty(3)
        Lbox[:] = metadata_dict['Lbox']
        assert (Lbox > 0).all(), "``Lbox`` must be positive"

        try:
            x, y, z = (
                self.ptcl_table['x'],
                self.ptcl_table['x'],
                self.ptcl_table['z'])

            assert np.all(x >= 0)
            assert np.all(x <= Lbox[0])
            assert np.all(y >= 0)
            assert np.all(y <= Lbox[1])
            assert np.all(z >= 0)
            assert np.all(z <= Lbox[2])
        except AssertionError:
            msg = ("The ``x``, ``y`` and ``z`` columns must only store "
                   "arrays\n that are bound by 0 and the input ``Lbox``. \n")
            raise HalotoolsError(msg)

        try:
            redshift = float(metadata_dict['redshift'])
        except:
            msg = ("\nThe ``redshift`` metadata must be a float.\n")
            raise HalotoolsError(msg)

    def add_ptclcat_to_cache(self, fname, simname, version_name,
                             processing_notes, overwrite=False):

        """
        Parameters
        ------------
        fname : string
            Absolute path of the file to be stored in cache.
            Must conclude with an `.hdf5` extension.

        simname : string
            Nickname of the simulation used as a shorthand way to keep track
            of the catalogs in your cache.

        version_name : string
            Nickname of the version of the particle catalog.
            The ``version_name`` is used as a bookkeeping tool in the cache log.
            As described in the `~halotools.sim_manager.UserSuppliedPtclCatalog` docstring,
            the version name selected here need not match the version name
            of the associated halo catalog.

        processing_notes : string
            String used to provide supplementary notes that will be attached to
            the hdf5 file storing your particle data.

        overwrite : bool, optional
            If the chosen ``fname`` already exists, then you must set ``overwrite``
            to True in order to write the file to disk. Default is False.

        """

        ############################################################
        # Perform some consistency checks in the fname
        if (os.path.isfile(fname)) & (overwrite is False):
            msg = ("\nYou attempted to store your particle catalog "
                   "in the following location: \n\n" + str(fname) +
                   "\n\nThis path points to an existing file. \n"
                   "Either choose a different fname or set ``overwrite`` to True.\n")
            raise HalotoolsError(msg)

        try:
            dirname = os.path.dirname(fname)
            assert os.path.exists(dirname)
        except:
            msg = ("\nThe directory you are trying to store the file does not exist. \n")
            raise HalotoolsError(msg)

        if fname[-5:] != '.hdf5':
            msg = ("\nThe fname must end with an ``.hdf5`` extension.\n")
            raise HalotoolsError(msg)

        ############################################################
        # Perform consistency checks on the remaining log entry attributes
        try:
            _ = str(simname)
            _ = str(version_name)
            _ = str(processing_notes)
        except:
            msg = ("\nThe input ``simname``, ``version_name`` "
                   "and ``processing_notes``\nmust all be strings.")
            raise HalotoolsError(msg)

        ############################################################
        # Now write the file to disk and add the appropriate metadata

        self.ptcl_table.write(fname, path='data', overwrite=overwrite)

        f = h5py.File(fname, 'a')

        redshift_string = get_redshift_string(self.redshift)

        f.attrs.create('simname', np.string_(simname))
        f.attrs.create('version_name', np.string_(version_name))
        f.attrs.create('redshift', np.string_(redshift_string))
        f.attrs.create('fname', np.string_(fname))

        f.attrs.create('Lbox', self.Lbox)
        f.attrs.create('particle_mass', self.particle_mass)

        time_right_now = datetime.datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S')
        f.attrs.create('time_catalog_was_originally_cached', np.string_(time_right_now))

        f.attrs.create('processing_notes', np.string_(processing_notes))

        f.close()

        ############################################################
        # Now that the file is on disk, add it to the cache
        cache = PtclTableCache()

        log_entry = PtclTableCacheLogEntry(
            simname=simname, version_name=version_name,
            redshift=self.redshift, fname=fname)

        cache.add_entry_to_cache_log(log_entry, update_ascii=True)
        self.log_entry = log_entry
