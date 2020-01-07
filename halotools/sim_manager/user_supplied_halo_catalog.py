""" Module containing the UserSuppliedHaloCatalog class.
"""

import numpy as np
import os

from warnings import warn
import datetime

from astropy.table import Table, Column

from .halo_table_cache import HaloTableCache
from .halo_table_cache_log_entry import HaloTableCacheLogEntry, get_redshift_string
from .user_supplied_ptcl_catalog import UserSuppliedPtclCatalog

from ..utils.array_utils import custom_len

from ..custom_exceptions import HalotoolsError

__all__ = ('UserSuppliedHaloCatalog', )


class UserSuppliedHaloCatalog(object):
    """ Class used to transform a user-provided halo catalog
    into the standard form recognized by Halotools.

    See :ref:`user_supplied_halo_catalogs` for a tutorial on this class.

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

        **halo_catalog_columns : sequence of arrays
            Sequence of length-*Nhalos* arrays passed in as keyword arguments.

            Each key will be the column name attached to the input array.
            All keys must begin with the substring ``halo_`` to help differentiate
            halo property from mock galaxy properties. At a minimum, there must be a
            ``halo_id`` keyword argument storing a unique integer for each halo,
            as well as columns ``halo_x``, ``halo_y`` and ``halo_z``.
            There must also be some additional mass-like variable,
            for which you can use any name that begins with ``halo_``
            See Examples section for further notes.

        user_supplied_ptclcat : table, optional
            Instance of the `~halotools.sim_manager.UserSuppliedPtclCatalog` class.
            If this keyword is passed, the `UserSuppliedHaloCatalog` instance
            will have a ``ptcl_table`` attribute bound to it storing dark matter particles
            randomly selected from the snapshot. At a minimum, the table must have
            columns ``x``, ``y`` and ``z``. Default is None.

        Examples
        ----------
        Here is an example using dummy data to show how to create a new `UserSuppliedHaloCatalog`
        instance from from your own halo catalog. First the setup:

        >>> redshift = 0.0
        >>> Lbox = 250.
        >>> particle_mass = 1e9
        >>> num_halos = 100
        >>> x = np.random.uniform(0, Lbox, num_halos)
        >>> y = np.random.uniform(0, Lbox, num_halos)
        >>> z = np.random.uniform(0, Lbox, num_halos)
        >>> mass = np.random.uniform(1e12, 1e15, num_halos)
        >>> ids = np.arange(0, num_halos)

        Now we simply pass in both the metadata and the halo catalog columns as keyword arguments:

        >>> halo_catalog = UserSuppliedHaloCatalog(redshift = redshift, Lbox = Lbox, particle_mass = particle_mass, halo_x = x, halo_y = y, halo_z = z, halo_id = ids, halo_mvir = mass)

        Your ``halo_catalog`` object can be used throughout the Halotools package.
        The halo catalog itself is stored in the ``halo_table`` attribute, with columns accessed as follows:

        >>> array_of_masses = halo_catalog.halo_table['halo_mvir']
        >>> array_of_x_positions = halo_catalog.halo_table['halo_x']

        Each piece of metadata you passed in can be accessed as an ordinary attribute:

        >>> halo_catalog_box_size = halo_catalog.Lbox
        >>> particle_mass = halo_catalog.particle_mass

        If you wish to pass in additional metadata, just include additional keywords:

        >>> simname = 'my_personal_sim'

        >>> halo_catalog = UserSuppliedHaloCatalog(redshift = redshift, simname = simname, Lbox = Lbox, particle_mass = particle_mass, halo_x = x, halo_y = y, halo_z = z, halo_id = ids, halo_mvir = mass)

        Similarly, if you wish to include additional columns for your halo catalog,
        Halotools is able to tell the difference between metadata and columns of halo data:

        >>> spin = np.random.uniform(0, 0.2, num_halos)
        >>> halo_catalog = UserSuppliedHaloCatalog(redshift = redshift, halo_spin = spin, simname = simname, Lbox = Lbox, particle_mass = particle_mass, halo_x = x, halo_y = y, halo_z = z, halo_id = ids, halo_mvir = mass)

        If you want to store your halo catalog in the Halotools cache,
        use the `add_halocat_to_cache` method.

        You also have the option to supply a randomly selected downsample of dark matter particles
        via the ``user_supplied_ptclcat`` keyword. This keyword have an instance of the
        `~halotools.sim_manager.UserSuppliedPtclCatalog` class bound to it, which
        helps ensure consistency between the halo catalog and particles.

        Here's an example of how to use this argument using some fake data:

        >>> num_ptcls = int(1e4)
        >>> ptcl_x = np.random.uniform(0, Lbox, num_ptcls)
        >>> ptcl_y = np.random.uniform(0, Lbox, num_ptcls)
        >>> ptcl_z = np.random.uniform(0, Lbox, num_ptcls)

        >>> from halotools.sim_manager import UserSuppliedPtclCatalog
        >>> ptclcat = UserSuppliedPtclCatalog(x = ptcl_x, y = ptcl_y, z = ptcl_z, Lbox = Lbox, particle_mass = particle_mass, redshift = redshift)
        >>> halo_catalog = UserSuppliedHaloCatalog(user_supplied_ptclcat = ptclcat, redshift = redshift, halo_spin = spin, simname = simname, Lbox = Lbox, particle_mass = particle_mass, halo_x = x, halo_y = y, halo_z = z, halo_id = ids, halo_mvir = mass)

        In some scenarios, you may already have tabular data stored in an Astropy `astropy.table.Table`
        or a Numpy structured array. If you want to transfer *all* the columns of
        your ``table_of_halos`` to the `UserSuppliedHaloCatalog`, then you can do so
        by splatting a python dictionary view of the table:

        >>> from astropy.table import Table
        >>> table_of_halos = Table()
        >>> num_halos = int(1e3)
        >>> Lbox = 250.
        >>> table_of_halos['halo_mass'] = np.random.uniform(1e10, 1e15, num_halos)
        >>> table_of_halos['halo_x'] = np.random.uniform(0, Lbox, num_halos)
        >>> table_of_halos['halo_y'] = np.random.uniform(0, Lbox, num_halos)
        >>> table_of_halos['halo_z'] = np.random.uniform(0, Lbox, num_halos)
        >>> table_of_halos['halo_jcvd'] = np.random.random(num_halos)

        >>> table_of_halos['halo_id'] = np.arange(num_halos).astype('i8')

        >>> d = {key:table_of_halos[key] for key in table_of_halos.keys()}
        >>> halocat = UserSuppliedHaloCatalog(simname = simname, redshift = redshift, Lbox = Lbox, particle_mass = particle_mass, **d)

        """
        halo_table_dict, metadata_dict = self._parse_constructor_kwargs(**kwargs)
        self.halo_table = Table(halo_table_dict)

        self._test_metadata_dict(**metadata_dict)

        # make Lbox a 3-vector
        _Lbox = metadata_dict.pop('Lbox')
        metadata_dict['Lbox'] = np.empty(3)
        metadata_dict['Lbox'][:] = _Lbox

        for key, value in metadata_dict.items():
            setattr(self, key, value)

        self._passively_bind_ptcl_table(**kwargs)

    def _parse_constructor_kwargs(self, **kwargs):
        """ Private method interprets constructor keyword arguments and returns two
        dictionaries. One stores the halo catalog columns, the other stores the metadata.

        Parameters
        ------------
        **kwargs : keyword arguments passed to constructor

        Returns
        ----------
        halo_table_dict : dictionary
            Keys are the names of the halo catalog columns, values are length-*Nhalos* ndarrays.

        metadata_dict : dictionary
            Dictionary storing the catalog metadata. Keys will be attribute names bound
            to the `UserSuppliedHaloCatalog` instance.
        """

        try:
            halo_id = np.array(kwargs['halo_id'])
            assert type(halo_id) is np.ndarray
            Nhalos = custom_len(halo_id)
        except KeyError:
            msg = ("\nThe UserSuppliedHaloCatalog requires a ``halo_id`` keyword argument.")
            raise HalotoolsError(msg)

        halo_table_dict = (
            {key: np.array(kwargs[key]) for key in kwargs
            if ((type(kwargs[key]) is np.ndarray) | (type(kwargs[key]) is Column)) and
            (custom_len(kwargs[key]) == Nhalos) and (key[:5] == 'halo_')})
        self._test_halo_table_dict(halo_table_dict)

        metadata_dict = (
            {key: kwargs[key] for key in kwargs
            if (key not in halo_table_dict) and (key != 'ptcl_table')}
            )

        return halo_table_dict, metadata_dict

    def _test_halo_table_dict(self, halo_table_dict):
        """
        """
        try:
            assert 'halo_x' in halo_table_dict
            assert 'halo_y' in halo_table_dict
            assert 'halo_z' in halo_table_dict
            assert len(halo_table_dict) >= 5
        except AssertionError:
            msg = ("\nThe UserSuppliedHaloCatalog requires keyword arguments ``halo_x``, "
                "``halo_y`` and ``halo_z``,\nplus one additional column storing a mass-like variable.\n"
                "Each of these keyword arguments must storing an ndarray of the same length\n"
                "as the ndarray bound to the ``halo_id`` keyword argument.\n")
            raise HalotoolsError(msg)

    def _test_metadata_dict(self, **metadata_dict):
        """
        """
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
            msg = ("\nThe UserSuppliedHaloCatalog requires "
                "keyword arguments ``particle_mass`` and ``redshift``\n"
                "storing scalars that will be interpreted as metadata about the halo catalog.\n")
            raise HalotoolsError(msg)

        Lbox = np.empty(3)
        Lbox[:] = metadata_dict['Lbox']
        try:
            x, y, z = (
                self.halo_table['halo_x'],
                self.halo_table['halo_y'],
                self.halo_table['halo_z']
                )
            assert np.all(x >= 0)
            assert np.all(x <= Lbox[0])
            assert np.all(y >= 0)
            assert np.all(y <= Lbox[1])
            assert np.all(z >= 0)
            assert np.all(z <= Lbox[2])
        except AssertionError:
            msg = ("The ``halo_x``, ``halo_y`` and ``halo_z`` columns must only store arrays\n"
                "that are bound by 0 and the input ``Lbox``. \n")
            raise HalotoolsError(msg)

        try:
            redshift = float(metadata_dict['redshift'])
        except:
            msg = ("\nThe ``redshift`` metadata must be a float.\n")
            raise HalotoolsError(msg)

        for key, value in metadata_dict.items():
            if (type(value) == np.ndarray):
                if custom_len(value) == len(self.halo_table['halo_id']):
                    msg = ("\nThe input ``" + key + "`` argument stores a length-Nhalos ndarray.\n"
                        "However, this key is being interpreted as metadata because \n"
                        "it does not begin with ``halo_``. If this is your intention, ignore this message.\n"
                        "Otherwise, rename this key to begin with ``halo_``. \n")
                    warn(msg, UserWarning)

    def _passively_bind_ptcl_table(self, **kwargs):
        """
        """

        try:
            user_supplied_ptclcat = kwargs['user_supplied_ptclcat']

            try:
                assert isinstance(user_supplied_ptclcat, UserSuppliedPtclCatalog)
            except AssertionError:
                msg = ("\n``user_supplied_ptclcat`` must be "
                    "an instance of UserSuppliedPtclCatalog\n")
                raise HalotoolsError(msg)

            ptcl_table = user_supplied_ptclcat.ptcl_table

            try:
                assert (user_supplied_ptclcat.Lbox == self.Lbox).all()
            except AssertionError:
                msg = ("\nInconsistent values of Lbox between halo and particle catalogs:\n"
                    "For the halo catalog, Lbox = " + str(self.Lbox) + "\n"
                    "For the ``user_supplied_ptclcat``, Lbox = " +
                    str(user_supplied_ptclcat.Lbox) + "\n\n")
                raise HalotoolsError(msg)

            try:
                assert user_supplied_ptclcat.particle_mass == self.particle_mass
            except AssertionError:
                msg = ("\nInconsistent values of particle_mass between halo and particle catalogs:\n"
                    "For the halo catalog, particle_mass = " + str(self.particle_mass) + "\n"
                    "For the ``user_supplied_ptclcat``, particle_mass = " +
                    str(user_supplied_ptclcat.particle_mass) + "\n\n")
                raise HalotoolsError(msg)

            try:
                z1 = get_redshift_string(user_supplied_ptclcat.redshift)
                z2 = get_redshift_string(self.redshift)
                assert z1 == z2
            except AssertionError:
                msg = ("\nInconsistent values of redshift between halo and particle catalogs:\n"
                    "For the halo catalog, redshift = " + str(self.redshift) + "\n"
                    "For the ``user_supplied_ptclcat``, redshift = " +
                    str(user_supplied_ptclcat.redshift) + "\n\n")
                raise HalotoolsError(msg)

            self.ptcl_table = ptcl_table

        except KeyError:
            pass

    def add_halocat_to_cache(self,
            fname, simname, halo_finder, version_name, processing_notes,
            overwrite=False, **additional_metadata):
        """
        Parameters
        ------------
        fname : string
            Absolute path of the file where you will store the halo catalog.
            Your filename must conclude with an `.hdf5` extension.

            The Halotools cache system will remember whatever location
            you choose, so try to choose a reasonably permanent resting place on disk.
            You can always relocate your catalog after caching it
            by following the :ref:`relocating_simulation_data` documentation page.

        simname : string
            Nickname of the simulation used as a shorthand way to keep track
            of the halo catalogs in your cache.

        halo_finder : string
            Nickname of the halo-finder used to generate the hlist file from particle data.

        version_name : string
            Nickname of the version of the halo catalog.
            The ``version_name`` is used as a bookkeeping tool in the cache log.

        processing_notes : string
            String used to provide supplementary notes that will be attached to
            the hdf5 file storing your halo catalog.

        overwrite : bool, optional
            If the chosen ``fname`` already exists, then you must set ``overwrite``
            to True in order to write the file to disk. Default is False.

        **additional_metadata : sequence of strings, optional
            Each keyword of ``additional_metadata`` defines the name
            of a piece of metadata stored in the hdf5 file. The
            value bound to each key can be any string. When you load your
            cached halo catalog into memory, each piece of metadata
            will be stored as an attribute of the
            `~halotools.sim_manager.CachedHaloCatalog` instance.

        """
        try:
            import h5py
        except ImportError:
            msg = ("\nYou must have h5py installed if you want to \n"
                "store your catalog in the Halotools cache. \n")
            raise HalotoolsError(msg)

        ############################################################
        # Perform some consistency checks in the fname
        if (os.path.isfile(fname)) & (overwrite is False):
            msg = ("\nYou attempted to store your halo catalog "
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
            _ = str(halo_finder)
            _ = str(version_name)
            _ = str(processing_notes)
        except:
            msg = ("\nThe input ``simname``, ``halo_finder``, ``version_name`` "
                "and ``processing_notes``\nmust all be strings.")
            raise HalotoolsError(msg)

        for key, value in additional_metadata.items():
            try:
                _ = str(value)
            except:
                msg = ("\nIf you use ``additional_metadata`` keyword arguments \n"
                    "to provide supplementary metadata about your catalog, \n"
                    "all such metadata will be bound to the hdf5 file in the "
                    "format of a string.\nHowever, the value you bound to the "
                    "``"+key+"`` keyword is not representable as a string.\n")
                raise HalotoolsError(msg)

        ############################################################
        # Now write the file to disk and add the appropriate metadata

        self.halo_table.write(fname, path='data', overwrite=overwrite)

        f = h5py.File(fname, 'a')

        redshift_string = get_redshift_string(self.redshift)

        f.attrs.create('simname', np.string_(simname))
        f.attrs.create('halo_finder', np.string_(halo_finder))
        f.attrs.create('version_name', np.string_(version_name))
        f.attrs.create('redshift', np.string_(redshift_string))
        f.attrs.create('fname', np.string_(fname))

        f.attrs.create('Lbox', self.Lbox)
        f.attrs.create('particle_mass', self.particle_mass)

        time_right_now = datetime.datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S')

        f.attrs.create('time_catalog_was_originally_cached', np.string_(time_right_now))

        f.attrs.create('processing_notes', np.string_(processing_notes))

        for key, value in additional_metadata.items():
            f.attrs.create(key, np.string_(value))

        f.close()
        ############################################################
        # Now that the file is on disk, add it to the cache
        cache = HaloTableCache()

        log_entry = HaloTableCacheLogEntry(
            simname=simname, halo_finder=halo_finder,
            version_name=version_name, redshift=self.redshift, fname=fname)

        cache.add_entry_to_cache_log(log_entry, update_ascii=True)
        self.log_entry = log_entry
