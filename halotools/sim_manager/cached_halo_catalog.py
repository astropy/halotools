""" Module storing the `~halotools.sim_manager.CachedHaloCatalog`,
the class responsible for retrieving halo catalogs from shorthand
keyword inputs such as ``simname`` and ``redshift``.
"""
import os
from warnings import warn
from copy import deepcopy
import numpy as np

from astropy.table import Table
from ..utils.python_string_comparisons import _passively_decode_string, compare_strings_py23_safe

try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False
    warn("Most of the functionality of the "
        "sim_manager sub-package requires h5py to be installed,\n"
        "which can be accomplished either with pip or conda. ")

from ..sim_manager import sim_defaults, supported_sims

from ..utils import broadcast_host_halo_property, add_halo_hostid

from .halo_table_cache import HaloTableCache
from .ptcl_table_cache import PtclTableCache
from .halo_table_cache_log_entry import get_redshift_string

from ..custom_exceptions import HalotoolsError, InvalidCacheLogEntry


__all__ = ('CachedHaloCatalog', )


class CachedHaloCatalog(object):
    """
    Container class for the halo catalogs and particle data
    that are stored in the Halotools cache log.
    `CachedHaloCatalog` is used to retrieve halo catalogs
    from shorthand keyword inputs such as
    ``simname``, ``halo_finder`` and ``redshift``.

    The halos are stored in the ``halo_table`` attribute
    in the form of an Astropy `~astropy.table.Table`.
    If available, another `~astropy.table.Table` storing
    a random downsampling of dark matter particles
    is stored in the ``ptcl_table`` attribute.
    See the Examples section below for details on how to
    access and manipulate this data.

    For a list of available snapshots provided by Halotools,
    see :ref:`supported_sim_list`.
    For information about the subhalo vs. host halo nomenclature
    conventions used throughout Halotools, see :ref:`rockstar_subhalo_nomenclature`.
    For a thorough discussion of the meaning of each column in the Rockstar halo catalogs,
    see the appendix of `Rodriguez Puebla et al 2016 <http://arxiv.org/abs/1602.04813>`_.
    """
    acceptable_kwargs = ('ptcl_version_name', 'fname', 'simname',
        'halo_finder', 'redshift', 'version_name', 'dz_tol', 'update_cached_fname',
        'preload_halo_table')

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ------------
        simname : string, optional
            Nickname of the simulation used as a shorthand way to keep track
            of the halo catalogs in your cache.
            The simnames of the Halotools-provided catalogs are
            'bolshoi', 'bolplanck', 'consuelo' and 'multidark'.

            Default is set by the ``default_simname`` variable in the
            `~halotools.sim_manager.sim_defaults` module.

        halo_finder : string, optional
            Nickname of the halo-finder used to generate the hlist file from particle data.

            Default is set by the ``default_halo_finder`` variable in the
            `~halotools.sim_manager.sim_defaults` module.

        redshift : float, optional
            Redshift of the halo catalog.

            Default is set by the ``default_redshift`` variable in the
            `~halotools.sim_manager.sim_defaults` module.

        version_name : string, optional
            Nickname of the version of the halo catalog.

            Default is set by the ``default_version_name`` variable in the
            `~halotools.sim_manager.sim_defaults` module.

        ptcl_version_name : string, optional
            Nicknake of the version of the particle catalog associated with
            the halos.

            This argument is typically only used if you have cached your own
            particles via the `~halotools.sim_manager.UserSuppliedPtclCatalog` class.
            Default is set by the ``default_version_name`` variable in the
            `~halotools.sim_manager.sim_defaults` module.

        fname : string, optional
            Absolute path to the location on disk storing the hdf5 file
            of halo data. If passing ``fname``, do not pass the metadata keys
            ``simname``, ``halo_finder``, ``version_name`` or ``redshift``.

        update_cached_fname : bool, optional
            If the hdf5 file storing the halos has been relocated to a new
            disk location after storing the data in cache,
            the ``update_cached_fname`` input can be used together with the
            ``fname`` input to update the cache log with the new disk location.

            See :ref:`relocating_simulation_data_instructions` for
            further instructions.

        dz_tol : float, optional
            Tolerance within to search for a catalog with a matching redshift.
            Halo catalogs in cache with a redshift that differs by greater
            than ``dz_tol`` will be ignored. Default is 0.05.

        Examples
        ---------
        If you followed the instructions in the
        :ref:`download_default_halos` section of the :ref:`getting_started` guide,
        then you can load the default halo catalog into memory by calling the
        `~halotools.sim_manager.CachedHaloCatalog` with no arguments:

        >>> halocat = CachedHaloCatalog() # doctest: +SKIP

        The halos are stored in the ``halo_table`` attribute
        in the form of an Astropy `~astropy.table.Table`.

        >>> halos = halocat.halo_table # doctest: +SKIP

        As with any Astropy `~astropy.table.Table`, the properties of the
        halos can be accessed in the same manner as a Numpy structured array
        or python dictionary:

        >>> array_of_masses = halocat.halo_table['halo_mvir'] # doctest: +SKIP
        >>> x_positions = halocat.halo_table['halo_x'] # doctest: +SKIP

        Note that all keys of a cached halo catalog begin with the substring
        ``halo_``. This is a bookkeeping device used to help
        the internals of Halotools differentiate
        between halo properties and the properties of mock galaxies
        populated into the halos with ambiguously similar names.

        The ``simname``, ``halo_finder``, ``version_name`` and ``redshift``
        keyword arguments fully specify the halo catalog that will be loaded.
        Omitting any of them will select the corresponding default value
        set in the `~halotools.sim_manager.sim_defaults` module.

        >>> halocat = CachedHaloCatalog(redshift = 1, simname = 'multidark') # doctest: +SKIP

        If you forget which catalogs you have stored in cache,
        you have two options for how to remind yourself.
        First, you can use the `~halotools.sim_manager.HaloTableCache` class:

        >>> from halotools.sim_manager import HaloTableCache
        >>> cache = HaloTableCache()
        >>> for entry in cache.log: print(entry) # doctest: +SKIP

        Alternatively, you can simply use a text editor to open the cache log,
        which is stored as ASCII data in the following location on your machine:

        $HOME/.astropy/cache/halotools/halo_table_cache_log.txt

        See also
        ----------
        :ref:`halo_catalog_analysis_quickstart`
        :ref:`halo_catalog_analysis_tutorial`
        """
        self._verify_acceptable_constructor_call(*args, **kwargs)

        assert _HAS_H5PY, "Must have h5py package installed to use CachedHaloCatalog objects"

        try:
            dz_tol = kwargs['dz_tol']
        except KeyError:
            dz_tol = 0.05
        self._dz_tol = dz_tol

        try:
            update_cached_fname = kwargs['update_cached_fname']
        except KeyError:
            update_cached_fname = False
        self._update_cached_fname = update_cached_fname

        self.halo_table_cache = HaloTableCache()

        self._disallow_catalogs_with_known_bugs(**kwargs)

        self.log_entry = self._determine_cache_log_entry(**kwargs)
        self.simname = self.log_entry.simname
        self.halo_finder = self.log_entry.halo_finder
        self.version_name = self.log_entry.version_name
        self.redshift = self.log_entry.redshift
        self.fname = self.log_entry.fname

        self._bind_additional_metadata()

        try:
            preload_halo_table = kwargs['preload_halo_table']
        except KeyError:
            preload_halo_table = False
        if preload_halo_table is True:
            _ = self.halo_table
            del _

        self._set_publication_list(self.simname)

    def _set_publication_list(self, simname):
        try:
            simclass = supported_sims.supported_sim_dict[simname]
            simobj = simclass()
            self.publications = simobj.publications
        except (KeyError, AttributeError):
            self.publications = []

    def _verify_acceptable_constructor_call(self, *args, **kwargs):
        """
        """

        try:
            assert len(args) == 0
        except AssertionError:
            msg = ("\nCachedHaloCatalog only accepts keyword arguments, not position arguments. \n")
            raise HalotoolsError(msg)

        for key in list(kwargs.keys()):
            try:
                assert key in self.acceptable_kwargs
            except AssertionError:
                msg = ("\nCachedHaloCatalog got an unexpected keyword ``" + key + "``\n"
                    "The only acceptable keywords are listed below:\n\n")
                for acceptable_key in self.acceptable_kwargs:
                    msg += "``" + acceptable_key + "``\n"
                raise HalotoolsError(msg)

    def _determine_cache_log_entry(self, **kwargs):
        """
        """
        try:
            self.ptcl_version_name = kwargs['ptcl_version_name']
            self._default_ptcl_version_name_choice = False
        except KeyError:
            self.ptcl_version_name = sim_defaults.default_ptcl_version_name
            self._default_ptcl_version_name_choice = True

        if 'fname' in kwargs:
            fname = kwargs['fname']

            if not os.path.isfile(fname):
                msg = ("\nThe ``fname`` you passed to the CachedHaloCatalog "
                    "constructor is a non-existent path.\n")
                raise HalotoolsError(msg)

            try:
                assert 'simname' not in kwargs
            except AssertionError:
                msg = ("\nIf you specify an input ``fname``, "
                    "do not also specify ``simname``.\n")
                raise HalotoolsError(msg)

            try:
                assert 'halo_finder' not in kwargs
            except AssertionError:
                msg = ("\nIf you specify an input ``fname``, "
                    "do not also specify ``halo_finder``.\n")
                raise HalotoolsError(msg)

            try:
                assert 'redshift' not in kwargs
            except AssertionError:
                msg = ("\nIf you specify an input ``fname``, "
                    "do not also specify ``redshift``.\n")
                raise HalotoolsError(msg)

            try:
                assert 'version_name' not in kwargs
            except AssertionError:
                msg = ("\nIf you specify an input ``fname``, "
                    "do not also specify ``version_name``.\n")
                raise HalotoolsError(msg)

            return self._retrieve_matching_log_entry_from_fname(fname)

        else:

            try:
                simname = str(kwargs['simname'])
                self._default_simname_choice = False
            except KeyError:
                simname = sim_defaults.default_simname
                self._default_simname_choice = True

            try:
                halo_finder = str(kwargs['halo_finder'])
                self._default_halo_finder_choice = False
            except KeyError:
                halo_finder = sim_defaults.default_halo_finder
                self._default_halo_finder_choice = True

            try:
                version_name = str(kwargs['version_name'])
                self._default_version_name_choice = False
            except KeyError:
                version_name = sim_defaults.default_version_name
                self._default_version_name_choice = True

            try:
                redshift = float(kwargs['redshift'])
                self._default_redshift_choice = False
            except KeyError:
                redshift = sim_defaults.default_redshift
                self._default_redshift_choice = True

            return self._retrieve_matching_log_entry_from_metadata(
                simname, halo_finder, version_name, redshift)

    def _retrieve_matching_log_entry_from_fname(self, fname):
        """
        """
        log_entry = self.halo_table_cache.determine_log_entry_from_fname(fname,
            overwrite_fname_metadata=False)

        if not compare_strings_py23_safe(log_entry.fname, fname):
            if self._update_cached_fname is True:
                old_fname = deepcopy(log_entry.fname)
                log_entry = (
                    self.halo_table_cache.determine_log_entry_from_fname(fname,
                        overwrite_fname_metadata=self._update_cached_fname)
                    )
                self.halo_table_cache.update_cached_file_location(
                    fname, old_fname)
            else:
                msg = ("\nThe ``fname`` you passed as an input to the "
                    "CachedHaloCatalog class \ndoes not match the ``fname`` "
                    "stored as metadata in the hdf5 file.\n"
                    "This means that at some point you manually relocated the catalog on disk \n"
                    "after storing its location in cache, "
                    "but you did not yet update the Halotools cache log. \n"
                    "When possible, try to keep your halo catalogs "
                    "at a fixed disk location \n"
                    "as this helps ensure reproducibility. \n"
                    "If the ``fname`` you passed to CachedHaloCatalog is the "
                    "new location you want to store the catalog, \n"
                    "then you can update the cache by calling the CachedHaloCatalog \n"
                    "constructor again and setting the ``update_cached_fname`` variable to True.\n")
                raise HalotoolsError(msg)

        return log_entry

    def _retrieve_matching_ptcl_cache_log_entry(self):
        """
        """

        ptcl_table_cache = PtclTableCache()
        if len(ptcl_table_cache.log) == 0:
            msg = ("\nThe Halotools cache log has no record of any particle catalogs.\n"
                "If you have never used Halotools before, "
                "you should read the Getting Started guide on halotools.readthedocs.io.\n"
                "If you have previously used the package before, \n"
                "try running the halotools/scripts/rebuild_ptcl_table_cache_log.py script.\n")
            raise HalotoolsError(msg)

        gen0 = ptcl_table_cache.matching_log_entry_generator(
            simname=self.simname, version_name=self.ptcl_version_name,
            redshift=self.redshift, dz_tol=self._dz_tol)
        gen1 = ptcl_table_cache.matching_log_entry_generator(
            simname=self.simname, version_name=self.ptcl_version_name)
        gen2 = ptcl_table_cache.matching_log_entry_generator(simname=self.simname)

        matching_entries = list(gen0)

        msg = ("\nYou tried to load a cached particle catalog "
            "with the following characteristics:\n\n")

        if self._default_simname_choice is True:
            msg += ("simname = ``" + str(self.simname) +
                "``  (set by sim_defaults.default_simname)\n")
        else:
            msg += "simname = ``" + str(self.simname) + "``\n"

        if self._default_ptcl_version_name_choice is True:
            msg += ("ptcl_version_name = ``" + str(self.ptcl_version_name) +
                "``  (set by sim_defaults.default_version_name)\n")
        else:
            msg += "ptcl_version_name = ``" + str(self.ptcl_version_name) + "``\n"

        if self._default_redshift_choice is True:
            msg += ("redshift = ``" + str(self.redshift) +
                "``  (set by sim_defaults.default_redshift)\n")
        else:
            msg += "redshift = ``" + str(self.redshift) + "``\n"

        msg += ("\nThere is no matching catalog in cache "
            "within dz_tol = "+str(self._dz_tol)+" of these inputs.\n"
                )

        if len(matching_entries) == 0:
            suggestion_preamble = ("\nThe following entries in the cache log "
                "most closely match your inputs:\n\n")
            alt_list1 = list(gen1)  # discard the redshift requirement
            if len(alt_list1) > 0:
                msg += suggestion_preamble
                for entry in alt_list1:
                    msg += str(entry) + "\n\n"
            else:
                alt_list2 = list(gen2)  # discard the version_name requirement
                if len(alt_list2) > 0:
                    msg += suggestion_preamble
                    for entry in alt_list2:
                        msg += str(entry) + "\n\n"
                else:
                    msg += "There are no simulations matching your input simname.\n"
            raise InvalidCacheLogEntry(msg)

        elif len(matching_entries) == 1:
            log_entry = matching_entries[0]
            return log_entry

        else:
            msg += ("There are multiple entries in the cache log \n"
                "within dz_tol = "+str(self._dz_tol)+" of your inputs. \n"
                "Try using the exact redshift and/or decreasing dz_tol.\n"
                "Now printing the matching entries:\n\n")
            for entry in matching_entries:
                msg += str(entry) + "\n"
            raise InvalidCacheLogEntry(msg)

    def _retrieve_matching_log_entry_from_metadata(self,
            simname, halo_finder, version_name, redshift):
        """
        """

        if len(self.halo_table_cache.log) == 0:
            msg = ("\nThe Halotools cache log is empty.\n"
                "If you have never used Halotools before, "
                "you should read the Getting Started guide on halotools.readthedocs.io.\n"
                "If you have previously used the package before, \n"
                "try running the halotools/scripts/rebuild_halo_table_cache_log.py script.\n")
            raise HalotoolsError(msg)

        gen0 = self.halo_table_cache.matching_log_entry_generator(
            simname=simname, halo_finder=halo_finder,
            version_name=version_name, redshift=redshift,
            dz_tol=self._dz_tol)
        gen1 = self.halo_table_cache.matching_log_entry_generator(
            simname=simname,
            halo_finder=halo_finder, version_name=version_name)
        gen2 = self.halo_table_cache.matching_log_entry_generator(
            simname=simname, halo_finder=halo_finder)
        gen3 = self.halo_table_cache.matching_log_entry_generator(
            simname=simname)

        matching_entries = list(gen0)

        msg = ("\nYou tried to load a cached halo catalog "
            "with the following characteristics:\n\n")

        if self._default_simname_choice is True:
            msg += ("simname = ``" + str(simname) +
                "``  (set by sim_defaults.default_simname)\n")
        else:
            msg += "simname = ``" + str(simname) + "``\n"

        if self._default_halo_finder_choice is True:
            msg += ("halo_finder = ``" + str(halo_finder) +
                "``  (set by sim_defaults.default_halo_finder)\n")
        else:
            msg += "halo_finder = ``" + str(halo_finder) + "``\n"

        if self._default_version_name_choice is True:
            msg += ("version_name = ``" + str(version_name) +
                "``  (set by sim_defaults.default_version_name)\n")
        else:
            msg += "version_name = ``" + str(version_name) + "``\n"

        if self._default_redshift_choice is True:
            msg += ("redshift = ``" + str(redshift) +
                "``  (set by sim_defaults.default_redshift)\n")
        else:
            msg += "redshift = ``" + str(redshift) + "``\n"

        msg += ("\nThere is no matching catalog in cache "
            "within dz_tol = "+str(self._dz_tol)+" of these inputs.\n"
                )

        if len(matching_entries) == 0:
            suggestion_preamble = ("\nThe following entries in the cache log "
                "most closely match your inputs:\n\n")
            alt_list1 = list(gen1)  # discard the redshift requirement
            if len(alt_list1) > 0:
                msg += suggestion_preamble
                for entry in alt_list1:
                    msg += str(entry) + "\n\n"
            else:
                alt_list2 = list(gen2)  # discard the version_name requirement
                if len(alt_list2) > 0:
                    msg += suggestion_preamble
                    for entry in alt_list2:
                        msg += str(entry) + "\n\n"
                else:
                    alt_list3 = list(gen3)  # discard the halo_finder requirement
                    if len(alt_list3) > 0:
                        msg += suggestion_preamble
                        for entry in alt_list3:
                            msg += str(entry) + "\n\n"
                    else:
                        msg += "There are no simulations matching your input simname.\n"
            raise InvalidCacheLogEntry(msg)

        elif len(matching_entries) == 1:
            log_entry = matching_entries[0]
            return log_entry

        else:
            msg += ("There are multiple entries in the cache log \n"
                "within dz_tol = "+str(self._dz_tol)+" of your inputs. \n"
                "Try using the exact redshift and/or decreasing dz_tol.\n"
                "Now printing the matching entries:\n\n")
            for entry in matching_entries:
                msg += str(entry) + "\n"
            raise InvalidCacheLogEntry(msg)

    @property
    def halo_table(self):
        """
        Astropy `~astropy.table.Table` object storing a catalog of dark matter halos.

        You can access the array storing, say, halo virial mass using the following syntax:

        >>> halocat = CachedHaloCatalog() # doctest: +SKIP
        >>> mass_array = halocat.halo_table['halo_mvir'] # doctest: +SKIP

        To see what halo properties are available in the catalog:

        >>> print(halocat.halo_table.keys()) # doctest: +SKIP
        """
        try:
            return self._halo_table
        except AttributeError:
            if self.log_entry.safe_for_cache is True:
                self._halo_table = Table.read(_passively_decode_string(self.fname), path='data')
                self._add_new_derived_columns(self._halo_table)
                return self._halo_table
            else:
                raise InvalidCacheLogEntry(self.log_entry._cache_safety_message)

    def _add_new_derived_columns(self, t):
        if 'halo_hostid' not in list(t.keys()):
            add_halo_hostid(t)

        if 'halo_mvir_host_halo' not in list(t.keys()):
            broadcast_host_halo_property(t, 'halo_mvir')

    def _bind_additional_metadata(self):
        """ Create convenience bindings of all metadata to the `CachedHaloCatalog` instance.
        """
        if not os.path.isfile(self.log_entry.fname):
            msg = ("The following input fname does not exist: \n\n" +
                self.log_entry.fname + "\n\n")
            raise InvalidCacheLogEntry(msg)

        f = h5py.File(self.log_entry.fname, 'r')
        for attr_key in list(f.attrs.keys()):
            if attr_key == 'redshift':
                setattr(self, attr_key, float(get_redshift_string(f.attrs[attr_key])))
            elif attr_key == 'Lbox':
                self.Lbox = np.empty(3)
                self.Lbox[:] = f.attrs['Lbox']
            else:
                setattr(self, attr_key, f.attrs[attr_key])
        f.close()

        matching_sim = self._retrieve_supported_sim()
        if matching_sim is not None:
            for attr in matching_sim._attrlist:
                if hasattr(self, attr):
                    try:
                        a = _passively_decode_string(getattr(self, attr))
                        b = _passively_decode_string(getattr(matching_sim, attr))
                        assert np.all(a == b)
                    except AssertionError:
                        msg = ("The ``" + attr + "`` metadata of the hdf5 file \n"
                            "is inconsistent with the corresponding attribute of the \n" +
                            matching_sim.__class__.__name__ + " class in the "
                            "sim_manager.supported_sims module.\n"
                            "Double-check the value of this attribute in the \n"
                            "NbodySimulation sub-class you added to the supported_sims module. \n"
                               )
                        raise HalotoolsError(msg)
                else:
                    setattr(self, attr, getattr(matching_sim, attr))

    def _retrieve_supported_sim(self):
        """
        """
        matching_sim = None
        for clname in supported_sims.__all__:
            try:
                cl = getattr(supported_sims, clname)
                obj = cl()
                if isinstance(obj, supported_sims.NbodySimulation):
                    if compare_strings_py23_safe(self.simname, obj.simname):
                        matching_sim = obj
            except TypeError:
                pass
        return matching_sim

    @property
    def ptcl_table(self):
        """
        Astropy `~astropy.table.Table` object storing
        a collection of ~1e6 randomly selected dark matter particles.
        """
        try:
            return self._ptcl_table
        except AttributeError:
            try:
                ptcl_log_entry = self.ptcl_log_entry
            except AttributeError:
                self.ptcl_log_entry = (
                    self._retrieve_matching_ptcl_cache_log_entry()
                    )
                ptcl_log_entry = self.ptcl_log_entry

            if ptcl_log_entry.safe_for_cache is True:
                self._ptcl_table = Table.read(_passively_decode_string(ptcl_log_entry.fname), path='data')
                return self._ptcl_table
            else:
                raise InvalidCacheLogEntry(ptcl_log_entry._cache_safety_message)

    def _disallow_catalogs_with_known_bugs(self, simname=sim_defaults.default_simname,
            version_name=sim_defaults.default_version_name, **kwargs):
        """
        """
        if (simname == 'bolplanck') and ('halotools_alpha_version' in version_name):
            msg = ("The ``{0}`` version of the ``{1}`` simulation \n"
            "is known to be spatially incomplete and should not be used.\n"
            "See https://github.com/astropy/halotools/issues/598.\n"
            "You can either download the original ASCII data and process it yourself, \n"
            "or use version_name = ``halotools_v0p4`` instead.\n")
            raise HalotoolsError(msg.format(version_name, simname))
