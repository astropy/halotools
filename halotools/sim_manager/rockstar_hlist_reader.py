r"""
Module storing the `~halotools.sim_manager.RockstarHlistReader` class,
the primary class used by Halotools to process
publicly available Rockstar hlist files and store them in cache.

"""

import os
import numpy as np
from warnings import warn
from astropy.table import Table
import datetime

try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False
    warn("Most of the functionality of the "
        "sim_manager sub-package requires h5py to be installed,\n"
        "which can be accomplished either with pip or conda. ")

from .tabular_ascii_reader import TabularAsciiReader
from .halo_table_cache import HaloTableCache
from .halo_table_cache_log_entry import HaloTableCacheLogEntry, get_redshift_string

from ..sim_manager import halotools_cache_dirname
from ..custom_exceptions import HalotoolsError
from ..utils.python_string_comparisons import _passively_decode_string


__all__ = ('RockstarHlistReader', )

uninstalled_h5py_msg = ("\nYou must have h5py installed if you want to \n"
    "use the RockstarHlistReader to store your catalog in the Halotools cache. \n"
    "For a stand-alone reader class, you should instead use TabularAsciiReader.\n")


def _infer_redshift_from_input_fname(fname):
    r""" Method extracts the portion of the Rockstar hlist fname
    that contains the scale factor of the snapshot, and returns a
    float for the redshift inferred from this substring.

    Parameters
    ----------
    fname : string
        Name of the Rockstar hlist file

    Returns
    -------
    rounded_redshift : float
        Redshift of the catalog, rounded to four decimals.

    Notes
    -----
    Assumes that the first character of the relevant substring
    is the one immediately following the first incidence of an underscore,
    and final character is the one immediately preceding the second decimal.
    These assumptions are valid for all catalogs currently on the hipacc website.

    """
    fname = os.path.basename(fname)
    first_index = fname.index('_')+1
    last_index = fname.index('.', fname.index('.')+1)
    scale_factor_substring = fname[first_index:last_index]
    scale_factor = float(scale_factor_substring)
    redshift = (1./scale_factor) - 1
    rounded_redshift = float(get_redshift_string(redshift))
    return rounded_redshift


class RockstarHlistReader(TabularAsciiReader):
    r"""
    The `RockstarHlistReader` reads Rockstar hlist ASCII files,
    stores them as hdf5 files in the Halotools cache, and updates the cache log.

    It is important that you carefully read the
    :ref:`reducing_and_caching_a_new_rockstar_catalog`
    before using this class.

    `RockstarHlistReader` is a subclass of
    `~halotools.sim_manager.TabularAsciiReader`, and supplements this behavior
    with the ability to read, update, and search the Halotools cache log.

    If you are planning to use the Halotools cache manager to store and
    keep track of your halo catalogs, this is the class to use. For a stand-alone
    reader of Rockstar hlists or large ASCII files in general, you should instead use the
    `~halotools.sim_manager.TabularAsciiReader` class.
    """

    def __init__(self, input_fname, columns_to_keep_dict,
            output_fname, simname, halo_finder, redshift, version_name,
            Lbox, particle_mass, header_char='#',
            row_cut_min_dict={}, row_cut_max_dict={},
            row_cut_eq_dict={}, row_cut_neq_dict={},
            overwrite=False, ignore_nearby_redshifts=False, dz_tol=0.05,
            processing_notes=' ', **kwargs):
        r"""
        Parameters
        -----------
        input_fname : string
            Absolute path of the file to be processed.

        columns_to_keep_dict : dict
            Dictionary used to define which columns
            of the tabular ASCII data will be kept.

            Each key of the dictionary will be the name of the
            column in the returned data table. The value bound to
            each key is a two-element tuple.
            The first tuple entry is an integer providing
            the *index* of the column to be kept, starting from 0.
            The second tuple entry is a string defining the Numpy dtype
            of the data in that column,
            e.g., 'f4' for a float, 'f8' for a double,
            or 'i8' for a long.

            Thus an example ``columns_to_keep_dict`` could be
            {'halo_mvir': (1, 'f4'), 'halo_id': (0, 'i8'), 'halo_spin': (45, 'f4')}

            The columns of all halo tables stored in the Halotools cache must
            must begin with the substring ``halo_``.
            At a minimum, any halo table stored in cache
            must have the following columns:
            ``halo_id``, ``halo_x``, ``halo_y``, ``halo_z``,
            plus at least one additional column (typically storing a mass-like variable).
            These requirements must be met if you want to use the Halotools cache system,
            or if you want Halotools to populate your halo catalog with mock galaxies.
            If you do not want to conform to these conventions, just use the
            `~halotools.sim_manager.TabularAsciiReader` and handle
            the file storage using your own preferred method.

        output_fname : string
            Absolute path to the location where the hdf5 file will be stored.
            The file extension must be '.hdf5'.
            If the file already exists, you must set
            the keyword argument ``overwrite`` to True.

            If output_fname is set to `std_cache_loc`, Halotools will place the
            catalog in the following location:

            $HOME/.astropy/cache/halotools/halo_catalogs/simname/halo_finder/input_fname.version_name.hdf5

        simname : string
            Nickname of the simulation used as a shorthand way to keep track
            of the halo catalogs in your cache.
            The simnames of the Halotools-provided catalogs are
            'bolshoi', 'bolplanck', 'consuelo' and 'multidark'.

        halo_finder : string
            Nickname of the halo-finder used to generate the hlist file from particle data.
            Most likely this should be 'rockstar', though there are also
            publicly available hlists processed with the 'bdm' halo-finder.

        redshift : float
            Redshift of the halo catalog.

        version_name : string
            Nickname of the version of the halo catalog you produce using RockstarHlistReader.
            The ``version_name`` is used as a bookkeeping tool in the cache log.

            If you process your own halo catalog with the RockstarHlistReader,
            you should choose your own version name that differs from the
            version names of the Halotools-provided catalogs.

        Lbox : float
            Box size of the simulation in Mpc/h.
            ``Lbox`` will automatically be added to the ``supplementary_metadata_dict``
            so that your hdf5 file will have the box size bound as metadata.

        particle_mass : float
            Mass of the dark matter particles of the simulation in Msun/h.
            ``particle_mass`` will automatically be added to the ``supplementary_metadata_dict``
            so that your hdf5 file will have the particle mass bound as metadata.

        row_cut_min_dict : dict, optional
            Dictionary used to place a lower-bound cut on the rows
            of the tabular ASCII data, e.g., to ignore halos below some mass cut.

            Each key of the dictionary must also
            be a key of the input ``columns_to_keep_dict``;
            for purposes of good bookeeping, you are not permitted
            to place a cut on a column that you do not keep. The value
            bound to each key serves as the lower bound on the data stored
            in that row. A row with a smaller value than this lower bound for the
            corresponding column will not appear in the returned data table.

            For example, if row_cut_min_dict = {'mass': 1e10}, then all rows of the
            returned data table will have a mass greater than 1e10.

        row_cut_max_dict : dict, optional
            Dictionary used to place an upper-bound cut on the rows
            of the tabular ASCII data, e.g., to ignore halos not satisfying some relaxation criterion.

            Each key of the dictionary must also
            be a key of the input ``columns_to_keep_dict``;
            for purposes of good bookeeping, you are not permitted
            to place a cut on a column that you do not keep. The value
            bound to each key serves as the upper bound on the data stored
            in that row. A row with a larger value than this upper bound for the
            corresponding column will not appear in the returned data table.

            For example, if row_cut_min_dict = {'mass': 1e15}, then all rows of the
            returned data table will have a mass less than 1e15.

        row_cut_eq_dict : dict, optional
            Dictionary used to place an equality cut on the rows
            of the tabular ASCII data, e.g., to ignore subhalos.

            Each key of the dictionary must also
            be a key of the input ``columns_to_keep_dict``;
            for purposes of good bookeeping, you are not permitted
            to place a cut on a column that you do not keep. The value
            bound to each key serves as the required value for the data stored
            in that row. Only rows with a value equal to this required value for the
            corresponding column will appear in the returned data table.

            For example, if row_cut_eq_dict = {'upid': -1}, then *all* rows of the
            returned data table will have a upid of -1.

        row_cut_neq_dict : dict, optional
            Dictionary used to place an inequality cut on the rows
            of the tabular ASCII data.

            Each key of the dictionary must also
            be a key of the input ``columns_to_keep_dict``;
            for purposes of good bookeeping, you are not permitted
            to place a cut on a column that you do not keep. The value
            bound to each key serves as a forbidden value for the data stored
            in that row. Rows with a value equal to this forbidden value for the
            corresponding column will not appear in the returned data table.

            For example, if row_cut_neq_dict = {'upid': -1}, then *no* rows of the
            returned data table will have a upid of -1.

        header_char : str, optional
            String to be interpreted as a header line of the ascii hlist file.
            Default is '#'.

        overwrite : bool, optional
            If the chosen ``output_fname`` already exists, then you must set ``overwrite``
            to True in order to write the file to disk. Default is False.

        ignore_nearby_redshifts : bool, optional
            Flag used to determine whether nearby redshifts in cache will be ignored.
            If there are existing halo catalogs in the Halotools cache with matching
            ``simname``, ``halo_finder`` and ``version_name``,
            and if one or more of those catalogs has a redshift within ``dz_tol``,
            then the ignore_nearby_redshifts flag must be set to True in order
            for the new halo catalog to be stored in cache.
            Default is False.

        dz_tol : float, optional
            Tolerance determining when another halo catalog in cache is deemed nearby.
            Default is 0.05.

        processing_notes : string, optional
            String used to provide supplementary notes that will be attached to
            the hdf5 file storing your halo catalog.

        Notes
        ------
        When the ``row_cut_min_dict``, ``row_cut_max_dict``,
        ``row_cut_eq_dict`` and ``row_cut_neq_dict`` keyword arguments are used
        simultaneously, only rows passing all cuts will be kept.

        Examples
        ----------
        Suppose you wish to reduce the ASCII data stored by ``input_fname``
        into a data structure with the following columns:
        halo ID, virial mass, x, y, z position, and peak circular velocity,
        where the data are stored in column 1, 45, 17, 18, 19 and 56,
        respectively, where the first column is index 0.
        If you wish to keep *all* rows of the halo catalog:

        >>> columns_to_keep_dict = {'halo_id': (1, 'i8'), 'halo_mvir': (45, 'f4'), 'halo_x': (17, 'f4'), 'halo_y': (18, 'f4'), 'halo_z': (19, 'f4'), 'halo_rvir': (36, 'f4')}
        >>> simname = 'any_nickname'
        >>> halo_finder = 'rockstar'
        >>> version_name = 'rockstar_v1.53_no_cuts'
        >>> redshift = 0.3478
        >>> Lbox, particle_mass = 400, 3.5e8

        >>> reader = RockstarHlistReader(input_fname, columns_to_keep_dict, output_fname, simname, halo_finder, redshift, version_name, Lbox, particle_mass) # doctest: +SKIP
        >>> reader.read_halocat(write_to_disk = True, update_cache_log = True) # doctest: +SKIP

        The halo catalog is now stored in cache and can be loaded into memory
        at any time using the `~halotools.sim_manager.CachedHaloCatalog` class
        with the following syntax.

        >>> from halotools.sim_manager import CachedHaloCatalog
        >>> halocat = CachedHaloCatalog(simname = 'any_nickname', halo_finder = 'rockstar', version_name = 'rockstar_v1.53_no_cuts', redshift = 0.3) # doctest: +SKIP

        Note that once you have stored the catalog with the precise redshift,
        to load it back into memory you do not need to remember the exact redshift
        to four significant digits, you just need to be within ``dz_tol``.
        You can always verify that you are working with the catalog you intended
        by inspecting the metadata:

        >>> print(halocat.redshift) # doctest: +SKIP
        >>> print(halocat.version_name) # doctest: +SKIP

        Now suppose that for your science target of interest,
        subhalos in your simulation with :math:`M_{\rm vir} < 10^{10} M_{\odot}/h`
        are not properly resolved. In this case you can use the ``row_cut_min_dict`` keyword
        argument to discard such halos as the file is read.

        >>> row_cut_min_dict = {'halo_mvir': 1e10}
        >>> version_name = 'rockstar_v1.53_mvir_gt_100'
        >>> processing_notes = 'All halos with halo_mvir < 1e10 km/s were thrown out during the initial catalog reduction'

        >>> reader = RockstarHlistReader(input_fname, columns_to_keep_dict, output_fname, simname, halo_finder, redshift, version_name, Lbox, particle_mass, row_cut_min_dict=row_cut_min_dict, processing_notes=processing_notes) # doctest: +SKIP
        >>> reader.read_halocat(['halo_rvir'], write_to_disk = True, update_cache_log = True) # doctest: +SKIP

        Note the list we passed to the `read_halocat` method via the columns_to_convert_from_kpc_to_mpc
        argument. In common rockstar catalogs, :math:`R_{\rm vir}` is stored in kpc/h units,
        while halo centers are stored in Mpc/h units, a potential source of buggy behavior.
        Take note of all units in your raw halo catalog before caching reductions of it.

        After calling `read_halocat`, the halo catalog is also stored in cache,
        and we load it in the same way as before
        but now using a different ``version_name``:

        >>> halocat = CachedHaloCatalog(simname = 'any_nickname', halo_finder = 'rockstar', version_name = 'rockstar_v1.53_mvir_gt_100', redshift = 0.3) # doctest: +SKIP

        Using the ``processing_notes`` argument is helpful
        in case you forgot exactly how the catalog was initially reduced.
        The ``processing_notes`` string you passed to the constructor
        is stored as metadata on the cached hdf5 file and is automatically
        bound to the `~halotools.sim_manager.CachedHaloCatalog` instance:

        >>> print(halocat.processing_notes) # doctest: +SKIP
        >>> 'All halos with halo_mvir < 1e10 were thrown out during the initial catalog reduction' # doctest: +SKIP

        Any cut you placed on the catalog during its initial
        reduction is automatically bound to the cached halo catalog as additional metadata.
        In this case, since we placed a lower bound on :math:`M_{\rm vir}`:

        >>> print(halocat.halo_mvir_row_cut_min) # doctest: +SKIP
        >>> 100 # doctest: +SKIP

        This metadata provides protection against typographical errors
        that may have been accidentally introduced in the hand-written ``processing_notes``.
        Additional metadata that is automatically bound to all cached catalogs
        includes other sanity checks on our bookkeeping such as ``orig_ascii_fname``
        and ``time_of_catalog_production``.

        See also
        ---------
        :ref:`reducing_and_caching_a_new_rockstar_catalog`

        """

        TabularAsciiReader.__init__(self,
            input_fname, columns_to_keep_dict,
            header_char, row_cut_min_dict, row_cut_max_dict,
            row_cut_eq_dict, row_cut_neq_dict)

        # Require that the minimum required columns have been selected,
        # and that they all begin with `halo_`
        self._enforce_halo_catalog_formatting_requirements()

        # Bind the constructor arguments to the instance
        self.simname = _passively_decode_string(simname)
        self.halo_finder = _passively_decode_string(halo_finder)
        self.redshift = float(get_redshift_string(redshift))
        self.version_name = _passively_decode_string(version_name)
        self.dz_tol = dz_tol
        self.Lbox = Lbox
        self.particle_mass = particle_mass
        self.overwrite = overwrite
        self.ignore_nearby_redshifts = ignore_nearby_redshifts
        self.processing_notes = _passively_decode_string(processing_notes)

        self.output_fname = _passively_decode_string(
            self._retrieve_output_fname(output_fname, self.overwrite, **kwargs)
            )

        self._check_cache_log_for_matching_catalog()

        msg = ("\n\nThe information about your ascii file "
            "and the metadata about the catalog \n"
            "have been processed and no exceptions were raised. \n"
            "Use the ``read_halocat`` method to read the ascii data, \n"
            "setting the write_to_disk and update_cache_log arguments as you like. \n"
            "See the docstring of the ``read_halocat`` method\n"
            "for details about these options. \n")
        print(msg)

    def _check_cache_log_for_matching_catalog(self):
        """ If this method raises no exceptions,
        then either there are no conflicting log entries
        or self.overwrite is set to True. In either case,
        no if there are no exceptions then it is safe to
        reduce the catalog, write it to disk, and request that
        the cache be updated.

        Note, though, that this does not guarantee that the
        resulting processed catalog will be safe to store in cache,
        as the processed catalog will be subject to further testing before
        it can be cached (for example, the halo_id column of the processed catalog
        must contain a set of unique integers).
        """
        self.halo_table_cache = HaloTableCache()
        self.log_entry = HaloTableCacheLogEntry(simname=self.simname,
            halo_finder=self.halo_finder, version_name=self.version_name,
            redshift=self.redshift, fname=self.output_fname)

        if self.log_entry in self.halo_table_cache.log:
            msg = ("\nThere is already an existing entry "
                "in the Halotools cache log\n"
                "that exactly matches the filename and "
                "metadata of the file you intend to write.\n")
            if self.overwrite is True:
                msg += ("Because you have set ``overwrite`` to True, "
                    "\ncalling the read_and_store_halocat_in_cache "
                    "method will overwrite the existing file and log entry.\n")
                warn(msg)
            else:
                linenum = self.halo_table_cache.log.index(self.log_entry) + 1
                msg += ("In order to proceed, "
                    "you must either set ``overwrite`` to True \n"
                    "or manually delete the existing file and also "
                    "remove the entry from the log.\n"
                    "To delete an entry from the log, \n"
                    "you can either use the `remove_entry_from_cache_log` method \n"
                    "of the HaloTableCache class, or equivalently, you can \n"
                    "manually delete line #"+str(linenum)+"from the log file. \n"
                    "The log file is stored in the following location:\n" +
                    self.halo_table_cache.cache_log_fname+"\n"
                        )
                raise HalotoolsError(msg)
        # there are no exact matches, but there may accidentally be nearby redshifts
        else:
            if self.ignore_nearby_redshifts is False:
                pass
            else:
                closely_matching_catalogs = list(
                    self.halo_table_cache.matching_log_entry_generator(
                        simname=self.simname,
                        halo_finder=self.halo_finder,
                        version_name=self.version_name,
                        redshift=self.redshift,
                        fname=self.output_fname, dz_tol=self.dz_tol)
                        )
                if len(closely_matching_catalogs) > 0:
                    msg = ("\nThe following filenames appear in the cache log \n"
                        "and exactly matching metadata and closely matching redshifts: \n\n")
                    for entry in closely_matching_catalogs:
                        msg += str(entry.fname) + "\n"
                    msg += ("In order to proceed, you must either set "
                        "the ``ignore_nearby_redshifts`` to True, \n"
                        "or choose a different ``version_name`` for your catalog.\n")
                    raise HalotoolsError(msg)

    def _enforce_halo_catalog_formatting_requirements(self):
        """ Private method enforces the halo_table formatting conventions of the package.
        """
        try:
            assert 'halo_id' in self.dt.names
            assert 'halo_x' in self.dt.names
            assert 'halo_y' in self.dt.names
            assert 'halo_z' in self.dt.names
            assert len(self.dt.names) > 4
        except AssertionError:
            msg = ("\nAll halo tables stored in cache \n"
            "must at least have the following columns:\n"
                "``halo_id``, ``halo_x``, ``halo_y``, ``halo_z``, \n"
                "plus at least one additional column (typically storing a mass-like variable).\n"
                "If you do not intend to use store the catalog in cache, \n"
                "you should instead process the file with the "
                "`~halotools.sim_manager.TabularAsciiReader`.\n"
                   )
            raise HalotoolsError(msg)

        for name in self.dt.names:
            try:
                assert name[0:5] == 'halo_'
            except AssertionError:
                msg = ("\nAll columns of halo tables stored in the Halotools cache\n"
                    "must begin with the substring ``halo_``.\n")
                msg += ("The column name ``"+name+"`` "
                    "appeared in your input ``columns_to_keep_dict``.\n"
                    "If you do not intend to use store the catalog in cache, \n"
                    "you should instead process the file with the "
                    "`~halotools.sim_manager.TabularAsciiReader`.\n"
                        )
                raise HalotoolsError(msg)

    def _get_default_output_fname(self):
        """
        """
        dirname = os.path.join(halotools_cache_dirname, 'halo_catalogs',
            self.simname, self.halo_finder)
        try:
            os.makedirs(dirname)
        except OSError:
            pass

        if self.input_fname[-3:] == '.gz':
            rootname = self.input_fname[:-3]
        else:
            rootname = self.input_fname

        basename = (
            os.path.basename(rootname) +
            '.' + self.version_name + '.hdf5'
            )
        default_fname = os.path.join(dirname, basename)
        return _passively_decode_string(default_fname)

    def _retrieve_output_fname(self, output_fname, overwrite, **kwargs):
        """ Private method checks to see whether the chosen
        ``output_fname`` already exists on disk, and enforces
        compatibility with ``overwrite``.
        """
        if output_fname == 'std_cache_loc':
            output_fname = self._get_default_output_fname()

        try:
            assert output_fname[-5:] == '.hdf5'
        except:
            msg = ("\nThe output_fname must be a string or unicode that concludes "
                " with file extension '.hdf5'\n")
            raise HalotoolsError(msg)

        return _passively_decode_string(output_fname)

    def read_halocat(self, columns_to_convert_from_kpc_to_mpc,
            write_to_disk=False, update_cache_log=False,
            add_supplementary_halocat_columns=True, **kwargs):
        r""" Method reads the ascii data and
        binds the resulting catalog to ``self.halo_table``.

        By default, the optional ``write_to_disk`` and ``update_cache_log``
        arguments are set to False because Halotools will not
        write large amounts of data to disk without your explicit instructions
        to do so.

        If you want an untouched replica of the downloaded halo catalog on disk,
        then you should set both of these arguments to True, in which case
        your reduced catalog will be saved on disk and stored in cache immediately.

        However, you may wish to supplement your halo catalog with additional
        halo properties before storing on disk. This can easily be accomplished by
        simply manually adding columns to the ``halo_table`` attribute of the
        `RockstarHlistReader` instance after reading in the data. In this case,
        set ``write_to_disk`` to False, add your new data, and then call the
        ``write_to_disk`` method and ``update_cache_log`` methods in succession.
        In such a case, it is good practice to make an explicit note of what you have done
        in the ``processing_notes`` attribute of the reader instance so that you will
        have a permanent record of how you processed the catalog.

        Parameters
        -----------
        columns_to_convert_from_kpc_to_mpc : list of strings
            List providing column names that should be divided by 1000
            in order to convert from kpc/h to Mpc/h units.
            This is necessary with typical rockstar catalogs for the
            ``halo_rvir``, ``halo_rs`` and ``halo_xoff`` columns, which are stored
            in kpc/h, whereas halo centers are typically stored in Mpc/h.
            All strings appearing in ``columns_to_convert_from_kpc_to_mpc``
            must also appear in the ``columns_to_keep_dict``.
            It is permissible for ``columns_to_convert_from_kpc_to_mpc``
            to be an empty list. See Notes for further discussion.

            Note that this feature is only temporary. The API of this function
            will change when Halotools adopts Astropy Units.

        write_to_disk : bool, optional
            If True, the `write_to_disk` method will be called automatically.
            Default is False, in which case you must call the `write_to_disk`
            method yourself to store the processed catalog. In that case,
            you will also need to manually call the ``update_cache_log`` method
            after writing to disk.

        update_cache_log : bool, optional
            If True, the `update_cache_log` method will be called automatically.
            Default is False, in which case you must call the `update_cache_log`
            method yourself to add the the processed catalog to the cache.

        add_supplementary_halocat_columns : bool, optional
            Boolean determining whether the halo_table will have additional
            columns added to it computed by the add_supplementary_halocat_columns method.
            Default is True.

            Note that this feature is rather bare-bones and is likely to significantly
            evolve and/or entirely vanish in future releases.

        chunk_memory_size : int, optional
            Determine the approximate amount of Megabytes of memory
            that will be processed in chunks. This variable
            must be smaller than the amount of RAM on your machine;
            choosing larger values typically improves performance.
            Default is 500 Mb.

        Notes
        -----
        Regarding the ``columns_to_convert_from_kpc_to_mpc`` argument,
        of course there could be other columns whose units you want to convert
        prior to caching the catalog, and simply division by 1000 may not be the
        appropriate unit conversion. To handle such cases, you should do
        the following. First, use the `read_halocat` method  with
        the ``write_to_disk`` and ``update_cache_log`` arguments both set to False.
        This will load the catalog from disk into memory.
        Now you are free to overwrite any column in the halo_table that you wish.
        When you have finished preparing the catalog, call the `write_to_disk`
        and `update_cache_log` methods (in that order).
        As you do so, be sure to include explicit notes of all manipulations you
        made on the halo_table between the time you called `read_halocat` and
        `write_to_disk`, and bind these notes to the ``processing_notes`` argument.

        """
        for key in columns_to_convert_from_kpc_to_mpc:
            try:
                assert key in self.columns_to_keep_dict
            except AssertionError:
                msg = ("\nYou included the ``" + key + "`` column in the input \n"
                    "``columns_to_convert_from_kpc_to_mpc`` but not in the input "
                    "``columns_to_keep_dict``\n")
                raise HalotoolsError(msg)

        result = self._read_ascii(**kwargs)
        self.halo_table = Table(result)

        for key in columns_to_convert_from_kpc_to_mpc:
            self.halo_table[key] /= 1000.

        if add_supplementary_halocat_columns is True:
            self.add_supplementary_halocat_columns()

        if write_to_disk is True:
            self.write_to_disk()
            self._file_has_been_written_to_disk = True
        else:
            self._file_has_been_written_to_disk = False

        if update_cache_log is True:
            if self._file_has_been_written_to_disk is True:
                self.update_cache_log()
            else:
                msg = ("\nYou set update_cache_log to True but the \n"
                    "newly processed halo_table has not yet been written to disk.\n")
                if write_to_disk is False:
                    msg += ("This is because you set write_to_disk to False, \n"
                        "in which case the read_halocat method "
                        "will not automatically update the cache.\n")
                    warn(msg)
                else:
                    msg += ("This indicates that there was a problem "
                        "writing the halo_table to disk.\n"
                        "After you resolve the problem, you can manually call \n"
                        "the write_to_disk and update_cache_log methods.\n")
                    raise HalotoolsError(msg)

    def _read_ascii(self, **kwargs):
        r""" Method reads the input ascii and returns
        a structured Numpy array of the data
        that passes the row- and column-cuts.

        Parameters
        ----------
        chunk_memory_size : int, optional
            Determine the approximate amount of Megabytes of memory
            that will be processed in chunks. This variable
            must be smaller than the amount of RAM on your machine;
            choosing larger values typically improves performance.
            Default is 500 Mb.

        Returns
        --------
        full_array : array_like
            Structured Numpy array storing the rows and columns
            that pass the input cuts. The columns of this array
            are those selected by the ``column_indices_to_keep``
            argument passed to the constructor.

        Notes
        -----
        The behavior of this function is entirely controlled in the
        `~halotools.sim_manager.TabularAsciiReader` superclass.
        This trivial reimplementation is simply here to guide readers
        of the source code to the location of the implementation -
        this private function should not be called by users.
        """
        return TabularAsciiReader.read_ascii(self, **kwargs)

    def write_to_disk(self):
        """ Method writes ``self.halo_table`` to ``self.output_fname``
        and also calls the ``self._write_metadata`` method to place the
        hdf5 file into standard form.

        It is likely that you will want to call the ``update_cache_log`` method
        after calling ``write_to_disk`` so that you can take advantage of the convenient
        syntax provided by the `~halotools.sim_manager.CachedHaloCatalog` class.
        """
        if not _HAS_H5PY:
            raise HalotoolsError(uninstalled_h5py_msg)

        self.halo_table.write(
            _passively_decode_string(self.output_fname), path='data', overwrite=self.overwrite)
        self._write_metadata()

    def _write_metadata(self):
        """ Private method to add metadata to the hdf5 file.
        """
        if not _HAS_H5PY:
            raise HalotoolsError(uninstalled_h5py_msg)

        # Now add the metadata
        f = h5py.File(self.output_fname, 'a')
        f.attrs.create('simname', np.string_(self.simname))
        f.attrs.create('halo_finder', np.string_(self.halo_finder))
        redshift_string = np.string_(get_redshift_string(self.redshift))
        f.attrs.create('redshift', redshift_string)
        f.attrs.create('version_name', np.string_(self.version_name))
        f.attrs.create('fname', np.string_(self.output_fname))

        f.attrs.create('Lbox', self.Lbox)
        f.attrs.create('particle_mass', self.particle_mass)
        f.attrs.create('orig_ascii_fname', np.string_(self.input_fname))

        time_right_now = np.string_(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        f.attrs.create('time_of_catalog_production', time_right_now)

        if self.processing_notes is not None:
            f.attrs.create('processing_notes', np.string_(self.processing_notes))

        # Store all the choices for row cuts as metadata
        for haloprop_key, cut_value in self.row_cut_min_dict.items():
            attrname = haloprop_key + '_row_cut_min'
            f.attrs.create(attrname, cut_value)

        for haloprop_key, cut_value in self.row_cut_max_dict.items():
            attrname = haloprop_key + '_row_cut_max'
            f.attrs.create(attrname, cut_value)

        for haloprop_key, cut_value in self.row_cut_eq_dict.items():
            attrname = haloprop_key + '_row_cut_eq'
            f.attrs.create(attrname, cut_value)

        for haloprop_key, cut_value in self.row_cut_neq_dict.items():
            attrname = haloprop_key + '_row_cut_neq'
            f.attrs.create(attrname, cut_value)

        f.close()

    def update_cache_log(self):
        """ Method updates the cache log with the new catalog,
        provided that it is safe to add to the cache.
        """
        self.halo_table_cache.add_entry_to_cache_log(
            self.log_entry, update_ascii=True)

    def add_supplementary_halocat_columns(self):
        """ Add the halo_nfw_conc and halo_hostid columns.
        This implementation will eventually change in favor of something
        more flexible.
        """
        # Add the halo_nfw_conc column
        if ('halo_rvir' in list(self.halo_table.keys())) & ('halo_rs' in list(self.halo_table.keys())):
            self.halo_table['halo_nfw_conc'] = (
                self.halo_table['halo_rvir'] / self.halo_table['halo_rs']
                )

        # Add the halo_nfw_conc column
        self.halo_table['halo_hostid'] = self.halo_table['halo_id']
        subhalo_mask = self.halo_table['halo_upid'] != -1
        self.halo_table['halo_hostid'][subhalo_mask] = (
            self.halo_table['halo_upid'][subhalo_mask]
            )
