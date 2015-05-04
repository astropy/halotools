# -*- coding: utf-8 -*-
"""
Methods and classes to load halo and particle catalogs into memory.

"""

__all__=['ProcessedSnapshot','CatalogManager', 'RockstarReader']

#from __future__ import (absolute_import, division, print_function,
#                        unicode_literals)

import numpy as np
import os, sys, warnings, urllib2, fnmatch
import pickle

HAS_SOUP = False
try:
    from bs4 import BeautifulSoup
    HAS_SOUP = True
except:
    pass

HAS_REQUESTS = False
try:
    import requests
    HAS_REQUESTS = True
except:
    pass

HAS_H5PY = False
try:
    import h5py
    HAS_H5PY = True
except:
    pass


from astropy.io import fits as fits
from astropy.table import Table
from astropy.utils.data import get_readable_fileobj
from astropy.utils.data import _get_download_cache_locs as get_download_cache_locs
from astropy.utils.data import _open_shelve as open_shelve

from . import configuration
from . import sim_specs
from . import sim_defaults

from ..utils.array_utils import find_idx_nearest_val
from ..utils.array_utils import array_like_length as aph_len
from ..utils.io_utils import download_file_from_url



class ProcessedSnapshot(object):
    """ Class containing halo and particle data taken from 
    a single snapshot of some Nbody simulation.
    """

    def __init__(self,
        simname=sim_defaults.default_simulation_name,
        scale_factor=sim_defaults.default_scale_factor,
        halo_finder=sim_defaults.default_halo_finder,
        download_yn=True, **kwargs):

        self.simulation_name = simname
        self.scale_factor = scale_factor
        self.halo_finder = halo_finder

        self.download_yn = download_yn

        catman = CatalogManager()
        self.CatalogManager = catman

        Lbox, mp, softening = catman.get_simulation_properties(self.simulation_name)
        self.Lbox = Lbox
        self.particle_mass = mp
        self.softening_length = softening

        ### OBSOLETE NOW - MUST BE REWRITTEN ###
        halo_catalog_filename,closest_scale_factor = (
            catman.find_nearest_snapshot_in_cache('subhalos',
                scale_factor = self.scale_factor,
                simname=self.simulation_name,
                halo_finder=self.halo_finder)
            )

        # If there are no matching halo catalogs in cache,
        # set the halo catalog to the default halo catalog
        if (halo_catalog_filename==None) or (closest_scale_factor != self.scale_factor):
            halo_catalog_filename = catman.default_halo_catalog_filename
            # Download the catalog, if desired
            if download_yn==True:
                catman.download_all_default_catalogs()

        self.halo_catalog_filename = halo_catalog_filename
        self.halo_catalog_dirname = configuration.get_catalogs_dir('subhalos')

        ### OBSOLETE NOW - MUST BE REWRITTEN ###
        particle_catalog_filename,closest_scale_factor = (
            catman.find_nearest_snapshot_in_cache('particles',
                scale_factor = self.scale_factor,
                simname=self.simulation_name,
                halo_finder=self.halo_finder)
            )

        # If there are no matching particle catalogs in cache,
        # set the particle catalog to the default particle catalog
        if (particle_catalog_filename==None) or (closest_scale_factor != self.scale_factor):
            particle_catalog_filename = catman.default_particle_catalog_filename
            # Download the catalog, if desired
            if download_yn==True:
                catman.download_all_default_catalogs()

        self.particle_catalog_filename = particle_catalog_filename
        self.particle_catalog_dirname = configuration.get_catalogs_dir('particles')

    @property
    def particles(self):
        """ Method to load simulation particle data into memory. 

        The property decorator syntax allows this method to be called 
        as if it is an attribute.

        """

        catalog_type = 'particles'
        particles = self.CatalogManager.load_catalog(catalog_type,
            dirname = self.particle_catalog_dirname,
            filename=self.particle_catalog_filename,
            download_yn = self.download_yn)

        return particles


    @property
    def halos(self):
        """ Method to load simulation halo catalog into memory. 

        The property decorator syntax allows this method to be called 
        as if it is an attribute.

        """
        catalog_type = 'subhalos'
        halos = self.CatalogManager.load_catalog(catalog_type,
            dirname = self.halo_catalog_dirname,
            filename=self.halo_catalog_filename,
            download_yn = self.download_yn)

        return halos

###################################################################################################

class CatalogManager(object):
    """ Container class for managing I/O of halo & particle catalogs.
    """

    def __init__(self):
        self.slac_urls = {'bolshoi_halos' : 'http://www.slac.stanford.edu/~behroozi/Bolshoi_Catalogs/',
        'bolshoi_bdm_halos' : 'http://www.slac.stanford.edu/~behroozi/Bolshoi_Catalogs_BDM/',
        'multidark_halos' : 'http://slac.stanford.edu/~behroozi/MultiDark_Hlists_Rockstar/',
        'consuelo_halos' : 'http://www.slac.stanford.edu/~behroozi/Consuelo_Catalogs/'
        }

        self.default_halo_catalog_filename = (
            sim_defaults.default_simulation_name+'_a'+
            str(np.round(sim_defaults.default_scale_factor,4))+'_'+
            sim_defaults.default_halo_finder+'_subhalos.hdf5')

        self.default_particle_catalog_filename = (
            sim_defaults.default_simulation_name+'_a'+
            str(np.round(sim_defaults.default_scale_factor,4))+
            '_particles.fits')


    def download_raw_halocat(self, simname, halo_finder, input_redshift, **kwargs):
        """ Method to download publicly available ascii data of 
        raw halo catalog from web location. 

        Parameters 
        ----------
        simname : string 
            Nickname of the simulation, e.g. `bolshoi`. 

        halo_finder : string 
            Nickname of the halo-finder, e.g. `rockstar`. 

        input_redshift : float 
            Redshift of the requested snapshot. Must match one of the 
            available snapshots, or a prompt will be issued providing the nearest 
            available snapshots to choose from. 

        dz_tol : float, optional
            Tolerance value determining how close the requested redshift must be to 
            some available snapshot before issuing a warning. Default value is 0.1. 

        download_loc : string, optional
            Absolute pathname of where the raw halo catalog will be stored. 
            Default is the halotools cache directory. 

        url : string, optional
            Web location of the halo catalog. Default behavior is for Halotools to use 
            pre-determined locations for the supported catalogs. If providing a url, 
            you must specify the full web location and filename, 
            e.g., `www.some.url/some.fname.dat`. 

        overwrite : boolean, optional
            If a file with the same filename already exists 
            in the requested download location, the `overwrite` boolean determines 
            whether or not to overwrite the file. Default is False. 
        """

        if HAS_SOUP == False:
            print("Must have BeautifulSoup installed to use Halotools Catalog Manager")
            return 

        if 'dz_tol' in kwargs.keys():
            dz_tol = kwargs['dz_tol']
        else:
            dz_tol = 0.1

        if 'url' in kwargs.keys():
            url = kwargs['url']
        else:
            # Check the default location url for the halo catalog 
            # that most closely matches the requested redshift
            list_of_available_snapshots = self.retrieve_available_raw_halocats(simname, halo_finder)
            closest_snapshot_fname = self.find_closest_raw_halocat(
                list_of_available_snapshots, input_redshift)
            scale_factor_of_closest_match = float(
                self.get_scale_factor_substring(closest_snapshot_fname)
                )
            redshift_of_closest_match = (1./scale_factor_of_closest_match) - 1
            if abs(redshift_of_closest_match - input_redshift) > dz_tol:
                msg = (
                    "No raw %s halo catalog has \na redshift within %.2f " + 
                    "of the input_redshift = %.2f.\n The closest redshift for these catalogs is %.2f"
                    )
                print(msg % (simname, dz_tol, input_redshift, redshift_of_closest_match))
                return 
            else:
                key = simname+'_'+halo_finder
                url = sim_defaults.raw_halocat_url[key]+closest_snapshot_fname


        if 'download_loc' in kwargs.keys():
            download_loc = kwargs['download_loc']
            # We were given an explicit path to store the catalog
            # Check that this path actually exists, and if so, use it 
            if not os.path.exists(download_loc):
                raise IOError("Input directory name %s for download location"
                    "of raw halo catalog does not exist" % download_loc)
            else:
                output_fname = os.path.join(download_loc, closest_snapshot_fname)
                self.update_list_of_previously_used_dirnames(download_loc)
        else:
            # We were not given an explicit path, so use the default Halotools cache dir
            cache_dirname = configuration.get_catalogs_dir('raw_halos', 
                simname=simname, halo_finder=halo_finder)
            output_fname = os.path.join(cache_dirname, closest_snapshot_fname)

        if os.path.isfile(output_fname):
            if ('overwrite' in kwargs.keys()) & (kwargs['overwrite'] ==True):
                warnings.warn("Downloading halo catalog and overwriting existing file %s" % output_fname)
            else:
                raise IOError("The following filename already exists: \n%s\n"
                    "If you really want to overwrite the file, \n"
                    "you must call this function again with the "
                    "keyword argument `overwrite` set to `True`" % output_fname)

        download_file_from_url(url, output_fname)

    def locate_raw_halocat(self, **kwargs):
        """ Return full path to the requested raw halo catalog. 

        Parameters 
        ----------
        fname : string, optional keyword argument 
            Absolute pathname to the raw halo catalog. 
            If not passed, must pass simname and redshift 
            keyword arguments.

        simname : string, optional keyword argument 
            Nickname of the simulation, e.g., `bolshoi`. 

        halo_finder : string, optional keyword argument 
            Nickname of the halo-finder, e.g., `rockstar` or `bdm`.
            Default is `rockstar`.  

        redshift : float, optional keyword argument 
            Redshift of the requested snapshot.

        Returns 
        -------
        halocat_fname : string 
            Absolute path to raw halo catalog

        Notes 
        -----
        If `fname`, `simname`, and `redshift` arguments are all supplied, 
        method will return `fname`. 

        """

        # First check that we were provided sufficient inputs
        if ('fname' not in kwargs.keys()):
            if ('simname' not in kwargs.keys()) or ('redshift' not in kwargs.keys()):
                msg = ("If not passing an absolute filename to locate_raw_halocat,\n"
                    "must pass both a simname and a redshift")
                raise IOError(msg)

        if 'fname' in kwargs.keys():
            return kwargs['fname']
        else:
            if 'halo_finder' not in kwargs.keys():
                halo_finder = sim_defaults.default_halo_finder
            else:
                halo_finder = kwargs['halo_finder']
            return self.full_fname_closest_raw_halocat_in_cache(
                kwargs['simname'], halo_finder, kwargs['redshift'])


    def process_raw_halocat(self, input_fname, simname, halo_finder, cuts_funcobj):
        """ Method reads in raw halo catalog ASCII data, makes the desired cuts, 
        and returns a numpy structured array of the rows passing the cuts. 

        Parameters 
        ----------
        input_fname : string 
            filename (including absolute path) where the ASCII data are stored. 

        simname : string 
            nickname of the simulation, e.g., `bolshoipl`. 

        halo_finder : string 
            Nickname of the halo-finder, e.g., `rockstar`. 

        cuts_funcobj : function object
            Function used to apply cuts to the rows of the ASCII data. 
            `cuts_funcobj` should accept a length-Nrows numpy structured array as input, 
            and return a length-Nrows boolean array.

        Returns 
        -------
        arr : array 
            Structured numpy array storing all rows passing the cuts. 

        """

        reader = RockstarReader(input_fname, simname=simname, halo_finder=halo_finder)
        arr = reader.read_halocat(cuts_funcobj=cuts_funcobj)
        reader._compress_ascii()

        return arr

    def store_processed_halocat(self, catalog, uncut_catalog_fname, simname, halo_finder, 
        cuts_funcobj, version_name, overwrite=False, **kwargs):
        """
        Method stores an hdf5 binary of the reduced halo catalog. 

        Parameters 
        ----------
        catalog : structured array 
            Numpy array of halo data. 

        uncut_catalog_fname : string 
            Filename of the location of the uncut halo catalog ASCII data. 

        simname : string 
            Nickname of the simulation, e.g., `bolshoi`. 

        halo_finder : string 
            Nickname of the halo-finder, e.g., `rockstar`. 

        cuts_funcobj : function object 
            Function used to apply cuts to the original halo catalog. 

        version_name : string 
            String that will be appended to `uncut_catalog_fname` to create a new 
            filename for the cut halo catalog. 

        overwrite : bool, optional 
            If True, and if there exists a catalog with the same filename in the 
            output location, the existing catalog will be overwritten. Default is False.  
        """

        if HAS_H5PY==False:
            raise ImportError("Must have h5py installed to use the "
                "store_processed_halocat method")
            return 

        if 'output_loc' in kwargs.keys():
            output_loc = kwargs['output_loc']
        else:
            output_loc = configuration.get_catalogs_dir(
                'halos', simname=simname, halo_finder=halo_finder)

        uncut_catalog_fname = os.path.basename(uncut_catalog_fname)
        if uncut_catalog_fname[-3:] == '.gz':
            uncut_catalog_fname = uncut_catalog_fname[:-3]

        output_fname = uncut_catalog_fname + '.' + version_name + '.hdf5'
        output_full_fname = os.path.join(output_loc, output_fname)
        t = Table(catalog)
        print("Storing reduced halo catalog in the following location:\n" + 
            output_full_fname)
        t.write(output_full_fname, path='halos', overwrite=overwrite)

        f = h5py.File(output_full_fname)
        pickled_cuts_funcobj = pickle.dumps(cuts_funcobj, protocol = 0)
        f.attrs['halocat_exact_cuts'] = pickled_cuts_funcobj
        f.close()


    def full_fname_closest_raw_halocat_in_cache(
        self, simname, halo_finder, input_redshift):

        filename_list = self.full_fnames_all_raw_halocats_in_cache(simname, halo_finder)

        closest_fname = self.find_closest_raw_halocat(
            filename_list, input_redshift)
        if closest_fname == None:
            return None

        dirname = configuration.get_catalogs_dir('raw_halos', 
            simname=simname, halo_finder=halo_finder)
        output_full_fname = os.path.join(dirname, closest_fname)

        return output_full_fname


    def full_fnames_all_raw_halocats_in_cache(self, **kwargs):
        """ Method returns the filenames of all snapshots 
        in the Halotools cache directory that match the input specifications. 

        Parameters 
        ----------
        simname : string, optional
            Nickname of the simulation, e.g. `bolshoi`. 

        halo_finder : string, optional
            Nickname of the halo-finder, e.g. `rockstar`. 

        Returns 
        -------
        file_list : list 
            List of strings of all raw catalogs in the cache directory. 
        """
        
        def find_files(catalog_dirname):
            file_list = []
            for path, dirlist, filelist in os.walk(catalog_dirname):
                for name in fnmatch.filter(filelist,'*hlist_*'):
                    file_list.append(os.path.join(path,name))
            return file_list

        catalog_dirname = configuration.get_catalogs_dir('raw_halos', **kwargs)

        return find_files(catalog_dirname)


    def find_closest_raw_halocat(self, filename_list, input_redshift):
        """ Method searches the url where the ``simname`` halo catalogs are stored, 
        and returns the filename of the closest matching snapshot to ``input_redshift``. 

        Parameters 
        ----------
        filename_list : list of strings
            Each entry of the list must be a filename of the type generated by Rockstar. 

        input_redshift : float
            Redshift of the requested snapshot.

        Returns
        -------
        output_fname : string 
            Filename of the closest matching snapshot. 
        """

        if aph_len(filename_list)==0:
            return None

        # First create a list of floats storing the scale factors of each hlist file
        scale_factor_list = []
        for full_fname in filename_list:
            fname = os.path.basename(full_fname)
            scale_factor_substring = self.get_scale_factor_substring(fname)
            scale_factor = float(scale_factor_substring)
            scale_factor_list.append(scale_factor)
        scale_factor_list = np.array(scale_factor_list)

        # Now use the array utils module to determine 
        # which scale factor is the closest
        input_scale_factor = 1./(1. + input_redshift)
        idx_closest_catalog = find_idx_nearest_val(scale_factor_list, input_scale_factor)

        closest_scale_factor = scale_factor_list[idx_closest_catalog]

        output_fname = filename_list[idx_closest_catalog]

        return output_fname


    def get_scale_factor_substring(self, fname):
        """ Method extracts the portion of the Rockstar hlist fname 
        that contains the scale factor of the snapshot. 

        Parameters 
        ----------
        fname : string 
            Filename of the hlist. 

        Returns 
        -------
        scale_factor_substring : string 
            The substring specifying the scale factor of the snapshot. 

        Notes 
        -----
        Assumes that the relevant substring always immediately follows 
        the first incidence of an underscore, and terminates immediately 
        preceding the second decimal. These assumptions are valid for 
        all catalogs currently on the hipacc website, including 
        'bolshoi', 'bolshoi_bdm', 'consuelo', and 'multidark'. 

        Examples
        --------
        >>> catman = CatalogManager()
        >>> fname = 'hlist_0.06630.list.gz'
        >>> scale_factor_string = catman.get_scale_factor_substring(fname)

        """
        first_index = fname.index('_')+1
        last_index = fname.index('.', fname.index('.')+1)
        scale_factor_substring = fname[first_index:last_index]
        return scale_factor_substring

    def retrieve_available_raw_halocats(self, simname, halo_finder, file_pattern = '*hlist_*'):
        """ Method searches the appropriate web location and 
        returns a list of the filenames of all relevant 
        raw halo catalogs that are available for download. 

        that are available at the host url. 

        Parameters 
        ----------
        simname : string 
            Nickname of the simulation, e.g., `bolshoi` or `bolshoipl`. 

        halo_finder : string 
            Nickname of the halo-finder that generated the catalog, 
            e.g., `rockstar` or `bdm`. 

        file_pattern : string, optional 
            String used to filter the list of filenames at the url so that 
            only files matching the requested pattern are returned. 
            Syntax is the same as running a grep search on a regular expression, 
            e.g., file_pattern = `*my_filter*`. Default is set to `*hlist_*`.
            Should be set to an empty string if no filtering is desired.  

        Returns 
        -------
        output : list 
            List of strings of all raw halo catalogs available for download 
            for the requested simulation. 

        Notes 
        ----- 
        Method assumes that the first characters of any halo catalog filename are `hlist_`. 

        """
        key = simname + '_' + halo_finder
        if key in sim_defaults.raw_halocat_url.keys():
            url = sim_defaults.raw_halocat_url[key]
        else:
            raise KeyError("Input simname + halo-finder does not correspond to "
                "any of the catalogs stored in sim_defaults.raw_halocat_url")

        soup = BeautifulSoup(requests.get(url).text)
        expected_filename_prefix = 'hlist_'
        file_list = []
        for a in soup.find_all('a'):
            file_list.append(a['href'])

        if file_pattern != '':
            output = fnmatch.filter(file_list, file_pattern)
        else:
            output = file_list

        return output


    def update_list_of_previously_used_dirnames(self, 
        catalog_type, input_full_fname, input_simname):
        """ Method determines whether the input `dirname` is a new location 
        that has not been used before to store halo catalogs, and updates the 
        package memory as necessary.
        """
        dirname = os.path.dirname(input_full_fname)
        if not os.path.exists(dirname):
            warnings.warn("Cannot append a non-existent path %s \n"
                "to Halotools memory of preferred user cache directory locations")
            return

        cache_memory_loc = configuration.get_catalogs_dir(catalog_type)
        cache_memory_full_fname = os.path.join(cache_memory_loc, 
            sim_defaults.cache_memory_fname)
        # If the cache memory file does not already exist, 
        # issue a warning, create it, and add the header
        if not os.path.isfile(cache_memory_full_fname):
            warnings.warn("Creating the following file: \n %s \n "
                "This file is used to store Halotools' memory\n"
                "of user-preferred locations for data of type %s" % 
                (cache_memory_full_fname, catalog_type)
                )
            with open(cache_memory_full_fname, 'w') as f:
                header_line1 = (
                    "# This file lists all previously used locations storing "
                    "catalogs of the following type: "+catalog_type+"\n"
                    )
                header_line2 = ("# Halotools uses this file "
                    "to automatically detect a previously used disk\n")
                f.write(header_line1)
                f.write(header_line2)

        add_new_simloc = True
        with open(cache_memory_full_fname, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line[0] != '#':
                    existing_fname, existing_simname = line.strip()[0], line.strip()[1]
                    if input_full_fname == existing_fname:
                        add_new_simloc = False
                        if existing_simname != input_simname:
                            msg = ("There already exists filename:\n%s\n "
                                "The existing file pertains to the %s simulation,\n"
                                "but you requested the cache memory to update this file \n"
                                "to correspond to the %s simulation.\n" 
                                "Take care that this was intentional.")

                            warnings.warn(msg % (existing_fname, existing_simname, input_simname))
                else:
                    pass

        if add_new_simloc == True:
            with open(cache_memory_full_fname, 'a') as f:
                f.write(input_full_fname + '  ' + input_simname)


    def get_simulation_properties(self,simname):
        """ Return a few characteristics of the input simulation.

        Parameters 
        ----------
        simname : string 
            Specifies the simulation of interest, e.g., 'bolshoi'.

        Returns 
        -------
        Lbox : float 
            Box size in Mpc/h.

        particle_mass : float 
            Particle mass in Msun/h.

        softening : float 
            Softening length in kpc/h.

        """

        Lbox, particle_mass, softening = None, None, None

        if (simname=='bolshoi'):
            Lbox = 250.0
            particle_mass = 1.35e8
            softening = 1.0

        return Lbox, particle_mass, softening 


    def load_catalog(self,catalog_type,
        dirname=None,filename=None,
        download_yn=False,url=sim_defaults.aph_web_location):
        """ Use the astropy reader to load the halo or particle catalog into memory.

        Parameters 
        ----------
        dirname : string 
            Name of directory where filename is stored.

        filename : string 
            Name of file being loaded into memory. 

        download_yn : boolean, optional
            If set to True, and if filename is not already stored in the cache directory, 
            method will attempt to download the file from the provided url. If there is no corresponding 
            file at the input url, an exception will be raised.

        url : string 
            Web location from which to download the catalog if it is not present in the cache directory.

        Returns 
        -------
        catalog : object
            Data structure located at the input filename.

        """
        if filename==None:
            if catalog_type=='subhalos':
                filename = self.default_halo_catalog_filename
            elif catalog_type=='particles':
                filename = self.default_particle_catalog_filename
            else:
                raise KeyError("Must supply catalog_type to be either "
                    "'particles' or 'subhalos'")
        if dirname==None:
            if catalog_type=='subhalos':
                dirname = self.halo_catalog_dirname
            elif catalog_type=='particles':
                dirname = self.particle_catalog_dirname
            else:
                raise KeyError("Must supply catalog_type to be either "
                    "'particles' or 'subhalos'")

        if os.path.isfile(os.path.join(dirname,filename)):
            catalog = Table.read(os.path.join(dirname,filename),path='data')
        else:
            ### Requested filename is not in cache, and external download is not requested
            if download_yn==False:
                return None
            else:
                # Download one of the default catalogs hosted at Yale
                if filename==self.default_halo_catalog_filename:
                    catalog_type='subhalos'
                if filename==self.default_particle_catalog_filename:
                    catalog_type='particles'
                else:
                    raise IOError("Input filename does not match one of the provided default catalogs")
                ###
                remote_filename = os.path.join(url,filename)
                fileobj = urllib2.urlopen(remote_filename)
                output_directory = configuration.get_catalogs_dir(catalog_type)
                output_filename = os.path.join(output_directory,filename)
                output = open(output_filename,'wb')
                output.write(fileobj.read())
                output.close()
                hdulist = fits.open(output_filename)
                catalog = Table(hdulist[1].data)

        return catalog


    def download_all_default_catalogs(self):
        """ If not already in cache, 
        download default particle and halo catalogs from Yale website.
        """

        url = sim_defaults.aph_web_location

        ### Download halo catalogs
        catalog_type = 'subhalos'
        output_directory = configuration.get_catalogs_dir(catalog_type)
        filename = self.default_halo_catalog_filename
        remote_filename = os.path.join(url,filename)
        if not os.path.isfile(os.path.join(output_directory,filename)):
            warnings.warn("Downloading default halo catalog")
            fileobj = urllib2.urlopen(remote_filename)
            output_filename = os.path.join(output_directory,filename)
            output = open(output_filename,'wb')
            output.write(fileobj.read())
            output.close()

        ### Download particle catalogs
        catalog_type = 'particles'
        output_directory = configuration.get_catalogs_dir(catalog_type)
        filename = self.default_particle_catalog_filename
        remote_filename = os.path.join(url,filename)
        if not os.path.isfile(os.path.join(output_directory,filename)):
            warnings.warn("Downloading default particle catalog")
            fileobj = urllib2.urlopen(remote_filename)
            output_filename = os.path.join(output_directory,filename)
            output = open(output_filename,'wb')
            output.write(fileobj.read())
            output.close()

    def clear_cache_memory(self, catalog_type):

        cache_memory_loc = configuration.get_catalogs_dir(catalog_type)
        cache_memory_fname = catalog_type+'_'+sim_defaults.cache_memory_fname
        cache_memory_full_fname = os.path.join(cache_memory_loc, cache_memory_fname)
        if os.path.exists(cache_memory_full_fname):
            os.system("rm "+cache_memory_full_fname)




###################################################################################################
class RockstarReader(object):

    def __init__(self, input_fname, **kwargs):

        if not os.path.isfile(input_fname):
            raise IOError("Input filename %s is not a file" % input_fname)
        self.fname = input_fname
        self._uncompress_ascii()

        if 'simobj' in kwargs.keys():
            simobj = kwargs['simobj']
            if not isistance(simobj, sim_specs.HaloCatSpecs):
                raise IOError("Input catalog object %s "
                    "must be a subclass of HaloCatSpecs" % simobj.__name__)
            self.simobj = simobj
        elif ('simname' in kwargs.keys()) & ('halo_finder' in kwargs.keys()):
            self.simname = kwargs['simname']
            self.halo_finder = kwargs['halo_finder']
            self.simobj = self.get_simobj(self.simname, self.halo_finder)
        else:
            raise IOError("Must either pass `simobj` keyword "
                "to the `RockstarReader` constructor,\n"
                "or both `simname` and `halo_finder` keywords")


    def default_mpeak_cut(self, x):

        return x['mpeak'] > ( 
            self.simobj.simulation.particle_mass*
            sim_defaults.Num_ptcl_requirement)


    def get_simobj(self, simname, halo_finder):
        """ Find and return the class instance that will be used to 
        convert raw ASCII halo catalog data into a reduced binary.

        Parameters 
        ----------
        simname : string 
            Nickname of the simulation, e.g., `bolshoi`. 

        halo_finder : string 
            Nickname of the halo-finder that generated the catalog, 
            e.g., `rockstar`. 

        Returns 
        -------
        simobj : object 
            Class instance of `~halotools.sim_manager.sim_specs.HaloCatSpecs`. 
            Used to read ascii data in the specific format of the 
            `simname` simulation and `halo_finder` halos. 
        """
        class_list = sim_specs.__all__
        parent_class = sim_specs.HaloCatSpecs

        supported_halocat_classes = []
        for clname in class_list:
            clobj = getattr(sim_specs, clname)
            if (issubclass(clobj, parent_class)) & (clobj.__name__ != parent_class.__name__):
                supported_halocat_classes.append(clobj())

        simobj = None
        for sim in supported_halocat_classes:
            if (sim.simname == simname) & (sim.halo_finder == halo_finder):
                simobj = sim
        if simobj==None:
            print("No simulation class found for %s simulation and %s halo-finder\n"
                "If you want to use Halotools to convert a raw halo catalog into a binary, \n"
                "you must either use an existing reader class or write your own\n" 
                % simname, halo_finder)
            return None
        else:
            return simobj


    def file_len(self):
        """ Compute the number of all rows in fname

        Parameters 
        ----------
        fname : string 

        Returns 
        -------
        Nrows : int
     
        """
        with open(self.fname) as f:
            for i, l in enumerate(f):
                pass
        Nrows = i + 1
        return Nrows

    def header_len(self,header_char='#'):
        """ Compute the number of header rows in fname. 

        Parameters 
        ----------
        fname : string 

        header_char : str, optional
            string to be interpreted as a header line

        Returns 
        -------
        Nheader : int

        Notes 
        -----
        All empty lines that appear in header 
        will be included in the count. 

        """
        Nheader = 0
        with open(self.fname) as f:
            for i, l in enumerate(f):
                if ( (l[0:len(header_char)]==header_char) or (l=="\n") ):
                    Nheader += 1
                else:
                    break

        return Nheader

    def get_header(self, Nrows_header_total=None):
        """ Return the header as a list of strings, 
        one entry per header row. 

        Parameters 
        ----------
        fname : string 

        Nrows_header_total :  int, optional
            If the total number of header rows is not known in advance, 
            method will call `header_len` to determine Nrows_header_total. 

        Notes 
        -----
        Empty lines will be included in the returned header. 

        """

        if Nrows_header_total==None:
            Nrows_header_total = self.header_len(self.fname)

        print("Reading the first %i lines of the ascii file" % Nrows_header_total)

        output = []
        with open(self.fname) as f:
            for i in range(Nrows_header_total):
                line = f.readline().strip()
                output.append(line)

        return output

    def _uncompress_ascii(self):
        """ If the input fname has file extension `.gz`, 
        then the method uses `gunzip` to decompress it, 
        and returns the input fname truncated to exclude 
        the `.gz` extension. If the input fname does not 
        end in `.gz`, method does nothing besides return 
        the input fname. 
        """
        if self.fname[-3:]=='.gz':
            print("...uncompressing ASCII data")
            os.system("gunzip "+self.fname)
            self.fname = self.fname[:-3]
        else:
            pass

    def _compress_ascii(self):
        """ Recompresses the halo catalog ascii data, 
        and returns the input filename appended with `.gz`.  
        """
        if self.fname[-3:]!='.gz':
            print("...compressing ASCII data")
            os.system("gzip "+self.fname)
            self.fname = self.fname + '.gz'
        else:
            pass


    def read_halocat(self, **kwargs):
        """ Reads fname in chunks and returns a structured array
        after applying cuts.

        Parameters 
        ----------
        cuts_funcobj : function object, optional keyword argument
            Function used to determine whether a row of the raw 
            halo catalog is included in the reduced binary. 
            Input of the `cut` function must be a structured array 
            with some subset of the field names of the 
            halo catalog dtype. Output of the `cut` function must 
            be a boolean array of length equal to the length of the 
            input structured array. Default is to make a cut on 
            `mpeak` at 300 particles, using `default_mpeak_cut` method. 

        nchunks : int, optional keyword argument
            `read_halocat` reads and processes ascii 
            in chunks at a time, both to improve performance and 
            so that the entire raw halo catalog need not fit in memory 
            in order to process it. The total number of chunks to use 
            can be specified with the `nchunks` argument. Default is 1000. 

        """

        if 'cuts_funcobj' in kwargs.keys():
            cuts_funcobj = kwargs['cuts_funcobj']
        else:
            cuts_funcobj = self.default_mpeak_cut

        if 'nchunks' in kwargs.keys():
            Nchunks = kwargs['nchunks']
        else:
            Nchunks = 1000

        dt = self.simobj.halocat_column_info

        file_length = self.file_len()
        chunksize = file_length / Nchunks
        chunk = []
        container = []
        for linenum, line in enumerate(open(self.fname)):
            if line[0] == '#':
                pass
            else:
                parsed_line = line.strip().split()
                chunk.append(tuple(parsed_line))  
        
            if (linenum % chunksize == 0) & (linenum > 0):        
                a = np.array(chunk, dtype = dt)
                container.append(a[cuts_funcobj(a)])
                chunk = []
    #Now for the final chunk missed by the above syntax
        a = np.array(chunk, dtype = dt)
        container.append(a[cuts_funcobj(a)])

    # Bundle up all array chunks into a single array
        for chunk in container:
            try:
                output = np.append(output, chunk)
            except NameError:
                output = np.array(chunk) 
                
        return output





###################################################################################################

