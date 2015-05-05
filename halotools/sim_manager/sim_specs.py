# -*- coding: utf-8 -*-

import numpy as np

from . import sim_defaults 

from abc import ABCMeta, abstractmethod, abstractproperty
from astropy.extern import six

from astropy import cosmology

__all__ = (
    ['SimulationSpecs', 'Bolshoi', 'BolshoiPl', 'MultiDark', 'Consuelo', 
    'HaloCat', 'BolshoiRockstar', 'BolshoiPlRockstar', 
    'BolshoiBdm', 'MultiDarkRockstar', 'ConsuleoRockstar']
    )




######################################################
########## Simulation classes appear below ########## 
######################################################

@six.add_metaclass(ABCMeta)
class SimulationSpecs(object):
    """ Abstract base class for any object used as a container for 
    simulation specs. 
    """

    def __init__(self, simname):
        self.simname = simname

    @abstractproperty
    def Lbox(self):
        """ Size of the simulated box in Mpc/h. 
        """
        pass

    @abstractproperty
    def particle_mass(self):
        """ Mass of the dark matter particles in Msun/h. 
        """
        pass

    @abstractproperty
    def softening_length(self):
        """ Softening scale of the particle interactions in kpc/h. 
        """
        pass

    @abstractproperty
    def cosmology(self):
        """ Astropy cosmology instance giving the 
        cosmological parameters with which the simulation was run. 
        """
        pass

    @abstractproperty
    def cosmology(self):
        """ Astropy cosmology instance giving the 
        cosmological parameters with which the simulation was run. 
        """
        pass


class Bolshoi(SimulationSpecs):

    def __init__(self):
        super(Bolshoi, self).__init__('bolshoi')

    @property
    def Lbox(self):
        return 250.0

    @property
    def particle_mass(self):
        return 1.35e8

    @property
    def softening_length(self):
        return 1.0

    @property
    def cosmology(self):
        return cosmology.WMAP5

class BolshoiPl(SimulationSpecs):

    def __init__(self):
        super(BolshoiPl, self).__init__('bolshoipl')

    @property
    def Lbox(self):
        return 250.0

    @property
    def particle_mass(self):
        return 1.35e8

    @property
    def softening_length(self):
        return 1.0

    @property
    def cosmology(self):
        return cosmology.Planck13

class MultiDark(SimulationSpecs):

    def __init__(self):
        super(MultiDark, self).__init__('multidark')

    @property
    def Lbox(self):
        return 1000.0

    @property
    def particle_mass(self):
        return 8.7e9

    @property
    def softening_length(self):
        return 7.0

    @property
    def cosmology(self):
        return cosmology.WMAP5

class Consuelo(SimulationSpecs):

    def __init__(self):
        super(Consuelo, self).__init__('consuelo')

    @property
    def Lbox(self):
        return 400.0

    @property
    def particle_mass(self):
        return 4.e8

    @property
    def softening_length(self):
        return 4.0

    @property
    def cosmology(self):
        return cosmology.WMAP5

######################################################
########## Halo-finder classes appear below ########## 
######################################################

@six.add_metaclass(ABCMeta)
class HaloCat(object):

    def __init__(self, simobj, halo_finder):
        self.simulation = simobj
        self.simname = self.simulation.simname

        self.halo_finder = halo_finder

    @abstractproperty
    def halocat_column_info(self):
        pass

    @abstractproperty
    def halocat_fname_pattern(self):
        pass

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
        Assumes that the first character of the relevant substring 
        is the one immediately following the first incidence of an underscore, 
        and final character is the one immediately preceding the second decimal. 
        These assumptions are valid for all catalogs currently on the hipacc website, 
        including `bolshoi`, `bolshoi_bdm`, `consuelo`, and `multidark`. 

        """
        first_index = fname.index('_')+1
        last_index = fname.index('.', fname.index('.')+1)
        scale_factor_substring = fname[first_index:last_index]
        return scale_factor_substring

class BolshoiRockstar(HaloCat):

    def __init__(self):

        bolshoi = Bolshoi()
        super(BolshoiRockstar, self).__init__(bolshoi, 'rockstar')

    @property 
    def raw_halocat_web_location(self):
        return 'http://www.slac.stanford.edu/~behroozi/Bolshoi_Catalogs/'

    @property 
    def halocat_fname_pattern(self):
        return '*hlist_*'

    @property 
    def halocat_column_info(self):

        dt = np.dtype([
            ('scale', 'f4'), 
            ('haloid', 'i8'), 
            ('scale_desc', 'f4'), 
            ('haloid_desc', 'i8'), 
            ('num_prog', 'i4'), 
            ('pid', 'i8'), 
            ('upid', 'i8'), 
            ('pid_desc', 'i8'), 
            ('phantom', 'i4'), 
            ('mvir_sam', 'f4'), 
            ('mvir', 'f4'), 
            ('rvir', 'f4'), 
            ('rs', 'f4'), 
            ('vrms', 'f4'), 
            ('mmp', 'i4'), 
            ('scale_lastmm', 'f4'), 
            ('vmax', 'f4'), 
            ('x', 'f4'), 
            ('y', 'f4'), 
            ('z', 'f4'), 
            ('vx', 'f4'), 
            ('vy', 'f4'), 
            ('vz', 'f4'), 
            ('jx', 'f4'), 
            ('jy', 'f4'), 
            ('jz', 'f4'), 
            ('spin', 'f4'), 
            ('haloid_breadth_first', 'i8'), 
            ('haloid_depth_first', 'i8'), 
            ('haloid_tree_root', 'i8'), 
            ('haloid_orig', 'i8'), 
            ('snap_num', 'i4'), 
            ('haloid_next_coprog_depthfirst', 'i8'), 
            ('haloid_last_prog_depthfirst', 'i8'), 
            ('rs_klypin', 'f4'), 
            ('mvir_all', 'f4'), 
            ('m200b', 'f4'), 
            ('m200c', 'f4'), 
            ('m500c', 'f4'), 
            ('m2500c', 'f4'), 
            ('xoff', 'f4'), 
            ('voff', 'f4'), 
            ('spin_bullock', 'f4'), 
            ('b_to_a', 'f4'), 
            ('c_to_a', 'f4'), 
            ('axisA_x', 'f4'), 
            ('axisA_y', 'f4'), 
            ('axisA_z', 'f4'), 
            ('b_to_a_500c', 'f4'), 
            ('c_to_a_500c', 'f4'), 
            ('axisA_x_500c', 'f4'), 
            ('axisA_y_500c', 'f4'), 
            ('axisA_z_500c', 'f4'), 
            ('t_by_u', 'f4'), 
            ('mass_pe_behroozi', 'f4'), 
            ('mass_pe_diemer', 'f4'), 
            ('macc', 'f4'), 
            ('mpeak', 'f4'), 
            ('vacc', 'f4'), 
            ('vpeak', 'f4'), 
            ('halfmass_scale', 'f4'), 
            ('dmvir_dt_inst', 'f4'), 
            ('dmvir_dt_100myr', 'f4'), 
            ('dmvir_dt_tdyn', 'f4'), 
            ('dmvir_dt_2dtyn', 'f4'), 
            ('dmvir_dt_mpeak', 'f4'), 
            ('scale_mpeak', 'f4'), 
            ('scale_lastacc', 'f4'), 
            ('scale_firstacc', 'f4'), 
            ('mvir_firstacc', 'f4'), 
            ('vmax_firstacc', 'f4'), 
            ('vmax_mpeak', 'f4')
            ])

        return dt

class BolshoiPlRockstar(HaloCat):

    def __init__(self):

        bolshoiPl = BolshoiPl()
        super(BolshoiPlRockstar, self).__init__(bolshoiPl, 'rockstar')

    @property 
    def raw_halocat_web_location(self):
        return 'http://www.slac.stanford.edu/~behroozi/BPlanck_Hlists/'

    @property 
    def halocat_fname_pattern(self):
        return '*hlist_*'

    @property 
    def halocat_column_info(self):

        dt = np.dtype([
            ('scale', 'f4'), 
            ('haloid', 'i8'), 
            ('scale_desc', 'f4'), 
            ('haloid_desc', 'i8'), 
            ('num_prog', 'i4'), 
            ('pid', 'i8'), 
            ('upid', 'i8'), 
            ('pid_desc', 'i8'), 
            ('phantom', 'i4'), 
            ('mvir_sam', 'f4'), 
            ('mvir', 'f4'), 
            ('rvir', 'f4'), 
            ('rs', 'f4'), 
            ('vrms', 'f4'), 
            ('mmp', 'i4'), 
            ('scale_lastmm', 'f4'), 
            ('vmax', 'f4'), 
            ('x', 'f4'), 
            ('y', 'f4'), 
            ('z', 'f4'), 
            ('vx', 'f4'), 
            ('vy', 'f4'), 
            ('vz', 'f4'), 
            ('jx', 'f4'), 
            ('jy', 'f4'), 
            ('jz', 'f4'), 
            ('spin', 'f4'), 
            ('haloid_breadth_first', 'i8'), 
            ('haloid_depth_first', 'i8'), 
            ('haloid_tree_root', 'i8'), 
            ('haloid_orig', 'i8'), 
            ('snap_num', 'i4'), 
            ('haloid_next_coprog_depthfirst', 'i8'), 
            ('haloid_last_prog_depthfirst', 'i8'), 
            ('rs_klypin', 'f4'), 
            ('mvir_all', 'f4'), 
            ('m200b', 'f4'), 
            ('m200c', 'f4'), 
            ('m500c', 'f4'), 
            ('m2500c', 'f4'), 
            ('xoff', 'f4'), 
            ('voff', 'f4'), 
            ('spin_bullock', 'f4'), 
            ('b_to_a', 'f4'), 
            ('c_to_a', 'f4'), 
            ('axisA_x', 'f4'), 
            ('axisA_y', 'f4'), 
            ('axisA_z', 'f4'), 
            ('b_to_a_500c', 'f4'), 
            ('c_to_a_500c', 'f4'), 
            ('axisA_x_500c', 'f4'), 
            ('axisA_y_500c', 'f4'), 
            ('axisA_z_500c', 'f4'), 
            ('t_by_u', 'f4'), 
            ('mass_pe_behroozi', 'f4'), 
            ('mass_pe_diemer', 'f4'), 
            ('macc', 'f4'), 
            ('mpeak', 'f4'), 
            ('vacc', 'f4'), 
            ('vpeak', 'f4'), 
            ('halfmass_scale', 'f4'), 
            ('dmvir_dt_inst', 'f4'), 
            ('dmvir_dt_100myr', 'f4'), 
            ('dmvir_dt_tdyn', 'f4'), 
            ('dmvir_dt_2dtyn', 'f4'), 
            ('dmvir_dt_mpeak', 'f4'), 
            ('scale_mpeak', 'f4'), 
            ('scale_lastacc', 'f4'), 
            ('scale_firstacc', 'f4'), 
            ('mvir_firstacc', 'f4'), 
            ('vmax_firstacc', 'f4'), 
            ('vmax_mpeak', 'f4')
            ])

        return dt

class BolshoiBdm(HaloCat):

    def __init__(self):

        bolshoi = Bolshoi()
        super(BolshoiBdm, self).__init__(bolshoi, 'bdm')

    @property 
    def raw_halocat_web_location(self):
        return 'http://www.slac.stanford.edu/~behroozi/Bolshoi_Catalogs_BDM/'

    @property 
    def halocat_fname_pattern(self):
        return '*hlist_*'

    @property 
    def halocat_column_info(self):
        d = {
        'scale': (0, 'int'), 
        'haloid': (1, 'int'), 
        'scale_desc': (2, 'int'), 
        'haloid_desc': (3, 'int'), 
        'num_prog': (4, 'int'), 
        'pid': (5, 'int'), 
        'upid': (6, 'int'), 
        'pid_desc': (7, 'int'), 
        'phantom': (8, 'int'), 
        'mvir_sam': (9, 'float'), 
        'mvir': (10, 'float'),
        'rvir': (11, 'float'), 
        'rs': (12, 'float'), 
        'vrms': (13, 'float'), 
        'mmp': (14, 'int'), 
        'scale_lastmm': (15, 'float'), 
        'vmax': (16, 'float'), 
        'x': (17, 'float'),  
        'y': (18, 'float'),  
        'z': (19, 'float'),  
        'vx': (20, 'float'),  
        'vy': (21, 'float'),  
        'vz': (22, 'float'),  
        'jx': (23, 'float'),
        'jy': (24, 'float'),
        'jz': (25, 'float'),
        'spin': (26, 'float'), 
        'haloid_breadth_first': (27, 'int'),
        'haloid_depth_first': (28, 'int'),
        'haloid_tree_root': (29, 'int'),
        'haloid_orig': (30, 'int'),
        'snap_num': (31, 'int'),
        'haloid_next_coprog_depthfirst': (32, 'int'), 
        'haloid_last_prog_depthfirst': (33, 'int'), 
        'xoff': (34, 'float'), 
        '2K/Ep-1': (35, 'float'), 
        'Rrms': (36, 'float'), 
        'b_to_a': (37, 'float'), 
        'c_to_a': (38, 'float'), 
        'axisA_x': (39, 'float'), 
        'axisA_y': (40, 'float'), 
        'axisA_z': (41, 'float'), 
        'macc': (42, 'float'), 
        'mpeak': (43, 'float'), 
        'vacc': (44, 'float'), 
        'vpeak': (45, 'float'), 
        }

        return d

class MultiDarkRockstar(HaloCat):

    def __init__(self):

        multidark = MultiDark()
        super(MultiDarkRockstar, self).__init__(multidark, 'rockstar')

    @property 
    def raw_halocat_web_location(self):
        return 'http://slac.stanford.edu/~behroozi/MultiDark_Hlists_Rockstar/'

    @property 
    def halocat_fname_pattern(self):
        return '*hlist_*'

    @property 
    def halocat_column_info(self):
        d = {
        'scale': (0, 'int'), 
        'haloid': (1, 'int'), 
        'scale_desc': (2, 'int'), 
        'haloid_desc': (3, 'int'), 
        'num_prog': (4, 'int'), 
        'pid': (5, 'int'), 
        'upid': (6, 'int'), 
        'pid_desc': (7, 'int'), 
        'phantom': (8, 'int'), 
        'mvir_sam': (9, 'float'), 
        'mvir': (10, 'float'),
        'rvir': (11, 'float'), 
        'rs': (12, 'float'), 
        'vrms': (13, 'float'), 
        'mmp': (14, 'int'), 
        'scale_lastmm': (15, 'float'), 
        'vmax': (16, 'float'), 
        'x': (17, 'float'),  
        'y': (18, 'float'),  
        'z': (19, 'float'),  
        'vx': (20, 'float'),  
        'vy': (21, 'float'),  
        'vz': (22, 'float'),  
        'jx': (23, 'float'),
        'jy': (24, 'float'),
        'jz': (25, 'float'),
        'spin': (26, 'float'), 
        'haloid_breadth_first': (27, 'int'),
        'haloid_depth_first': (28, 'int'),
        'haloid_tree_root': (29, 'int'),
        'haloid_orig': (30, 'int'),
        'snap_num': (31, 'int'),
        'haloid_next_coprog_depthfirst': (32, 'int'), 
        'haloid_last_prog_depthfirst': (33, 'int'), 
        'rs_klypin': (34, 'float'), 
        'mvir_all': (35, 'float'), 
        'm200b': (36, 'float'), 
        'm200c': (37, 'float'), 
        'm500c': (38 ,'float'), 
        'm2500c': (39, 'float'), 
        'xoff': (40, 'float'), 
        'voff': (41, 'float'), 
        'spin_bullock': (42, 'float'), 
        'b_to_a': (43, 'float'), 
        'c_to_a': (44, 'float'), 
        'axisA_x': (45, 'float'), 
        'axisA_y': (46, 'float'), 
        'axisA_z': (47, 'float'), 
        't_by_u': (48, 'float'), 
        'macc': (49, 'float'), 
        'mpeak': (50, 'float'), 
        'vacc': (51, 'float'), 
        'vpeak': (52, 'float'), 
        'halfmass_scale': (53, 'float'), 
        'dmvir_dt_inst': (54, 'float'), 
        'dmvir_dt_100myr': (55, 'float'), 
        'dmvir_dt_tdyn': (56, 'float'), 
        'dmvir_dt_2dtyn': (57, 'float'), 
        }

        return d



class ConsuleoRockstar(HaloCat):

    def __init__(self):

        consuelo = Consuelo()
        super(ConsuleoRockstar, self).__init__(consuelo, 'rockstar')

    @property 
    def web_location(self):
        return 'http://www.slac.stanford.edu/~behroozi/Consuelo_Catalogs/'

    @property 
    def halocat_fname_pattern(self):
        return '*hlist_*'

    @property 
    def halocat_column_info(self):
        d = {
        'scale': (0, 'int'), 
        'haloid': (1, 'int'), 
        'scale_desc': (2, 'int'), 
        'haloid_desc': (3, 'int'), 
        'num_prog': (4, 'int'), 
        'pid': (5, 'int'), 
        'upid': (6, 'int'), 
        'pid_desc': (7, 'int'), 
        'phantom': (8, 'int'), 
        'mvir_sam': (9, 'float'), 
        'mvir': (10, 'float'),
        'rvir': (11, 'float'), 
        'rs': (12, 'float'), 
        'vrms': (13, 'float'), 
        'mmp': (14, 'int'), 
        'scale_lastmm': (15, 'float'), 
        'vmax': (16, 'float'), 
        'x': (17, 'float'),  
        'y': (18, 'float'),  
        'z': (19, 'float'),  
        'vx': (20, 'float'),  
        'vy': (21, 'float'),  
        'vz': (22, 'float'),  
        'jx': (23, 'float'),
        'jy': (24, 'float'),
        'jz': (25, 'float'),
        'spin': (26, 'float'), 
        'haloid_breadth_first': (27, 'int'),
        'haloid_depth_first': (28, 'int'),
        'haloid_tree_root': (29, 'int'),
        'haloid_orig': (30, 'int'),
        'snap_num': (31, 'int'),
        'haloid_next_coprog_depthfirst': (32, 'int'), 
        'haloid_last_prog_depthfirst': (33, 'int'), 
        'rs_klypin': (34, 'float'), 
        'mvir_all': (35, 'float'), 
        'm200b': (36, 'float'), 
        'm200c': (37, 'float'), 
        'm500c': (38 ,'float'), 
        'm2500c': (39, 'float'), 
        'xoff': (40, 'float'), 
        'voff': (41, 'float'), 
        'spin_bullock': (42, 'float'), 
        'b_to_a': (43, 'float'), 
        'c_to_a': (44, 'float'), 
        'axisA_x': (45, 'float'), 
        'axisA_y': (46, 'float'), 
        'axisA_z': (47, 'float'), 
        't_by_u': (48, 'float'), 
        'macc': (49, 'float'), 
        'mpeak': (50, 'float'), 
        'vacc': (51, 'float'), 
        'vpeak': (52, 'float'), 
        'scale_halfmass': (53, 'float')
        }

        return d






























