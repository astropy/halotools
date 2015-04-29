# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod, abstractproperty
from astropy.extern import six

from astropy import cosmology

__all__ = ['SimulationSpecs', 'Bolshoi']


######################################################
########## Simulation classes appear below ########## 
######################################################

@six.add_metaclass(ABCMeta)
class SimulationSpecs(object):
	""" Abstract base class for any object used as a container for 
	simulation specs. 
	"""

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
class HaloCatSpecs(object):

	def __init__(self, simobj):
		self.simulation = simobj

	@abstractproperty
	def halocat_column_info(self):
		pass


class BolshoiRockstar(HaloCatSpecs):

	def __init__(self):

		bolshoi = Bolshoi()
		super(BolshoiRockstar, self).__init__(bolshoi)

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
		'b_to_a_500c': (48, 'float'), 
		'c_to_a_500c': (49, 'float'), 
		'axisA_x_500c': (50, 'float'), 
		'axisA_y_500c': (51, 'float'), 
		'axisA_z_500c': (52, 'float'), 
		't_by_u': (53, 'float'), 
		'mass_pe_behroozi': (54, 'float'), 
		'mass_pe_diemer': (55, 'float'), 
		'macc': (56, 'float'), 
		'mpeak': (57, 'float'), 
		'vacc': (58, 'float'), 
		'vpeak': (59, 'float'), 
		'halfmass_scale': (60, 'float'), 
		'dmvir_dt_inst': (61, 'float'), 
		'dmvir_dt_100myr': (62, 'float'), 
		'dmvir_dt_tdyn': (63, 'float'), 
		'dmvir_dt_2dtyn': (64, 'float'), 
		'dmvir_dt_mpeak': (65, 'float'), 
		'scale_mpeak': (66, 'float'), 
		'scale_lastacc': (67, 'float'), 
		'scale_firstacc': (68, 'float'), 
		'mvir_firstacc': (69, 'float'), 
		'vmax_firstacc': (70, 'float'), 
		'vmax_mpeak': (71, 'float')
		}

		return d

class BolshoiPlRockstar(HaloCatSpecs):

	def __init__(self):

		bolshoiPl = BolshoiPl()
		super(BolshoiPlRockstar, self).__init__(bolshoiPl)

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
		'b_to_a_500c': (48, 'float'), 
		'c_to_a_500c': (49, 'float'), 
		'axisA_x_500c': (50, 'float'), 
		'axisA_y_500c': (51, 'float'), 
		'axisA_z_500c': (52, 'float'), 
		't_by_u': (53, 'float'), 
		'mass_pe_behroozi': (54, 'float'), 
		'mass_pe_diemer': (55, 'float'), 
		'macc': (56, 'float'), 
		'mpeak': (57, 'float'), 
		'vacc': (58, 'float'), 
		'vpeak': (59, 'float'), 
		'halfmass_scale': (60, 'float'), 
		'dmvir_dt_inst': (61, 'float'), 
		'dmvir_dt_100myr': (62, 'float'), 
		'dmvir_dt_tdyn': (63, 'float'), 
		'dmvir_dt_2dtyn': (64, 'float'), 
		'dmvir_dt_mpeak': (65, 'float'), 
		'scale_mpeak': (66, 'float'), 
		'scale_lastacc': (67, 'float'), 
		'scale_firstacc': (68, 'float'), 
		'mvir_firstacc': (69, 'float'), 
		'vmax_firstacc': (70, 'float'), 
		'vmax_mpeak': (71, 'float')
		}

		return d

class MultiDarkRockstar(HaloCatSpecs):

	def __init__(self):

		multidark = MultiDark()
		super(MultiDarkRockstar, self).__init__(multidark)


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



class ConsuleoRockstar(HaloCatSpecs):

	def __init__(self):

		consuelo = Consuelo()
		super(ConsuleoRockstar, self).__init__(consuelo)


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






























