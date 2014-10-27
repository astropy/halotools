__all__ = ['fast3tree', 'get_distance']
import os
import numpy as np
import numpy.ctypeslib as C
import warnings

_ptr_dtype = np.uint64
_ptr_ctype = C.ctypes.c_uint64
_float_dtype = np.float32
_float_ctype = C.ctypes.c_float

def _load_fast3tree_lib(dim):
    mytype = np.dtype([('idx', np.int64), ('pos', _float_dtype, dim)], align=True)
    mytype_ctype =  C.ndpointer(mytype, ndim=1, flags='C,A')
    center_ctype = C.ndpointer(dtype=_float_dtype, ndim=1, shape=(dim,), flags='C,A')
    box_ctype = C.ndpointer(dtype=_float_dtype, ndim=2, shape=(2,dim), flags='C,A')
    tree_ptr_ptr = C.ndpointer(dtype=_ptr_dtype, ndim=1, shape=(1,), flags='C,A')

    c_lib = C.load_library('fast3tree_%dd'%(dim), __path__[0])

    c_lib.fast3tree_init.restype = _ptr_ctype
    c_lib.fast3tree_init.argtypes = [C.ctypes.c_int64, mytype_ctype]

    c_lib.fast3tree_rebuild.restype = None
    c_lib.fast3tree_rebuild.argtypes = [_ptr_ctype, C.ctypes.c_int64, mytype_ctype]

    c_lib.fast3tree_maxmin_rebuild.restype = None
    c_lib.fast3tree_maxmin_rebuild.argtypes = [_ptr_ctype]

    c_lib.fast3tree_free.restype = None
    c_lib.fast3tree_free.argtypes = [tree_ptr_ptr]

    c_lib.fast3tree_results_init.restype = _ptr_ctype
    c_lib.fast3tree_results_init.argtypes = None

    c_lib.fast3tree_find_sphere.restype = None
    c_lib.fast3tree_find_sphere.argtypes = [_ptr_ctype, _ptr_ctype, center_ctype, _float_ctype]

    c_lib.fast3tree_find_sphere_periodic.restype = C.ctypes.c_int
    c_lib.fast3tree_find_sphere_periodic.argtypes = [_ptr_ctype, _ptr_ctype, center_ctype, _float_ctype]

    c_lib.fast3tree_find_inside_of_box.restype = None
    c_lib.fast3tree_find_inside_of_box.argtypes = [_ptr_ctype, _ptr_ctype, box_ctype]

    c_lib.fast3tree_find_outside_of_box.restype = None
    c_lib.fast3tree_find_outside_of_box.argtypes = [_ptr_ctype, _ptr_ctype, box_ctype]

    c_lib.fast3tree_results_clear.restype = None
    c_lib.fast3tree_results_clear.argtypes = [_ptr_ctype]

    c_lib.fast3tree_results_free.restype = None
    c_lib.fast3tree_results_free.argtypes = [_ptr_ctype]

    c_lib.fast3tree_set_minmax.restype = None
    c_lib.fast3tree_set_minmax.argtypes = [_ptr_ctype, _float_ctype, _float_ctype]

    c_lib.fast3tree_find_next_closest_distance.restype = _float_ctype
    c_lib.fast3tree_find_next_closest_distance.argtypes = [_ptr_ctype, _ptr_ctype, center_ctype]

    return c_lib, mytype

_c_libs_dict = {}
for _i in filter(lambda s: s.endswith('d.so') and s.startswith('fast3tree_'), os.listdir(__path__[0])):
    _c_libs_dict[int(_i[10:-4])] = _load_fast3tree_lib(int(_i[10:-4]))
if len(_c_libs_dict) == 0:
    raise ImportError('Cannot find any fast3tree library.')

_results_dtype =  np.dtype([ \
        ('num_points', np.int64), ('num_allocated_points', np.int64), 
        ('points', np.uint64)], align=True)

def _read_from_address(ptr, dtype, count):
    return np.frombuffer(np.core.multiarray.int_asbuffer(\
            long(ptr), np.dtype(dtype).itemsize*count), dtype, count=count)

def get_distance(center, pos, box_size=-1):
    pos = np.asarray(pos)
    pos_dtype = pos.dtype.type
    d = pos - np.asarray(center, pos_dtype)
    if box_size > 0:
        box_size = pos_dtype(box_size)
        half_box_size = pos_dtype(box_size*0.5)
        d[d >  half_box_size] -= box_size
        d[d < -half_box_size] += box_size
    return np.sqrt(np.sum(d*d, axis=1))

# define fast3tree class
class fast3tree:
    def __init__(self, data, raw=False):
        '''
        Initialize a fast3tree from a list of points.
        Please call fast3tree with the `with` statment to ensure memory safe.
        
            with fast3tree(data) as tree:
        
        Parameters
        ----------
        data : array_like
            data to build the tree. must be a 2-d array.

        Member functions
        ----------------
        rebuild_boundaries()
        set_boundaries(Min, Max)
        query_nearest_distance(center)
        query_radius(center, r)
        query_box(corner1, corner2)
        '''
        self._dim = None
        if raw:
            self._set_dim(data.dtype[1].shape[0])
            if data.dtype != self._type:
                raise ValueError("raw data not in correct format.")
            self.data = data
        else:
            self._copy_data(data)
        self._tree_ptr =  self._lib.fast3tree_init( \
                np.int64(self.data.shape[0]), self.data)
        self._res_ptr = self._lib.fast3tree_results_init()
        self._check_opened_by_with = self._check_opened_by_with_warn

    def __enter__(self):
        self._check_opened_by_with = self._check_opened_by_with_pass
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.free()

    def _set_dim(self, dim):
        if dim not in _c_libs_dict:
            raise ValueError('data must have the last dim = %s.'%(\
                    ', '.join(map(str, _c_libs_dict.keys()))))
        self._dim = dim
        self._lib, self._type = _c_libs_dict[dim]

    def _copy_data(self, data):
        data = np.asarray(data)
        s = data.shape
        if len(s) != 2:
            raise ValueError('data must be a 2-d array.')
        if self._dim is None:
            self._set_dim(s[1])
        elif s[1] != self._dim:
            raise ValueError('data must have the last dim = %d.'%self._dim)

        self.data = np.empty(s[0], self._type)
        self.data['idx'] = np.arange(s[0], dtype=np.int64)
        self.data['pos'] = data

    def _check_opened_by_with_warn(self):
        warnings.warn("Please use `with` statment to open a fast3tree object.")

    def _check_opened_by_with_pass(self):
        pass

    def _read_results(self, output):
        o = output[0].lower()
        res = _read_from_address(self._res_ptr, _results_dtype, 1)[0]
        if o=='c':
            return res['num_points']
        if res[0]:
            points = _read_from_address(res['points'], _ptr_ctype, res[0])
            points = (points - self.data.ctypes.data)/self._type.itemsize
        else:
            points = []
        if o == 'i':
            return self.data['idx'][points]
        elif o == 'p':
            return self.data['pos'][points]
        elif o == 'b':
            return self.data['idx'][points], self.data['pos'][points]
        elif o == 'r':
            return self.data[points]
        else:
            return self.data['idx'][points]

    def rebuild(self, data=None, raw=False):
        '''
        Rebuild a fast3tree from a new (or the same) list of points.

        Parameters
        ----------
        data : array_like, optional
            data to rebuild the tree. must be a 2-d array.
            Default: None (use the exisiting data to rebuild)
        '''
        self._check_opened_by_with()
        if data is not None:
            if raw:
                if data.dtype != self._type:
                    raise ValueError("raw data not in correct format.")
                self.data = data
            else:
                self._copy_data(data)
        self._lib.fast3tree_rebuild(self._tree_ptr, \
                np.int64(self.data.shape[0]), self.data)

    def rebuild_boundaries(self):
        ''' Rebuilds the tree boundaries, but keeps structure the same. '''
        self._check_opened_by_with()
        self._lib.fast3tree_maxmin_rebuild(self._tree_ptr)

    def set_boundaries(self, Min, Max):
        '''
        Set the tree boundaries (for periodic boundary condition).

        Parameters
        ----------
        Min : float
        Max : float
        '''
        self._check_opened_by_with()
        self._lib.fast3tree_set_minmax(self._tree_ptr, _float_dtype(Min), \
                _float_dtype(Max))
        
    def free(self):
        ''' Frees the memory of the tree and the results. '''
        self._check_opened_by_with()
        self._lib.fast3tree_results_free(self._res_ptr)
        self._lib.fast3tree_free(np.asarray([self._tree_ptr], dtype=_ptr_dtype))
        self.data = None
        self._tree_ptr = None
        self._res_ptr = None

    def clear_results(self):
        ''' Frees the memory of the results. '''
        self._check_opened_by_with()
        self._lib.fast3tree_results_clear(self._res_ptr)

    def query_nearest_distance(self, center):
        '''
        Find the distance to the nearest point.

        Parameters
        ----------
        center : array_like
            
        Returns
        -------
        distance : float
        '''
        self._check_opened_by_with()
        center_arr = np.asarray(center, dtype=_float_dtype)
        d = self._lib.fast3tree_find_next_closest_distance(self._tree_ptr, \
                self._res_ptr, center_arr)
        return float(d)

    def query_radius(self, center, r, periodic=False, output='index'):
        '''
        Find all points within a sphere centered at center with radius r.

        Parameters
        ----------
        center : array_like
            center of the sphere
        r : float
            radius of the sphere
        periodic : bool, optional
            whether to use periodic boundary condition or not
        output : str, optional
            specify what to return

        Returns
        -------
        Could be one of the followings.
        count : int         [if output=='count']
        index : array_like  [if output=='index']
        pos : array_like    [if output=='pos']
        index, pos : tuple  [if output=='both']
        data : array_like   [if output=='raw']
        '''
        self._check_opened_by_with()
        center_arr = np.array(center, dtype=_float_dtype)
        if periodic:
            i = self._lib.fast3tree_find_sphere_periodic(self._tree_ptr, \
                    self._res_ptr, center_arr, _float_dtype(r))
        else:
            self._lib.fast3tree_find_sphere(self._tree_ptr, self._res_ptr, \
                    center_arr, _float_dtype(r))
        return self._read_results(output)

    def query_box(self, corner1, corner2, inside=True, output='index'):
        '''
        Find all points within a box.

        Parameters
        ----------
        corner1 : array_like
            position of the lower left corner of the box.
        corner2 : array_like
            position of the upper right corner of the box.
        inside : bool, optional
            whether to find the particles inside or outside the box
        output : str, optional
            specify what to return

        Returns
        -------
        Could be one of the followings.
        count : int         [if output=='count']
        index : array_like  [if output=='index']
        pos : array_like    [if output=='pos']
        index, pos : tuple  [if output=='both']
        data : array_like   [if output=='raw']
        '''
        self._check_opened_by_with()
        box_arr = np.array([corner1, corner2], dtype=_float_dtype)
        if inside:
            self._lib.fast3tree_find_inside_of_box(self._tree_ptr, \
                    self._res_ptr, box_arr)
        else:
            self._lib.fast3tree_find_outside_of_box(self._tree_ptr, \
                    self._res_ptr, box_arr)
        return self._read_results(output)

