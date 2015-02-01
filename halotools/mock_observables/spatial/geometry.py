#Duncan Campbell
#August 22, 2014
#Yale University

"""
Objects and methods for 2D and 3D shapes
"""

from __future__ import division, print_function
import math
import numpy as np
from distances import euclidean_distance

__all__=['polygon2D','circle','face','polygon3D','sphere','cylinder','inside_volume']

class polygon2D(object):
    """
    object that defines a 2-D polygon
    """
    def __init__(self,v=[(0,0),(0,0),(0,0)]):
        self.vertices = v # list of point objects
        for vert in self.vertices:
            if len(vert) != 2:
                raise ValueError('vertices must be len==2')
    
    def area(self,positive=False):
        A = 0.0
        for i in range(0,len(self.vertices)-1):
            A += 0.5*(self.vertices[i][0]*self.vertices[i+1][1]\
                 -self.vertices[i+1][0]*self.vertices[i][1])
        if positive==True: return math.fabs(A)
        else: return A
        
    def center(self):
        Cx = 0.0
        Cy = 0.0
        A = self.area()
        for i in range(0,len(self.vertices)-1):
            Cx += 1.0/(6.0*A)*(self.vertices[i][0]+self.vertices[i+1][0])\
                 *(self.vertices[i][0]*self.vertices[i+1][1]\
                 -self.vertices[i+1][0]*self.vertices[i][1])
            Cy += 1.0/(6.0*A)*(self.vertices[i][1]+self.vertices[i+1][1])\
                 *(self.vertices[i][0]*self.vertices[i+1][1]\
                 -self.vertices[i+1][0]*self.vertices[i][1])
        return Cx,Cy
        
    def circum_r(self):
        d = 0.0
        for i in range(0,len(self.vertices)):
            d = max(d,euclidean_distance(self.center(),self.vertices[i]))
        return d


class circle(object):
    """
    object that defines a 2-D circle
    
    Parameters
    ----------
    center: array_like
        center of mass of the circle, default is 0.0
    
    radius: float
        radius of circle, default is 1.0
    """
    def __init__(self,center=[0.0,0.0], radius=1.0):
        self.center = np.array(center)
        self.radius = radius
    
    def area(self):
        return math.pi*self.radius**2.0
        


class face(object):
    """
    object that defines a face of a 3-D polygone.
    """
    def __init__(self,v=[(0.0,0.0,0.0),(0.0,0.0,0.0),(0.0,0.0,0.0)]):
        self.vertices = v
        #check to see if vertices are co-planer
        if len(self.vertices)<3: 
            raise ValueError('not enough vertices to define face, N must be >=3')
        if len(self.vertices)>3:
            v0 = (self.vertices[0][0], self.vertices[0][1], self.vertices[0][2])
            v1 = (self.vertices[1][0]-v0[0], self.vertices[1][1]-v0[1], self.vertices[1][2]-v0[2])
            v2 = (self.vertices[2][0]-v0[0], self.vertices[2][1]-v0[1], self.vertices[2][2]-v0[2])
            for i in range(3,len(v)):
                v3 = (self.vertices[i][0]-v0[0], self.vertices[i][1]-v0[1], self.vertices[i][2]-v0[2])
                xprd = np.cross(v2,v3)
                vol = np.dot(v1,xprd)
                if vol != 0.0 :raise ValueError('vertices are not co-planer')
    
    def __iter__(self):
        self.start = 0
        self.stop = len(self.vertices)
        return self

    def next(self):
        start = self.start
        if start == self.stop: raise StopIteration
        else:
            self.start += 1
            return self.vertices[start]
            
    def area(self):
        v0 = np.array((self.vertices[0][0], self.vertices[0][1], self.vertices[0][2]))
        v1 = np.array((self.vertices[1][0]-v0[0], self.vertices[1][1]-v0[1], self.vertices[1][2]-v0[2]))
        v2 = np.array((self.vertices[2][0]-v0[0], self.vertices[2][1]-v0[1], self.vertices[2][2]-v0[2]))
        v3 = np.cross(v1,v2)
        v4 = np.cross(v3,v1)
        e1 = v1/math.sqrt(v1[0]**2.0+v1[1]**2.0+v1[2]**2.0)
        e2 = v4/math.sqrt(v4[0]**2.0+v4[1]**2.0+v4[2]**2.0)
        x = np.empty(len(self.vertices))
        y = np.empty(len(self.vertices))
        for i, vertex in enumerate(self.vertices):
            x[i] = (vertex[0]-v0[0])*e1[0]+(vertex[1]-v0[1])*e1[1]+(vertex[2]-v0[2])*e1[2]
            y[i] = (vertex[0]-v0[0])*e2[0]+(vertex[1]-v0[1])*e2[1]+(vertex[2]-v0[2])*e2[2]
        A = 0.0
        for i in range(0,len(self.vertices)-1):
            A += 0.5*(x[i]*y[i+1]-x[i+1]*y[i])
        return A
        
    def center(self):
        v0 = np.array((self.vertices[0][0], self.vertices[0][1], self.vertices[0][2]))
        v1 = np.array((self.vertices[1][0]-v0[0], self.vertices[1][1]-v0[1], self.vertices[1][2]-v0[2]))
        v2 = np.array((self.vertices[2][0]-v0[0], self.vertices[2][1]-v0[1], self.vertices[2][2]-v0[2]))
        v3 = np.cross(v1,v2)
        v4 = np.cross(v3,v1)
        e1 = v1/math.sqrt(v1[0]**2.0+v1[1]**2.0+v1[2]**2.0)
        e2 = v4/math.sqrt(v4[0]**2.0+v4[1]**2.0+v4[2]**2.0)
        x = np.empty(len(self.vertices))
        y = np.empty(len(self.vertices))
        for i, vertex in enumerate(self.vertices):
            x[i] = (vertex[0]-v0[0])*e1[0]+(vertex[1]-v0[1])*e1[1]+(vertex[2]-v0[2])*e1[2]
            y[i] = (vertex[0]-v0[0])*e2[0]+(vertex[1]-v0[1])*e2[1]+(vertex[2]-v0[2])*e2[2]
        Cx = 0.0
        Cy = 0.0
        A = self.area()
        for i in range(0,len(self.vertices)-1):
            Cx += 1.0/(6.0*A)*(x[i]+x[i+1])*(x[i]*y[i+1]-x[i+1]*y[i])
            Cy += 1.0/(6.0*A)*(y[i]+y[i+1])*(x[i]*y[i+1]-x[i+1]*y[i])
        Cxx = v0[0] + e1[0]*Cx+e2[0]*Cy
        Cyy = v0[1] + e1[1]*Cx+e2[1]*Cy
        Czz = v0[2] + e1[2]*Cx+e2[2]*Cy
        return Cxx,Cyy,Czz
        
    def normal(self):
        v0 = self.center()
        v1 = (self.vertices[1][0]-v0[0], self.vertices[1][1]-v0[1], self.vertices[1][2]-v0[2])
        v2 = (self.vertices[2][0]-v0[0], self.vertices[2][1]-v0[1], self.vertices[2][2]-v0[2])
        n = np.cross(v1,v2)
        n = n/(math.sqrt(n[0]**2.0+n[1]**2.0+n[2]**2.0))
        return n


class polygon3D(object):
    def __init__(self,f=[]):
        self.faces = f # list of point objects
        unq_verts = list({vert for sublist in self.faces for vert in sublist})
        self.vertices = [(unq_verts[i][0],unq_verts[i][1],unq_verts[i][2])\
                         for i in range(len(unq_verts))] 
        for f in self.faces:
            if not isinstance(f,face):
                raise ValueError('arguments must be of type face')
                
    def volume(self):
        vol = 0.0
        for f in self.faces: # the faces
            n = len(f.vertices)
            v2 = f.vertices[0] # the pivot of the fan
            #x2 = v2.x1
            #y2 = v2.x2
            #z2 = v2.x3
            x2,y2,z2 = v2
            for i in range(1,n-1): # divide into triangular fan segments
                v0 = f.vertices[i]
                #x0 = v0.x1
                #y0 = v0.x2
                #z0 = v0.x3
                x0,y0,z0 = v0
                v1 = f.vertices[i+1]
                #x1 = v1.x1
                #y1 = v1.x2
                #z1 = v1.x3
                x1,y1,z1 = v1
                # Add volume of tetrahedron formed by triangle and origin
                vol += math.fabs(x0 * y1 * z2 + x1 * y2 * z0 \
                                 + x2 * y0 * z1 - x0 * y2 * z1 \
                                 - x1 * y0 * z2 - x2 * y1 * z0)
        return vol/6.0
        
        def inside(self,point3D):
            pass


class volume(object):
    """
    abstract volume class
    """
    def __init__(self):
        self.center = (0.0,0.0,0.0)

class sphere(volume):
    """
    sphere volume object
    
    parameters
    ----------
    center: array_like, optional
        center of mass of the volume, default is the origin (0,0,0)
    
    radius: float, optional
        radius of the sphere, default is 1.0
    """
    
    def __init__(self,center=(0.0,0.0,0.0), radius=1.0):
        super(sphere, self).__init__()
        self.center = center
        self.radius = radius
    
    def volume(self):
        return (4.0/3.0)*math.pi*self.radius**3.0


class cylinder(volume):
    """
    cylinder volume object
    
    parameters
    ----------
    center: array_like, optional
        center of mass of the volume, default is the origin (0,0,0)
    
    radius: float, optional
        radius of the circular faces of the cylinder,default is 1.0
    
    length: float, optional
        length of the cylinder, default is 1.0
    
    normal: array_like, optional
        vector defining the normal to the circular fave of the cylinder, default is [0,0,1]
    """
    def __init__(self, center=np.array([0.0,0.0,0.0]), radius = 1.0, length=1.0, \
                 normal=np.array([0.0,0.0,1.0])):
        super(cylinder, self).__init__()
        self.center = center
        self.normal = normal
        self.radius = radius
        self.length = length
        
    def volume(self):
        """
        volume of cylinder
        """
        return math.pi * self.radius**2.0 * self.length
    
    def circum_r(self):
        """
        radius of sphere needed to circumscribe the volume centered on self.center
        """
        r = math.sqrt(self.radius**2.0+(self.length/2.0)**2.0)
        return r
    
    def inside(self, points=(0,0,0), period=None):
        """
        Calculate whether a point is inside or outside the volume.
        
        Parameters
        ----------
        points: array_like
            numpy.array or numpy.ndarray of points with shape (N,3).
        
        Returns
        -------
        inside: boolean
            True, point is inside the volume
            False, point is outside the volume
        """
        
        points = np.array(points)
        if len(points.shape) > 1:
            if points.shape[1] != 3: raise ValueError('points argument must have shape (3,N)')
            x = points[:,0]
            y = points[:,1]
            z = points[:,2]
            N=points.shape[0]
        else:
            if points.shape[0] != 3: raise ValueError('points argument must have shape (3,N)')
            x = points[0]
            y = points[1]
            z = points[2]
            N=1
        #process the period parameter
        if period is None:
            period = np.array([np.inf]*3)
        else:
            period = np.asarray(period).astype("float64")
        if period.shape[0]!=3: raise ValueError('period must be None or have shape (3,)')
        
        from halotools.mock_observables.spatial.distances import projected_distance
        
        #need to repeat this in order to use distance calculation
        center = np.tile(self.center,(N,1))
        normal = np.tile(self.normal,(N,1))
        
        d_para, d_perp = projected_distance(center,points,normal,period=period)
        
        return (d_para<self.length/2.0) & (d_perp<self.radius)


def inside_volume(shapes, points, period=None):
    """
    Check if a list of points is inside a volume.
    
    parameters
    ----------
    shapes: list
        a list of volume objects, or list of objects
    points: array_like
        a list of points or a ckdtree object
    period: array_like, optional
        length k array defining axis aligned PBCs. If set to none, PBCs = infinity.
    
    returns
    -------
    inside_points: np.array
        indices of points which fall within the specified volumes.
    inside_shapes: np.array
        array of booleans, True if any points fall within the shape, False otherwise
    points_in_shapes: list
        list of arrays of points inside each shape
    N_points_in_shapes: np.array
    """
    from halotools.mock_observables.spatial.kdtrees.ckdtree import cKDTree
    
    #check input
    if type(points) is not cKDTree:
        points = np.array(points) 
        if points.shape[-1] != 3:
            raise ValueError('points must be 3-dimensional.')
    elif type(points) is cKDTree:
        if points.m != 3:
            raise ValueError('kdtree must be made of 3-dimensional points.')
    else:
        raise ValueError('points must be either an array of points or ckdtree.')
    
    #check to see if a kdtree was passed and if not, create one.
    if type(points) is not cKDTree:
        points = np.array(points)
        KDT = cKDTree(points)
    else:
        KDT = points
        #points = KDT.data #will this make a copy and take up memory?
    
    #is shapes a list, or a single shape?
    if type(shapes) is list:
        #create arrays to store results
        inside_points = np.empty((0,), dtype=np.int)
        inside_shapes = np.empty((len(shapes),), dtype=bool)
        N_points_in_shapes = np.zeros((len(shapes),))
        points_in_shapes = []
        for i, shape in enumerate(shapes):
            points_to_test = np.array(KDT.query_ball_point(shape.center,shape.circum_r(),period=period))
            #need to do some special maneuvering to deal with kdtree/array input for points.
            if len(points_to_test)>0:
                if type(points) is cKDTree:
                    inside = shape.inside(KDT.data[points_to_test], period)
                    inside = points_to_test[inside]
                else:
                    inside = shape.inside(points[points_to_test], period)
                    inside = points_to_test[inside]
                #does a point fell within the volume?
                if len(inside)>0:
                     inside_shapes[i]=True
                else: inside_shapes[i]=False
                #append indices of points which fell within the volume
                inside_points = np.hstack((inside_points,inside))
                points_in_shapes.append(inside)
                N_points_in_shapes[i] = len(inside) 
        inside_points = np.unique(inside_points) #remove repeats
        return inside_points, inside_shapes, points_in_shapes, N_points_in_shapes
    else:
        shape = shapes
        points_to_test = np.array(KDT.query_ball_point(shape.center,shape.circum_r(),period=period))
        #need to do some special maneuvering to deal with kdtree/array input for points.
        if type(points) is cKDTree:
            inside = shape.inside(KDT.data[points_to_test], period)
            inside = points_to_test[inside]
        else:
            inside = shape.inside(points[points_to_test], period)
            inside = points_to_test[inside]
        #does a point fell within the volume?
        if len(inside)>0: inside_shapes = True
        else:  inside_shapes = False
        inside_points = inside
        N_points_in_shapes = len(inside)
        return inside_points, inside_shapes, inside_points, N_points_in_shapes
