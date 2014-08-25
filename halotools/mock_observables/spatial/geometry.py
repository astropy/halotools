#Duncan Campbell
#August 22, 2014
#Yale University
#This code includes geometric objects and methods to work on those objects.

from __future__ import division
import math
import numpy as np

#this function works with numpy arrays
def distance2D(point, points):
    '''
    Calculate the distance between two points with cartesian coordinates.
    '''
    if len(point) != 2: raise ValueError('first argument must be a list of 2 floats')
    x1 = point[0]
    y1 = point[1]
    points = np.array(points)
    if points.shape[0] != 2: raise ValueError('second argument must have shape (2,N)')
    if len(points.shape) > 1:
        x2 = points[:,0]
        y2 = points[:,1]
    else:
        x2 = np.array([points[0]])
        y2 = np.array([points[1]])
    
    return np.sqrt((x1-x2)**2.0+(y1-y2)**2.0)


#this function works with numpy arrays
def distance2D_periodic(point, points, box_size=0.0):
    '''
    Calculate the distance between two points with cartesian coordinates in a periodic 
    box with a corner of the box at (0,0,0).
    '''
    
    if len(point) != 2: raise ValueError('first argument must be a list of 2 floats')
    x1 = point[0]
    y1 = point[1]
    points = np.array(points)
    if points.shape[0] != 2: raise ValueError('second argument must have shape (2,N)')
    if len(points.shape) > 1:
        x2 = points[:,0]
        y2 = points[:,1]
    else:
        x2 = np.array([points[0]])
        y2 = np.array([points[1]])
    
    delta_x = np.fabs(x1 - x2)
    wrap = (delta_x > box_size/2.0)
    delta_x[wrap] = box_size - delta_x[wrap]
    delta_y = np.fabs(y1 - y2)
    wrap = (delta_y > box_size/2.0)
    delta_y[wrap] = box_size - delta_y[wrap]
    d = np.sqrt(delta_x ** 2.0 + delta_y ** 2.0)
    
    return d


#this function works with numpy arrays
def distance3D(point, points):
    '''
    Calculate the distance between two points with cartesian coordinates.
    '''
    if len(point) != 3: raise ValueError('first argument must be a list of 3 floats')
    x1 = point[0]
    y1 = point[1]
    z1 = point[2]
    points = np.array(points)
    if points.shape[0] != 3: raise ValueError('second argument must have shape (3,N)')
    if len(points.shape) > 1:
        x2 = points[:,0]
        y2 = points[:,1]
        z2 = points[:,2]
    else:
        x2 = np.array([points[0]])
        y2 = np.array([points[1]])
        z2 = np.array([points[2]])
    
    return np.sqrt((x1-x2)**2.0+(y1-y2)**2.0+(z1-z2)**2.0)


#this function works with numpy arrays
def distance3D_periodic(point, points, box_size=0.0):
    '''
    Calculate the distance between two points with cartesian coordinates in a periodic 
    box with a corner of the box at (0,0,0).
    '''
    
    if len(point) != 3: raise ValueError('first argument must be a list of 3 floats')
    x1 = point[0]
    y1 = point[1]
    z1 = point[2]
    points = np.array(points)
    if points.shape[0] != 3: raise ValueError('second argument must have shape (3,N)')
    if len(points.shape) > 1:
        x2 = points[:,0]
        y2 = points[:,1]
        z2 = points[:,2]
    else:
        x2 = np.array([points[0]])
        y2 = np.array([points[1]])
        z2 = np.array([points[2]])
    
    delta_x = np.fabs(x1 - x2)
    wrap = (delta_x > box_size/2.0)
    delta_x[wrap] = box_size - delta_x[wrap]
    delta_y = np.fabs(y1 - y2)
    wrap = (delta_y > box_size/2.0)
    delta_y[wrap] = box_size - delta_y[wrap]
    delta_z = np.fabs(z1 - z2)
    wrap = (delta_z > box_size/2.0)
    delta_z[wrap] = box_size - delta_z[wrap]
    d = np.sqrt(delta_x ** 2.0 + delta_y ** 2.0 + delta_z ** 2.0)
    
    return d


class polygon2D(object):
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
        print A
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
            print distance2D(self.center(),self.vertices[i])
            print max(d,distance2D(self.center(),self.vertices[i]))
            d = max(d,distance2D(self.center(),self.vertices[i]))
        return d


class circle(object):
    def __init__(self,center=[0.0,0.0], r=0.0):
        self.center = np.array(center)
        self.radius = r
    
    def area(self):
        return math.pi*self.radius**2.0
        


class face(object):
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
            print type(f)
            if not isinstance(f,face):
                print type(f)
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


class sphere(object):
    def __init__(self,center=(0.0,0.0,0.0), r=0.0):
        self.center = center
        self.radius = r
    
    def volume(self):
        return (4.0/3.0)*math.pi*self.radius**3.0


class cylinder(object):
    def __init__(self, center=(0.0,0.0,0.0), radius = 1.0, length=1.0, \
                 normal=np.array([0.0,0.0,1.0])):
        self.center = center
        self.normal = normal
        self.radius = radius
        self.length = length
        
    def volume(self):
        return math.pi * self.radius**2.0 * self.length
    
    def circum_r(self):
        '''
        radius to circumscribe the volume given the center
        '''
        r = math.sqrt(self.radius**2.0+(self.length/2.0)**2.0)
        return r
    
    def inside(self, points=(0,0,0), period=None):
        '''
        Calculate whether a point is inside or outside the volume.
        Parameters
            self: polygon3D object which defines volume
            point3D object to test
        Returns
            True: point is inside the volume
            False: point is outside the volume
        '''
        points = np.array(points)
        if len(points.shape) > 1:
            if points.shape[1] != 3: raise ValueError('argument must have shape (3,N)')
            x = points[:,0]
            y = points[:,1]
            z = points[:,2]
            N=points.shape[0]
        else:
            if points.shape[0] != 3: raise ValueError('argument must have shape (3,N)')
            x = points[0]
            y = points[1]
            z = points[2]
            N=1
        #process the period parameter
        if period is None:
            period = np.array([np.inf]*3)
        else:
            period = np.asarray(period).astype("float64")
        if period.shape[0]!=3: raise ValueError('period must have shape (3,)')
        
        if np.max(period)==np.inf:
            #define coordinate origin
            x0,y0,z0 = np.array(self.center)
            #recenter on origin
            x = x-x0
            y = y-y0
            z = z-z0
            #calculate new basis vectors
            v1 = self.normal
            #generate a random vector that is not parallel to v1
            ran_v = np.random.random(3)
            angle = np.dot(v1,ran_v)/(np.sqrt(np.dot(v1,v1)*np.dot(ran_v,ran_v)))
            while angle<0.02:
                ran_v = np.random.random(3)
                angle = np.dot(v1,ran_v)/(np.sqrt(np.dot(v1,v1)*np.dot(ran_v,ran_v)))
        
            #define new basis vectors
            e1= np.array([1,0,0])
            e2= np.array([0,1,0])
            e3= np.array([0,0,1])
            v1 = v1/np.sqrt(v1[0]**2.0+v1[1]**2.0+v1[2]**2.0) #normalize
            v2 = np.cross(v1,ran_v)
            v2 = v2/np.sqrt(v2[0]**2.0+v2[1]**2.0+v2[2]**2.0) #normalize
            v3 = np.cross(v1,v2)
            v3 = v3/np.sqrt(v3[0]**2.0+v3[1]**2.0+v3[2]**2.0) #normalize
        
            #calculate coordinate of point given new basis        
            Q = np.array([[np.dot(e1,v1),np.dot(e1,v2),np.dot(e1,v3)],
                      [np.dot(e2,v1),np.dot(e2,v2),np.dot(e2,v3)],
                      [np.dot(e3,v1),np.dot(e3,v2),np.dot(e3,v3)],])
            xp,yp,zp = np.dot(Q.T,np.array([x,y,z]))
        
            L_proj = np.fabs(xp)
            R_proj = np.sqrt(yp**2.0+zp**2.0)
            
            result = (L_proj<self.length/2.0) & (R_proj<self.radius)
    
        else:
            cases = np.array([[False,False,False],\
                              [True,False,False],[True,True,False],[True,True,True],\
                              [False,True,False],[False,True,True],\
                              [False,False,True]])
            dir = np.array(self.center)>period/2.0
            period[dir] = -period[dir]
            reflections = cases*period
            result = np.empty((N,), dtype=bool)
            result.fill(False)
            for reflection in reflections:
                #define coordinate origin
                x0,y0,z0 = reflection+np.array(self.center)
                #print reflection, x0,y0,z0
                #recenter on origin
                xp = x-x0
                yp = y-y0
                zp = z-z0
                #calculate new basis vectors
                v1 = self.normal
                #generate a random vector that is not parallel to v1
                ran_v = np.random.random(3)
                angle = np.dot(v1,ran_v)/(np.sqrt(np.dot(v1,v1)*np.dot(ran_v,ran_v)))
                while angle<0.02:
                        ran_v = np.random.random(3)
                        angle = np.dot(v1,ran_v)/(np.sqrt(np.dot(v1,v1)*np.dot(ran_v,ran_v)))
        
                #define new basis vectors
                e1= np.array([1,0,0])
                e2= np.array([0,1,0])
                e3= np.array([0,0,1])
                v1 = v1/np.sqrt(v1[0]**2.0+v1[1]**2.0+v1[2]**2.0) #normalize
                v2 = np.cross(v1,ran_v)
                v2 = v2/np.sqrt(v2[0]**2.0+v2[1]**2.0+v2[2]**2.0) #normalize
                v3 = np.cross(v1,v2)
                v3 = v3/np.sqrt(v3[0]**2.0+v3[1]**2.0+v3[2]**2.0) #normalize
        
                #calculate coordinate of point given new basis        
                Q = np.array([[np.dot(e1,v1),np.dot(e1,v2),np.dot(e1,v3)],
                              [np.dot(e2,v1),np.dot(e2,v2),np.dot(e2,v3)],
                              [np.dot(e3,v1),np.dot(e3,v2),np.dot(e3,v3)],])
                xp,yp,zp = np.dot(Q.T,np.array([xp,yp,zp]))
        
                L_proj = np.fabs(xp)
                R_proj = np.sqrt(yp**2.0+zp**2.0)
                
                result = result + np.array((L_proj<self.length/2.0) & (R_proj<self.radius))

        return result


def inside_volume(shapes, points, period=None):
    '''
    Check if a list of points is inside a volume.
    parameters
        shape: a geometry.py shape object, or list of objects
        points: a list of points or a ckdtree object
        period: length k array defining axis aligned PBCs. If set to none, PBCs = infinity.
    returns
        inside_points: np.array of ints of indices of points which fall within the 
            specified volumes
        inside_shapes: np.array of booleans, True if any points fall within the shape, 
            False otherwise
    '''
    
    if type(points) is not cKDTree:
        points = np.array(points)
        KDT = cKDTree(points)
    else:
        KDT=points
    
    #if shapes is a list of shapes
    if type(shapes) is list:
        inside_points=np.empty((0,), dtype=np.int)
        inside_shapes=np.empty((len(shapes),), dtype=bool)
        for i, shape in enumerate(shapes):
            points_to_test = np.array(KDT.query_ball_point(shape.center,shape.circum_r(),period=period))
    
            if type(points) is cKDTree:
                inside = shape.inside(KDT.data[points_to_test], period)
                inside = points_to_test[inside]
            else:
                inside = shape.inside(points[points_to_test], period)
                inside = points_to_test[inside]
            if len(inside)>0:
                inside_shapes[i]=True
            else: inside_shapes[i]=False
            inside_points = np.hstack((inside_points,inside))
        inside_points = np.unique(inside_points)
        return inside_points, inside_shapes
    else: #if shapes is a single shape object
        shape = shapes
        points_to_test = np.array(KDT.query_ball_point(shape.center,shape.circum_r(),period=period))
    
        if type(points) is cKDTree:
            inside = shape.inside(KDT.data[points_to_test], period)
            inside = points_to_test[inside]
        else:
            inside = shape.inside(points[points_to_test], period)
            inside = points_to_test[inside]
        
        if len(inside)>0: inside_shapes = True
        else:  inside_shapes = False
        inside_points = inside
        return inside_points, inside_shapes
