"""
This file define classes for generating 2D meshes.

"""
import numpy as np
from numba import jit, int32, float64
from numba.experimental import jitclass

class Mesh2D:
    def __init__(self, nx, ny, buff_size, xmin=0, xmax=1, ymin=0, ymax=1):
        self._nx = nx 
        self._ny = ny
        self._nbuff = buff_size
        self._xmin = xmin##
        self._xmax = xmax
        self._ymin = ymin
        self._ymax = ymax

        self._setup()
        return
    
    def _setup(self):
        self._dx = (self._xmax - self._xmin)/(self._nx+1)
        self._dy = (self._ymax - self._ymin)/(self._ny+1)
        self._istart = self._nbuff
        self._istartGC = 0
        self.iend = self._nbuff + self._nx-1
        self.iendGC = 2*self._nbuff + self._nx-1
        self._nxGC = 2*self._istart + self._nx
        

        self._jstart = self._nbuff
        self._jstartGC = 0
        self.jend = self._nbuff+self._ny-1
        self.jendGC = 2*self._nbuff+self._ny-1
        self._nyGC = 2*self._istart + self._ny

        x = np.linspace(self._xmin, self._xmax, self._nxGC)
        y = np.linspace(self._ymin, self._ymax, self._nyGC)
        xx,yy =np.meshgrid(x,y, indexing="ij")
        self._mesh =xx*0
        self._xx = xx
        self._yy = yy
        self._x = x 
        self._y = y
        return
    ## setup meanning?


    @property
    def nx(self):
        return self._nx
    

    @nx.setter
    def nx(self,nx):
        self._nx = nx
        self._setup()
        return
    @ny.setter
    def ny(self,ny):
        self._ny = ny
        self._setup()
        return
    @nbuff.setter
    def nbuff(self,nx):
        self._nx = nx
        self._setup()
        return
    @xmin.setter
    def xmin(self,nx):
        self._nx = nx
        self._setup()
        return
    @xmax.setter
    def xmax(self,nx):
        self._nx = nx
        self._setup()
        return
    @ymin.setter
    def nx(self,nx):
        self._nx = nx
        self._setup()
        return
    @ymax.setter
    def nx(self,nx):
        self._nx = nx
        self._setup()
        return


if __name__=='__main__':

    print("Testing ... nx=10, ny=10, buff =1")
    mesh = Mesh2D(nx=8,ny=8,buff_size=1)
    