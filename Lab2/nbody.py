from time import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import numba as na
from numba import jit, njit, prange, set_num_threads

"""

This program solve 3D direct N-particles simulations 
under gravitational forces. 

This file contains two classes:

1) Particles: describes the particle properties
2) NbodySimulation: describes the simulation

Usage:

    Step 1: import necessary classes

    from nbody import Particles, NbodySimulation

    Step 2: Write your own initialization function

    
        def initialize(particles:Particles):
            ....
            ....
            particles.set_masses(mass)
            particles.set_positions(pos)
            particles.set_velocities(vel)
            particles.set_accelerations(acc)

            return particles

    Step 3: Initialize your particles.

        particles = Particles(N=100)
        initialize(particles)


    Step 4: Initial, setup and start the simulation

        simulation = NbodySimulation(particles)
        simulation.setip(...)
        simulation.evolve(dt=0.001, tmax=10)


Author: Kuo-Chuan Pan, NTHU 2022.10.30
For the course, computational physics lab

"""

class Particles:
    """
    
    The Particles class handle all particle properties

    for the N-body simulation. 

    """
    def __init__(self,N:int=100):
        """
        Prepare memories for N particles

        :param N: number of particles.

        By default: particle properties include:
                nparticles: int. number of particles
                _masses: (N,1) mass of each particle
                _positions:  (N,3) x,y,z positions of each particle
                _velocities:  (N,3) vx, vy, vz velocities of each particle
                _accelerations:  (N,3) ax, ay, az accelerations of each partciel
                _tags:  (N)   tag of each particle
                _time: float. the simulation time 

        """
        #build array shape
        self.nparticles = N 
        self._time = 0
        self._masses = np.ones((N,1))
        self._positions = np.zeros((N,3))
        self._velocities = np.zeros((N,3))
        self._accelerations = np.zeros((N,3))
        self._tags = np.linspace(1,N,N)
        # self._Ek = np.zeros((N,1))
        # #self._U = np.zeros((N,1))
        return

    #getter

    def get_time(self):
        return self._time
    def get_masses(self):
        return self._masses
    def get_positions(self):
        return self._positions
    def get_velocities(self):
        return self._velocities 
    def get_accelerations(self):
        return self._accelerations
    def get_tags(self): 
        return self._tags
    # def get_Ek(self):
    #     self.Ek = np.sum(0.5*self._masses*np.sum(self._velocities**2,axis=1))
    #     return self._Ek
    # def get_U(self):
    #     for i in range(self.nparticles):
    #         for j in range(self.nparticles):
    #             if(j>i):
    #                 posx = self._positions[:,0]
    #                 posy = self._positions[:,1]
    #                 posz = self._positions[:,2]
    #                 x = (posx[i] - posx[j])
    #                 y = (posy[i] - posy[j])
    #                 z = (posz[i] - posz[j])
    #                 r = np.sqrt(x**2+y**2+z**2)+0.01
    #                 U = np.sum(-6.67e-8*self._masses[i]*self._masses[j]/r)
    #     return U

    
    #setter
    def set_time(self,time):
        self._time = time
        return
    def set_masses(self,masses):
        self._masses = masses
        return
    def set_positions(self,positions):
        self._positions = positions
        return
    def set_velocities(self,velocities):
        self._velocities = velocities
        return
    def set_accelerations(self,accelerations):
        self._accelerations = accelerations
        return
    def set_tags(self,tags): 
        self._tags = tags
        return
    # def set_Ek(self,Ek): 
    #     self._Ek = Ek
        return
    # def set_U(self,U): 
    #     self._U = U


    def output(self,fn, time):
        """
        Write simulation data into a file named "fn"


        """
        mass = self._masses
        pos  = self._positions
        vel  = self._velocities
        acc  = self._accelerations
        tag  = self._tags
        # Ek   = self._Ek
        # U    = self._U

        
        
        header = """
                ----------------------------------------------------
                Data from a 3D direct N-body simulation. 

                rows are i-particle; 
                coumns are :mass, tag, x ,y, z, vx, vy, vz, ax, ay, az

                NTHU, Computational Physics Lab

                ----------------------------------------------------
                """
        header += "Time = {}".format(time)
        np.savetxt(fn,(tag[:],mass[:,0],pos[:,0],pos[:,1],pos[:,2],
                            vel[:,0],vel[:,1],vel[:,2],
                            acc[:,0],acc[:,1],acc[:,2]),header=header)

        return
    




class NbodySimulation:
    """
    
    The N-body Simulation class.
    
    """

    def __init__(self,particles:Particles):
        """
        Initialize the N-body simulation with given Particles.

        :param particles: A Particles class.  
        
        """

        # store the particle information
        self.nparticles = particles.nparticles
        self.particles  = particles
        

        # Store physical information
        self.time  = 0.0  # simulation time

        # Set the default numerical schemes and parameters
        self.setup()
        
        return

    def setup(self, G=6.67e-8, #unit select
                    rsoft=0.01, 
                    method="Euler", 
                    io_freq=10, 
                    io_title="particles",
                    io_screen=True,
                    visualized=False
                    ):
        """
        Customize the simulation enviroments.

        :param G: the graivtational constant
        :param rsoft: float, a soften length
        :param meothd: string, the numerical scheme
                       support "Euler", "RK2", and "RK4"

        :param io_freq: int, the frequency to outupt data.
                        io_freq <=0 for no output. 
        :param io_title: the output header
        :param io_screen: print message on screen or not.
        :param visualized: on the fly visualization or not. 
        
        """
        #先設定class 
        self.G = G
        self.rsoft = rsoft
        self.method = method
        self.io_freq = io_freq
        self.io_title = io_title
        self.io_screen = io_screen
        self.visualized = visualized
        return 

    def evolve(self, dt:float=0.01, tmax:float=1):
        """

        Start to evolve the system

        :param dt: time step
        :param tmax: the finial time
        
        """
        # TODO:
        self.dt = dt
        self.tmax = tmax
        

        method = self.method
        if method=="Euler":
            _update_particles = self._update_particles_euler
        elif method=="RK2":
            _update_particles = self._update_particles_rk2
        elif method=="RK4":
            _update_particles = self._update_particles_rk4
        elif method=="LF":
            _update_particles = self._update_particles_LF
        else:
            print("No such update meothd", method)
            quit() 

        # prepare an output folder for lateron output
        io_folder = "data_"+self.io_title
        Path(io_folder).mkdir(parents=True, exist_ok=True)
        
        # ====================================================
        #
        # The main loop of the simulation
        #
        # =====================================================

        # TODO:
        time = self.time
        nsteps = np.ceil((tmax-time)/dt)
        #np.ceil np.floor
        for n in range(int(nsteps)):

            if (time+dt) > tmax: dt = tmax - time
            
            #update_particles 
            particles = self.particles
            # Ek[n]=np.sum(0.5*particles._masses*np.sum(particles._velocities**2,axis=1))
            # T[n] = time
            _update_particles(dt, particles)

            #check io/visual
            if (n % self.io_freq ==0):
                if self.io_screen:
                    print("n =",n, " time =",time, " dt =", dt)
                    # self.Ek[n]=np.sum(0.5*particles._masses*np.sum(particles._velocities**2,axis=1))
                    # self.T[n] = time

            
                    

                
                #visual
                if self.visualized:
                    pass

                #output data
                fn = io_folder+"/data_"+self.io_title+"_"+str(n).zfill(5)+".txt"
                self.particles.output(fn,time)
                

                
        


            #update time
            time += dt
        # print(self.Ek)
        # print(self.T)
        # plt.figure(1)
        # plt.plot(self.T,self.Ek,'--',label="RK4",markersize=3)
        # plt.xlabel('omega')
        # plt.ylabel('u')
        # plt.show()
        # print("Done!")
        return 

    # def _calculate_acceleration(self, mass, pos):
    #     """
    #     Calculate the acceleration.
    #     """
    #     # TODO:
    #     # acc =(1e13/86400**2)*np.ones((self.particles,3))
    #     posx = pos[:,0]
    #     posy = pos[:,1]
    #     posz = pos[:,2]
    #     G    = self.G
    #     npts = self.nparticles
    #     rsoft = self.rsoft
    #     acc = np.zeros((npts,3)) #reset acc

    #     for i in range(npts):
    #         for j in range(npts):
    #             if(j>i):
    #                 x = (posx[i] - posx[j])
    #                 y = (posy[i] - posy[j])
    #                 z = (posz[i] - posz[j])
    #                 rsq = x**2 + y**2 + z**2
    #                 req = np.sqrt(x**2+y**2)
    #                 force = -G*mass[i,0]*mass[j,0]/(rsq+rsoft)
    #                 theta = np.arctan2(y,x)
    #                 phi = np.arctan2(z,req)
    #                 fx = force*np.cos(theta)*np.cos(phi)
    #                 fy = force*np.sin(theta)*np.cos(phi)
    #                 fz = force*np.sin(phi)
    #                 acc[i,0] += fx/mass[i,0]
    #                 acc[i,1] += fy/mass[i,0]
    #                 acc[i,2] += fz/mass[i,0]

    #                 acc[j,0] -= fx/mass[j,0]
    #                 acc[j,1] -= fy/mass[j,0]
    #                 acc[j,2] -= fz/mass[j,0]
    #     return acc
    def _update_particles_euler(self, dt, particles:Particles):
        mass = particles.get_masses()
        pos = particles.get_positions() # y0[0]
        vel = particles.get_velocities()
        acc = particles.get_accelerations()
        posx = pos[:,0]
        posy = pos[:,1]
        posz = pos[:,2]
        G    = self.G
        npts = self.nparticles
        rsoft = self.rsoft
        # acc = np.zeros((npts,3)) #reset acc
        # acc = self._calculate_acceleration(mass,pos)
        acc = _calculate_acceleration_kernel(mass, posx, posy, posz, acc, G, rsoft, npts)
        pos = pos + vel * dt # y1[0]
        vel = vel + acc * dt 
        # acc = self._calculate_acceleration(mass,pos)
        acc = _calculate_acceleration_kernel(mass, posx, posy, posz, acc, G, rsoft, npts)
        particles.set_positions(pos)
        particles.set_velocities(vel)
        particles.set_accelerations(acc)
        return particles

    def _update_particles_rk2(self, dt, particles:Particles):
        mass = particles.get_masses()
        pos = particles.get_positions() # y0[0]
        vel = particles.get_velocities()
        acc = particles.get_accelerations()
        posx = pos[:,0]
        posy = pos[:,1]
        posz = pos[:,2]
        G    = self.G
        npts = self.nparticles
        rsoft = self.rsoft
        acc = _calculate_acceleration_kernel(mass, posx, posy, posz, acc, G, rsoft, npts)
        pos_1 = pos + vel*dt
        vel_1 = vel + acc*dt
        posx = pos_1[:,0]
        posy = pos_1[:,1]
        posz = pos_1[:,2]
        acc_1 = _calculate_acceleration_kernel(mass, posx, posy, posz, acc, G, rsoft, npts)

        pos = pos+0.5*dt*(vel+vel_1)
        vel = vel+0.5*dt*(acc+acc_1)
        posx = pos[:,0]
        posy = pos[:,1]
        posz = pos[:,2]
        acc = _calculate_acceleration_kernel(mass, posx, posy, posz, acc, G, rsoft, npts)
        
        particles.set_positions(pos)
        particles.set_velocities(vel)
        particles.set_accelerations(acc)
        return particles

    def _update_particles_rk4(self, dt, particles:Particles):
        mass = particles.get_masses()
        pos = particles.get_positions() # y0[0]
        vel = particles.get_velocities()
        acc = particles.get_accelerations()
        posx = pos[:,0]
        posy = pos[:,1]
        posz = pos[:,2]
        G    = self.G
        npts = self.nparticles
        rsoft = self.rsoft
        acc = _calculate_acceleration_kernel(mass, posx, posy, posz, acc, G, rsoft, npts)
        pos_1 = pos + 0.5*vel*dt
        vel_1 = vel + 0.5*acc*dt
        posx = pos_1[:,0]
        posy = pos_1[:,1]
        posz = pos_1[:,2]
        acc_1 = _calculate_acceleration_kernel(mass, posx, posy, posz, acc, G, rsoft, npts)

        pos_2 = pos_1 + 0.5*vel_1*dt
        vel_2 = vel_1 + 0.5*acc_1*dt
        posx = pos_2[:,0]
        posy = pos_2[:,1]
        posz = pos_2[:,2]
        acc_2 = _calculate_acceleration_kernel(mass, posx, posy, posz, acc, G, rsoft, npts)

        pos_3 = pos_2 + vel_2*dt
        vel_3 = vel_2 + acc_2*dt
        posx = pos_3[:,0]
        posy = pos_3[:,1]
        posz = pos_3[:,2]
        acc_3 = _calculate_acceleration_kernel(mass, posx, posy, posz, acc, G, rsoft, npts)

        pos = pos+(1/6)*dt*(vel+2*vel_1+2*vel_2+vel_3)
        vel = vel+(1/6)*dt*(acc+2*acc_1+2*acc_2+acc_3)
        posx = pos[:,0]
        posy = pos[:,1]
        posz = pos[:,2]
        acc = _calculate_acceleration_kernel(mass, posx, posy, posz, acc, G, rsoft, npts)
        
        particles.set_positions(pos)
        particles.set_velocities(vel)
        particles.set_accelerations(acc)
        return particles
    
    def _update_particles_LF(self, dt, particles:Particles):
        mass = particles.get_masses()
        pos = particles.get_positions() # y0[0]
        vel = particles.get_velocities()
        acc = particles.get_accelerations()
        posx = pos[:,0]
        posy = pos[:,1]
        posz = pos[:,2]
        G    = self.G
        npts = self.nparticles
        rsoft = self.rsoft
        acc = _calculate_acceleration_kernel(mass, posx, posy, posz, acc, G, rsoft, npts)
        pos = pos + vel * dt # y1[0] 
        vel = vel + 0.5 * acc * dt
        acc = _calculate_acceleration_kernel(mass, posx, posy, posz, acc, G, rsoft, npts)
        particles.set_positions(pos)
        particles.set_velocities(vel)
        particles.set_accelerations(acc)
        return particles
    



@jit(nopython=True)
def _calculate_acceleration_kernel(mass, posx ,posy ,posz , acc, G, rsoft,npts):

    for i in range(npts):
            for j in range(npts):
                if(j>i):
                    x = (posx[i] - posx[j])
                    y = (posy[i] - posy[j])
                    z = (posz[i] - posz[j])
                    rsq = x**2 + y**2 + z**2
                    req = np.sqrt(x**2+y**2)
                    force = -G*mass[i,0]*mass[j,0]/(rsq+rsoft**2)
                    theta = np.arctan2(y,x)
                    phi = np.arctan2(z,req)
                    fx = force*np.cos(theta)*np.cos(phi)
                    fy = force*np.sin(theta)*np.cos(phi)
                    fz = force*np.sin(phi)

                    acc[i,0] += fx/mass[i,0]
                    acc[i,1] += fy/mass[i,0]
                    acc[i,2] += fz/mass[i,0]

                    acc[j,0] -= fx/mass[j,0]
                    acc[j,1] -= fy/mass[j,0]
                    acc[j,2] -= fz/mass[j,0]
    return acc  

if __name__=='__main__':


    # test Particles() here
    particles = Particles(N=100)
    # test NbodySimulation(particles) here
    sim = NbodySimulation(particles=particles)
    sim.setup(G=6.67e-8, io_freq=1)
    sim.evolve(dt = 0.01, tmax=10)
    print(sim.G)
    print("Done")