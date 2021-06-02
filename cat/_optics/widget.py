#-----------------------------------------------------------------------------#
# Copyright (c) 2020 Institute of High Energy Physics Chinese Academy of 
#                    Science
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - heps hard x-ray scattering beamline (b4)"
__date__     = "date : 05.12.2021"
__version__  = "beta-0.3"
__email__    = "xuhan@ihep.ac.cn"


"""
propagate: the propagatation of coherent modes.

functions: 
           
classes: 
"""

#-----------------------------------------------------------------------------#
# library

import numpy as np
import matplotlib.pyplot as plt

from scipy import interpolate

#-----------------------------------------------------------------------------#
# function

#-----------------------------------------------------------------------------#
# class

#-------------------------------------------------------
# plot the results

class plot_optic(object):

    """
    description: plot the properties of optics
    
    methods: intensity, i1d, csd, sdc and coherent modes.
    """
    
    def __init__(self, optic):

        self.optic = optic

    #---------------------------------------------------
    # plot the intensity, 2d and 1d

    def intensity(self):

        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize = (12, 4))
        
        i2d = ax0.pcolor(self.optic.xtick * 1e6, self.optic.ytick * 1e6, self.optic.intensity)
        ax0.set_title('intenisty')
        fig.colorbar(i2d, ax = ax0)

        ax1.plot(self.optic.ytick * 1e6, np.sum(self.optic.intensity, 1))
        ax1.set_title('intensity_y')

        ax2.plot(self.optic.xtick * 1e6, np.sum(self.optic.intensity, 0))
        ax2.set_title('intensity_x')

    #---------------------------------------------------
    # plot the intensity, 1d

    def i1d(self):

        plt.figure(figsize = (4, 4))

        ix, = plt.plot(self.optic.xtick * 1e6, np.sum(self.optic.intensity, 1))
        iy, = plt.plot(self.optic.ytick * 1e6, np.sum(self.optic.intensity, 0))

    #---------------------------------------------------
    # plot the coherent modes

    def cmode(self, n = (3, 3)):
        
        fig, axes = plt.subplots(
            int(n[0]), int(n[1]), figsize = (4*int(n[1]), 4*int(n[0]))
            )
        idx = 0

        for i0 in range(int(n[0])):
            for i1 in range(int(n[1])):

                if int(n[0]) == 1: ax_i = axes[i1]
                else: ax_i = axes[i0, i1]

                ax_i.pcolor(
                    self.optic.xtick * 1e6, self.optic.ytick * 1e6, 
                    np.abs(self.optic.cmode[idx])**2
                    )
                idx += 1
    
    #---------------------------------------------------
    # plot the ratio

    def ratio(self, n = 50):

        plt.figure(figsize = (4, 4))        
        plt.scatter(
            range(n), 
            np.array(self.optic.ratio[0 : n])**2 / np.sum(np.array(self.optic.ratio[0 : n])**2)
            )
        plt.plot(
            range(n),
            np.array(self.optic.ratio[0 : n])**2 / np.sum(np.array(self.optic.ratio[0 : n])**2)
            )
    
    #---------------------------------------------------
    # plot longitudinal direction
    
    def longitude(self, zcoor):
        
        ztick = np.linspace(zcoor[0], zcoor[1], zcoor[2])
        
        zx_sum = np.sum(np.abs(np.array(self.optic.cmode))**2, 1)
        zy_sum = np.sum(np.abs(np.array(self.optic.cmode))**2, 2)
        
        fig, [ax_zx, ax_zy] = plt.subplots(1, 2, figsize = (8, 4))
        
        zx = ax_zx.pcolor(self.optic.xtick*1e6, ztick, zx_sum)
        ax_zx.set_title('zx plane')
        fig.colorbar(zx, ax = ax_zx)
        
        zy = ax_zy.pcolor(self.optic.ytick*1e6, ztick, zy_sum)
        ax_zy.set_title('zy plane')
        fig.colorbar(zy, ax = ax_zy)
        
    #---------------------------------------------------
    # plot the cross spectral density

    def csd(self):

        fig, axes = plt.subplots(2, 2, figsize = (8, 8))

        csd2x = axes[0, 0].pcolor(
            self.optic.xtick * 1e6, self.optic.ytick * 1e6, np.abs(self.optic.csd2x)
            )
        axes[0, 0].set_title('csd2x')
        fig.colorbar(csd2x, ax = axes[0, 0])

        axes[0, 1].plot(
            2*self.optic.xtick[int(self.optic.xcount/2) : -1] * 1e6, 
            self.optic.csd1x[int(self.optic.xcount/2) : -1]
            )
        axes[0, 1].set_title('csd1x')

        csd2y = axes[1, 0].pcolor(
            self.optic.xtick * 1e6, self.optic.ytick * 1e6, np.abs(self.optic.csd2y)
            )
        axes[1, 0].set_title('csd2y')
        fig.colorbar(csd2y, ax = axes[1, 0])

        axes[1, 1].plot(
            2*self.optic.ytick[int(self.optic.ycount/2) : -1] * 1e6,
            self.optic.csd1y[int(self.optic.ycount/2) : -1]
            )
        axes[1, 1].set_title('csd1y')

    #---------------------------------------------------
    # plot the sdc

    def sdc(self):

        fig, axes = plt.subplots(2, 2, figsize = (8, 8))

        sdc2x = axes[0, 0].pcolor(
            self.optic.xtick * 1e6, self.optic.ytick * 1e6, np.abs(self.optic.sdc2x)
            )
        axes[0, 0].set_title('sdc2x')
        fig.colorbar(sdc2x, ax = axes[0, 0])

        axes[0, 1].plot(
            2*self.optic.xtick[int(self.optic.xcount/2) : -1] * 1e6, 
            self.optic.sdc1x[int(self.optic.xcount/2) : -1] 
            )
        axes[0, 1].set_title('sdc1x')

        sdc2y = axes[1, 0].pcolor(
            self.optic.xtick * 1e6, self.optic.ytick * 1e6, 
            np.abs(self.optic.sdc2y)
            )
        axes[1, 0].set_title('sdc2y')
        fig.colorbar(sdc2y, ax = axes[1, 0])

        axes[1, 1].plot(
            2*self.optic.ytick[int(self.optic.ycount/2) : -1] * 1e6,
            self.optic.sdc1y[int(self.optic.ycount/2) : -1]
            )
        axes[1, 1].set_title('sdc1y')

    #---------------------------------------------------
    # save the figure

    def save(self, fig_name):

        plt.savefig(fig_name + '.png', dpi = 500)



#-------------------------------------------------------
# add error to the optics

class mirror_error(object):

    """
    description: load the error to the optic
    
    methods: 
    """
        
    def __init__(self, xcoor, ycoor, energy = 12.4):
        
        self.xstart, self.xend, self.xcount = xcoor
        self.ystart, self.yend, self.ycount = ycoor
        
        self.xtick = np.linspace(self.xstart, self.xend, self.xcount)
        self.ytick = np.linspace(self.ystart, self.yend, self.ycount)
        
        self.plane = np.zeros((self.ycount, self.xcount))
        
        self.wavelength = 1e-9 * 1239.8 / (1e3 * energy)
        
    #---------------------------------------------------
    # transform the dcm thermal deformation
    
    def dcm_deform(
        self, xcoor, ycoor, error_file = "test.dat", angle = 0.1521, 
        direction = 'h'
        ):
        
        deform = np.loadtxt(error_file)
        
        xstart, xend, xpixel = xcoor
        ystart, yend, ypixel = ycoor
        
        xcount = int(abs(xstart - xend) / xpixel)
        ycount = int(abs(ystart - yend) / ypixel)
        
        locx0 = np.argmin(np.abs(self.xtick - xstart))
        locx1 = np.argmin(np.abs(self.xtick - xend))
        locy0 = np.argmin(np.abs(self.ytick - ystart))
        locy1 = np.argmin(np.abs(self.ytick - yend))
            
        if (xstart < self.xstart or xend > self.xend or 
            ystart < self.ystart or yend > self.yend):
            
            raise ValueError("The range of error should smaller than the optic plane.")
        
        if direction == 'h':
            
            dxtick, dytick = np.meshgrid(
                np.linspace(
                    xstart/np.sin(angle), xend/np.sin(angle), xcount
                    ),
                np.linspace(ystart, yend, ycount)
                )
            
        elif direction == 'v':
            
            dxtick, dytick = np.meshgrid(
                np.linspace(xstart, xend, xcount),
                np.linspace(
                    ystart/np.sin(angle), yend/np.sin(angle), ycount
                    )
                )
            
        deform = interpolate.griddata(
            (deform[:, 0], deform[:, 1]), deform[:, 2], (dxtick, dytick), 
            method = 'cubic'
            )
        deform -= np.min(deform)
        phase = 2*np.pi * 2*np.sin(angle) * deform / self.wavelength
        
        plane = np.copy(self.plane)
        plane[locy0 : locy1, locx0 : locx1] = phase
        
        return plane
    
    #---------------------------------------------------
    # transform the plane error
    
    def plane_error(
        self, estart, eend, ecount, error_file = "test.dat", angle = 0.03, 
        direction = 'h'
        ):
        
        if direction == 'h':
            
            pxstart = self.xstart
            pxend = self.xend
            count = self.xcount
            
        elif direction == 'v':
            
            pxstart = self.ystart
            pxend = self.yend
            count = self.ycount
            
        error = np.loadtxt(error_file)
        
        func = interpolate.interp1d(np.linspace(estart, eend, ecount), error)
        
        if estart*np.sin(angle) > pxstart and eend*np.sin(angle) < pxend:
            
            line = np.zeros(ecount)
            etick = np.linspace(pxstart, pxend, count)
            
            start = np.argmin(np.abs(etick - estart*np.sin(angle)))
            end = np.argmin(np.abs(etick - eend*np.sin(angle)))
            
            line[start + 1 : end - 1] = (
                2*np.pi * 2*np.sin(angle) *
                func(etick[start + 1 : end - 1]) / 
                self.wavelength
                )
        
        elif estart*np.sin(angle) < pxstart and eend*np.sin(angle) > pxend:
            
            line = np.linspace(pxstart, pxend, count)
            line = (
                2*np.pi * 2*np.sin(angle) * func(line) / 
                self.wavelength
                )
            
        return line
    
    #---------------------------------------------------
    # crl error
    
    def crl_error(
        self, 
        xcoor, ycoor,
        efile_xcount = 121, efile_ycount = 121, delta = 2.216e-6, ne = 1,
        error_file = "test.dat"
        ):
        
        xstart, xend, xpixel = xcoor
        ystart, yend, ypixel = ycoor
        
        xcount = int(abs(xstart - xend) / xpixel)
        ycount = int(abs(ystart - yend) / ypixel)
        
        locx0 = np.argmin(np.abs(self.xtick - xstart))
        locx1 = np.argmin(np.abs(self.xtick - xend))
        locy0 = np.argmin(np.abs(self.ytick - ystart))
        locy1 = np.argmin(np.abs(self.ytick - yend))
        
        error = np.reshape(
            np.loadtxt(error_file), (efile_xcount, efile_ycount)
            )
        
        error_func = interpolate.interp2d(
            np.linspace(xstart, xend, efile_xcount),
            np.linspace(ystart, yend, efile_ycount),
            error
            )
        
        error = error_func(
            np.linspace(xstart, xend, xcount),
            np.linspace(ystart, yend, ycount)
            )
        
        phase = 2*np.pi * delta*error / self.wavelength
        
        plane = np.copy(self.plane)
        plane[locy0 : locy1, locx0 : locx1] = phase * ne
        
        return plane
        
        
        
        
        
        
        
        
        
    

        

