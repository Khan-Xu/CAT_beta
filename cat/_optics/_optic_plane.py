#-----------------------------------------------------------------------------#
# Copyright (c) 2020 Institute of High Energy Physics Chinese Academy of 
#                    Science
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - heps hard x-ray scattering beamline (b4)"
__date__     = "date : 05.12.2021"
__version__  = "beta-0.3"
__email__    = "xuhan@ihep.ac.cn"

"""
_source_utils: Source construction support.

Functions: None
           
Classes  : _optic_plane - the geometry structure of optics
"""

#-----------------------------------------------------------------------------#
# library

import pickle
import os
import math

import numpy as np
import h5py as h5

from scipy import interpolate
from copy import deepcopy

#-----------------------------------------------------------------------------#
# constant

#-----------------------------------------------------------------------------#
# function

def _locate(ticks, value):
    
    """
    return the nearest location of value among ticks
    
    args: ticks - numpy array of data.
          value - a value to be located
    """
    
    if value > np.max(ticks) or value < np.min(ticks):
        raise ValueError("The given value is out of range.")
    else:
        return np.argmin(np.abs(ticks - value))

#-----------------------------------------------------------------------------#
# class

class _op(object):
    
    #--------------------------------------------------------------------------
    # the initialization of the optic plane

    def __init__(
        self, 
        optic = None, optic_file = None, xcoor = None, ycoor = None, 
        name = "optic", 
        n_vector = 0, i_vector = None, position = 0,
        wavelength = None, ratio = None
        ):
        
        #------------------------------------------------------
        # geometery parameters
        
        if optic_file != None:

            with h5.File(self.optic_name + '.h5') as optic_file:

                self.xstart = np.array(optic_file["optic_plane/xstart"])
                self.xend = np.array(optic_file["optic_plane/xend"])
                self.xcount = int(np.array(optic_file["optic_plane/xcount"]))

                self.ystart = np.array(optic_file["optic_plane/ystart"])
                self.yend = np.array(optic_file["optic_plane/yend"])
                self.ycount = int(np.array(optic_file["optic_plane/ycount"]))

                self.wavelength = np.array(optic_file["optic_plane/wavelength"])
                self.position = np.array(optic_file["optic_plane/position"])
                self.n_vector = np.array(optic_file["optic_plane/n_vector"])

                self.cmode = list()
                self.ratio = list()
                self.center = [0, 0]

                if i_vector == None:
                    
                    for i in range(n_vector):
                        self.cmode.append(
                            np.reshape(
                                np.array(optic_file["coherence/coherent_mode"])[:, i], 
                                (self.n_row, self.n_column)
                                )
                            )
                        self.ratio.append(np.array(optic_file["coherence/ratio"])[i])
                else:
                    self.cmode = [np.array(optic_file["coherence/coherent_mode"])[:, i_vector]]
                    self.ratio = [np.array(optic_file["coherence/ratio"])[i_vector]]

                self.csd2x = np.array(optic_file["coherence/csd2x"])
                self.csd1x = np.array(optic_file["coherence/csd1x"])
                self.csd2y = np.array(optic_file["coherence/csd2y"])
                self.csd1y = np.array(optic_file["coherence/csd1y"])
                self.sdc2x = np.array(optic_file["coherence/sdc2x"])
                self.sdc1x = np.array(optic_file["coherence/sdc1x"])
                self.sdc2y = np.array(optic_file["coherence/sdc2y"])
                self.sdc1y = np.array(optic_file["coherence/sdc1y"])
                   
        else:

            if optic != None:
                
                self.xstart = np.copy(optic.xstart)
                self.xend   = np.copy(optic.xend)
                self.ystart = np.copy(optic.ystart)
                self.yend   = np.copy(optic.yend)

                self.xcount = int(np.copy(optic.xcount))
                self.ycount = int(np.copy(optic.ycount))

                self.n = np.copy(optic.n)

                self.wavelength = np.copy(optic.wavelength)
                self.ratio = np.copy(optic.ratio)

            else:

                self.xstart, self.xend, xcount = xcoor
                self.ystart, self.yend, ycount = ycoor

                self.xcount = int(xcount)
                self.ycount = int(ycount)

                self.wavelength = wavelength
                self.ratio = ratio

            self.csd2x = None
            self.csd2y = None
            self.csd1x = None
            self.csd1y = None
            
            # cal spectral degree of coherence
            
            self.sdc2x = None
            self.sdc2y = None
            self.sdc1x = None
            self.sdc1y = None

            #------------------------------------------------------
            # construct the optic plane
            
            self.xtick = np.linspace(self.xstart, self.xend, self.xcount)
            self.ytick = np.linspace(self.ystart, self.yend, self.ycount)
            
            self.gridx, self.gridy = np.meshgrid(self.xtick, self.ytick)
            
            self.xpixel = np.abs(self.xstart - self.xend) / self.xcount
            self.ypixel = np.abs(self.ystart - self.yend) / self.ycount
            
            self.n_row = self.ycount
            self.n_column = self.xcount

            self.position = position
            self.name = name

            self.center = [0, 0]
        
            #------------------------------------------------------
            # load coherent modes

            self.cmode = list()

            for i in range(self.n): 
                self.cmode.append(np.ones((self.n_row, self.n_column), dtype = complex))
    
    #--------------------------------------------------------------------------
    # the interpolate of the optic plane

    def interp(self, xpixel, ypixel, method = 'ri'):
        
        """
        interpolate the coherent mode data. 
        
        args: xpixel - the sampling density along x.
              ypixel - the sampling density along y.
              method - methods to interpolate the plane.
              
        return: interped data.
        """
        
        xcount = int((self.xend - self.xstart)/xpixel)
        ycount = int((self.yend - self.ystart)/ypixel)
        
        xcount += np.abs(xcount % 2 - 1)
        ycount += np.abs(ycount % 2 - 1)

        # remap the plane
            
        xtick = np.linspace(self.xstart, self.xend, xcount)
        ytick = np.linspace(self.ystart, self.yend, ycount)
        
        #------------------------------------------------------
        # interpolate real and image data

        if method == 'ri':
            
            for i in range(self.n):
            
                freal = interpolate.interp2d(
                    self.xtick, self.ytick, np.real(self.cmode[i]), 
                    kind = 'cubic')
                fimag = interpolate.interp2d(
                    self.xtick, self.ytick, np.imag(self.cmode[i]),
                    kind = 'cubic')
        
                self.cmode[i] = freal(xtick, ytick) + 1j*fimag(xtick, ytick)
        
        #------------------------------------------------------
        # interpolate abs and phase data

        elif method == 'ap':
            
            for i in range(self.n):
            
                fabs = interpolate.interp2d(
                    self.xtick, self.ytick, np.abs(self.cmode[i]), 
                    kind = 'cubic')
                fangle = interpolate.interp2d(
                    self.xtick, self.ytick, np.angle(self.cmode[i]),
                    kind = 'cubic')
        
                self.cmode[i] = fabs(xtick, ytick)*np.exp(1j*fangle(xtick, ytick))
        
        #------------------------------------------------------
        # interpolate abs and phase data after unwrap data

        elif method == 'unwrap':
                
            from skimage.restoration import unwrap_phase
            
            for i in range(self.n):
                
                unwraped_phase = unwrap_phase(np.angle(self.cmode[i]))
                
                fabs = interpolate.interp2d(
                    self.xtick, self.ytick, np.abs(self.cmode[i]), 
                    kind = 'cubic'
                    )
                
                fangle = interpolate.interp2d(
                    self.xtick, self.ytick, unwraped_phase,
                    kind = 'cubic'
                    )
        
                self.cmode[i] = fabs(xtick, ytick)*np.exp(1j*fangle(xtick, ytick))
        
        # reset the optic plane

        self.xcount = int(xcount)
        self.ycount = int(ycount)
        self.xpixel = xpixel
        self.ypixel = ypixel
        self.xtick  = xtick
        self.ytick  = ytick
        
        self.gridx, self.gridy = np.meshgrid(self.xtick, self.ytick)

        self.n_column = self.xcount
        self.n_row = self.ycount
        
    #--------------------------------------------------------------------------
    # calcualte the center of the coherent mode

    def cal_center(self):
        
        """
        calcualte the center of the modes.
        """
        intensity = np.abs(self.cmode[0])
        
        self.center = [self.xtick[np.argmax(np.sum(intensity, 0))], 
                       self.ytick[np.argmax(np.sum(intensity, 1))]]

     #--------------------------------------------------------------------------
     # add the mask of to the coheret mode
            
    def mask(self, xcoor = None, ycoor = None, r = None, s = "b"):
         
        """
        construct a mask. The shape could be box or circle
        
        args: xcoor - xstart, xend.
              ycoor - ystart, yend.
              r - radicus of mask.
              s - "b" for box; "c" for circle.
              
        return: mask.
        """
        
        mask = np.zeros((self.n_row, self.n_column))
        
        if s == "b":
        
            # construct mask
            
            mask[
                _locate(self.ytick, ycoor[0] + self.center[1]) : _locate(self.ytick, ycoor[1] + self.center[1]), 
                _locate(self.xtick, xcoor[0] + self.center[0]) : _locate(self.xtick, xcoor[1] + self.center[0])
                ] = 1
        
        elif s == "c":
            
            mask[np.sqrt(self.gridx**2 + self.gridy**2) < r] = 1
            
        for i in range(self.n): self.cmode[i] = self.cmode[i] * mask
    
    #--------------------------------------------------------------------------
    # shrink the optic plane

    def shrink(self, xcoor = None, ycoor = None):
        
        """
        shrink the optic plane to a center axis.
        
        args: xcoor - xstart, xend.
              ycoor - ystart, yend.
        """
        
        locxs = (0 if xcoor == None else _locate(self.xtick, xcoor[0] + self.center[0]))
        locxe = (
            self.xcount if xcoor == None else 
            _locate(self.xtick, xcoor[1] + self.center[0])
            )

        locys = (0 if ycoor == None else _locate(self.ytick, ycoor[0] + self.center[1]))
        locye = (
            self.ycount if ycoor == None else 
            _locate(self.ytick, ycoor[1] + self.center[1])
            )

        self.xstart = xcoor[0]
        self.xend   = xcoor[1]
        self.ystart = ycoor[0]
        self.yend   = ycoor[1]
        
        self.xcount = int(locxe - locxs)
        self.ycount = int(locye - locys)
        
        self.n_row = self.ycount
        self.n_column = self.xcount
        
        self.xtick = np.linspace(self.xstart, self.xend, self.xcount)
        self.ytick = np.linspace(self.ystart, self.yend, self.ycount)
        
        self.gridx, self.gridy = np.meshgrid(self.xtick, self.ytick)
        
        for i in range(self.n):
            self.cmode[i] = self.cmode[i][locys : locye, locxs : locxe]
    
    #--------------------------------------------------------------------------
    # expand the optic plane

    def expand(self, xcoor = None, ycoor = None):
        
        """
        expand the optic plane to a center axis.
        
        args: xcoor - xstart, xend.
              ycoor - ystart, yend.
        """

        eplx = int(np.abs(xcoor[0] + self.center[0] - self.xstart)/self.xpixel)
        eprx = int(np.abs(xcoor[1] + self.center[0] - self.xend)/self.xpixel)
        eply = int(np.abs(ycoor[0] + self.center[1] - self.ystart)/self.ypixel)
        epry = int(np.abs(ycoor[1] + self.center[1] - self.yend)/self.ypixel)
        
        xcount = eplx + eprx + self.xcount
        ycount = eply + epry + self.ycount
        
        xstart = self.xstart - eplx * self.xpixel
        xend = eprx * self.xpixel + self.xend
        ystart = self.ystart - eply * self.ypixel
        yend = epry * self.ypixel + self.yend
        
        cmode = np.zeros((ycount, xcount), dtype = complex)
        
        for i in range(self.n):

            cmode[eply : eply + self.ycount, eplx : eplx + self.xcount] = self.cmode[i]
            self.cmode[i] = np.copy(cmode)
            cmode = np.zeros((ycount, xcount), dtype = complex)
        
        self.xstart = xstart
        self.xend = xend
        self.ystart = ystart
        self.yend = yend
        self.xcount = xcount
        self.ycount = ycount
        self.xtick = np.linspace(xstart, xend, xcount)
        self.ytick = np.linspace(ystart, yend, ycount)
        self.gridx, self.gridy = np.meshgrid(self.xtick, self.ytick)
        
        self.n_row = self.ycount
        self.n_column = self.xcount
    
    #--------------------------------------------------------------------------
    # calcualte the csd

    def cal_csd(self):
        
        """
        calculate cross spectral density of the optic
        
        args: None
        
        return: None
        """
        
        # get center slice
        
        cmodex = np.zeros((self.n, self.n_column), dtype = np.complex128)
        cmodey = np.zeros((self.n_row, self.n), dtype = np.complex128)
        
        for i in range(self.n):
            
            cmodex[i, :] = (
                self.cmode[i][int(self.n_row/2), :] + self.cmode[i][math.ceil(self.n_row/2), :]
                ) * self.ratio[i] / 2

            cmodey[:, i] = (
                self.cmode[i][:, int(self.n_column/2)] + self.cmode[i][:, math.ceil(self.n_column/2)]
                ) * self.ratio[i] / 2
        
        # calcualte csd
        
        self.csd2x = np.dot(cmodex.T.conj(), cmodex)
        self.csd2y = np.dot(cmodey.conj(), cmodey.T)
        self.csd1x = np.abs(np.diag(np.fliplr(self.csd2x)))
        self.csd1y = np.abs(np.diag(np.fliplr(self.csd2y)))
    
        # cal spectral degree of coherence
        
        ix2 = np.zeros((self.xcount, self.xcount))
        iy2 = np.zeros((self.ycount, self.ycount))
        
        for i in range(self.xcount): ix2[i, :] = np.diag(np.abs(self.csd2x))
        for i in range(self.ycount): iy2[i, :] = np.diag(np.abs(self.csd2y))
        
        self.sdc2x = self.csd2x / np.sqrt(ix2 * ix2.T)
        self.sdc2y = self.csd2y / np.sqrt(iy2 * iy2.T)
        
        self.sdc1x = np.abs(np.diag(np.fliplr(self.sdc2x)))
        self.sdc1y = np.abs(np.diag(np.fliplr(self.sdc2y)))
    
    #--------------------------------------------------------------------------
    # calcualte the intensity

    def cal_i(self):
        
        self.intensity = np.zeros((self.n_row, self.n_column))
        
        for i, ic in enumerate(self.cmode):  
            self.intensity = self.intensity + self.ratio[i]**2 * np.abs(ic)**2

    #--------------------------------------------------------------------------
    # svd process

    def _svd(self):
        
        cmodes = np.zeros((self.n, self.xcount * self.ycount), dtype = np.complex128)
        
        for i in range(self.n):
            cmodes[i, :] = np.reshape(self.cmode[i], (self.xcount * self.ycount)) * self.ratio[i]
        
        import scipy.sparse.linalg as ssl
        
        svd_matrix = cmodes.T
        vector, value, evolution = ssl.svds(svd_matrix, k = self.n - 2)
        
        eig_vector = np.copy(vector[:, ::-1], order = 'C')
        value = np.copy(np.abs(value[::-1]), order = 'C')
        
        self.cmode = list()
        self.ratio = list()
        
        for i in range(self.n - 2):
            self.cmode.append(np.reshape(eig_vector[:, i], (self.xcount, self.ycount)))
            self.ratio.append(value[i])
    
    #--------------------------------------------------------------------------
    # cmd process
    
    def _cmd(self):
        
        cmodes = np.zeros((self.n, self.xcount * self.ycount), dtype = np.complex128)
        
        for i in range(self.n):
            cmodes[i, :] = np.reshape(self.cmode[i], (self.xcount * self.ycount)) * self.ratio[i] 
            
        import scipy.sparse.linalg as ssl
        
        csd = np.dot(cmodes.T.conj(), cmodes)
        
        eig_value, eig_vector = ssl.eigsh(csd, k = self.n)
        eig_vector = np.reshape(eig_vector, (self.n_row, self.n_column, self.n))
        
        cmode = np.zeros((self.n_row, self.n_column), dtype = np.complex128)
        
        self.ratio = list()
        
        for i in range(self.n):
            
            self.cmode[i] = np.copy(cmode)
            self.cmode[i] = eig_vector[:, :, i]
            self.ratio.append(eig_value[i])
            
    #--------------------------------------------------------------------------
    # svd process

    def save_h5(self):

        """
        save all the properties to h5 file
        """

        if os.path.isfile(self.name + '.h5'): os.remove(self.name + '.h5')

        with h5.File(self.optic_name + '.h5') as optic_file:

            parameters = optic_file.create_group("optic_plane")

            parameters.create_dataset('xstart', data = self.xstart)
            parameters.create_dataset('xend', data = self.xend)
            parameters.create_dataset('xcount', data = self.xcount)

            parameters.create_dataset('ystart', data = self.ystart)
            parameters.create_dataset('yend', data = self.yend)
            parameters.create_dataset('ycount', data = self.ycount)
            
            parameters.create_dataset('wavelength', data = self.wavelength)
            parameters.create_dataset('position', data = self.position)
            parameters.create_dataset('n_vector', data = self.n_vector)

            coherence = optic_file.create_group("coherence")

            coherence.create_dataset('coherent_mode', data = np.array(self.cmode))
            coherence.create_dataset('ratio', data = np.array(self.ratio))
            coherence.create_dataset('csd2x', data = np.array(self.csd2x))
            coherence.create_dataset('csd2y', data = np.array(self.csd2y))
            coherence.create_dataset('csd1x', data = np.array(self.csd1x))
            coherence.create_dataset('csd1y', data = np.array(self.csd1y))

            coherence.create_dataset('csd2x', data = np.array(self.sdc2x))
            coherence.create_dataset('csd2y', data = np.array(self.sdc2y))
            coherence.create_dataset('csd1x', data = np.array(self.sdc1x))
            coherence.create_dataset('csd1y', data = np.array(self.sdc1y))

    def save_pickle(self):
        
        """
        save all the properites to pickle file.
        """
        
        pickle.dump(self, open(self.name + '.pkl', 'wb'), True)
            