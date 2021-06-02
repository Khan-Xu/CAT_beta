#-----------------------------------------------------------------------------#
# Copyright (c) 2020 Institute of High Energy Physics Chinese Academy of 
#                    Science
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - heps hard x-ray scattering beamline (b4)"
__date__     = "date : 05.02.2021"
__version__  = "beta-0.2"


"""
_source_utils: Source construction support.

Functions: None
           
Classes  : _optic_plane - the geometry structure of optics
"""

#-----------------------------------------------------------------------------#
# library

import os
import numpy as np
import h5py as h5

from copy import deepcopy
from cat._optics._optic_plane import _locate
from cat._optics._optic_plane import _op

#-----------------------------------------------------------------------------#
# constant

#-----------------------------------------------------------------------------#
# function

#-----------------------------------------------------------------------------#
# class

#--------------------------------------------------------------------------
# the class of source

class source(_op):
    
    """
    construct the class of source.
    
    methods: 
    """
    
    def __init__(
        self, 
        source = None, file_name = "test.h5", name = "source",
        n_vector = 0, i_vector = None, position = 0,
        offx = 0, offy = 0, rotx = 0, roty = 0
        ):
        
        if os.path.isfile(file_name):
            pass
        else:
            raise ValueError("The source file don't exist.")

        with h5.File(file_name, 'a') as f:
            
            # the geometry structure of the source plane
            
            self.xstart = np.array(f["description/xstart"])
            self.xend   = np.array(f["description/xfin"])
            self.xcount = int(np.array(f["description/nx"]))
            self.ystart = np.array(f["description/ystart"])
            self.yend   = np.array(f["description/yfin"])
            self.ycount = int(np.array(f["description/ny"]))
            
            self.location = np.array(f["description/screen"])
            num_vector = int(np.array(f["description/n_vector"]))
            self.n_row = np.copy(self.ycount)
            self.n_column = np.copy(self.xcount)
            self.n = n_vector

            # the cooridnate of wavefront
            
            self.xtick = np.linspace(self.xstart, self.xend, self.xcount)
            self.ytick = np.linspace(self.ystart, self.yend, self.ycount)
            
            self.gridx, self.gridy = np.meshgrid(self.xtick, self.ytick)
            
            self.xpixel = np.abs(self.xstart - self.xend)/self.xcount
            self.ypixel = np.abs(self.ystart - self.yend)/self.ycount
            
            self.plane = np.zeros((self.n_row, self.n_column), dtype = complex)
            
            # the undulator parameters of source
            
            self.sigma_x0 = np.array(f["description/sigma_x0"])
            self.sigma_y0 = np.array(f["description/sigma_y0"])
            self.sigma_xd = np.array(f["description/sigma_xd"])
            self.sigma_yd = np.array(f["description/sigma_yd"])
            self.es = np.array(f["description/energy_spread"])
            self.current = np.array(f["description/current"])
            self.energy = np.array(f["description/hormonic_energy"])
            self.n_electron = np.array(f["description/n_electron"])
            
            # the coherence properites of source.
            
            self.position = self.location
            self.wavelength = np.array(f["description/wavelength"])
            self.ratio = np.array(f["coherence/eig_value"])
            
            if i_vector != None:
                cmode = np.array(f["coherence/eig_vector"])[:, i_vector]
                self.cmode = [np.reshape(cmode, (self.n_row, self.n_column))]
            else:
                self.n = n_vector
                self.cmode = [
                    np.reshape(
                        np.array(f["coherence/eig_vector"])[:, i], 
                        (self.n_row, self.n_column)) for i in range(n_vector)
                    ]
            
            self.name = name
            
            #---------------------------------------------------
            # add vibration of the source

            # add rotation of source
            
            rotx_phase = np.exp(
                -1j*(2*np.pi/self.wavelength)*
                (rotx*self.gridx - (1 - np.cos(rotx))*self.position)
                )
            roty_phase = np.exp(
                -1j*(2*np.pi/self.wavelength)*
                (roty*self.gridy - (1 - np.cos(roty))*self.position)
                )                

            # add vibration of the source
    
            offx = offx + np.sin(rotx) * self.position
            offy = offy + np.sin(roty) * self.position
            
            if offx > 0:
                loclx0 = _locate(self.xtick, self.xstart + offx)
                locrx0 = self.xcount - loclx0
                loclx1 = 0
                locrx1 = self.xcount - 2*loclx0
                
            elif offx <= 0:
                locrx0 = _locate(self.xtick, self.xend + offx)
                loclx0 = self.xcount - locrx0
                loclx1 = self.xcount - 2*locrx0
                locrx1 = self.xcount
            
            if offy > 0:
                locly0 = _locate(self.ytick, self.ystart + offy)
                locry0 = self.ycount - locly0
                locly1 = locly0
                locry1 = self.ycount - locly0
                
            elif offy <= 0:
                locry0 = _locate(self.ytick, self.yend + offy)
                locly0 = self.ycount - locry0
                locly1 = self.ycount - locry0
                locry1 = locry0
                
            for i in range(self.n):
            
                plane = deepcopy(self.plane)
                plane[locly0 : locry0, loclx0 : locrx0] = (
                    (self.cmode[i] * 
                     rotx_phase * roty_phase)[locly0 : locry0, loclx1 : locrx1]
                    )
                self.cmode[i] = plane

#--------------------------------------------------------------------------
# the class of a optic screen

class screen(_op):
    """
    construct the class of screen.
    
    methods: 
    """
    
    def __init__(
        self,
        optic = None, optic_file = None, xcoor = None, ycoor = None,
        name = "screen",
        n_vector = 0, i_vector = None, position = 0, 
        wavelength = None, ratio = None,
        offx = 0, offy = 0, rotx = 0, roty = 0,
        error = 0, direction = 'v'
        ):
        
        super().__init__(
            optic = optic, optic_file = optic_file, 
            xcoor = xcoor, ycoor = ycoor,
            name = name, n_vector = n_vector, i_vector = i_vector, 
            position = position, wavelength = wavelength, 
            ratio = ratio
            )

        #---------------------------------------------------
        # add roation to the screen

        rotx_phase = np.exp(
            -1j*(2*np.pi/self.wavelength)*np.sin(rotx)*self.gridx
            )
        roty_phase = np.exp(
            -1j*(2*np.pi/self.wavelength)*np.sin(roty)*self.gridy
            )
        
        for i in range(self.n):
            self.cmode[i] = rotx_phase*roty_phase

        #---------------------------------------------------
        # add vibration to the screen
            
        if offx > 0:
            loclx0 = _locate(self.xtick, self.xstart + offx)
            locrx0 = self.xcount - loclx0
            loclx1 = 0
            locrx1 = self.xcount - 2*loclx0
            
        elif offx <= 0:
            locrx0 = _locate(self.xtick, self.xend + offx)
            loclx0 = self.xcount - locrx0
            loclx1 = self.xcount - 2*locrx0
            locrx1 = self.xcount
        
        if offy > 0:
            locly0 = _locate(self.ytick, self.ystart + offy)
            locry0 = self.ycount - locly0
            locly1 = locly0
            locry1 = self.ycount - locly0
            
        elif offy <= 0:
            locry0 = _locate(self.ytick, self.yend + offy)
            locly0 = self.ycount - locry0
            locly1 = self.ycount - locry0
            locry1 = locry0
            
        for i in range(self.n):
        
            plane = np.zeros((self.n_row, self.n_column), dtype = complex)
            plane[:, loclx0 : locrx0] = (
                (self.cmode[i] * 
                 rotx_phase * roty_phase)[:, loclx1 : locrx1]
                )
            self.cmode[i] = plane
        
        #---------------------------------------------------
        # add error

        if not isinstance(error, np.ndarray): error_phase = 1

        else:
            if len(np.shape(error)) == 2:
                error_phase = np.exp(1j*error)
            
            elif len(np.shape(error)) == 1:

                e = np.zeros((self.n_row, self.n_column), dtype = complex)
                
                if direction == 'h':
                    for i in range(self.n_row): e[i, :] = np.exp(1j*error)
                elif direction == 'v':
                    for i in range(self.n_column): e[:, i] = np.exp(1j*error)
                    
                error_phase = np.exp(1j*e)
                
        for i in range(self.n):
            self.cmode[i] = self.cmode[i] * error_phase

#--------------------------------------------------------------------------
# the class of crl

class crl(_op):
    """
    construct the class of crl.
    
    methods: 
    """
    
    def __init__(
        self, 
        optic = None, optic_file = None, xcoor = None, ycoor = None, 
        name = "crl",
        n_vector = 0, i_vector = None, position = 0,
        wavelength = None, ratio = None,
        nlens = 0, delta = 2.216e-6, 
        rx = 0, ry = 0,
        offx = 0, offy = 0, rotx = 0, roty = 0, 
        error = 0
        ):
    
        
        super().__init__(
            optic = optic, optic_file = optic_file, 
            xcoor = xcoor, ycoor = ycoor,
            name = name, n_vector = n_vector, i_vector = i_vector, 
            position = position, wavelength = wavelength,
            ratio = ratio
            )
        
        #---------------------------------------------------
        # the focus length of lens
        
        self.focus_x = rx/(2*nlens*delta) if rx != 0 else 1e20
        self.focus_y = ry/(2*nlens*delta) if ry != 0 else 1e20
        
        #---------------------------------------------------
        # add vibration of source
        
        self.lens_phase = np.exp(
            1j*(2*np.pi/self.wavelength) *
            ((self.gridx + offx)**2/(2*self.focus_x) + 
             (self.gridy + offy)**2/(2*self.focus_y))
            )
        
        #---------------------------------------------------
        # add rotation of source
        
        rotx_phase = np.exp(
            -1j*(2*np.pi/self.wavelength)*np.sin(rotx)*self.gridx
            )
        roty_phase = np.exp(
            -1j*(2*np.pi/self.wavelength)*np.sin(roty)*self.gridy
            ) 
        
        #---------------------------------------------------
        # add error
        
        if not isinstance(error, np.ndarray): error_phase = 1
        
        else:
            if len(np.shape(error)) == 2:
                error_phase = np.exp(1j*error)
            else:
                raise TypeError("The dimension of the error should be two.")
            
        for i in range(self.n):
            self.cmode[i] = (
                self.lens_phase *
                rotx_phase * roty_phase *
                error_phase
                )

#--------------------------------------------------------------------------
# the class of kb mirror

class kb(_op):
    """
    construct the class of KB mirror. The effect of rocking and offset were
    considered.
    
    methods: 
    """

    def __init__(
        self, 
        optic = None, optic_file = None, xcoor = None, ycoor = None,
        name = "kb_mirror", direction = 'v', 
        n_vector = 0, i_vector = None, position = 0,
        wavelength = None, ratio = None,
        pfocus = 0, qfocus = 0, length = None, width = None, angle = 0,
        offset = 0,  rot = 0, error = 0
        ):
    
        super().__init__(
            optic = optic, optic_file = optic_file, 
            xcoor = None, ycoor = None,
            name = name, n_vector = n_vector, i_vector = i_vector, 
            position = position, wavelength = wavelength,
            ratio = ratio
            )
        
        #---------------------------------------------------
        # add vibration of source
        
        # T0DO: if the source of vibration is earth, this asumpation is wrong.
        # offx = np.sin(angle) * offx
        # offy = np.sin(angle) * offy
        
        rotx_phase = 1
        roty_phase = 1
        
        if direction == 'h':
            rotx_phase = np.exp(-1j*(2*np.pi/self.wavelength) *
                                np.sin(2*rot)*self.gridx)
        elif direction == 'v':
            roty_phase = np.exp(-1j*(2*np.pi/self.wavelength) *
                                np.sin(2*rot)*self.gridy)
        
        #---------------------------------------------------
        # add rotation of source
        
        # ideally the rocking angle of reflected light is double of rocking 
        # angle of mirror
        
        if direction == 'h':
            self.lens_phase = np.exp(
                1j*(2*np.pi/self.wavelength) *
                (np.sqrt((self.gridx + offset)**2 + pfocus**2) +
                 np.sqrt((self.gridx + offset)**2 + qfocus**2))
                )
        
        elif direction == 'v':
            self.lens_phase = np.exp(
                1j*(2*np.pi/self.wavelength) *
                (np.sqrt((self.gridy + offset)**2 + pfocus**2) +
                 np.sqrt((self.gridy + offset)**2 + qfocus**2))
                )
            
        #---------------------------------------------------
        # add error phase
        
        if not isinstance(error, np.ndarray): error_phase = 1
        
        else:
            if direction == 'v':
                e = np.zeros((self.n_row, self.n_column), dtype = float)
                for i in range(self.n_row): e[i, :] = error
            
            elif direction == 'h':
                e = np.zeros((self.n_row, self.n_column), dtype = float)
                for i in range(self.n_column): e[:, i] = error
                
            error_phase = np.exp(1j*e)
    
        for i in range(self.n):
            
            self.cmode[i] = self.cmode[i]*(
                self.lens_phase * 
                rotx_phase * roty_phase * error_phase
                )

#--------------------------------------------------------------------------
# the class of akb mirror

class akb(_op):
    """
    construct the class of KB mirror. The effect of rocking and offset were
    considered.
    
    methods: 
    """
    
    def __init__(
        self, 
        optic = None, optic_file = None, xcoor = None, ycoor = None, 
        name = "akb_mirror", direction = 'v', kind = 'ep',
        n_vector = 0, i_vector = None, position = 0,
        wavelength = None, ratio = None,
        pfocus = 0, qfocus = 0, afocus = 0, bfocus = 0,
        length = None, width = None, angle = 0,
        offset = 0,  rot = 0, error = None
        ):
    
        super().__init__(
            optic = optic, optic_file = optic_file, 
            xcoor = xcoor, ycoor = ycoor, 
            name = name, n_vector = n_vector, i_vector = i_vector,
            position = position, wavelength = wavelength, 
            ratio = ratio
            )
        
        #---------------------------------------------------
        # add vibration of source
        
        # T0DO: if the source of vibration is earth, this asumpation is wrong.
        # offx = np.sin(angle) * offx
        # offy = np.sin(angle) * offy
        
        rotx_phase = 1
        roty_phase = 1
        
        if direction == 'h':
            rotx_phase = np.exp(-1j*(2*np.pi/self.wavelength) *
                                np.sin(2*rot)*self.gridx)
        elif direction == 'v':
            roty_phase = np.exp(-1j*(2*np.pi/self.wavelength) *
                                np.sin(2*rot)*self.gridy)
        
        #---------------------------------------------------
        # add rotation of source
        
        # ideally the rocking angle of reflected light is double of rocking 
        # angle of mirror
        
        if direction == 'h':
            
            if kind == 'ep':
                
                self.lens_phase = np.exp(
                    1j*(2*np.pi/self.wavelength) *
                    (np.sqrt((self.gridx + offset)**2 + pfocus**2) +
                     np.sqrt((self.gridx + offset)**2 + qfocus**2))
                    )
            
            elif kind == 'hb':
                
                self.lens_phase = np.exp(
                    1j*(2*np.pi/self.wavelength) *
                    (np.sqrt((self.gridx + offset)**2 + afocus**2) -
                     np.sqrt((self.gridx + offset)**2 + bfocus**2))
                    )
        
        elif direction == 'v':
            
            if kind == 'ep':
            
                self.lens_phase = np.exp(
                    1j*(2*np.pi/self.wavelength) *
                    (np.sqrt((self.gridy + offset)**2 + pfocus**2) +
                     np.sqrt((self.gridy + offset)**2 + qfocus**2))
                    )
            
            elif kind == 'hb':
            
                self.lens_phase = np.exp(
                    1j*(2*np.pi/self.wavelength) *
                    (np.sqrt((self.gridy + offset)**2 + afocus**2) -
                     np.sqrt((self.gridy + offset)**2 + bfocus**2))
                    )
        
        #---------------------------------------------------
        # add error of the mirror
        
        if not isinstance(error, np.ndarray): 
            e = np.zeros((self.n_row, self._column), dtype = np.complex128)
        
        else:
            e = np.zeros((self.n_row, self._column), dtype = np.complex128)
            
            if direction == 'h':
                for i in range(self.n_row): 
                    e[i, :] = np.exp(1j*error)
            
            elif direction == 'v':
                for i in range(self.n_column): 
                    e[:, i] = np.exp(1j*error)
        
        error_phase = np.exp(1j*e)

        #--------------------------------------------------
        # construct phase and error
        
        for i in range(self.n):

            self.cmode[i] = self.cmode[i](
                self.lens_phase *
                rotx_phase * roty_phase *  error_phase
                )
            
#--------------------------------------------------------------------------
# the class of ideal lens     

class ideal_lens(_op):
    """
    construct the class of ideal lens
    
    methods
    """
    
    def __init__(
        self, 
        optic = None, optic_file = None, xcoor = None, ycoor = None,
        name = "ideal_lens", 
        n_vector = 0, i_vector = None, position = 0,
        wavelength = None, ratio = None,
        xfocus = 0, yfocus = 0, offx = 0, offy = 0, 
        rotx = 0, roty = 0, error = None
        ):
        
        super().__init__(
            optic = optic, optic_file = optic_file, 
            xcoor = xcoor, ycoor = ycoor,
            name = name,  n_vector = n_vector, i_vector = i_vector,
            position = position, wavelength = wavelength, 
            ratio = ratio
            )
        
        self.focus_x = xfocus
        self.focus_y = yfocus
    
        #---------------------------------------------------
        # add vibration of source
        
        self.lens_phase = np.exp(
            1j*(2*np.pi/self.wavelength) *
            ((self.gridx + offx)**2/(2*self.focus_x) + 
             (self.gridy + offy)**2/(2*self.focus_y))
            )
        
        #---------------------------------------------------
        # add rotation of source
        
        rotx_phase = np.exp(
            -1j*(2*np.pi/self.wavelength)*np.sin(rotx)*self.gridx
            )
        roty_phase = np.exp(
            -1j*(2*np.pi/self.wavelength)*np.sin(roty)*self.gridy
            )    
        #---------------------------------------------------
        # add error

        if not isinstance(error, np.ndarray): error_phase = 1

        else:
            error_phase = np.exp(1j*error)
            
        for i in range(self.n):
            self.cmode[i] = self.cmode[i] * (
                self.lens_phase * 
                rotx_phase * roty_phase *
                error_phase
                )
            
#-----------------------------------------------------------------------------#