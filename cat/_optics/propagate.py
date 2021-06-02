#-----------------------------------------------------------------------------#
# Copyright (c) 2020 Institute of High Energy Physics Chinese Academy of 
#                    Science
#-----------------------------------------------------------------------------#

__authors__  = "Han Xu - heps hard x-ray scattering beamline (b4)"
__date__     = "date : 05.02.2021"
__version__  = "beta-0.2"


"""
_source_utils: Source construction support.

functions: 

classes  : None
"""

#-----------------------------------------------------------------------------#
# library

import sys
import time
import numpy as np

# Warning! There is a bug in numpy.fft. Use scipy numpy fft instead.
# The pixel number of 2d arrray should be odd x odd. Or, half pxiel shift
# will be introudced. 
# For numpy, this shift was 1 and a half pxiel for odd x odd.
# Half pixel shift for even x even was oberved of both numpy fft and scipy fft.

from scipy import fft
from copy import deepcopy
from numpy import matlib 

#-----------------------------------------------------------------------------#
# constant

bar = "▋" 

#-----------------------------------------------------------------------------#
# function

#-----------------------------------------------------------------------------#
# propagate functions

#--------------------------------------------------------------------------
# fresnel fft propagator  

def _fresnel_dfft(cmode_to_propagate,  wavelength, nx, ny,
                  xstart, ystart, xend, yend, rx, ry, distance):

# Warning! The fft2 of impulse function and ic should be done together with
# numpy fft fft2. Or some speckle will apear.

    # wave number k
    wave_num = 2*np.pi / wavelength
    
    # the axis in frequency space
    qx = np.linspace(0.25/xstart, 0.25/xend, nx) * nx
    qy = np.linspace(0.25/ystart, 0.25/yend, ny) * ny
    
    mesh_qx, mesh_qy = np.meshgrid(qx, qy)
    
    # propagation function
    impulse_q = np.exp(
        (-1j * wave_num * distance) * 
        (1 - wavelength**2 * (mesh_qx**2 + mesh_qy**2))/2
        )
    
    # the multiply of coherent mode and propagation function
    propagated_cmode = fft.ifft2(
        fft.fft2(cmode_to_propagate) * 
        fft.ifftshift(impulse_q)
        )
      
    return propagated_cmode

#--------------------------------------------------------------------------
# angular spectrum fft propagator 

def _asm_sfft(cmode_to_propagate, wavelength, nx, ny, xstart, ystart,
             xend, yend, rx, ry, distance):
    
    dx = (xstart - xend) / nx
    dy = (ystart - yend) / ny
    
    fx = np.linspace(-1/(2*dx), 1/(2*dx), nx)
    fy = np.linspace(-1/(2*dy), 1/(2*dy), ny)
    
    mesh_fx, mesh_fy = np.meshgrid(fx, fy)
    
    impulse = np.exp(
        -1j * 2 * np.pi * distance *
        np.sqrt(1 / wavelength**2 - (mesh_fx**2 + mesh_fy**2))
        )
    
    cmode_to_propagate = fft.ifftshift(fft.fft2(cmode_to_propagate))
    
    propagated_cmode = fft.ifft2(
        fft.ifftshift(impulse * cmode_to_propagate)
        )
    
    return propagated_cmode

#--------------------------------------------------------------------------
# kirchhoff fresnel propagator

def _kirchoff_fresnel(cmode_to_propagate, wavelength, 
                      fnx, fny, fxstart, fystart, fxend, fyend, fgridx, fgridy,
                      bnx, bny, bxstart, bystart, bxend, byend, bgridx, bgridy,
                      distance):

    count = bnx * bny
    xpixel = (fxend - fxstart)/fnx
    ypixel = (fyend - fystart)/fny
    
    front_wave = cmode_to_propagate
    back_wave = np.zeros((bny, bnx), dtype = complex).flatten()
    bgridx = bgridx.flatten()
    bgridy = bgridy.flatten()
    
    for i in range(count):
        
        print(i, flush = True)
        
        path = np.sqrt(
            (bgridx[i] - fgridx)**2 + (bgridy[i] - fgridy)**2 + distance**2
            )
        
        path_phase = 2*np.pi * path / wavelength
        costhe = distance / path
        
        back_wave[i] = back_wave[i] + np.sum(
            ((fxstart - fxend)/fnx) * ((fystart - fyend)/fny) *
            np.abs(front_wave) *
            
            # TO DO: The sign of phase in this package should be checked again!
            # 1*np.angle(front_wave) - path_phase
            
            np.exp(1j * (1 * np.angle(front_wave) - path_phase)) *
            costhe /
            (wavelength * path)
            )

    back_wave = np.reshape(back_wave, (bny, bnx))

    
    return back_wave 

#--------------------------------------------------------------------------
# chirp-z transform propagator

# TODO: Unstable !!! The sampling ratio is not clear.
 
def _bluestein(cmode_to_propagate, wavelength, 
              fnx, fny, fxstart, fystart, fxend, fyend, fgridx, fgridy,
              bnx, bny, bxstart, bystart, bxend, byend, bgridx, bgridy,
              distance):
   

    #---------------------------------------------------
    # bluestein fft method

    def _bluestein_fft(g_input, fs, n, start, end, distance):
    
        n_ver, n_hor = np.shape(g_input)
        
        start_index = start + fs + 0.5 * (end - start) / n
        end_index = end + fs + 0.5 * (end - start) / n
        
        start_phase = np.exp(-1j * 2 * np.pi * start_index / fs)
        step_phase = np.exp(1j * 2 * np.pi * (end_index - start_index) / (n * fs))
        
        start_phase_neg_n = np.array(
            [start_phase**(-1*i) for i in range(n_ver)]
            )
        step_phase_n2 = np.array(
            [step_phase**(i**2/2) for i in range(n_ver)]
            )
        step_phase_k2 = np.array(
            [step_phase**(i**2/2) for i in range(n)]
            )
        step_phase_nk2 = np.array(
            [step_phase**(i**2/2) for i in range(-n_ver + 1, max(n_ver, n))]
            )
        step_phase_neg_nk2 = step_phase_nk2**(-1)
        
        fft_n = n_ver + n - 1
        count = 0
        while fft_n <= n_ver + n - 1:
            fft_n = 2**count
            count += 1
        
        conv_part0 = np.repeat(
            (start_phase_neg_n * step_phase_n2)[:, np.newaxis], 
            n_hor, axis = 1
            ) 
        conv_part1 = np.repeat(
            step_phase_neg_nk2[:, np.newaxis], 
            n_hor, axis = 1
            )
        
        # calcualte convoluation with FFT

        conved = (
            fft.fft(g_input * conv_part0, fft_n, axis = 0) * 
            fft.fft(conv_part1, fft_n, axis = 0)
            )
        
        # TODO: The flux was disturbed by convolution.
        
        g_output = fft.ifft(conved, axis = 0)
        g_output = (
            g_output[n_ver : n_ver + n, :] * 
            np.repeat(step_phase_k2[:, np.newaxis], n_hor, axis = 1)
            )
        
        l = (end_index - start_index) * np.linspace(0, n - 1, n)/n + start_index
        shift_phase = matlib.repmat(
            np.exp(1j * 2 * np.pi * l * (-n_ver/2 + 0.5)/fs), n_hor, 1
            )
        g_output = g_output.T * shift_phase 

        return g_output 

    #---------------------------------------------------
    # set plane geometry structure

    wave_num = 2 * np.pi / wavelength
    
    xpixel0 = (fxend - fxstart) / fnx
    ypixel0 = (fyend - fystart) / fny
    xpixel1 = (bxend - bxstart) / bnx
    ypixel1 = (byend - bystart) / bny
    
    fresnel0 = (
        np.exp(1j * wave_num * distance) * 
        np.exp(-0.5 * 1j * wave_num * (bgridx**2 + bgridy**2) / distance)
        )
    fresnel1 = (
        np.exp(-0.5 * 1j * wave_num * (fgridx**2 + fgridy**2) / distance)
        )
    g_input = cmode_to_propagate * fresnel1
    
    #------------------------------------------
    # propagate along y axis
    
    yfs = wavelength * distance / ypixel0
    yg_input = g_input
    
    g_input = _bluestein_fft(yg_input, yfs, bny, bystart, byend, distance)
    
    #------------------------------------------
    # propagate along x axis
    
    xfs = wavelength * distance / xpixel0
    xg_input = g_input
    
    g_input = _bluestein_fft(xg_input, xfs, bnx, bxstart, bxend, distance)
    
    #------------------------------------------
    # output
    
    g_output = g_input * fresnel0
    norm = (
        np.sum(np.abs(cmode_to_propagate)**2 * xpixel0 * ypixel0) / 
        np.sum(np.abs(g_output)**2 * xpixel1 * ypixel1)
        )
    
    return g_output * norm

#-----------------------------------------------------------------------------#
# function - propagation functions

#--------------------------------------------------------------------------
# fresnel propagate function

def fresnel(front, back):

    distance = back.position - front.position
     
    if distance == 0:
        
        for i in range(front.n):
            back.cmode[i] = back.cmode[i] * front.cmode[i]
            
    else:
        # the loop of every coherent mode
        
        for i in range(front.n):
            
            back_cmode = _fresnel_dfft(
                front.cmode[i], front.wavelength, 
                front.xcount, front.ycount, front.xstart, front.ystart, 
                front.xend, front.yend, front.xtick, front.ytick, 
                distance
                )
            
            back.cmode[i] = back.cmode[i] * back_cmode

#--------------------------------------------------------------------------
# angular spectrum propagate function

def asm(front, back):

    distance = back.position - front.position

    if distance == 0:

        for i in range(front.n):
            back.cmode[i] = back.cmode[i] * front.cmode[i]

    else:

        for i in range(front.n):

            back_cmode = _asm_sfft(
                front.cmode[i], front.wavelength,
                front.xcount, front.ycount, front.xstart, front.ystart, 
                front.xend, front.yend, front.xtick, front.ytick, 
                distance    
                )
            
            back.cmode[i] = back.cmode[i] * back_cmode

#--------------------------------------------------------------------------
# kirchhoff-fresnel propagate function

def kirchhoff(front, back):
    
    distance = back.position - front.position
    
    for i in range(front.n):
        
        back_cmode = _kirchoff_fresnel(
            front.cmode[i], front.wavelength,
            front.xcount, front.ycount, front.xstart, front.ystart,
            front.xend, front.yend, front.gridx, front.gridy,
            back.xcount, back.ycount, back.xstart, back.ystart, back.xend,
            back.yend, back.gridx, back.gridy,
            distance
            )
        
        back.cmode[i] = back.cmode[i] * back_cmode

#--------------------------------------------------------------------------
# chirp-z transform propagate function

def czt(front, back):
    
    distance = back.position - front.position
    
    for i in range(front.n):
        
        back_cmode = _bluestein(
            front.cmode[i], front.wavelength,
            front.xcount, front.ycount, front.xstart, front.ystart,
            front.xend, front.yend, front.gridx, front.gridy,
            back.xcount, back.ycount, back.xstart, back.ystart, back.xend,
            back.yend, back.gridx, back.gridy,
            distance
            )
        
        back.cmode[i] = back.cmode[i] * back_cmode
                
#-----------------------------------------------------------------------------#
# function - propagate beamline

#--------------------------------------------------------------------------
# propagate different coherent modes

def propagate_mode(n, vectors, beamline_func):
    
    t1 = time.time()
    
    for i, vector in enumerate(vectors):
        
        #------------------------------------------------------
        # processing bar
        
        print("\r", end = "")
        print(
            "propagate processs: {}%: ".format(int(100*(i + 1)/n)), 
            "▋" * int(1 + 10*i/n), end = ""
            )
        
        t2 = time.time()
        print("time cost: %.2f min" % (np.abs(t2 - t1)/60), end = "")
        
        sys.stdout.flush()
        time.sleep(0.005)
        
        #-----------------------------------------------------
        
        if i == 0:
            optic0 = beamline_func(vector)
        else:
            optic1 = beamline_func(vector)
            optic0.cmode.append(optic1.cmode[0])
    
    # t2 = time.time()
    
    # print("  time cost: %.2f min" % (np.abs(t2 - t1)/60))
    
    return optic0

#--------------------------------------------------------------------------
# propagate to different location

def propagate_plane(n, n_center, interval, beamline_func):
    
    t1 = time.time()
    
    for i in range(n):
        
        #------------------------------------------------------
        # processing bar
        
        print("\r", end = "")
        print(
            "propagate processs: {}% ".format(int(100*(i + 1)/n)), 
            "▋" * int(1 + 10*i/n), end = ""
            )
        
        t2 = time.time()
        print("  time cost: %.2f min" % (np.abs(t2 - t1)/60), end = "")
    
        sys.stdout.flush()
        time.sleep(0.005)
        
        #-----------------------------------------------------
        
        if i == 0:
            optic0 = beamline_func(i, n_center, interval)
        else:
            optic1 = beamline_func(i, n_center, interval)
            
            optic0.cmode.append(optic1.cmode[0])
    
    # t2 = time.time()
    
    # print("  time cost: %.2f min" % (np.abs(t2 - t1)/60))
    
    return optic0