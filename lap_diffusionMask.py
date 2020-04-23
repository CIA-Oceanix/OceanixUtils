#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 09:39:17 2018

@author: rfablet
"""
import numpy as np

def lap_diffusionMask(I,iter,lam,lamData=0.):
    ## assume missing data are NaN
    ## I : image to be filtered
    ## iter : number of iterations of the diffusion
    ## lam : diffusion coefficient, It=lam . laplacian(I)
    ## lamData : weight for data-driven term, It=lam*(I-Init)
    debug = 0
    Iinit = I
    
    lapI  = np.zeros((I.shape[0],I.shape[1],4))
    slapI = np.zeros((I.shape[0],I.shape[1]))
    for ii in range(0,iter):
        ## compute laplacian
        Ix           = I[1:,:]-I[0:-1,:]
        Iy           = I[:,1:]-I[:,0:-1]
        lapI[:,:,0]  = np.concatenate((-Ix[0:1,:],Ix),axis=0)
        lapI[:,:,1]  = np.concatenate((-Ix,Ix[I.shape[0]-2:,:]),axis=0)
        lapI[:,:,2]  = np.concatenate((-Iy[:,0:1],Iy),axis=1)
        lapI[:,:,3]  = np.concatenate((-Iy,Iy[:,I.shape[1]-2:]),axis=1)
        
        slapI = np.nansum(lapI,axis=2) / (1e-10+np.nansum(1.0-np.isnan(lapI).astype(float),axis=2))
        I     = I - lam * slapI + lamData * (Iinit - I)
  
        if debug:
            plt.figure(10)
            plt.clf()
            plt.subplot(1,3,1)
            imgplot = plt.imshow(lapI)
            imgplot.set_cmap('hot')
            plt.colorbar()
            plt.subplot(1,3,2)
            imgplot = plt.imshow(Iinit)
            imgplot.set_cmap('hot')
            plt.colorbar()
            plt.subplot(1,3,3)
            imgplot = plt.imshow(I)
            imgplot.set_cmap('hot')
            plt.colorbar()
            plt.plot()
            
            input("Press Enter to continue...")
 
    return I
