# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 09:05:27 2018

@author: Sonny
"""

import skimage.io
import numpy as np
import skimage.viewer
from matplotlib import pyplot as plt
import math
import sys
import numpy as np
import Functions as F
N=7
Isolu=skimage.io.imread('Isle_of_new.jpg',as_grey=True).astype(np.float32)
Iori = skimage.io.imread('Isle_of_grey.jpg',as_grey=True).astype(np.float32)
Iori1 = skimage.io.imread('Isle_of_grey.jpg',as_grey=True).astype(np.float32)
Imask = skimage.io.imread('Isle_of_mask.jpg',as_grey=True).astype(np.float32)



#-----------------------------This code does not work
omega=F.neumann(Imask, Iori)
A=F.Amatrix(Imask, omega)
Ain=np.linalg.inv(A[0])

omegatot=[0]*Ain.shape[0]
for t in range(N):
    print(t)
    omega=F.newomega(omega, Imask, Iori)
    Sw=F.Somega(omega,Imask)
    for i in range(len(Sw)):
        omegatot[i]=Sw[i]+A[1][i]
    I=[0]*len(omegatot)
    for i in range(len(omegatot)):
        for j in range(len(omegatot)): 
            I[i] += Ain[i][j]*omegatot[j]
    I=np.dot(Ain,omegatot)
    I=F.Romega(I,Imask)
    omega=F.Romega(Sw,Imask)
#---------------------------This code does not work

#-------------- Euler

#for i in range(200):
#    I=F.eul(0.1)
N=len(Imask)
M=len(Imask[0])
for i in range(N):
        for j in range(M):
            if Imask[i][j]==0:
                continue
            else:
                Iori[i][j]=I[i][j]
#                
err=F.error(Isolu, Iori1, Iori)
#print(err)
#plt.figure(1)
#plt.imshow(A, cmap='gray', interpolation='nearest')
plt.figure(2)
plt.imshow(Iori1, cmap='gray', interpolation='nearest')
plt.figure(3)
plt.imshow(Iori, cmap='gray', interpolation='nearest')
plt.figure(4)
plt.imshow(Isolu, cmap='gray', interpolation='nearest')
plt.imsave('Isle_of_result.jpg', arr=Iori, cmap='gray')