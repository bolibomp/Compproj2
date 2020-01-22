# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 12:20:38 2018

@author: Sonny
"""
import skimage.io
import numpy as np
import skimage.viewer
from matplotlib import pyplot as plt
import math
import sys
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv

Iori = skimage.io.imread('Isle_of_new.jpg',as_grey=True).astype(np.float32)
Imask = skimage.io.imread('Isle_of_mask.jpg',as_grey=True).astype(np.float32)
N=len(Imask)
M=len(Imask[0])


def num(i,j): #Gives the number boundary points (2 for corner, 3 for side, else 4)
    if (any([i==0, i==N]) and any([j==0, j==M])):
        return 2
    if (any([i==0, i==N]) or any([j==0, j==M])):
        return 3
    else:
        return 4

def Ik(i,j,I=Iori,mask=Imask): #0 if outside matrix
    if (0>i or i>(N-1) or 0>j or j>(M-1)):
        return 0
    elif mask[i][j]==0:
        return I[i][j]
    else:
        return mask[i][j]

def eul(D): #Explicit euler
    for i in range(N):
        for j in range(M):
            if Imask[i][j]==0:
                continue
            else:
                n=num(i,j)
                Imask[i][j]=Imask[i][j]+D*(Ik(i-1,j)+Ik(i+1,j)+Ik(i,j-1)+Ik(i,j+1)-n*Ik(i,j))
    return Imask

#-----------

def centeracc(f1,f,f3): #O(a^2)
    return (f1-2*f+f3)
def acc3(f1,f2,f): #O(a) #backwards and forwards is the same
    return(f1-2*f2+f)
def acc4(f1,f2,f3,f): #O(a^2)  
    return (2*f1-5*f2+4*f3-f)

def centerder(f1,f3):
    return (f3-f1)/2
def der2(f,f1):
    return (f1-f)
def der3(f,f1,f2):
    return (-3/2*f+2*f1-1/2*f2)

def I(i,j,matris): #0 if outside matrix
    n=len(matris)
    m=len(matris[0])
    if (0>i or i>(n-1) or 0>j or j>(m-1)):
        return 0
    else:
        return matris[i][j]

def findpoint(matris):
    n=len(matris)
    m=len(matris[0])
    matrix=np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            if not(I(i,j,matris)==1 or I(i,j,matris)==I(i,j+1,matris)==I(i,j-1,matris)==I(i+1,j,matris)==I(i-1,j,matris)==0):
                matrix[i][j]=1
    return matrix
def Ineu(i,j,matris): #0 if outside matrix
    n=len(matris)
    m=len(matris[0])
    if (0>i or i>(n-1) or 0>j or j>(m-1)):
        return -1
    else:
        return matris[i][j]
    

def THELOOP(matris, ori):
    n=len(matris)
    m=len(matris[0])
    boundary=findpoint(matris)
    neufield=np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            if boundary[i,j]==0:
                continue
            g0=Ineu(i,j-3,ori)
            g1=Ineu(i,j-2,ori)
            g2=Ineu(i,j-1,ori)
            g3=Ineu(i,j-0,ori)
            g4=Ineu(i,j+1,ori)
            
            I1=Ineu(i,j+1,matris)
            I2=Ineu(i,j-1,matris)
            if I1==I2==0:
                neufield[i][j]=centeracc(g2,g3,g4)
                continue
            I3=Ineu(i,j-2,matris)
            I4=Ineu(i,j-3,matris)
            if I4==I3==I2==0:
                neufield[i][j]=acc4(g0,g1,g2,g3)
                
            elif I3==I2==0:
                neufield[i][j]=acc3(g1,g2,g3)
            
            else:
                neufield[i][j]
    return neufield


def neumann(matris, ori): #mask in here #returns (-u_y, v_x)
    n=len(matris)
    m=len(matris[0])
    neu=[]
    for i in range(4):
        m1=np.rot90(matris,i)
        m2=np.rot90(ori,i)
        neu.append(np.rot90(THELOOP(m1,m2),4-i))
    neux=[np.zeros((n,m)),np.zeros((n,m))]
    neunew=np.zeros((n,m))
    for k in range(2):
        for i in range(n):
            for j in range(m):
                if (neu[k][i][j]==0 and neu[k+2][i][j]==0):
                    continue
                if (neu[k][i][j]!=0 and neu[k+2][i][j]!=0):
                    neux[k][i][j]=(neu[k][i][j]-neu[k+2][i][j])/2
                elif neu[k][i][j]!=0:
                    neux[k][i][j]=neu[k][i][j]
                else:
                    neux[k][i][j]=neu[k+2][i][j]
                neunew[i][j]=neunew[i][j]+neux[k][i][j]
    return neunew

def getvelo(matris, Iori): #gets the velocityfield
    n=len(matris)
    m=len(matris[0])
    veloxy=[np.zeros((n,m)), np.zeros((n,m))]
    for i in range(n):
        for j in range(m):
            if matris[i][j]==0:
                continue
            if not(i==0 or i==(n-1)):
                veloxy[0][i][j]=centerder(Iori[i-1][j],Iori[i+1][j])
            elif i==0:
                veloxy[0][i][j]=der3(Iori[i][j],Iori[i+1][j],Iori[i+2][j])
            else:
                veloxy[0][i][j]=-der3(Iori[i][j],Iori[i-1][j],Iori[i-2][j])
            
            if not(j==0 or j==(m-1)):
                veloxy[1][i][j]=centerder(Iori[i][j-1],Iori[i][j+1])
            elif j==0:
                veloxy[1][i][j]=der3(Iori[i][j],Iori[i][j+1],Iori[i][j+2])
            else:
                veloxy[1][i][j]=-der3(Iori[i][j],Iori[i][j-1],Iori[i][j-2])
    
    return veloxy

def absomega(deromega): #takes the matrix of derivates of omega and returns abs, also no touchie
    n=len(deromega[0])
    m=len(deromega[0][0])
    abomega=np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            if deromega[0][i][j]==deromega[1][i][j]==0:
                continue   
            abomega[i][j]=np.sqrt((deromega[0][i][j])**2+(deromega[1][i][j])**2)
    return abomega

def g(velofield, K=0.5): #No touchie
    n=len(velofield[0])
    m=len(velofield[0][0])
    gfield=[np.zeros((n,m)), np.zeros((n,m))]
    s=absomega(velofield)
    for i in range(n):
        for j in range(m):
            if velofield[0][i][j]==velofield[1][i][j]==0 :
                continue
            gfield[0][i][j]=1/(1+(s[i][j]/K)**2)*velofield[0][i][j]
            gfield[1][i][j]=1/(1+(s[i][j]/K)**2)*velofield[1][i][j]
    return gfield


def newomega(oldomega, matris, I):
    n=len(I)
    m=len(I[0])

    velo=getvelo(matris, I) #u and v, here is why we need "I"
    u=np.multiply(-1, velo[1])
    v=velo[0]
    
    veloomega=(getvelo(matris, oldomega)[0],getvelo(matris, oldomega)[1]) #w_x and w_y

    gfield=g(veloomega) #g(|lap(w)|)*w_x, g(|lap(w)|)*w_y
    
    dx=getvelo(matris, gfield[0])[0] 
    dy=getvelo(matris, gfield[1])[1]
    diffusion=[x+y for x,y in zip(dx,dy)] #this is the whole diffusion term.. I think
    
    omega=oldomega
    for i in range(n):
        for j in range(m):
            if matris[i][j]==0:
                continue
            try:
                uw=abs(u)*(oldomega[i+int(np.sign(u))][j]-oldomega[i][j])
            except IndexError:
                uw=0
            try:
                vw=abs(v)*(oldomega[i][j+int(np.sign(v))]-oldomega[i][j])
            except IndexError:
                vw=0
            omega[i][j]=omega[i][j]-uw-vw+diffusion[i][j]
    return omega

def Somega(omega,matris):
    S=[]
    n=len(matris)
    m=len(matris[0])
    for i in range(0,n):
        for j in range(0,m):
            if matris[i][j]==1:
                S.append(omega[i][j])
    return S

def Romega(omega,matris):
    n=len(matris)
    m=len(matris[0])
    R=np.zeros((n,m))
    k=0
    n=len(matris)
    m=len(matris[0])
    for i in range(0,n):
        for j in range(0,m):
            if matris[i][j]==1:
                R[i,j]=omega[k]
                k+=1
                
    return R

#-----------------------------------------
def up(a):
    b=[x for x in a]
    b[0]=b[0]-1
    return b
def down(a):
    b=[x for x in a]
    b[0]=b[0]+1
    return b
def left(a):
    b=[x for x in a]
    b[1]=b[1]-1
    return b
def right(a):
    b=[x for x in a]
    b[1]=b[1]+1
    return b          

def SList(matris):
    S=[]
    n=len(matris)
    m=len(matris[0])
    for j in range(0,n):
        for i in range(0,m):
            if matris[j][i]==1:
                S.append([j,i])
    return S

def Amatrix(matris, neumann1):
    n=len(matris)
    m=len(matris[0])
    M=SList(matris)
    b=[]
    A=-4*np.identity((len(M)))
    for i in range(len(M)):
        b.append(0)
        upper=up(M[i])
        downer=down(M[i])
        righter=right(M[i])
        lefter=left(M[i]) #These can generate coordinates outside of boundary
        try:
            if upper in M:
                index = M.index(upper)
                A[i][index]+=1
            else:
                haha=neumann1[upper[0]-n][upper[1]]
                b[i]+=neumann1[upper[0]][upper[1]]
        except IndexError:
            A[i][i]+=1
            pass
        
        try:
            if downer in M:
                index = M.index(downer)
                A[i][index]+=1
            else:
                b[i]+=neumann1[downer[0]][downer[1]]
        except IndexError:
            A[i][i]+=1
            pass
        
        try:
            if righter in M:
                index = M.index(righter)
                A[i][index]+=1
            else:
                b[i]+=neumann1[righter[0]][righter[1]]
        except IndexError:
            A[i][i]+=1
            pass

        try:
            if lefter in M:
                index = M.index(lefter)
                A[i][index]+=1
            else:
                haha=neumann1[upper[0]][upper[1]-m-1]
                b[i]+=neumann1[lefter[0]][lefter[1]]
        except IndexError:
            A[i][i]+=1
            pass

    return A,b          
            
#a=[[1,1,1,1,1,0.5,1],
#   [1,1,1,1,1,1,1],
#   [0.5,0.5,0.5,0.5,0.5,0.5,0.5],
#   [0.5,0.5,0,0,0,0,0],
#   [0.5,0,0,0,0,0,0]]   
#b=[[1,1,1,1,1,0,1],
#   [1,1,1,1,1,1,1],
#   [0,0,0,0,0,0,0],
#   [0,0,0,0,0,0,0],
#   [0,0,0,0,0,0,0]] 
#
##print(len(a[0]),len(b[0]))
#
#c=Amatrix(b,a)
#d=np.linalg.det(c[0])
#print(d)
#
##print(c[0])
##d=newomega(c, b, a)
#plt.figure(2)
#plt.imshow(c[0], cmap='gray', interpolation='nearest')
#plt.figure(3)
#plt.imshow(a, cmap='gray', interpolation='nearest')
            
#----------------------------------
def mean(Iori):
    summ=0
    n=0
    for i in range(N):
        for j in range(M):
            if Imask[i][j]==0:
                continue
            else:
                n=n+1
                summ=+Iori[i][j]
    return summ/n,n
def error(Iori, I, Isolu):
    sumt=0
    sumn=0
    meanori=mean(Iori)[0]
    n=mean(Iori)[1]
    print(n)
    for i in range(N):
        for j in range(M):
            if Imask[i][j]==0:
                continue
            else:
                sumt=sumt+(I[i][j]-Isolu[i][j])**2
                sumn=sumn+(I[i][j]-meanori)**2
    return ((n-1)/n*(sumt/sumn))
#--------------------------------

def invmatris(MATRISEN):
    A=csc_matrix(MATRISEN)
    Ainv=inv(A)
    return Ainv

#-------------------------
def Slist(realneu):
    S=[]
    n=len(realneu)
    m=len(realneu[0])
    for j in range(0,n):
        for i in range(0,m):
            if realneu[j][i]==1:
                S.append((j,i))
    return S


                   
        