#!/usr/bin/env python
# coding: utf-8

# In[46]:


import numpy as np
from scipy.sparse import find
import sys
from uvw import RectilinearGrid, DataArray
import source.pade_compact as pc
from scipy.fftpack import dct
import matplotlib.pyplot as plt


######################################################
# Its a magicbox for all required auxiliary function.#
######################################################

def aero_sq(ny,nx,p,eps,rho,u_vel):
    p_chop = p*eps
    a,b,P = find(p_chop)
    P = P.reshape((a.max()-a.min()+1,b.max()-b.min()+1))
    cp = (P-p[int(ny/2),int(nx/4)])/(0.5*rho*u_vel**2)
    cplo,cpup,cple,cpri = cp[0,:],cp[-1,:],cp[:,0],cp[:,-1]
    cl = np.mean(cpup) - np.mean(cplo)
    cd = np.mean(cple) - np.mean(cpri)
    cp_mean = np.mean(cp)
    return cl,cd,cp_mean

def aero_ir(ny,nx,plxlf,plylf,plxrg,plyrg,p):
    p_lf,p_rt = np.zeros(len(plxlf)),np.zeros(len(plxrg))

    for i in range(len(plxlf)):
        p_lf[i] = p[int(plxlf[i]),int(plylf[i])]
        
    for j in range(len(plxrg)):
        p_rt[j] = p[int(plxrg[j]),int(plyrg[j])]
        
    cd = 2*(np.mean(p_lf)-np.mean(p_rt))
    
    p_lfd,p_rtd = np.array_split(p_lf,2),np.array_split(p_rt,2)
    plf1,prt1 = np.array(list(p_lfd[0])+ list(p_rtd[0])),np.array(list(p_lfd[1])+list(p_rtd[1]))
    cl = 2*(np.mean(plf1)-np.mean(prt1))
    
    return 1.5*cl,cd

def nusnum_ir(dx,dy,plxlf,plylf,plxrg,plyrg,T,T0):
    plx,ply = list(plxlf) + list(np.flip(plxrg)),list(plylf) + list(np.flip(plyrg))
    T_ext = np.zeros(len(plx))
    for i in range(len(plx)):
        T_ext[i] = T[int(plx[i]),int(ply[i])]
    Nu = (T0-T)/(min(dx,dy))
    Nuavg = np.mean(Nu)
    return Nuavg

def nusnum(T,T0,eps,dx,dy):
    a,b,V = find(eps)
    a = a.reshape((a.max()-a.min()+1,b.max()-b.min()+1))
    b = b.reshape((b.max()-b.min()+1,a.max()-a.min()+1))
    
    Tbot = T[a.min()-1,b.min():b.max()+2]
    Ttop = T[a.max()+1,b.min():b.max()+2]
    Tlef = T[a.min():a.max()+2,b.min()-1]
    Trig = T[a.min():a.max()+2,b.max()+1]
    
    Nu_top = (T0-Ttop)/(dx)
    Nu_bot = (T0-Tbot)/(dx)
    Nu_lef = (T0-Tlef)/(dy)
    Nu_rig = (T0-Trig)/(dy)
    
    Nu = np.array([])
    Nu = np.append(Nu,Nu_top)
    Nu = np.append(Nu,Nu_lef)
    Nu = np.append(Nu,Nu_bot)
    Nu = np.append(Nu,Nu_rig)
    
    Nu_avg = np.mean(Nu)
    
    return Nu_avg

def vtr_exp(sdi,i,L,H,ny,nx,u,v,p,vort,T):
    sivgt = np.zeros((ny,nx))
    dx,dy = L/(nx-1),H/(ny-1)
    sivgt[1:-1,1:-1] = ((u[1:-1,2:]-u[1:-1,:-2])/(dx**2))*((v[2:,1:-1]-v[:-2,1:-1])/(dy**2)) - ((u[2:,1:-1]-u[:-2,1:-1])/(dy**2))*((v[1:-1,2:]-v[1:-1,:-2])/(dx**2))
    # Creating coordinates
    y = np.linspace(0, L, nx)
    x = np.linspace(0, H, ny)
    
    xx, yy = np.meshgrid(x, y, indexing='ij')
    
    original_stdout = sys.stdout
    with open(sdi+'/out_it'+str(i+1)+'.vtr','w') as f:
        sys.stdout = f
        with RectilinearGrid(sys.stdout, (x, y)) as grid:
            grid.addPointData(DataArray(u, range(2), 'u velocity'))
            grid.addPointData(DataArray(v, range(2), 'v velocity'))
            grid.addPointData(DataArray(p, range(2), 'pressure'))
            grid.addPointData(DataArray(vort, range(2), 'vorticity'))
            grid.addPointData(DataArray(T, range(2), 'temperature'))
            grid.addPointData(DataArray(sivgt, range(2), 'SIVGT'))
        sys.stdout = original_stdout

def pres_pois_spec(bod,poi_b,bcxu,bcxv,bcyu,bcyv,ny,nx,dy,dx,rho,dt,u,v,eps=None):
    RHS = np.ones([ny,nx])
    if bod == 'IBM':
        u,v = (1-eps)*u,(1-eps)*v
        
    RHS = (pc.dx(u,ny,nx,dy,dx,bcxu) + pc.dy(v,ny,nx,dy,dx,bcyv))*(rho/dt)
    # FFT coefficients
    kp = 2*np.pi*np.fft.fftfreq(nx,d=dx)
    om = 2*np.pi*np.fft.fftfreq(ny,d=dy)
    kx,ky = np.meshgrid(kp,om)
    delsq = -(kx**2 + ky**2)
    delsq[0,0] = 1e-6
        
    # FFT integration
    # For periodic in axis=0 i.e. y-axis and dirichilet/pressure neumann in x-axis for inflow and outflow
    if poi_b == 'x dirichilet':
        RHS_hat = dct(np.fft.fft(RHS,axis=1),type = 1,axis = 0)
        p = RHS_hat*dt/(delsq)
        p = np.fft.ifft(dct(p,type=1,axis=0)/(2*(ny+1)),axis=1).real
        p = p - p[0,0]
            
    elif poi_b == 'double periodic':
        # For double periodic domain
        RHS_hat = np.fft.fft2(RHS)
        p = RHS_hat/delsq
        p = np.fft.ifft2(p).real
        p = p - p[0,0]                           
        
    return p

def resids(ene,u,v,p,T,ny,nx,dy,dx,nu,Pr,H):
    dpdx,dpdy,dudx,dudy,dvdx,dvdy,ddudx,ddudy,ddvdx,ddvdy,residT = np.zeros((ny,nx)),np.zeros((ny,nx)),np.zeros((ny,nx)),np.zeros((ny,nx)),np.zeros((ny,nx)),np.zeros((ny,nx)),np.zeros((ny,nx)),np.zeros((ny,nx)),np.zeros((ny,nx)),np.zeros((ny,nx)),np.zeros((ny,nx))
    
    dpdx[1:-1,1:-1] = -(p[1:-1,2:]- p[1:-1,:-2])/(2*dx)
    dpdy[1:-1,1:-1] = -(p[2:,1:-1]- p[:-2,1:-1])/(2*dy)
    dudx[1:-1,1:-1] = (u[1:-1,2:]- u[1:-1,:-2])/(2*dx)
    dudy[1:-1,1:-1] = (u[2:,1:-1]- u[:-2,1:-1])/(2*dy)
    dvdx[1:-1,1:-1] = (v[1:-1,2:]- v[1:-1,:-2])/(2*dx)
    dvdy[1:-1,1:-1] = (v[2:,1:-1]- v[:-2,1:-1])/(2*dy)
    
    ddudx[1:-1,1:-1] = (u[1:-1,2:] - 2*u[1:-1,1:-1] + u[1:-1,:-2])/(dx**2)
    ddudy[1:-1,1:-1] = (u[2:,1:-1] - 2*u[1:-1,1:-1] + u[:-2,1:-1])/(dy**2)
    ddvdx[1:-1,1:-1] = (v[1:-1,2:] - 2*v[1:-1,1:-1] + v[1:-1,:-2])/(dx**2)
    ddvdy[1:-1,1:-1] = (v[2:,1:-1] - 2*v[1:-1,1:-1] + v[:-2,1:-1])/(dy**2)
    
    if ene == 'on':
        dTdx,dTdy,ddTdx,ddTdy = np.zeros((ny,nx)),np.zeros((ny,nx)),np.zeros((ny,nx)),np.zeros((ny,nx))
        dTdx[1:-1,1:-1] = (T[1:-1,2:]- T[1:-1,:-2])/(2*dx)
        dTdy[1:-1,1:-1] = (T[2:,1:-1]- T[:-2,1:-1])/(2*dy)
        ddTdx[1:-1,1:-1] = (T[1:-1,2:] - 2*T[1:-1,1:-1] + T[1:-1,:-2])/(dx**2)
        ddTdy[1:-1,1:-1] = (T[2:,1:-1] - 2*T[1:-1,1:-1] + T[:-2,1:-1])/(dy**2)
        
        residT = u*dudx + v*dudy - (nu/Pr)*(ddTdx+ddTdy)
#         rmsT = np.sqrt(np.mean(residT**2))
    
    residu = u*dudx + v*dudy + dpdx - nu*(ddudx+ddudy)
    residv = u*dvdx + v*dvdy + dpdy - nu*(ddvdx+ddvdy)
    
    cont = dudx + dvdy
    con_r = abs(np.mean(cont))
    
#     rmsu,rmsv,rmsc = np.sqrt(np.mean(residu**2)),np.sqrt(np.mean(residv**2)),np.sqrt(np.mean(cont**2))
    
    return con_r*abs(np.mean(residu)),con_r*abs(np.mean(residv)),con_r,con_r*abs(np.mean(residT))
