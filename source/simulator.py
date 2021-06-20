#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import source.time_integrators as TIM
import time
import source.IBM as IBM
import source.auxfunx as af
# import probes as pbs

class FluiDNS():
    # Initialization constructor
    def __init__(self,L,H,nx,ny,ene,restart,save_dir,itr):
        self.L,self.H,self.nx,self.ny= L,H,nx,ny
        self.dx,self.dy = self.L/(self.nx),self.H/(self.ny)
        self.declarations(ny,nx,restart,itr,save_dir,ene)
    
    ##########################################################################################
    # A function to populate all required parameters used in simulation as a class attribute
    # to be available throughout the class anywhere
    ##########################################################################################
    def parameters(self,SimTit,save_dir,isave,idata,aeros,Re,Ri,Pr,bod,ene,buo,schema,pois_bound,bcxu,bcxv,bcyu,bcyv,bcxt,bcyt,rho,dt,vel,A0,T0,omp,shape,x_pos,y_pos,length,width,radius,mot,amp,fn,st,dog,hflux,probes,prints):
        self.title = SimTit
        self.sdi = save_dir
        self.isave,self.idata,self.aeros = isave,idata,aeros
        self.Re = Re
        self.dt,self.rho,self.nu = dt,rho,1/self.Re
        self.ischeme = schema
        self.vel = vel
        self.poi_b = pois_bound
        self.bcxu,self.bcxv,self.bcyu,self.bcyv,self.bod = bcxu,bcxv,bcyu,bcyv,bod
        self.A0,self.omp = A0,omp
        self.x_pos,self.y_pos,self.length,self.width = x_pos,y_pos,length,width
        self.shape = shape
        self.ra = radius
        self.ene,self.T0 = ene,T0
        self.Ri,self.Pr = Ri,Pr
        self.buo = buo
        self.bcxt,self.bcyt = bcxt,bcyt
        self.motion = mot
        self.amp,self.fn,self.st = amp,fn,st
        self.dog,self.hflux,self.probes,self.iprints = dog,hflux,probes,prints
    
    ########################################################################################################
    # Declaring all required arrays for starting the simulation based on the fresh simulation or a restart
    ########################################################################################################
    def declarations(self,ny,nx,restart,itr,save_dir,ene):
        self.x = np.linspace(0, self.L, nx)
        self.y = np.linspace(0, self.H, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        if restart == 'new':
            self.u,self.v,self.p = np.zeros([ny,nx]),np.zeros([ny,nx]),np.zeros([ny,nx])
            self.vort = np.zeros([ny,nx])
            self.T = np.zeros([ny,nx])

        elif restart == 'restart':
            print('It is a restart...')
            self.u,self.v = np.fromfile(save_dir+'/out_it' + str(itr)+ '_u.dat'),np.fromfile(save_dir+'/out_it' + str(itr)+ '_v.dat')
            self.p = np.fromfile(save_dir+'/out_it' + str(itr)+ '_p.dat')
            self.vort = np.fromfile(save_dir+'/out_it' + str(itr)+ '_vort.dat')
            
            self.u,self.v,self.p = self.u.reshape((ny,nx)),self.v.reshape((ny,nx)),self.p.reshape((ny,nx)) 
            self.vort = self.vort.reshape((ny,nx))
#             self.T = np.zeros([ny,nx])
            self.T = np.fromfile(save_dir +'/out_it' + str(itr)+ '_temp.dat')
            self.T = self.T.reshape((ny,nx))
            
               
    def solver(self,nt_st,nt_en):
        # time initialization
        t,itr = nt_st*self.dt,0
        # time step calculation
        nt = nt_en-nt_st
        ntn = nt_st
        print("total iterations = " +str(nt))
        print("Start time = "+str(nt_st*self.dt))
        print("End time = "+str(nt_en*self.dt))
        est_time = int(0.2*nt)
        t_rec = np.zeros([est_time,1])
        pro_u,pro_v,pro_p = np.array([]),np.array([]),np.array([])
        pro_vo,pro_T = np.array([]),np.array([])
        Nur,eps,eps_in = np.array([]),np.array([]),np.array([])
        cpr,clr,cdr = np.array([]),np.array([]),np.array([])
        resur,resvr,mdefr,resTr = np.array([]),np.array([]),np.array([]),np.array([])
        
        for i in range(nt):
            st_t = time.time()
            ntn += 1
            t = ntn*self.dt
            
            # IBM Mask formulation
            if self.bod == 'IBM':
                if self.shape == 'square':
                    eps,eps_in = IBM.square(self.motion,self.ene,self.ny,self.nx,self.dy,self.dx,self.x_pos,self.y_pos,self.length,self.width,self.amp,self.st,self.fn,t)
                elif self.shape == 'circle':
                    eps,eps_in,plxlf,plylf,plxrg,plyrg = IBM.circle(self.motion,self.ene,self.ny,self.nx,self.dy,self.dx,self.x_pos,self.y_pos,self.ra,self.amp,self.st,self.fn,t)

            ######### Time Integrators come here  #############################
            if self.ischeme == 'RK1':
                self.u,self.v,self.p,self.T = TIM.RK1(self.bod,self.ischeme,self.poi_b,self.ene,self.buo,self.bcxu,self.bcxv,self.bcyu,self.bcyv,self.bcxt,self.bcyt,                 self.L,self.H,self.ny,self.nx,self.dy,self.dx,self.rho,self.nu,self.Ri,self.Pr,self.dt,self.u,self.v,self.p,self.T,self.vel,t,self.A0,self.omp,eps,eps_in,self.T0,self.dog,self.hflux)
                
            elif self.ischeme == 'CPM':
                self.u,self.v,self.p,self.T = TIM.CPM(self.bod,self.ischeme,self.poi_b,self.ene,self.buo,self.bcxu,self.bcxv,self.bcyu,self.bcyv,self.bcxt,self.bcyt,                                                self.L,self.H,self.ny,self.nx,self.dy,self.dx,self.rho,self.nu,self.Ri,self.Pr,self.dt                                                ,self.u,self.v,self.p,self.T,self.vel,t,self.A0,self.omp                 ,eps,eps_in,self.T0,self.dog,self.hflux,i)
                
            elif self.ischeme == 'AB2':
                if i == 0:
                    un,vn,pn,Tn = TIM.RK1(self.bod,self.ischeme,self.poi_b,self.ene,self.buo,self.bcxu,self.bcxv,self.bcyu,self.bcyv,self.bcxt,self.bcyt,                                                self.L,self.H,self.ny,self.nx,self.dy,self.dx,self.rho,self.nu,self.Ri,self.Pr,self.dt                                                ,self.u,self.v,self.p,self.T,self.vel,t,self.A0,self.omp                                               ,eps,eps_in,self.T0,self.dog,self.hflux)
                    
                else:
                    self.u,self.v,self.p,self.T,un,vn,pn,Tn = TIM.AB2(self.bod,self.ischeme,self.poi_b,self.ene,self.buo,self.bcxu,self.bcxv,self.bcyu,self.bcyv,self.bcxt,self.bcyt,                                                self.L,self.H,self.ny,self.nx,self.dy,self.dx,self.rho,self.nu,self.Ri,self.Pr,self.dt                                                ,self.u,un,self.v,vn,self.p,pn,self.T,Tn,self.vel,t,self.A0,self.omp                                               ,eps,eps_in,self.T0,self.dog,self.hflux)
            
            ###################################################################
            # vorticity field calculation
            self.vort[1:-1,1:-1] = (self.v[1:-1,2:]-self.v[1:-1,:-2])/(2*self.dx) - (self.u[2:,1:-1]-self.u[:-2,1:-1])/(2*self.dy)
           
            # probing velocities to calculate shedding frequncies if required
            if self.probes == 'on':
                x_loc,y_loc = 15,1.5
                x_index,y_index = int(y_loc/dy),int(x_loc/dx)
                ppu = self.u[x_index,y_index]
                ppv = self.v[x_index,y_index]
                ppp = self.p[x_index,y_index]
                ppT = self.T[x_index,y_index]
                ppvo = self.vort[x_index,y_index]
                pro_u,pro_v,pro_p,pro_T  = np.append(pro_u,ppu),np.append(pro_v,ppv),np.append(pro_p,ppp),np.append(pro_T,ppT)
                pro_vo = np.append(pro_vo,ppvo)
                
            # Residue calculations
            resu,resv,mdef,resT = af.resids(self.ene,self.u,self.v,self.p,self.T,self.ny,self.nx,self.dy,self.dx,self.nu,self.Pr,self.H)
            resur,resvr,mdefr,resTr = np.append(resur,resu),np.append(resvr,resv),np.append(mdefr,mdef),np.append(resTr,resT)
            
            # calculation of aero coeffs and recording
            if self.aeros == 'on' and self.bod == 'IBM':
                if self.shape == 'square':
                    cl,cd,cp = af.aero_sq(self.ny,self.nx,self.p,eps,self.rho,self.vel)
                    clr,cdr,cpr = np.append(clr,cl),np.append(cdr,cd),np.append(cpr,cp)
                    if self.ene == 'on':
                        Nu = af.nusnum(self.T,self.T0,eps,self.dx,self.dy)
                        Nur = np.append(Nur,Nu)
                elif self.shape == 'circle':
                    cl,cd = af.aero_ir(self.ny,self.nx,plxlf,plylf,plxrg,plyrg,self.p)
                    clr,cdr = np.append(clr,cl),np.append(cdr,cd)
                    if self.ene == 'on':
                        Nu = af.nusnum_ir(self.dx,self.dy,plxlf,plylf,plxrg,plyrg,self.T,self.T0)
                        Nur = np.append(Nur,Nu)
                
            # time estimation lines
            if i < est_time:
                en_t = time.time() - st_t
                t_rec[i,0] = en_t
            if i == est_time:
                ti = np.average(t_rec[:,0])*nt
                if ti > 60:
                    ti = ti/60
                    print("ETC = "+str(ti)+ ' mins')
                elif ti < 60:
                    print("ETC = "+str(ti)+ ' secs')  
                    
            if (i+1)%self.isave == 0:
                af.vtr_exp(self.sdi,nt_st+i,self.L,self.H,self.ny,self.nx,self.u,self.v,self.p,self.vort,self.T)
                
                    
            ### CFL and inertia time scale check #############################
            C_til = self.dt *(self.u.max()/self.dx + self.v.max()/self.dy)
            ##################################################################
                    
                    
            # print displays and progression show
            if (i+1)%self.iprints == 0:
                print('##################################################################')
                print('Title : '+str(self.title))
                print('##################################################################')
                print('')
                print('Max u-vel = ' + str(round(self.u.max(),4)) + '; Min u-vel = ' + str(round(self.u.min(),4)))
                print('Max v-vel = ' + str(round(self.v.max(),4)) + '; Min v-vel = ' + str(round(self.v.min(),4)))
                print('Max vorti = ' + str(round(self.vort.max(),4)) + '; Min vorti = ' + str(round(self.vort.min(),4)))
                if self.ene == 'on':
                    print('Max tempe = ' + str(round(self.T.max(),4)) + '; Min tempe = ' + str(round(self.T.min(),4)))
                print('')
                print('Res_u = {} ; Res_v = {} ; continuity = {}'.format(round(resu,6),round(resv,6),round(mdef,6)))
                if self.ene == 'on':
                    print('Res_T = {}'.format(round(resT,6)))
                print('')
                if self.bod == 'IBM' and self.aeros == 'on':
                    print('Cd = '+str(round(cd,4)))
                    print('Cl = '+str(round(cl,4)))

                print('')
                print('CFL = {}'.format(round(C_til,4)))
                if C_til > 0.2:
                    print('Warning: Inertial timescale being violated. Solution may diverge.')
                print('Right now ' + str(nt_st + i+1) + ' iterations done, ' + str(nt-i-1) + ' more to go.')
                print('Time Elapsed is '+str(t) + ' secs.')
                print('')
                print('##################################################################')
            
            if (i+1)%self.idata == 0:    
                # Saving fields at the end of time loop end for restarting anytime after that
                self.u.tofile(self.sdi+'/out_it' + str(nt_st+i+1)+'_u.dat')
                self.v.tofile(self.sdi+'/out_it' + str(nt_st+i+1)+ '_v.dat')
                self.p.tofile(self.sdi+'/out_it' + str(nt_st+i+1)+ '_p.dat')
                self.vort.tofile(self.sdi+'/out_it' + str(nt_st+i+1)+ '_vort.dat')
                self.T.tofile(self.sdi+'/out_it' + str(nt_st+i+1)+ '_temp.dat')
                    
            if self.vort.max() > 1000 or self.T.max() > 100:
                print('##########################################################')
                print('Divergence Detected. Exiting Simulation.')
                print('##########################################################')
                exit()
        
        np.savetxt(self.sdi+'/pro_u.dat',pro_u,delimiter=',')
        np.savetxt(self.sdi+ '/pro_v.dat',pro_v,delimiter=',')
        np.savetxt(self.sdi+'/pro_p.dat',pro_p,delimiter=',')
        np.savetxt(self.sdi+'/pro_T.dat',pro_T,delimiter=',')
        np.savetxt(self.sdi+'/pro_vo.dat',pro_vo,delimiter=',')
        np.savetxt(self.sdi+'/Nur.dat',Nur,delimiter=',')
        np.savetxt(self.sdi+'/clr.dat',clr,delimiter=',')
        np.savetxt(self.sdi+'/cdr.dat',cdr,delimiter=',')
        np.savetxt(self.sdi+'/cpr.dat',cpr,delimiter=',')
        np.savetxt(self.sdi+'/residue_u.dat',resur,delimiter=',')
        np.savetxt(self.sdi+'/residue_v.dat',resvr,delimiter=',')
        np.savetxt(self.sdi+'/residue_cont.dat',mdefr,delimiter=',')
        np.savetxt(self.sdi+'/residue_tem.dat',resTr,delimiter=',')
        return self.u,self.v,self.p,self.vort,self.T
            

