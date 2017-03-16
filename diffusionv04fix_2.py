'''
Created on Feb 5, 2015

@author: H.xie

|----------------------------|
|                            |
|                            |
|                            |    ^y
|                            |    |
|----------------------------|    |
------------------------------>z
(z,y,vz,vy,v,ene,x)
{
The probability density function for expon is:

expon.pdf(x) = lambda * exp(- lambda*x)
for x >= 0.

The scale parameter is equal to scale = 1.0 / lambda.
rvs(loc=0, scale=1, size=1)
}
{
A special case of a chi distribution, with df = 3, loc = 0.0, and given scale = a, where a is the parameter used in the Mathworld description [R240].

The probability density function for maxwell is:

maxwell.pdf(x) = sqrt(2/pi)x**2 * exp(-x**2/2)
for x > 0.
rvs(loc=0, scale=1, size=1)    Random variates.
pdf(x, loc=0, scale=1)    Probability density function.
}
version v0: basic version, only e-p scattering
version v1: add photon energy scan function. could generate QE-Emittance, electrons loss factor
version v2, add e-h scattering function, add e-e scattering opinion, add EA scan
version v3, add e-e scattering function, ea scan for both emittance and electron loss. save figures. add green laser mark on scanned figure.
version v4, use a mfp/5 as step, reduce the step length, fix the first step too large issue
in v4plus, consider the e-p scatter may gain the energy
'''

import random
from scipy.stats import maxwell,expon
from scipy.constants import *
import matplotlib.pyplot as plt
import numpy as np

import math
from numpy import genfromtxt, dtype
from scipy.interpolate import interp1d
from scipy import integrate
import time
import os

start_time=time.time()
'''
def Max_boltE(ene,le):  ## Maxwell b distribution
    return 2(((ene/pi)**0.5)/le**1.5)*np.exp(-ene/le)
'''
ma=9.11*10**-31  # electron mess,kg
eff=0.1176
me=eff*ma  # effective electron mess,kg
bandgap=1.6   #1.2DOS12or 1.64(DOS16) or 1.6(DOS160)


ec=1.6*(10**(-19))

def MBdist(n,e_photon,thick): # n: particle number, loct: start point(x-x0), scale: sigma, wl: wavelength,thick: thickness of the cathode
    assert e_photon> bandgap
    if e_photon-bandgap-0.8<=0:
        scale=e_photon-bandgap
        loct=0
    else:
        scale=0.8
        loct=e_photon-bandgap-scale
    data = maxwell.rvs(loc=loct, scale=scale, size=n)
    data_ene=np.array(data)
    params = maxwell.fit(data, floc=0)
    data_v=np.sqrt(2*data_ene*ec/me)*10**9
    p2D=[]
    wl=((19.82-27.95*e_photon+11.15*e_photon**2)*10**-3 )**-1
    pens = expon.rvs(loc=0, scale=wl,size=n)
    penss=filter(lambda x:x<=thick,pens)
    params_exp=expon.fit(pens,floc=0)
    i=0
    
    for n in range(len(penss)):
        phi=random.uniform(0,2*math.pi) # initial angular
        poy=random.uniform(-1*10**6,1*10**6)  # initial y direction position
        p2D.append([penss[i],poy,data_v[i]*math.cos(phi),data_v[i]*math.sin(phi),data_v[i],data[i]])  #p2D: (z,y,vz,vy,v,ene)
        i+=1  
    p2D=np.array(p2D)     
    return params,p2D,penss,params_exp




def DosDist(n,e_photon,thick,data,absorb_data):# n, the photon numbers; loct, the postion(z) of the electrons;thick, unit, nm.
    
    f = interp1d(data[:,0], data[:,1])
    f2 = interp1d(absorb_data[:,0],absorb_data[:,1])

    n1 = int((e_photon-1)/0.01) # change to n 
    energy = np.linspace(1.,e_photon,n1)
    norm,err= integrate.quad(lambda e: f(e-e_photon)*f(e), 1,e_photon,limit=10000)
    
    data_ene = []
    num_energy = []
    i=0
    while i< n1:
        n3 =round(1.5*n*f(energy[i]-e_photon)*f(energy[i])*0.01/norm) #using n instead of n1
        num_energy.append(n3)
        ener_array = np.empty(n3)
        ener_array.fill(energy[i])
        data_ene.extend(ener_array)
        i+=1
    np.random.shuffle(data_ene)
    
    '''
    plt.subplot(211)
    plt.plot(data[:,0],data[:,1])
    plt.subplot(212)
    plt.hist(data_ene,bins=30)
    plt.show()
    '''
    p2D=[]
    '''wl=((19.82-27.95*e_photon+11.15*e_photon**2)*10**-3 )**-1'''
    wl=(f2(e_photon)*10**-3 )**-1
    pens = expon.rvs(loc=0, scale=wl,size=n)
    penss=list(filter(lambda x:x<=thick,pens))
    params_exp=expon.fit(pens,floc=0)    
   
    i=0
    for i in range(len(penss)):
        phi=random.uniform(0,2*math.pi) # initial angular on 2D surface
        php=random.uniform(0,2*math.pi)# initial angular on perpdiculat face
        poy=random.uniform(-1*10**6,1*10**6)  # initial y direction position, nm
        v = np.sqrt(2 * np.abs((data_ene[i]-bandgap))*ec/me)*10**9
        p2D.append([penss[i],poy,v*math.cos(phi)*math.cos(php),v*math.sin(phi)*math.cos(php),v,np.abs(data_ene[i]-bandgap)])  #p2D: (z,y,vz,vy,v,ene)
        i+=1  
    p2D=np.array(p2D) 
   
    #print p2D
    return params_exp,p2D,penss,params_exp
    
def diff(stept,endT,p2Di,mfp_ps,eloss,bounde,bounda,types):
    
    
    p2D_emission=[]
    p2D_trap=[]
    p2D_back=[]

    
        
    if types==1:
        tmatix_ns=np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[endT,0,1,0,0,0],[0,endT,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])# non scattering transistion matrix
        p2De=np.dot(p2Di,tmatix_ns)
        #p2De[:,0]=np.clip(p2De[:,0],bounde,bounda) 
        se,st,sb,p2De=swap(bounde,bounda,0,p2De)
        p2D_emission.extend(se.tolist())
        p2D_trap.extend(st.tolist())
        p2D_back.extend(sb.tolist())
        p2De=p2De.tolist()
        
        p2De.extend(p2D_back)
        p2De.extend(p2D_trap)
        p2De.extend(p2D_emission)
        
        p2De=np.array(p2De)
        p2De[:,5]=np.maximum(p2De[:,5],0)
        p2De[:,0]=np.clip(p2De[:,0],bounde,bounda) 
        
    elif types==2:
        tmatix_diff=np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[endT,0,1,0,0,0],[0,endT,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
        t=0
        for t in range(int(endT/stept)):
            q=random.uniform(0,2*math.pi)
            tmatix_difft=np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[stept,0,0,0,0,0],[0,stept,0,0,0,0],[0,0,math.cos(q),math.sin(q),1,0],[0,0,0,0,0,1]])
            tmatix_diff=np.dot(tmatix_diff,tmatix_difft)
            t+=1
        p2De=np.dot(p2Di,tmatix_diff)
        #p2De[:,0]=np.clip(p2De[:,0],bounde,bounda) 
        se,st,sb,p2De=swap(bounde,bounda,0,p2De)
        p2D_emission.extend(se.tolist())
        p2D_trap.extend(st.tolist())
        p2D_back.extend(sb.tolist())

        p2De=p2De.tolist()
        p2De.extend(p2D_back)
        p2De.extend(p2D_trap)
        p2De.extend(p2D_emission)
   
   
        p2De=np.array(p2De)
        print (p2De)
        p2De[:,5]=np.maximum(p2De[:,5],0)
        
        print (p2De,bounde,bounda)
        
        p2De[:,0]=np.clip(p2De[:,0],bounde,bounda) 
    elif types==3:
        #tmatix_i=np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[endT,0,1,0,0,0],[0,endT,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
        t=0
        #p2De=np.dot(p2Di,tmatix_i)
        
        
        eemfpdata = genfromtxt('K2CsSb_ees.csv',delimiter=',')


        eemfp= interp1d(eemfpdata[:,0], eemfpdata[:,1])
        p2De=p2Di
        tempres=[]
        while t < endT:
            
            ee=eloss  # random energy loss  2.5 * np.random.randn(2, 4) + 3// sigma * np.random.randn(...) + mu
            mfp=np.abs((mfp_ps/3)*np.random.randn()+mfp_ps)   #random mfp
           
            stept=mfp/(np.sqrt(2*np.mean(p2De[:,5])*ec/me)*10**9)/5.0  # 0.2*tau is the step for collision
            # print np.mean(p2De[:,5]),'   ',np.sqrt(2*np.mean(p2De[:,5])*ec/me),'   ', mfp,'   ', stept
            tmatix_epsim=np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[stept,0,1,0,0,0],[0,stept,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
            p2De=np.dot(p2De,tmatix_epsim)
            
             
            
            lossoverall=1  #1
                        
            ranlist=np.array([ee]*int(len(p2De)*0.2*lossoverall)+ [-ee]*int(len(p2De)*0.2*(1-lossoverall)) +[0]*(len(p2De)-int(len(p2De)*0.2*lossoverall)-int(len(p2De)*0.2*(1-lossoverall))))
            np.random.shuffle(ranlist)
            nor_ranlist=(ranlist/ee).astype(int)
            p2De[:,5]=p2De[:,5]-ranlist
            
            
            
            for k in range(len(p2De)):           
                
                # start a e-e scattering   stept     ,eemfp, and e-h scattering
                if p2De[k-1,5]>bandgap:
                    r=float(mfp/(eemfp(p2De[k-1,5]-bandgap)/10)/5)# mfp/2 is average loss? remove later, /5 is match the tstep
                    
                    sec1=random.randint(0,1) ## ee-eh random.randint(0,1); if ee set to 1
                    selc=[sec1,1-sec1] # the ratio of e-e and e-h scattering
                    #print int(r*100)
                    if r<=1:
                        #ranlist=[0]*(100-int(r*100))+[1]*int(r*100)
                        if random.random()<=r:
                            happ=1
                        else:
                            happ=0
                        
                        p2De[k-1,5]=p2De[k-1,5]- random.uniform(bandgap,p2De[k-1,5])*happ*selc[0] ##random.uniform(bandgap,p2De[k-1,5])*happ*selc[0] #e-e scattering
                        p2De[k-1,5]=p2De[k-1,5]- random.uniform(0,1.27)*happ*selc[1] #e-h scattering
                     
                    else:
                        happ=1
                        p2De[k-1,5]=p2De[k-1,5]-np.minimum(int(r),int(p2De[k-1,5]/bandgap))*random.uniform(bandgap,p2De[k-1,5])*selc[0]      ##*random.uniform(bandgap,p2De[k-1,5])*selc[0]      e-e scattering *int(mfp/(eemfp(p2De[k-1,5]-bandgap)/10))
                        p2De[k-1,5]=p2De[k-1,5]-np.minimum(int(r),int(p2De[k-1,5]/bandgap))*random.uniform(0,1.27)*selc[1]#e-h scattering
                else:
                    happ=0   
                
                
                if p2De[k-1,5]>0:
                    p2De[k-1,4]=np.sqrt(2*p2De[k-1,5]*ec/me)*10**9
                    q=random.uniform(0,2*math.pi)
                    qp=random.uniform(0,2*math.pi)
                    if np.maximum(happ,nor_ranlist[k-1])==1:
                        p2De[k-1,2]=p2De[k-1,4]*math.cos(q)*math.cos(qp)*np.maximum(happ,nor_ranlist[k-1])
                        p2De[k-1,3]=p2De[k-1,4]*math.sin(q)*math.cos(qp)*np.maximum(happ,nor_ranlist[k-1])
                   
            
            se,st,sb,p2De=swap(bounde,bounda,0,p2De,ee)    
                
            
            p2D_emission.extend(se.tolist())
            p2D_trap.extend(st.tolist())
            p2D_back.extend(sb.tolist())
            ##print (t,'   ', len(p2De),'  ',len(se),'   ',len(st),'    ',len(sb))
            
            tempres.append([t*10**15,len(se)])# change s to fs
            
            
            t+=stept
            
            if len(p2De)==0:
                #print p2De,p2D_trap
                break
            
        p2De=p2De.tolist()
        p2De.extend(p2D_back)
        p2De.extend(p2D_trap)
        p2De.extend(p2D_emission)
        p2De=np.array(p2De)
        p2De[:,5]=np.maximum(p2De[:,5],0)
        p2De[:,0]=np.clip(p2De[:,0],bounde,bounda) 
            
    
    else:
        print ("wrong type: 1 is non scattering; 2 is brown diffusion 3 is consider scattering")
            
   
    '''
    p2De[:,0]=map(lambda x:bounde if (x < bounde) else x,p2De[:,0])
    p2De[:,0]=map(lambda x:bounda if (x > bounda) else x,p2De[:,0])
    '''
    p2D_emission= np.array(p2D_emission)
    
    if len(p2D_emission)!=0:
        
    
        p2D_emission[:,0]=bounde
    
    
    p2D_back= np.array(p2D_back)
    
    if len(p2D_back)!=0: 
        p2D_back[:,0]=bounda
    
    p2D_trap = np.array(p2D_trap)
    if len(p2D_trap)!=0:
        p2D_trap[:,2]=0
    tempres=np.array(tempres)
    
    return p2D_emission,p2D_trap,p2D_back,p2De,tempres

def swap(be,ba,bb,mat,el):
    assert ba > be
    emissind=mat[:,0]<=be
    backind=mat[:,0]>=ba
    loweind=mat[:,5]<=bb
    
    swape=mat[emissind,:]
    swape[:,5]=swape[:,5]+el
    swapt=mat[(~emissind) & (~backind)& loweind,:]
    swapb=mat[backind,:]
    swapr=mat[(~emissind) & (~loweind) & (~backind),:]
    #print swapt.shape
    return swape, swapt, swapb, swapr

def emission(mat,ea,sctk):
    
    matind=mat[:,5]>=(ea+sctk)
    mat2=mat[matind,:]
    
    mat2[:,5]=mat2[:,5]-ea-sctk
    phpangle=np.random.uniform(0,2*math.pi,(len(mat2),1))
    mat2=np.append(mat2, phpangle, 1)
    mat2[:,4]=np.sqrt(2*mat2[:,5]*ec/me)*np.cos(mat2[:,6])*10**9
    mat2ind=np.abs(mat2[:,4])> np.abs(mat2[:,3])
    #print mat2ind
    emission=mat2[mat2ind,:]
    
    emission[:,2]=np.sqrt(emission[:,4]**2-emission[:,3]**2)
    surface_trap=len(mat)-len(emission)
    '''
    gamma=(np.mean(emission[:,5])+511000.0)/511000.0
    beta=np.sqrt(1-gamma**-2)
    Temittance= gamma*beta*np.sqrt(np.mean(emission[:,1]**2)*(10**-9)*np.mean((emission[:,3]/emission[:,2])**2)-np.mean(emission[:,1]*(10**-9)*(emission[:,3]/emission[:,2]))**2)  #sqrt(<x**2><x'**2>-<xx'>**2)
    '''
    Temittance=np.sqrt(np.mean((emission[:,1]*10**-9)**2))*(np.sqrt(np.mean(((emission[:,3]*10**-9)/np.cos(emission[:,6]))**2)))*np.sqrt(eff)/(3*10**8)  #sigma_x*sqrt(<v_x**2>)/c
    #print np.mean(emission[:,1]**2),np.mean(emission[:,1]**2),'\n',np.mean((emission[:,3]/emission[:,2])**2),'\n',np.mean(emission[:,1]*(emission[:,3]/emission[:,2]))**2
    #print np.sqrt(np.mean((emission[:,1]*10**-9)**2)),np.sqrt(np.mean(emission[:,3]**2))/(3*10**8)
    return emission, Temittance,surface_trap

def plot(p2D,pens,p2De,para,params_exp,thickness,tempres,petest,electron_affinity,finame):
    #print tempres
    with open(finame+'tempres2_tk_%d.csv' %(thickness),'ab') as csv:
        np.savetxt(csv,tempres,fmt='%.8e',delimiter=',')
    plt.subplot(511)
    plt.xlabel('thickness[nm]')
    plt.ylabel('P')
    plt.hist(p2D[:,5], bins=20, normed=True)
    x = np.linspace(0, 5, 80)  
    if para !=None :
        plt.plot(x, maxwell.pdf(x, *para),'r',x,maxwell.cdf(x, *para), 'g')
    
    plt.subplot(512)
    plt.hist(pens, bins=20, normed=True)
    z=np.linspace(0,500,200)
    plt.xlabel('thickness[nm]')
    plt.ylabel('P')
    plt.xlim(0,2*thickness)  
    plt.plot(z, expon.pdf(z, *params_exp),'r')#plt.plot(z, expon.pdf(z, *params_exp),'r',z,expon.cdf(z, *params_exp), 'g')
    
    plt.subplot(513)
    plt.xlim(0,thickness+1)  
    plt.xlabel('thickness[nm]')
    plt.ylabel('size[nm]')
    plt.plot(p2D[:,0],p2D[:,1],'b,')
    
    plt.subplot(514)
    plt.ylim(-1*10**6,1*10**6)
    plt.xlim(0,thickness+1)
    plt.xlabel('thickness[nm]')
    plt.ylabel('size[nm]')
    plt.plot(p2De[:,0],p2De[:,1],'b,')
    
    
    plt.subplot(515)

    
    plt.xlabel('Time[s]')
    plt.ylabel('count[1]')
    smooth=80

    for i in np.arange(0,len(tempres)):   
        
        tempres[i,1]=(np.sum(tempres[i:i+smooth,1])/smooth).astype(float)
    
    plt.ylim(0,tempres[1,1]*1.5)
    plt.plot(tempres[:,0],tempres[:,1],'r-')
    
    
    
    plt.savefig(finame+'full_tk_%dpe_%.3fea_%.3f.pdf'%(thickness,petest,electron_affinity))
    #plt.show()
    return

def emittancescanplot(wlscan,start,stop,ea,thickness,mfp,expdata,finame):
    with open(finame+'emittance_ea_%.3fwl_scan_mfp_%.3ftk_%dbg_%.3feff_%.4f.csv' %(ea,mfp,thickness,bandgap,eff),'ab') as csv:
        np.savetxt(csv,wlscan,fmt='%.8e',delimiter=',')
    '''with open(finame+'greenqe.csv','ab') as csv:
        np.savetxt(csv,greenqe,fmt='%.8e',delimiter=',')'''
  
   
    fig,ax1=plt.subplots(figsize=(16,8))
    ax2=ax1.twinx()
    
    ax1.plot(wlscan[:,0],wlscan[:,1]*10**6,'g-')
    ax1.plot(2.3, 0.39,'go')
    ax2.plot(wlscan[:,0],wlscan[:,2],'b-')
    ax2.plot(expdata[:,0],expdata[:,1],'ro',label='%.1f' % expdata[0,1])
    ax1.set_xlabel('photon energy[eV]')
    #plt.xlim(1.6,6.5)
    ticks=np.arange(start-0.2,stop+0.2,0.2)
    ax1.set_xticks(ticks)
    #print ticks
    ax1.set_ylabel('emittance [mrad]',color='g')
   
    ax2.set_ylabel('QE[%]',color='b')
    ax2.set_yscale('log')
    ax2.margins(0.02)        
    plt.savefig(finame+'emittance_ea_%.3fmfp_%.3ftk_%dbg_%.3feff_%.4f2.pdf' %(ea,mfp,thickness,bandgap,eff))
    

def electronlossscanplot(wlscan,start,stop, pestep,mfp,electron_affinity,thickness,ee,eh,expdata,finame):
    with open(finame+'wl_scan_ea_%.3fmfp_%.3fee_%deh_%d.csv' %(electron_affinity,mfp,ee,eh),'ab') as csv:
        np.savetxt(csv,wlscan,fmt='%.8e',delimiter=',')
    
            
    fig,ax1=plt.subplots(figsize=(16,8))
            
    ax1.plot(wlscan[:,0],wlscan[:,2],'b-',label='emission')
    ax1.plot(wlscan[:,0],wlscan[:,3],'r-',label='laser loss')
    ax1.plot(wlscan[:,0],wlscan[:,4],'g-',label='diffuse to back')
    ax1.plot(wlscan[:,0],wlscan[:,5],'y-',label='trap in crystal')
    ax1.plot(wlscan[:,0],wlscan[:,6],'k-',label='trap on surface')
    ax1.plot(expdata[:,0],expdata[:,1],'bo',label='%.1f' % expdata[0,1])
    ind_annot=(2.3-(start))/pestep
            
    #print bandgap1,electron_affinity,(2.3-(start))/pestep,wlscan[ind_annot,0],wlscan[0,0]
    ax1.annotate('(%s,%s)' % (wlscan[ind_annot+1,0],wlscan[ind_annot+1,2]),xy=(wlscan[ind_annot+1,0],wlscan[ind_annot+1,2]), textcoords='offset points')
            
            
            
    legend = ax1.legend(loc='best')
    ax1.set_xlabel('photon energy[eV]')
    #plt.xlim(1.6,6.5)
    ticks=np.arange(start-0.2,stop+0.2,0.2)
    ax1.set_xticks(ticks)
         
    ax1.set_ylabel('rate[%]',color='b')
    ax1.set_yscale('log')
    ax1.margins(0.02)        
    plt.savefig(finame+'mfp_%.3fea_%.3ftk_%dee_%deh_%d2.pdf' %(mfp,electron_affinity,thickness,ee,eh))

                    


def main(op):
    
    dosdata = genfromtxt('K2CsSb DOS16.csv',delimiter=',')  #DOS16: bandgap:1.64eV,  DOS12: bandgap:1.23eV
    #print(dosdata)
    expdata= genfromtxt('cooling_exp.csv',delimiter=',')
    absorb_data = genfromtxt('absorption.csv',delimiter=',') #absorption coefficient from reference
    electron_affinity=0.3  #0.4
    thickness=40
    n=300000
    pestart=1.4
    peend=4.7
    pestep=0.1
    petest=2.3
    bandgap1=float('%.1f' % bandgap)
    schottky=0
    tstep=0.00001 
    tend=0.01
    mfp=3  #5
    emisposition=0
    eloss=0.027#0.01  mfp  3nm,0.027eV
    ee=1
    eh=1
    finame=os.path.basename(__file__)[:-3]

    
    if op == 0:# emittance vs qe with maxwell
        para,p2D,pens,params_exp=MBdist(n,petest,thickness)# partcle #, start energy, sigma, apsorbtion length, thichness
        #p2D: initial energy distribution, pens: initial depth distribution, thickness: sample thickness, para and params_exp are fitting parameters
        esurf,trap,back,p2De,tempres=diff(tstep,tend,p2D,mfp,eloss,emisposition,thickness,3) # diff(stept,endT,p2Di,mfp_ps,bounde,bounda,types):
        # time step, time end, inital beam, mfp, emission surface, thickness, calculation type
        # esurf: reach to surface, trap: trap in body, back: get to back, p2De: all the particulates information
        emiss,emittance,surtrap=emission(esurf,electron_affinity,schottky)  # beam reach to surface, electron affinity, schottky
        #emiss: emit out beam, emittance:rms nor thermal emittance, surtrap: trap at surface
        #print emittance, float(len(emiss)/n)
        plot(p2D,pens,p2De,para,params_exp,thickness,tempres,finame)
        
        for electron_affinity in np.arange(0.1,0.8,0.1):
            wlscan=[]
            start=bandgap1+electron_affinity+0.1
            for ep in np.arange(start,peend,pestep):
                emisst,emittancet, surtrapt=emission(diff(tstep,tend,MBdist(n,ep,thickness)[1],mfp,eloss,emisposition,thickness,3)[0],electron_affinity,schottky)
                qe=len(emisst)*100/float(n)
                print (ep, '    ',emittancet,'   ', qe,'%' )
                wlscan.append([ep, emittancet,qe])
            wlscan=np.array(wlscan)
            emittancescanplot(wlscan,start,peend,electron_affinity,thickness,mfp,expdata,finame)
        print (wlscan)
        



    elif op==1:   # wavelength and ea scan with electrons go
        '''
        data_ene,p2D,pens,params_exp=DosDist(n,petest,thickness,dosdata,absorb_data)
        #p2D: initial energy distribution, pens: initial depth distribution, thickness: sample thickness, para and params_exp are fitting parameters
        esurf,trap,back,p2De,tempres=diff(tstep,tend,p2D,mfp,eloss,emisposition,thickness,3) # diff(stept,endT,p2Di,mfp_ps,bounde,bounda,types):
        # time step, time end, inital beam, mfp, emission surface, thickness, calculation type
        # esurf: reach to surface, trap: trap in body, back: get to back, p2De: all the particulates information
        emiss,emittance,surtrap=emission(esurf,electron_affinity,schottky)  # beam reach to surface, electron affinity, schottky
        #emiss: emit out beam, emittance:rms nor thermal emittance, surtrap: trap at surface
        #print emittance, float(len(emiss)/n)
        #print emiss
        plot(p2D,pens,p2De,None,params_exp,thickness)
        #des=open('eaep_scan.dat','w')
        '''
        
        for electron_affinity in np.arange(0.02,0.1,0.02):
            wlscan=[]
            bandgap1=float('%.1f' % bandgap)
            start=bandgap1+electron_affinity+0.1
            for ep in np.arange(start,peend,pestep):
                ppd=DosDist(n,ep,thickness,dosdata,absorb_data)[1]
                es,tr,ba,p2dee,tempres=diff(tstep,tend,ppd,mfp,eloss,emisposition,thickness,3)
                emisst,emittancet, surtrapt=emission(es,electron_affinity,schottky)
                
                qe=len(emisst)*100/float(n)
                labs=(float(n)-len(ppd))*100/float(n)
                backd=len(ba)*100/float(n)
                trapin=len(tr)*100/float(n)
                trapsur=surtrapt*100/float(n)
                
                '''
                print ep,'  ',len(emisst)+(float(n)-len(ppd))+len(ba)+len(tr)+surtrapt
                
                
                qe=len(emisst)*100/float(len(ppd)) 
                labs=0         
                backd=len(ba)*100/float(len(ppd))
                trapin=len(tr)*100/float(len(ppd))
                trapsur=surtrapt*100/float(len(ppd))
                '''
                print (ep, '    ',emittancet,'   ', qe,'%' )
                wlscan.append([ep, emittancet,qe,labs,backd,trapin,trapsur,electron_affinity])
                
            wlscan=np.array(wlscan)           
            stop=peend
            electronlossscanplot(wlscan,start,stop, pestep,mfp,electron_affinity,thickness,ee,eh,expdata,finame)
        plot(p2D,pens,p2De,None,params_exp,thickness)
        plt.show()
        with open(finame+'p2De.csv','ab') as csv:
            np.savetxt(csv,p2De,fmt='%.8e',delimiter=',') 
        #plt.show()
        #print wlscan  
        
        
        
    
    elif op==2:  # emittance vs qe with Dos, wavelength and ea scan
        '''
        data_ene,p2D,pens,params_exp=DosDist(n,petest,thickness,dosdata,absorb_data)
        #p2D: initial energy distribution, pens: initial depth distribution, thickness: sample thickness, para and params_exp are fitting parameters
        esurf,trap,back,p2De,tempres=diff(tstep,tend,p2D,mfp,eloss,emisposition,thickness,3) # diff(stept,endT,p2Di,mfp_ps,bounde,bounda,types):
        # time step, time end, inital beam, mfp, emission surface, thickness, calculation type
        # esurf: reach to surface, trap: trap in body, back: get to back, p2De: all the particulates information
        emiss,emittance,surtrap=emission(esurf,electron_affinity,schottky)  # beam reach to surface, electron affinity, schottky
        #emiss: emit out beam, emittance:rms nor thermal emittance, surtrap: trap at surface
        #print emittance, float(len(emiss)/n)
        #print emiss
        plot(p2D,pens,p2De,None,params_exp,thickness,tempres,petest,electron_affinity)
        '''
        
        '''for electron_affinity in np.arange(0.2,0.6,0.01):'''
      
        
        '''for thickness in np.arange(10,300,30):'''
        '''for theta in np.arange(0,pi/2,0.01):
            electron_affinity = 0.3-(np.math.sin(theta))**0.5*0.11 # at peak surface field, the change in surface potential is 0.15 eV'''
        for electron_affinity in np.arange(0.0,0.6,0.05):
            wlscan=[]
            greenqe = []
            bandgap1=float('%.1f' % bandgap)
            start=bandgap1+0.22+0.1
            
            for ep in np.arange(start,peend,pestep):
                emisst,emittancet, surtrapt=emission(diff(tstep,tend,DosDist(n,ep,thickness,dosdata,absorb_data)[1],mfp,eloss,emisposition,thickness,3)[0],electron_affinity,schottky)
                '''plot(p2D,pens,p2De,para,params_exp,thickness,tempres,finame)'''
                qe=len(emisst)*100/float(n)
                print (ep, '    ',emittancet,'   ', qe,'%' )
                wlscan.append([ep, emittancet,qe,electron_affinity])                
                
             
            wlscan=np.array(wlscan)
            greenqe.append([electron_affinity,wlscan[0][2]])
            greenqe = np.array(greenqe)
            emittancescanplot(wlscan,start,peend,electron_affinity,thickness,mfp,expdata,greenqe,finame)
          
        
    elif op==3:   # wavelength scan with single ea electrons go
        
            wlscan=[]
            
            for ep in np.arange(start,peend,pestep):
                ppd=DosDist(n,ep,thickness,dosdata,absorb_data)[1]
                es,tr,ba,p2dee,tempres=diff(tstep,tend,ppd,mfp,eloss,emisposition,thickness,3)
                emisst,emittancet, surtrapt=emission(es,electron_affinity,schottky)
                
                qe=len(emisst)*100/float(n)
                labs=(float(n)-len(ppd))*100/float(n)
                backd=len(ba)*100/float(n)
                trapin=len(tr)*100/float(n)
                trapsur=surtrapt*100/float(n)
                
                '''
                print ep,'  ',len(emisst)+(float(n)-len(ppd))+len(ba)+len(tr)+surtrapt
                
                
                qe=len(emisst)*100/float(len(ppd)) 
                labs=0         
                backd=len(ba)*100/float(len(ppd))
                trapin=len(tr)*100/float(len(ppd))
                trapsur=surtrapt*100/float(len(ppd))
                '''
                print (ep, '    ',emittancet,'   ', qe,'%')
                wlscan.append([ep, emittancet,qe,labs,backd,trapin,trapsur,electron_affinity])
                
            wlscan=np.array(wlscan)           
            stop=peend
            electronlossscanplot(wlscan,start,stop, pestep,mfp,electron_affinity,thickness,ee,eh,expdata,finame)
        #plt.show()
        #print wlscan  
           
    elif op==4:   # emittance vs qe with Dos, wavelength scan wt single ea
          
        wlscan=[]
        start=bandgap1+electron_affinity+0.1
            
        for ep in np.arange(start,peend,pestep):
            emisst,emittancet, surtrapt=emission(diff(tstep,tend,DosDist(n,ep,thickness,dosdata,absorb_data)[1],mfp,eloss,emisposition,thickness,3)[0],electron_affinity,schottky)
            qe=len(emisst)*100/float(n)
            print (ep, '    ',emittancet,'   ', qe,'%' )
            wlscan.append([ep, emittancet,qe])
               
        wlscan=np.array(wlscan)
        emittancescanplot(wlscan,start,peend,electron_affinity,thickness,mfp,expdata,finame)

    elif op==5: #for test
        
        data_ene,p2D,pens,params_exp=DosDist(n,petest,thickness,dosdata,absorb_data)
        #p2D: initial energy distribution, pens: initial depth distribution, thickness: sample thickness, para and params_exp are fitting parameters
        esurf,trap,back,p2De,tempres=diff(tstep,tend,p2D,mfp,eloss,emisposition,thickness,3) # diff(stept,endT,p2Di,mfp_ps,bounde,bounda,types):
        # time step, time end, inital beam, mfp, emission surface, thickness, calculation type
        # esurf: reach to surface, trap: trap in body, back: get to back, p2De: all the particulates information
        emiss,emittance,surtrap=emission(esurf,electron_affinity,schottky)  # beam reach to surface, electron affinity, schottky
        #emiss: emit out beam, emittance:rms nor thermal emittance, surtrap: trap at surface
        #print emittance, float(len(emiss)/n)
        #print emiss
        
        print (petest, '    ',emittance,'   ',len(emiss)*100/float(n),'%' )
        plot(p2D,pens,p2De,None,params_exp,thickness,tempres,petest,electron_affinity,finame)      
        
        
    #print("----%s sceonds----" %(time.time()-start_time))
    
    elif op==6: #thickness scan of option 5
        
        
        for thickness in np.arange(10,300,30):
        
            data_ene,p2D,pens,params_exp=DosDist(n,petest,thickness,dosdata,absorb_data)
            #p2D: initial energy distribution, pens: initial depth distribution, thickness: sample thickness, para and params_exp are fitting parameters
            esurf,trap,back,p2De,tempres=diff(tstep,tend,p2D,mfp,eloss,emisposition,thickness,3) # diff(stept,endT,p2Di,mfp_ps,bounde,bounda,types):
            # time step, time end, inital beam, mfp, emission surface, thickness, calculation type
            # esurf: reach to surface, trap: trap in body, back: get to back, p2De: all the particulates information
            emiss,emittance,surtrap=emission(esurf,electron_affinity,schottky)  # beam reach to surface, electron affinity, schottky
            #emiss: emit out beam, emittance:rms nor thermal emittance, surtrap: trap at surface
            #print emittance, float(len(emiss)/n)
            #print emiss
            
           
            plot(p2D,pens,p2De,None,params_exp,thickness,tempres,petest,electron_affinity,finame)      
    elif op==7:
        '''
        
        for thickness in np.arange(10,300,30):'''
        for theta in np.arange(0,pi/2,0.01):
            electron_affinity = 0.15-(np.math.sin(theta))**0.5*0.15 # at peak surface field, the change in surface potential is 0.15 eV
            wlscan=[]
            greenqe = []
            bandgap1=float('%.1f' % bandgap)
            start=bandgap1+0.22+0.1
            
            for ep in np.arange(start,peend,pestep):
                data_ene,p2D,pens,params_exp=DosDist(n,ep,thickness,dosdata,absorb_data)
                esurf,trap,back,p2De,tempres=diff(tstep,tend,p2D,mfp,eloss,emisposition,thickness,3)
                emiss,emittance,surtrap=emission(esurf,electron_affinity,schottky)
                plot(p2D,pens,p2De,None,params_exp,thickness,tempres,petest,electron_affinity,finame) 
                qe=len(emiss)*100/float(n)
                print (ep, '    ',emittance,'   ', qe,'%' )
                wlscan.append([ep, emittance,qe,electron_affinity])                
                
          
            wlscan=np.array(wlscan)           
            greenqe.append([theta,thickness,wlscan[0][2]])            
            greenqe = np.array(greenqe)
            emittancescanplot(wlscan,start,peend,electron_affinity,thickness,mfp,expdata,greenqe,finame) 
        '''print (tempres2)
        tempres2 = np.array(tempres2).tolist()  
        print (len(tempres2))
        print (tempres)
        print (tempres2)'''
        '''with open(finame+'tempres2.csv','ab') as csv:
            np.savetxt(csv,tempres2,fmt='%.8e',delimiter=',')'''
    elif op==8:    
    
        wlscan=[]
        ep = 2.32   
        bandgap1=float('%.1f' % bandgap)
        start=bandgap1+0.22+0.1
            
        for electron_affinity in np.arange(0,0.6,0.01):
            emisst,emittancet, surtrapt=emission(diff(tstep,tend,DosDist(n,ep,thickness,dosdata,absorb_data)[1],mfp,eloss,emisposition,thickness,3)[0],electron_affinity,schottky)
            '''plot(p2D,pens,p2De,para,params_exp,thickness,tempres,finame)'''
            qe=len(emisst)*100/float(n)
            print (electron_affinity, '    ',emittancet,'   ', qe,'%' )
            wlscan.append([ep, emittancet,qe,electron_affinity])                
                
             
        wlscan=np.array(wlscan)
        print wlscan
        
        
        emittancescanplot(wlscan,start,peend,electron_affinity,thickness,mfp,expdata,finame)
          
            
        
        
    print("----%s sceonds----" %(time.time()-start_time))
        
if __name__ == '__main__':
    
    main(8)