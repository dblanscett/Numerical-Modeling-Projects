# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 14:26:37 2021

This code simulates a molecular dynamic model for which we have N particles 
in a box with periodic boundary conditions and the assumption that energy is 
conserved. Several different initial states are chosen and run to equilibrium.
these equilibirum states are compared to see if our initial conditions 
influence the outcome. 

@author: Devan
"""

from numpy import array,copy, zeros_like, zeros, histogram, linspace
from numpy.random import rand
import matplotlib.pyplot as plt
from numpy.linalg import norm

class MD:#Parameters: Number Particles, Box length, lattice vctors and run time
    def __init__(s,N=20, bLen=10,a1=[2,0],a2=[0,2],runTime=5):
        s.N=N
        s.bLen=bLen
        s.a1=array(a1)
        s.a2=array(a2)
        s.dt=0.02
        s.runTime=runTime
        
        lattice=[]#holds positions
        start=[0.0,0.0]    
        
        firstCol=[]
        firstCol.append([start[0],start[1]])
        
        while start[1]+s.a2[1]<bLen:
            start+=copy(s.a2)
            if start[0]>s.a2[0]:
                start[0]-=2*s.a2[0] #first column steps up left, then up right, to not lose space in the top left corner of box
            firstCol.append([start[0],start[1]])
        
        for i in range(len(firstCol)):
            if len(lattice)>=s.N:
                s.lattice=array(lattice)%s.bLen
                break
            lattice.append(firstCol[i])
            rowStart=array(firstCol[i])
            while rowStart[0]+s.a1[0]<bLen:
                if len(lattice)>=s.N:
                    s.lattice=array(lattice)%s.bLen
                    break
                rowStart+=copy(s.a1)
                lattice.append([rowStart[0],rowStart[1]])
        s.lattice=array(lattice)%s.bLen
        
        #lattice made, time to randomize it
        s.lattice=array([[x[0]+(rand()-0.5),x[1]+(rand()-0.5)] 
                         for x in s.lattice])%s.bLen
        
    def initialize(s,percentX=0.7):
        s.pos=copy(s.lattice) #current positions
        s.vel=zeros_like(s.pos) #current velocity
        #if len(s.vel) is odd the last vel in the array is 0,0 but this is ok
        #because our average KE is zero
        #need to update so v pos == v neg
        trackVel=0.0 #make sure we have just as much positive vel as neg

        for i in range(int(len(s.vel)/2)):
            randVel=rand()
            trackVel+=randVel
            #print(trackVel)
            s.vel[i,0],s.vel[i,1]=randVel*percentX,randVel*(1-percentX)
        
        for i in range(int(len(s.vel)/2)):
            j=i+int(len(s.vel)/2)
            randVel=-rand()
            if i == int(len(s.vel)/2)-1:
                s.vel[j,0],s.vel[j,1]=-trackVel*percentX,-trackVel*(1-percentX)
                #print(trackVel)
            elif randVel > trackVel:
                s.vel[j,0],s.vel[j,1]=-trackVel*percentX,-trackVel*(1-percentX)
                trackVel=0
                #print(trackVel)
            else:
                s.vel[j,0],s.vel[j,1]=randVel*percentX,randVel*(1-percentX)
                trackVel+=randVel #randVel already negative
                #print("minus",trackVel)
        
        s.posPrev=(s.pos-s.vel*s.dt)%s.bLen
        
    def showParticles(s,positions):
        plt.plot([x[0] for x in positions],[x[1] for x in positions],'b.',ms=10)
        plt.xlim(0,s.bLen)
        plt.ylim(0,s.bLen)
        plt.show()
        plt.draw()
        plt.clf()
        
    def getShortestDistance(s,tPos,pPos):  
        periods=array([pPos, pPos+array([-s.bLen,0]),pPos+array([s.bLen,0]),
                       pPos+array([0,-s.bLen]),pPos+array([0,s.bLen])])
        distances = [norm(x) for x in periods-tPos] 
        minDex=distances.index(min(distances))
        return(tPos-periods[minDex])
        
    def getNetForceOnSingleParticle(s,particle):
        forceOnOne=0
        for i in range(len(s.pos)):
            distBetween=s.getShortestDistance(particle,s.pos[i])
            if norm(distBetween) > 0.8 and norm(distBetween) < 3.0:
                rMag=norm(distBetween)
                fMag=24*(2/rMag**13-1/rMag**7)
                forceOnOne+=(distBetween/rMag)*fMag
        return(forceOnOne)
    
    def simulate(s,xPercent=0.7):
        s.initialize(xPercent)
        time = 0
        index = 0
        lineNumber=1
        s.showParticles(s.posPrev)
        s.showParticles((s.pos))
        countAvg=0
        countList=[]
        binList=[]
        
        while time < 4:
            forces=zeros([len(s.pos),2])
            for i,k in enumerate(s.pos):
                forces[i]=s.getNetForceOnSingleParticle(k)
            s.posNew=(2*s.pos-s.posPrev+forces*s.dt**2)%s.bLen
            s.vel=(s.posNew-s.posPrev)/(2*s.dt)
            time+=s.dt
            index+=1
            s.posPrev=copy(s.pos)
            s.pos=copy(s.posNew)
            s.showParticles(s.pos)
            speeds=[norm(x) for x in s.vel]
            
            counts,bins=histogram(speeds,density=True)
            countAvg+=counts
            
            if index == 20:
                #print("plot")
                countList.append(countAvg/20)
                binList.append((bins[1:]+bins[:-1])/2)
                countAvg=0
                lineNumber+=1
                index=0
        #print(binList,countList)
        return(binList,countList)
        
    def compareInitialVel(s):
        xPercent=linspace(0,1,9)
        z=0
        fig,axs=plt.subplots(3,3)
        for i in range(3):
            for j in range(3):
                title='percent x: ' + str(xPercent[z])
                binList,countList=s.simulate(xPercent[z])
                axs[i,j].set_title(title)
                for k in range(len(binList)):
                    axs[i,j].plot(binList[k],countList[k])
                z+=1
        plt.show()
                
                
#Parameters: Number Particles, Box length, lattice vctors and run time
partyBox=MD()
partyBox.simulate()
partyBox.compareInitialVel()
