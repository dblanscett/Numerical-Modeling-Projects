# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 13:24:29 2021

@author: Devan
This code models a three body planetary system with the Sun, Jupiter and Earth 
with the infastructure set up to allow addition of a user menu and the addition
of additional planets, allowing for a n body system. In this code we explore
the effects of initial conditions and investigate the affect of changing 
Jupiter's mass.
"""

from numpy import array,zeros, zeros_like, pi, sqrt, copy
from numpy.linalg import norm
import matplotlib.pyplot as plt

class planets:
    def __init__(s,nPlanets,planetNames,dt,runtime,axisLim,jMassMult=1):
        s.r = zeros((nPlanets,2)) #distances in AU, initially at perihelion
        s.vel = zeros_like(s.r) #AU/yr
        s.masses = zeros(nPlanets) #in kg
        s.pointColors = [] #holds the colors of the planets on the graph
        s.pointSize = [] #holds the size of the points on the graph
        s.eccentricity = zeros(nPlanets) #used to find initial V
        s.semiMajorAxis = zeros(nPlanets) #also used to find initial V
        s.nPlanets = nPlanets #number of planets in this system
        s.planetNames = planetNames #which planets were passed in
        s.dt = dt #years
        s.maxT = runtime #also in years
        s.jupiterMassMult = jMassMult #used to alt jupiter's mass
        s.lineR = zeros([int(runtime/dt),nPlanets*2]) #holds vars for line plot
        s.axisLim = axisLim #sets plot size
        s.initialize() #fills lists, adjust r for CM, find start V
        #alt initial conditions are set by function fall after initialization
        
    def initialize(s):
        scalePoint = 10 #marker size for sun, all planets scaled accordingly
        s.sunIndex = 0 #used to pull out Sun's info from arrays (for M_s etc.)
        
        #Loop to assign perihelion distances for initial positions, point color
        #for the plot, size of the point, planet mass, eccentricity, and semi 
        #major axis (or a)
        for i in range(s.nPlanets):
            if planetNames[i]=='Sun':
                s.r[i] = [0,0] 
                s.pointColors.append('yo')
                s.pointSize.append(scalePoint)
                s.masses[i] = 1.989e30
                s.sunIndex=i #so we can pull sun values out later
                
            elif planetNames[i]=='Earth':
                s.r[i] = [0.98,0] #perihelion distance
                s.pointColors.append('bo')
                s.pointSize.append(int(scalePoint*0.3))
                s.masses[i] = 6.0e24
                s.eccentricity[i] = 0.017
                s.semiMajorAxis[i] = 1.00
                
            elif planetNames[i]=='Jupiter':
                s.r[i] = [4.95,0] #perihelion distance
                s.pointColors.append('ro')
                s.pointSize.append(int(scalePoint*0.75))
                s.masses[i] = 1.9e27 * s.jupiterMassMult
                s.eccentricity[i] = 0.048
                s.semiMajorAxis[i] = 5.20
               
        CM = sum(s.r[:,0]*s.masses)/sum(s.masses)
        s.r -= [CM,0] #sets CM as origin
        
        #loop to set initial velocites
        for i in range(len(s.vel)):
            if s.planetNames[i] == 'Sun': #calculated outside of loop
                continue
            else:
                #eq for vel has prefactor of sqrt(G*M_s) which = 2pi in AU
                #velocities found using equation 4.11 for vmax from textbook
                #we're using vmax which corresonds to rmin (perihelion)
                eccen = (1+s.eccentricity[i])/(s.semiMajorAxis[i]*(1-s.eccentricity[i]))
                mRatio = s.masses[i]/s.masses[s.sunIndex]
                s.vel[i] = [0,2*pi*(sqrt(eccen*(1+mRatio)))]
        velSun = sum(s.vel[:,1]*s.masses)/s.masses[s.sunIndex]
        s.vel[s.sunIndex] = [0,-velSun] #if total momentum = 0 CM doesn't move
        #print(sum(s.masses*s.vel[:,1]))
        #print(s.r)
    
    def altInitialize(s,r,v):
        #This function allows us to test different initial conditions
        s.r = r
        s.vel = v
        
    def plotSys(s):
        for i,k in enumerate(s.r):
            plt.plot(k[0],k[1],s.pointColors[i],ms=s.pointSize[i],
                     label=s.planetNames[i])
        plt.legend()
        plt.ylim(-s.axisLim,s.axisLim)
        plt.ylabel('y (AU)')
        plt.xlim(-s.axisLim,s.axisLim)
        plt.xlabel('x (AU)')
        plt.draw()
        plt.pause(1e-2)
        plt.clf()
    
    def linePlot(s, plotTitle):
        plt.plot(s.lineR[:,0],s.lineR[:,1],'y-',label='Sun')
        plt.plot(s.lineR[:,2],s.lineR[:,3],'b-',label='Earth')
        plt.plot(s.lineR[:,4],s.lineR[:,5],'r-',label='Jupiter')
        plt.legend()
        plt.title(plotTitle)
        plt.ylim(-s.axisLim,s.axisLim)
        plt.ylabel('y (AU)')
        plt.xlim(-s.axisLim,s.axisLim)
        plt.xlabel('x (AU)')
        plt.show()
    
    def derivs(s,vari):
        deriv=zeros_like(vari)
        deriv[:,:2]=vari[:,2:]
        forces = zeros_like(vari[:,2:])
        prefactor = -(4*pi**2)/s.masses[s.sunIndex] #force prefactor
        for i in range(len(forces)):
            forceOnI = zeros(2)
            for j in range(len(forces)):
                if i == j:
                    continue
                else:
                    distBetween = (vari[i,:2]-vari[j,:2])
                    dBetMag = norm((vari[j,:2]-vari[i,:2]))
                    forceOnI += (s.masses[j]/(dBetMag**3))*distBetween
            forces[i] = prefactor*forceOnI
        deriv[:,2:]=forces
        return(deriv)
    
    def rungeKutta(s):
        pos = copy(s.r) #copying arrays to avoid overwriting too soon
        vel = copy(s.vel)
        
        variRows = len(pos)
        variCol = len(pos[0])+len(vel[0])
        vari = zeros([variRows,variCol])
        
        for i in range(len(vari)):
            vari[i]=[pos[i,0],pos[i,1],vel[i,0],vel[i,1]]
        
        k1 = s.dt*s.derivs(vari)
        k2 = s.dt*s.derivs(vari+0.5*k1)
        k3 = s.dt*s.derivs(vari+0.5*k2)
        k4 = s.dt*s.derivs(vari+k3)
        
        vari += (1/6)*(k1+2*k2+2*k3+k4)
        s.r = vari[:,:2]
        s.vel = vari[:,2:]
        
    def simulate(s,showPlot):
        #s.plotSys()
        t = 0
        counter=0
        while t < s.maxT-s.dt:
            s.rungeKutta() #update system
            #plot all points for line graph as gen sum of system at end
            s.lineR[counter]=[s.r[0,0],s.r[0,1],s.r[1,0],s.r[1,1],s.r[2,0],s.r[2,1]]
            #used to print initial conditions used later
            #if round(t,2) % 0.25 == 0 and t < 1.1:
            #    print('t',t)
            #    print('r',s.r)
            #    print('vel',s.vel)
            #if we're plotting in real time we only plot every 100 steps to 
            #reduce runtime/compuational burden
            if counter%100 == 0 and showPlot:
                s.plotSys()
                #s.linePlot()
            t += s.dt
            counter += 1
        
#if time set up user input for which planets to add
nPlanets=3 #number of planets
planetNames = ['Sun','Earth','Jupiter']
dt=0.001 #years
runtime = 12
axisLim = 10
showPlot = 0#boolean, I got tired of typing True/False
jMassMult = [1,10,100,1000] #allows us to scale Jupiter's mass

#alt init conditions pulled from running with dt = 0.001, runtime = 12, 3 
#planets, jMassMult = 1 at t=0.250, 0.500, 0.750 respectively
altR = array([[[-4.675e-3,-6.9413e-4],[-5.285e-02, 9.933e-01],
               [4.895e+00, 7.235e-01]],
              [[-4.529e-03, -1.365e-03],[-1.013e+00, -3.225e-02],
               [4.745e+00,  1.429e+00]],
              [[-4.298e-03, -2.009e-03], [1.025e-02, -9.9626e-01],
               [ 4.499e+00,  2.106e+00]]])
altV = array([[[4.038e-04, -2.7349e-03],[-6.293e+00, -2.147e-01],
               [-4.028e-01,  2.8637e+00]],
              [[7.585e-04, -2.632e-03], [1.936e-01, -6.207e+00], 
               [-7.9476e-01,  2.775e+00]],
              [[ 1.097e-03, -2.515e-03], [6.301e+00, 1.826e-01],
               [-1.168e+00,  2.632e+00]]])

#plotTitle='Ordinary System'
#SolarSystem = planets(nPlanets,planetNames,dt,runtime,axisLim,jMassMult[0])
#SolarSystem.simulate(showPlot)
#SolarSystem.linePlot(plotTitle)

#testing different initial conditions
#startTimes = array([0.250,0.500,0.750]) #times we pulled start conditions from
#for i in startTimes:
#    plotTitle = 'Starting from ' + str(i) + '% of Earths 1st orbit'
#    testInit = planets(nPlanets,planetNames,dt,runtime,axisLim,jMassMult[0])
#    testInit.simulate(showPlot)
#    testInit.linePlot(plotTitle)

#testing different Jupiter Masses
#for i in range(3): #we've already used the first
#    plotTitle = 'Jupiters Mass at ' + str(jMassMult[i+1]) + '* original value'
#    testInit = planets(nPlanets,planetNames,dt,runtime,axisLim,jMassMult[i+1])
#    testInit.simulate(showPlot)
#    testInit.linePlot(plotTitle) 

#ah. Right. Starting at different positions from original system aren't 
#different initial conditons, especially for stable orbits

#Original initial conditions which we'll alter
altR = array([[-4.72693337e-03,  0.00000000e+00],
              [ 9.75273067e-01,  0.00000000e+00],
              [ 4.94527307e+00,  0.00000000e+00]])
altV = array([[ 0.00000000e+00, -2.78218740e-03],
              [ 0.00000000e+00,  6.39093265e+00],
              [ 0.00000000e+00,  2.89232902e+00]])

#plotTitle = 'Earth Sun Jupiter'
#fixTest = planets(nPlanets,planetNames,dt,runtime,axisLim,jMassMult[0])
#altR[1] *= -1 #flip the earth to the other side of the Sun
#fixTest.altInitialize(altR, altV)
#fixTest.simulate(showPlot)
#fixTest.linePlot(plotTitle)

#plotTitle = 'Jupiter Sun Earth'
#fixTest = planets(nPlanets,planetNames,dt,runtime,axisLim,jMassMult[0])
#altR[2] *= -1 #flip jupiter to the other side of the Sun
#fixTest.altInitialize(altR, altV)
#fixTest.simulate(showPlot)
#fixTest.linePlot(plotTitle)

#plotTitle = 'Flip Earth Velocity'
#fixTest = planets(nPlanets,planetNames,dt,runtime,axisLim,jMassMult[0])
#altV[1] *= -1 #Start Earth going the other way
#fixTest.altInitialize(altR, altV)
#fixTest.simulate(showPlot)
#fixTest.linePlot(plotTitle)

plotTitle = 'Flip Jupiter Velocity'
fixTest = planets(nPlanets,planetNames,dt,runtime,axisLim,jMassMult[0])
altV[2] *= -1 #Start Earth going the other way
fixTest.altInitialize(altR, altV)
fixTest.simulate(showPlot)
fixTest.linePlot(plotTitle)
