"""
Created on Thu Feb 11 14:35:27 2021

@author: Devan

Class: Numerical Modeling
Problem: 7.9 from Nicholas J. Giordano's "Computational Physics" Second edition

Description:
    This code shows the diffusion equation (7.21) in action as a highly peaked
    density plot in 1d spreads out over time. It also compares the time 
    dependent diffusion against eq (7.22) which the book claims is a solution
    to our diffusion equation
     
"""

from numpy import linspace, exp, zeros_like, copy, sqrt
from numpy.random import random
import matplotlib.pyplot as plt

class CoffeeCreamer: #One cube or two? Come diffuse your stress 
    def __init__(s, length=100, dx=1, dt=0.5, D=5):
        #parameters describing the systems
        s.length=length
        s.dx=dx
        s.steps=int(length/s.dx)
        s.x=linspace(-int(s.length),int(s.length),s.steps)
        s.D=D #affects the speed at which the wave disperses
        if dt <= dx**2/(2*s.D):
            s.dt=dt
        else:
            s.dt=random()*(dx**2/(2*s.D))#prevent instability if user passes
                                         #dt too large
        
        s.rho=zeros_like(s.x)#for the diffusion eqatuion (7.21)
        s.rhoNext=copy(s.rho)
        
        s.checkSol=(zeros_like(s.x))#to check the proposed solution
        s.checkSolNext=copy(s.checkSol) 
        
        
    
    def initialize(s):
        #parameters for gaussian "pluck" on our string
        s.t=0
        x=copy(s.x)
        s.standDev = .2 #1/m^2 
        s.midPoint = 0
        s.height=1
        s.rho=s.height*exp(-(x-s.midPoint)**2/(2*s.standDev**2))
        s.checkSol=copy(s.rho)
                
        plt.plot(s.x,s.rho)
        plt.plot(s.x,s.checkSol)
        plt.show()
        print(s.rho)

    def updateY(s,y):
        y1=copy(y)
        y1[1:-1]=y[1:-1]+(s.D*s.dt)/(s.dx**2)*(y[2:]+y[:-2]-2*y[1:-1])
        return y1
    
    def checkSolCalc(s,t):
        sigma=sqrt(2*s.D*t)
        y1=(1/sigma)*exp(-(s.x**2/(2*sigma**2)))
        return y1
        
    def movie(s):
        s.initialize()
        
        while s.t<10.5:
        #for i in range(20):
            #print(t)
            s.rho=s.updateY(s.rho)
            
            #movie time
            plt.subplot(211)
            plt.plot(s.x,s.rho,'g-',label="density plot")
            plt.legend()
            plt.subplot(212)
            plt.ylim(0,1)
            plt.plot(s.x,s.checkSolCalc(s.t),'b-',label="solution check")
            plt.legend()
            plt.draw()
            plt.pause(0.0001)
            plt.clf()
            
            s.t+=s.dt
        plt.subplot(211)
        plt.plot(s.x,s.rho,'g-',label="density plot") 
        plt.subplot(212)
        plt.plot(s.x,s.checkSolCalc(s.t),'b-',label="solution check")
        plt.legend()
        plt.show()

densityProfile = CoffeeCreamer()
densityProfile.initialize()
densityProfile.movie()
