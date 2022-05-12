# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 14:35:27 2021

@author: Devan

Class: Numerical Modeling
Problem: 6.1 from Nicholas J. Giordano's "Computational Physics" Second edition

Description:
    This code plots two gaussian "plucks", or wave packets, on two different 
    strings. One string has the ends fixed and the other has the ends free. 
    It should be observed that the string that has freedom in the end will
    refuse to invert the wave pack, preferring to maintain a positive outlook
    on life. Confining a wave packet to a string that is imprisoned on either 
    end results in inverted smiles the like of which no primary song can change,
    only time will do that. 
"""

from numpy import linspace, exp, zeros_like, copy
import matplotlib.pyplot as plt

class Iroh: #Go with the flow 
    def __init__(s, length=1, dx=0.01, c=300):
        #parameters describing the systems
        s.length=length
        s.dx=dx
        s.steps=int(length/s.dx)
        #print(s.steps)
        s.x=linspace(0,s.length,s.steps)
        s.y1=zeros_like(s.x)
        s.y2=copy(s.y1) #two empty arrays for the time steps
        s.y3=copy(s.y1)
        
        #bound string
        s.yb1=copy(s.y1)
        s.yb2=copy(s.y2)
        s.yb3=copy(s.y3)
        
        #constants to help calc next time y
        s.c=c
        s.dt=s.dx/c
        s.r=s.c*(s.dt/s.dx)
        #print(s.r)
    
    def initialize(s):
        #paramets for gaussian "pluck" on our string
        x=copy(s.x)
        s.k = 1000 #1/m^2 
        s.x0 = 0.3
        s.y2=exp(-s.k*(x-s.x0)**2)
        s.y1=copy(s.y2)
        
        #bound states
        s.yb2=exp(-s.k*(x-s.x0)**2)
        s.yb1=copy(s.yb2)
        
        #plt.plot(s.x,s.y2)
        #plt.plot(s.x,s.yb2)
        #plt.show()
        

    def updateY(s,y0,y1):
        yn0=copy(y0)
        yn=copy(y1)
        yn1=copy(yn) #just for the size
        yn1[1:-1]=2*(1-s.r**2)*yn[1:-1]-yn0[1:-1]+s.r**2*(yn[2:]+yn[:-2])
        return yn1
        
    def movie(s):
        s.initialize()#we have our first and now our second (with the pluck)
        t=0
        #this will run for 
        while t<0.007:
            #print(t)
            s.y3=s.updateY(s.y1,s.y2)
            #update ends for free string
            s.y3[0]=s.y3[1]
            s.y3[-1]=s.y3[-2]
            
            #update values to prep for next propagation
            s.y1=copy(s.y2)
            s.y2=copy(s.y3)
            
            #repeat for bound plots
            s.yb3=s.updateY(s.yb1,s.yb2)
            s.yb1=copy(s.yb2)
            s.yb2=copy(s.yb3)
            
            #movie time
            plt.subplot(211)
            plt.plot(s.x,s.y3,'g-',label='free')
            plt.legend()
            plt.ylim(-1.1,1.1)
            plt.subplot(212)
            plt.plot(s.x,s.yb3,label='bound')
            plt.legend()
            plt.ylim(-1.1,1.1)
            plt.draw()
            plt.pause(0.0001)
            plt.clf()
            
            t+=s.dt
        
        

string = Iroh()
string.movie()