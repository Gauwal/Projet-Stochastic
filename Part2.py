# -*- coding: utf-8 -*-
"""
Created on Sat May  8 14:27:27 2021

@author: Arthur Gengler
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import math 




#Lecture de u 
file1 = open('Input_linear_case.txt', "r")
lines = file1.readlines()
i=0
u=np.zeros(len(lines)-1)
for line in lines:
    if(i==0):
        pass
    else:
        u[i-1]=float(line.strip().split(" ")[0])
    i+=1
file1.close()

#Lecture de y
file2 = open('Measured_output_linear_case.txt', "r")
lines = file2.readlines()
i=0
y=np.zeros(len(lines)-1)
for line in lines:
    if(i==0):
        pass
    else:
        y[i-1]=float(line.strip().split(" ")[0])
    i+=1
file2.close()

#Matrice of Kalman filter
A=[[0.9512,0],[0.0476,0.9512]]
B=[0.0975,0.0024]
C=[0,1]


Q=[[10**(-3)*9.506,10**(-3)*0.0234],[10**(-3)*0.234,10**(-3)*9.512]]
R=0.0125

#x(-1)
mu_x_init=[5,5]
Cov_x_init=[[1,0],[0,1]]

Inx=[[1,0],[0,1]]

#titde (µ_k )= Aµk−1 + Buk
def mu_tilde(A,B,mukprev,ucurrent):
    return np.dot(A,mukprev)+np.dot(B,ucurrent)

#tilde Pk = APk−1 AT + Q
def P_tilde(A,Q,Pprev):
    return np.dot(np.dot(A,Pprev),np.transpose(A))+Q

#Kk = P˜kCT(CP˜kCT + R)−1
def K_k(Ptilde,C,R):
    abc=1/(np.dot(np.dot(C,Ptilde),np.transpose(C))+R)
    return np.dot(Ptilde,np.transpose(C))*abc

#Calcul mu_k
def mu_k(mutilde,Kk,yk,C):
    return mutilde+Kk*(yk-np.dot(C,mutilde))

#Pk = (Inx − KkC)P˜k
def P_k(Inx,Kk,C,Ptilde):
    return np.dot(Inx-Kk*C,Ptilde)

k=len(y)
x1=np.zeros(len(y))
x2=np.zeros(len(y))

for i in range(0,k):
    ucurrent=u[i]
    yk=y[i]
    if i==0:
        mukprev=mu_x_init
        Pprev=Cov_x_init
        mutilde=mu_tilde(A,B,mukprev,ucurrent)
        Ptilde=P_tilde(A,Q,Pprev)
        Kk=K_k(Ptilde,C,R)
        muk=mu_k(mutilde,Kk,yk,C)
        Pk=P_k(Inx,Kk,C,Ptilde)
        x1[i]=np.random.normal(muk[0],Pk[0][0])
    else:
        mukprev=muk
        Pprev=Pk
        mutilde=mu_tilde(A,B,mukprev,ucurrent)
        Ptilde=P_tilde(A,Q,Pprev)
        Kk=K_k(Ptilde,C,R)
        muk=mu_k(mutilde,Kk,yk,C)
        Pk=P_k(Inx,Kk,C,Ptilde)
        x1[i]=np.random.normal(muk[0],Pk[0][0])
    print(x1[2])

        
      
        
        
        
        
        
        