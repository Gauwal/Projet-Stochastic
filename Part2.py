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

#x1_true
file3 = open('True_state_x1_linear_case.txt', "r")
lines = file3.readlines()
i=0
x1_true=np.zeros(len(lines)-1)
for line in lines:
    if(i==0):
        pass
    else:
        x1_true[i-1]=float(line.strip().split(" ")[0])
    i+=1
file3.close()

#x2_true
file4 = open('True_state_x1_linear_case.txt', "r")
lines = file4.readlines()
i=0
x2_true=np.zeros(len(lines)-1)
for line in lines:
    if(i==0):
        pass
    else:
        x2_true[i-1]=float(line.strip().split(" ")[0])
    i+=1
file4.close()

#x1_true_nonlin
file4 = open('True_state_x1_nonlinear_case.txt', "r")
lines = file4.readlines()
i=0
x1_true_nl=np.zeros(len(lines)-1)
for line in lines:
    if(i==0):
        pass
    else:
        x1_true_nl[i-1]=float(line.strip().split(" ")[0])
    i+=1
file4.close()

#x2_true_nonlin
file5 = open('True_state_x2_nonlinear_case.txt', "r")
lines = file5.readlines()
i=0
x2_true_nl=np.zeros(len(lines)-1)
for line in lines:
    if(i==0):
        pass
    else:
        x2_true_nl[i-1]=float(line.strip().split(" ")[0])
    i+=1
file5.close()

#Input nl
file6 = open('Input_nonlinear_case.txt', "r")
lines = file6.readlines()
i=0
u_nl=np.zeros(len(lines)-1)
for line in lines:
    if(i==0):
        pass
    else:
        u_nl[i-1]=float(line.strip().split(" ")[0])
    i+=1
file6.close()

#Lecture de y_nl
file7 = open('Measured_output_nonlinear_case.txt', "r")
lines = file7.readlines()
i=0
y_nl=np.zeros(len(lines)-1)
for line in lines:
    if(i==0):
        pass
    else:
        y_nl[i-1]=float(line.strip().split(" ")[0])
    i+=1
file7.close()

#Sampling Period
h=0.1



#Matrice of Kalman filter
A=np.array([[0.9512,0],[0.0476,0.9512]])
A.shape=(2,2)
B=np.array([0.0975,0.0024])
B.shape=(2,1)
C=np.array([0,1])
C.shape=(1,2)

Q=np.array([[10**(-3)*9.506,10**(-3)*0.0234],[10**(-3)*0.234,10**(-3)*9.512]])
Q.shape=(2,2)
R=np.array(0.0125)

#x(-1)
mu_x_init=np.array([5,5])
mu_x_init.shape=(2,1)
Cov_x_init=np.array([[1,0],[0,1]])
Cov_x_init.shape=(2,2)
Inx=np.array([[1,0],[0,1]])
Inx.shape=(2,2)

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
 #   return np.dot(C,mutilde)
    return mutilde+Kk*(yk-np.dot(C,mutilde))

#Pk = (Inx − KkC)P˜k
def P_k(Inx,Kk,C,Ptilde):
    return np.dot(Inx-np.dot(Kk,C),Ptilde)

k=len(y)
x1=np.zeros(len(y))
x2=np.zeros(len(y))

x1sigmaplus=np.zeros(len(y))
x1sigmamoins=np.zeros(len(y))
x2sigmaplus=np.zeros(len(y))
x2sigmamoins=np.zeros(len(y))
x1rmds=np.zeros(len(y))
x2rmds=np.zeros(len(y))

for i in range(0,k):
    ucurrent=u[i]
    yk=y[i]
    if i==0:
        mukprev=mu_x_init
        Pprev=Cov_x_init
        mutilde=mu_tilde(A,B,mukprev,ucurrent)
        Ptilde=P_tilde(A,Q,Pprev)
        Kk=K_k(Ptilde,C,R)
        #print("Kk",Kk)
        #print("Azer",(yk-np.dot(C,mutilde)))
        #print("Kk",Kk*(yk-np.dot(C,mutilde)))
        muk=mu_k(mutilde,Kk,yk,C)
        Pk=P_k(Inx,Kk,C,Ptilde)
        x1[i]=np.random.normal(muk[0],Pk[0][0])
        
        x2[i]=np.random.normal(muk[1],Pk[1][1])
        
    else:
        mukprev=muk
        Pprev=Pk
        mutilde=mu_tilde(A,B,mukprev,ucurrent)
        Ptilde=P_tilde(A,Q,Pprev)
        Kk=K_k(Ptilde,C,R)
        muk=mu_k(mutilde,Kk,yk,C)
        Pk=P_k(Inx,Kk,C,Ptilde)
        x1[i]=np.random.normal(muk[0],Pk[0][0])
        x2[i]=np.random.normal(muk[1],Pk[1][1])
    x1sigmaplus[i]=x1[i]+1.96*(Pk[0][0])**(1/2)
    x1sigmamoins[i]=x1[i]-1.96*(Pk[0][0])**(1/2)
    x2sigmaplus[i]=x2[i]+1.96*(Pk[1][1])**(1/2)
    x2sigmamoins[i]=x2[i]-1.96*(Pk[1][1])**(1/2)
    x1rmds[i]=((x1[i]-x1_true[i])**(2))**(1/2)
    x2rmds[i]=((x2[i]-x2_true[i])**(2))**(1/2)



#a=True
a=False

Linewdth = 1
fig, axis = plt.subplots(1, 4,figsize=(20, 5))
axis[0].plot(np.arange(0,1000,1)*(h),x1sigmaplus,color="lightsteelblue", linewidth=Linewdth)
axis[0].plot(np.arange(0,1000,1)*(h),x1sigmamoins,color="lightsteelblue", linewidth=Linewdth)
axis[0].fill_between(np.arange(0,1000,1)*(h),y1=x1sigmaplus, y2=x1sigmamoins,color="lightsteelblue",label="95% CI")
axis[0].plot(np.arange(0,1000,1)*(h),x1_true,color="royalblue",label="True value", linewidth=Linewdth)
axis[0].plot(np.arange(0,1000,1)*(h),x1,color="blue",label="Filtered Data", linewidth=Linewdth)
axis[0].legend()
axis[0].set_aspect(1.0/axis[0].get_data_ratio(), adjustable='box')
axis[0].set_xlabel('Time [s]')
axis[0].set_ylabel('Height [m]')
axis[0].set_title("X1 Kf")
  


axis[1].plot(np.arange(0,1000,1)*h,x2sigmaplus,color="pink", linewidth=Linewdth)
axis[1].plot(np.arange(0,1000,1)*h,x2sigmamoins,color="pink", linewidth=Linewdth)
axis[1].fill_between(np.arange(0,1000,1)*h,y1=x2sigmaplus, y2=x2sigmamoins,color="pink",label="95% CI")
axis[1].plot(np.arange(0,1000,1)*h,x2_true,color="indianred",label="True value", linewidth=Linewdth)
axis[1].plot(np.arange(0,1000,1)*h,x2,color="red",label="Filtered Data", linewidth=Linewdth)
axis[1].legend()
axis[1].set_aspect(1.0/axis[1].get_data_ratio(), adjustable='box')
axis[1].set_xlabel('Time [s]')
axis[1].set_ylabel('Height [m]')
axis[1].set_title("X2 Kf")
  

axis[2].plot(x1rmds,color="blue",label="RMSD x1", linewidth=Linewdth)
axis[2].plot(x2rmds,color="red",label="RMSD x2", linewidth=Linewdth)
axis[2].legend()
axis[2].set_aspect(1.0/axis[2].get_data_ratio(), adjustable='box')
axis[2].set_xlabel('Time [s]')
axis[2].set_ylabel('RMSD [m]')
axis[2].set_title("RMSD")


axis[3].plot(np.arange(0,1000,1)*(h),x1sigmaplus-x1sigmamoins,color="blue", linewidth=Linewdth, label="95% Ci of x1")
axis[3].plot(np.arange(0,1000,1)*h,x2sigmaplus-x2sigmamoins,color="red", linewidth=Linewdth, label="95% Ci of x1")
axis[3].legend()
axis[3].set_aspect(1.0/axis[3].get_data_ratio(), adjustable='box')
axis[3].set_xlabel('Time [s]')
axis[3].set_ylabel('Width [m]')
axis[3].set_title("95% Confidence Intervals")


plt.show()



###------EnKF------###


def sample_mean(zer):
    sum=0
    for i in range(len(zer)):
        sum=sum+zer[i]
    return sum/len(zer)

#faut Rajouter le bruit
def x_tilde(A,B,ucurrent,x1_hat_i_kprev,x2_hat_i_kprev):
    return A@[x1_hat_i_kprev,x2_hat_i_kprev]+B@ucurrent

#faut Rajouter le bruit
def y_tilde(C,xtilde):
    return C@xtilde

def K_k_approx(Papprox,C,R):
    abc=1/(np.dot(np.dot(C,Papprox),np.transpose(C))+R)
    return np.dot(Papprox,np.transpose(C))*abc     
    
    

N=30
x1_hat_prev=np.array(np.zeros((N,1)))
x2_hat_prev=np.array(np.zeros((N,1)))
xtilde=np.array(np.zeros((N,2,1)))
ytilde=np.array(np.zeros((N,1)))
x_hat=np.array(np.zeros((N,2,1)))
x1_hat=np.zeros(len(y))
x2_hat=np.zeros(len(y))

x1rmdsmoy=np.zeros(k)
x2rmdsmoy=np.zeros(k)
for j in range(0,k):
    ucurrent=u[j]
    yk=y[j]
    for i in range(0,N):

        if j==0:
            x1_hat_prev[i][0]=np.random.normal(mu_x_init[0],Cov_x_init[0][0])
            x2_hat_prev[i][0]=np.random.normal(mu_x_init[1],Cov_x_init[1][1])
        else:
            x1_hat_prev[i][0]=x_hat[i][0][0]
            x2_hat_prev[i][0]=x_hat[i][1][0]
    
        x_hat_prev=np.array([[x1_hat_prev[i][0]],[x2_hat_prev[i][0]]])
        x_hat_prev.shape=(2,1)
        #print(xtilde[:][:][i].shape)
        w1=np.random.normal(0,Q[0][0])
        w2=np.random.normal(0,Q[1][1])
        
        w=[[np.random.normal(0,Q[0][0])],[np.random.normal(0,Q[0][0])]]
        v=np.random.normal(0,0.0125)
        xtilde[i][:][:]=A@x_hat_prev+B*ucurrent+w 
        ytilde[i][:]=C@xtilde[i][:][:]+v
        #print("xtilde",xtilde[i][:][:],"i",i)
        #print("C",C.shape)
        #print(ytilde[i][:].shape)
        
        
    #N-2 because our N-1 = N of the formula

    covxtilde12=(1/(N-2))*np.sum([(xtilde[n][0][0]-np.mean([(xtilde[m][0][0]) for m in range(0,N)]))*(xtilde[n][1][0]-np.mean([(xtilde[m][1][0]) for m in range(0,N)])) for n in range(0,N)])
    varxtilde1=(1/(N-2))*np.sum([(xtilde[n][0][0]-np.mean([(xtilde[m][0][0]) for m in range(0,N)]))*(xtilde[n][0][0]-np.mean([(xtilde[m][0][0]) for m in range(0,N)])) for n in range(0,N)])
    varxtilde2=(1/(N-2))*np.sum([(xtilde[n][1][0]-np.mean([(xtilde[m][1][0]) for m in range(0,N)]))*(xtilde[n][1][0]-np.mean([(xtilde[m][1][0]) for m in range(0,N)])) for n in range(0,N)])
    
    Ptildeapprox=np.array([varxtilde1,covxtilde12,covxtilde12,varxtilde2]).reshape((2,2))
   # print(Ptildeapprox)
    #print(xtilde[1][0][0]-np.mean([(xtilde[m][0][0]) for m in range(0,N)]))
    
    #print(Ptildeapprox)

    Kkapprox=Ptildeapprox@C.T*(1/(C@Ptildeapprox@C.T+R))
    for i in range(0,N):
        x_hat[i][:][:]=xtilde[i][:][:]+Kkapprox*(yk-ytilde[i][:])
        #print("hat",x_hat[i][:][:],"i",i)
    x1_hat[j]=np.mean([(x_hat[i][0][0]) for i in range(0,N)])
    x2_hat[j]=np.mean([(x_hat[i][1][0]) for i in range(0,N)])

    x1rmds=np.zeros(N)
    for i in range(0,N):
        x1rmds[i]=(x_hat[i][0][0]-x1_true[j])
    x1rmdsmoy[j]=np.mean([(x1rmds[i]) for i in range(0,N)])
    x2rmds=np.zeros(N)
    for i in range(0,N):
        x2rmds[i]=(x_hat[i][1][0]-x2_true[j])
    x2rmdsmoy[j]=np.mean([(x2rmds[i]) for i in range(0,N)])

    x1sigmaplus[j]=x1_hat[j]+1.96*(Ptildeapprox[0][0])**(1/2)
    x1sigmamoins[j]=x1_hat[j]-1.96*(Ptildeapprox[0][0])**(1/2)
    x2sigmaplus[j]=x2_hat[j]+1.96*(Ptildeapprox[1][1])**(1/2)
    x2sigmamoins[j]=x2_hat[j]-1.96*(Ptildeapprox[1][1])**(1/2)






Linewdth = 1
fig, axis = plt.subplots(1, 4,figsize=(20, 5))
#axis[0].plot(np.arange(0,1000,1)*(h),x1sigmaplus,color="lightsteelblue", linewidth=Linewdth)
#axis[0].plot(np.arange(0,1000,1)*(h),x1sigmamoins,color="lightsteelblue", linewidth=Linewdth)
#axis[0].fill_between(np.arange(0,1000,1)*(h),y1=x1sigmaplus, y2=x1sigmamoins,color="lightsteelblue",label="95% CI")
axis[0].plot(np.arange(0,1000,1)*(h),x1_true,color="royalblue",label="True value", linewidth=Linewdth)
axis[0].plot(np.arange(0,1000,1)*(h),x1,color="blue",label="Filtered Data", linewidth=Linewdth)
axis[0].legend()
axis[0].set_aspect(1.0/axis[0].get_data_ratio(), adjustable='box')
axis[0].set_xlabel('Time [s]')
axis[0].set_ylabel('Height [m]')
axis[0].set_title("X1 linear EnKf")
  


#axis[1].plot(np.arange(0,1000,1)*h,x2sigmaplus,color="pink", linewidth=Linewdth)
#axis[1].plot(np.arange(0,1000,1)*h,x2sigmamoins,color="pink", linewidth=Linewdth)
#axis[1].fill_between(np.arange(0,1000,1)*h,y1=x2sigmaplus, y2=x2sigmamoins,color="pink",label="95% CI")
axis[1].plot(np.arange(0,1000,1)*h,x2_true,color="indianred",label="True value", linewidth=Linewdth)
axis[1].plot(np.arange(0,1000,1)*h,x2,color="red",label="Filtered Data", linewidth=Linewdth)
axis[1].legend()
axis[1].set_aspect(1.0/axis[1].get_data_ratio(), adjustable='box')
axis[1].set_xlabel('Time [s]')
axis[1].set_ylabel('Height [m]')
axis[1].set_title("X2 linear EnKf")
  

axis[2].plot(np.arange(0,1000,1)*h,x1rmdsmoy,color="blue",label="RMSD x1", linewidth=Linewdth)
axis[2].plot(np.arange(0,1000,1)*h,x2rmdsmoy,color="red",label="RMSD x2", linewidth=Linewdth)
axis[2].legend()
axis[2].set_aspect(1.0/axis[2].get_data_ratio(), adjustable='box')
axis[2].set_xlabel('Time [s]')
axis[2].set_ylabel('RMSD [m]')
axis[2].set_title("RMSD")
  

axis[3].plot(np.arange(0,1000,1)*(h),x1sigmaplus-x1sigmamoins,color="blue", linewidth=Linewdth, label="95% Ci of x1")
axis[3].plot(np.arange(0,1000,1)*h,x2sigmaplus-x2sigmamoins,color="red", linewidth=Linewdth, label="95% Ci of x1")
axis[3].legend()
axis[3].set_aspect(1.0/axis[3].get_data_ratio(), adjustable='box')
axis[3].set_xlabel('Time [s]')
axis[3].set_ylabel('Width [m]')
axis[3].set_title("95% Confidence Intervals")


plt.show()

###-------------2.2


x1_true=x1_true_nl
x2_true=x2_true_nl
y=y_nl
u=u_nl

def f(xk,uk,ind):
    if ind == 1:
        return np.array([-math.sqrt(abs(xk[0][0]))+uk,math.sqrt(abs(xk[0][0]))-math.sqrt(abs(xk[1][0]))]).reshape((2,1))
    elif ind == 4:
        return f(xk+(h)*f(xk,uk,ind-1),uk,1)
    else:
        return f(xk+(h/2)*f(xk,uk,ind-1),uk,1)

uprev=u[0]


x1rmdsmoy=np.zeros(k)
x2rmdsmoy=np.zeros(k)
for j in range(0,k):
    if j>0 :
        uprev=u[j-1]
    yk=y[j]
    for i in range(0,N):

        if j==0:
            x1_hat_prev[i][0]=np.random.normal(mu_x_init[0],Cov_x_init[0][0])
            x2_hat_prev[i][0]=np.random.normal(mu_x_init[1],Cov_x_init[1][1])
        else:
            x1_hat_prev[i][0]=x_hat[i][0][0]
            x2_hat_prev[i][0]=x_hat[i][1][0]
    
        x_hat_prev=np.array([[x1_hat_prev[i][0]],[x2_hat_prev[i][0]]])
        x_hat_prev.shape=(2,1)
        #print(xtilde[:][:][i].shape)
        w1=np.random.normal(0,Q[0][0])
        w2=np.random.normal(0,Q[1][1])
        
    #print("x1_hat_prev",x1_hat_prev.shape)
    #x_hat_prev=np.array([[x1_hat_prev],[x2_hat_prev]])
    #x_hat_prev.shape=(2,1,3)
    #print("x_hat_prev",x_hat_prev.shape)
    #print("x_hat_prev",x_hat_prev)    
    #xtilde=A@x_hat_prev+B@u    #+bruit
    #ytilde=C@xtilde            #+bruit
    #Ptildeapprox=np.cov(xtilde)
    #Kkapprox=Ptildeapprox@C.T*(1/(C@Ptildeapprox@C.T+R))
    #x_hat=xtilde+Kkapprox*(yk-ytilde)
        w=[[np.random.normal(0,Q[0][0])],[np.random.normal(0,Q[0][0])]]
        v=np.random.normal(0,0.0125)
        xtilde[i][:][:]=x_hat_prev+(h/6)*(f(x_hat_prev,uprev,1)+2*f(x_hat_prev,uprev,2)+2*f(x_hat_prev,uprev,3)+f(x_hat_prev,uprev,4))+w 
        ytilde[i][:]=xtilde[i][1][0]+v
        #print("xtilde",xtilde[i][:][:],"i",i)
        #print("C",C.shape)
        #print(ytilde[i][:].shape)
        
        
    #N-2 because our N-1 = N of the formula

    covxtilde12=(1/(N-2))*np.sum([(xtilde[n][0][0]-np.mean([(xtilde[m][0][0]) for m in range(0,N)]))*(xtilde[n][1][0]-np.mean([(xtilde[m][1][0]) for m in range(0,N)])) for n in range(0,N)])
    varxtilde1=(1/(N-2))*np.sum([(xtilde[n][0][0]-np.mean([(xtilde[m][0][0]) for m in range(0,N)]))*(xtilde[n][0][0]-np.mean([(xtilde[m][0][0]) for m in range(0,N)])) for n in range(0,N)])
    varxtilde2=(1/(N-2))*np.sum([(xtilde[n][1][0]-np.mean([(xtilde[m][1][0]) for m in range(0,N)]))*(xtilde[n][1][0]-np.mean([(xtilde[m][1][0]) for m in range(0,N)])) for n in range(0,N)])
    
    Ptildeapprox=np.array([varxtilde1,covxtilde12,covxtilde12,varxtilde2]).reshape((2,2))
   # print(Ptildeapprox)
    #print(xtilde[1][0][0]-np.mean([(xtilde[m][0][0]) for m in range(0,N)]))
    
    #print(Ptildeapprox)

    Kkapprox=Ptildeapprox@C.T*(1/(C@Ptildeapprox@C.T+R))
    for i in range(0,N):
        x_hat[i][:][:]=xtilde[i][:][:]+Kkapprox*(yk-ytilde[i][:])
        #print("hat",x_hat[i][:][:],"i",i)
        
        
        
    x1rmds=np.zeros(N)
    for i in range(0,N):
        x1rmds[i]=(x_hat[i][0][0]-x1_true[j])
    x1rmdsmoy[j]=np.mean([(x1rmds[i]) for i in range(0,N)])
    x2rmds=np.zeros(N)
    for i in range(0,N):
        x2rmds[i]=(x_hat[i][1][0]-x2_true[j])
    x2rmdsmoy[j]=np.mean([(x2rmds[i]) for i in range(0,N)])
    
    x1_hat[j]=np.mean([(x_hat[i][0][0]) for i in range(0,N)])
    x2_hat[j]=np.mean([(x_hat[i][1][0]) for i in range(0,N)])
    

    x1sigmaplus[j]=x1_hat[j]+1.96*(Ptildeapprox[0][0])**(1/2)
    x1sigmamoins[j]=x1_hat[j]-1.96*(Ptildeapprox[0][0])**(1/2)
    x2sigmaplus[j]=x2_hat[j]+1.96*(Ptildeapprox[1][1])**(1/2)
    x2sigmamoins[j]=x2_hat[j]-1.96*(Ptildeapprox[1][1])**(1/2)
    
    



Linewdth = 1
fig, axis = plt.subplots(1, 4,figsize=(20, 5))
#axis[0].plot(np.arange(0,1000,1)*(h),x1sigmaplus,color="lightsteelblue", linewidth=Linewdth)
#axis[0].plot(np.arange(0,1000,1)*(h),x1sigmamoins,color="lightsteelblue", linewidth=Linewdth)
#axis[0].fill_between(np.arange(0,1000,1)*(h),y1=x1sigmaplus, y2=x1sigmamoins,color="lightsteelblue",label="95% CI")
axis[0].plot(np.arange(0,1000,1)*(h),x1_true,color="royalblue",label="True value", linewidth=Linewdth)
axis[0].plot(np.arange(0,1000,1)*(h),x1,color="blue",label="Filtered Data", linewidth=Linewdth)
axis[0].legend()
axis[0].set_aspect(1.0/axis[0].get_data_ratio(), adjustable='box')
axis[0].set_xlabel('Time [s]')
axis[0].set_ylabel('Height [m]')
axis[0].set_title("X1 non-linear EnKf")
  


#axis[1].plot(np.arange(0,1000,1)*h,x2sigmaplus,color="pink", linewidth=Linewdth)
#axis[1].plot(np.arange(0,1000,1)*h,x2sigmamoins,color="pink", linewidth=Linewdth)
#axis[1].fill_between(np.arange(0,1000,1)*h,y1=x2sigmaplus, y2=x2sigmamoins,color="pink",label="95% CI")
axis[1].plot(np.arange(0,1000,1)*h,x2_true,color="indianred",label="True value", linewidth=Linewdth)
axis[1].plot(np.arange(0,1000,1)*h,x2,color="red",label="Filtered Data", linewidth=Linewdth)
axis[1].legend()
axis[1].set_aspect(1.0/axis[1].get_data_ratio(), adjustable='box')
axis[1].set_xlabel('Time [s]')
axis[1].set_ylabel('Height [m]')
axis[1].set_title("X2 non-linear EnKf")
  

axis[2].plot(np.arange(0,1000,1)*h,x1rmdsmoy,color="blue",label="RMSD x1", linewidth=Linewdth)
axis[2].plot(np.arange(0,1000,1)*h,x2rmdsmoy,color="red",label="RMSD x2", linewidth=Linewdth)
axis[2].legend()
axis[2].set_aspect(1.0/axis[2].get_data_ratio(), adjustable='box')
axis[2].set_xlabel('Time [s]')
axis[2].set_ylabel('RMSD [m]')
axis[2].set_title("RMSD")
  

axis[3].plot(np.arange(0,1000,1)*(h),x1sigmaplus-x1sigmamoins,color="blue", linewidth=Linewdth, label="95% Ci of x1")
axis[3].plot(np.arange(0,1000,1)*h,x2sigmaplus-x2sigmamoins,color="red", linewidth=Linewdth, label="95% Ci of x1")
axis[3].legend()
axis[3].set_aspect(1.0/axis[3].get_data_ratio(), adjustable='box')
axis[3].set_xlabel('Time [s]')
axis[3].set_ylabel('Width [m]')
axis[3].set_title("95% Confidence Intervals")


plt.show()    
    
    
    
