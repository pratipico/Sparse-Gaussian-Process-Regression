import numpy as np
import matplotlib.pyplot as plt
import PeteGP as GP
import GPy
# GP verification code for a 1D regression problem. Compares PeteGP with standard regression using GPy. 

# Make some 1D training data
N = 50
X = np.linspace(0,10,N)
F = np.sin(X)
Y = F + 0.2*np.random.randn(N)
X_Star = np.linspace(0,10,200)
N_Star = len(X_Star)

# GP
L = np.array([0.1,1,5])
Sigma = 0.2

for i in range(3):
    K,C,InvC = GP.FindGramMatrix(X,L[i],Sigma,N)

    # Make some predictions 
    Y_StarMean = np.zeros(N_Star)
    Y_StarStd = np.zeros(N_Star)
    for n in range(N_Star):
        xStar = X_Star[n]
        Y_StarMean[n], Y_StarStd[n] = GP.Predict(X,xStar,L[i],Sigma,Y,K,C,InvC,N)

    # Plots
    plt.subplot(3,1,i+1)
    plt.plot(X_Star,Y_StarMean,color='black',label='GP')
    plt.plot(X_Star,Y_StarMean+3*Y_StarStd,color='black')
    plt.plot(X_Star,Y_StarMean-3*Y_StarStd,color='black')    
    plt.plot(X,Y,'o',label='Observations')
    plt.plot(X,F,color='red',label='f')
    if i == 0:
        plt.title('L = 0.5')
    elif i == 1:
        plt.title('L = 1')
    elif i == 2:
        plt.title('L = 5')
        
plt.xlabel('x')
plt.legend()
plt.show()

