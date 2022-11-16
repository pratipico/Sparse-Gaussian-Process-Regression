import numpy as np
import matplotlib.pyplot as plt

N = 100
X = np.linspace(0,10,N)
K = np.zeros([N,N])

def FindK(X,L):
    for i in range(N):
        for j in range(N):
            K[i,j] = np.exp(-1/(2*L**2) * (X[i]-X[j])**2 )
    return K

m = np.zeros(N)

plt.subplot(3,1,1)
K = FindK(X,10.0)
for i in range(10):
    f = np.random.multivariate_normal(m,K)
    plt.plot(X,f,color='black',label='l = 10')
    #if i == 0:
    #    plt.legend(location=1)
plt.title('l = 10')

plt.subplot(3,1,2)
K = FindK(X,1.0)
for i in range(10):
    f = np.random.multivariate_normal(m,K)
    plt.plot(X,f,color='black',label='l = 1')
    #if i == 0:
    #    plt.legend()
plt.ylabel('f')
plt.title('l = 1')

plt.subplot(3,1,3)
K = FindK(X,0.1)
for i in range(10):
    f = np.random.multivariate_normal(m,K)
    plt.plot(X,f,color='black',label='l = 0.1')
    #if i == 0:
    #    plt.legend()
plt.xlabel('x')
plt.title('l = 0.1')


plt.show()
