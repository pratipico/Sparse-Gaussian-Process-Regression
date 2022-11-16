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


K = FindK(X,10.0)
plt.subplot(1,3,1)
plt.imshow(K)
plt.title('l = 10')

K = FindK(X,1.0)
plt.subplot(1,3,2)
plt.imshow(K)
plt.title('l = 1')

K = FindK(X,0.1)
plt.subplot(1,3,3)
plt.imshow(K)
plt.title('l = 0.1')

plt.show()
