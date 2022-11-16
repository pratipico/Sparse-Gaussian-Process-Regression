import numpy as np
import matplotlib.pyplot as plt
import LIRU_GP as GP
"""
This code runs a 1D GP regression example using the module LIRU_GP
P.L.Green
University of Liverpool
22/05/19
"""

# Make some 1D training data
N = 100  # No. points in training data
X = np.linspace(0, 10, N)  # Inputs spaced evenly between 0 and 10
F = X * np.sin(X)          # True fuction f is f = x*sin(x)
Y = F + 0.2*np.random.randn(N)   # Observations are corrupted with noise
N_Star = 200   # No. points where we are going to make predictions
X_Star = np.linspace(0, 10, N_Star)  # Inputs where we will make predictions

# Train GP
L0 = 0.5        # Initial estimate of the length scale
Sigma0 = 0.1    # Initial estimate of the noise standard deviation
L, Sigma, K, C, InvC, elapsed_time = GP.Train(L0, Sigma0, X, Y, N)
print('Hyperparameters: ', L, Sigma)    # Print hyperparameters
print('Elapsed Time:', elapsed_time)    # Print the time taken to train the GP

# Make predictions
Y_StarMean = np.zeros(N_Star)   # Array of prediction means
Y_StarStd = np.zeros(N_Star)    # Array of prediction standard deviations

# Make single prediction at the nth point in X_Star (Note that this is a
# slow way of doing it - vectorising the prediction operation would speed
# things up a lot).
for n in range(N_Star):
    Y_StarMean[n], Y_StarStd[n] = GP.Predict(X, X_Star[n], L, Sigma,
                                             Y, K, C, InvC, N)

# Plot Results
plt.figure()
plt.plot(X_Star, Y_StarMean, 'black', label='GP')
plt.plot(X_Star, Y_StarMean+3*Y_StarStd, 'black')
plt.plot(X_Star, Y_StarMean-3*Y_StarStd, 'black')
plt.plot(X, Y, '.', label='Observation Data')
plt.legend()
plt.xlabel('x')
plt.show()
