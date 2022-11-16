from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import LIRU_GP as GP
"""
This code runs a 2D GP regression example using the module LIRU_GP
P.L.Green
University of Liverpool
22/05/19
"""


# Make some 2D training data
N_grid = 10         # Input points will be distributed across a 10x10 grid
N = N_grid**2       # Total no. training data points is therefore N_grid**2
r = np.linspace(0, 10, N_grid)  # Inputs range from 0 to 10 in both directions
X_grid1, X_grid2 = np.meshgrid(r, r)  # Form mesh grid for X
Y_grid = np.zeros([N_grid, N_grid])   # Form mesh grid for y
X = np.zeros([N, 2])    # Initialise X values
Y = np.zeros(N)         # Initialise y values
n = 0
for i in range(N_grid):
    for j in range(N_grid):
        x = [X_grid1[i, j], X_grid2[i, j]]
        X[n, :] = x

        # True function is f(x)=sin(x_1)+cos(x_2)
        y = np.sin(x[0]) + np.cos(x[1]) + 0.2 * np.random.randn()

        Y_grid[i, j] = y
        Y[n] = y
        n += 1

# Train GP
L0 = 0.5        # Initial length scale
Sigma0 = 0.1    # Initial noise standard deviation
L, Sigma, K, C, InvC, elapsed_time = GP.Train(L0, Sigma0, X, Y, N)  # Train GP
print('Hyperparameters:', L, Sigma)      # Print hyperparameters
print('Elapsed Time:', elapsed_time)     # Print time taken to train GP

# Make some predictions
N_Star = 30   # Predictions will be distributed across a 30x30 grid
r_star = np.linspace(0, 10, N_Star)             # Input points
X_star1, X_star2 = np.meshgrid(r_star, r_star)  # Form mesh grid for X_star
Y_StarMean = np.zeros([N_Star, N_Star])         # mean of GP predictions
Y_StarStd = np.zeros([N_Star, N_Star])          # std of GP predictions
for i in range(N_Star):
    for j in range(N_Star):
        x_star = [X_star1[i, j], X_star2[i, j]]
        Y_StarMean[i, j], Y_StarStd[i, j] = GP.Predict(X, x_star, L, Sigma,
                                                       Y, K, C, InvC, N)

# Plot Results
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_wireframe(X_star1, X_star2, Y_StarMean, label='GP mean prediction')
ax.scatter(X_grid1, X_grid2, Y_grid, color='black', label='Training data')
plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
