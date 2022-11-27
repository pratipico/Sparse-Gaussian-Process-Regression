import GPy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
kernel = GPy.kern.RBF(1)

#X = np.random.uniform(-3.,3.,(20,1))
#Y = np.sin(X) + np.random.randn(20,1)*0.05

N = 1000
X = dist.Uniform(0.0, 5.0).sample(sample_shape=(N,))
Y = 0.5 * torch.sin(3 * X) + dist.Normal(0.0, 0.2).sample(sample_shape=(N,))

model = GPy.models.GPRegression(X, Y, kernel=kernel)
model.optimize()
model.plot()
plt.savefig('fig2.png')
plt.show()