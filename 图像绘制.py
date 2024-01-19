import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-5, 5, 0.1)

# sigmoid
y_sigmoid = 1 / (1 + np.exp(-x))

# tanh
y_tanh = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

# relu
y_relu = np.maximum(0, x)

# leaky relu
y_leaky_relu = np.maximum(0.01 * x, x)

# elu
y_elu = np.where(x < 0, np.exp(x) - 1, x)

plt.figure(figsize=(8,6))
plt.plot(x, y_sigmoid, label='sigmoid')
plt.plot(x, y_tanh, label='tanh')
plt.plot(x, y_relu, label='relu')
plt.plot(x, y_leaky_relu, label='leaky relu')
plt.plot(x, y_elu, label='elu')
plt.legend()
plt.grid()
plt.show()