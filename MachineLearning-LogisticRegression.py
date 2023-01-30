import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate sample data
np.random.seed(0)
x = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(x).ravel()
y[::5] += 3 * (0.5 - np.random.rand(16))

# Train the model
reg = LinearRegression().fit(x, y)

# Predict the values
y_pred = reg.predict(x)

# Plot the data and predictions
plt.scatter(x, y, color='red')
plt.plot(x, y_pred, color='brown', linewidth=3)
plt.show()
