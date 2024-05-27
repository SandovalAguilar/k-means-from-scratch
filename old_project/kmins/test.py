import kmins as k
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Simulación de datos
np.random.seed(21212)

x1 = np.random.standard_normal((100, 2)) * 0.6 + np.ones((100, 2))
x2 = np.random.standard_normal((100, 2)) * 0.5 - np.ones((100, 2))
x3 = np.random.standard_normal((100, 2)) * 0.4 - 2 * np.ones((100, 2)) + 5
X = np.concatenate((x1, x2, x3), axis=0)

plt.plot(X[:, 0], X[:, 1], 'k.')

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-means
kmeans = k.KMins(k=3, max_iter=500, tol=1e-3)
kmeans.ajustar(X_scaled)