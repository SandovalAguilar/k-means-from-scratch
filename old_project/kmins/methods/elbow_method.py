import numpy as np
import matplotlib.pyplot as plt


class ElbowMethod:
    def __init__(self, max_k):
        self.max_k = max_k

    def ajustar(self, X):
        self.X = X
        self.sse = []

        for k in range(1, self.max_k + 1):
            kmeans = KMins(k=k, max_iter=500, tol=1e-3)
            kmeans.ajustar(X)
            self.sse.append(self.calcular_sse(
                X, kmeans.centroides, kmeans.etiquetas))

    def calcular_sse(self, X, centroides, etiquetas):
        sse = 0
        for i, centroide in enumerate(centroides):
            sse += np.sum((X[etiquetas == i] - centroide) ** 2)
        return sse

    def graficar(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.max_k + 1), self.sse, marker='o')
        plt.title('Método del Codo')
        plt.xlabel('Número de clusters (k)')
        plt.ylabel('SSE')
        plt.grid(True)
        plt.show()
