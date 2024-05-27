import numpy as np
import matplotlib.pyplot as plt
 

class KMins:
    """
    Implementación del algoritmo K-means para agrupamiento de datos.

    Parametros:
        k: Numero de clusters a encontrar.
        max_iter: Numero maximo de iteraciones.
        tol: Tolerancia para la convergencia.

    Atributos:
        centroides: Posiciones de los centroides de los clusters.
        etiquetas: Etiqueta de cluster asignada a cada punto de datos.

    Metodos:
        ajustar(X): Ajusta el modelo a un conjunto de datos X.
        predecir(X): Predice las etiquetas de cluster para un nuevo conjunto de datos X.
        visualizar(X): Visualiza los datos y los clusters.
    """

    def __init__(self, k, max_iter, tol):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol

    def ajustar(self, X):
        """
        Ajusta el modelo a un conjunto de datos X.

        Parametros:
            X: Matriz numpy que contiene los datos de entrada.

        Retorno:
            Ninguno.
        """
        self.X = X
        self.n_datos, self.n_caracteristicas = X.shape

        # Inicializar centroides al azar
        self.centroides = X[np.random.choice(self.n_datos, size=self.k), :]

        for _ in range(self.max_iter):
            # Asignar puntos de datos a los centroides mas cercanos
            etiquetas = self.asignar_etiquetas()

            # Recalcular centroides
            nuevos_centroides = self.calcular_centroides(etiquetas)

            # Comprobar convergencia
            if np.linalg.norm(self.centroides - nuevos_centroides) < self.tol:
                break

            # Actualizar centroides
            self.centroides = nuevos_centroides

        self.etiquetas = etiquetas

    def asignar_etiquetas(self):
        """
        Asigna puntos de datos a los centroides mas cercanos.

        Retorno:
            Arreglo numpy que contiene la etiqueta de cluster para cada punto de datos.
        """
        etiquetas = np.zeros(self.n_datos, dtype=np.int32)
        for i, punto in enumerate(self.X):
            distancias = np.linalg.norm(punto - self.centroides, axis=1)
            etiquetas[i] = np.argmin(distancias)
        return etiquetas

    def calcular_centroides(self, etiquetas):
        """
        Recalcula los centroides de los clusters.

        Parametros:
            etiquetas: Arreglo numpy que contiene la etiqueta de cluster para cada punto de datos.

        Retorno:
            Matriz numpy que contiene las posiciones de los nuevos centroides.
        """
        nuevos_centroides = np.zeros((self.k, self.n_caracteristicas))
        for i in range(self.k):
            nuevos_centroides[i] = np.mean(self.X[etiquetas == i], axis=0)
        return nuevos_centroides

    def predecir(self, X):
        """
        Predice las etiquetas de cluster para un nuevo conjunto de datos X.

        Parametros:
            X: Matriz numpy que contiene los datos nuevos.

        Retorno:
            Arreglo numpy que contiene la etiqueta de cluster predicha para cada punto de datos en X.
        """
        etiquetas_predichas = np.zeros(X.shape[0], dtype=np.int32)
        for i, punto in enumerate(X):
            distancias = np.linalg.norm(punto - self.centroides, axis=1)
            etiquetas_predichas[i] = np.argmin(distancias)
        return etiquetas_predichas

    def visualizar(self, X=None):
        """
        Visualiza los datos y los clusters.

        Parametros:
            X: Matriz numpy que contiene los datos a visualizar. (Opcional, si se omite se utilizan los datos de entrenamiento)
        """
        if X is None:
            X = self.X

        plt.figure(figsize=(10, 6))

        # Gráfico de dispersión para los datos
        plt.subplot(1, 2, 1)
        plt.scatter(X[:, 0], X[:, 1], c=self.etiquetas)
        plt.title('Datos preprocesados')
        plt.xlabel('Característica 1')
        plt.ylabel('Característica 2')

        # Gráfico de dispersión para los datos con clusters
        plt.subplot(1, 2, 2)
        colores = ['b', 'g', 'r', 'c', 'm', 'y']
        plt.scatter(X[:, 0], X[:, 1], c=[colores[label]
                    for label in self.etiquetas])
        plt.scatter(self.centroides[:, 0], self.centroides[:, 1],
                    marker='x', s=150, c='black', linewidths=2)
        plt.title('Datos con clusters')
        plt.xlabel('Característica 1')
        plt.ylabel('Característica 2')
        plt.grid(True)

        plt.tight_layout()
        plt.show()
