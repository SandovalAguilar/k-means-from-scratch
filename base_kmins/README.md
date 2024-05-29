# KMins: Implementación del Algoritmo K-means

KMins es una clase en Python para implementar el algoritmo de agrupamiento K-means, que permite dividir un conjunto de datos en `k` clusters. Esta implementación incluye métodos para ajustar el modelo, predecir etiquetas de cluster para nuevos datos, visualizar los resultados y utilizar el método del codo para determinar el número óptimo de clusters.

## Instalación

Para instalar el módulo KMins desde PyPI, utiliza el siguiente comando:

```bash
pip install kmins
```

## Descripción de la Clase

```python
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
        metodo_del_codo(X, max_k): Aplica el método del codo para encontrar el número óptimo de clusters.
    """
```

## Parámetros de Inicialización

- `k` (int): Número de clusters a encontrar.
- `max_iter` (int, opcional): Número máximo de iteraciones (por defecto 100).
- `tol` (float, opcional): Tolerancia para la convergencia (por defecto 1e-4).

## Atributos

- `centroides` (numpy.ndarray): Posiciones de los centroides de los clusters.
- `etiquetas` (numpy.ndarray): Etiqueta de cluster asignada a cada punto de datos.

## Métodos

### `ajustar(X)`

Ajusta el modelo K-means a un conjunto de datos.

- **Parámetros:**
  - `X` (numpy.ndarray): Matriz que contiene los datos de entrada.
- **Retorno:** Ninguno.

### `asignar_etiquetas()`

Asigna puntos de datos a los centroides más cercanos.

- **Retorno:** 
  - `etiquetas` (numpy.ndarray): Etiqueta de cluster para cada punto de datos.

### `calcular_centroides(etiquetas)`

Recalcula los centroides de los clusters.

- **Parámetros:**
  - `etiquetas` (numpy.ndarray): Etiqueta de cluster para cada punto de datos.
- **Retorno:** 
  - `nuevos_centroides` (numpy.ndarray): Posiciones de los nuevos centroides.

### `predecir(X)`

Predice las etiquetas de cluster para un nuevo conjunto de datos.

- **Parámetros:**
  - `X` (numpy.ndarray): Matriz que contiene los datos nuevos.
- **Retorno:** 
  - `etiquetas` (numpy.ndarray): Etiqueta de cluster predicha para cada punto de datos en `X`.

### `visualizar(X=None)`

Visualiza los datos y los clusters.

- **Parámetros:**
  - `X` (numpy.ndarray, opcional): Matriz que contiene los datos a visualizar. Si se omite, se utilizan los datos de entrenamiento.
- **Retorno:** Ninguno.

### `metodo_del_codo(X, max_k)`

Aplica el método del codo para encontrar el número óptimo de clusters.

- **Parámetros:**
  - `X` (numpy.ndarray): Matriz que contiene los datos de entrada.
  - `max_k` (int): Número máximo de clusters a probar.
- **Retorno:** 
  - `sse` (list): Lista de sumas de errores cuadráticos (SSE) para cada valor de `k`.

### `suma_errores_cuadraticos(kmeans)`

Calcula la suma de errores cuadráticos (SSE) para un modelo K-means ajustado.

- **Parámetros:**
  - `kmeans` (KMins): Instancia de la clase KMins ajustada.
- **Retorno:** 
  - `sse` (float): Suma de errores cuadráticos.

## Ejemplo de Uso

```python
import numpy as np
import matplotlib.pyplot as plt
from kmins import KMins

# Crear un conjunto de datos de ejemplo
X = np.random.rand(100, 2)

# Inicializar el modelo K-means
kmeans = KMins(k=3)

# Ajustar el modelo a los datos
kmeans.ajustar(X)

# Visualizar los resultados
kmeans.visualizar()

# Aplicar el método del codo
kmeans.metodo_del_codo(X, max_k=10)
```

Este ejemplo muestra cómo utilizar la clase `KMins` para ajustar un modelo K-means a un conjunto de datos, visualizar los clusters resultantes y aplicar el método del codo para determinar el número óptimo de clusters.

## Requisitos

- `numpy`
- `matplotlib`

Para instalar los requisitos, puedes usar:

```bash
pip install numpy 
pip install matplotlib
```

## Notas

- Este código es para propósitos educativos y puede necesitar ajustes para ser utilizado en producción.
- Asegúrate de tener instaladas las bibliotecas necesarias antes de ejecutar el código.

## Licencia

Este proyecto está bajo la Licencia MIT.

