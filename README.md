# Laboratorio de análisis de datos

Aplicación MVC en Jupyter para cargar cualquier archivo CSV, realizar análisis exploratorio de datos (EDA), preprocesamiento y métodos de aprendizaje no supervisado.

## Uso rápido

1. Abra [notebooks/Version3_MVC.ipynb](notebooks/Version3_MVC.ipynb).
2. Ejecute las primeras cuatro celdas en orden.
3. Escriba la ruta de un archivo CSV en el campo **Archivo** y presione **Cargar archivo**.
4. Seleccione un método. La interfaz mostrará únicamente los parámetros necesarios para esa operación.

El archivo `data/drug200.csv` se incluye como ejemplo, pero la aplicación no depende de él.

## Procedimientos disponibles

### Consulta y estructura

| Procedimiento | Significado |
| --- | --- |
| Información | Resume la cantidad de filas, columnas y sus nombres. |
| Mostrar | Presenta el conjunto de datos cargado. |
| Primeras filas | Muestra los primeros registros para una inspección rápida. |
| Últimas filas | Muestra los registros finales. |
| Obtener fila | Recupera una fila según su posición numérica. |
| Obtener columna | Recupera una variable por su nombre. |
| Dimensiones | Devuelve la cantidad de filas y columnas $(n, p)$. |
| Nombres columnas | Lista las variables disponibles. |
| Tipos de datos | Indica si cada variable es numérica, texto, fecha, etc. |

### Limpieza y transformación

| Procedimiento | Significado |
| --- | --- |
| Contar nulos | Cuenta los valores ausentes por columna. |
| Eliminar nulos | Quita las filas que contienen al menos un valor ausente. |
| Reemplazar nulos | Sustituye los valores ausentes por el valor indicado. |
| Imputar nulos | Completa nulos con la mediana en variables numéricas y la moda en categóricas. |
| Detectar duplicados | Identifica registros repetidos. |
| Eliminar duplicados | Conserva una sola copia de cada registro repetido. |
| Convertir tipos | Cambia el tipo de una variable; en la interfaz actual convierte la columna elegida a `float64`. |
| Normalizar | Escala cada variable numérica al intervalo $[0, 1]$. |
| Estandarizar | Transforma variables numéricas para que tengan media $0$ y desviación estándar $1$. |
| Agregar columna | Crea una variable nueva; en la interfaz se agrega con valor constante `1`. |
| Eliminar columna | Borra una variable del conjunto de datos. |
| Seleccionar variables | Devuelve un subconjunto con las columnas seleccionadas. |

### Manipulación de registros

| Procedimiento | Significado |
| --- | --- |
| Ordenar | Reordena los registros ascendentemente según la columna elegida. |
| Filtrar | Conserva filas que cumplen una comparación: `==`, `!=`, `>`, `>=`, `<` o `<=`. |

### EDA y visualización

| Procedimiento | Significado |
| --- | --- |
| Estadísticas | Calcula conteos, media, desviación, cuartiles, mínimos y máximos; para categorías muestra frecuencia y moda. |
| Correlación | Calcula la relación lineal entre pares de variables numéricas, con valores entre $-1$ y $1$. |
| Visualizar datos | Muestra primeras filas, dimensiones y tipos de datos en una sola salida. |
| Frecuencias | Cuenta cuántas veces aparece cada valor en cada columna. |
| Distribución variables | Separa el resumen descriptivo de variables numéricas y categóricas. |
| Histogramas | Muestran la distribución de frecuencias de variables numéricas. |
| Boxplots | Resumen gráfico con mediana, cuartiles y posibles valores atípicos. |
| Scatterplots | Matriz de dispersión para observar relaciones entre pares de variables numéricas. |
| Mapa de calor | Representa visualmente la matriz de correlación. |
| Detectar outliers | Detecta valores atípicos con el rango intercuartílico: valores fuera de $[Q_1 - 1.5IQR, Q_3 + 1.5IQR]$. |

### Persistencia

| Procedimiento | Significado |
| --- | --- |
| Guardar CSV | Guarda el estado actual de los datos en `dataframe_guardado.csv`. |
| Exportar resultados | Exporta el estado resultante del EDA en `resultados_eda.csv`. |
| Restaurar datos | Recupera la copia original del último CSV cargado y descarta transformaciones hechas en memoria. |

### Reducción de dimensionalidad y clustering

Antes de estos métodos, la aplicación codifica variables categóricas mediante *one-hot encoding* y estandariza los atributos. Esto evita que la escala de una variable domine los resultados.

| Procedimiento | Significado |
| --- | --- |
| PCA | Análisis de Componentes Principales (ACP): crea componentes ortogonales que resumen la mayor cantidad posible de variabilidad. Se visualizan las dos primeras componentes. |
| PCA (variación) | Ejecuta PCA con `whiten=True` y el solucionador aleatorizado; sirve para comparar configuraciones. |
| HAC | Agrupamiento Jerárquico Aglomerativo: inicia con un grupo por observación y une los más similares hasta obtener el número de clústeres elegido. El enlace puede ser `ward`, `average`, `complete` o `single`. |
| HAC Dendrograma | Árbol que muestra el orden y la distancia a la que se fusionan grupos en HAC. |
| K-Means | Agrupa observaciones alrededor de $k$ centroides, minimizando la distancia interna de cada grupo. |
| K-Means Codo | Grafica la inercia para varios valores de $k$; el punto donde la mejora empieza a disminuir orienta la elección de clústeres. |
| t-SNE | Proyección no lineal que prioriza conservar vecindades locales; se utiliza principalmente para visualizar datos de muchas dimensiones. Su parámetro clave es la *perplexity*. |
| UMAP | Proyección no lineal que preserva estructura local y parte de la global. `n_neighbors` controla el tamaño del vecindario y `min_dist` la separación mínima en la proyección. |

Para HAC y K-Means se reporta el **silhouette score**, donde valores cercanos a $1$ sugieren grupos más separados. K-Means además reporta la **inercia**, suma de distancias cuadradas de cada observación a su centroide.

### Métricas impresas en consola

Cada vez que se ejecuta una técnica, el programa imprime su métrica principal. Esto permite revisar el resultado sin leer el código.

| Técnica | Mensaje en consola | Interpretación simple |
| --- | --- | --- |
| PCA | `PCA - Varianza explicada acumulada: X%` | Indica cuánta información de los datos conservan las componentes calculadas. Un porcentaje más alto es mejor; como referencia, superar $70\%$ suele ser una representación razonable. |
| HAC | `HAC - Correlación cofenética: X` | Mide qué tan bien el dendrograma conserva las distancias originales entre observaciones. Va aproximadamente de $0$ a $1$; valores cercanos a $1$ son mejores. |
| K-Means | `K-Means - Silhouette score: X` | Mide si los grupos están compactos y separados. Va de $-1$ a $1$; cercano a $1$ es bueno, cercano a $0$ indica grupos mezclados y negativo es desfavorable. |
| t-SNE | `t-SNE - Trustworthiness: X` | Indica si los vecinos cercanos de los datos originales continúan cerca en el plano proyectado. Va de $0$ a $1$; valores cercanos a $1$ son mejores. |
| UMAP | `UMAP - Trustworthiness: X` | Mide la conservación de vecinos al reducir los datos. Va de $0$ a $1$; valores cercanos a $1$ son mejores. |

Las funciones también devuelven el modelo o resultado. En PCA y clustering se retorna un diccionario con el modelo y la métrica; en t-SNE y UMAP, el `DataFrame` del embedding conserva el modelo y `trustworthiness` en `resultado.attrs`.

## Estructura

```text
data/            Archivos CSV de entrada
src/             Implementación MVC y clase base de EDA
src/unsupervised/ Subclase con PCA, HAC, K-Means, t-SNE y UMAP
src/supervised/   Espacio para métodos supervisados futuros
notebooks/        Punto de entrada en Jupyter
```

