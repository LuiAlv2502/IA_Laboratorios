import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from scipy.cluster.hierarchy import dendrogram, linkage as scipy_linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    import umap
except ImportError:
    umap = None


class UnsupervisedAnalysis:
    """Implementa reducción de dimensionalidad y clustering."""

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def preparar_datos_numericos(self, columnas=None):
        datos = self.dataframe if columnas is None else self.dataframe[columnas]

        if datos.empty:
            raise ValueError("No hay datos para procesar.")

        datos_codificados = pd.get_dummies(datos, drop_first=True)

        if datos_codificados.empty:
            raise ValueError("No existen columnas utilizables.")

        matriz = StandardScaler().fit_transform(
            datos_codificados.to_numpy(dtype=float)
        )
        return matriz, list(datos_codificados.columns)

    def pca(self, n_components=2, whiten=False, svd_solver="auto", columnas=None):
        matriz, _ = self.preparar_datos_numericos(columnas)
        modelo = PCA(
            n_components=n_components,
            whiten=whiten,
            svd_solver=svd_solver,
            random_state=42,
        )
        componentes = modelo.fit_transform(matriz)
        nombres = [f"PC{i + 1}" for i in range(componentes.shape[1])]

        return {
            "componentes": pd.DataFrame(componentes, columns=nombres),
            "varianza_explicada": modelo.explained_variance_ratio_,
            "varianza_acumulada": np.cumsum(modelo.explained_variance_ratio_),
        }

    def pca_grafico(self, resultado_pca):
        componentes = resultado_pca["componentes"]

        if componentes.shape[1] < 2:
            raise ValueError("Se requieren al menos dos componentes.")

        figura, eje = plt.subplots(figsize=(8, 6))
        eje.scatter(componentes["PC1"], componentes["PC2"])
        eje.set_xlabel(f"PC1 ({resultado_pca['varianza_explicada'][0]:.1%})")
        eje.set_ylabel(f"PC2 ({resultado_pca['varianza_explicada'][1]:.1%})")
        eje.set_title("PCA - Componentes principales")
        plt.tight_layout()
        display(figura)
        plt.close(figura)
        return eje

    def hac(self, n_clusters=3, linkage="ward", metric="euclidean", columnas=None):
        matriz, _ = self.preparar_datos_numericos(columnas)
        modelo = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            metric=metric if linkage != "ward" else "euclidean",
        )
        etiquetas = modelo.fit_predict(matriz)
        resultado = self.dataframe.copy()
        resultado["cluster"] = etiquetas

        return {
            "resultado": resultado,
            "silhouette": silhouette_score(matriz, etiquetas),
        }

    def hac_dendrograma(self, metodo="ward", columnas=None):
        matriz, _ = self.preparar_datos_numericos(columnas)
        enlaces = scipy_linkage(matriz, method=metodo)
        figura, eje = plt.subplots(figsize=(12, 6))
        dendrogram(enlaces, ax=eje)
        eje.set_title(f"Dendrograma HAC (método={metodo})")
        eje.set_xlabel("Índice de muestra")
        eje.set_ylabel("Distancia")
        plt.tight_layout()
        display(figura)
        plt.close(figura)
        return eje

    def kmeans(self, n_clusters=3, init="k-means++", n_init=10, columnas=None):
        matriz, _ = self.preparar_datos_numericos(columnas)
        modelo = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            random_state=42,
        )
        etiquetas = modelo.fit_predict(matriz)
        resultado = self.dataframe.copy()
        resultado["cluster"] = etiquetas

        return {
            "resultado": resultado,
            "inercia": modelo.inertia_,
            "silhouette": silhouette_score(matriz, etiquetas),
        }

    def kmeans_codo(self, k_max=10, columnas=None):
        if k_max < 2:
            raise ValueError("k_max debe ser mayor o igual que 2.")

        matriz, _ = self.preparar_datos_numericos(columnas)
        inercias = []

        for k in range(1, k_max + 1):
            modelo = KMeans(n_clusters=k, n_init=10, random_state=42)
            modelo.fit(matriz)
            inercias.append(modelo.inertia_)

        figura, eje = plt.subplots(figsize=(8, 6))
        eje.plot(range(1, k_max + 1), inercias, marker="o")
        eje.set_xlabel("Número de clústeres (k)")
        eje.set_ylabel("Inercia")
        eje.set_title("Método del codo")
        plt.tight_layout()
        display(figura)
        plt.close(figura)
        return pd.Series(inercias, index=range(1, k_max + 1), name="inercia")

    def cluster_grafico(self, etiquetas, titulo="Clústeres"):
        """Proyecta las observaciones a 2D y las colorea por clúster."""
        matriz, _ = self.preparar_datos_numericos()
        proyeccion = PCA(n_components=2, random_state=42).fit_transform(matriz)

        figura, eje = plt.subplots(figsize=(8, 6))
        puntos = eje.scatter(
            proyeccion[:, 0],
            proyeccion[:, 1],
            c=etiquetas,
            cmap="tab10",
            alpha=0.8,
        )
        eje.set_xlabel("Componente principal 1")
        eje.set_ylabel("Componente principal 2")
        eje.set_title(titulo)
        figura.colorbar(puntos, ax=eje, label="Clúster")
        plt.tight_layout()
        display(figura)
        plt.close(figura)
        return eje

    def tsne(self, n_components=2, perplexity=30.0, learning_rate="auto", columnas=None):
        matriz, _ = self.preparar_datos_numericos(columnas)
        modelo = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            random_state=42,
            init="pca",
        )
        embedding = modelo.fit_transform(matriz)
        nombres = [f"Dim{i + 1}" for i in range(embedding.shape[1])]
        return pd.DataFrame(embedding, columns=nombres)

    def umap_embedding(self, n_components=2, n_neighbors=15, min_dist=0.1, columnas=None):
        if umap is None:
            raise ImportError("Instale umap-learn con: pip install umap-learn")

        matriz, _ = self.preparar_datos_numericos(columnas)
        modelo = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42,
        )
        embedding = modelo.fit_transform(matriz)
        nombres = [f"Dim{i + 1}" for i in range(embedding.shape[1])]
        return pd.DataFrame(embedding, columns=nombres)

    def embedding_grafico(self, embedding, titulo="Embedding"):
        if embedding.shape[1] < 2:
            raise ValueError("Se requieren al menos dos dimensiones.")

        columnas = list(embedding.columns)
        figura, eje = plt.subplots(figsize=(8, 6))
        eje.scatter(embedding[columnas[0]], embedding[columnas[1]])
        eje.set_xlabel(columnas[0])
        eje.set_ylabel(columnas[1])
        eje.set_title(titulo)
        plt.tight_layout()
        display(figura)
        plt.close(figura)
        return eje