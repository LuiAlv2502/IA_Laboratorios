from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataframe_desarrollado import DataFrame
from model import DataModel


DATASET = ROOT / "data" / "drug200.csv"


@pytest.fixture
def model():
    return DataModel(DATASET)


def test_cargar_csv_generico(model):
    assert model.dimensiones() == (200, 6)
    assert set(model.nombres_columnas()) == {
        "Age", "Sex", "BP", "Cholesterol", "Na_to_K", "Drug"
    }


def test_restaurar_recupera_datos_originales(model):
    model.eliminar_columna("Drug")
    model.restaurar()

    assert "Drug" in model.nombres_columnas()


def test_imputar_nulos_y_normalizar(tmp_path):
    ruta = tmp_path / "datos.csv"
    pd.DataFrame({"valor": [1.0, None, 3.0], "grupo": ["a", None, "b"]}).to_csv(
        ruta, index=False
    )
    model = DataModel(ruta)
    model.imputar_nulos()
    model.normalizar()

    assert model.contar_nulos().sum() == 0
    assert model.obtener_columna("valor").between(0, 1).all()


def test_pca_devuelve_dos_componentes(model):
    resultado = model.pca(n_components=2)

    assert resultado["componentes"].shape == (200, 2)
    assert len(resultado["varianza_explicada"]) == 2
    assert resultado["varianza_acumulada"][-1] <= 1


@pytest.mark.parametrize("algoritmo", ["hac", "kmeans"])
def test_clustering_asigna_grupo_y_metrica(model, algoritmo):
    resultado = getattr(model, algoritmo)(n_clusters=3)

    assert "cluster" in resultado["resultado"].columns
    assert resultado["resultado"]["cluster"].nunique() == 3
    assert -1 <= resultado["silhouette"] <= 1


def test_kmeans_reporta_inercia(model):
    resultado = model.kmeans(n_clusters=3)

    assert resultado["inercia"] > 0


@pytest.mark.parametrize(
    ("metodo", "parametros"),
    [
        ("tsne", {"perplexity": 5.0}),
        ("umap_embedding", {"n_neighbors": 5}),
    ],
)
def test_embeddings_tienen_dos_dimensiones(model, metodo, parametros):
    embedding = getattr(model, metodo)(**parametros)

    assert embedding.shape == (200, 2)


def test_pca_rechaza_dataframe_vacio():
    datos = DataFrame(pd.DataFrame())

    with pytest.raises(ValueError, match="No hay datos"):
        datos.pca()