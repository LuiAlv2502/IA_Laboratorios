import pandas as pd
from DataFrame_desarrollado import DataFrame


class TitanicModel:
    """
    MODEL
    Encapsula el objeto de la clase DataFrame y expone
    operaciones que serán utilizadas por el Controller.
    """

    def __init__(self, ruta_csv):
        self.__ruta_csv = ruta_csv
        self.__datos_originales = pd.read_csv(ruta_csv)
        self.__dataframe = DataFrame(self.__datos_originales.copy())

    @property
    def dataframe(self):
        return self.__dataframe

    def restaurar(self):
        self.__dataframe = DataFrame(self.__datos_originales.copy())
        return self.__dataframe

    def informacion(self):
        return str(self.__dataframe)

    def mostrar(self):
        return self.__dataframe.mostrar()

    def primeras_filas(self, n=10):
        return self.__dataframe.primeras_filas(n)

    def ultimas_filas(self, n=10):
        return self.__dataframe.ultimas_filas(n)

    def obtener_fila(self, indice):
        return self.__dataframe.obtener_fila(indice)

    def obtener_columna(self, nombre):
        return self.__dataframe.obtener_columna(nombre)

    def dimensiones(self):
        return self.__dataframe.dimensiones()

    def nombres_columnas(self):
        return self.__dataframe.nombres_columnas()

    def tipos_datos(self):
        return self.__dataframe.tipos_datos()

    def contar_nulos(self):
        return self.__dataframe.contar_nulos()

    def eliminar_nulos(self):
        return self.__dataframe.eliminar_nulos()

    def reemplazar_nulos(self, valor=0):
        return self.__dataframe.reemplazar_nulos(valor)

    def ordenar(self, columna):
        return self.__dataframe.ordenar(columna)

    def filtrar(self, columna, operador, valor):
        return self.__dataframe.filtrar(columna, operador, valor)

    def agregar_columna(self, nombre, valor):
        return self.__dataframe.agregar_columna(nombre, valor)

    def eliminar_columna(self, nombre):
        return self.__dataframe.eliminar_columna(nombre)

    def estadisticas(self):
        return self.__dataframe.estadisticas()

    def correlacion(self):
        return self.__dataframe.correlacion()

    def visualizar_datos(self):
        return self.__dataframe.visualizar_datos()

    def frecuencias(self):
        return self.__dataframe.frecuencias()

    def detectar_duplicados(self):
        return self.__dataframe.detectar_duplicados()

    def eliminar_duplicados(self):
        return self.__dataframe.eliminar_duplicados()

    def convertir_tipos(self, conversiones):
        return self.__dataframe.convertir_tipos(conversiones)

    def distribucion_variables(self):
        return self.__dataframe.distribucion_variables()

    def histogramas(self):
        return self.__dataframe.histogramas()

    def boxplots(self):
        return self.__dataframe.boxplots()

    def scatterplots(self):
        return self.__dataframe.scatterplots()

    def mapa_calor(self):
        return self.__dataframe.mapa_calor()

    def detectar_outliers(self):
        return self.__dataframe.detectar_outliers()

    def imputar_nulos(self):
        return self.__dataframe.imputar_nulos()

    def normalizar(self):
        return self.__dataframe.normalizar()

    def estandarizar(self):
        return self.__dataframe.estandarizar()

    def seleccionar_variables(self, columnas):
        return self.__dataframe.seleccionar_variables(columnas)

    def guardar_csv(self, ruta="dataframe_guardado.csv"):
        return self.__dataframe.guardar_csv(ruta)

    def exportar_resultados(self, ruta="resultados_eda.csv"):
        return self.__dataframe.exportar_resultados(ruta)
