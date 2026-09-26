import pandas as pd
from IPython.display import display, clear_output


class TitanicController:
    """
    CONTROLLER
    Recibe Model y View.
    Gestiona eventos de la interfaz y decide qué método del Model ejecutar.
    """

    def __init__(self, model, view):
        self.model = model
        self.view = view

        self.view.actualizar_columnas(
            self.model.nombres_columnas()
        )

        self.view.boton_ejecutar.on_click(self.ejecutar)
        self.view.boton_restaurar.on_click(self.restaurar)

    def iniciar(self):
        self.view.mostrar()

    def restaurar(self, _):
        self.model.restaurar()
        self.view.actualizar_columnas(
            self.model.nombres_columnas()
        )

        with self.view.salida:
            clear_output()
            print("Datos restaurados correctamente.")
            print(self.model.informacion())

    def _convertir_valor(self, columna, valor):
        serie = self.model.obtener_columna(columna)

        if pd.api.types.is_numeric_dtype(serie):
            try:
                return float(valor)
            except ValueError:
                return valor

        return valor

    def ejecutar(self, _):
        metodo = self.view.metodo.value

        with self.view.salida:
            clear_output()

            try:
                if metodo == "Información":
                    print(self.model.informacion())

                elif metodo == "Mostrar":
                    display(self.model.mostrar().head(100))

                elif metodo == "Primeras filas":
                    display(self.model.primeras_filas(10))

                elif metodo == "Últimas filas":
                    display(self.model.ultimas_filas(10))

                elif metodo == "Obtener fila":
                    display(
                        self.model.obtener_fila(
                            self.view.indice.value
                        ).to_frame().T
                    )

                elif metodo == "Obtener columna":
                    display(
                        self.model.obtener_columna(
                            self.view.columna.value
                        ).head(100).to_frame()
                    )

                elif metodo == "Dimensiones":
                    print(self.model.dimensiones())

                elif metodo == "Nombres columnas":
                    print(self.model.nombres_columnas())

                elif metodo == "Tipos de datos":
                    display(
                        self.model.tipos_datos().to_frame("tipo")
                    )

                elif metodo == "Contar nulos":
                    display(
                        self.model.contar_nulos().to_frame("nulos")
                    )

                elif metodo == "Eliminar nulos":
                    display(self.model.eliminar_nulos().head(50))

                elif metodo == "Reemplazar nulos":
                    valor = self.view.valor.value or 0
                    display(
                        self.model.reemplazar_nulos(valor).head(50)
                    )

                elif metodo == "Ordenar":
                    display(
                        self.model.ordenar(
                            self.view.columna.value
                        ).head(100)
                    )

                elif metodo == "Filtrar":
                    columna = self.view.columna.value
                    valor = self._convertir_valor(
                        columna,
                        self.view.valor.value
                    )

                    display(
                        self.model.filtrar(
                            columna,
                            self.view.operador.value,
                            valor
                        ).head(100)
                    )

                elif metodo == "Agregar columna":
                    nombre = self.view.valor.value or "nueva_columna"
                    display(
                        self.model.agregar_columna(
                            nombre,
                            1
                        ).head(50)
                    )
                    self.view.actualizar_columnas(
                        self.model.nombres_columnas()
                    )

                elif metodo == "Eliminar columna":
                    display(
                        self.model.eliminar_columna(
                            self.view.columna.value
                        ).head(50)
                    )
                    self.view.actualizar_columnas(
                        self.model.nombres_columnas()
                    )

                elif metodo == "Estadísticas":
                    display(self.model.estadisticas().T)

                elif metodo == "Correlación":
                    display(self.model.correlacion())

                elif metodo == "Visualizar datos":
                    self.model.visualizar_datos()

                elif metodo == "Frecuencias":
                    resultado = self.model.frecuencias()

                    for columna, frecuencias in resultado.items():
                        print(f"\n--- {columna} ---")
                        display(
                            frecuencias.head(20).to_frame("frecuencia")
                        )

                elif metodo == "Detectar duplicados":
                    resultado = self.model.detectar_duplicados()
                    print("Cantidad:", len(resultado))
                    display(resultado.head(100))

                elif metodo == "Eliminar duplicados":
                    display(
                        self.model.eliminar_duplicados().head(50)
                    )

                elif metodo == "Convertir tipos":
                    columna = self.view.columna.value
                    display(
                        self.model.convertir_tipos(
                            {columna: "float64"}
                        ).head(50)
                    )

                elif metodo == "Distribución variables":
                    resultado = self.model.distribucion_variables()

                    print("NUMÉRICAS")
                    display(resultado["numericas"])

                    print("CATEGÓRICAS")
                    display(resultado["categoricas"])

                elif metodo == "Histogramas":
                    self.model.histogramas()

                elif metodo == "Boxplots":
                    self.model.boxplots()

                elif metodo == "Scatterplots":
                    self.model.scatterplots()

                elif metodo == "Mapa de calor":
                    self.model.mapa_calor()

                elif metodo == "Detectar outliers":
                    resultado = self.model.detectar_outliers()

                    resumen = {
                        columna: len(datos)
                        for columna, datos in resultado.items()
                    }

                    display(
                        pd.Series(
                            resumen,
                            name="outliers"
                        ).to_frame()
                    )

                elif metodo == "Imputar nulos":
                    display(
                        self.model.imputar_nulos().head(50)
                    )

                elif metodo == "Normalizar":
                    display(
                        self.model.normalizar().head(50)
                    )

                elif metodo == "Estandarizar":
                    display(
                        self.model.estandarizar().head(50)
                    )

                elif metodo == "Seleccionar variables":
                    columnas = list(
                        self.view.columnas_multiple.value
                    )

                    display(
                        self.model.seleccionar_variables(
                            columnas
                        ).head(100)
                    )

                elif metodo == "Guardar CSV":
                    print(
                        "Archivo creado:",
                        self.model.guardar_csv(
                            "dataframe_guardado.csv"
                        )
                    )

                elif metodo == "Exportar resultados":
                    print(
                        "Archivo creado:",
                        self.model.exportar_resultados(
                            "resultados_eda.csv"
                        )
                    )

            except Exception as e:
                print(
                    f"ERROR: {type(e).__name__}: {e}"
                )
