import ipywidgets as widgets
from IPython.display import display


class TitanicView:
    """
    VIEW
    Solo contiene componentes visuales.
    No conoce la lógica del DataFrame.
    """

    def __init__(self):
        self.titulo = widgets.HTML("<h2>Titanic MVC - Jupyter</h2>")

        self.metodo = widgets.Dropdown(
            options=[
                "Información",
                "Mostrar",
                "Primeras filas",
                "Últimas filas",
                "Obtener fila",
                "Obtener columna",
                "Dimensiones",
                "Nombres columnas",
                "Tipos de datos",
                "Contar nulos",
                "Eliminar nulos",
                "Reemplazar nulos",
                "Ordenar",
                "Filtrar",
                "Agregar columna",
                "Eliminar columna",
                "Estadísticas",
                "Correlación",
                "Visualizar datos",
                "Frecuencias",
                "Detectar duplicados",
                "Eliminar duplicados",
                "Convertir tipos",
                "Distribución variables",
                "Histogramas",
                "Boxplots",
                "Scatterplots",
                "Mapa de calor",
                "Detectar outliers",
                "Imputar nulos",
                "Normalizar",
                "Estandarizar",
                "Seleccionar variables",
                "Guardar CSV",
                "Exportar resultados",
            ],
            description="Método:",
            layout=widgets.Layout(width="430px")
        )

        self.columna = widgets.Dropdown(
            options=[],
            description="Columna:",
            layout=widgets.Layout(width="330px")
        )

        self.operador = widgets.Dropdown(
            options=["==", "!=", ">", ">=", "<", "<="],
            description="Operador:"
        )

        self.valor = widgets.Text(
            description="Valor:",
            placeholder="Valor a usar"
        )

        self.indice = widgets.IntText(
            value=0,
            description="Fila:"
        )

        self.columnas_multiple = widgets.SelectMultiple(
            options=[],
            description="Columnas:",
            layout=widgets.Layout(width="350px", height="130px")
        )

        self.boton_ejecutar = widgets.Button(
            description="Ejecutar",
            button_style="primary"
        )

        self.boton_restaurar = widgets.Button(
            description="Restaurar datos"
        )

        self.salida = widgets.Output()

    def actualizar_columnas(self, columnas):
        self.columna.options = columnas
        self.columnas_multiple.options = columnas

    def mostrar(self):
        display(self.titulo)
        display(widgets.HBox([self.metodo, self.boton_ejecutar, self.boton_restaurar]))
        display(widgets.HBox([self.columna, self.operador, self.valor, self.indice]))
        display(self.columnas_multiple)
        display(self.salida)
