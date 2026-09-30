"""Algoritmos de aprendizaje supervisado."""

from ._base import SplitData, SupervisadoBase
from .Classification import ClasificacionModelos

__all__ = [
	"ClasificacionModelos",
	"SplitData",
	"SupervisadoBase",
]