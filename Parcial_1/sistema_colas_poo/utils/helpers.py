#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de utilidades y funciones auxiliares para la calculadora de teoría de colas.
"""

from typing import Union


def parse_float(val: Union[str, float, int]) -> float:
    """
    Convierte una cadena, entero o flotante a float, aceptando comas (,) o puntos (.)
    como separador decimal.
    
    Parámetros:
        val: Valor de entrada.
        
    Retorna:
        float: Valor numérico en formato flotante.
        
    Excepciones:
        ValueError: Si el formato no es numérico válido.
    """
    if isinstance(val, (float, int)):
        return float(val)
    if not isinstance(val, str):
        raise ValueError(f"No se puede convertir el tipo {type(val)} a float.")
    
    cleaned = val.strip().replace(',', '.')
    if not cleaned:
        raise ValueError("Entrada vacía.")
    return float(cleaned)


def format_percent(val: float) -> str:
    """Formatea un valor decimal (ej. 0.85) como porcentaje (85.00%)."""
    return f"{val * 100.0:.2f}%"


def print_divider(char: str = "-", length: int = 72):
    """Imprime una línea divisoria."""
    print(char * length)


def print_header(title: str, length: int = 72):
    """Imprime un encabezado centrado con bordes."""
    print("=" * length)
    print(title.center(length))
    print("=" * length)
