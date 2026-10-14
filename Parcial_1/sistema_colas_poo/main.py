#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Punto de Entrada Principal del Sistema de Resolución de Teoría de Colas (POO).
Universidad José Antonio Páez (UJAP) - Métodos Cuantitativos

Módulos soportados:
  - M/M/1 (Servidor único, llegadas Poisson, servicio exponencial)
  - M/M/s (Múltiples servidores, llegadas Poisson, servicio exponencial)
  - M/D/1 (Servidor único, servicio determinista/constante σ=0)
  - M/G/1 (Servidor único, servicio general con desviación típica σ)

Ejecución:
  python3 sistema_colas_poo/main.py
"""

import sys
from pathlib import Path

# Añadir el directorio raíz al path para soporte de ejecuciones directas
RAIZ_PROYECTO = str(Path(__file__).parent.parent)
if RAIZ_PROYECTO not in sys.path:
    sys.path.insert(0, RAIZ_PROYECTO)

from sistema_colas_poo.cli import menu_principal


def main():
    """Inicia la interfaz de línea de comandos."""
    try:
        menu_principal()
    except KeyboardInterrupt:
        print("\n\n  [Info] Programa interrumpido por el usuario. Saliendo...\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
