#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Servicio de Optimización Económica de Servidores.
Determina el número óptimo de servidores (s*) en un sistema M/M/s
para minimizar el costo total del sistema (CTS).
"""

import math
from typing import Dict, Any, List
from ..models.mms_model import ModeloMMs


def optimizar_servidores(
    lambda_rate: float,
    mu_rate: float,
    cq: float,
    cs: float,
    cw: float = 0.0,
    time_unit: str = "minutos",
    max_s: int = 50
) -> Dict[str, Any]:
    """
    Busca iterativamente el número óptimo de servidores s* que minimiza el costo total.

    Parámetros:
        lambda_rate (float): Tasa media de llegadas (λ)
        mu_rate (float): Tasa media de servicio (μ)
        cq (float): Costo de espera por cliente en cola por unidad de tiempo
        cs (float): Costo por servidor por unidad de tiempo
        cw (float): Costo por atención por cliente
        time_unit (str): Unidad de tiempo
        max_s (int): Límite máximo de servidores a buscar

    Retorna:
        Dict con s_optimo, costo_optimo, lista de resultados por s y reporte formateado.
    """
    if lambda_rate <= 0 or mu_rate <= 0:
        raise ValueError("Las tasas λ y μ deben ser positivas.")
    if cs <= 0 and cq <= 0:
        raise ValueError("Se requiere al menos un costo positivo (Cs o Cq) para realizar la optimización.")

    # Servidores mínimos requeridos para estabilidad r = λ/μ
    r = lambda_rate / mu_rate
    s_min = int(math.floor(r)) + 1

    resultados = []
    best_s = s_min
    min_costo = float('inf')
    best_modelo = None

    for s in range(s_min, s_min + max_s):
        modelo = ModeloMMs(
            lambda_rate=lambda_rate,
            mu_rate=mu_rate,
            s=s,
            time_unit=time_unit,
            cq=cq,
            cs=cs,
            cw=cw
        )
        costos = modelo.calcular_costos()
        costo_total = costos["CTS"]

        item = {
            "s": s,
            "rho": modelo.rho,
            "pw": modelo.pw(),
            "lq": modelo.lq(),
            "ls": modelo.ls(),
            "wq": modelo.wq(),
            "ws": modelo.ws(),
            "CS": costos["CS"],
            "CQ": costos["CQ"],
            "CTS": costo_total,
            "modelo": modelo
        }
        resultados.append(item)

        if costo_total < min_costo:
            min_costo = costo_total
            best_s = s
            best_modelo = modelo
        elif costo_total > min_costo and s > best_s + 2:
            # Si el costo empieza a subir sostenidamente tras pasar el óptimo, detenemos la búsqueda
            break

    # Construcción de la tabla de texto
    lineas = [
        "=" * 85,
        "          TABLA DE OPTIMIZACIÓN ECONÓMICA DE SERVIDORES (BÚSQUEDA DE s*)",
        "=" * 85,
        f"  λ = {lambda_rate:.4f} | μ = {mu_rate:.4f} | Cs = {cs:.4f} | Cq = {cq:.4f} | Cw = {cw:.4f} ({time_unit})",
        "-" * 85,
        f"  {'s':<4} | {'ρ (Utiliz)':<10} | {'Pw (Espera)':<11} | {'Lq':<9} | {'Wq':<9} | {'CS':<10} | {'CQ':<10} | {'CTS Total':<10}",
        "  " + "-" * 81
    ]

    for item in resultados:
        es_optimo = " (ÓPTIMO s*)" if item["s"] == best_s else ""
        lineas.append(
            f"  {item['s']:<4} | {item['rho']:<10.4f} | {item['pw']:<11.4f} | {item['lq']:<9.4f} | "
            f"{item['wq']:<9.4f} | {item['CS']:<10.2f} | {item['CQ']:<10.2f} | {item['CTS']:<10.2f}{es_optimo}"
        )

    lineas.extend([
        "=" * 85,
        f"  >>> RESULTADO: El número óptimo de servidores es s* = {best_s}",
        f"  >>> Costo Total Mínimo (CTS) = {min_costo:.4f} por {time_unit}",
        "=" * 85
    ])

    reporte_texto = "\n".join(lineas)

    return {
        "s_optimo": best_s,
        "costo_optimo": min_costo,
        "modelo_optimo": best_modelo,
        "evaluaciones": resultados,
        "reporte_texto": reporte_texto
    }
