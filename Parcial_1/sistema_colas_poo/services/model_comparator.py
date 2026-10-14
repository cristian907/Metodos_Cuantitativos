#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Servicio de Comparación Multimodelo de Teoría de Colas.
Permite evaluar y comparar en paralelo los modelos M/M/1, M/M/s, M/D/1 y M/G/1
bajo los mismos parámetros de entrada para analizar el impacto de la variabilidad del servicio.
"""

from typing import Dict, Any, List
from ..models.mm1_model import ModeloMM1
from ..models.mms_model import ModeloMMs
from ..models.md1_model import ModeloMD1
from ..models.mg1_model import ModeloMG1


def comparar_modelos(
    lambda_rate: float,
    mu_rate: float,
    s: int = 1,
    sigma: float = 0.5,
    time_unit: str = "minutos",
    cq: float = 0.0,
    cs: float = 0.0,
    cw: float = 0.0
) -> Dict[str, Any]:
    """
    Compara las métricas de desempeño entre los modelos M/M/1 (o M/M/s), M/D/1 y M/G/1.

    Parámetros:
        lambda_rate (float): Tasa media de llegada (λ)
        mu_rate (float): Tasa media de servicio (μ)
        s (int): Número de servidores (para M/M/s)
        sigma (float): Desviación típica del tiempo de servicio (para M/G/1)
        time_unit (str): Unidad de tiempo
        cq, cs, cw: Parámetros de costo

    Retorna:
        Dict conteniendo las instancias de modelos, tabla comparativa y reporte en texto.
    """
    modelos = []

    # Modelo Exponencial (M/M/1 o M/M/s)
    if s == 1:
        modelos.append(ModeloMM1(lambda_rate, mu_rate, time_unit, cq, cs, cw))
    else:
        modelos.append(ModeloMMs(lambda_rate, mu_rate, s, time_unit, cq, cs, cw))

    # Modelo Determinista M/D/1 (servidor único)
    modelos.append(ModeloMD1(lambda_rate, mu_rate, time_unit, cq, cs, cw))

    # Modelo General M/G/1 (servidor único con σ)
    modelos.append(ModeloMG1(lambda_rate, mu_rate, sigma, time_unit, cq, cs, cw))

    # Formatear tabla de comparación
    lineas = [
        "=" * 88,
        "                 TABLA COMPARATIVA DE MODELOS DE TEORÍA DE COLAS",
        "=" * 88,
        f"  Parámetros base: λ = {lambda_rate:.4f} | μ = {mu_rate:.4f} | Unidad: {time_unit}",
        f"  Servidores: M/M/{s} | M/D/1 (σ=0) | M/G/1 (σ={sigma:.4f})",
        "-" * 88,
        f"  {'Métrica / Indicador':<30} | {'M/M/' + str(s):<15} | {'M/D/1':<15} | {'M/G/1':<15}",
        "  " + "-" * 84
    ]

    metricas = [
        ("Factor Utilización (ρ)", lambda m: f"{m.rho:.4f} ({m.rho*100:.2f}%)"),
        ("Prob. Sistema Vacío (P0)", lambda m: f"{m.p0():.4f}"),
        ("Prob. de Espera (Pw)", lambda m: f"{m.pw():.4f}"),
        ("Clientes en Cola (Lq)", lambda m: f"{m.lq():.4f}"),
        ("Clientes en Sistema (Ls)", lambda m: f"{m.ls():.4f}"),
        (f"Tiempo en Cola (Wq) [{time_unit}]", lambda m: f"{m.wq():.4f}"),
        (f"Tiempo en Sistema (Ws) [{time_unit}]", lambda m: f"{m.ws():.4f}"),
    ]

    if cs > 0 or cq > 0 or cw > 0:
        metricas.append((f"Costo Total (CTS) [/{time_unit}]", lambda m: f"{m.calcular_costos()['CTS']:.4f}"))

    for nombre_metrica, extractor in metricas:
        vals = [extractor(m) for m in modelos]
        lineas.append(f"  {nombre_metrica:<30} | {vals[0]:<15} | {vals[1]:<15} | {vals[2]:<15}")

    lineas.extend([
        "=" * 88,
        "  ANÁLISIS DE VARIABILIDAD:",
        "  • El modelo M/D/1 (servicio constante σ=0) siempre reduce Lq y Wq exactamente a la mitad",
        "    con respecto a M/M/1 para un solo servidor.",
        "  • En M/G/1, a mayor desviación estándar σ, mayor es la congestión y el tiempo de espera en cola.",
        "=" * 88
    ])

    reporte_texto = "\n".join(lineas)

    return {
        "modelos": modelos,
        "reporte_texto": reporte_texto
    }
