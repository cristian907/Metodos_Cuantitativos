#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
SISTEMA DE RESOLUCIÓN DE TEORÍA DE COLAS (M/M/1, M/M/s, M/D/1, M/G/1) - UNIFICADO (POO)
===============================================================================
Universidad José Antonio Páez (UJAP) - Investigación de Operaciones / Métodos

Este archivo único integra completamente la arquitectura orientada a objetos (POO)
para la resolución, optimización económica, comparación multimodelo y generación
de reportes de sistemas de teoría de colas:

Modelos Incluidos:
  1. Modelo M/M/1: Servidor único con llegadas Poisson y servicio exponencial.
  2. Modelo M/M/s: Múltiples servidores en paralelo con Erlang C (s >= 1).
  3. Modelo M/D/1: Servidor único con tiempo de servicio determinista/constante (σ = 0).
  4. Modelo M/G/1: Servidor único con servicio general/arbitrario (desviación típica σ)
                   mediante la Fórmula de Pollaczek-Khinchine (P-K) y ajuste Curry-Feldman.

Características Adicionales:
  - Métricas completas: ρ, r, P0, Pw, Ls, Lq, Ws, Wq, Lb, Wb.
  - Probabilidades de estado: P(N = n), P(N <= n), P(N > n) en tabla o consulta.
  - Probabilidades de tiempo: P(Wq > t) y P(Ws > t).
  - Estructura de costos y optimización del número de servidores (s*).
  - Comparación multimodelo lado a lado.
  - Exportación de reportes detallados en archivos .txt.
  - Suite de pruebas unitarias integradas.
===============================================================================
"""

import math
import sys
import unittest
from typing import Dict, Any, List, Optional, Tuple, Union


# =============================================================================
# 1. FUNCIONES AUXILIARES Y FORMATO
# =============================================================================

def parse_float(val: Union[str, float, int]) -> float:
    """
    Convierte una cadena, entero o flotante a float, aceptando comas (,) o puntos (.)
    como separador decimal.
    """
    if isinstance(val, (float, int)):
        return float(val)
    if not isinstance(val, str):
        raise ValueError(f"No se puede convertir el tipo {type(val)} a float.")
    
    cleaned = val.strip().replace(',', '.')
    if not cleaned:
        raise ValueError("Entrada vacía.")
    return float(cleaned)


def print_divider(char: str = "-", length: int = 72):
    """Imprime una línea divisoria."""
    print(char * length)


def print_header(title: str, length: int = 72):
    """Imprime un encabezado centrado con bordes."""
    print("=" * length)
    print(title.center(length))
    print("=" * length)


# =============================================================================
# 2. CLASE BASE ABSTRACTA
# =============================================================================

class BaseQueueModel:
    """
    Clase base abstracta para modelos de teoría de colas.
    Define atributos comunes, validaciones de estabilidad y firma de métodos.
    """
    nombre_display = "Modelo Base"

    def __init__(
        self,
        lambda_rate: float,
        mu_rate: float,
        s: int = 1,
        time_unit: str = "minutos",
        cq: float = 0.0,
        cs: float = 0.0,
        cw: float = 0.0
    ):
        """
        Inicializa y valida los parámetros del modelo.
        """
        if lambda_rate <= 0:
            raise ValueError("La tasa de llegada (λ) debe ser estrictamente positiva (> 0).")
        if mu_rate <= 0:
            raise ValueError("La tasa de servicio (μ) debe ser strictly positiva (> 0).")
        if s < 1:
            raise ValueError("El número de servidores (s) debe ser un entero positivo (>= 1).")
        if cq < 0 or cs < 0 or cw < 0:
            raise ValueError("Los parámetros de costo (cq, cs, cw) no pueden ser negativos.")

        self.lambda_rate = float(lambda_rate)
        self.mu_rate = float(mu_rate)
        self.s = int(s)
        self.time_unit = str(time_unit)
        self.cq = float(cq)
        self.cs = float(cs)
        self.cw = float(cw)

        # Factor de utilización por servidor ρ = λ / (s * μ)
        self.rho = self.lambda_rate / (self.s * self.mu_rate)
        self.r = self.lambda_rate / self.mu_rate  # Intensidad de tráfico en erlangs (r = λ/μ)

        # Validación de estabilidad en estado estacionario
        if self.rho >= 1.0:
            raise ValueError(
                f"El sistema es INESTABLE (ρ = {self.rho:.4f} >= 1.0).\n"
                f"La tasa de llegada (λ = {self.lambda_rate}) supera o iguala la capacidad "
                f"máxima del sistema (s * μ = {self.s} * {self.mu_rate} = {self.s * self.mu_rate})."
            )

    def p0(self) -> float:
        """Probabilidad de que el sistema esté totalmente vacío (P0)."""
        raise NotImplementedError

    def pw(self) -> float:
        """Probabilidad de que un cliente entrante tenga que esperar (Pw)."""
        raise NotImplementedError

    def lq(self) -> float:
        """Número promedio de clientes en la cola (Lq)."""
        raise NotImplementedError

    def ls(self) -> float:
        """Número promedio de clientes en el sistema (Ls)."""
        raise NotImplementedError

    def wq(self) -> float:
        """Tiempo promedio de espera en la cola (Wq)."""
        raise NotImplementedError

    def ws(self) -> float:
        """Tiempo promedio de permanencia en el sistema (Ws)."""
        raise NotImplementedError

    def lb(self) -> float:
        """Número promedio de clientes en cola cuando el sistema está ocupado (Lb)."""
        pw_val = self.pw()
        return self.lq() / pw_val if pw_val > 0 else 0.0

    def wb(self) -> float:
        """Tiempo promedio de espera en cola cuando el sistema está ocupado (Wb)."""
        pw_val = self.pw()
        return self.wq() / pw_val if pw_val > 0 else 0.0

    def probabilidad_n(self, n: int) -> float:
        """Probabilidad de que haya exactamente n clientes en el sistema (Pn)."""
        raise NotImplementedError

    def probabilidad_n_o_menos(self, n: int) -> float:
        """Calcula P(N <= n): Probabilidad de que haya n o menos clientes en el sistema."""
        if n < 0:
            return 0.0
        return sum(self.probabilidad_n(i) for i in range(n + 1))

    def probabilidad_mas_de_n(self, n: int) -> float:
        """Calcula P(N > n): Probabilidad de que haya más de n clientes en el sistema."""
        return 1.0 - self.probabilidad_n_o_menos(n)

    def prob_tiempo_cola_mayor_t(self, t: float) -> float:
        """Aproximación / Cálculo de P(Wq > t): Probabilidad de esperar en cola más de t."""
        if t < 0:
            return 1.0
        wq_val = self.wq()
        if wq_val <= 0:
            return 0.0
        return self.pw() * math.exp(-t / wq_val)

    def prob_tiempo_sistema_mayor_t(self, t: float) -> float:
        """Aproximación / Cálculo de P(Ws > t): Probabilidad de permanecer en sistema más de t."""
        if t < 0:
            return 1.0
        ws_val = self.ws()
        if ws_val <= 0:
            return 0.0
        return math.exp(-t / ws_val)

    def calcular_costos(self) -> Dict[str, float]:
        """Calcula la estructura de costos detallada por unidad de tiempo."""
        lq_val = self.lq()
        ls_val = self.ls()
        servidores_ocupados = ls_val - lq_val  # r = λ/μ
        servidores_ociosos = float(self.s) - servidores_ocupados

        cs_total = self.cs * self.s
        cq_cola = self.cq * lq_val
        cq_sistema = self.cq * ls_val
        cw_atencion = self.cw * servidores_ocupados

        cts_total = cq_sistema + cs_total + cw_atencion

        costo_servidor_ocupado_unid = self.cs + self.cq + self.cw
        cts_busy = costo_servidor_ocupado_unid * servidores_ocupados
        cts_bar = self.cs * servidores_ociosos

        return {
            "servidores_ocupados": servidores_ocupados,
            "servidores_ociosos": servidores_ociosos,
            "CS": cs_total,
            "CQ": cq_cola,
            "C_espera_sistema": cq_sistema,
            "CTS_busy": cts_busy,
            "CTS_bar": cts_bar,
            "CTQ": cq_cola,
            "CTS": cts_total,
            "CT_total": cts_total
        }

    def resumen(self) -> Dict[str, Any]:
        """Genera un diccionario estructurado con las métricas del modelo."""
        return {
            "modelo": self.__class__.__name__,
            "nombre_modelo": getattr(self, "nombre_display", self.__class__.__name__),
            "lambda": self.lambda_rate,
            "mu": self.mu_rate,
            "s": self.s,
            "rho": self.rho,
            "utilizacion_pct": self.rho * 100.0,
            "r": self.r,
            "p0": self.p0(),
            "p0_pct": self.p0() * 100.0,
            "pw": self.pw(),
            "pw_pct": self.pw() * 100.0,
            "ls": self.ls(),
            "lq": self.lq(),
            "ws": self.ws(),
            "wq": self.wq(),
            "lb": self.lb(),
            "wb": self.wb(),
            "time_unit": self.time_unit
        }

    def generar_reporte_texto(
        self,
        consultas_adicionales: Optional[List[str]] = None,
        k_max_tabla: Optional[int] = None
    ) -> str:
        """Genera un reporte analítico detallado en formato texto plano."""
        r = self.resumen()
        lineas = [
            "=" * 72,
            f"       REPORTE DE TEORÍA DE COLAS - MODELO {r['nombre_modelo']}",
            "=" * 72,
            f"  Tasa de llegada (λ)                   : {r['lambda']:.6f} clientes / {self.time_unit}",
            f"  Tasa de servicio por servidor (μ)    : {r['mu']:.6f} clientes / {self.time_unit}",
            f"  Número de servidores (s)              : {r['s']}",
            "-" * 72,
            f"  Factor de utilización por servidor (ρ): {r['rho']:.4f} ({r['utilizacion_pct']:.2f}%)",
            f"  Intensidad de tráfico (r = λ/μ)       : {r['r']:.4f} erlangs",
            f"  Probabilidad de sistema vacío (P0)    : {r['p0']:.4f} ({r['p0_pct']:.2f}%)",
            f"  Probabilidad de tener que esperar (Pw): {r['pw']:.4f} ({r['pw_pct']:.2f}%)",
            "-" * 72,
            f"  N° medio de clientes en el sistema (Ls): {r['ls']:.4f} clientes",
            f"  N° medio de clientes en la cola (Lq)  : {r['lq']:.4f} clientes",
            f"  N° medio en cola si está ocupada (Lb) : {r['lb']:.4f} clientes",
            "-" * 72,
            f"  Tiempo medio en el sistema (Ws)       : {r['ws']:.4f} {self.time_unit}",
            f"  Tiempo medio de espera en cola (Wq)   : {r['wq']:.4f} {self.time_unit}",
            f"  Tiempo medio de espera si espera (Wb) : {r['wb']:.4f} {self.time_unit}",
            "=" * 72
        ]

        if self.cs > 0 or self.cq > 0 or self.cw > 0:
            c = self.calcular_costos()
            lineas.extend([
                "",
                "-" * 72,
                "                   ESTRUCTURA DE COSTOS DEL SISTEMA",
                "-" * 72,
                f"  Costos unitarios: Cs = {self.cs:.4f} | Cq = {self.cq:.4f} | Cw = {self.cw:.4f}",
                f"  Servidores ocupados esperados (r)     : {c['servidores_ocupados']:.4f}",
                f"  Servidores ociosos esperados (s - r)  : {c['servidores_ociosos']:.4f}",
                "-" * 72,
                f"  Coste total de servicio (CS = cs·s)   : {c['CS']:.4f} / {self.time_unit}",
                f"  Coste total espera en cola (CQ=cq·Lq) : {c['CQ']:.4f} / {self.time_unit}",
                f"  Coste total espera sistema (cq·Ls)    : {c['C_espera_sistema']:.4f} / {self.time_unit}",
                "-" * 72,
                "  Desglose Formato WinQSB / Teórico:",
                f"    • Costo serv. ocupados (CTS_busy)   : {c['CTS_busy']:.4f} / {self.time_unit}",
                f"    • Costo serv. ociosos (CTS_bar)     : {c['CTS_bar']:.4f} / {self.time_unit}",
                f"    • Costo espera en cola (CTQ)        : {c['CTQ']:.4f} / {self.time_unit}",
                "-" * 72,
                f"  COSTE TOTAL EN SISTEMA (CTS)          : {c['CTS']:.4f} / {self.time_unit}",
                "=" * 72
            ])

        if consultas_adicionales:
            lineas.extend([
                "",
                "-" * 72,
                "        RESULTADOS ADICIONALES (CONSULTAS DEL MENÚ SECUNDARIO)",
                "-" * 72
            ])
            for item in consultas_adicionales:
                lineas.append(f"  • {item}")
            lineas.append("=" * 72)

        if k_max_tabla is not None and k_max_tabla >= 0:
            lineas.extend([
                "",
                "-" * 72,
                f"        TABLA DE PROBABILIDADES DE ESTADO Pn (n = 0 a {k_max_tabla})",
                "-" * 72,
                f"  {'n':<5} | {'P(N = n)':<12} | {'P(N <= n)':<12} | {'P(N > n)':<12}",
                "  " + "-" * 48
            ])
            for i in range(k_max_tabla + 1):
                pn = self.probabilidad_n(i)
                pc = self.probabilidad_n_o_menos(i)
                pg = self.probabilidad_mas_de_n(i)
                lineas.append(f"  {i:<5} | {pn:<12.6f} | {pc:<12.6f} | {pg:<12.6f}")
            lineas.append("=" * 72)

        return "\n".join(lineas)

    def exportar_txt(
        self,
        filepath: str = "reporte_cola.txt",
        consultas_adicionales: Optional[List[str]] = None,
        k_max_tabla: Optional[int] = None
    ) -> str:
        """Exporta el reporte analítico a un archivo de texto plano."""
        contenido = self.generar_reporte_texto(
            consultas_adicionales=consultas_adicionales,
            k_max_tabla=k_max_tabla
        )
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(contenido)
        return filepath


# =============================================================================
# 3. IMPLEMENTACIÓN DE MODELOS ESPECÍFICOS
# =============================================================================

class ModeloMMs(BaseQueueModel):
    """
    Modelo M/M/s: Múltiples servidores en paralelo (s >= 1).
    Llegadas de Poisson y tiempos de servicio exponenciales.
    """
    nombre_display = "M/M/s"

    def __init__(
        self,
        lambda_rate: float,
        mu_rate: float,
        s: int = 1,
        time_unit: str = "minutos",
        cq: float = 0.0,
        cs: float = 0.0,
        cw: float = 0.0
    ):
        super().__init__(
            lambda_rate=lambda_rate,
            mu_rate=mu_rate,
            s=s,
            time_unit=time_unit,
            cq=cq,
            cs=cs,
            cw=cw
        )
        self.nombre_display = f"M/M/{self.s}"

    def p0(self) -> float:
        """P0 = 1 / [ sum_{n=0}^{s-1} (r^n / n!) + (r^s / (s! * (1 - ρ))) ]"""
        r = self.r
        sumatoria = sum([(r ** n) / math.factorial(n) for n in range(self.s)])
        ultimo_termino = (r ** self.s) / (math.factorial(self.s) * (1.0 - self.rho))
        return 1.0 / (sumatoria + ultimo_termino)

    def pw(self) -> float:
        """Fórmula Erlang C (Pw) = P(N >= s)."""
        r = self.r
        p0_val = self.p0()
        return ((r ** self.s) / (math.factorial(self.s) * (1.0 - self.rho))) * p0_val

    def lq(self) -> float:
        """Lq = (Pw * ρ) / (1 - ρ)"""
        return (self.pw() * self.rho) / (1.0 - self.rho)

    def ls(self) -> float:
        """Ls = Lq + (λ / μ)"""
        return self.lq() + self.r

    def wq(self) -> float:
        """Wq = Lq / λ"""
        return self.lq() / self.lambda_rate

    def ws(self) -> float:
        """Ws = Ls / λ = Wq + 1/μ"""
        return self.ls() / self.lambda_rate

    def probabilidad_n(self, n: int) -> float:
        """Probabilidad de exactamente n clientes en el sistema."""
        if n < 0:
            return 0.0
        p0_val = self.p0()
        r = self.r
        if n <= self.s:
            return ((r ** n) / math.factorial(n)) * p0_val
        else:
            denom = math.factorial(self.s) * (self.s ** (n - self.s))
            return ((r ** n) / denom) * p0_val

    def prob_tiempo_cola_mayor_t(self, t: float) -> float:
        """P(Wq > t) = Pw * exp(-s * μ * (1 - ρ) * t)"""
        if t < 0:
            return 1.0
        exponent = -self.s * self.mu_rate * (1.0 - self.rho) * t
        return self.pw() * math.exp(exponent)

    def prob_tiempo_sistema_mayor_t(self, t: float) -> float:
        """P(Ws > t)"""
        if t < 0:
            return 1.0
        mu = self.mu_rate
        pw_val = self.pw()
        r = self.r
        denom = float(self.s - 1 - r)

        if abs(denom) < 1e-9:
            factor = 1.0 + pw_val * mu * t
        else:
            factor = 1.0 + pw_val * ((1.0 - math.exp(-mu * denom * t)) / denom)

        return math.exp(-mu * t) * factor


class ModeloMM1(ModeloMMs):
    """
    Modelo M/M/1: Especialización de ModeloMMs para servidor único (s = 1).
    Utiliza fórmulas directas y cerradas simplificadas.
    """
    nombre_display = "M/M/1"

    def __init__(
        self,
        lambda_rate: float,
        mu_rate: float,
        time_unit: str = "minutos",
        cq: float = 0.0,
        cs: float = 0.0,
        cw: float = 0.0
    ):
        super().__init__(
            lambda_rate=lambda_rate,
            mu_rate=mu_rate,
            s=1,
            time_unit=time_unit,
            cq=cq,
            cs=cs,
            cw=cw
        )
        self.nombre_display = "M/M/1"

    def p0(self) -> float:
        """P0 = 1 - ρ"""
        return 1.0 - self.rho

    def pw(self) -> float:
        """Pw = ρ"""
        return self.rho

    def lq(self) -> float:
        """Lq = λ^2 / (μ * (μ - λ)) = ρ^2 / (1 - ρ)"""
        return (self.lambda_rate ** 2) / (self.mu_rate * (self.mu_rate - self.lambda_rate))

    def ls(self) -> float:
        """Ls = λ / (μ - λ) = ρ / (1 - ρ)"""
        return self.lambda_rate / (self.mu_rate - self.lambda_rate)

    def wq(self) -> float:
        """Wq = λ / (μ * (μ - λ))"""
        return self.lambda_rate / (self.mu_rate * (self.mu_rate - self.lambda_rate))

    def ws(self) -> float:
        """Ws = 1 / (μ - λ)"""
        return 1.0 / (self.mu_rate - self.lambda_rate)

    def lb(self) -> float:
        """Lb = Lq / Pw = ρ / (1 - ρ)"""
        return self.rho / (1.0 - self.rho)

    def wb(self) -> float:
        """Wb = Wq / Pw = 1 / (μ - λ)"""
        return 1.0 / (self.mu_rate - self.lambda_rate)

    def probabilidad_n(self, n: int) -> float:
        """Pn = (1 - ρ) * ρ^n"""
        if n < 0:
            return 0.0
        return (1.0 - self.rho) * (self.rho ** n)

    def prob_tiempo_cola_mayor_t(self, t: float) -> float:
        """P(Wq > t) = ρ * exp(-μ * (1 - ρ) * t)"""
        if t < 0:
            return 1.0
        return self.rho * math.exp(-self.mu_rate * (1.0 - self.rho) * t)

    def prob_tiempo_sistema_mayor_t(self, t: float) -> float:
        """P(Ws > t) = exp(-μ * (1 - ρ) * t)"""
        if t < 0:
            return 1.0
        return math.exp(-self.mu_rate * (1.0 - self.rho) * t)


class ModeloMG1(BaseQueueModel):
    """
    Modelo M/G/1: Llegadas Poisson, Servicio General/Arbitrario (1 servidor).
    Fórmula de Pollaczek-Khinchine (P-K).
    """
    nombre_display = "M/G/1"

    def __init__(
        self,
        lambda_rate: float,
        mu_rate: float,
        sigma: float = 0.0,
        time_unit: str = "minutos",
        cq: float = 0.0,
        cs: float = 0.0,
        cw: float = 0.0
    ):
        super().__init__(
            lambda_rate=lambda_rate,
            mu_rate=mu_rate,
            s=1,
            time_unit=time_unit,
            cq=cq,
            cs=cs,
            cw=cw
        )
        if sigma < 0:
            raise ValueError("La desviación estándar σ no puede ser negativa.")
        self.sigma = float(sigma)
        self.nombre_display = f"M/G/1 (σ={self.sigma:.4f})"

    def p0(self) -> float:
        """P0 = 1 - ρ"""
        return 1.0 - self.rho

    def pw(self) -> float:
        """Pw = ρ"""
        return self.rho

    def lq(self) -> float:
        """Fórmula de Pollaczek-Khinchine: Lq = [ λ^2 * σ^2 + ρ^2 ] / [ 2 * (1 - ρ) ]"""
        num = (self.lambda_rate ** 2) * (self.sigma ** 2) + (self.rho ** 2)
        denom = 2.0 * (1.0 - self.rho)
        return num / denom

    def ls(self) -> float:
        """Ls = Lq + ρ"""
        return self.lq() + self.rho

    def wq(self) -> float:
        """Wq = Lq / λ"""
        return self.lq() / self.lambda_rate

    def ws(self) -> float:
        """Ws = Ls / λ = Wq + 1/μ"""
        return self.ls() / self.lambda_rate

    def factor_curry_feldman(self) -> float:
        """Factor Curry-Feldman = (1 + μ^2 * σ^2) / 2"""
        return (1.0 + (self.mu_rate ** 2) * (self.sigma ** 2)) / 2.0

    def probabilidad_n(self, n: int) -> float:
        """Aproximación Pn ≈ (1 - ρ) * ρ^n"""
        if n < 0:
            return 0.0
        return (1.0 - self.rho) * (self.rho ** n)


class ModeloMD1(ModeloMG1):
    """
    Modelo M/D/1: Llegadas Poisson, Tiempo de Servicio Determinista/Constante (σ = 0).
    """
    nombre_display = "M/D/1"

    def __init__(
        self,
        lambda_rate: float,
        mu_rate: float,
        time_unit: str = "minutos",
        cq: float = 0.0,
        cs: float = 0.0,
        cw: float = 0.0
    ):
        super().__init__(
            lambda_rate=lambda_rate,
            mu_rate=mu_rate,
            sigma=0.0,
            time_unit=time_unit,
            cq=cq,
            cs=cs,
            cw=cw
        )
        self.nombre_display = "M/D/1"

    def lq(self) -> float:
        """Lq = ρ^2 / [ 2 * (1 - ρ) ] = Lq(M/M/1) / 2"""
        return (self.rho ** 2) / (2.0 * (1.0 - self.rho))


# =============================================================================
# 4. SERVICIOS DE OPTIMIZACIÓN Y COMPARACIÓN
# =============================================================================

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
    Busca iterativamente el número óptimo de servidores s* que minimiza el costo total (CTS).
    """
    if lambda_rate <= 0 or mu_rate <= 0:
        raise ValueError("Las tasas λ y μ deben ser positivas.")
    if cs <= 0 and cq <= 0:
        raise ValueError("Se requiere al menos un costo positivo (Cs o Cq) para realizar la optimización.")

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
            break

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
    Compara de forma simultánea los modelos M/M/1 (o M/M/s), M/D/1 y M/G/1.
    """
    modelos = []
    if s == 1:
        modelos.append(ModeloMM1(lambda_rate, mu_rate, time_unit, cq, cs, cw))
    else:
        modelos.append(ModeloMMs(lambda_rate, mu_rate, s, time_unit, cq, cs, cw))

    modelos.append(ModeloMD1(lambda_rate, mu_rate, time_unit, cq, cs, cw))
    modelos.append(ModeloMG1(lambda_rate, mu_rate, sigma, time_unit, cq, cs, cw))

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

    return {
        "modelos": modelos,
        "reporte_texto": "\n".join(lineas)
    }


# =============================================================================
# 5. PRUEBAS AUTOMATIZADAS INTERNAS
# =============================================================================

class TestSistemaColasUnificado(unittest.TestCase):
    """Pruebas unitarias para la validación de las clases y funciones del archivo unificado."""

    def test_mm1(self):
        m = ModeloMM1(0.8, 1.0)
        self.assertAlmostEqual(m.rho, 0.8)
        self.assertAlmostEqual(m.p0(), 0.2)
        self.assertAlmostEqual(m.pw(), 0.8)
        self.assertAlmostEqual(m.ls(), 4.0)
        self.assertAlmostEqual(m.lq(), 3.2)

    def test_mms(self):
        m = ModeloMMs(15.0, 6.0, 3)
        self.assertAlmostEqual(m.p0(), 0.0449438, places=4)
        self.assertAlmostEqual(m.lq(), 3.511235, places=4)

    def test_md1(self):
        m = ModeloMD1(0.8, 1.0)
        self.assertAlmostEqual(m.lq(), 1.6)

    def test_mg1(self):
        m = ModeloMG1(0.8, 1.0, 0.5)
        self.assertAlmostEqual(m.lq(), 2.0)
        self.assertAlmostEqual(m.factor_curry_feldman(), 0.625)

    def test_inestabilidad(self):
        with self.assertRaises(ValueError):
            ModeloMM1(10.0, 5.0)


# =============================================================================
# 6. INTERFAZ DE LÍNEA DE COMANDOS (CLI)
# =============================================================================

def pedir_float(prompt: str, min_val: Optional[float] = None, allow_zero: bool = False) -> float:
    """Solicita un flotante por consola con validación."""
    while True:
        try:
            val_str = input(prompt)
            val = parse_float(val_str)
            if min_val is not None:
                if allow_zero and val < min_val:
                    print(f"   [Error] El valor debe ser mayor o igual a {min_val}.")
                    continue
                elif not allow_zero and val <= min_val:
                    print(f"   [Error] El valor debe ser estrictamente mayor a {min_val}.")
                    continue
            return val
        except ValueError:
            print("   [Error] Entrada inválida. Ingrese un número (ej. 5 u 0.8).")


def pedir_int(prompt: str, min_val: int = 1) -> int:
    """Solicita un entero por consola."""
    while True:
        try:
            val_str = input(prompt)
            val = int(val_str.strip())
            if val < min_val:
                print(f"   [Error] Debe ingresar un entero >= {min_val}.")
                continue
            return val
        except ValueError:
            print("   [Error] Entrada inválida. Ingrese un entero.")


def pedir_costos() -> tuple:
    """Solicita parámetros de costo opcionales."""
    print("\n  --- Configuración Opcional de Costos del Sistema ---")
    desea = input("  ¿Desea incluir análisis de costos? (s/n, por defecto 'n'): ").strip().lower()
    if desea in ['s', 'si', 'y', 'yes']:
        cs = pedir_float("  • Costo por servidor por unidad de tiempo (Cs) [>= 0]: ", min_val=0.0, allow_zero=True)
        cq = pedir_float("  • Costo de espera por cliente en cola (Cq) [>= 0]: ", min_val=0.0, allow_zero=True)
        cw = pedir_float("  • Costo de atención por cliente (Cw) [>= 0]: ", min_val=0.0, allow_zero=True)
        return cs, cq, cw
    return 0.0, 0.0, 0.0


def submenu_consultas(modelo: BaseQueueModel):
    """Submenú interactivo para consultas adicionales."""
    consultas_realizadas: List[str] = []
    k_max_tabla: Optional[int] = None

    while True:
        print("\n" + "-" * 72)
        print("                   MENÚ DE CONSULTAS ADICIONALES")
        print("-" * 72)
        print("  1. Consultar P(N = n)   - Probabilidad de exactamente n clientes")
        print("  2. Consultar P(N <= n)  - Probabilidad de n o menos clientes")
        print("  3. Consultar P(N > n)   - Probabilidad de más de n clientes")
        print("  4. Consultar P(Wq > t)  - Probabilidad de esperar en cola más de t")
        print("  5. Consultar P(Ws > t)  - Probabilidad de permanecer en sistema más de t")
        print("  6. Generar Tabla de Probabilidades Pn (n = 0 a k)")
        print("  7. Exportar Reporte Completo a archivo .txt")
        print("  0. Volver al Menú Principal")
        print("-" * 72)

        opc = input("  Seleccione una opción: ").strip()

        if opc == "0":
            break
        elif opc == "1":
            n = pedir_int("  Ingrese el número de clientes (n >= 0): ", min_val=0)
            pn = modelo.probabilidad_n(n)
            msg = f"P(N = {n}) = {pn:.6f} ({pn * 100:.2f}%)"
            print(f"\n   >>> RESULTADO: {msg}")
            consultas_realizadas.append(msg)
        elif opc == "2":
            n = pedir_int("  Ingrese el número de clientes (n >= 0): ", min_val=0)
            pc = modelo.probabilidad_n_o_menos(n)
            msg = f"P(N <= {n}) = {pc:.6f} ({pc * 100:.2f}%)"
            print(f"\n   >>> RESULTADO: {msg}")
            consultas_realizadas.append(msg)
        elif opc == "3":
            n = pedir_int("  Ingrese el número de clientes (n >= 0): ", min_val=0)
            pg = modelo.probabilidad_mas_de_n(n)
            msg = f"P(N > {n}) = {pg:.6f} ({pg * 100:.2f}%)"
            print(f"\n   >>> RESULTADO: {msg}")
            consultas_realizadas.append(msg)
        elif opc == "4":
            t = pedir_float(f"  Ingrese el tiempo t ({modelo.time_unit}): ", min_val=0.0, allow_zero=True)
            pwq = modelo.prob_tiempo_cola_mayor_t(t)
            msg = f"P(Wq > {t:.2f} {modelo.time_unit}) = {pwq:.6f} ({pwq * 100:.2f}%)"
            print(f"\n   >>> RESULTADO: {msg}")
            consultas_realizadas.append(msg)
        elif opc == "5":
            t = pedir_float(f"  Ingrese el tiempo t ({modelo.time_unit}): ", min_val=0.0, allow_zero=True)
            pws = modelo.prob_tiempo_sistema_mayor_t(t)
            msg = f"P(Ws > {t:.2f} {modelo.time_unit}) = {pws:.6f} ({pws * 100:.2f}%)"
            print(f"\n   >>> RESULTADO: {msg}")
            consultas_realizadas.append(msg)
        elif opc == "6":
            k_max_tabla = pedir_int("  Ingrese el límite superior k (n = 0 a k): ", min_val=0)
            print("\n" + modelo.generar_reporte_texto(consultas_adicionales=consultas_realizadas, k_max_tabla=k_max_tabla))
        elif opc == "7":
            nombre_archivo = input("  Nombre del archivo (por defecto 'reporte_cola.txt'): ").strip() or "reporte_cola.txt"
            filepath = modelo.exportar_txt(
                filepath=nombre_archivo,
                consultas_adicionales=consultas_realizadas,
                k_max_tabla=k_max_tabla
            )
            print(f"\n   >>> ÉXITO: Reporte exportado a '{filepath}'")
        else:
            print("   [Error] Opción no válida.")


def main():
    """Función de bucle del menú principal."""
    while True:
        print_header("SISTEMA DE RESOLUCIÓN DE TEORÍA DE COLAS (UNIFICADO - POO)")
        print("  Universidad José Antonio Páez (UJAP) - Métodos Cuantitativos")
        print_divider("-")
        print("  1. Modelo M/M/1 (Servidor Único, Exponencial)")
        print("  2. Modelo M/M/s (Múltiples Servidores, Exponencial)")
        print("  3. Modelo M/D/1 (Servidor Único, Servicio Constante / Determinista)")
        print("  4. Modelo M/G/1 (Servidor Único, Servicio General con Desviación σ)")
        print("  5. Optimización Económica de Servidores (Búsqueda de s* Mínimo Costo)")
        print("  6. Comparación Multimodelo Lado a Lado (M/M/1 vs M/D/1 vs M/G/1)")
        print("  7. Ejecutar Pruebas Automatizadas del Sistema")
        print("  0. Salir")
        print_divider("-")

        opcion = input("  Seleccione una opción [0-7]: ").strip()

        if opcion == "0":
            print("\n  ¡Gracias por utilizar el Sistema de Teoría de Colas! Hasta luego.\n")
            break

        elif opcion == "1":
            print_header("CONFIGURACIÓN DE MODELO M/M/1")
            lam = pedir_float("  • Tasa media de llegadas (λ): ", min_val=0.0)
            mu = pedir_float("  • Tasa media de servicio (μ): ", min_val=0.0)
            tu = input("  • Unidad de tiempo [por defecto 'minutos']: ").strip() or "minutos"
            cs, cq, cw = pedir_costos()

            try:
                modelo = ModeloMM1(lambda_rate=lam, mu_rate=mu, time_unit=tu, cq=cq, cs=cs, cw=cw)
                print("\n" + modelo.generar_reporte_texto())
                submenu_consultas(modelo)
            except ValueError as e:
                print(f"\n   [ERROR DE ESTABILIDAD]: {e}\n")

        elif opcion == "2":
            print_header("CONFIGURACIÓN DE MODELO M/M/s")
            lam = pedir_float("  • Tasa media de llegadas (λ): ", min_val=0.0)
            mu = pedir_float("  • Tasa media de servicio por servidor (μ): ", min_val=0.0)
            s = pedir_int("  • Número de servidores (s >= 1): ", min_val=1)
            tu = input("  • Unidad de tiempo [por defecto 'minutos']: ").strip() or "minutos"
            cs, cq, cw = pedir_costos()

            try:
                modelo = ModeloMMs(lambda_rate=lam, mu_rate=mu, s=s, time_unit=tu, cq=cq, cs=cs, cw=cw)
                print("\n" + modelo.generar_reporte_texto())
                submenu_consultas(modelo)
            except ValueError as e:
                print(f"\n   [ERROR DE ESTABILIDAD]: {e}\n")

        elif opcion == "3":
            print_header("CONFIGURACIÓN DE MODELO M/D/1 (SERVICIO CONSTANTE)")
            lam = pedir_float("  • Tasa media de llegadas (λ): ", min_val=0.0)
            mu = pedir_float("  • Tasa media de servicio (μ): ", min_val=0.0)
            tu = input("  • Unidad de tiempo [por defecto 'minutos']: ").strip() or "minutos"
            cs, cq, cw = pedir_costos()

            try:
                modelo = ModeloMD1(lambda_rate=lam, mu_rate=mu, time_unit=tu, cq=cq, cs=cs, cw=cw)
                print("\n" + modelo.generar_reporte_texto())
                submenu_consultas(modelo)
            except ValueError as e:
                print(f"\n   [ERROR DE ESTABILIDAD]: {e}\n")

        elif opcion == "4":
            print_header("CONFIGURACIÓN DE MODELO M/G/1 (SERVICIO GENERAL)")
            lam = pedir_float("  • Tasa media de llegadas (λ): ", min_val=0.0)
            mu = pedir_float("  • Tasa media de servicio (μ): ", min_val=0.0)
            sigma = pedir_float("  • Desviación estándar del servicio (σ) [>= 0]: ", min_val=0.0, allow_zero=True)
            tu = input("  • Unidad de tiempo [por defecto 'minutos']: ").strip() or "minutos"
            cs, cq, cw = pedir_costos()

            try:
                modelo = ModeloMG1(lambda_rate=lam, mu_rate=mu, sigma=sigma, time_unit=tu, cq=cq, cs=cs, cw=cw)
                print("\n" + modelo.generar_reporte_texto())
                print(f"  • Factor de modificación Curry & Feldman: {modelo.factor_curry_feldman():.4f}")
                submenu_consultas(modelo)
            except ValueError as e:
                print(f"\n   [ERROR DE ESTABILIDAD]: {e}\n")

        elif opcion == "5":
            print_header("OPTIMIZACIÓN ECONÓMICA DE SERVIDORES (BÚSQUEDA DE s*)")
            lam = pedir_float("  • Tasa media de llegadas (λ): ", min_val=0.0)
            mu = pedir_float("  • Tasa media de servicio por servidor (μ): ", min_val=0.0)
            cs = pedir_float("  • Costo por servidor por unidad de tiempo (Cs): ", min_val=0.0)
            cq = pedir_float("  • Costo de espera por cliente en cola (Cq): ", min_val=0.0)
            cw = pedir_float("  • Costo por servicio por cliente (Cw) [opcional, def 0]: ", min_val=0.0, allow_zero=True)
            tu = input("  • Unidad de tiempo [por defecto 'horas']: ").strip() or "horas"

            try:
                resultado = optimizar_servidores(lambda_rate=lam, mu_rate=mu, cq=cq, cs=cs, cw=cw, time_unit=tu)
                print("\n" + resultado["reporte_texto"])
            except ValueError as e:
                print(f"\n   [ERROR]: {e}\n")

        elif opcion == "6":
            print_header("COMPARACIÓN MULTIMODELO LADO A LADO")
            lam = pedir_float("  • Tasa media de llegadas (λ): ", min_val=0.0)
            mu = pedir_float("  • Tasa media de servicio por servidor (μ): ", min_val=0.0)
            s = pedir_int("  • Número de servidores para M/M/s [def 1]: ", min_val=1)
            sigma = pedir_float("  • Desviación estándar σ para M/G/1: ", min_val=0.0, allow_zero=True)
            tu = input("  • Unidad de tiempo [por defecto 'minutos']: ").strip() or "minutos"
            cs, cq, cw = pedir_costos()

            try:
                res = comparar_modelos(lambda_rate=lam, mu_rate=mu, s=s, sigma=sigma, time_unit=tu, cq=cq, cs=cs, cw=cw)
                print("\n" + res["reporte_texto"])
            except ValueError as e:
                print(f"\n   [ERROR]: {e}\n")

        elif opcion == "7":
            print_header("EJECUTANDO PRUEBAS AUTOMATIZADAS DEL SISTEMA UNIFICADO")
            suite = unittest.TestLoader().loadTestsFromTestCase(TestSistemaColasUnificado)
            runner = unittest.TextTestRunner(verbosity=2)
            runner.run(suite)

        else:
            print("   [Error] Opción no válida.")

        input("\n  Presione ENTER para continuar...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  [Info] Programa interrumpido por el usuario. Saliendo...\n")
        sys.exit(0)
