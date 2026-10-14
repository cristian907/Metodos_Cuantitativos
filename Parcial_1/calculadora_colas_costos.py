#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
MÓDULO DE RESOLUCIÓN DE MODELOS DE TEORÍA DE COLAS (M/M/1 y M/M/s) CON COSTOS - POO
===============================================================================
Universidad José Antonio Páez (UJAP) - Investigación de Operaciones / Métodos

Este módulo implementa una arquitectura orientada a objetos (POO) para resolver
sistemas de colas estocásticos de población infinita con análisis económico y de costos:
  - Modelo M/M/1: Servidor único (s = 1) con costos
  - Modelo M/M/s: Múltiples servidores en paralelo (s >= 1) con costos

Incluye:
  - Métricas de desempeño: Ls, Lq, Ws, Wq, Lb, Wb
  - Probabilidades de estado (P0, Pn, P(N <= n), P(N > n)), Erlang C (Pw)
  - Estructura de costos:
      • Cs: Costo por servidor por unidad de tiempo
      • Cq: Costo de espera por cliente en cola por unidad de tiempo
      • Cw: Costo de atención/servicio por cliente atendido
      • CTS: Costo de servidores ocupados
      • CTS_bar: Costo de servidores ociosos
      • CTQ: Costo de tiempo de espera en cola
      • CT_total: Costo total del sistema
  - Optimización económica: Búsqueda del número óptimo de servidores (s*) que minimiza el costo total.
===============================================================================
"""

import math
from typing import Dict, Any, Tuple, List, Optional


def parse_float(val_str: str) -> float:
    """Convierte una cadena a float aceptando comas o puntos decimales."""
    return float(val_str.strip().replace(',', '.'))


class BaseQueueModel:
    """
    Clase base abstracta para modelos de teoría de colas con análisis de costos.
    Define los atributos comunes, validaciones de estabilidad y firma de métodos.
    """
    def __init__(
        self,
        lambda_rate: float,
        mu_rate: float,
        s: int = 1,
        time_unit: str = "unidad de tiempo",
        cq: float = 0.0,
        cs: float = 0.0,
        cw: float = 0.0
    ):
        """
        Inicializa el modelo de colas.

        Parámetros:
            lambda_rate (float): Tasa media de llegadas (λ)
            mu_rate (float): Tasa media de servicio por servidor (μ)
            s (int): Número de servidores en paralelo (s >= 1)
            time_unit (str): Nombre de la unidad de tiempo (ej. 'minutos', 'horas')
            cq (float): Costo de espera por cliente en cola por unidad de tiempo (Cq)
            cs (float): Costo por servidor por unidad de tiempo (Cs)
            cw (float): Costo por servicio/atención por cliente (Cw)
        """
        if lambda_rate <= 0:
            raise ValueError("La tasa de llegada (λ) debe ser estrictamente positiva (> 0).")
        if mu_rate <= 0:
            raise ValueError("La tasa de servicio (μ) debe ser estrictamente positiva (> 0).")
        if s < 1:
            raise ValueError("El número de servidores (s) debe ser un entero positivo (>= 1).")
        if cq < 0 or cs < 0 or cw < 0:
            raise ValueError("Los costos (Cq, Cs, Cw) no pueden ser negativos.")

        self.lambda_rate = float(lambda_rate)
        self.mu_rate = float(mu_rate)
        self.s = int(s)
        self.time_unit = time_unit
        self.cq = float(cq)
        self.cs = float(cs)
        self.cw = float(cw)

        # Factor de utilización por servidor ρ = λ / (s * μ)
        self.rho = self.lambda_rate / (self.s * self.mu_rate)
        self.r = self.lambda_rate / self.mu_rate  # Intensidad de tráfico (r = λ/μ)

        # Validación de estado estable
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
        """Probabilidad de que un cliente entrante tenga que esperar P(N >= s)."""
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

    def calcular_costos(self) -> Dict[str, float]:
        """
        Calcula la estructura de costos detallada por unidad de tiempo:
          - CS: Costo total de servicio de servidores (cs * s)
          - CQ / CTQ: Costo de espera en cola (cq * Lq)
          - C_espera_sistema: Costo de espera en sistema (cq * Ls)
          - CTS: Costo Total en Sistema (cq * Ls + cs * s + cw * r) [Formato Ejercicio]
          - CTS_busy: Costo de servidores ocupados ( (cs + cq + cw) * r ) [Formato WinQSB]
          - CTS_bar: Costo de servidores ociosos ( cs * (s - r) ) [Formato WinQSB]
        """
        lq_val = self.lq()
        ls_val = self.ls()

        # Servidores ocupados esperados = Ls - Lq = r
        servidores_ocupados = ls_val - lq_val
        servidores_ociosos = float(self.s) - servidores_ocupados

        cs_total = self.cs * self.s
        cq_cola = self.cq * lq_val
        cq_sistema = self.cq * ls_val
        cw_atencion = self.cw * servidores_ocupados

        # Coste total en el sistema = cq * Ls + cs * s + cw * r
        cts_total = cq_sistema + cs_total + cw_atencion

        # Desglose formato WinQSB:
        # En WinQSB, el costo de servidor ocupado incluye la tarifa de espera del cliente atendiéndose: cb = cs + cq + cw
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
        """Genera un diccionario estructurado con todos los indicadores clave del modelo."""
        return {
            "modelo": f"M/M/{self.s}",
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
            "cs": self.cs,
            "cq": self.cq,
            "cw": self.cw,
            "time_unit": self.time_unit
        }

    def generar_reporte_texto(
        self,
        consultas_adicionales: Optional[List[str]] = None,
        k_max_tabla: Optional[int] = None
    ) -> str:
        """
        Genera una representación en formato texto plano completa con todos los
        resultados del modelo, costos y las consultas adicionales del menú secundario.
        """
        r = self.resumen()
        lineas = []
        lineas.append("=" * 72)
        lineas.append(f"       REPORTE DE TEORÍA DE COLAS CON COSTOS - MODELO {r['modelo']}")
        lineas.append("=" * 72)
        lineas.append(f"  Tasa de llegada (λ)                   : {r['lambda']:.4f} clientes / {self.time_unit}")
        lineas.append(f"  Tasa de servicio por servidor (μ)    : {r['mu']:.4f} clientes / {self.time_unit}")
        lineas.append(f"  Número de servidores (s)              : {r['s']}")
        lineas.append("-" * 72)
        lineas.append(f"  Factor de utilización por servidor (ρ): {r['rho']:.4f} ({r['utilizacion_pct']:.2f}%)")
        lineas.append(f"  Intensidad de tráfico (r = λ/μ)       : {r['r']:.4f} erlangs")
        lineas.append(f"  Probabilidad de sistema vacío (P0)    : {r['p0']:.4f} ({r['p0_pct']:.2f}%)")
        lineas.append(f"  Probabilidad de tener que esperar (Pw): {r['pw']:.4f} ({r['pw_pct']:.2f}%)")
        lineas.append("-" * 72)
        lineas.append(f"  N° medio de clientes en el sistema (Ls): {r['ls']:.4f} clientes")
        lineas.append(f"  N° medio de clientes en la cola (Lq)  : {r['lq']:.4f} clientes")
        lineas.append("-" * 72)
        lineas.append(f"  Tiempo medio en el sistema (Ws)       : {r['ws']:.4f} {self.time_unit}")
        lineas.append(f"  Tiempo medio de espera en cola (Wq)   : {r['wq']:.4f} {self.time_unit}")
        lineas.append("=" * 72)

        # Sección de costos
        c = self.calcular_costos()
        lineas.append("")
        lineas.append("-" * 72)
        lineas.append("                   ESTRUCTURA DE COSTOS DEL SISTEMA")
        lineas.append("-" * 72)
        lineas.append(f"  Costo por servidor (Cs)               : {self.cs:.4f} / {self.time_unit}")
        lineas.append(f"  Costo de espera en cola (Cq)          : {self.cq:.4f} / cliente · {self.time_unit}")
        lineas.append(f"  Costo de atención por cliente (Cw)    : {self.cw:.4f} / cliente")
        lineas.append("-" * 72)
        lineas.append(f"  Servidores ocupados esperados (r)     : {c['servidores_ocupados']:.4f}")
        lineas.append(f"  Servidores ociosos esperados (s - r)  : {c['servidores_ociosos']:.4f}")
        lineas.append("-" * 72)
        lineas.append(f"  Coste total de servicio (CS = cs·s)   : {c['CS']:.4f} / {self.time_unit}")
        lineas.append(f"  Coste total espera en cola (CQ=cq·Lq) : {c['CQ']:.4f} / {self.time_unit}")
        lineas.append(f"  Coste total espera sistema (cq·Ls)    : {c['C_espera_sistema']:.4f} / {self.time_unit}")
        lineas.append("-" * 72)
        lineas.append("  Desglose Formato WinQSB / Teórico:")
        lineas.append(f"    • Costo serv. ocupados (CTS_busy)   : {c['CTS_busy']:.4f} / {self.time_unit}")
        lineas.append(f"    • Costo serv. ociosos (CTS_bar)     : {c['CTS_bar']:.4f} / {self.time_unit}")
        lineas.append(f"    • Costo espera en cola (CTQ)        : {c['CTQ']:.4f} / {self.time_unit}")
        lineas.append("-" * 72)
        lineas.append(f"  COSTE TOTAL EN SISTEMA (CTS)          : {c['CTS']:.4f} / {self.time_unit}")
        lineas.append("=" * 72)

        # Consultas adicionales realizadas del menú secundario
        if consultas_adicionales:
            lineas.append("")
            lineas.append("-" * 72)
            lineas.append("        RESULTADOS ADICIONALES (CONSULTAS DEL MENÚ SECUNDARIO)")
            lineas.append("-" * 72)
            for item in consultas_adicionales:
                lineas.append(f"  • {item}")
            lineas.append("=" * 72)

        # Tabla de probabilidades Pn si se especificó K
        if k_max_tabla is not None and k_max_tabla >= 0:
            lineas.append("")
            lineas.append("-" * 72)
            lineas.append(f"        TABLA DE PROBABILIDADES DE ESTADO Pn (n = 0 a {k_max_tabla})")
            lineas.append("-" * 72)
            lineas.append(f"  {'n':<5} | {'P(N = n)':<12} | {'P(N <= n)':<12} | {'P(N > n)':<12}")
            lineas.append("  " + "-" * 48)
            for i in range(k_max_tabla + 1):
                pn = self.probabilidad_n(i)
                pc = self.probabilidad_n_o_menos(i)
                pg = self.probabilidad_mas_de_n(i)
                lineas.append(f"  {i:<5} | {pn:<12.6f} | {pc:<12.6f} | {pg:<12.6f}")
            lineas.append("=" * 72)

        return "\n".join(lineas)

    def exportar_txt(
        self,
        filepath: str = "resultados_cola_costos.txt",
        consultas_adicionales: Optional[List[str]] = None,
        k_max_tabla: Optional[int] = None
    ) -> str:
        """Guarda el reporte completo en un archivo .txt en el sistema de archivos."""
        contenido = self.generar_reporte_texto(consultas_adicionales, k_max_tabla)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(contenido)
        return filepath

    def mostrar_reporte(self):
        """Imprime un reporte analítico detallado y visualmente refinado por consola."""
        print(self.generar_reporte_texto())


class ModeloMMs(BaseQueueModel):
    """
    Representa un modelo de colas M/M/s con múltiples servidores en paralelo y costos.
    Válido para s >= 1.
    """
    def p0(self) -> float:
        """
        Calcula P0 (Probabilidad de que el sistema esté vacío).
        P0 = 1 / [ sum_{n=0}^{s-1} (r^n / n!) + (r^s / (s! * (1 - ρ))) ]
        """
        r = self.r
        sumatoria = sum([(r ** n) / math.factorial(n) for n in range(self.s)])
        ultimo_termino = (r ** self.s) / (math.factorial(self.s) * (1.0 - self.rho))
        return 1.0 / (sumatoria + ultimo_termino)

    def pw(self) -> float:
        """
        Probabilidad de que un cliente que llega deba esperar (Fórmula Erlang C).
        Pw = P(N >= s) = (r^s / (s! * (1 - ρ))) * P0
        """
        r = self.r
        p0_val = self.p0()
        return ((r ** self.s) / (math.factorial(self.s) * (1.0 - self.rho))) * p0_val

    def lq(self) -> float:
        """
        Número promedio de clientes en la cola (Lq).
        Lq = (Pw * ρ) / (1 - ρ)
        """
        return (self.pw() * self.rho) / (1.0 - self.rho)

    def ls(self) -> float:
        """
        Número promedio de clientes en el sistema (Ls).
        Ls = Lq + (λ / μ)
        """
        return self.lq() + self.r

    def wq(self) -> float:
        """
        Tiempo promedio de espera en la cola (Wq).
        Wq = Lq / λ
        """
        return self.lq() / self.lambda_rate

    def ws(self) -> float:
        """
        Tiempo promedio en el sistema (Ws).
        Ws = Wq + 1 / μ = Ls / λ
        """
        return self.ls() / self.lambda_rate

    def lb(self) -> float:
        """Número promedio de clientes en cola para un sistema ocupado (Lb = Lq / Pw)."""
        pw_val = self.pw()
        return self.lq() / pw_val if pw_val > 0 else 0.0

    def wb(self) -> float:
        """Tiempo promedio en cola para un sistema ocupado (Wb = Wq / Pw)."""
        pw_val = self.pw()
        return self.wq() / pw_val if pw_val > 0 else 0.0

    def probabilidad_n(self, n: int) -> float:
        """Calcula Pn: Probabilidad de que haya exactamente n clientes en el sistema."""
        if n < 0:
            return 0.0
        p0_val = self.p0()
        r = self.r
        if n <= self.s:
            return ((r ** n) / math.factorial(n)) * p0_val
        else:
            denom = math.factorial(self.s) * (self.s ** (n - self.s))
            return ((r ** n) / denom) * p0_val

    def probabilidad_n_o_menos(self, n: int) -> float:
        """Calcula P(N <= n): Probabilidad de que haya n o menos clientes en el sistema."""
        if n < 0:
            return 0.0
        return sum(self.probabilidad_n(i) for i in range(n + 1))

    def probabilidad_mas_de_n(self, n: int) -> float:
        """Calcula P(N > n): Probabilidad de que haya más de n clientes en el sistema."""
        return 1.0 - self.probabilidad_n_o_menos(n)

    def prob_tiempo_cola_mayor_t(self, t: float) -> float:
        """Calcula P(Wq > t): Probabilidad de esperar en cola más de un tiempo t."""
        if t < 0:
            return 1.0
        exponent = -self.s * self.mu_rate * (1.0 - self.rho) * t
        return self.pw() * math.exp(exponent)

    def prob_tiempo_sistema_mayor_t(self, t: float) -> float:
        """Calcula P(Ws > t): Probabilidad de permanecer en el sistema más de un tiempo t."""
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
    Especialización de ModeloMMs para un único servidor (s = 1) con costos.
    Aprovecha fórmulas simplificadas cerradas directas para M/M/1.
    """
    def __init__(
        self,
        lambda_rate: float,
        mu_rate: float,
        time_unit: str = "unidad de tiempo",
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

    def p0(self) -> float:
        """P0 = 1 - ρ"""
        return 1.0 - self.rho

    def pw(self) -> float:
        """En M/M/1, la probabilidad de esperar es igual a la ocupación ρ."""
        return self.rho

    def lq(self) -> float:
        """Lq = λ^2 / (μ * (μ - λ)) = ρ^2 / (1 - ρ)"""
        return (self.lambda_rate ** 2) / (self.mu_rate * (self.mu_rate - self.lambda_rate))

    def ls(self) -> float:
        """Ls = λ / (μ - λ) = ρ / (1 - ρ)"""
        return self.lambda_rate / (self.mu_rate - self.lambda_rate)

    def wq(self) -> float:
        """Wq = λ / (μ * (μ - λ)) = ρ / (μ - λ)"""
        return self.lambda_rate / (self.mu_rate * (self.mu_rate - self.lambda_rate))

    def ws(self) -> float:
        """Ws = 1 / (μ - λ)"""
        return 1.0 / (self.mu_rate - self.lambda_rate)

    def probabilidad_n(self, n: int) -> float:
        """Pn = (1 - ρ) * ρ^n"""
        if n < 0:
            return 0.0
        return (1.0 - self.rho) * (self.rho ** n)

    def probabilidad_n_o_menos(self, n: int) -> float:
        """P(N <= n) = 1 - ρ^(n+1)"""
        if n < 0:
            return 0.0
        return 1.0 - (self.rho ** (n + 1))

    def probabilidad_mas_de_n(self, n: int) -> float:
        """P(N > n) = ρ^(n+1)"""
        if n < 0:
            return 1.0
        return self.rho ** (n + 1)

    def prob_tiempo_sistema_mayor_t(self, t: float) -> float:
        """P(Ws > t) = exp(-μ * (1 - ρ) * t)"""
        if t < 0:
            return 1.0
        return math.exp(-self.mu_rate * (1.0 - self.rho) * t)

    def prob_tiempo_cola_mayor_t(self, t: float) -> float:
        """P(Wq > t) = ρ * exp(-μ * (1 - ρ) * t)"""
        if t < 0:
            return 1.0
        return self.rho * math.exp(-self.mu_rate * (1.0 - self.rho) * t)


def optimizar_servidores_mms(
    lambda_rate: float,
    mu_rate: float,
    cs: float,
    cq: float,
    cw: float = 0.0,
    s_max: int = 30,
    time_unit: str = "unidad de tiempo"
) -> Tuple[int, float, List[Dict[str, Any]]]:
    """
    Encuentra el número óptimo de servidores s* que minimiza el Costo Total (CT).
    Evalúa s desde s_min = floor(λ/μ) + 1 hasta s_max.
    """
    s_min = math.floor(lambda_rate / mu_rate) + 1
    if s_max < s_min:
        s_max = s_min + 10

    tabla_costos = []
    mejor_s = s_min
    menor_costo = float('inf')

    for s in range(s_min, s_max + 1):
        m = ModeloMMs(lambda_rate=lambda_rate, mu_rate=mu_rate, s=s, time_unit=time_unit, cq=cq, cs=cs, cw=cw)
        c = m.calcular_costos()
        ct = c["CT_total"]
        tabla_costos.append({
            "s": s,
            "rho": m.rho,
            "lq": m.lq(),
            "ls": m.ls(),
            "wq": m.wq(),
            "ws": m.ws(),
            "ct_total": ct,
            "costo_servidores": c["costo_servidores_totales"],
            "costo_espera": c["CTQ"]
        })
        if ct < menor_costo:
            menor_costo = ct
            mejor_s = s

    return mejor_s, menor_costo, tabla_costos


# =============================================================================
# MENÚ INTERACTIVO CON INTEGRACIÓN COMPLETA DE COSTOS
# =============================================================================

def menu_interactivo():
    """Ejecuta una interfaz interactiva de consola con opciones completas de costos."""
    while True:
        print("\n" + "=" * 72)
        print("   MENÚ PRINCIPAL - CALCULADORA DE COLAS CON COSTOS (M/M/1 y M/M/s)")
        print("=======================================================================")
        print("  1. Resolver Modelo M/M/1 (Servidor Único con Costos)")
        print("  2. Resolver Modelo M/M/s (Múltiples Servidores con Costos)")
        print("  3. Optimización Económica: Encontrar N° óptimo de servidores (s*)")
        print("  0. Salir")
        print("=" * 72)

        opc = input(" Seleccione una opción [0-3]: ").strip()

        if opc == "1":
            try:
                print("\n--- INGRESO DE DATOS MODELO M/M/1 ---")
                lam = parse_float(input(" Ingrese la tasa de llegada (λ): "))
                mu = parse_float(input(" Ingrese la tasa de servicio (μ): "))
                unidad = input(" Ingrese la unidad de tiempo [minutos]: ").strip() or "minutos"

                print("\n--- INGRESO DE COSTOS DEL SISTEMA ---")
                cs = parse_float(input(" Costo por servidor por unidad de tiempo (Cs) [0.0]: ") or "0")
                cq = parse_float(input(" Costo de espera por cliente en cola por unidad de tiempo (Cq) [0.0]: ") or "0")
                cw = parse_float(input(" Costo de atención por cliente atendido (Cw) [0.0]: ") or "0")

                modelo = ModeloMM1(lambda_rate=lam, mu_rate=mu, time_unit=unidad, cq=cq, cs=cs, cw=cw)
                modelo.mostrar_reporte()

                _menu_secundario_costos(modelo)
            except ValueError as e:
                print(f"\n [ERROR]: {e}")

        elif opc == "2":
            try:
                print("\n--- INGRESO DE DATOS MODELO M/M/s ---")
                lam = parse_float(input(" Ingrese la tasa de llegada (λ): "))
                mu = parse_float(input(" Ingrese la tasa de servicio por servidor (μ): "))
                s = int(parse_float(input(" Ingrese el número de servidores (s): ")))
                unidad = input(" Ingrese la unidad de tiempo [horas]: ").strip() or "horas"

                print("\n--- INGRESO DE COSTOS DEL SISTEMA ---")
                cs = parse_float(input(" Costo por servidor por unidad de tiempo (Cs) [0.0]: ") or "0")
                cq = parse_float(input(" Costo de espera por cliente en cola por unidad de tiempo (Cq) [0.0]: ") or "0")
                cw = parse_float(input(" Costo de atención por cliente atendido (Cw) [0.0]: ") or "0")

                modelo = ModeloMMs(lambda_rate=lam, mu_rate=mu, s=s, time_unit=unidad, cq=cq, cs=cs, cw=cw)
                modelo.mostrar_reporte()

                _menu_secundario_costos(modelo)
            except ValueError as e:
                print(f"\n [ERROR]: {e}")

        elif opc == "3":
            try:
                print("\n--- OPTIMIZACIÓN ECONÓMICA DE SERVIDORES (s*) ---")
                lam = parse_float(input(" Ingrese la tasa de llegada (λ): "))
                mu = parse_float(input(" Ingrese la tasa de servicio por servidor (μ): "))
                cs = parse_float(input(" Costo por servidor por unidad de tiempo (Cs): "))
                cq = parse_float(input(" Costo de espera por cliente en cola por unidad de tiempo (Cq): "))
                cw = parse_float(input(" Costo por servicio por cliente (Cw) [0.0]: ") or "0")
                unidad = input(" Ingrese la unidad de tiempo [horas]: ").strip() or "horas"
                s_max = int(parse_float(input(" Ingrese s máximo a evaluar [20]: ") or "20"))

                mejor_s, menor_costo, tabla = optimizar_servidores_mms(lam, mu, cs, cq, cw, s_max, unidad)

                print("\n" + "=" * 76)
                print("   TABLA DE COMPARACIÓN ECONÓMICA DE SERVIDORES")
                print("=" * 76)
                print(f"  {'s':<4} | {'ρ':<7} | {'Lq':<8} | {'Ls':<8} | {'C. Servidores':<14} | {'C. Espera':<12} | {'CT Total':<12}")
                print("  " + "-" * 72)
                for fila in tabla:
                    marca = " <-- OPTIMO" if fila["s"] == mejor_s else ""
                    print(
                        f"  {fila['s']:<4} | {fila['rho']:<7.4f} | {fila['lq']:<8.4f} | {fila['ls']:<8.4f} | "
                        f"{fila['costo_servidores']:<14.4f} | {fila['costo_espera']:<12.4f} | {fila['ct_total']:<12.4f}{marca}"
                    )
                print("=" * 76)
                print(f"  >>> EL NÚMERO ÓPTIMO DE SERVIDORES ES s* = {mejor_s} CON COSTO MÍNIMO CT = {menor_costo:.4f} / {unidad}")

                exp = input("\n  ¿Desea crear el modelo M/M/s* óptimo y ver reporte completo? (s/n) [s]: ").strip().lower()
                if exp in ['', 's', 'si', 'sí', 'y', 'yes']:
                    mod_opt = ModeloMMs(lambda_rate=lam, mu_rate=mu, s=mejor_s, time_unit=unidad, cq=cq, cs=cs, cw=cw)
                    mod_opt.mostrar_reporte()
                    _menu_secundario_costos(mod_opt)

            except ValueError as e:
                print(f"\n [ERROR]: {e}")

        elif opc == "0":
            print("\n ¡Gracias por utilizar la calculadora de teoría de colas con costos!")
            break
        else:
            print(" Opción no válida. Por favor, intente de nuevo.")


def _menu_secundario_costos(modelo: BaseQueueModel):
    """Menú secundario para consultar probabilidades, modificar costos y exportar a TXT."""
    consultas_adicionales = []
    k_max_tabla = None

    while True:
        print("\n" + "-" * 60)
        print("     CONSULTAS DE PROBABILIDADES, TIEMPOS Y COSTOS")
        print("-" * 60)
        print("  1. Probabilidad de exactamente 'n' clientes P(N = n)")
        print("  2. Probabilidad de 'n' o menos clientes P(N <= n)")
        print("  3. Probabilidad de más de 'n' clientes P(N > n)")
        print("  4. Probabilidad de tiempo en sistema mayor a 't' P(Ws > t)")
        print("  5. Probabilidad de tiempo en cola mayor a 't' P(Wq > t)")
        print("  6. Mostrar tabla de probabilidades Pn (hasta K)")
        print("  7. Modificar / Actualizar Costos (Cs, Cq, Cw)")
        print("  8. Exportar todos los resultados a un archivo TXT (.txt)")
        print("  0. Volver al menú principal")
        print("-" * 60)

        sub_opc = input(" Seleccione una opción (0-8): ").strip()

        if sub_opc == "1":
            try:
                n = int(parse_float(input("  Ingrese n: ")))
                pn = modelo.probabilidad_n(n)
                res_str = f"P(N = {n}) = {pn:.6f} ({pn * 100:.2f}%)"
                print(f"\n  >> {res_str}")
                consultas_adicionales.append(res_str)
            except ValueError as e:
                print(f"  [ERROR]: {e}")

        elif sub_opc == "2":
            try:
                n = int(parse_float(input("  Ingrese n: ")))
                p_cum = modelo.probabilidad_n_o_menos(n)
                res_str = f"P(N <= {n}) = {p_cum:.6f} ({p_cum * 100:.2f}%)"
                print(f"\n  >> {res_str}")
                consultas_adicionales.append(res_str)
            except ValueError as e:
                print(f"  [ERROR]: {e}")

        elif sub_opc == "3":
            try:
                n = int(parse_float(input("  Ingrese n: ")))
                p_gt = modelo.probabilidad_mas_de_n(n)
                res_str = f"P(N > {n}) = {p_gt:.6f} ({p_gt * 100:.2f}%)"
                print(f"\n  >> {res_str}")
                consultas_adicionales.append(res_str)
            except ValueError as e:
                print(f"  [ERROR]: {e}")

        elif sub_opc == "4":
            try:
                t = parse_float(input(f"  Ingrese tiempo t (en {modelo.time_unit}): "))
                p_ws = modelo.prob_tiempo_sistema_mayor_t(t)
                res_str = f"P(Ws > {t}) = {p_ws:.6f} ({p_ws * 100:.2f}%)"
                print(f"\n  >> {res_str}")
                consultas_adicionales.append(res_str)
            except ValueError as e:
                print(f"  [ERROR]: {e}")

        elif sub_opc == "5":
            try:
                t = parse_float(input(f"  Ingrese tiempo t (en {modelo.time_unit}): "))
                p_wq = modelo.prob_tiempo_cola_mayor_t(t)
                res_str = f"P(Wq > {t}) = {p_wq:.6f} ({p_wq * 100:.2f}%)"
                print(f"\n  >> {res_str}")
                consultas_adicionales.append(res_str)
            except ValueError as e:
                print(f"  [ERROR]: {e}")

        elif sub_opc == "6":
            try:
                k_max = int(parse_float(input("  Ingrese K máximo: ")))
                k_max_tabla = k_max
                _mostrar_tabla_pn(modelo, k_max)
            except ValueError as e:
                print(f"  [ERROR]: {e}")

        elif sub_opc == "7":
            try:
                modelo.cs = parse_float(input(f"  Nuevo costo por servidor (Cs) [{modelo.cs}]: ") or str(modelo.cs))
                modelo.cq = parse_float(input(f"  Nuevo costo de espera en cola (Cq) [{modelo.cq}]: ") or str(modelo.cq))
                modelo.cw = parse_float(input(f"  Nuevo costo de atención (Cw) [{modelo.cw}]: ") or str(modelo.cw))
                print("\n  [ÉXITO]: Costos actualizados. Nuevo reporte:")
                modelo.mostrar_reporte()
            except ValueError as e:
                print(f"  [ERROR]: {e}")

        elif sub_opc == "8":
            try:
                nombre_archivo = input("  Ingrese el nombre del archivo TXT [resultados_cola_costos.txt]: ").strip()
                if not nombre_archivo:
                    nombre_archivo = "resultados_cola_costos.txt"
                if not nombre_archivo.endswith(".txt"):
                    nombre_archivo += ".txt"

                if k_max_tabla is None:
                    incluir = input("  ¿Desea incluir la tabla de probabilidades Pn en el archivo? (s/n) [n]: ").strip().lower()
                    if incluir in ['s', 'si', 'sí', 'y', 'yes']:
                        k_max_tabla = int(parse_float(input("    Ingrese K máximo para la tabla Pn [10]: ") or "10"))

                path_guardado = modelo.exportar_txt(
                    filepath=nombre_archivo,
                    consultas_adicionales=consultas_adicionales,
                    k_max_tabla=k_max_tabla
                )
                print(f"\n  [ÉXITO]: Reporte exportado correctamente a '{path_guardado}'.")
            except Exception as e:
                print(f"  [ERROR al exportar TXT]: {e}")

        elif sub_opc == "0":
            break
        else:
            print("  Opción no válida.")


def _mostrar_tabla_pn(modelo: BaseQueueModel, k_max: int):
    """Muestra una tabla ordenada de probabilidades Pn, P(<=n), P(>n)."""
    print(f"\n  {'n':<5} | {'P(N = n)':<12} | {'P(N <= n)':<12} | {'P(N > n)':<12}")
    print("  " + "-" * 48)
    for i in range(k_max + 1):
        pn = modelo.probabilidad_n(i)
        pc = modelo.probabilidad_n_o_menos(i)
        pg = modelo.probabilidad_mas_de_n(i)
        print(f"  {i:<5} | {pn:<12.6f} | {pc:<12.6f} | {pg:<12.6f}")


if __name__ == "__main__":
    menu_interactivo()
