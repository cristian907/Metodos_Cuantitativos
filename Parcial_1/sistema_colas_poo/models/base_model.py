#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clase base abstracta para modelos de teoría de colas.
Define los atributos comunes, validaciones de estabilidad, firma de métodos
y funciones genéricas de costos, reportes y probabilidades.
"""

import math
from typing import Dict, Any, List, Optional


class BaseQueueModel:
    """
    Clase base abstracta para todos los modelos de teoría de colas (POO).
    """
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
        Inicializa los parámetros del modelo de colas y realiza validaciones.

        Parámetros:
            lambda_rate (float): Tasa media de llegadas (λ > 0)
            mu_rate (float): Tasa media de servicio por servidor (μ > 0)
            s (int): Número de servidores en paralelo (s >= 1)
            time_unit (str): Nombre de la unidad de tiempo (ej. 'minutos', 'horas')
            cq (float): Costo de espera por cliente por unidad de tiempo (Cq >= 0)
            cs (float): Costo por servidor por unidad de tiempo (Cs >= 0)
            cw (float): Costo de atención/servicio por cliente (Cw >= 0)
        """
        if lambda_rate <= 0:
            raise ValueError("La tasa de llegada (λ) debe ser estrictamente positiva (> 0).")
        if mu_rate <= 0:
            raise ValueError("La tasa de servicio (μ) debe ser estrictamente positiva (> 0).")
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

    # --- MÉTODOS ABSTRACTOS / VIRTUALES A IMPLEMENTAR POR SUBCLASES ---

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

    # --- MÉTODOS CUMULATIVOS Y PROBABILIDADES DE TIEMPO ---

    def probabilidad_n_o_menos(self, n: int) -> float:
        """Calcula P(N <= n): Probabilidad de que haya n o menos clientes en el sistema."""
        if n < 0:
            return 0.0
        return sum(self.probabilidad_n(i) for i in range(n + 1))

    def probabilidad_mas_de_n(self, n: int) -> float:
        """Calcula P(N > n): Probabilidad de que haya más de n clientes en el sistema."""
        return 1.0 - self.probabilidad_n_o_menos(n)

    def prob_tiempo_cola_mayor_t(self, t: float) -> float:
        """
        Aproximación / Cálculo de P(Wq > t): Probabilidad de esperar en cola más de t.
        P(Wq > t) ≈ Pw * exp(-t / Wq)
        """
        if t < 0:
            return 1.0
        wq_val = self.wq()
        if wq_val <= 0:
            return 0.0
        return self.pw() * math.exp(-t / wq_val)

    def prob_tiempo_sistema_mayor_t(self, t: float) -> float:
        """
        Aproximación / Cálculo de P(Ws > t): Probabilidad de permanecer en sistema más de t.
        P(Ws > t) ≈ exp(-t / Ws)
        """
        if t < 0:
            return 1.0
        ws_val = self.ws()
        if ws_val <= 0:
            return 0.0
        return math.exp(-t / ws_val)

    # --- ANÁLISIS DE COSTOS ---

    def calcular_costos(self) -> Dict[str, float]:
        """
        Calcula la estructura de costos detallada por unidad de tiempo:
          - CS: Costo total de servidores (cs * s)
          - CQ / CTQ: Costo de espera en cola (cq * Lq)
          - C_espera_sistema: Costo de espera en el sistema (cq * Ls)
          - CTS_busy: Costo de servidores ocupados ( (cs + cq + cw) * r ) [WinQSB]
          - CTS_bar: Costo de servidores ociosos ( cs * (s - r) ) [WinQSB]
          - CTS: Costo total en el sistema (cq * Ls + cs * s + cw * r)
        """
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

    # --- RESUMEN Y REPORTES ---

    def resumen(self) -> Dict[str, Any]:
        """Genera un diccionario estructurado con los indicadores clave del modelo."""
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
