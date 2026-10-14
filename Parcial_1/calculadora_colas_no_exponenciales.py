#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
MÓDULO DE RESOLUCIÓN DE MODELOS DE TEORÍA DE COLAS NO EXPONENCIALES, 
INTERRUPCIONES Y PRIORIDADES - POO
===============================================================================
Universidad José Antonio Páez (UJAP) - Investigación de Operaciones / Métodos

Este módulo implementa una arquitectura orientada a objetos (POO) para resolver
modelos avanzadas de teoría de colas con tiempos de servicio no exponenciales,
sistemas de pérdidas, modelos con prioridades e interrupciones del servidor:
  1. Modelo G/G/s (Aproximación de Allen-Cunneen)
  2. Modelo M/G/1 (Fórmula de Pollaczek-Khinchine y ajuste de Curry-Feldman)
  3. Modelo M/D/1 (Servicio constante, 1 servidor)
  4. Modelo M/D/s (Servicio constante, s servidores)
  5. Modelo M/E_k/1 (Servicio con distribución Erlang-k, 1 servidor)
  6. Modelo Erlang-B / M/G/s/s (Líneas telefónicas / Sistema de pérdidas sin cola)
  7. Modelo de Prioridades Sin Interrupción (Non-Preemptive Priority)
  8. Modelo de Prioridades Con Interrupción (Preemptive Priority)
  9. Modelo con Interrupciones / Fallas del Servidor (Server Breakdown and Repair)

Incluye la resolución completa del problema del traslado de empleados en el aeropuerto.
===============================================================================
"""

import math
from typing import Dict, Any, List, Optional, Tuple


def parse_float(val_str: str) -> float:
    """Convierte una cadena a float aceptando comas o puntos decimales."""
    return float(val_str.strip().replace(',', '.'))


class BaseNonExponentialModel:
    """
    Clase base abstracta para modelos no exponenciales de teoría de colas.
    Define atributos comunes, cálculos de costos y exportación de reportes.
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
        if lambda_rate <= 0:
            raise ValueError("La tasa de llegada (λ) debe ser positiva (> 0).")
        if mu_rate <= 0:
            raise ValueError("La tasa de servicio (μ) debe ser positiva (> 0).")
        if s < 1:
            raise ValueError("El número de servidores (s) debe ser >= 1.")

        self.lambda_rate = float(lambda_rate)
        self.mu_rate = float(mu_rate)
        self.s = int(s)
        self.time_unit = time_unit
        self.cq = float(cq)
        self.cs = float(cs)
        self.cw = float(cw)

        self.rho = self.lambda_rate / (self.s * self.mu_rate)
        self.r = self.lambda_rate / self.mu_rate

    def p0(self) -> float:
        """Probabilidad de que el sistema esté vacío (P0)."""
        raise NotImplementedError

    def pw(self) -> float:
        """Probabilidad de espera en cola (Pw)."""
        raise NotImplementedError

    def lq(self) -> float:
        """Número promedio de clientes en cola (Lq)."""
        raise NotImplementedError

    def ls(self) -> float:
        """Número promedio de clientes en el sistema (Ls)."""
        raise NotImplementedError

    def wq(self) -> float:
        """Tiempo promedio de espera en cola (Wq)."""
        raise NotImplementedError

    def ws(self) -> float:
        """Tiempo promedio en el sistema (Ws)."""
        raise NotImplementedError

    def probabilidad_n(self, n: int) -> float:
        """
        Aproximación de la probabilidad Pn de tener n clientes en el sistema.
        Pn ≈ (1 - ρ) * ρ^n
        """
        if n < 0:
            return 0.0
        return (1.0 - self.rho) * (self.rho ** n)

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
        Aproximación de P(Wq > t): Probabilidad de esperar en cola más de un tiempo t.
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
        Aproximación de P(Ws > t): Probabilidad de permanencia en sistema mayor a t.
        P(Ws > t) ≈ exp(-t / Ws)
        """
        if t < 0:
            return 1.0
        ws_val = self.ws()
        if ws_val <= 0:
            return 0.0
        return math.exp(-t / ws_val)

    def calcular_costos(self) -> Dict[str, float]:
        """Calcula los costos del sistema por unidad de tiempo."""
        lq_val = self.lq()
        ls_val = self.ls()
        servidores_ocupados = ls_val - lq_val
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
        """Genera un diccionario estructurado con las métricas clave."""
        return {
            "modelo": self.__class__.__name__,
            "lambda": self.lambda_rate,
            "mu": self.mu_rate,
            "s": self.s,
            "rho": self.rho,
            "utilizacion_pct": self.rho * 100.0,
            "r": self.r,
            "p0": self.p0(),
            "pw": self.pw(),
            "ls": self.ls(),
            "lq": self.lq(),
            "ws": self.ws(),
            "wq": self.wq(),
            "time_unit": self.time_unit
        }

    def generar_reporte_texto(
        self,
        notas_adicionales: Optional[List[str]] = None,
        consultas_adicionales: Optional[List[str]] = None,
        k_max_tabla: Optional[int] = None
    ) -> str:
        """Genera un reporte detallado en texto plano."""
        r = self.resumen()
        lineas = [
            "=" * 72,
            f"       REPORTE DE TEORÍA DE COLAS NO EXPONENCIAL - MODELO {r['modelo']}",
            "=" * 72,
            f"  Tasa de llegada (λ)                   : {r['lambda']:.6f} clientes / {self.time_unit}",
            f"  Tasa de servicio por servidor (μ)    : {r['mu']:.6f} clientes / {self.time_unit}",
            f"  Número de servidores (s)              : {r['s']}",
            "-" * 72,
            f"  Factor de utilización por servidor (ρ): {r['rho']:.4f} ({r['utilizacion_pct']:.2f}%)",
            f"  Intensidad de tráfico (r = λ/μ)       : {r['r']:.4f} erlangs",
            f"  Probabilidad de sistema vacío (P0)    : {r['p0']:.4f} ({r['p0'] * 100:.2f}%)",
            f"  Probabilidad de tener que esperar (Pw): {r['pw']:.4f} ({r['pw'] * 100:.2f}%)",
            "-" * 72,
            f"  N° medio de clientes en el sistema (Ls): {r['ls']:.4f} clientes",
            f"  N° medio de clientes en la cola (Lq)  : {r['lq']:.4f} clientes",
            "-" * 72,
            f"  Tiempo medio en el sistema (Ws)       : {r['ws']:.4f} {self.time_unit}",
            f"  Tiempo medio de espera en cola (Wq)   : {r['wq']:.4f} {self.time_unit}",
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

        if notas_adicionales or consultas_adicionales:
            lineas.extend([
                "",
                "-" * 72,
                "                       NOTAS Y RESULTADOS ADICIONALES",
                "-" * 72
            ])
            if notas_adicionales:
                for nota in notas_adicionales:
                    lineas.append(f"  • {nota}")
            if consultas_adicionales:
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
        filepath: str = "reporte_cola_no_exponencial.txt",
        notas_adicionales: Optional[List[str]] = None,
        consultas_adicionales: Optional[List[str]] = None,
        k_max_tabla: Optional[int] = None
    ) -> str:
        """Exporta el reporte a un archivo .txt."""
        contenido = self.generar_reporte_texto(
            notas_adicionales=notas_adicionales,
            consultas_adicionales=consultas_adicionales,
            k_max_tabla=k_max_tabla
        )
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(contenido)
        return filepath

    def mostrar_reporte(self):
        """Muestra el reporte por consola."""
        print(self.generar_reporte_texto())


class ModeloGGs(BaseNonExponentialModel):
    """
    Modelo de Cola General G/G/s (Aproximación de Allen-Cunneen).
    - Tiempos de llegada con media 1/λ y desviación típica σ_λ
    - Tiempos de servicio con media 1/μ y desviación típica σ_μ
    - s servidores en paralelo
    """
    def __init__(
        self,
        lambda_rate: float,
        sigma_lambda: float,
        mu_rate: float,
        sigma_mu: float,
        s: int = 1,
        time_unit: str = "minutos",
        cq: float = 0.0,
        cs: float = 0.0,
        cw: float = 0.0
    ):
        super().__init__(lambda_rate, mu_rate, s, time_unit, cq, cs, cw)
        if sigma_lambda < 0:
            raise ValueError("σ_λ no puede ser negativa.")
        if sigma_mu < 0:
            raise ValueError("σ_μ no puede ser negativa.")

        self.sigma_lambda = float(sigma_lambda)
        self.sigma_mu = float(sigma_mu)

        if self.rho >= 1.0:
            raise ValueError(f"El sistema es inestable (ρ = {self.rho:.4f} >= 1.0).")

    def _erlang_c_pw(self) -> float:
        """Calcula la probabilidad de espera Erlang C para M/M/s equivalente."""
        r = self.r
        sumatoria = sum([(r ** n) / math.factorial(n) for n in range(self.s)])
        ultimo = (r ** self.s) / (math.factorial(self.s) * (1.0 - self.rho))
        p0_mms = 1.0 / (sumatoria + ultimo)
        return ultimo * p0_mms

    def p0(self) -> float:
        """P0 aproximado mediante la suma ponderada del sistema M/M/s."""
        r = self.r
        sumatoria = sum([(r ** n) / math.factorial(n) for n in range(self.s)])
        ultimo = (r ** self.s) / (math.factorial(self.s) * (1.0 - self.rho))
        return 1.0 / (sumatoria + ultimo)

    def pw(self) -> float:
        """Probabilidad de espera en el sistema M/M/s equivalente."""
        return self._erlang_c_pw()

    def wq(self) -> float:
        """
        Tiempo promedio de espera en cola (Aproximación de Allen-Cunneen):
        Wq = [ P_M/M/s(n >= s) / (2 * s * μ * (1 - ρ)) ] * (λ^2 * σ_λ^2 + μ^2 * σ_μ^2)
        """
        pw_mms = self._erlang_c_pw()
        denom = 2.0 * self.s * self.mu_rate * (1.0 - self.rho)
        variabilidad = (self.lambda_rate ** 2) * (self.sigma_lambda ** 2) + (self.mu_rate ** 2) * (self.sigma_mu ** 2)
        return (pw_mms / denom) * variabilidad

    def lq(self) -> float:
        """Lq = λ * Wq"""
        return self.lambda_rate * self.wq()

    def ls(self) -> float:
        """Ls = Lq + λ/μ"""
        return self.lq() + self.r

    def ws(self) -> float:
        """Ws = Wq + 1/μ"""
        return self.wq() + (1.0 / self.mu_rate)


class ModeloMG1(BaseNonExponentialModel):
    """
    Modelo M/G/1: Llegadas de Poisson (exponencial), Servicio General/Arbitrario (1 servidor).
    Usa la Fórmula de Pollaczek-Khinchine (P-K).
    Parámetros:
      - lambda_rate (λ): Tasa de llegada
      - mu_rate (μ): Tasa de servicio
      - sigma (σ): Desviación típica del tiempo de servicio
    """
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
        super().__init__(lambda_rate, mu_rate, s=1, time_unit=time_unit, cq=cq, cs=cs, cw=cw)
        if sigma < 0:
            raise ValueError("La desviación típica σ no puede ser negativa.")
        self.sigma = float(sigma)

        if self.rho >= 1.0:
            raise ValueError(f"El sistema es inestable (ρ = {self.rho:.4f} >= 1.0).")

    def p0(self) -> float:
        """P0 = 1 - ρ"""
        return 1.0 - self.rho

    def pw(self) -> float:
        """En M/G/1 la utilización ρ representa la probabilidad de que el servidor esté ocupado."""
        return self.rho

    def lq(self) -> float:
        """
        Fórmula de Pollaczek-Khinchine para Lq:
        Lq = [ λ^2 * σ^2 + ρ^2 ] / [ 2 * (1 - ρ) ]
        """
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
        """Ws = Wq + 1/μ = Ls / λ"""
        return self.ls() / self.lambda_rate

    def probabilidad_n(self, n: int) -> float:
        """Aproximación Pn = (1 - ρ) * ρ^n (expresión aproximada para M/G/1 según documento)."""
        if n < 0:
            return 0.0
        return (1.0 - self.rho) * (self.rho ** n)

    def factor_curry_feldman(self) -> float:
        """
        Factor de modificación de Curry & Feldman:
        Factor = (1 + μ^2 * σ^2) / 2
        Permite relacionar Lq(M/G/1) = Lq(M/M/1) * Factor
        """
        return (1.0 + (self.mu_rate ** 2) * (self.sigma ** 2)) / 2.0


class ModeloMD1(ModeloMG1):
    """
    Modelo M/D/1: Llegadas de Poisson, Servicio Determinista/Constante (σ = 0, 1 servidor).
    Hereda de ModeloMG1 fijando σ = 0.
    """
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

    def lq(self) -> float:
        """Lq = ρ^2 / [ 2 * (1 - ρ) ]"""
        return (self.rho ** 2) / (2.0 * (1.0 - self.rho))


class ModeloMDs(BaseNonExponentialModel):
    """
    Modelo M/D/s: Llegadas de Poisson, Servicio Determinista/Constante (s >= 1 servidores).
    """
    def __init__(
        self,
        lambda_rate: float,
        mu_rate: float,
        s: int,
        time_unit: str = "minutos",
        cq: float = 0.0,
        cs: float = 0.0,
        cw: float = 0.0
    ):
        super().__init__(lambda_rate, mu_rate, s, time_unit, cq, cs, cw)
        if self.rho >= 1.0:
            raise ValueError(f"El sistema M/D/{s} es inestable (ρ = {self.rho:.4f} >= 1.0).")

    def p0(self) -> float:
        """P0 según la fórmula del modelo M/D/s."""
        r = self.r
        sumatoria = sum([(r ** n) / math.factorial(n) for n in range(self.s)])
        ultimo = (r ** self.s) / (math.factorial(self.s) * (1.0 - self.rho))
        return 1.0 / (sumatoria + ultimo)

    def pw(self) -> float:
        """Probabilidad de esperar en cola Pw."""
        r = self.r
        p0_val = self.p0()
        return ((r ** self.s) / (math.factorial(self.s) * (1.0 - self.rho))) * p0_val

    def lq(self) -> float:
        """
        Lq para M/D/s:
        Lq = [ (1 / s!) * (λ/μ)^s * (ρ / (2 * (1 - ρ)^2)) ] * P0
        """
        r = self.r
        p0_val = self.p0()
        coef = (1.0 / math.factorial(self.s)) * (r ** self.s)
        factor_rho = self.rho / (2.0 * ((1.0 - self.rho) ** 2))
        return coef * factor_rho * p0_val

    def ls(self) -> float:
        """Ls = Lq + λ/μ"""
        return self.lq() + self.r

    def wq(self) -> float:
        """Wq = Lq / λ"""
        return self.lq() / self.lambda_rate

    def ws(self) -> float:
        """Ws = Wq + 1/μ"""
        return self.wq() + (1.0 / self.mu_rate)


class ModeloMEk1(ModeloMG1):
    """
    Modelo M/E_k/1: Llegadas de Poisson, Servicio Erlang-k (1 servidor).
    Parámetro k: Forma de la distribución Erlang (k >= 1).
    Note que la varianza σ^2 = 1 / (k * μ^2), por lo que σ = 1 / (μ * sqrt(k)).
    """
    def __init__(
        self,
        lambda_rate: float,
        mu_rate: float,
        k: float,
        time_unit: str = "minutos",
        cq: float = 0.0,
        cs: float = 0.0,
        cw: float = 0.0
    ):
        if k <= 0:
            raise ValueError("El parámetro de forma k de Erlang debe ser > 0.")
        self.k = float(k)
        sigma = 1.0 / (mu_rate * math.sqrt(self.k))
        super().__init__(
            lambda_rate=lambda_rate,
            mu_rate=mu_rate,
            sigma=sigma,
            time_unit=time_unit,
            cq=cq,
            cs=cs,
            cw=cw
        )

    def lq(self) -> float:
        """
        Lq = [ (1 + k) / (2 * k) ] * [ ρ^2 / (1 - ρ) ]
        """
        factor_k = (1.0 + self.k) / (2.0 * self.k)
        return factor_k * ((self.rho ** 2) / (1.0 - self.rho))


class ModeloErlangB(BaseNonExponentialModel):
    """
    Modelo Erlang-B / M/G/s/s: Sistema de líneas telefónicas / Pérdidas (sin cola).
    Los clientes que encuentran todos los servidores ocupados se pierden (son bloqueados).
    """
    def __init__(
        self,
        lambda_rate: float,
        mu_rate: float,
        s: int,
        time_unit: str = "minutos",
        cs: float = 0.0
    ):
        super().__init__(lambda_rate, mu_rate, s, time_unit, cq=0.0, cs=cs, cw=0.0)
        self.A = self.r  # Tráfico ofrecido en Erlangs

    def p0(self) -> float:
        """Probabilidad de que el sistema esté completamente libre P0."""
        denom = sum([(self.A ** n) / math.factorial(n) for n in range(self.s + 1)])
        return 1.0 / denom

    def probabilidad_bloqueo(self) -> float:
        """
        Fórmula de Erlang-B P_s (Probabilidad de bloqueo / todas las líneas ocupadas):
        P_s = (A^s / s!) / [ sum_{h=0}^s (A^h / h!) ]
        """
        p0_val = self.p0()
        return ((self.A ** self.s) / math.factorial(self.s)) * p0_val

    def pw(self) -> float:
        """En Erlang B, Pw es la probabilidad de ser bloqueado P_s."""
        return self.probabilidad_bloqueo()

    def probabilidad_j(self, j: int) -> float:
        """Probabilidad de que exactamente j servidores estén ocupados (0 <= j <= s)."""
        if j < 0 or j > self.s:
            return 0.0
        return ((self.A ** j) / math.factorial(j)) * self.p0()

    def trafico_ofrecido(self) -> float:
        """A = λ / μ (Erlangs)"""
        return self.A

    def trafico_servido(self) -> float:
        """H = A * (1 - P_s) (Erlangs) - Número medio de servidores ocupados."""
        return self.A * (1.0 - self.probabilidad_bloqueo())

    def lq(self) -> float:
        """En un sistema sin cola, Lq = 0."""
        return 0.0

    def ls(self) -> float:
        """Número promedio de clientes en el sistema (servidores ocupados H)."""
        return self.trafico_servido()

    def wq(self) -> float:
        """En un sistema sin cola, Wq = 0."""
        return 0.0

    def ws(self) -> float:
        """Tiempo promedio en el sistema para clientes atendidos = 1/μ."""
        return 1.0 / self.mu_rate


class ModeloPrioridades:
    """
    Modelo de Colas con Clases de Prioridad (FIFO dentro de cada clase).
    Soporta:
      - Prioridades Sin Interrupción (Non-Preemptive) para s=1 o s>=1
      - Prioridades Con Interrupción (Preemptive) para s=1
    Prioridad 1 es la más alta (Crítica).
    """
    def __init__(
        self,
        lambda_rates: List[float],
        mu_rate: float,
        s: int = 1,
        interrupcion: bool = False,
        time_unit: str = "minutos"
    ):
        if not lambda_rates or any(lam <= 0 for lam in lambda_rates):
            raise ValueError("Las tasas de llegada (λ_i) deben ser positivas.")
        if mu_rate <= 0:
            raise ValueError("La tasa de servicio (μ) debe ser positiva.")
        if s < 1:
            raise ValueError("El número de servidores (s) debe ser >= 1.")

        self.lambda_rates = [float(l) for l in lambda_rates]
        self.num_clases = len(lambda_rates)
        self.mu_rate = float(mu_rate)
        self.s = int(s)
        self.interrupcion = bool(interrupcion)
        self.time_unit = time_unit

        self.lambda_total = sum(self.lambda_rates)
        self.rho_total = self.lambda_total / (self.s * self.mu_rate)

        if self.rho_total >= 1.0:
            raise ValueError(f"El sistema de prioridades es inestable (ρ_total = {self.rho_total:.4f} >= 1.0).")

    def _b_k(self, k: int) -> float:
        """Calcula B_k = 1 - (1 / (s * μ)) * sum_{i=1}^k λ_i"""
        sum_lambda = sum(self.lambda_rates[:k])
        return 1.0 - (sum_lambda / (self.s * self.mu_rate))

    def calcular_metricas(self) -> List[Dict[str, float]]:
        """
        Calcula W_k, Wq_k, L_k, Lq_k para cada clase de prioridad k (1-indexed).
        """
        resultados = []
        mu = self.mu_rate
        s = self.s

        if not self.interrupcion:
            # Sin interrupción (Non-preemptive)
            if s == 1:
                # Fórmulas simplificadas para s=1 (pág. 38 del documento)
                # A_k = (1 / μ^2) * sum_{i=1}^N λ_i = λ_total / μ^2
                A_const = self.lambda_total / (mu ** 2)
                for k in range(1, self.num_clases + 1):
                    lam_k = self.lambda_rates[k - 1]
                    b_prev = self._b_k(k - 1)
                    b_curr = self._b_k(k)
                    
                    wq_k = A_const / (b_prev * b_curr)
                    ws_k = wq_k + (1.0 / mu)
                    lq_k = lam_k * wq_k
                    ls_k = lam_k * ws_k

                    resultados.append({
                        "clase": k,
                        "lambda": lam_k,
                        "Wq": wq_k,
                        "Ws": ws_k,
                        "Lq": lq_k,
                        "Ls": ls_k
                    })
            else:
                # Múltiples servidores s > 1 (Non-preemptive)
                r_total = self.lambda_total / mu
                # A = (s! (s μ - λ_total) / r_total^s) * sum_{j=0}^{s-1} (r_total^j / j!) + s μ
                denom_a = (r_total ** s)
                sum_r = sum([(r_total ** j) / math.factorial(j) for j in range(s)])
                A_factor = (math.factorial(s) * (s * mu - self.lambda_total) / denom_a) * sum_r + (s * mu)
                
                for k in range(1, self.num_clases + 1):
                    lam_k = self.lambda_rates[k - 1]
                    b_prev = self._b_k(k - 1)
                    b_curr = self._b_k(k)
                    
                    ws_k = (1.0 / (A_factor * b_prev * b_curr)) + (1.0 / mu)
                    wq_k = ws_k - (1.0 / mu)
                    lq_k = lam_k * wq_k
                    ls_k = lam_k * ws_k

                    resultados.append({
                        "clase": k,
                        "lambda": lam_k,
                        "Wq": wq_k,
                        "Ws": ws_k,
                        "Lq": lq_k,
                        "Ls": ls_k
                    })
        else:
            # Con interrupción (Preemptive) - s=1 (pág. 39 del documento)
            for k in range(1, self.num_clases + 1):
                lam_k = self.lambda_rates[k - 1]
                b_prev = self._b_k(k - 1)
                b_curr = self._b_k(k)

                ws_k = (1.0 / mu) / (b_prev * b_curr)
                wq_k = ws_k - (1.0 / mu)
                lq_k = lam_k * wq_k
                ls_k = lam_k * ws_k

                resultados.append({
                    "clase": k,
                    "lambda": lam_k,
                    "Wq": wq_k,
                    "Ws": ws_k,
                    "Lq": lq_k,
                    "Ls": ls_k
                })

        return resultados

    def generar_reporte_texto(self) -> str:
        res = self.calcular_metricas()
        tipo_str = "CON Interrupción (Preemptive)" if self.interrupcion else "SIN Interrupción (Non-Preemptive)"
        lineas = [
            "=" * 72,
            f"   REPORTE DE SISTEMA DE COLAS CON PRIORIDADES - {tipo_str}",
            "=" * 72,
            f"  Tasa global de llegada (λ_total) : {self.lambda_total:.4f} clientes / {self.time_unit}",
            f"  Tasa de servicio (μ)             : {self.mu_rate:.4f} clientes / {self.time_unit}",
            f"  Número de servidores (s)         : {self.s}",
            f"  Factor de utilización global (ρ): {self.rho_total:.4f} ({self.rho_total * 100:.2f}%)",
            "-" * 72,
            f"  {'Clase':<6} | {'λ_k':<8} | {'Lq_k':<10} | {'Ls_k':<10} | {'Wq_k ('+self.time_unit+')':<14} | {'Ws_k ('+self.time_unit+')':<14}",
            "  " + "-" * 68
        ]

        lq_total = sum(r["Lq"] for r in res)
        ls_total = sum(r["Ls"] for r in res)
        wq_prom = lq_total / self.lambda_total
        ws_prom = ls_total / self.lambda_total

        for r in res:
            lineas.append(
                f"  {r['clase']:<6} | {r['lambda']:<8.4f} | {r['Lq']:<10.4f} | {r['Ls']:<10.4f} | "
                f"{r['Wq']:<14.4f} | {r['Ws']:<14.4f}"
            )

        lineas.extend([
            "-" * 72,
            f"  TOTALES / PROMEDIOS PONDERADOS:",
            f"  N° total de clientes en cola (Lq)     : {lq_total:.4f} clientes",
            f"  N° total de clientes en sistema (Ls)  : {ls_total:.4f} clientes",
            f"  Tiempo medio ponderado en cola (Wq)   : {wq_prom:.4f} {self.time_unit}",
            f"  Tiempo medio ponderado en sistema (Ws): {ws_prom:.4f} {self.time_unit}",
            "=" * 72
        ])
        return "\n".join(lineas)

    def mostrar_reporte(self):
        print(self.generar_reporte_texto())


class ModeloInterrupcionesServidor:
    """
    Modelo de Colas con Interrupciones / Fallas del Servidor (Server Breakdowns and Repair).
    Llegadas y servicios de Poisson con fallas aleatorias (tasa λ_R) y reparaciones (tasa μ_R).
    Páginas 42-48 del documento de la UAM.
    """
    def __init__(
        self,
        lambda_rate: float,
        mu_rate: float,
        lambda_r: float,
        mu_r: float,
        time_unit: str = "horas"
    ):
        if lambda_rate <= 0 or mu_rate <= 0 or lambda_r <= 0 or mu_r <= 0:
            raise ValueError("Todas las tasas (λ, μ, λ_R, μ_R) deben ser estrictamente positivas (> 0).")

        self.lambda_rate = float(lambda_rate)
        self.mu_rate = float(mu_rate)
        self.lambda_r = float(lambda_r)
        self.mu_r = float(mu_r)
        self.time_unit = time_unit

        self.T_A = 1.0 / self.lambda_rate
        self.T_S = 1.0 / self.mu_rate
        self.T_R = 1.0 / self.lambda_r
        self.T_SR = 1.0 / self.mu_r

        self.rho = self.lambda_rate / self.mu_rate  # p(H)

        if self.rho >= 1.0:
            raise ValueError(f"El sistema es inestable (ρ nominal = {self.rho:.4f} >= 1.0).")

    def calcular_probabilidades_estado(self) -> Dict[str, float]:
        """Calcula p(0,F), p(F), p(S), p(R), p(0), p(0,R)."""
        T_S = self.T_S
        T_SR = self.T_SR
        T_R = self.T_R
        pH = self.rho

        # p(0,F) de ecuación (6) pág 46
        num_p0f = T_R * (T_R + T_S - pH * (T_S + T_SR + T_R))
        denom_p0f = (T_R + T_S) * (T_R + T_SR)
        p0_f = num_p0f / denom_p0f

        # p(F) de ecuación (1) pág 44
        pF = (T_R + p0_f * T_S) / (T_S + T_SR + T_R)

        # p(S) de ecuación (4) pág 45
        pS = (T_S / T_R) * (T_R - p0_f * T_R - p0_f * T_SR) / (T_S + T_SR + T_R)

        # p(R) de ecuación (2) pág 44
        pR = T_SR / (T_S + T_SR + T_R - T_S * (p0_f / pF))

        # p(0,R) de la página 48
        p0_r = p0_f * (2.0 * self.lambda_rate + self.lambda_r + self.mu_rate) * self.lambda_r / (
            self.lambda_r * self.mu_r + (self.mu_r + self.lambda_rate) * (self.lambda_r + self.mu_rate)
        )

        p0 = p0_f + p0_r

        return {
            "p0_F": p0_f,
            "p0_R": p0_r,
            "p0": p0,
            "pF": pF,
            "pS": pS,
            "pR": pR
        }

    def calcular_metricas(self) -> Dict[str, float]:
        """Calcula Wc, Ws, Lc, Ls, H, λ_C."""
        probs = self.calcular_probabilidades_estado()
        pF = probs["pF"]
        pS = probs["pS"]
        pR = probs["pR"]

        T_S = self.T_S
        T_SR = self.T_SR
        T_R = self.T_R

        lambda_c = 1.0 / (T_S + T_SR + T_R)

        # Ecuación (8) pág 47 para Wc
        p_RS = pR + pS
        num_wc = self.rho * T_S + T_SR * (p_RS - lambda_c * T_SR * p_RS - lambda_c * T_S * pS)
        denom_wc = 1.0 - self.rho - lambda_c * T_SR
        wc = num_wc / denom_wc

        ws = wc + T_S
        lc = self.lambda_rate * wc
        ls = self.lambda_rate * ws
        h = self.rho

        return {
            "lambda_C": lambda_c,
            "Wc": wc,
            "Ws": ws,
            "Lc": lc,
            "Ls": ls,
            "H": h,
            "p0_F": probs["p0_F"],
            "p0_R": probs["p0_R"],
            "p0": probs["p0"],
            "pF": pF,
            "pS": pS,
            "pR": pR
        }

    def generar_reporte_texto(self) -> str:
        m = self.calcular_metricas()
        lineas = [
            "=" * 72,
            "   REPORTE DE SISTEMA DE COLAS CON INTERRUPCIONES DEL SERVIDOR",
            "=" * 72,
            f"  Tasa de llegada (λ)                   : {self.lambda_rate:.4f} clientes / {self.time_unit}",
            f"  Tasa de servicio (μ)                  : {self.mu_rate:.4f} clientes / {self.time_unit}",
            f"  Tasa de fallas (λ_R)                  : {self.lambda_r:.4f} fallas / {self.time_unit}",
            f"  Tasa de reparación (μ_R)              : {self.mu_r:.4f} reparaciones / {self.time_unit}",
            "-" * 72,
            f"  Tiempo medio entre llegadas (T_A)     : {self.T_A:.4f} {self.time_unit}",
            f"  Tiempo medio de servicio (T_S)        : {self.T_S:.4f} {self.time_unit}",
            f"  Tiempo medio hasta falla (T_R)        : {self.T_R:.4f} {self.time_unit}",
            f"  Tiempo medio de reparación (T_SR)     : {self.T_SR:.4f} {self.time_unit}",
            "-" * 72,
            f"  Prob. Sistema Operativo p(F)          : {m['pF']:.6f} ({m['pF']*100:.2f}%)",
            f"  Prob. Temp. Operativo p(S)            : {m['pS']:.6f} ({m['pS']*100:.2f}%)",
            f"  Prob. No Operativo / Reparación p(R)  : {m['pR']:.6f} ({m['pR']*100:.2f}%)",
            f"  Probabilidad de Vacío Total p(0)      : {m['p0']:.6f} ({m['p0']*100:.2f}%)",
            "-" * 72,
            f"  Tiempo medio en cola (Wc)             : {m['Wc']:.6f} {self.time_unit}",
            f"  Tiempo medio en sistema (Ws)          : {m['Ws']:.6f} {self.time_unit}",
            f"  N° medio de clientes en cola (Lc)     : {m['Lc']:.6f} clientes",
            f"  N° medio de clientes en sistema (Ls)  : {m['Ls']:.6f} clientes",
            "=" * 72
        ]
        return "\n".join(lineas)

    def mostrar_reporte(self):
        print(self.generar_reporte_texto())


def resolver_ejemplo_aeropuerto() -> Tuple[ModeloMD1, ModeloMDs]:
    """
    Resuelve explícitamente el problema del servicio de traslado del aeropuerto de la página 17:
    'El aeropuerto dispone de un servicio de traslado en el que consiste en llevar a cada empleado
    que lo solicite a su casa, hotel o alrededores. Este servicio puede atender a un empleado cada
    7 minutos. El promedio de llegada de empleados es cada 8 minutos, siguiendo una distribución de Poisson.
    a) Encontrar las medidas de eficiencia del servicio.
    b) ¿Se podría mejorar el tiempo medio de un empleado en el sistema?'
    """
    print("\n" + "=" * 76)
    print(" RESOLUCIÓN DEL PROBLEMA DEL AEROPUERTO (SERVICIO DE TRASLADO DE EMPLEADOS)")
    print("=" * 76)
    print(" Datos iniciales del problema:")
    print("  • Tiempo medio entre llegadas: 8 minutos (Poisson -> λ = 1/8 = 0.125 empleados/min)")
    print("  • Tiempo medio de servicio   : 7 minutos constantes (-> μ = 1/7 ≈ 0.142857 empleados/min)")
    print("-" * 76)

    # Parte a: Modelo M/D/1 (s = 1)
    lambda_val = 1.0 / 8.0
    mu_val = 1.0 / 7.0

    m_d_1 = ModeloMD1(lambda_rate=lambda_val, mu_rate=mu_val, time_unit="minutos")
    
    print("\n>>> PARTE A: Medidas de eficiencia con s = 1 servidor (Modelo M/D/1)")
    print(f"  • Factor de utilización (ρ)            : {m_d_1.rho:.4f} ({m_d_1.rho*100:.2f}%)")
    print(f"  • Probabilidad de sistema vacío (P0)   : {m_d_1.p0():.4f} ({m_d_1.p0()*100:.2f}%)")
    print(f"  • N° medio de empleados en cola (Lq)   : {m_d_1.lq():.4f} empleados")
    print(f"  • N° medio de empleados en sistema (Ls): {m_d_1.ls():.4f} empleados")
    print(f"  • Tiempo medio de espera en cola (Wq)  : {m_d_1.wq():.4f} minutos")
    print(f"  • Tiempo medio en el sistema (Ws)      : {m_d_1.ws():.4f} minutos")

    # Parte b: Evaluación de mejora añadiendo un segundo vehículo (s = 2) -> Modelo M/D/2
    m_d_2 = ModeloMDs(lambda_rate=lambda_val, mu_rate=mu_val, s=2, time_unit="minutos")

    print("\n>>> PARTE B: ¿Se podría mejorar el tiempo medio de un empleado en el sistema?")
    print("  -> SÍ. Añadiendo un segundo servidor/vehículo de traslado (s = 2), el sistema pasa a ser M/D/2.")
    print(f"  • Nuevo factor de utilización (ρ)      : {m_d_2.rho:.4f} ({m_d_2.rho*100:.2f}%)")
    print(f"  • Nueva probabilidad de vacío (P0)    : {m_d_2.p0():.4f} ({m_d_2.p0()*100:.2f}%)")
    print(f"  • Nuevo N° medio en cola (Lq)          : {m_d_2.lq():.4f} empleados")
    print(f"  • Nuevo N° medio en sistema (Ls)       : {m_d_2.ls():.4f} empleados")
    print(f"  • Nuevo tiempo medio en cola (Wq)      : {m_d_2.wq():.4f} minutos")
    print(f"  • Nuevo tiempo medio en sistema (Ws)   : {m_d_2.ws():.4f} minutos")

    mejora_pct = ((m_d_1.ws() - m_d_2.ws()) / m_d_1.ws()) * 100.0
    print(f"\n  CONCLUSIÓN:")
    print(f"  El tiempo medio de permanencia en el sistema (Ws) se reduce drásticamente")
    print(f"  de {m_d_1.ws():.3f} minutos a {m_d_2.ws():.3f} minutos (una mejora del {mejora_pct:.2f}%).")
    print("=" * 76 + "\n")

    return m_d_1, m_d_2


def menu_interactivo():
    """Menú principal interactivo de la calculadora no exponencial."""
    while True:
        print("\n" + "=" * 72)
        print("     CALCULADORA DE TEORÍA DE COLAS NO EXPONENCIALES Y AVANZADAS")
        print("=======================================================================")
        print("  1. Resolver Ejemplo del Aeropuerto (Servicio de Traslado M/D/1 vs M/D/2)")
        print("  2. Modelo G/G/s (Aproximación Allen-Cunneen)")
        print("  3. Modelo M/G/1 (Fórmula de Pollaczek-Khinchine)")
        print("  4. Modelo M/D/1 (Servicio constante, 1 servidor)")
        print("  5. Modelo M/D/s (Servicio constante, s servidores)")
        print("  6. Modelo M/E_k/1 (Servicio Erlang-k, 1 servidor)")
        print("  7. Modelo Erlang-B / M/G/s/s (Líneas telefónicas / Sin cola)")
        print("  8. Sistema con Prioridades (Sin / Con interrupción)")
        print("  9. Modelo con Interrupciones / Fallas del Servidor")
        print("  0. Salir")
        print("=" * 72)

        opcion = input("  Seleccione una opción [0-9]: ").strip()

        if opcion == "1":
            m1, m2 = resolver_ejemplo_aeropuerto()
            exp = input("  ¿Desea exportar los resultados del problema del aeropuerto a TXT? (s/n) [s]: ").strip().lower()
            if exp in ['', 's', 'si', 'sí', 'y', 'yes']:
                rep1 = m1.generar_reporte_texto(notas_adicionales=["Problema Aeropuerto - Caso a (s=1)"])
                rep2 = m2.generar_reporte_texto(notas_adicionales=["Problema Aeropuerto - Caso b (s=2)"])
                with open("resultado_aeropuerto.txt", "w", encoding="utf-8") as f:
                    f.write(rep1 + "\n\n" + rep2)
                print("  [ÉXITO]: Reporte guardado en 'resultado_aeropuerto.txt'.")

        elif opcion == "2":
            try:
                lam = parse_float(input("  Ingrese tasa de llegada λ: "))
                sig_lam = parse_float(input("  Ingrese desv. típica des llegadas σ_λ: "))
                mu = parse_float(input("  Ingrese tasa de servicio μ: "))
                sig_mu = parse_float(input("  Ingrese desv. típica del servicio σ_μ: "))
                s = int(parse_float(input("  Ingrese número de servidores s [1]: ") or "1"))
                unid = input("  Ingrese unidad de tiempo [minutos]: ").strip() or "minutos"

                mod = ModeloGGs(lam, sig_lam, mu, sig_mu, s=s, time_unit=unid)
                mod.mostrar_reporte()

                _menu_secundario(mod)
            except Exception as e:
                print(f"  [ERROR]: {e}")

        elif opcion == "3":
            try:
                lam = parse_float(input("  Ingrese tasa de llegada λ: "))
                mu = parse_float(input("  Ingrese tasa de servicio μ: "))
                sig = parse_float(input("  Ingrese desv. típica del servicio σ: "))
                unid = input("  Ingrese unidad de tiempo [minutos]: ").strip() or "minutos"

                mod = ModeloMG1(lam, mu, sigma=sig, time_unit=unid)
                mod.mostrar_reporte()
                print(f"  • Factor de Curry & Feldman: {mod.factor_curry_feldman():.4f}")

                _menu_secundario(mod)
            except Exception as e:
                print(f"  [ERROR]: {e}")

        elif opcion == "4":
            try:
                lam = parse_float(input("  Ingrese tasa de llegada λ: "))
                mu = parse_float(input("  Ingrese tasa de servicio μ: "))
                unid = input("  Ingrese unidad de tiempo [minutos]: ").strip() or "minutos"

                mod = ModeloMD1(lam, mu, time_unit=unid)
                mod.mostrar_reporte()

                _menu_secundario(mod)
            except Exception as e:
                print(f"  [ERROR]: {e}")

        elif opcion == "5":
            try:
                lam = parse_float(input("  Ingrese tasa de llegada λ: "))
                mu = parse_float(input("  Ingrese tasa de servicio μ: "))
                s = int(parse_float(input("  Ingrese número de servidores s: ")))
                unid = input("  Ingrese unidad de tiempo [minutos]: ").strip() or "minutos"

                mod = ModeloMDs(lam, mu, s=s, time_unit=unid)
                mod.mostrar_reporte()

                _menu_secundario(mod)
            except Exception as e:
                print(f"  [ERROR]: {e}")

        elif opcion == "6":
            try:
                lam = parse_float(input("  Ingrese tasa de llegada λ: "))
                mu = parse_float(input("  Ingrese tasa de servicio μ: "))
                k = parse_float(input("  Ingrese parámetro k de Erlang (k >= 1): "))
                unid = input("  Ingrese unidad de tiempo [minutos]: ").strip() or "minutos"

                mod = ModeloMEk1(lam, mu, k=k, time_unit=unid)
                mod.mostrar_reporte()

                _menu_secundario(mod)
            except Exception as e:
                print(f"  [ERROR]: {e}")

        elif opcion == "7":
            try:
                lam = parse_float(input("  Ingrese tasa de llegada λ: "))
                mu = parse_float(input("  Ingrese tasa de servicio μ: "))
                s = int(parse_float(input("  Ingrese número de líneas/servidores s: ")))
                unid = input("  Ingrese unidad de tiempo [minutos]: ").strip() or "minutos"

                mod = ModeloErlangB(lam, mu, s=s, time_unit=unid)
                mod.mostrar_reporte()
                print(f"  • Tráfico Ofrecido (A) : {mod.trafico_ofrecido():.4f} Erlangs")
                print(f"  • Tráfico Servido (H)  : {mod.trafico_servido():.4f} Erlangs")
                print(f"  • Prob. Bloqueo (P_s)  : {mod.probabilidad_bloqueo():.4f} ({mod.probabilidad_bloqueo()*100:.2f}%)")

                _menu_secundario(mod)
            except Exception as e:
                print(f"  [ERROR]: {e}")

        elif opcion == "8":
            try:
                num_c = int(parse_float(input("  Ingrese número de clases de prioridad N: ")))
                lams = []
                for i in range(1, num_c + 1):
                    lams.append(parse_float(input(f"    Ingrese λ_{i} para prioridad {i}: ")))
                mu = parse_float(input("  Ingrese tasa de servicio común μ: "))
                s = int(parse_float(input("  Ingrese número de servidores s [1]: ") or "1"))
                inter = input("  ¿Es con interrupción (preemptive)? (s/n) [n]: ").strip().lower() in ['s', 'si', 'sí', 'y', 'yes']
                unid = input("  Ingrese unidad de tiempo [minutos]: ").strip() or "minutos"

                mod = ModeloPrioridades(lams, mu, s=s, interrupcion=inter, time_unit=unid)
                mod.mostrar_reporte()

                exp = input("  ¿Desea exportar el reporte a TXT? (s/n) [n]: ").strip().lower()
                if exp in ['s', 'si', 'sí', 'y', 'yes']:
                    fname = input("  Nombre del archivo TXT [reporte_prioridades.txt]: ").strip() or "reporte_prioridades.txt"
                    with open(fname, "w", encoding="utf-8") as f:
                        f.write(mod.generar_reporte_texto())
                    print(f"  [ÉXITO]: Exportado a '{fname}'.")
            except Exception as e:
                print(f"  [ERROR]: {e}")

        elif opcion == "9":
            try:
                lam = parse_float(input("  Ingrese tasa de llegada de clientes λ: "))
                mu = parse_float(input("  Ingrese tasa de servicio μ: "))
                lam_r = parse_float(input("  Ingrese tasa de fallas λ_R: "))
                mu_r = parse_float(input("  Ingrese tasa de reparación μ_R: "))
                unid = input("  Ingrese unidad de tiempo [horas]: ").strip() or "horas"

                mod = ModeloInterrupcionesServidor(lam, mu, lam_r, mu_r, time_unit=unid)
                mod.mostrar_reporte()

                exp = input("  ¿Desea exportar el reporte a TXT? (s/n) [n]: ").strip().lower()
                if exp in ['s', 'si', 'sí', 'y', 'yes']:
                    fname = input("  Nombre del archivo TXT [reporte_interrupciones.txt]: ").strip() or "reporte_interrupciones.txt"
                    with open(fname, "w", encoding="utf-8") as f:
                        f.write(mod.generar_reporte_texto())
                    print(f"  [ÉXITO]: Exportado a '{fname}'.")
            except Exception as e:
                print(f"  [ERROR]: {e}")

        elif opcion == "0":
            print("\n  ¡Gracias por utilizar el módulo de teoría de colas no exponenciales!")
            break
        else:
            print("  Opción inválida. Intente de nuevo.")


def _mostrar_tabla_pn(modelo: BaseNonExponentialModel, k_max: int):
    """Muestra una tabla ordenada de probabilidades Pn, P(<=n), P(>n)."""
    print(f"\n  {'n':<5} | {'P(N = n)':<12} | {'P(N <= n)':<12} | {'P(N > n)':<12}")
    print("  " + "-" * 48)
    for i in range(k_max + 1):
        pn = modelo.probabilidad_n(i)
        pc = modelo.probabilidad_n_o_menos(i)
        pg = modelo.probabilidad_mas_de_n(i)
        print(f"  {i:<5} | {pn:<12.6f} | {pc:<12.6f} | {pg:<12.6f}")


def _menu_secundario(modelo: BaseNonExponentialModel):
    """Menú secundario interactivo para consultar probabilidades, tiempos, costos y exportar."""
    consultas_adicionales = []
    k_max_tabla = None

    while True:
        print("\n" + "-" * 55)
        print("     CONSULTAS ADICIONALES Y OPCIONES DEL MODELO")
        print("-" * 55)
        print("  1. Probabilidad de exactamente 'n' clientes P(N = n)")
        print("  2. Probabilidad de 'n' o menos clientes P(N <= n)")
        print("  3. Probabilidad de más de 'n' clientes P(N > n)")
        print("  4. Probabilidad de tiempo en sistema mayor a 't' P(Ws > t)")
        print("  5. Probabilidad de tiempo en cola mayor a 't' P(Wq > t)")
        print("  6. Mostrar tabla de probabilidades Pn (hasta K)")
        print("  7. Ingresar / Actualizar Costos del Sistema (Cs, Cq, Cw)")
        print("  8. Exportar todos los resultados a un archivo TXT (.txt)")
        print("  0. Volver al menú principal")
        print("-" * 55)

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
                print("\n  --- INGRESO / ACTUALIZACIÓN DE COSTOS ---")
                modelo.cs = parse_float(input(f"  Costo por servidor (Cs) [{modelo.cs}]: ") or str(modelo.cs))
                modelo.cq = parse_float(input(f"  Costo de espera en cola (Cq) [{modelo.cq}]: ") or str(modelo.cq))
                modelo.cw = parse_float(input(f"  Costo de atención por cliente (Cw) [{modelo.cw}]: ") or str(modelo.cw))
                print("\n  [ÉXITO]: Costos actualizados. Reporte actualizado:")
                modelo.mostrar_reporte()
            except ValueError as e:
                print(f"  [ERROR]: {e}")

        elif sub_opc == "8":
            try:
                nombre_archivo = input("  Ingrese el nombre del archivo TXT [reporte_no_exponencial.txt]: ").strip() or "reporte_no_exponencial.txt"
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


if __name__ == "__main__":
    menu_interactivo()
