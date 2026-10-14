#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modelo de Colas M/M/s: Múltiples servidores en paralelo (s >= 1),
llegadas de Poisson (exponenciales) y tiempos de servicio exponenciales.
"""

import math
from .base_model import BaseQueueModel


class ModeloMMs(BaseQueueModel):
    """
    Representa un modelo de teoría de colas M/M/s con múltiples servidores.
    Válido para s >= 1.
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
        """
        Calcula P0: Probabilidad de que el sistema esté totalmente vacío.
        P0 = 1 / [ sum_{n=0}^{s-1} (r^n / n!) + (r^s / (s! * (1 - ρ))) ]
        """
        r = self.r
        sumatoria = sum([(r ** n) / math.factorial(n) for n in range(self.s)])
        ultimo_termino = (r ** self.s) / (math.factorial(self.s) * (1.0 - self.rho))
        return 1.0 / (sumatoria + ultimo_termino)

    def pw(self) -> float:
        """
        Calcula Erlang C (Pw): Probabilidad de que un cliente entrante deba esperar P(N >= s).
        Pw = (r^s / (s! * (1 - ρ))) * P0
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
        Tiempo promedio de permanencia en el sistema (Ws).
        Ws = Ls / λ = Wq + 1/μ
        """
        return self.ls() / self.lambda_rate

    def probabilidad_n(self, n: int) -> float:
        """
        Calcula Pn: Probabilidad de que haya exactamente n clientes en el sistema.
        - Si 0 <= n <= s: Pn = (r^n / n!) * P0
        - Si n > s:       Pn = (r^n / (s! * s^(n-s))) * P0
        """
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
        """
        Calcula P(Wq > t): Probabilidad de esperar en cola más de un tiempo t.
        P(Wq > t) = Pw * exp(-s * μ * (1 - ρ) * t)
        """
        if t < 0:
            return 1.0
        exponent = -self.s * self.mu_rate * (1.0 - self.rho) * t
        return self.pw() * math.exp(exponent)

    def prob_tiempo_sistema_mayor_t(self, t: float) -> float:
        """
        Calcula P(Ws > t): Probabilidad de permanecer en el sistema más de un tiempo t.
        """
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
