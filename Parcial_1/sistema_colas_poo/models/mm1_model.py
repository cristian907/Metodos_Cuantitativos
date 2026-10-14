#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modelo de Colas M/M/1: Servidor único (s = 1), llegadas de Poisson
y tiempos de servicio exponenciales.
Aprovecha fórmulas simplificadas y cerradas directas para M/M/1.
"""

import math
from .mms_model import ModeloMMs


class ModeloMM1(ModeloMMs):
    """
    Especialización de ModeloMMs para servidor único (s = 1).
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
        """En M/M/1, la probabilidad de esperar es igual al factor de utilización ρ."""
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
