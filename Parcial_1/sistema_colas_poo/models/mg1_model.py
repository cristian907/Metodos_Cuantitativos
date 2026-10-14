#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modelo de Colas M/G/1: Llegadas de Poisson, Servicio General / Arbitrario (1 servidor).
Utiliza la Fórmula de Pollaczek-Khinchine (P-K) y permite el cálculo
del factor de ajuste de Curry & Feldman.
"""

from .base_model import BaseQueueModel


class ModeloMG1(BaseQueueModel):
    """
    Modelo M/G/1: Llegadas de Poisson, Servicio General con desviación estándar σ.
    Válido para s = 1.
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
        """En M/G/1, la probabilidad de esperar en el sistema es igual a la ocupación ρ."""
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
        """Ws = Ls / λ = Wq + 1/μ"""
        return self.ls() / self.lambda_rate

    def factor_curry_feldman(self) -> float:
        """
        Calcula el Factor de Modificación de Curry & Feldman:
        Factor = (1 + μ^2 * σ^2) / 2
        Relación: Lq(M/G/1) = Lq(M/M/1) * Factor
        """
        return (1.0 + (self.mu_rate ** 2) * (self.sigma ** 2)) / 2.0

    def probabilidad_n(self, n: int) -> float:
        """
        Aproximación de Pn para M/G/1:
        Pn ≈ (1 - ρ) * ρ^n
        """
        if n < 0:
            return 0.0
        return (1.0 - self.rho) * (self.rho ** n)
