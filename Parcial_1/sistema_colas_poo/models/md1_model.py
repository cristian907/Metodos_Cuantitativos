#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modelo de Colas M/D/1: Llegadas de Poisson, Servicio Determinista / Constante (1 servidor).
Hereda de ModeloMG1 fijando la desviación típica σ = 0.
"""

from .mg1_model import ModeloMG1


class ModeloMD1(ModeloMG1):
    """
    Modelo M/D/1: Llegadas de Poisson, Tiempo de Servicio Constante (σ = 0).
    Válido para s = 1.
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
        """
        Fórmula de M/D/1 para Lq (Servicio determinista σ = 0):
        Lq = ρ^2 / [ 2 * (1 - ρ) ] = Lq(M/M/1) / 2
        """
        return (self.rho ** 2) / (2.0 * (1.0 - self.rho))
