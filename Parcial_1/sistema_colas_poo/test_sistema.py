#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Suite de Pruebas Unitarias e Integración para el Sistema de Teoría de Colas (POO).
Valida las fórmulas matemáticas, estabilidad, costos y servicios contra soluciones teóricas conocidas.
"""

import sys
import unittest
from pathlib import Path

# Permitir importación del paquete desde la raíz
sys.path.insert(0, str(Path(__file__).parent.parent))

from sistema_colas_poo.models.mm1_model import ModeloMM1
from sistema_colas_poo.models.mms_model import ModeloMMs
from sistema_colas_poo.models.md1_model import ModeloMD1
from sistema_colas_poo.models.mg1_model import ModeloMG1
from sistema_colas_poo.services.cost_optimizer import optimizar_servidores
from sistema_colas_poo.services.model_comparator import comparar_modelos


class TestSistemaColas(unittest.TestCase):
    """Casos de prueba para verificar los 4 modelos de colas y servicios auxiliaries."""

    def test_mm1_problema1(self):
        """Verifica M/M/1 con λ = 0.8, μ = 1.0 (Problema1.txt)."""
        modelo = ModeloMM1(lambda_rate=0.8, mu_rate=1.0, time_unit="minutos")
        self.assertAlmostEqual(modelo.rho, 0.8, places=4)
        self.assertAlmostEqual(modelo.p0(), 0.2, places=4)
        self.assertAlmostEqual(modelo.pw(), 0.8, places=4)
        self.assertAlmostEqual(modelo.ls(), 4.0, places=4)
        self.assertAlmostEqual(modelo.lq(), 3.2, places=4)
        self.assertAlmostEqual(modelo.ws(), 5.0, places=4)
        self.assertAlmostEqual(modelo.wq(), 4.0, places=4)
        self.assertAlmostEqual(modelo.lb(), 4.0, places=4)
        self.assertAlmostEqual(modelo.wb(), 5.0, places=4)
        # Probabilidad de estado P(N > 3)
        self.assertAlmostEqual(modelo.probabilidad_mas_de_n(3), 0.4096, places=4)

    def test_mms_problema2(self):
        """Verifica M/M/3 con λ = 15.0, μ = 6.0 (Problema2.txt)."""
        modelo = ModeloMMs(lambda_rate=15.0, mu_rate=6.0, s=3, time_unit="horas")
        self.assertAlmostEqual(modelo.rho, 15.0 / 18.0, places=4)
        self.assertAlmostEqual(modelo.p0(), 0.0449438, places=4)
        self.assertAlmostEqual(modelo.pw(), 0.702247, places=4)
        self.assertAlmostEqual(modelo.lq(), 3.511235, places=4)
        self.assertAlmostEqual(modelo.ls(), 6.011235, places=4)
        self.assertAlmostEqual(modelo.wq(), 0.234082, places=4)
        self.assertAlmostEqual(modelo.ws(), 0.400749, places=4)

    def test_md1_formula(self):
        """Verifica M/D/1 con λ = 0.8, μ = 1.0 (σ = 0). Lq debe ser la mitad de M/M/1."""
        modelo = ModeloMD1(lambda_rate=0.8, mu_rate=1.0, time_unit="minutos")
        self.assertAlmostEqual(modelo.rho, 0.8, places=4)
        self.assertAlmostEqual(modelo.p0(), 0.2, places=4)
        self.assertAlmostEqual(modelo.lq(), 1.6, places=4)  # 3.2 / 2 = 1.6
        self.assertAlmostEqual(modelo.ls(), 2.4, places=4)  # 1.6 + 0.8 = 2.4
        self.assertAlmostEqual(modelo.wq(), 2.0, places=4)  # 1.6 / 0.8 = 2.0
        self.assertAlmostEqual(modelo.ws(), 3.0, places=4)  # 2.4 / 0.8 = 3.0

    def test_mg1_pollaczek_khinchine(self):
        """Verifica M/G/1 con λ = 0.8, μ = 1.0, σ = 0.5 y el factor Curry-Feldman."""
        modelo = ModeloMG1(lambda_rate=0.8, mu_rate=1.0, sigma=0.5, time_unit="minutos")
        self.assertAlmostEqual(modelo.lq(), 2.0, places=4)
        self.assertAlmostEqual(modelo.ls(), 2.8, places=4)
        self.assertAlmostEqual(modelo.wq(), 2.5, places=4)
        self.assertAlmostEqual(modelo.ws(), 3.5, places=4)
        self.assertAlmostEqual(modelo.factor_curry_feldman(), 0.625, places=4)

    def test_instabilidad(self):
        """Verifica que el sistema rechace configuraciones inestables ρ >= 1."""
        with self.assertRaises(ValueError):
            ModeloMM1(lambda_rate=10.0, mu_rate=5.0)  # ρ = 2.0
        with self.assertRaises(ValueError):
            ModeloMMs(lambda_rate=12.0, mu_rate=4.0, s=3)  # ρ = 1.0

    def test_costos_y_optimizacion(self):
        """Verifica el cálculo de costos y la búsqueda de servidores óptimos s*."""
        modelo = ModeloMM1(
            lambda_rate=0.8, mu_rate=1.0, time_unit="horas",
            cq=10.0, cs=20.0, cw=5.0
        )
        costos = modelo.calcular_costos()
        # CS = 20 * 1 = 20
        self.assertAlmostEqual(costos["CS"], 20.0, places=4)
        # CQ = 10 * Lq = 10 * 3.2 = 32.0
        self.assertAlmostEqual(costos["CQ"], 32.0, places=4)

        opt = optimizar_servidores(
            lambda_rate=15.0, mu_rate=6.0, cq=100.0, cs=50.0, time_unit="horas"
        )
        self.assertGreaterEqual(opt["s_optimo"], 3)
        self.assertIn("reporte_texto", opt)

    def test_comparador_modelos(self):
        """Verifica que el comparador ejecute la evaluación multimodelo correctamente."""
        res = comparar_modelos(lambda_rate=0.8, mu_rate=1.0, s=1, sigma=0.5)
        self.assertEqual(len(res["modelos"]), 3)
        self.assertIn("TABLA COMPARATIVA", res["reporte_texto"])


if __name__ == "__main__":
    unittest.main()
