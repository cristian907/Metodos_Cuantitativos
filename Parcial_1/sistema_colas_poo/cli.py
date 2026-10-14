#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Interfaz de Línea de Comandos (CLI) para la Calculadora de Teoría de Colas.
Proporciona menús interactivos, entrada asistida de datos, consultas de probabilidad,
desglose de costos y exportación de reportes.
"""

import sys
import unittest
from typing import Optional, List

from .utils.helpers import parse_float, print_header, print_divider
from .models.mm1_model import ModeloMM1
from .models.mms_model import ModeloMMs
from .models.md1_model import ModeloMD1
from .models.mg1_model import ModeloMG1
from .models.base_model import BaseQueueModel
from .services.cost_optimizer import optimizar_servidores
from .services.model_comparator import comparar_modelos


def pedir_float(prompt: str, min_val: Optional[float] = None, allow_zero: bool = False) -> float:
    """Solicita un número flotante por consola con validación de rango."""
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
            print("   [Error] Entrada inválida. Ingrese un número válido (ej. 5 u 0.8).")


def pedir_int(prompt: str, min_val: int = 1) -> int:
    """Solicita un número entero por consola con validación."""
    while True:
        try:
            val_str = input(prompt)
            val = int(val_str.strip())
            if val < min_val:
                print(f"   [Error] Debe ingresar un entero mayor o igual a {min_val}.")
                continue
            return val
        except ValueError:
            print("   [Error] Entrada inválida. Ingrese un número entero.")


def pedir_costos() -> tuple:
    """Solicita opcionalmente los parámetros de costo Cq, Cs, Cw."""
    print("\n  --- Configuración Opcional de Costos del Sistema ---")
    desea = input("  ¿Desea incluir análisis de costos? (s/n, por defecto 'n'): ").strip().lower()
    if desea in ['s', 'si', 'y', 'yes']:
        cs = pedir_float("  • Costo por servidor por unidad de tiempo (Cs) [>= 0]: ", min_val=0.0, allow_zero=True)
        cq = pedir_float("  • Costo de espera por cliente por unidad de tiempo (Cq) [>= 0]: ", min_val=0.0, allow_zero=True)
        cw = pedir_float("  • Costo de atención por cliente atendido (Cw) [>= 0]: ", min_val=0.0, allow_zero=True)
        return cs, cq, cw
    return 0.0, 0.0, 0.0


def submenu_consultas(modelo: BaseQueueModel):
    """Menú secundario para consultas de probabilidad y exportación de reportes."""
    consultas_realizadas: List[str] = []
    k_max_tabla: Optional[int] = None

    while True:
        print("\n" + "-" * 72)
        print("                   MENÚ DE CONSULTAS ADICIONALES")
        print("-" * 72)
        print("  1. Consultar P(N = n)   - Probabilidad de exactamente n clientes")
        print("  2. Consultar P(N <= n)  - Probabilidad de n o menos clientes")
        print("  3. Consultar P(N > n)   - Probabilidad de más de n clientes")
        print("  4. Consultar P(Wq > t)  - Probabilidad de esperar en cola más de un tiempo t")
        print("  5. Consultar P(Ws > t)  - Probabilidad de permanencia en sistema más de t")
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
            nombre_archivo = input("  Nombre del archivo (por defecto 'reporte_cola.txt'): ").strip()
            if not nombre_archivo:
                nombre_archivo = "reporte_cola.txt"
            filepath = modelo.exportar_txt(
                filepath=nombre_archivo,
                consultas_adicionales=consultas_realizadas,
                k_max_tabla=k_max_tabla
            )
            print(f"\n   >>> ÉXITO: Reporte exportado correctamente a '{filepath}'")
        else:
            print("   [Error] Opción no válida.")


def menu_principal():
    """Función de bucle del menú principal."""
    while True:
        print_header("SISTEMA DE RESOLUCIÓN DE TEORÍA DE COLAS (POO)")
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
            lam = pedir_float("  • Tasa media de llegadas (λ) [clientes/unidad]: ", min_val=0.0)
            mu = pedir_float("  • Tasa media de servicio (μ) [clientes/unidad]: ", min_val=0.0)
            tu = input("  • Unidad de tiempo (ej. 'minutos', 'horas') [por defecto 'minutos']: ").strip() or "minutos"
            cs, cq, cw = pedir_costos()

            try:
                modelo = ModeloMM1(lambda_rate=lam, mu_rate=mu, time_unit=tu, cq=cq, cs=cs, cw=cw)
                print("\n" + modelo.generar_reporte_texto())
                submenu_consultas(modelo)
            except ValueError as e:
                print(f"\n   [ERROR DE ESTABILIDAD]: {e}\n")

        elif opcion == "2":
            print_header("CONFIGURACIÓN DE MODELO M/M/s")
            lam = pedir_float("  • Tasa media de llegadas (λ) [clientes/unidad]: ", min_val=0.0)
            mu = pedir_float("  • Tasa media de servicio por servidor (μ) [clientes/unidad]: ", min_val=0.0)
            s = pedir_int("  • Número de servidores en paralelo (s >= 1): ", min_val=1)
            tu = input("  • Unidad de tiempo (ej. 'minutos', 'horas') [por defecto 'minutos']: ").strip() or "minutos"
            cs, cq, cw = pedir_costos()

            try:
                modelo = ModeloMMs(lambda_rate=lam, mu_rate=mu, s=s, time_unit=tu, cq=cq, cs=cs, cw=cw)
                print("\n" + modelo.generar_reporte_texto())
                submenu_consultas(modelo)
            except ValueError as e:
                print(f"\n   [ERROR DE ESTABILIDAD]: {e}\n")

        elif opcion == "3":
            print_header("CONFIGURACIÓN DE MODELO M/D/1 (SERVICIO CONSTANTE)")
            lam = pedir_float("  • Tasa media de llegadas (λ) [clientes/unidad]: ", min_val=0.0)
            mu = pedir_float("  • Tasa media de servicio (μ) [clientes/unidad]: ", min_val=0.0)
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
            lam = pedir_float("  • Tasa media de llegadas (λ) [clientes/unidad]: ", min_val=0.0)
            mu = pedir_float("  • Tasa media de servicio (μ) [clientes/unidad]: ", min_val=0.0)
            sigma = pedir_float("  • Desviación estándar del tiempo de servicio (σ) [>= 0]: ", min_val=0.0, allow_zero=True)
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
            print_header("EJECUTANDO PRUEBAS AUTOMATIZADAS DEL SISTEMA")
            from .test_sistema import TestSistemaColas
            suite = unittest.TestLoader().loadTestsFromTestCase(TestSistemaColas)
            runner = unittest.TextTestRunner(verbosity=2)
            runner.run(suite)

        else:
            print("   [Error] Opción no válida. Por favor seleccione un número del 0 al 7.")

        input("\n  Presione ENTER para continuar...")
