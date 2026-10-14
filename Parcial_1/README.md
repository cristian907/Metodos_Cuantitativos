# Parcial I: Teoría de Líneas de Espera (Teoría de Colas)

> **Universidad José Antonio Páez (UJAP)**  
> **Facultad de Ingeniería**  
> **Materia:** Métodos Cuantitativos
> **Unidad I / Parcial I:** Modelado, Simulación y Optimización de Teoría de Colas  

---

## 📌 1. Introducción

La **Teoría de Líneas de Espera** (o Teoría de Colas) es una rama de la investigación de operaciones y la matemática aplicada que estudia el comportamiento de sistemas estocásticos donde clientes o transacciones solicitan un servicio a uno o más servidores con capacidad finita.

El propósito principal de los modelos en esta carpeta es proporcionar soluciones computacionales exactas basadas en **Programación Orientada a Objetos (POO)** para evaluar:
1. **Nivel de congestión:** Longitud promedio de la cola ($L_q$) y de clientes en el sistema ($L_s$).
2. **Tiempos de retardo:** Tiempo promedio de espera en cola ($W_q$) y de permanencia total ($W_s$).
3. **Probabilidades de estado:** Probabilidad de que el sistema esté desocupado ($P_0$), probabilidad de encontrar $n$ clientes ($P_n$), y probabilidad de demora ($P_w$).
4. **Análisis económico:** Equilibrio óptimo entre el costo de proveer servidores ($C_s$) y el costo de oportunidad o penalización por la espera de los clientes ($C_q$), hallando el número óptimo de servidores ($s^*$).

---

## 📐 2. Fundamentos y Notación de Kendall

Los sistemas se clasifican bajo la notación estándar de Kendall:

$$\mathbf{A / B / s / K / N / D}$$

* **$A$ (Proceso de llegadas):** $M$ (Markoviano / Poisson), $D$ (Determinista / Constante), $G$ (General).
* **$B$ (Proceso de servicio):** $M$ (Exponencial), $D$ (Determinista), $E_k$ (Erlang-$k$), $G$ (General).
* **$s$ (Número de servidores):** Canales o estaciones de servicio en paralelo ($s \ge 1$).
* **$K$ (Capacidad máxima):** Capacidad del sistema (cola + servidores). Por defecto $\infty$.
* **$N$ (Población de clientes):** Tamaño de la fuente de entrada. Por defecto $\infty$.
* **$D$ (Disciplina de servicio):** Orden de atención. Por defecto FIFO (*First In, First Out*).

### Variables y Parámetros Principales

| Parámetro | Significado | Restricción / Fórmula |
| :--- | :--- | :--- |
| $\lambda$ | Tasa media de llegadas | Clientes por unidad de tiempo ($\lambda > 0$) |
| $\mu$ | Tasa media de servicio por servidor | Clientes atendidos por unidad de tiempo ($\mu > 0$) |
| $s$ | Número de servidores en paralelo | Entero $s \ge 1$ |
| $r = \frac{\lambda}{\mu}$ | Intensidad de tráfico | Demanda total expresada en servidores equivalentes |
| $\rho = \frac{\lambda}{s\mu}$ | Factor de utilización del sistema | Proporción de ocupación global (debe ser $\rho < 1$) |

### Relaciones Operativas de Little

1. **Ley de Little en el sistema:** $L_s = \lambda \cdot W_s$
2. **Ley de Little en la cola:** $L_q = \lambda \cdot W_q$
3. **Relación entre tiempos:** $W_s = W_q + \frac{1}{\mu}$
4. **Relación entre longitudes:** $L_s = L_q + \frac{\lambda}{\mu} = L_q + r$
5. **Cola no vacía (Busy Queue):**
   * $L_b = \frac{L_q}{P_w}$ (Promedio de clientes cuando hay cola)
   * $W_b = \frac{W_q}{P_w}$ (Tiempo promedio de espera para clientes que sí esperan)

---

## 🗂️ 3. Estructura de Archivos y Directorios

```text
Parcial_1/
├── README.md                              <- Este documento explicativo
├── calculadora_colas_mm1_mms.py          <- Script interactivo para modelos M/M/1 y M/M/s
├── calculadora_colas_costos.py           <- Modelos M/M/1 y M/M/s con análisis y optimización de costos
├── calculadora_colas_no_exponenciales.py <- Modelos M/G/1, M/D/1, M/Ek/1, G/G/s, Erlang-B, prioridades y fallas
├── calculadora_colas_unificada.py        <- Script monolítico todo-en-uno con comparador y suite de pruebas
└── sistema_colas_poo/                    <- Paquete modular orientado a objetos (arquitectura limpia)
    ├── __init__.py
    ├── main.py                           <- Punto de entrada CLI principal
    ├── cli.py                            <- Menús interactivos y visualización en consola
    ├── test_sistema.py                   <- Suite completa de pruebas unitarias (unittest)
    ├── models/                           <- Clases de los modelos matemáticos
    │   ├── __init__.py
    │   ├── base_model.py                 <- Clase abstracta BaseQueueModel
    │   ├── mm1_model.py                  <- Implementación del modelo M/M/1
    │   ├── mms_model.py                  <- Implementación del modelo M/M/s
    │   ├── md1_model.py                  <- Implementación del modelo M/D/1
    │   └── mg1_model.py                  <- Implementación del modelo M/G/1
    ├── services/                         <- Lógica de negocio y optimización
    │   ├── __init__.py
    │   ├── cost_optimizer.py             <- Algoritmo de optimización de número de servidores (s*)
    │   └── model_comparator.py          <- Comparación multimodelo lado a lado
    └── utils/                            <- Helpers y formato
        ├── __init__.py
        └── helpers.py                    <- Parsing decimal y formato de cadenas
```

---

## ⚙️ 4. Modelos Matemáticos Implementados

### 4.1. Modelo $M/M/1$
* **Hipótesis:** 1 servidor, llegadas Poisson, servicio exponencial, cola ilimitada, disciplina FIFO.
* **Fórmulas clave:**
  $$\rho = \frac{\lambda}{\mu}, \quad P_0 = 1 - \rho, \quad P_n = (1 - \rho)\rho^n$$
  $$L_s = \frac{\lambda}{\mu - \lambda} = \frac{\rho}{1 - \rho}, \quad L_q = \frac{\lambda^2}{\mu(\mu - \lambda)} = \frac{\rho^2}{1 - \rho}$$
  $$W_s = \frac{1}{\mu - \lambda}, \quad W_q = \frac{\lambda}{\mu(\mu - \lambda)}$$
  $$P(W_s > t) = e^{-\mu(1 - \rho)t}, \quad P(W_q > t) = \rho e^{-\mu(1 - \rho)t}$$

---

### 4.2. Modelo $M/M/s$ (Erlang C)
* **Hipótesis:** $s$ servidores idénticos en paralelo, una única cola central.
* **Probabilidad de sistema vacío ($P_0$):**
  $$P_0 = \left[ \sum_{n=0}^{s-1} \frac{(\lambda / \mu)^n}{n!} + \frac{(\lambda / \mu)^s}{s! (1 - \rho)} \right]^{-1}$$
* **Probabilidad de esperar en cola ($P_w$, Erlang C):**
  $$P_w = \frac{(\lambda / \mu)^s}{s! (1 - \rho)} P_0$$
* **Medidas de desempeño:**
  $$L_q = \frac{P_w \cdot \rho}{1 - \rho}, \quad W_q = \frac{L_q}{\lambda}, \quad W_s = W_q + \frac{1}{\mu}, \quad L_s = L_q + \frac{\lambda}{\mu}$$
* **Probabilidad de espera en cola mayor a $t$:**
  $$P(W_q > t) = P_w \cdot e^{-s\mu(1 - \rho)t}$$

---

### 4.3. Modelo $M/D/1$ (Servicio Determinista)
* **Hipótesis:** 1 servidor con tiempo de servicio constante ($\sigma = 0$).
* **Fórmulas:**
  $$L_q = \frac{\rho^2}{2(1 - \rho)} = \frac{L_{q(M/M/1)}}{2}$$
  $$W_q = \frac{L_q}{\lambda}, \quad W_s = W_q + \frac{1}{\mu}, \quad L_s = L_q + \rho$$
  *(Nótese que la varianza nula reduce exactamente a la mitad la cola promedio frente a $M/M/1$)*.

---

### 4.4. Modelo $M/G/1$ (Fórmula de Pollaczek-Khinchine)
* **Hipótesis:** 1 servidor con tiempo de servicio de distribución general con media $1/\mu$ y varianza conocida $\sigma^2$.
* **Fórmula P-K:**
  $$L_q = \frac{\lambda^2 \sigma^2 + \rho^2}{2(1 - \rho)}$$
  $$W_q = \frac{L_q}{\lambda}, \quad W_s = W_q + \frac{1}{\mu}, \quad L_s = L_q + \rho$$

---

### 4.5. Modelos Avanzados Adicionales (`calculadora_colas_no_exponenciales.py`)
1. **$G/G/s$ (Aproximación Allen-Cunneen):** Llegadas y servicio con coeficientes de variación $C_a^2$ y $C_s^2$.
2. **$M/E_k/1$:** Tiempo de servicio con distribución Erlang-$k$.
3. **Erlang-B ($M/G/s/s$):** Sistema sin cola (los clientes que llegan cuando todos los servidores están ocupados se pierden).
4. **Prioridades:**
   * *Sin interrupción (Non-preemptive):* Los clientes de mayor prioridad van al frente de la cola pero no interrumpen una atención en curso.
   * *Con interrupción (Preemptive):* Los clientes prioritarios suspenden la atención de clientes de menor prioridad.
5. **Averías y Reparaciones:** Servidores con tasas de fallo ($\lambda_f$) y tiempo de reparación exponencial ($\mu_r$).

---

### 4.6. Estructura de Costos y Optimización

El costo total por unidad de tiempo del sistema se define como:
$$CT(s) = \text{Costo de Servidores} + \text{Costo de Espera} + \text{Costo de Atención}$$
$$CT(s) = s \cdot C_s + L_q \cdot C_q + \lambda \cdot C_w$$

* **Costo de servidores ocupados:** $CTS = r \cdot C_s$
* **Costo de servidores ociosos:** $CTS_{ocioso} = (s - r) \cdot C_s$
* **Costo de espera en cola:** $CTQ = L_q \cdot C_q$

La función de optimización evalúa $CT(s)$ iterativamente a partir del mínimo número de servidores que garantiza estabilidad ($s_{min} = \lfloor \lambda/\mu \rfloor + 1$) hasta encontrar el valor $s^*$ donde $CT(s^*)$ es mínimo.

---

## 🚀 5. Guía de Ejecución

### Requisitos Previos
* **Python 3.8+**
* No requiere librerías externas de terceros (utiliza exclusivamente la biblioteca estándar: `math`, `sys`, `unittest`, `typing`, `pathlib`).

### 5.1. Ejecutar el Sistema Modular (Recomendado)
Para iniciar la interfaz interactiva completa:
```bash
python3 sistema_colas_poo/main.py
```

### 5.2. Ejecutar la Suite de Pruebas Unitarias
Para validar matemáticamente todos los modelos y servicios contra casos resueltos:
```bash
python3 sistema_colas_poo/test_sistema.py
```
O desde el módulo unificado:
```bash
python3 -m unittest calculadora_colas_unificada.py
```

### 5.3. Ejecutar las Calculadoras Específicas
Si se desea interactuar con un script enfocado en un subtema particular:
* **Modelos básicos $M/M/1$ y $M/M/s$:**
  ```bash
  python3 calculadora_colas_mm1_mms.py
  ```
* **Optimización económica y costos:**
  ```bash
  python3 calculadora_colas_costos.py
  ```
* **Modelos no exponenciales, prioridades y fallas:**
  ```bash
  python3 calculadora_colas_no_exponenciales.py
  ```
* **Versión todo-en-uno:**
  ```bash
  python3 calculadora_colas_unificada.py
  ```