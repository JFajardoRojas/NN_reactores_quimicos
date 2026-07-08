# -*- coding: utf-8 -*-
"""
============================================================================
 DEMO — Herramientas de IA para el dimensionamiento de un reactor PFR
 Datos multi-fidelidad + modelo FÍSICO-INFORMADO vs modelo CAJA-NEGRA
============================================================================

Idea central de la clase (20 min, pregrado ing. química)
------------------------------------------------------------
Queremos estimar el VOLUMEN de un reactor de flujo pistón (PFR) necesario
para alcanzar una conversión objetivo X*, gastando la MENOR cantidad de
datos experimentales posible (los experimentos son caros y lentos).

Reacción:  A -> B   (fase líquida, isotérmica, flujo volumétrico constante)

Lo que SIEMPRE conocemos (física dura):  el BALANCE MOLAR del PFR
        dX/dV = (-r_A) / F_A0
Lo que NO conocemos:  la LEY DE VELOCIDAD  -r_A(C_A)

Dos filosofías de modelado que comparamos:

  (1) MODELO FÍSICO-INFORMADO  (Universal ODE / UDE)
      - Respeta el balance molar (lo metemos "a mano" en la ecuación).
      - Una RED NEURONAL aprende solo la parte desconocida: la ley -r_A(C_A),
        entrenada DENTRO de la EDO (aquí está el machine learning).
      - Usa datos MULTI-FIDELIDAD:
          * baja fidelidad  = modelo barato y SESGADO (cinética 1er orden),
                              muchísimos puntos casi gratis  -> lo usa como "prior".
          * alta fidelidad  = pocos experimentos reales (con ruido) -> corrige el prior.
      => Necesita MUY POCOS experimentos porque la física hace el trabajo pesado.

  (2) MODELO CAJA-NEGRA  (black-box)
      - Aprende directamente el mapa V -> X con una red neuronal.
      - NO sabe nada del balance molar.
      => Necesita MUCHOS más datos para "descubrir" el patrón, y extrapola mal.

El script:
  - genera los datos multi-fidelidad y los guarda en CSV,
  - entrena ambos modelos con distinta cantidad de experimentos,
  - compara error de conversión y error en el VOLUMEN predicho,
  - guarda 4 figuras y una tabla de resultados.

Autor: material de apoyo para la clase de Fernando.
============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import os

RNG_GLOBAL = np.random.default_rng(7)
OUT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# 1) PARÁMETROS DEL SISTEMA (la "realidad" que la naturaleza esconde)
# ---------------------------------------------------------------------------
C_A0 = 2.0        # mol/L  concentración de entrada de A
V0   = 1.0        # L/s    flujo volumétrico
F_A0 = C_A0 * V0  # mol/s  flujo molar de entrada de A  (= 2.0)

# Ley de velocidad VERDADERA: tipo Langmuir-Hinshelwood (NO es simple 1er orden)
#   -r_A = k * C_A^2 / (1 + K_ads * C_A)
K_TRUE   = 0.18   # L/(mol·s)
KADS_TRUE = 0.55  # L/mol

# Modelo BARATO de baja fidelidad: supone cinética de 1er orden (sesgado)
#   -r_A ≈ k_lf * C_A
K_LF = 0.085      # 1/s  (ajustado "a ojo" por el ingeniero, queda sesgado ~18%)

X_TARGET = 0.80   # conversión objetivo para dimensionar el reactor
V_MAX    = 25.0   # L    dominio del reactor para simular / graficar


def rate_true(C_A):
    """Ley de velocidad verdadera -r_A(C_A) [mol/(L·s)]."""
    return K_TRUE * C_A**2 / (1.0 + KADS_TRUE * C_A)


def rate_lf(C_A):
    """Ley de velocidad de BAJA FIDELIDAD (barata, sesgada): 1er orden."""
    return K_LF * C_A


# ---------------------------------------------------------------------------
# 2) INTEGRADOR DEL BALANCE MOLAR DEL PFR  (RK4, paso fijo)
#    dX/dV = (-r_A(C_A)) / F_A0 ,  con  C_A = C_A0 (1 - X)
# ---------------------------------------------------------------------------
def integrate_pfr(rate_fn, V_grid):
    """Integra el perfil X(V) usando una ley de velocidad dada rate_fn(C_A)."""
    X = np.zeros_like(V_grid, dtype=float)
    x = 0.0
    for i in range(1, len(V_grid)):
        h = V_grid[i] - V_grid[i - 1]

        def f(xx):
            C_A = C_A0 * (1.0 - xx)
            val = rate_fn(C_A)
            return float(np.asarray(val).reshape(-1)[0]) / F_A0

        k1 = f(x)
        k2 = f(x + 0.5 * h * k1)
        k3 = f(x + 0.5 * h * k2)
        k4 = f(x + h * k3)
        x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        x = min(max(x, 0.0), 0.999)   # conversión física acotada
        X[i] = x
    return X


V_DENSE = np.linspace(0.0, V_MAX, 400)
X_TRUE_DENSE = integrate_pfr(rate_true, V_DENSE)
X_LF_DENSE   = integrate_pfr(rate_lf,   V_DENSE)


def volume_for_conversion(V_grid, X_grid, X_star):
    """Interpola el volumen al que se alcanza la conversión X_star."""
    if X_grid.max() < X_star:
        return np.nan
    return float(np.interp(X_star, X_grid, V_grid))


V_STAR_TRUE = volume_for_conversion(V_DENSE, X_TRUE_DENSE, X_TARGET)
V_STAR_LF   = volume_for_conversion(V_DENSE, X_LF_DENSE,   X_TARGET)


# ---------------------------------------------------------------------------
# 3) GENERACIÓN DE DATOS MULTI-FIDELIDAD  ->  CSV
# ---------------------------------------------------------------------------
def generar_datos():
    rng = np.random.default_rng(20260707)

    # --- BAJA FIDELIDAD: barato, abundante (modelo 1er orden), leve ruido ---
    V_lf = np.linspace(0.5, V_MAX, 60)
    X_lf = integrate_pfr(rate_lf, np.concatenate([[0.0], V_lf]))[1:]
    X_lf = np.clip(X_lf + rng.normal(0, 0.004, size=X_lf.shape), 0, 0.999)
    df_lf = pd.DataFrame({
        "V_L": V_lf, "X": X_lf,
        "C_A_mol_L": C_A0 * (1 - X_lf),
        "fidelidad": "baja",
    })

    # --- ALTA FIDELIDAD: caro, escaso (experimentos reales), ruido de medición ---
    V_hf_pool = np.array([1.5, 3.5, 5.5, 8.0, 11.0, 14.0, 17.0, 20.0, 22.0])
    X_hf = integrate_pfr(rate_true,
                         np.concatenate([[0.0], V_hf_pool]))[1:]
    X_hf = np.clip(X_hf + rng.normal(0, 0.03, size=X_hf.shape), 0, 0.999)
    df_hf = pd.DataFrame({
        "V_L": V_hf_pool, "X": X_hf,
        "C_A_mol_L": C_A0 * (1 - X_hf),
        "fidelidad": "alta",
    })

    # --- SET DE PRUEBA (verdad sin ruido, para medir error) ---
    V_test = np.linspace(0.5, V_MAX, 80)
    X_test = integrate_pfr(rate_true, np.concatenate([[0.0], V_test]))[1:]
    df_test = pd.DataFrame({"V_L": V_test, "X": X_test})

    df_lf.to_csv(os.path.join(OUT, "datos_baja_fidelidad.csv"), index=False)
    df_hf.to_csv(os.path.join(OUT, "datos_alta_fidelidad.csv"), index=False)
    df_test.to_csv(os.path.join(OUT, "datos_prueba.csv"), index=False)
    return df_lf, df_hf, df_test


DF_LF, DF_HF, DF_TEST = generar_datos()


# ---------------------------------------------------------------------------
# 4) MODELO FÍSICO-INFORMADO = RED NEURONAL DENTRO DE LA EDO  (Universal ODE / UDE)
#
#    La ÚNICA parte desconocida es la ley de velocidad -r_A(C_A). La reemplazamos
#    por una RED NEURONAL pequeña NN_θ(C_A) (aquí ESTÁ el machine learning), pero
#    la escribimos de forma multiplicativa sobre el prior barato de baja fidelidad:
#
#        -r_A(C_A) = r_LF(C_A) · exp( NN_θ(C_A) )
#
#    - Si la red vale 0  =>  recuperamos EXACTAMENTE el prior de 1er orden.
#    - El exp() garantiza que la velocidad sea SIEMPRE positiva (la EDO nunca
#      produce perfiles absurdos, aunque la red aún no esté entrenada).
#    - La red es un MLP: 1 capa oculta (tanh). Sus pesos θ son lo que se APRENDE.
#
#    ENTRENAMIENTO "physics-informed": metemos la red DENTRO del balance molar,
#    integramos el PFR, y ajustamos los pesos θ para que X(V) reproduzca los pocos
#    experimentos caros. La física (balance molar) hace el trabajo pesado; la red
#    solo tiene que aprender la forma de la velocidad.
#
#    MULTI-FIDELIDAD: los pesos arrancan ~0 (=> modelo barato) y la regularización
#    L2 (lam·θ) los mantiene pequeños salvo que los datos exijan corregir. Es decir,
#    fusionamos el modelo barato (prior) con los pocos datos caros.
# ---------------------------------------------------------------------------

H_NN = 3                    # neuronas en la capa oculta de la red (pequeña: pocos datos)
CA_MU, CA_SD = 1.0, 0.55    # escalado de la entrada C_A (condiciona el entrenamiento)


def _nn_forward(theta, C_A):
    """MLP de 1 capa oculta (tanh). Devuelve el log-multiplicador de la velocidad.
    θ = [W1(H), b1(H), W2(H), b2(1)]  ->  3·H + 1 pesos."""
    W1 = theta[:H_NN]
    b1 = theta[H_NN:2 * H_NN]
    W2 = theta[2 * H_NN:3 * H_NN]
    b2 = theta[3 * H_NN]
    xs = (np.atleast_1d(C_A) - CA_MU) / CA_SD
    h = np.tanh(xs[:, None] * W1[None, :] + b1[None, :])   # (m, H)
    return h @ W2 + b2                                      # (m,)


def rate_pinn(theta, C_A):
    """Ley de velocidad aprendida por la red: prior barato · exp(red). Siempre > 0."""
    return rate_lf(np.atleast_1d(C_A)) * np.exp(_nn_forward(theta, C_A))


def fit_fisico_informado(V_hf, X_hf, lam=0.4, seed=0):
    """Entrena la RED dentro del balance molar con pocos experimentos caros.
    'lam' es la regularización L2 sobre los pesos (los mantiene cerca del prior)."""
    n_theta = 3 * H_NN + 1
    rng = np.random.default_rng(seed)
    theta0 = rng.normal(0, 0.05, size=n_theta)   # pesos pequeños => arranca en el prior
    V_int = np.linspace(0.0, V_MAX, 70)

    def residuals(theta):
        Xpred_grid = integrate_pfr(lambda CA: rate_pinn(theta, CA), V_int)
        Xpred_hf = np.interp(V_hf, V_int, Xpred_grid)
        r_data = (Xpred_hf - X_hf) / HF_SIGMA          # ajuste a los datos caros
        r_reg = lam * theta                            # regularización L2 de los pesos
        return np.concatenate([r_data, r_reg])

    sol = least_squares(residuals, theta0, method="trf", max_nfev=150)
    return sol.x


def predict_fisico_informado(theta, V_grid):
    V_int = np.linspace(0.0, V_MAX, 160)
    Xg = integrate_pfr(lambda CA: rate_pinn(theta, CA), V_int)
    return np.interp(V_grid, V_int, Xg)


# ---------------------------------------------------------------------------
# 5) MODELO CAJA-NEGRA: red neuronal que aprende V -> X sin física
# ---------------------------------------------------------------------------
def fit_caja_negra(V_hf, X_hf, seed=0):
    sc = StandardScaler().fit(V_hf.reshape(-1, 1))
    mlp = MLPRegressor(hidden_layer_sizes=(32, 32), activation="tanh",
                       solver="lbfgs", alpha=1e-3, max_iter=4000,
                       random_state=seed)
    mlp.fit(sc.transform(V_hf.reshape(-1, 1)), X_hf)
    return (sc, mlp)


def predict_caja_negra(model, V_grid):
    sc, mlp = model
    y = mlp.predict(sc.transform(np.asarray(V_grid).reshape(-1, 1)))
    return np.clip(y, 0.0, 0.999)


# ---------------------------------------------------------------------------
# 6) EXPERIMENTO: barrido de cantidad de datos + métricas
# ---------------------------------------------------------------------------
Xtest_true = DF_TEST["X"].values
Vtest = DF_TEST["V_L"].values

HF_SIGMA = 0.03    # ruido de medición de los experimentos de alta fidelidad (~3%)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b))**2)))


def generar_experimentos(n, seed):
    """Genera n experimentos de alta fidelidad: repartidos en el reactor,
    con una NUEVA realización de ruido de medición (para promediar el error)."""
    rng = np.random.default_rng(seed)
    V_loc = np.linspace(1.5, 22.0, n)
    Xg = integrate_pfr(rate_true, np.concatenate([[0.0], V_loc]))[1:]
    X_noisy = np.clip(Xg + rng.normal(0, HF_SIGMA, size=Xg.shape), 0, 0.999)
    return V_loc, X_noisy


# --- Barrido: para cada nº de experimentos promediamos sobre N_TRIALS
#     realizaciones de ruido distintas (así la curva de error es estable). ---
N_LIST = [4, 5, 6, 8, 10, 14]
N_TRIALS = 8

# El barrido es REANUDABLE: cada fila se guarda apenas se calcula, y si el
# script se interrumpe, al re-ejecutarlo continúa donde quedó.
RAW = os.path.join(OUT, "resultados_crudos_ude.csv")
done = set()
if os.path.exists(RAW):
    _prev = pd.read_csv(RAW)
    done = {(int(a), int(b)) for a, b in zip(_prev.n_exp, _prev.trial)}

for n in N_LIST:
    for t in range(N_TRIALS):
        if (n, t) in done:
            continue
        Vh, Xh = generar_experimentos(n, seed=1000 * n + t)

        # (1) físico-informado
        p = fit_fisico_informado(Vh, Xh, seed=t)
        Xpi = predict_fisico_informado(p, Vtest)
        rmse_pi = rmse(Xpi, Xtest_true)
        Vpi = volume_for_conversion(Vtest, Xpi, X_TARGET)
        err_vol_pi = abs(Vpi - V_STAR_TRUE) / V_STAR_TRUE * 100 if not np.isnan(Vpi) else np.nan

        # (2) caja-negra
        m = fit_caja_negra(Vh, Xh, seed=t)
        Xbb = predict_caja_negra(m, Vtest)
        rmse_bb = rmse(Xbb, Xtest_true)
        Vbb = volume_for_conversion(Vtest, Xbb, X_TARGET)
        err_vol_bb = abs(Vbb - V_STAR_TRUE) / V_STAR_TRUE * 100 if not np.isnan(Vbb) else np.nan

        row = dict(n_exp=n, trial=t,
                   rmse_fisico=rmse_pi, rmse_cajanegra=rmse_bb,
                   errvol_fisico_pct=err_vol_pi, errvol_cajanegra_pct=err_vol_bb)
        pd.DataFrame([row]).to_csv(RAW, mode="a", index=False,
                                   header=not os.path.exists(RAW))

res = pd.read_csv(RAW).sort_values(["n_exp", "trial"])
agg = res.groupby("n_exp").mean(numeric_only=True).reset_index().drop(columns=["trial"])
sd = res.groupby("n_exp").std(numeric_only=True).reset_index()
agg.to_csv(os.path.join(OUT, "resumen_resultados.csv"), index=False)


# ---------------------------------------------------------------------------
# 7) FIGURAS
# ---------------------------------------------------------------------------
plt.rcParams.update({"figure.dpi": 130, "font.size": 11})
C_PI, C_BB, C_TRUE, C_LF = "#1b7837", "#c51b7d", "#2166ac", "#f1a340"

# --- Fig 1: error de conversión vs número de experimentos ---
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.fill_between(agg.n_exp, np.maximum(agg.rmse_fisico - sd.rmse_fisico, 1e-4),
                agg.rmse_fisico + sd.rmse_fisico, color=C_PI, alpha=0.15)
ax.fill_between(agg.n_exp, np.maximum(agg.rmse_cajanegra - sd.rmse_cajanegra, 1e-4),
                agg.rmse_cajanegra + sd.rmse_cajanegra, color=C_BB, alpha=0.15)
ax.plot(agg.n_exp, agg.rmse_fisico, "o-", color=C_PI, lw=2.2, label="Físico-informado (+ multi-fidelidad)")
ax.plot(agg.n_exp, agg.rmse_cajanegra, "s--", color=C_BB, lw=2.2, label="Caja-negra (sin física)")
ax.set_xlabel("Nº de experimentos de alta fidelidad")
ax.set_ylabel("RMSE de la conversión X (set de prueba)")
ax.set_yscale("log")
ax.set_title("Eficiencia de datos: menos experimentos con física", fontsize=12)
ax.grid(alpha=0.3, which="both")
ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_error_vs_datos.png")); plt.close(fig)

# --- Fig 2: error en el VOLUMEN predicho vs número de experimentos ---
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.fill_between(agg.n_exp, np.maximum(agg.errvol_fisico_pct - sd.errvol_fisico_pct, 0),
                agg.errvol_fisico_pct + sd.errvol_fisico_pct, color=C_PI, alpha=0.15)
ax.fill_between(agg.n_exp, np.maximum(agg.errvol_cajanegra_pct - sd.errvol_cajanegra_pct, 0),
                agg.errvol_cajanegra_pct + sd.errvol_cajanegra_pct, color=C_BB, alpha=0.15)
ax.plot(agg.n_exp, agg.errvol_fisico_pct, "o-", color=C_PI, lw=2.2, label="Físico-informado")
ax.plot(agg.n_exp, agg.errvol_cajanegra_pct, "s--", color=C_BB, lw=2.2, label="Caja-negra")
ax.axhline(5, color="gray", ls=":", label="Tolerancia de diseño (5%)")
ax.set_xlabel("Nº de experimentos de alta fidelidad")
ax.set_ylabel(f"Error en el volumen para X={X_TARGET:.0%}  [%]")
ax.set_title("Error en el dimensionamiento del PFR")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_error_volumen.png")); plt.close(fig)

# --- Fig 3: perfiles X(V) con POCOS datos (n=4) ---
n_demo = 4
Vh, Xh = generar_experimentos(n_demo, seed=1000 * n_demo + 0)
p = fit_fisico_informado(Vh, Xh, seed=0)
m = fit_caja_negra(Vh, Xh, seed=0)
Xpi = predict_fisico_informado(p, V_DENSE)
Xbb = predict_caja_negra(m, V_DENSE)

fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.plot(V_DENSE, X_TRUE_DENSE, color=C_TRUE, lw=3, label="Realidad (verdad)")
ax.plot(V_DENSE, X_LF_DENSE, color=C_LF, lw=1.8, ls=":", label="Baja fidelidad (barato, sesgado)")
ax.plot(V_DENSE, Xpi, color=C_PI, lw=2.2, label="Físico-informado")
ax.plot(V_DENSE, Xbb, color=C_BB, lw=2.2, ls="--", label="Caja-negra")
ax.scatter(Vh, Xh, color="k", zorder=5, s=55, label=f"{n_demo} experimentos (alta fidelidad)")
ax.axhline(X_TARGET, color="gray", ls=":", lw=1)
ax.set_xlabel("Volumen del reactor V  [L]")
ax.set_ylabel("Conversión X")
ax.set_title(f"Perfil de conversión con solo {n_demo} experimentos")
ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_perfiles_XV.png")); plt.close(fig)

# --- Fig 4: ley de velocidad recuperada -r_A(C_A) ---
CA_grid = np.linspace(0.2, C_A0, 200)
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.plot(CA_grid, rate_true(CA_grid), color=C_TRUE, lw=3, label="Ley verdadera  (Langmuir-Hinshelwood)")
ax.plot(CA_grid, rate_lf(CA_grid), color=C_LF, lw=1.8, ls=":", label="Prior baja fidelidad (1er orden)")
ax.plot(CA_grid, rate_pinn(p, CA_grid), color=C_PI, lw=2.2, label=f"Recuperada por la red neuronal ({n_demo} exp.)")
ax.set_xlabel("Concentración $C_A$  [mol/L]")
ax.set_ylabel(r"$-r_A$  [mol/(L·s)]")
ax.set_title("El físico-informado APRENDE la ley de velocidad", fontsize=12)
ax.grid(alpha=0.3); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_ley_velocidad.png")); plt.close(fig)

# --- Fig 5: EXTRAPOLACIÓN — datos solo en baja conversión (V<=8), predecir X=0.8 ---
V_TRAIN = 8.0
Vh_e = np.linspace(1.5, V_TRAIN, 6)
Xh_e = np.clip(integrate_pfr(rate_true, np.concatenate([[0.0], Vh_e]))[1:]
               + np.random.default_rng(1).normal(0, HF_SIGMA, 6), 0, 0.999)
p_e = fit_fisico_informado(Vh_e, Xh_e, seed=1)
m_e = fit_caja_negra(Vh_e, Xh_e, seed=1)
Xpi_e = predict_fisico_informado(p_e, V_DENSE)
Xbb_e = predict_caja_negra(m_e, V_DENSE)
Vpi_e = volume_for_conversion(V_DENSE, Xpi_e, X_TARGET)
Vbb_e = volume_for_conversion(V_DENSE, Xbb_e, X_TARGET)

fig, ax = plt.subplots(figsize=(8, 5))
ax.axvspan(0, V_TRAIN, color="0.92")
ax.text(V_TRAIN / 2, 0.05, "datos\n(reactores pequeños)", ha="center", fontsize=8.5, color="0.4")
ax.text(V_TRAIN + 0.4, 0.05, "← extrapolación (sin datos) →", fontsize=8.5, color="0.4")
ax.plot(V_DENSE, X_TRUE_DENSE, color=C_TRUE, lw=3, label="Realidad (verdad)")
ax.plot(V_DENSE, X_LF_DENSE, color=C_LF, lw=1.8, ls=":", label="Baja fidelidad (1er orden)")
ax.plot(V_DENSE, Xpi_e, color=C_PI, lw=2.4, label="Físico-informado")
ax.plot(V_DENSE, Xbb_e, color=C_BB, lw=2.4, ls="--", label="Caja-negra")
ax.scatter(Vh_e, Xh_e, color="k", s=55, zorder=6, label="6 experimentos (baja conversión)")
ax.axhline(X_TARGET, color="gray", ls=":", lw=1)
ax.plot([V_STAR_TRUE], [X_TARGET], "*", color=C_TRUE, ms=15, zorder=7)
ax.annotate(f"V* real = {V_STAR_TRUE:.1f} L", (V_STAR_TRUE, X_TARGET),
            xytext=(V_STAR_TRUE - 1.5, X_TARGET - 0.13), fontsize=9, color=C_TRUE)
if not np.isnan(Vpi_e):
    ax.plot([Vpi_e], [X_TARGET], "o", color=C_PI, ms=9, zorder=7)
    ax.annotate(f"físico: {Vpi_e:.1f} L", (Vpi_e, X_TARGET),
                xytext=(Vpi_e - 0.5, X_TARGET + 0.06), fontsize=9, color=C_PI)
if not np.isnan(Vbb_e):
    ax.plot([Vbb_e], [X_TARGET], "s", color=C_BB, ms=9, zorder=7)
    ax.annotate(f"caja-negra: {Vbb_e:.1f} L (sub-dimensiona)", (Vbb_e, X_TARGET),
                xytext=(Vbb_e - 2.5, X_TARGET - 0.19), fontsize=9, color=C_BB)
ax.set_xlim(0, V_MAX); ax.set_ylim(0, 0.95)
ax.set_xlabel("Volumen del reactor V  [L]"); ax.set_ylabel("Conversión X")
ax.set_title("Extrapolación: dimensionar el reactor grande con datos de baja conversión",
             fontsize=11.5)
ax.grid(alpha=0.3); ax.legend(fontsize=8.5, loc="center right")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_extrapolacion.png")); plt.close(fig)


# ---------------------------------------------------------------------------
# 8) RESUMEN EN CONSOLA
# ---------------------------------------------------------------------------
def n_para_umbral(col, umbral):
    ok = agg[agg[col] <= umbral]
    return int(ok.n_exp.min()) if len(ok) else None

print("=" * 70)
print("RESUMEN — Dimensionamiento de PFR con IA (datos multi-fidelidad)")
print("=" * 70)
print(f"Reacción A->B, PFR isotérmico.  C_A0={C_A0} mol/L, v0={V0} L/s, F_A0={F_A0} mol/s")
print(f"Ley verdadera: -r_A = {K_TRUE}·C_A^2/(1+{KADS_TRUE}·C_A)")
print(f"Conversión objetivo X* = {X_TARGET:.0%}")
print(f"Volumen VERDADERO requerido:            V* = {V_STAR_TRUE:.2f} L")
print(f"Volumen con modelo barato (baja fid.):  V  = {V_STAR_LF:.2f} L  "
      f"(error {abs(V_STAR_LF-V_STAR_TRUE)/V_STAR_TRUE*100:.1f}%)")
print("-" * 70)
print("Promedios por nº de experimentos (RMSE de X y error de volumen %):")
print(agg.to_string(index=False,
      formatters={"rmse_fisico": "{:.4f}".format,
                  "rmse_cajanegra": "{:.4f}".format,
                  "errvol_fisico_pct": "{:.1f}".format,
                  "errvol_cajanegra_pct": "{:.1f}".format}))
print("-" * 70)
u = 5.0
print(f"Experimentos para error de volumen < {u:.0f}%:")
print(f"   Físico-informado : {n_para_umbral('errvol_fisico_pct', u)} experimentos")
print(f"   Caja-negra       : {n_para_umbral('errvol_cajanegra_pct', u)} experimentos")
print("=" * 70)
print("Archivos generados en la carpeta de salida:")
for f in ["datos_baja_fidelidad.csv", "datos_alta_fidelidad.csv", "datos_prueba.csv",
          "resumen_resultados.csv", "fig1_error_vs_datos.png", "fig2_error_volumen.png",
          "fig3_perfiles_XV.png", "fig4_ley_velocidad.png", "fig5_extrapolacion.png"]:
    print("  -", f)
