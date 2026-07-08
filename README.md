# Nota de métodos — Dimensionamiento de un PFR con IA (datos multi-fidelidad)

Material de apoyo para la clase de 20 min (pregrado ing. química). Todo lo que
está aquí se genera con `demo_pfr_ia.py`; los números coinciden con las figuras.
Se incluye in Jupyter Notebook `demo_pfr_ia.ipynb` para que el estudiante tenga 
una experiencia más interactiva.

---

## 1. El problema

Reacción `A → B` en un PFR **isotérmico**, fase líquida, flujo volumétrico
constante. Queremos **dimensionar el reactor**: hallar el volumen `V` necesario
para una conversión objetivo `X* = 0.80`, gastando la **menor cantidad de
experimentos** posible (cada experimento es caro y lento).

Datos del sistema: `C_A0 = 2.0 mol/L`, `v0 = 1.0 L/s`, `F_A0 = 2.0 mol/s`.

**Lo que SIEMPRE conocemos** — la física dura, el balance molar del PFR:

```
dX/dV = (-r_A) / F_A0 ,      con   C_A = C_A0 (1 - X)
```

**Lo que NO conocemos** — la ley de velocidad `-r_A(C_A)`. En este ejemplo la
"realidad" (oculta al modelo) es una cinética tipo Langmuir-Hinshelwood:

```
-r_A = k · C_A² / (1 + K_ads · C_A) ,     k = 0.18,  K_ads = 0.55
```

Referencia exacta (integrando la EDO): **V\* = 16.0 L** para `X = 0.80`.

---

## 2. Dato multi-fidelidad

| Fuente | Qué es | Costo | Cantidad | Sesgo |
|---|---|---|---|---|
| **Baja fidelidad** | Modelo barato: suponer cinética de **1er orden** `-r_A ≈ 0.085·C_A` | casi gratis | mucha | **sí** (error de 18% en el volumen) |
| **Alta fidelidad** | Experimentos reales (verdad + ruido de medición ~3%) | caro | poca | no (pero con ruido) |

El modelo barato solo, para dimensionar, da **V = 18.9 L** → **18% de error**.
La idea multi-fidelidad: usar el modelo barato como **punto de partida (prior)**
y corregirlo con **unos pocos** experimentos caros.

---

## 3. Los dos modelos que comparamos

### (A) Físico-informado = red neuronal dentro de la EDO (UDE)
**Aquí está el machine learning.** La incógnita (la ley de velocidad) se reemplaza
por una **red neuronal** `NN_θ(C_A)` (un MLP de 1 capa oculta, `tanh`), escrita de
forma multiplicativa sobre el prior barato:

```
-r_A(C_A) = r_LF(C_A) · exp( NN_θ(C_A) )
```

- Si la red vale 0 ⇒ recupera exactamente el prior de 1er orden (**ancla multi-fidelidad**).
- El `exp(·)` garantiza velocidad **siempre positiva** ⇒ la EDO nunca da perfiles absurdos.
- La red es **pequeña a propósito** (3 neuronas ocultas): con 4–8 datos, más
  capacidad = más sobreajuste (justo lo que le pasa a la caja-negra).

Entrenamiento (physics-informed): se mete la red **dentro** del balance molar, se
integra el PFR y se ajustan los **pesos θ** por optimización:

```
min_θ  Σ_i [ (X_pred(V_i; θ) − X_i) / σ ]²  +  λ² · ‖θ‖²
        └──────── ajuste a datos caros ────┘   └─ regulariza (cerca del prior) ─┘
```

Esto es una **Universal Differential Equation / Neural ODE**: la red aprende la
parte desconocida de la física. Como el balance molar hace el trabajo pesado,
**bastan muy pocos experimentos** y el ruido se filtra.

> Nota de diseño: empezamos con una red más grande (5–6 neuronas) pero con 4–6
> datos ruidosos se sobreajustaba; una red pequeña + regularización L2 hacia el
> prior la estabiliza. Esa tensión *flexibilidad ↔ datos* **es** el mensaje de la clase.

### (B) Caja-negra — sin física
Una red neuronal (MLP 32×32, tanh) aprende directamente el mapa `V → X`. No sabe
nada del balance molar ni de que existe una ley de velocidad. Con pocos puntos
**sobreajusta el ruido** y la curvatura queda mal.

---

## 4. Resultados (promedios sobre 8 realizaciones de ruido)

RMSE de la conversión sobre el set de prueba y error en el volumen para `X=0.8`:

Físico-informado = red neuronal dentro de la EDO (3 neuronas). RMSE de la conversión
y error en el volumen para `X=0.8`:

| Nº exp. | RMSE físico | RMSE caja-negra | Err. vol. físico | Err. vol. caja-negra |
|---:|---:|---:|---:|---:|
| 4  | 0.025 | 0.032 | 11.5% |  9.9% |
| 5  | 0.023 | 0.031 |  8.4% | 14.1% |
| 6  | 0.022 | 0.029 |  **3.2%** |  6.8% |
| 8  | 0.017 | 0.024 |  4.3% | 15.3% |
| 10 | 0.015 | 0.020 |  4.5% |  6.1% |
| 14 | 0.013 | 0.020 |  4.0% |  5.9% |

**Titulares:**
- El físico-informado (red neuronal) tiene un RMSE **~1.5× menor** en todo el rango.
- Alcanza la **tolerancia de diseño (< 5%)** con **6 experimentos**; la caja-negra
  **no la alcanza de forma fiable** en el rango probado (queda entre 6% y 15% y es
  **errática** — más datos no ayudan de forma monótona porque sobreajusta ruido).
- Con **solo 4 experimentos**, la red ya **reconstruye la ley de velocidad
  verdadera** (fig. 4), algo que la caja-negra ni siquiera intenta.

---

## 5. Guion sugerido (20 min) y qué figura usar

1. **(2 min) El problema.** Dimensionar un PFR = necesito `V` para `X*`. Necesito
   `-r_A(C_A)`, y medirla cuesta. Balance molar `dX/dV = -r_A/F_A0`.
2. **(3 min) Multi-fidelidad.** Modelo barato sesgado (18% de error) vs
   experimentos caros. → tabla de la sección 2.
3. **(4 min) Dos filosofías.** Meter la física (aprender solo `-r_A`) vs caja-negra
   (aprender `V→X`). Ecuación de la pérdida física de la sección 3A.
4. **(4 min) Perfil con pocos datos.** → **fig. 3**: con 4 experimentos, el
   físico-informado sigue la verdad; la caja-negra serpentea y cruza `X=0.8` en el
   `V` equivocado.
5. **(3 min) Eficiencia de datos.** → **fig. 1** (RMSE vs Nº exp.) y **fig. 2**
   (error de volumen; el físico baja de la línea del 5%, la caja-negra no).
6. **(2 min) Lo que la física regala.** → **fig. 4**: recupera la ley de velocidad
   real desde 4 puntos. Interpretabilidad + extrapolación.
7. **(2 min) Cierre.** Menos costo (computacional y de experimentos) = atar el
   modelo a la física. La física reduce los grados de libertad → menos datos.

---

## 5b. El caso más revelador: EXTRAPOLACIÓN (fig. 5)

Situación realista: los experimentos se hicieron **solo en reactores pequeños**
(baja conversión, `X ≲ 0.64`, `V ≤ 8 L`) porque son baratos y rápidos. Hay que
dimensionar el **reactor grande** para `X = 0.8` (`V ≈ 16 L`), **fuera** del rango
de los datos.

- **Físico-informado:** el balance molar le permite **extrapolar** con sentido
  físico → alcanza `X=0.8` cerca del volumen real. Con 6 experimentos:
  **V\* ≈ 14 L (error ~11%)**, y **siempre** alcanza el objetivo.
- **Caja-negra:** fuera de sus datos extrapola sin control. En el ejemplo
  **sub-dimensiona el reactor ~30%** (V\* ≈ 11 L) y, en varias realizaciones de
  ruido, **su perfil ni siquiera llega a `X=0.8`** → no puede dimensionar el reactor.

Promediando 4 realizaciones (datos en `V≤8`, `n=6-8`): físico-informado ~15% de
error alcanzando el objetivo el 100% de las veces; caja-negra RMSE **3–5× peor** y
falla en alcanzar el objetivo buena parte de las veces. Moraleja: **la física
permite extrapolar; la caja-negra no.** (Un reactor sub-dimensionado no cumple la
especificación de conversión — es un error de diseño costoso.)

## 6. Honestidad intelectual (por si preguntan)
- La caja-negra **interpola** una curva suave razonablemente; su debilidad real es
  el **ruido con pocos datos** y la **falta de estructura** (no garantiza
  monotonía ni una ley de velocidad física, ni extrapola).
- Si los experimentos se hicieran **solo en reactores pequeños** (baja conversión)
  y hubiera que predecir el reactor grande, la ventaja del físico-informado es aún
  mayor: el balance molar permite **extrapolar**; la caja-negra no.
- El físico-informado supone que el balance molar es correcto. Si la física
  asumida está mal (p. ej. no isotérmico, mezclado no ideal), ese sesgo se traslada.

---

## 7. Archivos
- `demo_pfr_ia.ipynb` — **notebook Jupyter** con celdas explicativas (incluye la
  sección de extrapolación). Recomendado para explorar y para la clase.
- `demo_pfr_ia.py` — mismo contenido como script (reanudable), genera todas las figuras.
- `datos_baja_fidelidad.csv`, `datos_alta_fidelidad.csv`, `datos_prueba.csv` — datos.
- `resultados_crudos.csv` / `resumen_resultados.csv` — métricas crudas y promediadas.
- `fig1_error_vs_datos.png`, `fig2_error_volumen.png`, `fig3_perfiles_XV.png`,
  `fig4_ley_velocidad.png`, `fig5_extrapolacion.png` — figuras.

**Cómo correrlo:** abre `demo_pfr_ia.ipynb` en Jupyter y ejecuta de arriba a abajo;
o `python3 demo_pfr_ia.py`. Necesita numpy, scipy, scikit-learn, matplotlib, pandas.
Tarda ~1–2 min; si el script se interrumpe, al re-ejecutarlo continúa.
