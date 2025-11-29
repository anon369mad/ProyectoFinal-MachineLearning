# 🛰️ Detección de Anomalías en Series Temporales IQ mediante Autoencoders (Conv1D + LSTM)

Este proyecto implementa un sistema completo para detectar **ataques de interferencia (jamming)** en una red de comunicaciones simulada, utilizando **autoencoders para series temporales** basados en **CNN + LSTM**.
La detección se basa en el **error de reconstrucción** sobre ventanas de datos IQ.

---

## 📁 Estructura del Proyecto

```
TIME-SERIES-ANOMALY-DETECTION/
│
├── data/
│   ├── pure_samples_1/
│   │   └── pure_samples_1.csv
│   ├── under_attack_samples_1/
│   │   └── under_attack_samples_1.csv
│   └── intrusion_detected_plots/
│       ├── intrusion_sequence_30.png
│       ├── intrusion_sequence_54.png
│       └── ...
│
├── src/
│   ├── data_generator.py
│   ├── model.py
│   ├── hybrid.py
│   ├── anomaly_detection.py
│   └── __pycache__/
│
├── hybrid_conv_lstm_simple.keras
├── hybrid_conv_lstm_simple_final.keras
├── hybrid_simple_best.keras
├── rnn_autoencoder_model.h5
├── README.md
└── requirements.txt
```

---

## 🚀 Objetivo del Proyecto

Detectar **anomalías en señales IQ** provenientes de un sistema de comunicaciones.
Un ataque jammer introduce patrones anómalos que **incrementan el error de reconstrucción del autoencoder**, lo que permite distinguir situaciones normales de eventos maliciosos.

---

# 🔧 Componentes Principales

---

## 1. Data Generator (`data_generator.py`)

El Data Generator:

* Lee archivos `.csv` con muestras IQ.
* Separa parte real e imaginaria.
* Normaliza dinámicamente por ventana.
* Crea ventanas deslizantes (por defecto `SEQ_LEN = 100`).
* Retorna tensores con forma:

```
(batch_size, seq_len, 2)
```

Incluye características avanzadas:

* Reinicio automático del generador.
* Manejo de archivos grandes sin cargarlos en memoria.
* Compatibilidad con data augmentation en tiempo real.

---

## 2. Modelo Híbrido Conv1D + LSTM (`hybrid.py`)

Arquitectura del autoencoder:

### **Encoder**

* Capas `Conv1D` con `BatchNorm` + `MaxPooling`.
* LSTM de 256 unidades para capturar dinámica temporal.
* Capa latente entre 32–64 dimensiones.

### **Decoder**

* `RepeatVector(seq_len)`
* LSTM de 64 unidades
* Capas densas temporales para reconstruir IQ.

Compilado con:

```python
optimizer = Adam(1e-3)
loss = MeanSquaredError()
```

Entrenado mediante:

```python
train_on_batch(X_aug, X_original)
```

---

## 3. Inferencia y Anomaly Detection (`anomaly_detection.py`)

Incluye:

* Cálculo de error por ventana (MSE).

* Histogramas de errores.

* Selección automática del **umbral óptimo** usando:

  ```
  precision_recall_curve → mejor F1
  ```

* Generación automática de gráficos bajo ataque en:

```
data/intrusion_detected_plots/
```

---

# 🧪 Experimentos

---

## 🧪 **Experimento A — Conv–LSTM + Ruido Gaussiano**

Aumentación simple:

```
X_aug = X + N(0, 0.01 * std)
```

**Resultados:**

* Muy buena separación entre pure y jammer.
* **F1 ≈ 0.50** (vs baseline ≈ 0.13).

---

## 🧪 **Experimento B — Aumentación Avanzada**

Incluye:

* Ruido gaussiano
* Amplitude scaling
* Circular time shifting
* Impulse noise

**Resultados:**

* Mayor robustez general
* F1 se mantiene estable ≈ **0.50**
* Limitación: falta regularización temporal explícita.

---

## 🧪 **Experimento C — Denoising Autoencoder + Regularización Temporal**

### ✔ Entrenamiento con ruido pesado

El modelo aprende a reconstruir señales limpias a partir de señales distorsionadas.

### ✔ Regularización temporal añadida

Pérdida total:

```
L = MSE(x, x_hat) + λ · Σ_t (x̂[t+1] – x̂[t])²
```

### ✔ Beneficios

* Reduce sobreajuste a transitorios irrelevantes.
* Decoder más estable.
* Aumenta separación entre pure y jammer.
* F1 puede subir a **0.60–0.70**.

➡️ Diseñado para empujar el sistema hacia la meta **F1 = 0.7–0.85**.

---

# 📊 Visualización de Intrusiones

El sistema genera imágenes como:

```
intrusion_detected_plots/
│ intrusion_sequence_30.png
│ intrusion_sequence_54.png
│ ...
```

Cada figura muestra:

* Señal original vs reconstruida.
* Error punto a punto.
* Marcadores cuando la ventana fue clasificada como anomalía.

---

# ▶️ Cómo Entrenar el Modelo

Desde `src/`:

```bash
python hybrid.py
```

Esto:

* Inicializa modelo + generadores.
* Entrena por 30 epochs (configurable).
* Guarda mejores versiones como:

```
hybrid_conv_lstm_simple.keras
hybrid_simple_best.keras
```

---

# ▶️ Cómo Realizar Detección de Anomalías

Ejecutar:

```bash
python anomaly_detection.py
```

El script:

* Carga modelo entrenado.
* Procesa señales bajo ataque.
* Calcula errores por ventana.
* Determina mejor umbral según F1.
* Guarda gráficos de secuencias anomalía.

---

# 📝 Requisitos

Archivo `requirements.txt` incluye:

* tensorflow / keras
* numpy
* pandas
* matplotlib
* scikit-learn

Instalación:

```bash
pip install -r requirements.txt
```

---

# 🏁 Conclusión

El sistema implementa un pipeline completo para detección de ataques de jamming usando autoencoders temporales con:

* Arquitectura híbrida **CNN + LSTM**
* Data augmentation avanzado
* Regularización temporal para mejorar discriminación
* Umbral óptimo automático vía curva PR

