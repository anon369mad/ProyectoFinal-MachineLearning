import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from data_generator import DataGenerator
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# ============================
#     CONFIG
# ============================
sequence_length = 10
batch_size = 20
num_batches = 100

# ============================
#     LOAD MODEL
# ============================
model = load_model(
    'rnn_autoencoder_model.h5',
    custom_objects={'mse': MeanSquaredError()}
)

# ============================
#   DATA GENERATOR (TEST)
# ============================
gen = DataGenerator(
    './data/under_attack_samples_1/under_attack_samples_1.csv',
    batch_size=batch_size,
    sequence_length=sequence_length,
    for_training=False
)

# ============================
#   STORAGE
# ============================
reconstruction_errors = []   # will store arrays shape (batch, seq_len)
all_X_test = []
all_X_pred = []

print("Starting anomaly detection...")

# ============================
#   PREDICTION LOOP
# ============================
for batch_idx in range(num_batches):
    try:
        X = next(gen)
    except StopIteration:
        print("Generator finished early.")
        break

    print(f"Batch {batch_idx+1}/{num_batches}")

    X_pred = model.predict(X)

    # Error por muestra en ventana -> shape (batch, seq_len)
    err = np.mean((X - X_pred)**2, axis=2)

    reconstruction_errors.append(err)
    all_X_test.append(X)
    all_X_pred.append(X_pred)

# ============================
#   SAFETY: verificar que recolectamos algo
# ============================
if len(reconstruction_errors) == 0:
    raise RuntimeError("No se recolectaron errores: el generador no devolvió ventanas.")

# ============================
#   CONCATENAR
# ============================
# reconstruction_errors -> list of arrays (batch, seq_len) -> concat filas
reconstruction_errors = np.vstack(reconstruction_errors)   # shape = (N_windows, seq_len)

# Sólo concatenar all_X_* si realmente los necesitas (pueden ser grandes)
try:
    all_X_test = np.vstack(all_X_test)
    all_X_pred = np.vstack(all_X_pred)
except Exception:
    # si falla, no es crítico para métricas; mantener como None
    all_X_test = None
    all_X_pred = None

# ============================
#   ERROR POR SECUENCIA
# ============================
# toma el error máximo entre los dos canales por timestep y luego promedio por ventana
# (tu diseño original promediaba real+imag sobre axis=2 y luego aquí promedia; mantener similar)
seq_error = reconstruction_errors.mean(axis=1)   # 1 valor por ventana

# Threshold dinámico percentil 99 (baseline)
threshold = np.percentile(seq_error, 99)

y_pred = (seq_error > threshold).astype(int)

# ============================
#   CARGAR LABELS REALES (por muestra)
# ============================
df = pd.read_csv('./data/under_attack_samples_1/under_attack_samples_1.csv')
if 'label' not in df.columns:
    raise RuntimeError("CSV no contiene columna 'label'")

sample_labels = (df['label'] == 'jammer').astype(int).values  # 0/1 por muestra

# ============================
#   CONSTRUIR LABELS POR VENTANA (sliding windows)
#   ventana = 1 si ALGUNA muestra dentro es jammer
# ============================
window_labels = []
for i in range(0, len(sample_labels) - sequence_length + 1):
    window_labels.append(1 if sample_labels[i:i+sequence_length].any() else 0)
window_labels = np.array(window_labels)

# Alinear longitudes (usar min para evitar mismatches)
min_len = min(len(window_labels), len(seq_error))
y_true_win = window_labels[:min_len]
y_pred_win = y_pred[:min_len]

print(f"Windows used for metrics: {min_len}")
print(f"Total windows (from samples): {len(window_labels)}, windows predicted: {len(seq_error)}")

# ============================
#   METRICAS
# ============================
precision = precision_score(y_true_win, y_pred_win, zero_division=0)
recall    = recall_score(y_true_win, y_pred_win, zero_division=0)
f1        = f1_score(y_true_win, y_pred_win, zero_division=0)
cm        = confusion_matrix(y_true_win, y_pred_win)

print("\n=== FINAL DETECTION METRICS ===")
print(f"Threshold: {threshold:.6f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

# ============================
#   PLOT ERROR VS THRESHOLD
# ============================
plt.figure(figsize=(14, 6))
plt.plot(seq_error[:min_len], label='Error por Secuencia')
plt.axhline(threshold, color='r', linestyle='--', label='Threshold 99%')
plt.title("Error de reconstrucción por secuencia (ventanas alineadas)")
plt.xlabel("Ventana index")
plt.ylabel("Error")
plt.legend()
plt.show()
