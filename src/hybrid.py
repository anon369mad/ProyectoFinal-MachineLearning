# Importamos las librerías necesarias para todo el proyecto
import os
import numpy as np 
import pandas as pd       
import matplotlib.pyplot as plt
import random             # Para generar aumentos simples a los datos
import tensorflow as tf
from tensorflow.keras.models import Model, load_model   # Crear/cargar modelos
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, LSTM,RepeatVector,TimeDistributed, Dense,BatchNormalization, Dropout)
from data_generator import DataGenerator   
from tensorflow.keras.optimizers import Adam     #sirve para optimizar el modelo    
from tensorflow.keras.losses import MeanSquaredError  # Se usa para medir el error

from sklearn.metrics import (precision_score, recall_score, f1_score,confusion_matrix,precision_recall_curve)# Métricas para evaluar detección de anomalías

# Configuración básica
SEQ_LEN= 100
BATCH_SIZE= 64
EPOCHS= 30
LR= 1e-3
MODEL_PATH= "hybrid_conv_lstm_simple.keras"

TRAIN_CSV = "./data/pure_samples_1/pure_samples_1.csv"
VAL_CSV   = "./data/under_attack_samples_1/under_attack_samples_1.csv"

"""
#Experimento B Con varias técnicas de aumento de datos para series temporales
def augment_batch(X, p_noise=0.6,p_scale=0.6,p_shift=0.5,p_impulse=0.1, max_shift=5):
    #X: batch (batch, seq_len, features)
    X_aug = X.copy()
    if random.random() < p_noise:
        std = 0.01*(np.std(X_aug)+1e-12)
        X_aug += np.random.normal(0,std, size=X_aug.shape)
    if random.random()< p_scale:
        scale = np.random.uniform(0.9,1.1) 
        X_aug *= scale
   if random.random() < p_shift:
        shift = np.random.randint(-max_shift, max_shift+1)
        X_aug = np.roll(X_aug, shift, axis=1) 
   if random.random() < p_impulse:
        batch, seq, feat = X_aug.shape
        n_impulses = int(seq*0.01)   
        for b in range(batch):
            idx = np.random.choice(seq, size=n_impulses, replace=False)
            X_aug[b, idx, :] += np.random.uniform(0.1, 0.2) * np.sign(np.random.randn(n_impulses, feat))
    return X_aug
"""
# Esto es para experimento A: solo ruido gaussiano
def augment_batch(X, p_noise=0.6):
    #  agregamos un poco de ruido
    X_aug = X.copy()
    if random.random()< p_noise:
        std = 0.01*np.std(X) #Experimento C: 0.01 -> 0.03
        X_aug = X_aug+np.random.normal(0, std, size=X_aug.shape)
    return X_aug

# nuestro modelo híbrido Conv1D+LSTM 
def build_model(seq_len,n_features=2, latent=32):
    inp =Input(shape=(seq_len, n_features))
    # Encoder: primero conv1D+pooling para comprimir un poco la secuencia
    x =Conv1D(64,3, padding='same',activation='relu')(inp)
    x =BatchNormalization()(x)
    x =MaxPooling1D(2, padding='same')(x)
    x =Conv1D(128, 3, padding='same', activation='relu')(x)
    x =BatchNormalization()(x)
    x =MaxPooling1D(2,padding='same')(x)
    # LSTM que resume la secuencia comprimida
    x =LSTM(256, return_sequences=False)(x)
    x = Dropout(0.15)(x)
    # Capa de Representación comprimida final
    bottleneck = Dense(latent, activation='relu')(x)
    # Reconstruimos la secuencia original osea decoder
    x =RepeatVector(seq_len)(bottleneck)
    x =LSTM(64,return_sequences=True)(x)
    x =TimeDistributed(Dense(32,activation='relu'))(x)
    #x = Conv1D(16, 3, padding='same', activation='relu')(x)#Experimento C Con una capa conv1D adicional en el decoder
    # Reconstrucción final de I/Q
    out =TimeDistributed(Dense(n_features, activation='linear'),name='reconstruction')(x)
    model =Model(inp, out,name='hybrid_simple')
    model.compile(optimizer=Adam(LR), loss=MeanSquaredError())
    return model

# Convierte etiquetas por muestra a etiquetas por ventana
def samples_to_window_labels(slabels, seq_len, rule="any"):
    n =len(slabels)
    if n <seq_len:
        return np.array([], dtype=int)
    ws =[]
    for i in range(0, n-seq_len+1):
        w = slabels[i:i+seq_len]
        # "any": si hay alguna etiqueta anómala → ventana anómala
        if rule == "any":
            ws.append(1 if np.any(w == 1) else 0)
        # centro de la ventana define la etiqueta
        elif rule == "center":
            ws.append(int(w[seq_len//2] ==1))
        else:
            ws.append(1 if np.mean(w)>0.5 else 0)
    return np.array(ws, dtype=int)

# Entrenamiento manual por lotes (más estable que fit_generator)
def train_model(model, gen,epochs=EPOCHS,steps_per_epoch=None):
    # Si el generador conoce el total de muestras:
    if steps_per_epoch is None and getattr(gen, "total_samples", None) is not None:
        total_windows = max(0, gen.total_samples-gen.sequence_length+1)
        steps_per_epoch = max(1, int(np.ceil(total_windows/gen.batch_size)))
    steps_per_epoch = steps_per_epoch or 100
    best_loss = np.inf
    for ep in range(epochs):
        #imprimir progreso
        print(f"\nEpoch {ep+1}/{epochs} (steps_per_epoch={steps_per_epoch})")
        gen.reset()
        batch = 0
        losses = []
        while batch <steps_per_epoch:
            try:
                X_batch, _ = next(gen)
            except StopIteration:
                print(" Generador agotado. Se reinicia.")
                gen.reset()
                try:
                    X_batch, _ = next(gen)
                except StopIteration:
                    print(" Sin más datos. Fin del entrenamiento.")
                    return
            # Aumentación
            X_aug = augment_batch(X_batch)
            # Entrenar por batch
            loss = model.train_on_batch(X_aug, X_batch)
            losses.append(loss)
            batch += 1
            if batch%50 == 0 or batch == steps_per_epoch:
                print(f"  batch {batch}/{steps_per_epoch} pérdida(media)={np.mean(losses[-50:]):.6f}")
        mean_loss = np.mean(losses)
        print(f" Pérdida media epoch {ep+1}: {mean_loss:.6f}")
        # Guardamos mejor modelo
        if mean_loss < best_loss:
            best_loss = mean_loss
            model.save(MODEL_PATH)
            print("####Mejor pérdida encontrada. Modelo guardado.#####")

    print("Entrenamiento terminado.")

# Inferencia rápida en modo batch
def infer_errors(gen, model, max_batches=None):
    gen.reset()
    errors = []
    if max_batches is None and getattr(gen,"total_samples", None) is not None:
        total_windows = max( 0, gen.total_samples-gen.sequence_length+1)
        max_batches = int(np.ceil(total_windows/gen.batch_size))
    max_batches = max_batches or 200
    for b in range(max_batches):
        try:
            X = next(gen)
        except StopIteration:
            break
        # Usar predict_on_batch (más rápido)
        X_pred = model.predict_on_batch(X)
        # Calculamos error MSE por ventana para I y Q
        mse_I =np.mean((X[:,:,0]-X_pred[:,:,0])**2,axis=1)
        mse_Q = np.mean((X[:,:,1]-X_pred[:,:,1])**2,axis=1)
        mse = mse_I * 0.5 + mse_Q * 0.5
        errors.append(mse)
    if not errors:
        return np.array([])
    return np.concatenate(errors, axis=0)

# Encuentra el mejor umbral usando PR para maximizar F1
def find_best_threshold(errors, y_true):
    if len(errors) == 0:
        return None, 0.0
    prec, rec, thr = precision_recall_curve(y_true, errors)
    f1s = 2*prec*rec / (prec+rec+1e-12)
    f1s = f1s[:-1]  # descartamo el último umbral
    if len(f1s) == 0:
        return np.percentile(errors, 99),0.0
    best_idx = np.argmax(f1s)
    return thr[best_idx], f1s[best_idx]

#  Main
if __name__ == "__main__":
    # Crear modelo
    model = build_model(SEQ_LEN, n_features=2, latent=32)
    model.summary()
    # Generadores con tu DataGenerator
    train_gen = DataGenerator(TRAIN_CSV, batch_size=BATCH_SIZE,sequence_length=SEQ_LEN, for_training=True)
    val_gen   = DataGenerator(VAL_CSV, batch_size=BATCH_SIZE,sequence_length=SEQ_LEN, for_training=False)
    # Entrenar
    print("\nINICIO DEL ENTRENAMIENTO")
    train_model(model, train_gen, epochs=EPOCHS, steps_per_epoch=100)
    # Cargar mejor modelo
    if os.path.exists(MODEL_PATH):
        model =load_model(MODEL_PATH, custom_objects={'mse': MeanSquaredError()})
        print(f"Modelo cargado desde: {MODEL_PATH}")
    else:
        print("No se encontró un modelo previo. Se usa el entrenado.")
    # Validación
    print("\nVALIDACIÓN")
    val_errors = infer_errors(val_gen, model)

    if val_errors.size == 0:
        print("No se pudieron procesar ventanas de validación.")
    else:
        df_val = pd.read_csv(VAL_CSV)
        sample_labels = (df_val['label'] == 'jammer').astype(int).values
        win_labels = samples_to_window_labels(sample_labels, SEQ_LEN, rule="any")
        min_len = min(len(win_labels), len(val_errors))
        win_labels = win_labels[:min_len]
        val_errors = val_errors[:min_len]
        print(f"Ventanas val: {len(win_labels)}, errores: {len(val_errors)}")
        best_thr, best_f1 = find_best_threshold(val_errors, win_labels)
        if best_thr is None:
            best_thr = np.percentile(val_errors, 99)
        print(f"Mejor umbral: {best_thr:.6e}, Mejor F1: {best_f1:.4f}")
        y_pred = (val_errors > best_thr).astype(int)
        p =precision_score(win_labels, y_pred, zero_division=0)
        r =recall_score(win_labels, y_pred, zero_division=0)
        f1 =f1_score(win_labels, y_pred, zero_division=0)
        cm =confusion_matrix(win_labels, y_pred)
        print("\nMÉTRICAS (mejor umbral)")
        print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
        print("Confusion matrix:\n", cm)
        # Gráfico simple de distribución de errores
        plt.figure(figsize=(9,4))
        plt.hist(val_errors[win_labels==0], bins=150, alpha=0.6, label='pure')
        plt.hist(val_errors[win_labels==1], bins=150, alpha=0.6, label='jammer')
        plt.axvline(best_thr, color='k', linestyle='--', label='best_thr')
        plt.legend()
        plt.title('Distribución de errores de reconstrucción')
        plt.show()
    # Guardar modelo final
    model.save("hybrid_conv_lstm_simple_final.keras")
    print("Modelo final guardado como: hybrid_conv_lstm_simple_final.keras")