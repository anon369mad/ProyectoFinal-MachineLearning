import os
import numpy as np
import pandas as pd
import struct

class DataGenerator:
    def __init__(self, filepath, batch_size, sequence_length, max_samples=None, for_training=True):
        self.filepath = filepath
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.max_samples = max_samples
        self.for_training = for_training

        _, self.file_extension = os.path.splitext(filepath)

        # -------------------------------
        #   MODO CSV
        # -------------------------------
        if self.file_extension == ".csv":
            print(f"Loading CSV IQ data from {filepath}...")
            self.samples = self.load_csv(filepath)
            self.total_samples = len(self.samples)

        # -------------------------------
        #   MODO DAT
        # -------------------------------
        elif self.file_extension == ".dat":
            print("Reading DAT binary file...")
            self.binary_file = open(self.filepath, "rb")
            self.total_samples = None  # unknown size

        else:
            raise ValueError(f"Unsupported file format {self.file_extension}")

        self.reset()

    def reset(self):
        self.pos = 0
        if self.file_extension == ".dat":
            self.binary_file.seek(0)

    def close(self):
        if self.file_extension == ".dat" and not self.binary_file.closed:
            self.binary_file.close()

    # ================================================================
    #   CSV LOADER — Soporta pure_samples y under_attack_samples
    # ================================================================
    def load_csv(self, filepath):

        # Detecta si tiene encabezados
        df = pd.read_csv(filepath)

        # Caso 1: pure_samples_1.csv → sin encabezado → 1 columna
        if df.shape[1] == 1:
            raw = df.iloc[:, 0].astype(str)

        # Caso 2: under_attack_samples_1.csv → columnas: IQ Data,label
        elif "IQ Data" in df.columns:
            raw = df["IQ Data"].astype(str)

        else:
            raise ValueError("CSV format not recognized: expected 1 column or 'IQ Data' column.")

        iq_list = []

        for val in raw:
            try:
                clean = (
                    val.replace("(", "")
                    .replace(")", "")
                    .replace(" ", "")
                )
                iq_list.append(complex(clean))
            except:
                continue

        return np.array(iq_list)

    # ================================================================
    #   Convert complex samples → windows (batch, seq_len, 2)
    # ================================================================
    def create_windows(self, arr):

        real = np.real(arr)
        imag = np.imag(arr)

        # Normalización por ventana
        real = (real - real.mean()) / (real.std() + 1e-8)
        imag = (imag - imag.mean()) / (imag.std() + 1e-8)

        X = []
        for i in range(0, len(real) - self.sequence_length):
            seq = np.column_stack([
                real[i:i+self.sequence_length],
                imag[i:i+self.sequence_length]
            ])
            X.append(seq)

        return np.array(X)

    # ================================================================
    #   ITERACIÓN
    # ================================================================
    def __next__(self):

        # ------------------------------
        #   MODO CSV
        # ------------------------------
        if self.file_extension == ".csv":
            batch_end = self.pos + self.batch_size + self.sequence_length
            if batch_end >= self.total_samples:
                raise StopIteration

            batch_samples = self.samples[self.pos:batch_end]
            self.pos += self.batch_size

            X = self.create_windows(batch_samples)

            return (X, X) if self.for_training else X

        # ------------------------------
        #   MODO DAT
        # ------------------------------
        else:
            samples = []
            chunksize = self.batch_size * self.sequence_length

            while len(samples) < chunksize:
                binary_data = self.binary_file.read(8)
                if not binary_data:
                    raise StopIteration

                r, im = struct.unpack('ff', binary_data)
                c = complex(r, im)
                samples.append(c)

                if self.max_samples and self.pos >= self.max_samples:
                    raise StopIteration
                self.pos += 1

            samples = np.array(samples)
            X = self.create_windows(samples)

            return (X, X) if self.for_training else X
