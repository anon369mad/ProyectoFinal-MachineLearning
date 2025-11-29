import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from data_generator import DataGenerator

sequence_length = 10

# Define the RNN autoencoder model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 2), return_sequences=True))
model.add(LSTM(25, activation='relu', return_sequences=False))
model.add(RepeatVector(sequence_length))
model.add(LSTM(25, activation='relu', return_sequences=True))
model.add(LSTM(50, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(2)))

model.summary()
model.compile(optimizer='adam', loss='mse')

batch_size = 20
max_train_samples = 100000
train_steps = max_train_samples // (batch_size * sequence_length)

# Initialize the data generator for training
train_gen_instance = DataGenerator(
    './data/pure_samples_1/pure_samples_1.csv',
    batch_size=batch_size,
    sequence_length=sequence_length,
    max_samples=max_train_samples,
    for_training=True
)
# Train the model
num_epochs = 9
steps_per_epoch = train_steps

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_gen_instance.reset()
    for step in range(steps_per_epoch):
        try:
            X_chunk, Y_chunk = next(train_gen_instance)
        except StopIteration:
            train_gen_instance.reset()
            X_chunk, Y_chunk = next(train_gen_instance)

        model.train_on_batch(X_chunk, Y_chunk)
        print(f"Step {step + 1}/{steps_per_epoch}", end='\r')
    print()

# Save the trained model
model.save('rnn_autoencoder_model.h5')

train_gen_instance.close()
