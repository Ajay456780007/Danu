import tensorflow as tf
from tensorflow.keras import layers, models
from Sub_Functions.Evaluate import main_est_parameters
import numpy as np


def build_autoencoder(input_shape=(10, 1), num_classes=2):
    """
    Fully connected autoencoder / classifier for 1D input sequence.
    Matches the proposed model input/output.
    """
    input_seq = layers.Input(shape=input_shape)  # e.g., (10, 1)

    # Flatten input if needed
    x = layers.Flatten()(input_seq)

    # ---------- ENCODER / Dense layers ----------
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # ---------- Bottleneck ----------
    x = layers.Dense(32, activation='relu')(x)

    # ---------- Decoder / Classification ----------
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dense(8, activation='relu')(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=input_seq, outputs=output)
    return model


def AE(x_train, x_test, y_train, y_test, epochs):
    # Build model
    num_classes = len(np.unique(y_train))
    input_shape = (x_train.shape[1], 1)  # (sequence_length, 1)
    model = build_autoencoder(input_shape=input_shape, num_classes=num_classes)

    # Ensure proper shape
    x_train = np.array(x_train).reshape(-1, x_train.shape[1], 1)
    x_test = np.array(x_test).reshape(-1, x_test.shape[1], 1)

    # Compile & train
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2, verbose=2)

    # Predict & evaluate
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    metrics = main_est_parameters(y_test, y_pred_classes)

    return metrics
