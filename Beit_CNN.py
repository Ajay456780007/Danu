import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Dropout
from keras.models import Model
import numpy as np
from Sub_Functions.Evaluate import main_est_parameters


def transformer_block(x, num_heads=2, key_dim=32):
    """Transformer block for 1D sequence input."""
    # Layer normalization
    norm1 = LayerNormalization()(x)
    # Self-attention
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(norm1, norm1)
    # Add & Norm
    x = tf.keras.layers.Add()([x, attn_output])
    norm2 = LayerNormalization()(x)
    # MLP block
    mlp_output = Dense(64, activation='relu')(norm2)
    mlp_output = Dense(x.shape[-1])(mlp_output)
    # Final skip connection
    return tf.keras.layers.Add()([x, mlp_output])


def build_sequence_transformer_classifier(input_shape=(10, 1), num_classes=2):
    inputs = Input(shape=input_shape)  # e.g., (sequence_length, 1)

    # Optional embedding / dense projection
    x = Dense(32, activation='relu')(inputs)

    # Transformer block
    x = transformer_block(x, num_heads=2, key_dim=16)

    # Global pooling and classification
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model


def BeiT_CNN(x_train, x_test, y_train, y_test, epochs):
    num_classes = len(np.unique(y_train))
    input_shape = (x_train.shape[1], 1)

    # Build model
    model = build_sequence_transformer_classifier(input_shape=input_shape, num_classes=num_classes)

    # Reshape data for (samples, sequence_length, 1)
    x_train = np.array(x_train).reshape(-1, x_train.shape[1], 1)
    x_test = np.array(x_test).reshape(-1, x_test.shape[1], 1)

    # Compile and train
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_split=0.2, epochs=epochs, batch_size=32, verbose=2)

    # Predict & evaluate
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    metrics = main_est_parameters(y_test, y_pred_classes)

    return metrics
