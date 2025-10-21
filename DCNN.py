import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from Sub_Functions.Evaluate import main_est_parameters

def DCNN(x_train, x_test, y_train, y_test, epochs):
    """
    Deep CNN for proposed model input shape (10, 1)
    """

    # Reshape inputs to 3D for Conv1D
    if x_train.ndim == 2:
        x_train = x_train.reshape(-1, x_train.shape[1], 1)
    if x_test.ndim == 2:
        x_test = x_test.reshape(-1, x_test.shape[1], 1)

    input_shape = x_train.shape[1:]  # (10,1)
    num_classes = len(np.unique(y_train))

    model = Sequential()
    # Conv1D layers
    model.add(Conv1D(32, kernel_size=2, activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64, kernel_size=2, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=epochs, batch_size=16, validation_split=0.2)

    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)

    metrics = main_est_parameters(y_test, y_pred)
    return metrics
