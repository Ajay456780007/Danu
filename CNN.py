import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from keras.optimizers import Adam
from Sub_Functions.Evaluate import main_est_parameters


def CNN(x_train, x_test, y_train, y_test, epochs):
    # Ensure input shape: (samples, sequence_length, 1)
    x_train = np.array(x_train).reshape(-1, x_train.shape[1], 1)
    x_test = np.array(x_test).reshape(-1, x_test.shape[1], 1)

    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, padding="same", activation="relu", input_shape=x_train.shape[1:]))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(2, activation="softmax"))  # 2 classes

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])  # use sparse_categorical_crossentropy if labels are integers

    model.fit(x_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2, verbose=2)

    # Predictions
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Evaluate using your custom metrics function
    metrics = main_est_parameters(y_test, y_pred_classes)
    return metrics
