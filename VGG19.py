from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np
from Sub_Functions.Evaluate import main_est_parameters
# from Sub_Functions.Load_data import train_test_splitter


# from Sub_Functions.Load_data import train_test_split2

def VGG_19(x_train, x_test, y_train, y_test, epochs):
    NUM_CLASSES = len(np.unique(y_train))
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(50, 50, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='relu'))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, validation_split=0.2, epochs=epochs)

    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)

    metrics = main_est_parameters(y_test, y_pred)
    return metrics