from keras.layers import BatchNormalization, Dropout
from keras.applications import InceptionV3
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Dense
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf
import os
from Sub_Functions.Evaluate import main_est_parameters
# from Sub_Functions.Load_data import train_test_splitter
from sklearn.metrics import classification_report, confusion_matrix

# from Sub_Functions.Load_data import train_test_splitter


def build_model(input_shape, num_classes):
    input_tensor = Input(shape=input_shape)

    # Load InceptionV3 without the top layers and with pretrained weights
    base_model = InceptionV3(include_top=False, weights='imagenet', input_tensor=input_tensor)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax', name='classification')(x)

    model = Model(inputs=base_model.input, outputs=output)

    # Freeze the base model layers initially to avoid overfitting
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def Inception_V3(x_train, x_test, y_train, y_test, epochs, DB):
    num_classes = len(np.unique(y_train))

    # Resize input to 75x75 for InceptionV3
    x_train_resized = tf.image.resize(x_train, [75, 75]).numpy()
    x_test_resized = tf.image.resize(x_test, [75, 75]).numpy()

    # Convert labels to one-hot encoding
    y_train_categorical = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test_categorical = tf.keras.utils.to_categorical(y_test, num_classes)

    # Build the model
    model = build_model((75, 75, 3), num_classes)

    # Reduce learning rate if validation loss plateaus
    # lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    # Early stopping to avoid overfitting
    # early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    model.fit(x_train_resized, y_train_categorical, epochs=epochs, batch_size=32, validation_split=0.2)

    # Save the model
    # os.makedirs("Saved_models/", exist_ok=True)
    # model.save(f"Saved_models/{DB}_model.h5")

    # Make predictions
    y_pred = model.predict(x_test_resized)
    y_pred_labels = np.argmax(y_pred, axis=1)

    # Evaluate model performance
    metrics = main_est_parameters(y_test, y_pred_labels)
    # confusion_matrix_ = confusion_matrix(y_test, y_pred_labels)
    # classification_report_ = classification_report(y_test, y_pred_labels)
    # print(confusion_matrix_)
    # print(classification_report_)
    return metrics


# Load data
# x_train, x_test, y_train, y_test = train_test_splitter("MaskedDFER", percent=0.8)
#
# # Run model
# metrics = InceptionV3_model(x_train, x_test, y_train, y_test, 50, "MaskedDFER")
# print(metrics)
