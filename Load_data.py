import numpy as np
import pandas as pd
import os
from Comparative_model.CNN import CNN
from Comparative_model.InceptionV3 import Inception_V3
from Comparative_model.SVM import SVM
from Comparative_model.VGG19 import VGG_19
from Comparative_model.Beit_CNN import BeiT_CNN
from Comparative_model.DCNN import DCNN
from Comparative_model.AE import AE
from Proposed_model.proposed_model import proposed_model

def Load_data(DB):
    feat = np.load(f"Data_loader/{DB}/features.npy")
    label = np.load(f"Data_loader/{DB}/labels.npy")

    return feat, label


def Load_data2(DB):
    feat = np.load(f"../Data_loader/{DB}/features.npy")
    label = np.load(f"../Data_loader/{DB}/labels.npy")

    return feat, label


def train_test_splitter(data, percent, num=500):
    feat, label = Load_data2(data)  # load your features and labels
    unique_classes = np.unique(label)

    selected_indices_per_class = []

    for cls in unique_classes:
        class_indices = np.where(label == cls)[0]
        # Randomly choose `num` samples per class without replacement
        selected_class_indices = np.random.choice(class_indices, num, replace=True)
        selected_indices_per_class.append(selected_class_indices)

    # Concatenate all selected indices from each class
    selected_indices = np.concatenate(selected_indices_per_class)

    # Shuffle the combined indices
    np.random.shuffle(selected_indices)

    # Get balanced features and labels
    balanced_feat = feat[selected_indices]
    balanced_label = label[selected_indices]

    data_size = balanced_feat.shape[0]
    split_point = int(data_size * percent)  # training data size

    # Split into train/test sets
    training_sequence = balanced_feat[:split_point]
    training_labels = balanced_label[:split_point]
    testing_sequence = balanced_feat[split_point:]
    testing_labels = balanced_label[split_point:]

    return training_sequence, testing_sequence, training_labels, testing_labels


def models_return_metrics(data, epochs, ok=True, percents=None, force_retrain=False):
    import os

    training_percentages = percents if percents is not None else [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    model_registry = {
        # "AE": AE,
        # "BeiT_CNN": BeiT_CNN,
        # "CNN": CNN,
        # "DCNN": DCNN,
        "Base_model":proposed_model,
        "opt1":proposed_model,
        "opt2":proposed_model
    }

    if ok:
        for model_name, model_fn in model_registry.items():
            print(f"\n==== Training model: {model_name} ====")
            all_metrics = []

            for percent in training_percentages:
                print(f"  → Training {model_name} with {int(percent * 100)}% training data...")

                x_train, x_test, y_train, y_test = train_test_splitter1(data, percent=percent)
                if model_name == "Inceptionv3":

                    metrics = model_fn(x_train, x_test, y_train, y_test, epochs, data)
                    print("metrics ok ")
                elif model_name == "Base_model":
                    metrics = model_fn(x_train, x_test, y_train, y_test, percent, "DB1", opt=0)

                    # result = proposed_model(x_train, x_test, y_train, y_test, 0.7, "DB1", opt=3)
                elif model_name =="opt1":
                    metrics = model_fn(x_train, x_test, y_train, y_test, percent, "DB1", opt=1)

                elif model_name =="opt2":
                    metrics = model_fn(x_train, x_test, y_train, y_test, percent, "DB1", opt=2)
                else:
                    metrics = model_fn(x_train, x_test, y_train, y_test, epochs)

                all_metrics.append(metrics)

            # Save after all percentages
            save_path = f"Temp/{data}/Comp/{model_name}/all_metrics.npy"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, np.array(all_metrics, dtype=object))

            print(f"✔ Saved all metrics for {model_name} to {save_path}")


def train_test_splitter1(data, percent, num=50):
    feat, label = Load_data(data)  # load your features and labels
    unique_classes = np.unique(label)

    selected_indices_per_class = []

    for cls in unique_classes:
        class_indices = np.where(label == cls)[0]
        # Randomly choose `num` samples per class without replacement
        selected_class_indices = np.random.choice(class_indices, num, replace=True)
        selected_indices_per_class.append(selected_class_indices)

    # Concatenate all selected indices from each class
    selected_indices = np.concatenate(selected_indices_per_class)

    # Shuffle the combined indices
    np.random.shuffle(selected_indices)

    # Get balanced features and labels
    balanced_feat = feat[selected_indices]
    balanced_label = label[selected_indices]

    data_size = balanced_feat.shape[0]
    split_point = int(data_size * percent)  # training data size

    # Split into train/test sets
    training_sequence = balanced_feat[:split_point]
    training_labels = balanced_label[:split_point]
    testing_sequence = balanced_feat[split_point:]
    testing_labels = balanced_label[split_point:]

    return training_sequence, testing_sequence, training_labels, testing_labels
