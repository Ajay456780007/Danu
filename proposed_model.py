import os
import numpy as np
import random
from keras import Input, models
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.utils import plot_model, to_categorical
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from termcolor import colored

# Your external imports (kept as-is)
from optimizations.BCO.border import BCO
from Sub_Functions.Evaluate import main_est_parameters
from optimizations.Walrus_opt.main1 import Walrus_opt
# from optimizations.PROPOSED import PROP_opt  # if you have a proposed optimizer module
from mealpy import FloatVar

# ---------------------------
# Your optimization class (kept unchanged)
# ---------------------------
class optimization:
    def __init__(self, model, x_test, y_test):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test

    def fitness_function1(self, solution):
        print(colored("Fitness Function >> ", color='blue', on_color='on_grey'))
        weights_to_train = self.model.get_weights()
        weight_list = weights_to_train[5]

        if np.ndim(weight_list) == 1:
            weights = solution.flatten()
        else:
            weights = solution.reshape(weight_list.shape[0], weight_list.shape[1])

        weights_to_train[5] = weights
        self.model.set_weights(weights_to_train)

        preds = self.model.predict(self.x_test)

        # Convert both to integer labels
        if self.y_test.ndim > 1 and self.y_test.shape[1] > 1:
            y_true = np.argmax(self.y_test, axis=1)
        else:
            y_true = self.y_test

        y_pred = np.argmax(preds, axis=1)
        acc = accuracy_score(y_true, y_pred)
        return acc

    def optimization(self, c_weights, opt):
        problem_dict1 = {
            "bounds": FloatVar(lb=(c_weights.min(),) * c_weights.size,
                               ub=(c_weights.max(),) * c_weights.size,
                               name="delta"),
            "minmax": "min",
            "obj_func": self.fitness_function1,
            "log_to": None,
            "save_population": False,
            "Curr_Weight": c_weights,
            "Model_trained_Partial": self.model,
            "test_loader": self.x_test,
            "tst_lab": self.y_test,
        }
        if opt == 1:
            best_solution, best_fitness = BCO(epochs=100, pop_size=10, to_optimize=c_weights,fitness_func=self.fitness_function1)
        elif opt == 2:
            best_solution, best_fitness = Walrus_opt(epochs=100, pop_size=10)
        else:
            best_solution, best_fitness = BCO(epochs=100, pop_size=10, to_optimize=c_weights,fitness_func=self.fitness_function1)

        return best_solution

    def main_update_hyperparameters(self, option):
        weights_to_train = self.model.get_weights()

        # Debug print (optional)
        for i, w in enumerate(weights_to_train):
            print(f"weights index {i}: shape={np.shape(w)}")

        # find first 2D weight matrix (kernel) to optimize
        target_idx = None
        for i, w in enumerate(weights_to_train):
            if np.ndim(w) == 2:
                target_idx = i
                break

        if target_idx is None:
            raise RuntimeError("No 2D weight matrix found in model.get_weights()")

        # select the 2D weight matrix
        to_op_1 = np.array(weights_to_train[target_idx])  # shape (rows, cols)

        # No need to reshape if already 2D; keep as is for optimization input
        sh_to_op_1 = to_op_1.copy()

        # run optimization (your optimization() expects a 2D array)
        # Suppose weights_flat_solution is from BCO or WOA
        weights_flat_solution = self.optimization(sh_to_op_1, option)

        # Convert to np.array
        weights_flat_solution = np.array(weights_flat_solution)

        # Only reshape if the sizes match
        if weights_flat_solution.size == to_op_1.size:
            to_opt_new = weights_flat_solution.reshape(to_op_1.shape)
        else:
            # If sizes differ, handle separately for WOA
            # Option 1: pad with zeros
            if weights_flat_solution.size < to_op_1.size:
                padded = np.zeros(to_op_1.size)
                padded[:weights_flat_solution.size] = weights_flat_solution
                to_opt_new = padded.reshape(to_op_1.shape)
            # Option 2: truncate if too large
            else:
                to_opt_new = weights_flat_solution[:to_op_1.size].reshape(to_op_1.shape)

        # Now to_opt_new is safely shaped for both BCO and WOA

        # put back into full weights list and set model weights
        weights_to_train[target_idx] = to_opt_new
        self.model.set_weights(weights_to_train)
        return self.model


# ---------------------------
# Model builder: adjusted
# - IMPORTANT CHANGE: When opt==3 we just return the compiled model (federated training handled in proposed_model)
# - For opt==1/2 keep previous behavior of running optimization if you want (same as original code)
# ---------------------------
def build_model(x_test, y_test, input_shape, num_classes, opt):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(10, return_sequences=True))
    model.add(LSTM(10, return_sequences=False))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    if opt == 0:
        return model
    else:
        if opt == 1:
            print(colored("[INFO] Updating weights using Border Collie Algorithm", "yellow"))
            op = optimization(model, x_test, y_test)
            return op.main_update_hyperparameters(opt)
        elif opt == 2:
            print(colored("[INFO] Updating weights using Walrus Algorithm", "yellow"))
            op = optimization(model, x_test, y_test)
            return op.main_update_hyperparameters(opt)
        elif opt == 3:
            # For hybrid (opt==3) just return the base model.
            # Federated training will be executed in proposed_model (keeps flow clearer).
            print(colored("[INFO] Building model for Hybrid Federated Learning (opt==3)", "yellow"))
            return model
        else:
            # fallback: return model
            return model

# ---------------------------
# Helper: FedAvg averaging for Keras weights
# ---------------------------
def federated_average(weights_list, sample_counts):
    """
    weights_list: list of lists (each inner list are model.get_weights() from a client)
    sample_counts: list of ints per client
    returns: averaged weights (list)
    """
    # number of weight arrays
    avg_weights = []
    total_samples = float(sum(sample_counts))
    # iterate through each weight tensor index
    for layer_idx in range(len(weights_list[0])):
        # sum of weighted arrays
        accum = None
        for client_idx, client_weights in enumerate(weights_list):
            w = np.array(client_weights[layer_idx], dtype=np.float64)
            weight_factor = sample_counts[client_idx] / total_samples
            if accum is None:
                accum = w * weight_factor
            else:
                accum = accum + w * weight_factor
        avg_weights.append(accum.astype(weights_list[0][layer_idx].dtype))
    return avg_weights

# ---------------------------
# Main proposed model function (with federated logic for opt==3)
# ---------------------------
def proposed_model(x_train, x_test, y_train, y_test, train_percent, DB, opt, epochs_list=None):
    """
    x_train, x_test: numpy arrays of data
    y_train, y_test: label arrays (integer encoded)
    train_percent: float (e.g., 0.6)
    DB: dataset name used for folder creation
    opt: 0/1/2/3 (3 = Hybrid federated)
    epochs_list: list, e.g. [1,200,300,400,500]
    """
    if epochs_list is None:
        epochs_list = [1, 200, 300, 400, 500]

    input_shape = x_train.shape
    num_classes = len(np.unique(y_train))
    # input_shape = (x_train.shape[1], x_train.shape[2])
    # num_classes = len(np.unique(y_train))

    # Debug check
    print("Before reshape:", x_train.shape)

    # Reshape for LSTM input -> (samples, timesteps, features)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Debug check
    print("After reshape:", x_train.shape)

    # Now safe to define input_shape
    input_shape = (x_train.shape[1], x_train.shape[2])  # (timesteps, features)
    print("Input shape for model:", input_shape)

    # One-hot encode labels
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    # Build the model
    model = build_model(x_train, y_train_cat, input_shape, num_classes, opt)

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    # Create folder structure (unchanged)
    opt_names = {0: 'NoOpt', 1: 'BCO', 2: 'Walrus', 3: 'Hybrid'}
    opt_name = opt_names.get(opt, 'NoOpt')

    checkpoint_dir = f"Checkpoint/{DB}/{opt_name}/TP_{int(train_percent * 100)}"
    metric_path = f"Analysis/Performance_Analysis/{opt_name}/{DB}/"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(metric_path, exist_ok=True)
    os.makedirs("Architectures/", exist_ok=True)
    os.makedirs("Saved_model/", exist_ok=True)
    os.makedirs("Hybrid_drift", exist_ok=True)  # ensure drift folder exists

    # try to find the highest saved epoch (like before)
    prev_epoch = 0
    for ep in reversed(epochs_list):
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{ep}.weights.h5")
        metrics_file = os.path.join(metric_path, f"metrics_{train_percent}percent_epoch{ep}.npy")
        if os.path.exists(checkpoint_path) and os.path.exists(metrics_file):
            print(f"Found existing checkpoint and metrics for epoch {ep}, loading model...")
            model.load_weights(checkpoint_path)
            prev_epoch = ep
            break

    metrics_all = {}

    # ---------------------------
    # Federated-specific config (used only for opt==3)
    # ---------------------------
    NUM_CLIENTS = 3  # you can make this configurable
    CLIENT_BATCH = 8
    # split x_train into NUM_CLIENTS non-overlapping parts (simple IID split)
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    client_indices = np.array_split(indices, NUM_CLIENTS)
    clients_data = []
    for idxs in client_indices:
        clients_data.append((x_train[idxs], y_train_cat[idxs]))  # y as categorical for Keras .fit

    # Read drift values if present; expected file contains 3 numbers e.g. [acc_prev, spec_prev, sens_prev]
    drift_path = os.path.join("Hybrid_drift", "Check.npy")
    if os.path.exists(drift_path):
        try:
            drift_vals = np.load(drift_path)
        except Exception as e:
            print(f"[WARN] Could not read drift file {drift_path}: {e}. Using zeros.")
            drift_vals = np.array([0.0, 0.0, 0.0])
    else:
        drift_vals = np.array([0.0, 0.0, 0.0])

    # ---------------------------
    # TRAINING loop: iterate epoch segments (1,200,...)
    # If opt != 3, original single-machine .fit is used (keeps flow)
    # If opt == 3, run federated training where each epoch number corresponds to that many FL rounds
    # ---------------------------
    for end_epoch in epochs_list:
        if end_epoch <= prev_epoch:
            continue

        print(f"\nTraining segment: epochs up to {end_epoch} (prev_epoch={prev_epoch}) for TP={train_percent * 100}% ...")

        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{end_epoch}.weights.h5")
        metrics_file = os.path.join(metric_path, f"metrics_{train_percent}percent_epoch{end_epoch}.npy")

        if opt != 3:
            # --- Original behavior (no federated learning) ---
            try:
                model.fit(x_train, y_train_cat,
                          epochs=end_epoch,
                          initial_epoch=prev_epoch,
                          batch_size=8,
                          validation_split=0.2,
                          verbose=2)

                # plot_model(model, to_file="Architectures/model_architecture.png", show_shapes=True, show_layer_names=True)

                model.save(f"Saved_model/{DB}_model.h5")
                model.save_weights(checkpoint_path)
                print(f"Checkpoint saved at: {checkpoint_path}")

                preds = model.predict(x_test)
                y_pred = np.argmax(preds, axis=1)
                y_test_labels = np.argmax(y_test_cat, axis=1)
                metrics = main_est_parameters(y_test_labels, y_pred)

                metrics_all[f"epoch_{end_epoch}"] = metrics
                np.save(metrics_file, metrics)
                print(f"Metrics saved at: {metrics_file}")

                prev_epoch = end_epoch
                model.save(f"Saved_model/{DB}_model.keras")

            except KeyboardInterrupt:
                print(f"Training interrupted during epochs {prev_epoch + 1} to {end_epoch}. Not saving checkpoint or metrics.")
                raise

        else:
            # --- Federated training for opt==3 (Hybrid) ---
            rounds_to_run = end_epoch - prev_epoch
            print(f"[FEDERAL] Running {rounds_to_run} federated rounds (each client trains 1 epoch per round).")

            def run_federated_rounds(num_rounds):
                nonlocal model
                for r in range(num_rounds):
                    print(f"[FEDERAL] Round {r+1}/{num_rounds} ...")
                    global_weights = model.get_weights()
                    client_weights_list = []
                    client_sizes = []
                    # each client trains on its local data for 1 epoch
                    for cid, (cx, cy) in enumerate(clients_data):
                        client_model = build_model(x_test, y_test, input_shape, num_classes, opt=0)
                        client_model.compile(optimizer=Adam(learning_rate=0.001),
                                             loss='categorical_crossentropy', metrics=['accuracy'])
                        # set global weights
                        client_model.set_weights(global_weights)
                        if len(cx) == 0:
                            print(f"Client {cid} has 0 samples, skipping...")
                            continue
                        # train locally for 1 epoch
                        client_model.fit(cx, cy, epochs=1, batch_size=CLIENT_BATCH, verbose=0)
                        cw = client_model.get_weights()
                        client_weights_list.append(cw)
                        client_sizes.append(len(cx))
                        print(f"    Client {cid} done (samples={len(cx)}).")
                    if len(client_weights_list) == 0:
                        print("[FEDERAL] No client updates this round. Skipping aggregation.")
                        continue
                    # aggregate
                    new_global = federated_average(client_weights_list, client_sizes)
                    model.set_weights(new_global)
                    print("[FEDERAL] Aggregation complete for this round.")
                return model

            try:
                # Run federated rounds once
                model = run_federated_rounds(rounds_to_run)

                # After finishing the rounds, evaluate
                plot_model(model, to_file="Architectures/model_architecture.png", show_shapes=True, show_layer_names=True)

                model.save(f"Saved_model/{DB}_model.h5")
                model.save_weights(checkpoint_path)
                print(f"Checkpoint saved at: {checkpoint_path}")

                preds = model.predict(x_test)
                y_pred = np.argmax(preds, axis=1)
                y_test_labels = np.argmax(y_test_cat, axis=1)
                metrics = main_est_parameters(y_test_labels, y_pred)

                metrics_all[f"epoch_{end_epoch}"] = metrics
                np.save(metrics_file, metrics)
                print(f"Metrics saved at: {metrics_file}")

                # ----- DRIFT CHECK logic -----
                # metrics must contain Accuracy/Specificity/Sensitivity (variants tolerated)
                def extract_metric(mdict, key_options):
                    for k in key_options:
                        if k in mdict:
                            return float(mdict[k])
                    # fallback 0
                    return 0.0

                acc = extract_metric(metrics, ['Accuracy', 'accuracy', 'ACC', 'acc'])
                spec = extract_metric(metrics, ['Specificity', 'specificity', 'SPEC', 'spec'])
                sens = extract_metric(metrics, ['Sensitivity', 'sensitivity', 'SENS', 'sens', 'Recall', 'recall'])

                print(f"[DRIFT] Current metrics -> Acc: {acc}, Spec: {spec}, Sens: {sens}")
                print(f"[DRIFT] Stored drift values from {drift_path}: {drift_vals}")

                # Condition: if all three drift_vals are less than 10% of current metric => retrain this segment once more
                retrain_flag = True
                # protect division by zero if metric is zero; if metric is zero and drift_val < 0.1*0 => 0 compare; we treat as False so no retrain.
                comparisons = []
                for dv, cur in zip(drift_vals, [acc, spec, sens]):
                    if float(cur) <= 0.0:
                        comparisons.append(False)
                    else:
                        comparisons.append(float(dv) < 0.1 * float(cur))
                retrain_flag = all(comparisons)

                if retrain_flag:
                    print(colored("[DRIFT] Triggered drift condition -> Retraining this federated portion one more time.", "red"))
                    # run federated rounds one more time for the same number of rounds
                    model = run_federated_rounds(rounds_to_run)
                    # re-evaluate and overwrite checkpoint & metrics
                    model.save(f"Saved_model/{DB}_model.h5")
                    model.save_weights(checkpoint_path)
                    preds = model.predict(x_test)
                    y_pred = np.argmax(preds, axis=1)
                    metrics = main_est_parameters(y_test_labels, y_pred)
                    metrics_all[f"epoch_{end_epoch}_retrained"] = metrics
                    np.save(metrics_file, metrics)  # overwrite (or you can save another file)
                    print(f"[DRIFT] Retrain complete and metrics overwritten at {metrics_file}")

                # update prev_epoch after success
                prev_epoch = end_epoch
                model.save(f"Saved_model/{DB}_model.keras")

            except KeyboardInterrupt:
                print(f"Federated training interrupted during epochs {prev_epoch + 1} to {end_epoch}. Not saving checkpoint or metrics.")
                raise

        print(f"Completed training segment up to epoch {end_epoch} (TP={train_percent * 100}%).")

    return metrics_all

# ---------------------------
# End of file
# ---------------------------
