import os
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from termcolor import cprint
import random

warnings.filterwarnings('ignore')

# Global dictionary to store registered vehicles (simulating a database)
vehicle_db = {}


def Registered(data, num):
    """
    Simulate vehicle registration: encrypt and store data with a generated AES key.
    """
    data_bytes = data.to_json().encode('utf-8')
    key = get_random_bytes(16)
    cipher = AES.new(key, AES.MODE_EAX)
    ciphertext, tag = cipher.encrypt_and_digest(data_bytes)
    nonce = cipher.nonce

    # Store encrypted data and related info (simulated database)
    vehicle_db[num] = {
        'key': key,
        'ciphertext': ciphertext,
        'tag': tag,
        'nonce': nonce
    }

    print("========================================")
    cprint(f"Vehicle {num} Registered Successfully ", "green")
    return key


def login(num, key_input):
    """
    Simulate vehicle login: decrypt only if the correct key is provided.
    """
    record = vehicle_db.get(num)
    if record is None:
        cprint(f"Vehicle {num} not found in database ❌", "red")
        return None

    try:
        cipher = AES.new(key_input, AES.MODE_EAX, record['nonce'])
        decrypted_data = cipher.decrypt_and_verify(record['ciphertext'], record['tag'])
        cprint(f"Vehicle {num} Login Successful ✅", "cyan")
        return decrypted_data

    except Exception:
        cprint(f"Vehicle {num} Login Failed ❌ due to incorrect key", "red")
        return None


def Cleaning_Data(DB):
    D1 = pd.read_csv(f'Dataset/{DB}/data.csv')
    D1 = D1.iloc[1:100, 0:10]
    D2 = pd.read_csv(f'Dataset/{DB}/rsu1.csv')
    NEW_D = pd.concat([D1, D2], axis=1)
    NEW_D.to_csv(f'Dataset\\{DB}\\Cleaned.csv')
    return NEW_D


def preprocessing(DB):
    if DB == "DB1":
        data = Cleaning_Data(DB)
        data = data.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

        # Step 1: Register all vehicles
        keys = []
        for i in range(len(data)):
            key = Registered(data.iloc[i], i)
            keys.append(key)

        # Step 2: Randomly choose 3 users with wrong keys
        w = random.sample(range(len(data)), 3)

        valid_data = []
        for i in range(len(data)):
            if i in w:
                # Generate an incorrect key intentionally
                wk = get_random_bytes(16)
                decrypted = login(i, wk)
            else:
                decrypted = login(i, keys[i])

            if decrypted is not None:
                valid_data.append(data.iloc[i])

        # Continue only with successfully logged in data
        data = pd.DataFrame(valid_data)
        print(f"\nTotal successful logins: {len(data)} / {len(keys)}")

        # Drop unnecessary columns
        data.drop(['**** SCALARS ****', 'Unnamed: 2', 'run', 'Module'], axis=1, inplace=True, errors='ignore')

        new_list = data['TotalLostPackets'].tolist()
        Avg = np.mean(new_list)
        labels = [1 if val > Avg else 0 for val in new_list]
        data['Attack'] = labels

        feat = data.drop('Attack', axis=1)
        lab = data['Attack']

        # Standardization
        scaler = StandardScaler()
        feat = scaler.fit_transform(feat)
        feat = feat.astype(np.float32) / feat.max()

        # SMOTE balancing
        sm = SMOTE(sampling_strategy={0: 10000}, random_state=42)
        feat, lab = sm.fit_resample(feat, lab)

        # PCA feature selection
        pca = PCA(n_components=10)
        feat = pca.fit_transform(feat)

        os.makedirs(f"Data_loader\\{DB}", exist_ok=True)
        np.save(f'Data_loader\\{DB}\\features.npy', feat)
        np.save(f'Data_loader\\{DB}\\labels.npy', lab)

        print("\nThe data saved successfully.......................🚗🚗 ")


