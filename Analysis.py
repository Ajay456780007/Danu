import os

import numpy as np
from sklearn.model_selection import StratifiedKFold
from termcolor import cprint, colored

from Comparative_model.AE import AE
from Comparative_model.Beit_CNN import BeiT_CNN
from Comparative_model.CNN import CNN
from Comparative_model.DCNN import DCNN
from Comparative_model.InceptionV3 import Inception_V3
from Comparative_model.SVM import SVM
from Comparative_model.VGG19 import VGG_19
from Sub_Functions.Load_data import models_return_metrics, Load_data2, Load_data, train_test_splitter, \
    train_test_splitter1
# from Proposed_model.Proposed_model import proposed_model
from Proposed_model.proposed_model import proposed_model


class Analysis:
    def __init__(self, Data):
        self.lab = None
        self.feat = None
        self.DB = Data
        self.E = [20, 40, 60, 80, 100]
        self.save = False

    def Data_loading(self):
        self.feat = np.load(f"data_loader/{self.DB}/features.npy")
        # loading the labels
        self.lab = np.load(f"data_loader/{self.DB}/labels.npy")

    def COMP_Analysis(self):
        self.Data_loading()
        tr = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        models_return_metrics(self.DB, ok=True, epochs=100)

        perf_names = ["ACC", "SEN", "SPE", "F1score", "REC", "PRE", "TPR", "FPR", "PC1", "RE1"]
        files_name = [f"Analysis/Comparative_Analysis/{self.DB}/{name}_1.npy" for name in perf_names]
        # model_registry = {
        #     "AA_DCN": AA_DCN,
        #     "MUFM": MUFM,
        #     "Facenet_512": Facenet_512,
        #     "CVVGG_19": CVVGG_19,
        #     "EA_Net": EA_Net,
        #     "CNN": CNN,
        #     "SimFLE": SimFLE
        # }
        A = np.load(f"Temp/{self.DB}/Comp/AA_DCN/all_metrics.npy", allow_pickle=True).tolist()
        B = np.load(f"Temp/{self.DB}/Comp/MUFM/all_metrics.npy", allow_pickle=True).tolist()
        C = np.load(f"Temp/{self.DB}/Comp/Facenet_512/all_metrics.npy", allow_pickle=True).tolist()
        D = np.load(f"Temp/{self.DB}/Comp/CVVGG_19/all_metrics.npy", allow_pickle=True).tolist()
        E = np.load(f"Temp/{self.DB}/Comp/EA_Net/all_metrics.npy", allow_pickle=True).tolist()
        F = np.load(f"Temp/{self.DB}/Comp/CNN/all_metrics.npy", allow_pickle=True).tolist()
        G = np.load(f"Temp/{self.DB}/Comp/SimFLE/all_metrics.npy", allow_pickle=True).tolist()

        all_models = [A, B, C, D, E, F, G]

        if self.save:
            for j in range(len(perf_names)):
                new = []
                for model_metrics in all_models:
                    x = [row[j] for row in model_metrics]
                    new.append(x)
                if self.save:
                    np.save(files_name[j], np.array(new))

    def KF_Analysis(self):
        kr = [6, 7, 8, 9, 10]
        model = {
            "AE": AE,
            "BeiT_CNN": BeiT_CNN,
            "CNN": CNN,
            "DCNN": DCNN,
            "Inceptionv3": Inception_V3,
            "SVM": SVM,
            "VGG19": VGG_19
        }

        for model_name, model_fn in model.items():
            if os.path.isfile(f"Temp/KF/{self.DB}/{model_name}.npy"):
                print(f"Skipping the {model_name}_model as it has already been run")
                continue

            output = []
            for i in range(len(kr)):
                kr[i] = 2
                kf = StratifiedKFold(n_splits=kr[i], shuffle=True, random_state=42)
                self.Data_loading()
                kfold = kf.split(self.feat, self.lab)
                out = []
                for k, (train_index, test_index) in enumerate(kfold):
                    # feat1, label1 = Load_data2(self.DB)
                    # feat, label = balance2(self.DB, feat1, label1)
                    print(f"  → Training {model_name}....")
                    x_train, x_test, y_train, y_test = train_test_splitter1(self.DB, 0.7, num=30)

                    # Train and evaluate model
                    model2 = model_fn(x_train, x_test, y_train, y_test, 1)
                    out.append(model2)
                a = np.array(out)
                mean = np.mean(a, axis=0)
                output.append(mean)

            os.makedirs(f"Temp/KF/{self.DB}", exist_ok=True)
            np.save(f"Temp/KF/{self.DB}/{model_name}.npy", np.array(output))

        A = np.load(f"Temp/KF/{self.DB}/AE.npy")
        B = np.load(f"Temp/KF/{self.DB}/BeiT_CNN.npy")
        C = np.load(f"Temp/KF/{self.DB}/CNN.npy")
        D = np.load(f"Temp/KF/{self.DB}/DCNN.npy")
        E = np.load(f"Temp/KF/{self.DB}/Inceptionv3.npy")
        F = np.load(f"Temp/KF/{self.DB}/SVM.npy")
        G = np.load(f"Temp/KF/{self.DB}/VGG19.npy")
        H = np.load(f"Temp/KF/{self.DB}/PM_WCL.npy")

        perf_names = ["ACC", "SEN", "SPE", "F1score", "REC", "PRE", "TPR", "FPR"]
        files_name = [f'Analysis/KF_Analysis/{self.DB}/{name}_2.npy' for name in perf_names]

        for i in range(len(perf_names)):
            max_out = []
            max_out.append(A.T[i])
            max_out.append(B.T[i])
            max_out.append(C.T[i])
            max_out.append(D.T[i])
            max_out.append(E.T[i])
            max_out.append(F.T[i])
            max_out.append(G.T[i])
            max_out.append(H.T[i])
            os.makedirs(f"Analysis1/KF_Analysis/{self.DB}/", exist_ok=True)
            if self.save:
                np.save(files_name[i], np.array(max_out))

    def PERF_Analysis(self):
        epoch = [0]
        Performance_Results = []
        Training_Percentage = 0.4
        print("The performance Analysis starts....")

        for i in range(6):
            cprint(f"[⚠️] Performance Analysis Count  Is {i + 1} Out Of 6", 'cyan', on_color='on_grey')
            output = []
            for ep in epoch:
                x_train, x_test, y_train, y_test = train_test_splitter1(self.DB, percent=Training_Percentage)
                result = proposed_model(x_train, x_test, y_train, y_test, Training_Percentage, self.DB)
                output.append(result)
            Performance_Results.append(output)

            Training_Percentage += 0.1

        cprint("The results are saved successfully")
        cprint("[✅] Execution of Performance Analysis Completed", 'green', on_color='on_grey')
