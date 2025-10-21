# import pandas as pd
# import os
# base_dir_p3 = "Results/DB1/Comparative_Analysis/Bar"
# metrics = ["Accuracy", "F1 Score", "Precision", "Recall", "Sensitivity", "Specificity"]
#
# for metric in metrics:
#     # Find actual filename in Results_P3 directory containing the metric string
#     file_p3 = next((f for f in os.listdir(base_dir_p3) if metric in f and f.endswith(".csv")), None)
#
#     if file_p3:
#         path_p3 = os.path.join(base_dir_p3, file_p3)
#         df_p3 = pd.read_csv(path_p3)
#
#         # Swap row at index 0 with row at index 5
#         temp = df_p3.iloc[0].copy()
#         df_p3.iloc[0] = df_p3.iloc[5]
#         df_p3.iloc[5] = temp
#
#         # Save the swapped DataFrame to the same file
#         df_p3.to_csv(path_p3, index=False)
#         print(f"Swapped rows 0 and 5 in {file_p3}")
#     else:
#         print(f"No file found for metric {metric}")
import os
base_dir_p3 = "Results_P3/DB1/Comparative_Analysis/Bar"
save_dir = "Analysis/Comparative_Analysis/DB1"
os.makedirs(save_dir, exist_ok=True)

# Mapping: metric keyword (as in CSV) to .npy filename
npy_map = {
    "Accuracy": "ACC_1.npy",
    "F1 Score": "F1score_1.npy",
    "Precision": "PRE_1.npy",
    "Recall": "REC_1.npy",
    "Sensitivity": "SEN_1.npy",
    "Specificity": "SPE_1.npy"
}

for metric, npy_name in npy_map.items():
    file_p3 = next((f for f in os.listdir(base_dir_p3) if metric in f and f.endswith(".csv")), None)
    if file_p3:
        path_p3 = os.path.join(base_dir_p3, file_p3)
        df_p3 = pd.read_csv(path_p3, index_col=0)  # remove row names (index column)
        np.save(os.path.join(save_dir, npy_name), df_p3.values)
        print(f"Saved: {npy_name}")
    else:
        print(f"No file found for {metric}")

