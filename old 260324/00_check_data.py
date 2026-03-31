# 資料驗證階段，僅確認讀檔沒問題，不做任何模型訓練

import os
import numpy as np
from keel_io import load_5fold_dataset

# 三個 dataset
DATASETS = [
    "ecoli-0-1-3-7_vs_2-6-5-fold",
    "glass-0-1-2-3_vs_4-5-6-5-fold",
    "yeast-0-5-6-7-9_vs_4-5-fold",
]

BASE_PATH = "data"


for dataset in DATASETS:

    print(f"\n===== {dataset} =====")

    # 讀取該 dataset 的 5 folds
    folds = load_5fold_dataset(os.path.join(BASE_PATH, dataset))

    # 每個 fold 都印出資訊
    for i, (X_train, y_train, X_test, y_test) in enumerate(folds):

        print(f"\nFold {i+1}")

        # 看資料維度
        print("Train shape:", X_train.shape)
        print("Test shape :", X_test.shape)

        # 看 label 分布
        values, counts = np.unique(y_train, return_counts=True)
        print("Train label distribution:", dict(zip(values, counts)))