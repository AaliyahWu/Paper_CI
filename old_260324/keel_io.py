# 讀 KEEL 5fcv 的 train/test 檔

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def read_keel_dat(filepath):
    """
    讀取 KEEL .dat 檔案
    
    KEEL 格式：
        @relation ...
        @attribute ...
        ...
        @data
        1,2,3,class1
        4,5,6,class2
    
    目的：
        - 解析 attribute 名稱
        - 把資料轉成 DataFrame
        - 將類別特徵轉數值
        - 將 label 轉成 0/1
    """

    # 讀取整個檔案
    with open(filepath, "r") as f:
        lines = f.readlines()

    data_start = False  # 用來判斷是否已進入 @data 區塊
    attributes = []     # 存欄位名稱
    data = []           # 存真正資料

    for line in lines:
        line = line.strip()

        # 如果是 attribute 行，記錄欄位名稱
        if line.lower().startswith("@attribute"):
            parts = line.split()
            attr_name = parts[1]
            attributes.append(attr_name)

        # 如果看到 @data，代表後面開始是資料
        if line.lower().startswith("@data"):
            data_start = True
            continue

        # 進入資料區塊後開始收集資料
        if data_start:
            if line != "":
                data.append(line.split(","))

    # 建立 DataFrame
    df = pd.DataFrame(data, columns=attributes)

    # 最後一欄是 label
    X = df.iloc[:, :-1]  # 所有特徵
    y = df.iloc[:, -1]   # 最後一欄（分類標籤）

    # 如果某些特徵是字串（類別型）
    # 要轉成數值（機器學習只能處理數值）
    for col in X.columns:
        if X[col].dtype == object:
            X.loc[:, col] = LabelEncoder().fit_transform(X[col])

    # 將 label 轉為 0 / 1
    # 例如：
    #   minority_class -> 1
    #   majority_class -> 0
    y = LabelEncoder().fit_transform(y)

    # 轉成 numpy
    return X.values.astype(float), y.astype(int)


def load_5fold_dataset(dataset_folder):
    """
    載入 KEEL 的 5-fold 資料夾
    
    每個 fold 有：
        - 5-1tra.dat
        - 5-1tst.dat
        ...
        - 5-5tra.dat
        - 5-5tst.dat

    回傳：
        list of (X_train, y_train, X_test, y_test)
    """

    files = os.listdir(dataset_folder)

    folds = []

    # KEEL 已經切好 5 folds
    for i in range(1, 6):

        # 找到對應 fold 的 train/test 檔
        tra_file = [f for f in files if f"-5-{i}tra.dat" in f][0]
        tst_file = [f for f in files if f"-5-{i}tst.dat" in f][0]

        # 讀檔
        X_train, y_train = read_keel_dat(os.path.join(dataset_folder, tra_file))
        X_test, y_test = read_keel_dat(os.path.join(dataset_folder, tst_file))

        folds.append((X_train, y_train, X_test, y_test))

    return folds