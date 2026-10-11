# ============================================================
# 【Baseline A v1】One-Class Classification (OCC) Baseline
# ============================================================
# 使用 OF_maj (Only Majority) 策略 + OneClassSVM
# - Kernel: RBF
# - Gamma: "scale"
# - 評估方式：8 個資料集 × 5-fold = 40 個實驗
# - 評估指標：AUC / Recall / F1 / G-mean
# 
# ★ v1 核心原則：
#   1) StandardScaler 只在 majority 類別上 fit（符合 OCC 原則）
#   2) 只用 majority 訓練 OneClassSVM
#   3) 參數調優（nu）只在 training 內部 validation 進行
#   4) 不使用 test set 進行任何調參
# 
# 【流程概覽】
#   ├─ 讀取 8 個資料集 (KEEL .dat 格式)
#   ├─ 對每個資料集的 5 折進行：
#   │  ├─ 載入 train/test 檔案
#   │  ├─ 自動判別 majority/minority 類別
#   │  ├─ 在 training 內部進行 80/20 stratified split
#   │  ├─ 測試 nu ∈ {0.01, 0.05, 0.1}，選最佳 AUC
#   │  ├─ 用全部 training majority 重訓最佳模型
#   │  └─ 在 test set 評估：AUC、F1、Recall、G-mean
#   └─ 輸出 3 張 Excel 表（per_fold / summary / overall_mean）
# ============================================================

import os
import numpy as np
import pandas as pd

# sklearn：我們只用最基本的 OCC baseline（One-Class SVM）
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

# 用來在 training fold 內再切 validation（不能用 test 來調參）
from sklearn.model_selection import StratifiedShuffleSplit

# 評估指標
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    confusion_matrix
)

# ============================================================
# 1) 讀取 KEEL .dat 檔案
#    KEEL 格式通常是：
#    @relation ...
#    @attribute f1 ...
#    ...
#    @attribute class {negative,positive}
#    @data
#    0.1,0.2,...,negative
#    ...
# ============================================================
def load_keel_dat(path):
    """
    讀取 KEEL 的 .dat 格式
    回傳：
      X: 特徵矩陣 (n_samples, n_features)
      y: 類別標籤 (n_samples,)
      feature_names: 特徵名稱

    ★ 支援類別型特徵（如 abalone 的 Sex: M/F/I）：
      - 若某欄無法轉換為 float，則對該欄做 Label Encoding
      - 編碼依照該欄在「整份資料」中出現的字串排序，確保一致性
    """
    feature_names = []
    data_started = False
    rows, y = [], []

    # encoding/errors 用 ignore 是為了避免某些資料集含特殊字元導致讀檔失敗
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            # 空行或註解行略過
            if not line or line.startswith("%"):
                continue

            # 在 @data 之前：解析 attribute 名稱
            if not data_started:
                if line.lower().startswith("@attribute"):
                    parts = line.split()

                    # parts[1] = 欄位名稱
                    # 但 class 欄位不是 feature，因此要排除
                    if len(parts) >= 3:
                        name = parts[1]
                        if name.lower() != "class":
                            feature_names.append(name)

                # 碰到 @data 後開始讀資料列
                if line.lower().startswith("@data"):
                    data_started = True
                continue

            # 在 @data 之後：每一行就是一筆資料
            # 格式：f1, f2, ..., classLabel
            parts = [p.strip() for p in line.split(",")]
            *feat, label = parts

            rows.append(feat)
            y.append(label)

    if not rows:
        return np.array([]), np.array([]), feature_names

    # ── 逐欄嘗試轉 float，失敗則 Label Encode ──────────────────
    n_cols = len(rows[0])
    X = np.zeros((len(rows), n_cols), dtype=float)

    for col_idx in range(n_cols):
        col_vals = [r[col_idx] for r in rows]
        try:
            # 嘗試整欄直接轉 float
            X[:, col_idx] = [float(v) for v in col_vals]
        except ValueError:
            # 有字串值 → Label Encoding（依排序後的唯一值編碼）
            unique_vals = sorted(set(col_vals))
            mapping = {v: i for i, v in enumerate(unique_vals)}
            X[:, col_idx] = [float(mapping[v]) for v in col_vals]

    return X, np.array(y, dtype=str), feature_names


# ============================================================
# 2) G-mean 計算
#    G-mean = sqrt(TPR * TNR)
#    - TPR = Recall(minority) = tp / (tp + fn)
#    - TNR = Specificity = tn / (tn + fp)
#    在 class imbalance 下，比 accuracy 更有意義
# ============================================================
def gmean(cm):
    """
    cm 是 confusion matrix：
    [[tn, fp],
     [fn, tp]]
    """
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # minority recall
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # specificity
    return np.sqrt(tpr * tnr)


# ============================================================
# 3) Baseline a：OF_maj -> OCC
#
#    使用預先切好的五折資料：
#    - 訓練集 (tra.dat)：只用 majority 類別訓練
#    - 測試集 (tst.dat)：評估性能
#
#    參數調優策略：
#    - gamma 固定 "scale"
#    - nu 只測 {0.01, 0.05, 0.1}
#    - 在訓練集內做 internal stratified split (80/20)
#    - 用 validation AUC 選最好的 nu
# ============================================================
def tune_baseline_a(X, y, maj_label, min_label, random_state=42):
    """
    在訓練集內部做輕量調參：
      1) Split: 80% train / 20% validation（分層抽樣）
      2) Fit: 只用 train split 的 majority
      3) Select nu: 用 validation AUC 最高者
      4) Refit: 用整個訓練集的 majority 重訓最佳模型

    回傳：
      final_model: 重新用全 training majority 訓練後的模型
      best_nu: 被選到的 nu
    """

    # Stratified: 保持 validation 中 minority/majority 的比例
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=random_state
    )
    idx_train, idx_val = next(splitter.split(X, y))

    X_train, y_train = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]

    # ★ Baseline a 核心：只用 majority 訓練 OCC
    X_train_maj = X_train[y_train == maj_label]

    # nu：OneClassSVM 允許多少比例的資料被視為 outlier
    candidate_nu = [0.01, 0.05, 0.1]
    best_auc = -1
    best_nu = None

    for nu in candidate_nu:
        # ★ StandardScaler 只在 majority 上 fit，符合 OCC 原則
        scaler = StandardScaler()
        scaler.fit(X_train_maj)  # 只看 majority 的 mean/std
        X_train_maj_scaled = scaler.transform(X_train_maj)
        X_val_scaled = scaler.transform(X_val)  # val 用同一組統計量 transform

        ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)
        ocsvm.fit(X_train_maj_scaled)

        scores = ocsvm.decision_function(X_val_scaled).ravel()

        # decision_function 越大越像 inlier（像 majority）
        # 但 AUC 我們希望：minority = anomaly = score 越大越好
        # 所以用 anomaly = -decision_function
        anomaly = -scores

        # minority 當作正類(1)
        y_val_bin = (y_val == min_label).astype(int)

        # validation AUC：只看排序能力，不受 threshold 影響
        try:
            auc = roc_auc_score(y_val_bin, anomaly)
        except:
            # 若 validation 裡剛好沒有 minority，AUC 會算不了
            auc = 0

        if auc > best_auc:
            best_auc = auc
            best_nu = nu

    # --------------------------------------------------------
    # 用「整個訓練集的 majority」重訓最佳模型
    # Scaler 只 fit majority，再 transform 全體
    # --------------------------------------------------------
    X_maj_full = X[y == maj_label]
    final_scaler = StandardScaler()
    final_scaler.fit(X_maj_full)  # 只用 majority fit

    # 建立一個包含 scaler 和 ocsvm 的簡單物件方便後續 evaluate 使用
    class MajScalerOCSVM:
        def __init__(self, scaler, ocsvm):
            self.scaler = scaler
            self.ocsvm = ocsvm
        def decision_function(self, X):
            return self.ocsvm.decision_function(self.scaler.transform(X))
        def predict(self, X):
            return self.ocsvm.predict(self.scaler.transform(X))

    final_ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=best_nu)
    final_ocsvm.fit(final_scaler.transform(X_maj_full))
    final_model = MajScalerOCSVM(final_scaler, final_ocsvm)

    return final_model, best_nu


# ============================================================
# 4) Test 評估
#
#    - AUC：使用 anomaly score（-decision_function）
#    - F1 / Recall / G-mean：需要 binary prediction
#
#    baseline-friendly 的做法：
#    - 直接用 OneClassSVM.predict() 的結果
#      predict(): inlier -> 1, outlier -> -1
# ============================================================
def evaluate(model, X_test, y_test, maj_label, min_label):
    """
    回傳：
      auc, f1, recall, gmean
    """

    # 用於 AUC（排序能力）
    scores = model.decision_function(X_test).ravel()
    anomaly = -scores

    # ground truth：minority=1
    y_true = (y_test == min_label).astype(int)

    # predict()：模型自己內建的 outlier 判斷
    # outlier -> -1 => minority(1)
    y_pred_raw = model.predict(X_test)
    y_pred = (y_pred_raw == -1).astype(int)

    auc = roc_auc_score(y_true, anomaly)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    # confusion matrix 固定 labels=[0,1]，確保 tn/fp/fn/tp 順序不會亂
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    gm = gmean(cm)

    return auc, f1, rec, gm


# ============================================================
# 5) 主程式：跑八個資料集 × 五折
#    - 不同資料集放不同資料夾
#    - 每個 fold 都有 xxx-tra.dat / xxx-tst.dat
#    - 輸出 baseline_a_v1_results.xlsx
# ============================================================
BASE_DIR = "data"

DATASETS = {
    # ── 原始三個資料集 ──────────────────────────────────────────
    "ecoli-0137_vs_26": "ecoli-0-1-3-7_vs_2-6-5-fold",
    "glass-01236_vs_456": "glass-0-1-2-3_vs_4-5-6-5-fold",
    "yeast-05679_vs_45": "yeast-0-5-6-7-9_vs_4-5-fold",
    # ── 低 IR 對照組 ─────────────────────────────────────────────
    "glass1": "glass1-5-fold",                           # IR ≈ 2.0:1，低難度
    "yeast1": "yeast1-5-fold",                           # IR ≈ 2.46:1，低難度
    # ── 新增三個資料集 ────────────────────────────────────────────
    "cleveland-0_vs_4": "cleveland-0_vs_4-5-fold",       # 新增
    "yeast-2_vs_8": "yeast-2_vs_8-5-fold",               # 新增
    "abalone-17_vs_7-8-9-10": "abalone-17_vs_7-8-9-10-5-fold",  # 新增
}

results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

output_lines = []
per_fold_records = []
summary_records = []

for dataset_name, folder in DATASETS.items():

    dataset_path = os.path.join(BASE_DIR, folder)
    output_lines.append(f"\n===== Dataset: {dataset_name} =====")

    auc_list = []
    recall_list = []
    f1_list = []
    gmean_list = []

    for fold in range(1, 6):
        # 自動找尋該 fold 的訓練與測試檔案
        # 檔名格式：*{fold}tra.dat 和 *{fold}tst.dat
        files = os.listdir(dataset_path)
        tr_files = [f for f in files if f.endswith(f"{fold}tra.dat")]
        ts_files = [f for f in files if f.endswith(f"{fold}tst.dat")]

        if not tr_files or not ts_files:
            print(f"Warning: {dataset_name} Fold {fold} 的 train 或 test 檔案不存在，跳過")
            continue

        tr_path = os.path.join(dataset_path, tr_files[0])
        ts_path = os.path.join(dataset_path, ts_files[0])

        # 讀取資料
        Xtr, ytr, _ = load_keel_dat(tr_path)
        Xts, yts, _ = load_keel_dat(ts_path)

        # ----------------------------------------------------
        # 自動找 majority / minority
        # 因為 KEEL 類別名稱可能是 negative/positive 或其他字
        # 用 training fold 的分布來定義 majority/minority（符合研究設定）
        # ----------------------------------------------------
        labels, counts = np.unique(ytr, return_counts=True)
        maj_label = labels[np.argmax(counts)]
        min_label = labels[np.argmin(counts)]

        # 1) tuning（只用 training 內部 validation，不看 test）
        model, best_nu = tune_baseline_a(
            Xtr, ytr, maj_label, min_label,
            random_state=100 + fold
        )

        # 2) test evaluation
        auc, f1, rec, gm = evaluate(model, Xts, yts, maj_label, min_label)

        auc_list.append(auc)
        recall_list.append(rec)
        f1_list.append(f1)
        gmean_list.append(gm)

        per_fold_records.append({
            "dataset": dataset_name,
            "dataset_dir": folder,
            "fold": fold,
            "best_nu": best_nu,
            "auc": float(auc),
            "gmean": float(gm),
            "recall_min": float(rec),
            "f1": float(f1),
        })

        output_lines.append(
            f"Fold {fold}: nu={best_nu} | AUC={auc:.4f} | G-mean={gm:.4f} | Recall(min)={rec:.4f} | F1={f1:.4f}"
        )

    summary_records.append({
        "dataset": dataset_name,
        "dataset_dir": folder,
        "n_folds": len(auc_list),
        "auc_mean": float(np.mean(auc_list)),
        "auc_std": float(np.std(auc_list, ddof=0)),
        "gmean_mean": float(np.mean(gmean_list)),
        "gmean_std": float(np.std(gmean_list, ddof=0)),
        "recall_min_mean": float(np.mean(recall_list)),
        "recall_min_std": float(np.std(recall_list, ddof=0)),
        "f1_mean": float(np.mean(f1_list)),
        "f1_std": float(np.std(f1_list, ddof=0)),
    })

    # 5-fold 平均
    output_lines.append(
        f"Average: AUC={np.mean(auc_list):.4f} ± {np.std(auc_list, ddof=0):.4f} | "
        f"G-mean={np.mean(gmean_list):.4f} ± {np.std(gmean_list, ddof=0):.4f} | "
        f"Recall(min)={np.mean(recall_list):.4f} ± {np.std(recall_list, ddof=0):.4f} | "
        f"F1={np.mean(f1_list):.4f} ± {np.std(f1_list, ddof=0):.4f}"
    )


# ============================================================
# 6) 只寫出 Excel 結果檔
# ============================================================
xlsx_path = os.path.join(results_dir, "baseline_a_v1_results.xlsx")

per_fold_df = pd.DataFrame(per_fold_records)
summary_df = pd.DataFrame(summary_records)

overall_df = pd.DataFrame([{
    "dataset": "ALL",
    "n_datasets": int(summary_df["dataset"].nunique()) if not summary_df.empty else 0,
    "n_folds": int(len(per_fold_df)),
    "auc_mean": float(per_fold_df["auc"].mean()) if not per_fold_df.empty else np.nan,
    "auc_std": float(per_fold_df["auc"].std(ddof=0)) if not per_fold_df.empty else np.nan,
    "gmean_mean": float(per_fold_df["gmean"].mean()) if not per_fold_df.empty else np.nan,
    "gmean_std": float(per_fold_df["gmean"].std(ddof=0)) if not per_fold_df.empty else np.nan,
    "recall_min_mean": float(per_fold_df["recall_min"].mean()) if not per_fold_df.empty else np.nan,
    "recall_min_std": float(per_fold_df["recall_min"].std(ddof=0)) if not per_fold_df.empty else np.nan,
    "f1_mean": float(per_fold_df["f1"].mean()) if not per_fold_df.empty else np.nan,
    "f1_std": float(per_fold_df["f1"].std(ddof=0)) if not per_fold_df.empty else np.nan,
}])

with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    per_fold_df.to_excel(writer, sheet_name="per_fold", index=False)
    summary_df.to_excel(writer, sheet_name="summary", index=False)
    overall_df.to_excel(writer, sheet_name="overall_mean", index=False)

print("\nBaseline A v1 finished.")
print(f"Excel saved to {xlsx_path}")
