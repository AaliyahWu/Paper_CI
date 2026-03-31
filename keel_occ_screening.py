import os
import io
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 載入單一 .dat 檔
# ============================================================

def load_dat_file(filepath):
    """載入 KEEL .dat，回傳 X (ndarray) 和 y (0=normal, 1=anomaly)"""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    # 過濾 KEEL 非標準 ARFF 行（@inputs / @outputs / @input / @output）
    filtered = [l for l in lines if not l.strip().lower().startswith(('@inputs', '@outputs', '@input', '@output'))]
    data, meta = arff.loadarff(io.StringIO(''.join(filtered)))
    df = pd.DataFrame(data)
    for col in df.select_dtypes([object]):
        df[col] = df[col].str.decode('utf-8')

    label_col = df.columns[-1]
    normal_class = df[label_col].value_counts().idxmax()
    y = (df[label_col] != normal_class).astype(int).values

    X_df = df.drop(columns=[label_col])
    # 類別型特徵欄轉成整數 label encoding
    for col in X_df.select_dtypes(['object']):
        X_df[col] = pd.Categorical(X_df[col]).codes
    X = X_df.astype(float).values
    return X, y

# ============================================================
# 2. 掃描資料集結構：每個資料集找出所有 fold 的 (tra, tst) 配對
# ============================================================

def scan_keel_folds(root_dir):
    """
    掃描 root_dir 下每個子資料夾，
    找出所有 *{n}tra.dat / *{n}tst.dat 配對。

    回傳：
      dict { dataset_name: [ (tra_path, tst_path), ... ] }
    """
    dataset_folds = {}

    for ds_name in sorted(os.listdir(root_dir)):
        ds_path = os.path.join(root_dir, ds_name)
        if not os.path.isdir(ds_path):
            continue

        # 只收集資料夾直接下的 .dat 檔，不進子資料夾
        all_dats = [
            os.path.join(ds_path, fname)
            for fname in os.listdir(ds_path)
            if fname.endswith('.dat') and os.path.isfile(os.path.join(ds_path, fname))
        ]

        # 配對 tra / tst（依 fold 編號）
        tra_files = sorted([p for p in all_dats if 'tra.dat' in os.path.basename(p)])
        tst_files = sorted([p for p in all_dats if 'tst.dat' in os.path.basename(p)])

        pairs = list(zip(tra_files, tst_files))
        if pairs:
            dataset_folds[ds_name] = pairs

    return dataset_folds

# ============================================================
# 3. 對單一資料集做 5-fold OCC 評估（預設參數）
# ============================================================

def evaluate_occ_kfold(fold_pairs):
    """
    fold_pairs: list of (tra_path, tst_path)
    各 fold：用 tra 的 normal 樣本訓練，在 tst 上算 AUC。
    回傳各 OCC 的平均 AUC（across folds）。
    """
    auc_records = {'OCSVM': [], 'LOF': [], 'IF': []}

    for tra_path, tst_path in fold_pairs:
        try:
            X_tra, y_tra = load_dat_file(tra_path)
            X_tst, y_tst = load_dat_file(tst_path)
        except Exception as e:
            print(f"    載入失敗: {e}")
            continue

        # 測試集需有異常樣本才能算 AUC
        if y_tst.sum() < 1:
            continue

        # 只用訓練集的 normal 樣本訓練
        X_train_normal = X_tra[y_tra == 0]
        if len(X_train_normal) < 5:
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_normal)
        X_tst_scaled   = scaler.transform(X_tst)

        # OCSVM
        try:
            clf = OneClassSVM()
            clf.fit(X_train_scaled)
            scores = -clf.decision_function(X_tst_scaled)
            auc_records['OCSVM'].append(roc_auc_score(y_tst, scores))
        except: pass

        # LOF
        try:
            clf = LocalOutlierFactor(novelty=True)
            clf.fit(X_train_scaled)
            scores = -clf.decision_function(X_tst_scaled)
            auc_records['LOF'].append(roc_auc_score(y_tst, scores))
        except: pass

        # Isolation Forest
        try:
            clf = IsolationForest(random_state=42)
            clf.fit(X_train_scaled)
            scores = -clf.decision_function(X_tst_scaled)
            auc_records['IF'].append(roc_auc_score(y_tst, scores))
        except: pass

    return {
        k: round(np.mean(v), 4) if v else np.nan
        for k, v in auc_records.items()
    }

# ============================================================
# 4. 跑全部資料集，篩出至少兩種 OCC AUC < 0.7
# ============================================================

def run_all_keel(root_dir, threshold=0.7):
    dataset_folds = scan_keel_folds(root_dir)
    print(f"找到 {len(dataset_folds)} 個資料集，開始 5-fold 評估...\n")

    records = []

    for ds_name, fold_pairs in dataset_folds.items():
        print(f"處理: {ds_name} ({len(fold_pairs)} folds) ...", end=' ', flush=True)

        # 取第一個 fold 的 tra 來抓 meta 資訊
        try:
            X_sample, y_sample = load_dat_file(fold_pairs[0][0])
            n_features  = X_sample.shape[1]
            # 用所有 fold 的 tra 估算整體異常比例
            all_y = np.concatenate([load_dat_file(tp)[1] for tp, _ in fold_pairs])
            anomaly_ratio = round(all_y.mean(), 4)
            n_samples     = len(all_y)
            n_anomaly     = int(all_y.sum())
        except Exception as e:
            print(f"!! meta 載入失敗: {e}")
            continue

        aucs = evaluate_occ_kfold(fold_pairs)

        n_normal = n_samples - n_anomaly
        ir = round(n_normal / n_anomaly, 4) if n_anomaly > 0 else np.nan

        record = {
            'dataset':       ds_name,
            'n_samples':     n_samples,      # 所有 fold tra 合計
            'n_features':    n_features,
            'n_anomaly':     n_anomaly,
            'anomaly_ratio': anomaly_ratio,
            'IR':            ir,
            'AUC_OCSVM':     aucs['OCSVM'],
            'AUC_LOF':       aucs['LOF'],
            'AUC_IF':        aucs['IF'],
            'AUC_mean':      round(np.nanmean(list(aucs.values())), 4),
        }
        records.append(record)
        print(f"OCSVM={aucs['OCSVM']}  LOF={aucs['LOF']}  IF={aucs['IF']}")

    df_all = pd.DataFrame(records)

    if df_all.empty or 'AUC_OCSVM' not in df_all.columns:
        print("!! 沒有成功評估任何資料集，請確認 KEEL_ROOT 路徑與 .dat 檔案結構。")
        return df_all, pd.DataFrame()

    auc_cols = ['AUC_OCSVM', 'AUC_LOF', 'AUC_IF']
    df_all['n_below'] = (df_all[auc_cols] < threshold).sum(axis=1)
    df_hard = df_all[df_all['n_below'] >= 2].copy()

    print(f"\n=== 至少2種OCC AUC < {threshold}：共 {len(df_hard)} 個資料集 ===")
    print(df_hard[['dataset','n_samples','n_features','IR','anomaly_ratio',
                   'AUC_OCSVM','AUC_LOF','AUC_IF','AUC_mean']].to_string(index=False))

    return df_all, df_hard

# ============================================================
# 5. 挑兩個前測資料集（依 IR 中位數分兩組，各取中位數代表）
# ============================================================

def select_pilot_datasets(df_hard):
    if len(df_hard) < 2:
        print("候選資料集不足2個。")
        return []

    df_sorted = df_hard.dropna(subset=['IR']).sort_values('IR').reset_index(drop=True)
    med_ir = df_sorted['IR'].median()
    print(f"\nIR 中位數 = {med_ir:.4f}")

    low_ir  = df_sorted[df_sorted['IR'] <  med_ir]   # 低IR（不平衡程度較輕）
    high_ir = df_sorted[df_sorted['IR'] >= med_ir]   # 高IR（不平衡程度較重）

    def pick_median_row(df_sub):
        """取 IR 最接近該子群中位數的那一筆"""
        med = df_sub['IR'].median()
        idx = (df_sub['IR'] - med).abs().idxmin()
        return df_sub.loc[idx]

    pilots = []
    if len(low_ir) > 0:
        pilots.append(('低IR組（較平衡）', pick_median_row(low_ir)))
    if len(high_ir) > 0:
        pilots.append(('高IR組（較不平衡）', pick_median_row(high_ir)))

    print("\n" + "="*55)
    print("建議前測資料集")
    print("="*55)
    for label, row in pilots:
        print(f"\n【{label}】{row['dataset']}")
        print(f"  樣本數: {row['n_samples']}  特徵數: {row['n_features']}")
        print(f"  IR: {row['IR']}  異常比例: {row['anomaly_ratio']:.2%}")
        print(f"  AUC_OCSVM: {row['AUC_OCSVM']}  AUC_LOF: {row['AUC_LOF']}  AUC_IF: {row['AUC_IF']}")
        print(f"  AUC_mean:  {row['AUC_mean']}")

    return pilots

# ============================================================
# 6. 主程式
# ============================================================

if __name__ == '__main__':
    KEEL_ROOT = './KEEL_Dataset'    # ← 對應你截圖中的資料夾名稱

    df_all, df_hard = run_all_keel(KEEL_ROOT, threshold=0.7)

    out_xlsx = 'occ_auc_results.xlsx'
    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
        df_all.to_excel(writer,  sheet_name='all',  index=False)
        df_hard.to_excel(writer, sheet_name='hard', index=False)
    print(f"\n結果已儲存至 {out_xlsx}（sheet: all / hard）")

    pilots = select_pilot_datasets(df_hard)