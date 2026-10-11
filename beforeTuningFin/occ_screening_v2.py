"""
occ_screening_v2.py — 困難資料集（HARD）篩選，一次輸出兩種設定以供對照

兩種設定
  neutral : StandardScaler + sklearn 預設參數（= 原本 occ_screening.py 的設定）
            OCSVM(nu=0.5, gamma=scale) / LOF(k=20) / IF(n_estimators=100)
  aligned : MinMaxScaler + baseline A 的固定參數（= occ_tune_core 的 LEGACY_DEFAULT）
            OCSVM(nu=0.1, gamma=scale) / LOF(k=min(20, n-1)) / IF(100, max_samples=256)

兩種設定的共同點（都與 A~N 對齊）
  - scaler 只 fit 在「訓練集的多數類（normal）」上，test 只做 transform
  - 只用訓練多數類訓練 OCC
  - 分數一律 -score_samples（越高越異常），AUC 只看排序
  - normal/anomaly 的定義由 train fold 決定，test fold 沿用，避免 train/test 各自判定造成標籤反轉

HARD 規則：三種 OCC 中至少 2 種的 5-fold 平均 AUC < 0.7

輸出：results/occ_auc_results_v2.xlsx
  all_neutral / hard_neutral / all_aligned / hard_aligned / compare / config
"""
import os
import io
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

KEEL_ROOT = './KEEL_Dataset'
THRESHOLD = 0.7
OUT_XLSX  = 'results/occ_auc_results_v2.xlsx'

# 重複的資料夾（同一個資料集的另一種編碼版本），避免同一個資料集被算兩次
EXCLUDE_DIRS = {'abalone19-5-fold_Encoding', 'abalone9-18-5-fold_Encoding'}

# 兩種篩選設定
CONFIGS = {
    'neutral': {
        'scaler': 'standard',
        'ocsvm':  {'nu': 0.5, 'gamma': 'scale'},      # sklearn 預設
        'lof':    {'n_neighbors': 20},
        'iforest': {'n_estimators': 100, 'max_samples': 'auto'},
    },
    'aligned': {
        'scaler': 'minmax',
        'ocsvm':  {'nu': 0.1, 'gamma': 'scale'},      # baseline A 固定參數
        'lof':    {'n_neighbors': 20},
        'iforest': {'n_estimators': 100, 'max_samples': 256},
    },
}


# ============================================================
# 1. 載入單一 .dat 檔
# ============================================================
def load_dat_file(filepath, normal_class=None):
    """載入 KEEL .dat。

    回傳 X, y(0=normal, 1=anomaly), normal_class, n_cat

    normal_class：train fold 傳 None（自行以多數類判定），
    test fold 請傳入 train 判定的結果，避免兩邊各自判定造成標籤反轉。
    """
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    filtered = [l for l in lines
                if not l.strip().lower().startswith(('@inputs', '@outputs', '@input', '@output'))]
    data, meta = arff.loadarff(io.StringIO(''.join(filtered)))
    df = pd.DataFrame(data)
    for col in df.select_dtypes([object]):
        df[col] = df[col].str.decode('utf-8')

    label_col = df.columns[-1]
    if normal_class is None:
        normal_class = df[label_col].value_counts().idxmax()
    y = (df[label_col] != normal_class).astype(int).values

    X_df = df.drop(columns=[label_col])
    n_cat = 0
    for col in X_df.select_dtypes(['object']):
        n_cat += 1
        X_df[col] = pd.Categorical(X_df[col]).codes
    X = X_df.astype(float).values
    return X, y, normal_class, n_cat


# ============================================================
# 2. 掃描資料集結構
# ============================================================
def scan_keel_folds(root_dir):
    """回傳 dict { dataset_name: [(tra_path, tst_path), ...] }"""
    dataset_folds = {}
    for ds_name in sorted(os.listdir(root_dir)):
        if ds_name in EXCLUDE_DIRS:
            print(f"  (略過重複資料夾: {ds_name})")
            continue
        ds_path = os.path.join(root_dir, ds_name)
        if not os.path.isdir(ds_path):
            continue
        all_dats = [os.path.join(ds_path, f) for f in os.listdir(ds_path)
                    if f.endswith('.dat') and os.path.isfile(os.path.join(ds_path, f))]
        tra_files = sorted([p for p in all_dats if 'tra.dat' in os.path.basename(p)])
        tst_files = sorted([p for p in all_dats if 'tst.dat' in os.path.basename(p)])
        pairs = list(zip(tra_files, tst_files))
        if pairs:
            dataset_folds[ds_name] = pairs
    return dataset_folds


# ============================================================
# 3. 建立 OCC（依設定）
# ============================================================
def build_occ(occ_name, cfg, n_train):
    """n_train = 訓練多數類樣本數，用於 LOF 的 k 與 iForest 的 max_samples 上限。"""
    if occ_name == 'OCSVM':
        p = cfg['ocsvm']
        return OneClassSVM(kernel='rbf', nu=p['nu'], gamma=p['gamma'])
    if occ_name == 'LOF':
        k = int(min(cfg['lof']['n_neighbors'], n_train - 1))
        return LocalOutlierFactor(n_neighbors=max(1, k), novelty=True)
    p = cfg['iforest']
    ms = p['max_samples']
    if ms != 'auto':
        ms = int(min(ms, n_train))
    return IsolationForest(n_estimators=p['n_estimators'], max_samples=ms, random_state=42)


# ============================================================
# 4. 單一資料集 5-fold 評估
# ============================================================
def evaluate_occ_kfold(fold_pairs, cfg):
    auc_records = {'OCSVM': [], 'LOF': [], 'IF': []}

    for tra_path, tst_path in fold_pairs:
        try:
            X_tra, y_tra, normal_class, _ = load_dat_file(tra_path)
            X_tst, y_tst, _, _ = load_dat_file(tst_path, normal_class=normal_class)
        except Exception as e:
            print(f"    載入失敗: {e}")
            continue

        if y_tst.sum() < 1 or len(np.unique(y_tst)) < 2:
            continue

        X_train_normal = X_tra[y_tra == 0]
        if len(X_train_normal) < 5:
            continue

        scaler = StandardScaler() if cfg['scaler'] == 'standard' else MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train_normal)   # 只 fit 訓練多數類
        X_tst_scaled   = scaler.transform(X_tst)

        for occ_name in ('OCSVM', 'LOF', 'IF'):
            try:
                clf = build_occ(occ_name, cfg, len(X_train_scaled))
                clf.fit(X_train_scaled)
                scores = -clf.score_samples(X_tst_scaled)       # 越高越異常
                auc_records[occ_name].append(roc_auc_score(y_tst, scores))
            except Exception as e:
                print(f"    {occ_name} 失敗: {e}")

    return {k: round(float(np.mean(v)), 4) if v else np.nan
            for k, v in auc_records.items()}, \
           {k: len(v) for k, v in auc_records.items()}


# ============================================================
# 5. 跑全部資料集（單一設定）
# ============================================================
def run_all_keel(dataset_folds, cfg, cfg_name, threshold=THRESHOLD):
    print(f"\n{'='*60}\n設定 {cfg_name}: scaler={cfg['scaler']}, "
          f"OCSVM nu={cfg['ocsvm']['nu']}, LOF k={cfg['lof']['n_neighbors']}, "
          f"IF max_samples={cfg['iforest']['max_samples']}\n{'='*60}")

    records = []
    for ds_name, fold_pairs in dataset_folds.items():
        print(f"處理: {ds_name} ({len(fold_pairs)} folds) ...", end=' ', flush=True)
        try:
            X_sample, _, _, n_cat = load_dat_file(fold_pairs[0][0])
            n_features = X_sample.shape[1]
            all_y = np.concatenate([load_dat_file(tp)[1] for tp, _ in fold_pairs])
            anomaly_ratio = round(float(all_y.mean()), 4)
            n_samples = len(all_y)          # 注意：所有 fold 的 tra 合計，非原始樣本數
            n_anomaly = int(all_y.sum())
        except Exception as e:
            print(f"!! meta 載入失敗: {e}")
            continue

        aucs, n_folds = evaluate_occ_kfold(fold_pairs, cfg)
        n_normal = n_samples - n_anomaly
        ir = round(n_normal / n_anomaly, 4) if n_anomaly > 0 else np.nan

        records.append({
            'dataset': ds_name,
            'n_samples_all_folds': n_samples,
            'n_features': n_features,
            'n_categorical': n_cat,          # >0 表示含名目屬性（train/test 編碼可能不一致）
            'n_anomaly': n_anomaly,
            'anomaly_ratio': anomaly_ratio,
            'IR': ir,
            'AUC_OCSVM': aucs['OCSVM'],
            'AUC_LOF': aucs['LOF'],
            'AUC_IF': aucs['IF'],
            'AUC_mean': round(float(np.nanmean(list(aucs.values()))), 4),
            'valid_folds': min(n_folds.values()),
        })
        print(f"OCSVM={aucs['OCSVM']}  LOF={aucs['LOF']}  IF={aucs['IF']}")

    df_all = pd.DataFrame(records)
    if df_all.empty:
        print("!! 沒有成功評估任何資料集，請確認 KEEL_ROOT 路徑。")
        return df_all, pd.DataFrame()

    auc_cols = ['AUC_OCSVM', 'AUC_LOF', 'AUC_IF']
    df_all['n_below'] = (df_all[auc_cols] < threshold).sum(axis=1)
    df_all['is_HARD'] = df_all['n_below'] >= 2
    df_hard = df_all[df_all['is_HARD']].copy()

    print(f"\n=== [{cfg_name}] 至少 2 種 OCC AUC < {threshold}：{len(df_hard)} 個 ===")
    print(df_hard[['dataset', 'IR', 'AUC_OCSVM', 'AUC_LOF', 'AUC_IF', 'n_below']]
          .to_string(index=False))
    return df_all, df_hard


# ============================================================
# 6. 兩種設定的名單對照
# ============================================================
def compare_hard(res):
    names = list(res.keys())
    all_ds = sorted(set().union(*[set(res[n][0]['dataset']) for n in names]))
    rows = []
    for ds in all_ds:
        row = {'dataset': ds}
        flags = []
        for n in names:
            d = res[n][0]
            r = d[d.dataset == ds]
            hard = bool(r['is_HARD'].iloc[0]) if len(r) else False
            flags.append(hard)
            row[f'{n}_OCSVM'] = r['AUC_OCSVM'].iloc[0] if len(r) else np.nan
            row[f'{n}_LOF']   = r['AUC_LOF'].iloc[0] if len(r) else np.nan
            row[f'{n}_IF']    = r['AUC_IF'].iloc[0] if len(r) else np.nan
            row[f'{n}_n_below'] = r['n_below'].iloc[0] if len(r) else np.nan
            row[f'{n}_HARD'] = hard
        row['一致'] = '一致' if len(set(flags)) == 1 else '★不一致'
        rows.append(row)
    df = pd.DataFrame(rows)

    diff = df[df['一致'] == '★不一致']
    print(f"\n{'='*60}\n兩種設定判定不同的資料集：{len(diff)} 個\n{'='*60}")
    if len(diff):
        print(diff[['dataset'] + [c for c in df.columns
                                  if c.endswith(('_HARD', '_n_below'))]].to_string(index=False))
    for n in names:
        hard = df[df[f'{n}_HARD']]['dataset'].tolist()
        print(f"\n[{n}] HARD {len(hard)} 個：{hard}")
    return df


# ============================================================
# 7. 主程式
# ============================================================
if __name__ == '__main__':
    dataset_folds = scan_keel_folds(KEEL_ROOT)
    print(f"找到 {len(dataset_folds)} 個資料集（已排除重複資料夾）")

    res = {name: run_all_keel(dataset_folds, cfg, name) for name, cfg in CONFIGS.items()}
    df_cmp = compare_hard(res)

    cfg_rows = [{'設定': n,
                 'scaler': c['scaler'],
                 'OCSVM': f"nu={c['ocsvm']['nu']}, gamma={c['ocsvm']['gamma']}, kernel=rbf",
                 'LOF': f"n_neighbors=min({c['lof']['n_neighbors']}, n_train-1), novelty=True",
                 'iForest': f"n_estimators={c['iforest']['n_estimators']}, "
                            f"max_samples={c['iforest']['max_samples']}, random_state=42",
                 '共同設定': 'scaler 只 fit 訓練多數類；只用訓練多數類訓練；'
                             '分數 = -score_samples；test 標籤沿用 train 判定的 normal class',
                 'HARD 規則': f'3 種 OCC 中至少 2 種 5-fold 平均 AUC < {THRESHOLD}'}
                for n, c in CONFIGS.items()]

    os.makedirs('results', exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine='openpyxl') as w:
        for n, (df_all, df_hard) in res.items():
            df_all.to_excel(w, sheet_name=f'all_{n}', index=False)
            df_hard.to_excel(w, sheet_name=f'hard_{n}', index=False)
        df_cmp.to_excel(w, sheet_name='compare', index=False)
        pd.DataFrame(cfg_rows).to_excel(w, sheet_name='config', index=False)
    print(f"\n結果已儲存至 {OUT_XLSX}")
