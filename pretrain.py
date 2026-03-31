"""
前測主程式 v3
架構搜尋設計：
  - 層數：h2、h3（KEEL 小資料集，h4+ 易 overfit）
  - 瓶頸維度：input_dim × [1/4, 1/3, 1/2, 1/1]（動態，依各資料集決定）
    * 1/1 = 無壓縮版本，作為對照
  - 組合：2 層 × 4 比例 = 8 種架構 × 4 AE × 3 OCC = 96 組/資料集
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────
# 設定
# ─────────────────────────────────────────
DATA_DIR    = "./preTraData"
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

HIDDEN_LAYERS      = [2, 3]
BOTTLENECK_RATIOS  = [1/4, 1/3, 1/2, 1/1]   # 相對於 input_dim 的比例
RATIO_LABELS       = ["1/4", "1/3", "1/2", "1/1"]

AE_TYPES  = ["AE", "DAE", "SAE", "VAE"]
OCC_TYPES = ["OCSVM", "LOF", "iForest"]


def get_ae_configs(input_dim):
    """
    依 input_dim 動態產生 AE 架構清單。
    回傳 list of (n_layers, n_units, label)
    n_units 至少為 2，避免瓶頸層太小導致訓練不穩。
    """
    configs = []
    for n_layers in HIDDEN_LAYERS:
        for ratio, label in zip(BOTTLENECK_RATIOS, RATIO_LABELS):
            n_units = max(2, round(input_dim * ratio))
            configs.append((n_layers, n_units, f"h{n_layers}-{label}"))
    return configs

AE_EPOCHS     = 100
AE_BATCH_SIZE = 64
AE_LR         = 1e-3
DAE_NOISE     = 0.1
SAE_SPARSITY  = 1e-3
VAE_BETA      = 1.0


# ─────────────────────────────────────────
# AE 模型定義
# ─────────────────────────────────────────
class AEModel(nn.Module):
    def __init__(self, input_dim, n_layers, n_units):
        super().__init__()
        dims = [input_dim] + [n_units] * n_layers
        enc, dec = [], []
        for i in range(len(dims) - 1):
            enc += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        for i in range(len(dims) - 1):
            d_in, d_out = dims[-(i+1)], dims[-(i+2)]
            act = nn.Sigmoid() if i == len(dims) - 2 else nn.ReLU()
            dec += [nn.Linear(d_in, d_out), act]
        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class VAEModel(nn.Module):
    def __init__(self, input_dim, n_layers, n_units):
        super().__init__()
        dims = [input_dim] + [n_units] * n_layers
        base = []
        for i in range(len(dims) - 2):
            base += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        self.enc_base  = nn.Sequential(*base) if base else nn.Identity()
        mid = dims[-2] if len(dims) >= 2 else input_dim
        self.fc_mu     = nn.Linear(mid, dims[-1])
        self.fc_logvar = nn.Linear(mid, dims[-1])
        dec = []
        dec_dims = dims[::-1]
        for i in range(len(dec_dims) - 1):
            act = nn.Sigmoid() if i == len(dec_dims) - 2 else nn.ReLU()
            dec += [nn.Linear(dec_dims[i], dec_dims[i+1]), act]
        self.decoder = nn.Sequential(*dec)

    def reparameterize(self, mu, lv):
        return mu + torch.exp(0.5 * lv) * torch.randn_like(lv)

    def forward(self, x):
        h  = self.enc_base(x)
        mu = self.fc_mu(h)
        lv = self.fc_logvar(h)
        z  = self.reparameterize(mu, lv)
        return self.decoder(z), z, mu, lv


# ─────────────────────────────────────────
# AE 訓練 + 特徵提取
# ─────────────────────────────────────────
def train_and_extract(ae_type, X_maj, X_all, n_layers, n_units):
    """
    用 majority 資料訓練 AE，再對 X_all 提取深度特徵
    回傳 (feat_maj, feat_all)
    """
    input_dim = X_maj.shape[1]
    model = VAEModel(input_dim, n_layers, n_units) if ae_type == "VAE" \
            else AEModel(input_dim, n_layers, n_units)

    optim  = torch.optim.Adam(model.parameters(), lr=AE_LR)
    mse    = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(X_maj, dtype=torch.float32)),
        batch_size=AE_BATCH_SIZE, shuffle=True
    )

    for _ in range(AE_EPOCHS):
        model.train()
        for (xb,) in loader:
            optim.zero_grad()
            if ae_type == "DAE":
                xb_n = torch.clamp(xb + DAE_NOISE * torch.randn_like(xb), 0, 1)
                xr, _ = model(xb_n)
                loss   = mse(xr, xb)
            elif ae_type == "SAE":
                xr, z = model(xb)
                loss   = mse(xr, xb) + SAE_SPARSITY * z.abs().mean()
            elif ae_type == "VAE":
                xr, _, mu, lv = model(xb)
                kl   = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean()
                loss = mse(xr, xb) + VAE_BETA * kl
            else:
                xr, _ = model(xb)
                loss   = mse(xr, xb)
            loss.backward()
            optim.step()

    # 提取特徵
    model.eval()
    def extract(X):
        xt = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            if ae_type == "VAE":
                _, _, mu, _ = model(xt)
                return mu.numpy()
            else:
                _, z = model(xt)
                return z.numpy()

    return extract(X_maj), extract(X_all)


# ─────────────────────────────────────────
# OCC 評估
# ─────────────────────────────────────────
def run_occ(occ_type, feat_maj, feat_all, y):
    scaler    = MinMaxScaler()
    feat_maj_s = scaler.fit_transform(feat_maj)
    feat_all_s = scaler.transform(feat_all)

    if occ_type == "OCSVM":
        clf = OneClassSVM(nu=0.1, kernel="rbf")
        clf.fit(feat_maj_s)
        scores = -clf.decision_function(feat_all_s)
    elif occ_type == "LOF":
        clf = LocalOutlierFactor(n_neighbors=min(20, len(feat_maj_s)-1), novelty=True)
        clf.fit(feat_maj_s)
        scores = -clf.decision_function(feat_all_s)
    else:  # iForest
        clf = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
        clf.fit(feat_maj_s)
        scores = -clf.decision_function(feat_all_s)

    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, scores)


# ─────────────────────────────────────────
# 讀取 KEEL .dat
# ─────────────────────────────────────────
def load_keel_dat(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    attr_names, data_start = [], 0
    for i, line in enumerate(lines):
        l = line.strip()
        if l.lower().startswith("@attribute"):
            parts = l.split()
            attr_names.append(parts[1])
        if l.lower() == "@data":
            data_start = i + 1
            break

    rows = [l.strip().split(",") for l in lines[data_start:]
            if l.strip() and not l.startswith("%")]
    df = pd.DataFrame(rows, columns=attr_names)
    class_col = attr_names[-1]
    y_raw = df[class_col].str.strip().values
    X = df.iloc[:, :-1].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)

    unique, counts = np.unique(y_raw, return_counts=True)
    minority_label = unique[np.argmin(counts)]
    y = (y_raw == minority_label).astype(int)
    return X, y


def scan_pretrain_folds(root_dir):
    """
    掃描 preTraData/ 下每個子資料夾，
    找出所有 *tra.dat / *tst.dat 配對（只看直接層檔案，不進子資料夾）。
    回傳 dict { dataset_name: [(tra_path, tst_path), ...] }
    """
    dataset_folds = {}
    for ds_name in sorted(os.listdir(root_dir)):
        ds_path = os.path.join(root_dir, ds_name)
        if not os.path.isdir(ds_path):
            continue
        files = [f for f in os.listdir(ds_path)
                 if f.endswith('.dat') and os.path.isfile(os.path.join(ds_path, f))]
        tra = sorted([os.path.join(ds_path, f) for f in files if 'tra.dat' in f])
        tst = sorted([os.path.join(ds_path, f) for f in files if 'tst.dat' in f])
        pairs = list(zip(tra, tst))
        if pairs:
            dataset_folds[ds_name] = pairs
    return dataset_folds


def compute_ir(y):
    n_min = y.sum()
    n_maj = len(y) - n_min
    return n_maj / n_min if n_min > 0 else np.inf


def baseline_a_auc(X, y):
    X_maj = X[y == 0]
    sc = MinMaxScaler()
    X_maj_s = sc.fit_transform(X_maj)
    X_all_s = sc.transform(X)
    clf = OneClassSVM(nu=0.1, kernel="rbf")
    clf.fit(X_maj_s)
    scores = -clf.decision_function(X_all_s)
    if len(np.unique(y)) < 2:
        return 1.0
    return roc_auc_score(y, scores)


# ─────────────────────────────────────────
# 篩選 + 代表性資料集
# ─────────────────────────────────────────
def select_datasets(data_dir, auc_threshold=0.7):
    dat_files = glob.glob(os.path.join(data_dir, "*.dat"))
    if not dat_files:
        print(f"[警告] 找不到 .dat 檔，請確認 {data_dir}/ 目錄")
        return []

    print(f"共 {len(dat_files)} 個資料集，篩選 Baseline-A AUC < {auc_threshold} ...\n")
    selected = []
    for fp in sorted(dat_files):
        name = Path(fp).stem
        try:
            X, y = load_keel_dat(fp)
            if X.shape[0] < 20 or len(np.unique(y)) < 2:
                continue
            auc = baseline_a_auc(X, y)
            ir  = compute_ir(y)
            mark = "✓" if auc < auc_threshold else " "
            print(f"  [{mark}] {name:40s}  AUC={auc:.3f}  IR={ir:6.1f}")
            if auc < auc_threshold:
                selected.append((name, fp, X, y, ir))
        except Exception as e:
            print(f"  [跳過] {name}: {e}")

    print(f"\n篩選後：{len(selected)} 個資料集")
    pd.DataFrame([{"name": s[0], "IR": round(s[4],2)} for s in selected]) \
      .to_csv(os.path.join(RESULTS_DIR, "selected_datasets.csv"), index=False)
    return selected


def pick_representative(selected):
    if not selected:
        return []
    irs = np.array([s[4] for s in selected])
    med = np.median(irs)
    low  = [s for s in selected if s[4] <= med]
    high = [s for s in selected if s[4] >  med]

    def closest_to_median(grp):
        if not grp: return None
        g_irs = np.array([s[4] for s in grp])
        return grp[np.argmin(np.abs(g_irs - np.median(g_irs)))]

    reps = [r for r in [closest_to_median(low), closest_to_median(high)] if r]
    print("\n代表性資料集：")
    for r in reps:
        print(f"  {r[0]}  IR={r[4]:.1f}  shape={r[2].shape}")
    return reps


# ─────────────────────────────────────────
# 前測主流程
# ─────────────────────────────────────────
def run_pretrain(dataset_folds):
    """
    dataset_folds: dict { ds_name: [(tra_path, tst_path), ...] }
    每個資料集做 5-fold：用 tra 的 normal 樣本訓練 AE，在 tst 上評估 OCC AUC。
    """
    all_rows = []

    for ds_name, fold_pairs in dataset_folds.items():
        # 用第一個 fold 的 tra 取得 input_dim 和 IR
        try:
            X0, _ = load_keel_dat(fold_pairs[0][0])
            input_dim = X0.shape[1]
            all_y = np.concatenate([load_keel_dat(tp)[1] for tp, _ in fold_pairs])
            ir = compute_ir(all_y)
        except Exception as e:
            print(f"\n[跳過] {ds_name} meta 載入失敗: {e}")
            continue

        ae_configs = get_ae_configs(input_dim)
        print(f"\n{'='*60}")
        print(f"資料集：{ds_name}  folds={len(fold_pairs)}  IR={ir:.1f}  input_dim={input_dim}")
        print(f"{'='*60}")

        for ae_type in AE_TYPES:
            for (n_layers, n_units, cfg_label) in ae_configs:
                print(f"  [{ae_type}] {cfg_label}(units={n_units}) ", end="", flush=True)
                fold_aucs = {occ: [] for occ in OCC_TYPES}

                for tra_path, tst_path in fold_pairs:
                    try:
                        X_tra, y_tra = load_keel_dat(tra_path)
                        X_tst, y_tst = load_keel_dat(tst_path)
                        if y_tst.sum() < 1:
                            continue

                        sc = MinMaxScaler()
                        X_tra_s = sc.fit_transform(X_tra)
                        X_tst_s = sc.transform(X_tst)
                        X_maj   = X_tra_s[y_tra == 0]
                        if len(X_maj) < 5:
                            continue

                        feat_maj, feat_tst = train_and_extract(
                            ae_type, X_maj, X_tst_s, n_layers, n_units)

                        for occ in OCC_TYPES:
                            auc = run_occ(occ, feat_maj, feat_tst, y_tst)
                            if not np.isnan(auc):
                                fold_aucs[occ].append(auc)
                    except Exception as e:
                        print(f"[fold err: {e}] ", end="", flush=True)

                row = {"dataset": ds_name, "input_dim": input_dim,
                       "IR": round(ir, 2), "AE_type": ae_type,
                       "config": cfg_label, "n_units": n_units}
                for occ in OCC_TYPES:
                    val = round(np.mean(fold_aucs[occ]), 3) if fold_aucs[occ] else "-"
                    row[occ] = val
                    print(f"{occ}={val} ", end="", flush=True)
                print()
                all_rows.append(row)

    df = pd.DataFrame(all_rows)
    out_path = os.path.join(RESULTS_DIR, "pretrain_results.xlsx")
    df.to_excel(out_path, index=False, engine="openpyxl")
    print(f"\n✅ 結果儲存：{out_path}")
    return df


# ─────────────────────────────────────────
# 輸出最佳參數 & 圖二格式表
# ─────────────────────────────────────────
def summarize_best(df):
    print("\n" + "="*60 + "\n最佳參數總結\n" + "="*60)
    rows = []
    for ae in AE_TYPES:
        sub = df[df["AE_type"] == ae].copy()
        for occ in OCC_TYPES:
            sub[occ] = pd.to_numeric(sub[occ], errors="coerce")
            best_cfg = sub.groupby("config")[occ].mean().idxmax()
            best_val = sub.groupby("config")[occ].mean().max()
            print(f"  {ae:5s} + {occ:8s} => {best_cfg}  avg AUC={best_val:.3f}")
            rows.append({"AE": ae, "OCC": occ, "best_config": best_cfg, "avg_AUC": round(best_val,3)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS_DIR, "best_params.csv"), index=False, encoding="utf-8-sig")
    print(f"\n最佳參數 => {RESULTS_DIR}/best_params.csv")
    return out


def make_fig2_table(df, tag="avg"):
    # 從資料本身取得 config 順序（h2-1/4, h2-1/3 ... h3-1/1）
    configs_order = []
    for l in HIDDEN_LAYERS:
        for r in RATIO_LABELS:
            configs_order.append(f"h{l}-{r}")

    cols = pd.MultiIndex.from_tuples([(ae, occ) for ae in AE_TYPES for occ in OCC_TYPES])
    out  = pd.DataFrame(index=configs_order, columns=cols, dtype=object)

    for ae in AE_TYPES:
        sub = df[df["AE_type"] == ae].copy()
        for occ in OCC_TYPES:
            sub[occ] = pd.to_numeric(sub[occ], errors="coerce")
            means = sub.groupby("config")[occ].mean()
            for cfg in configs_order:
                val = means.get(cfg, np.nan)
                out.loc[cfg, (ae, occ)] = round(val, 3) if not pd.isna(val) else "-"

    path = os.path.join(RESULTS_DIR, f"table_fig2_{tag}.csv")
    out.to_csv(path, encoding="utf-8-sig")
    print(f"圖二格式表 => {path}")
    return out


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("【步驟 1】掃描 preTraData 資料集")
    dataset_folds = scan_pretrain_folds(DATA_DIR)
    if not dataset_folds:
        print(f"\n請確認 {DATA_DIR}/ 下有含 tra.dat/tst.dat 的子資料夾。")
        exit(0)
    print(f"找到 {len(dataset_folds)} 個資料集：{list(dataset_folds.keys())}")

    print("\n【步驟 2】AE 架構前測 × OCC 前測（5-fold）")
    results_df = run_pretrain(dataset_folds)

    print("\n【步驟 3】最佳參數")
    summarize_best(results_df)

    print("\n【步驟 4】圖二格式表")
    for ds_name in dataset_folds:
        sub = results_df[results_df["dataset"] == ds_name]
        if not sub.empty:
            make_fig2_table(sub, tag=ds_name)
    make_fig2_table(results_df, tag="average")

    print("\n✅ 前測全部完成！請查看 ./results/ 資料夾。")
