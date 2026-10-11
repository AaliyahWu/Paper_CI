"""
M_study2_fw_then_us_DF_final_v1.py
==================================
實驗二（Study Two）整合：Feature Weighting × Under-sampling，順序固定為
        ── FW 先 → US 後 ──（在 J 的 DF 端清理框架上）

== 這支檔案在做什麼 ==
    把 H（DF 端 feature weighting）的 FW 維度，與 J（DF 端 under-sampling）的
    Sampler 維度疊在一起，並明確固定「先加權、再清理」的順序。
    研究問題：在 J（DF 側 US，最佳 = VAE+ENN+LOF, AUC≈0.7066）的基礎上，
              先做 FW 再做 US，是否帶來顯著幫助？

== 與 J 的唯一結構差異 ==
    J ：MinMax(fit 未清理 DF_maj) → US（在【未加權】DF 空間清理）→ OCC
    M ：MinMax(fit 未清理 DF_maj) → FW（套到 maj/min/test）
                                  → US（在【已加權】DF_w 空間清理）→ OCC
    重點：ENN/CNN/TL 都是距離（kNN）型 sampler。先 FW 會改變 sampler 看到的
          幾何 → 哪些 majority 被判為「異常鄰域中的雜訊」會與 J 不同。
          這就是「FW→US」這個 ordering 想測的效應本身。

== Pipeline（OCC 永遠在最後）==
    OF (train: maj+min)
        ──MinMax(fit on X_maj)──> X_maj_s / X_min_s / X_tst_s
        ──AE 訓練(只用 X_maj_s)──> 抽 DF_maj / DF_min / DF_tst
        ──MinMax(fit on 未清理 DF_maj)──> DF_maj_s / DF_min_s / DF_tst_s   ← 座標固定一次
        ──FW(在 DF_maj_s 算 w、normalize max=1、套到三者)──> DF_maj_w / DF_min_w / DF_tst_w
        ──US(在【加權後】DF_w 空間，只刪 DF_maj_w，min 當參考點)──> DF_maj_clean_w
        ──OCC(LOF；do_scale=False，不再二次 MinMax)──> prediction

== 為什麼這個設計能直接回答「FW→US 有沒有幫助」==
    自然形成 FW × Sampler 二維表，四角落都是已知 baseline，可互相驗證：
        FW=none, Sampler=none  → 應重現 baseline B（VAE+LOF）
        FW=none, Sampler=ENN   → 應重現 J（VAE+ENN+LOF；你的 0.7066 在此格）
        FW=var , Sampler=none  → 應重現 H（VAE+var+LOF）
        FW≠none, Sampler≠none  → ★ M 的新貢獻：FW→US 組合格 ★
    判斷增益：固定看某個 Sampler 欄（如 ENN），比較不同 FW 列的 AUC，
              差值即為「先 FW」帶來的幫助。三角落能重現舊結果，
              同時驗證 RNG 對齊與「座標固定一次」沒寫壞。

== 「座標固定一次」與無 leakage ==
    • DF 的 MinMax 只用【未清理】DF_maj 來 fit；FW 的 weights 只在
      DF_maj_s（train majority）上計算。兩者都不碰 test，無 leakage。
    • FW=none → normalize(ones)=ones → DF_w = DF_s → US 退化回 J；
      再加 Sampler=none → 退化回 baseline B。故控制組精準可重現。

== 這次的精簡（pilot）==
    • 只跑一個資料集：見下方 RUN_ONLY_DATASETS。
    • 沿用 H 的聚焦：AE 固定 VAE、OCC 固定 LOF（你的最佳組合都落在這）。
      要擴成 4 AE / 3 OCC 只要改 AE_TYPES / OCC_TYPES 兩行。

搜尋空間（單一資料集）：1 AE × 5 FW × 4 Sampler × 1 OCC × 21 config = 420 組 / fold。
VAE 訓練數 = 21 config × 5 fold = 105 次（與 H 在單一資料集相同，sampler 只是刪列，便宜）。

輸出：results/M_study2_fw_then_us_DF.xlsx
      • all_per_fold     ：每 (FW, Sampler, OCC, Config, Fold) 一列
      • best_grid        ：★核心比較表★ FW(列) × Sampler(欄) 的 best-config 平均
      • ak_all_export    ：A~K 統整用 flat 表（含 Order / FW / Sampler 欄）
      • ak_best_export   ：同上，best-config 版
"""

import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor, kneighbors_graph
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, f1_score, recall_score, confusion_matrix

# imbalanced-learn（需 pip install imbalanced-learn）
from imblearn.under_sampling import (
    EditedNearestNeighbours, CondensedNearestNeighbour, TomekLinks,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

# ─────────────────────────── 路徑設定 ────────────────────────────────────────
DATA_ROOT   = Path("data")
RESULTS_DIR = Path("results")
OUTPUT_FILE = RESULTS_DIR / "M_study2_fw_then_us_DF.xlsx"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5
EPS     = 1e-12

# ── ★單一資料集 pilot 控制★ ──
#   None      → 取 DATA_ROOT 下排序後的「第一個」資料夾（最省事）
#   ["名稱"]  → 只跑指定資料夾名稱（建議填你 J 跑出最佳的那個資料集）
#   例：RUN_ONLY_DATASETS = ["yeast3"]
RUN_ONLY_DATASETS = ["vehicle2-5-fold"]

# ── AE 訓練超參數（與 B/C/H/J 完全一致）──
AE_EPOCHS     = 100
AE_BATCH_SIZE = 64
AE_LR         = 1e-3
DAE_NOISE     = 0.1
SAE_SPARSITY  = 1e-3
VAE_BETA      = 1.0

# ── 聚焦策略（沿用 H）：AE 固定 VAE、OCC 固定 LOF（最佳組合都在此）──
#   要做完整 ablation 再改回：AE_TYPES=["AE","DAE","SAE","VAE"], OCC_TYPES=["OCSVM","LOF","iForest"]
AE_TYPES    = ["VAE"]
OCC_TYPES   = ["LOF"]
METRIC_COLS = ["AUC", "F1", "Recall", "G-mean"]

# ── Feature Weighting 方法（與 H 一致；含 none 內部 baseline）──
FW_METHODS  = ["none", "var", "ivar", "mad", "lap"]
LAPLACIAN_K = 5

# ── Under-sampling 方法（與 J 一致；含 none 內部 baseline）──
SAMPLERS  = ["none", "ENN", "CNN", "TL"]
ENN_K     = 3
CNN_K     = 1
CNN_SEED  = 42

# ── A~K 統整 metadata（不影響計算）──
STUDY_ID    = "M"
METHOD_ID   = "M_FW_then_US"
FEATURE_SET = "DF_maj"
ORDER       = "FW_then_US"          # ★ ordering 作為明確欄位，方便日後與 US→FW 對照 ★
BASELINE_REF = "B / H(FW) / J(US)"  # 三角落可重現的對照
OCC_SCOPE = "focused_VAE_LOF"       # 與 J 的 ak_export 欄位對齊；M pilot 目前只跑 VAE × LOF
SAMPLER_SCALE_MODE = "df_majority_minmax_once_before_FW_then_sampler_and_occ"

COMPARISON_EXPORT_COLS = [
    "Study", "Method", "Order", "FeatureSet", "Dataset", "AE", "FW", "Sampler", "OCC",
    "Config", "Fold", "ConfigPolicy", "MajKept", "MajRemoved", "RemovedRate",
    "SamplerStatus", "BaselineRef", "OCCScope", "SamplerScaleMode",
] + METRIC_COLS

# ── Grid（與 B/C/G/H/J 對齊）──
N_LAYERS_LIST     = [1, 2, 3]
BOTTLENECK_RATIOS = {
    "1/4": 0.25, "1/3": 1/3, "1/2": 0.5, "1/1": 1.0,
    "2/1": 2.0,  "3/1": 3.0, "4/1": 4.0,
}
ALL_CONFIGS = [f"h{nl}-{rl}" for nl in N_LAYERS_LIST for rl in BOTTLENECK_RATIOS]


def safe_removed_rate(n_removed, n_kept):
    denom = int(n_removed) + int(n_kept)
    return float(n_removed / denom) if denom > 0 else 0.0


# ─────────────────────────── Feature Weighting（與 H 完全一致）──────────────
def compute_feature_weights(X, method, k=LAPLACIAN_K, eps=EPS):
    """在 train majority 的 DF 上計算每個特徵的非負權重（公式與 H 一致）。"""
    n, d = X.shape

    if method == "none":
        return np.ones(d, dtype=float)
    elif method == "var":
        return np.clip(X.var(axis=0), 0, None)
    elif method == "ivar":
        return 1.0 / (X.var(axis=0) + eps)
    elif method == "mad":
        med = np.median(X, axis=0)
        return np.clip(np.median(np.abs(X - med), axis=0), 0, None)
    elif method == "lap":
        if n < 3:
            return np.ones(d, dtype=float)
        k_eff = max(1, min(k, n - 1))
        try:
            S = kneighbors_graph(X, n_neighbors=k_eff, mode="distance",
                                 include_self=False).toarray()
        except Exception:
            return np.ones(d, dtype=float)
        pos = S[S > 0]
        t = (np.median(pos) ** 2 + eps) if len(pos) > 0 else 1.0
        S = np.where(S > 0, np.exp(-(S ** 2) / t), 0.0)
        S = 0.5 * (S + S.T)
        D_diag = S.sum(axis=1)
        D = np.diag(D_diag)
        L = D - S
        w = np.zeros(d, dtype=float)
        D_sum = D_diag.sum() + eps
        for j in range(d):
            f = X[:, j]
            f_bar = (f * D_diag).sum() / D_sum
            f_tilde = f - f_bar
            num = float(f_tilde @ L @ f_tilde)
            den = float(f_tilde @ D @ f_tilde) + eps
            ls = num / den
            w[j] = 1.0 / (ls + eps)
        return np.clip(w, 0, None)
    else:
        raise ValueError(f"Unknown FW method: {method}")


def normalize_weights(w):
    """歸一化使 max(w)=1（與 H 一致）：最重要的 feature 保留 100%，其餘衰減。
    保證 weighted DF ∈ [0, DF_s] ⊆ [0,1]，維持 VAE sigmoid decoder 假設與 OCC scale 正常。"""
    w = np.asarray(w, dtype=float)
    w = np.clip(w, 0, None)
    m = w.max()
    if m <= EPS:
        return np.ones_like(w)
    return w / m


def apply_weights(DF_maj_s, DF_min_s, DF_tst_s, method):
    """在【已 MinMax-scaled、未清理】的 DF 上算 weights，並套到 maj/min/test 三者。

    為什麼三者都要套同一組 w：
      • maj/test 加權 → 影響最終 OCC 距離（與 H 相同效果）。
      • min 也要加權 → 因為接下來 US 在加權空間清理，sampler 用 min 當參考點，
        若 min 不加權，sampler 的距離判斷會在「不一致座標」上，FW→US 的語意就破了。
    weights 只用 DF_maj_s（train majority）計算 → 無 leakage。
    回傳：(DF_maj_w, DF_min_w, DF_tst_w, w_norm)
    """
    w_raw  = compute_feature_weights(DF_maj_s, method)
    w_norm = normalize_weights(w_raw)
    return DF_maj_s * w_norm, DF_min_s * w_norm, DF_tst_s * w_norm, w_norm


# ─────────────────────────── Under-sampling（在【加權後】DF 空間）────────────
def make_sampler(name):
    if name == "ENN":
        return EditedNearestNeighbours(
            n_neighbors=ENN_K, kind_sel="all", sampling_strategy="auto")
    if name == "CNN":
        return CondensedNearestNeighbour(
            n_neighbors=CNN_K, random_state=CNN_SEED, sampling_strategy="auto")
    if name == "TL":
        return TomekLinks(sampling_strategy="auto")
    return None


def undersample_df_weighted(DF_maj_w, DF_min_w, sampler_name):
    """在【已加權】DF 空間清理 majority（只刪 maj，min 當參考點，test 全程不碰）。

    與 J 的 undersample_df 唯一差別：輸入是加權後的 DF_w（FW→US 的核心）。
    回傳：(DF_maj_clean_w, n_removed, sampler_status)
    """
    if sampler_name == "none":
        return DF_maj_w, 0, "none_baseline"

    n_maj = len(DF_maj_w)
    DF_all_w = np.vstack([DF_maj_w, DF_min_w])
    y_all = np.array([0] * n_maj + [1] * len(DF_min_w))

    sampler = make_sampler(sampler_name)
    if sampler is None:
        return DF_maj_w, 0, "fallback_unknown_sampler"
    sampler.fit_resample(DF_all_w, y_all)
    idx = sampler.sample_indices_

    # 確認 minority 完整保留（sampling_strategy="auto" 只該動 majority）
    if int(np.sum(idx >= n_maj)) != len(DF_min_w):
        return DF_maj_w, 0, "fallback_minority_changed"

    keep_maj_local = idx[idx < n_maj]
    DF_maj_clean_w = DF_maj_w[keep_maj_local]
    n_removed = n_maj - len(DF_maj_clean_w)
    status = "ok_removed" if n_removed > 0 else "ok_no_removed"
    return DF_maj_clean_w, n_removed, status


# ─────────────────────────── AE 模型（與 B/C/J 完全一致）────────────────────
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
        dec_dims = dims[::-1]
        dec = []
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


def train_ae_and_get_extractor(ae_type, X_maj_s, n_layers, n_units):
    """訓練 AE（只用 majority），回傳 extract 函式（與 J 完全一致）。
    VAE 取 μ，其餘取 encoder z。"""
    input_dim = X_maj_s.shape[1]
    model = VAEModel(input_dim, n_layers, n_units) if ae_type == "VAE" \
            else AEModel(input_dim, n_layers, n_units)

    optim_ = torch.optim.Adam(model.parameters(), lr=AE_LR)
    mse    = nn.MSELoss()
    bs     = min(AE_BATCH_SIZE, len(X_maj_s))
    loader = DataLoader(
        TensorDataset(torch.tensor(X_maj_s, dtype=torch.float32)),
        batch_size=bs, shuffle=True,
    )

    for _ in range(AE_EPOCHS):
        model.train()
        for (xb,) in loader:
            optim_.zero_grad()
            if ae_type == "DAE":
                xb_n  = torch.clamp(xb + DAE_NOISE * torch.randn_like(xb), 0, 1)
                xr, _ = model(xb_n)
                loss  = mse(xr, xb)
            elif ae_type == "SAE":
                xr, z = model(xb)
                loss  = mse(xr, xb) + SAE_SPARSITY * z.abs().mean()
            elif ae_type == "VAE":
                xr, _, mu, lv = model(xb)
                kl   = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean()
                loss = mse(xr, xb) + VAE_BETA * kl
            else:
                xr, _ = model(xb)
                loss  = mse(xr, xb)
            loss.backward()
            optim_.step()

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
    return extract


# ─────────────────────────── 評估指標（與 B/J 一致）────────────────────────
def gmean_score(y_true, y_pred_binary):
    cm = confusion_matrix(y_true, y_pred_binary, labels=[1, 0])
    if cm.shape == (2, 2):
        tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return float(np.sqrt(sens * spec))
    return 0.0


def run_occ_eval(occ_type, feat_maj, feat_test, y_test, n_neighbors_cap, do_scale=False):
    """OCC 評估：M 一律 do_scale=False，因為輸入已是
    『MinMax-scaled → weighted →(可能)under-sampled』的 DF，再 MinMax 會 cancel FW。"""
    if do_scale:
        scaler    = MinMaxScaler()
        feat_maj  = scaler.fit_transform(feat_maj)
        feat_test = scaler.transform(feat_test)

    if occ_type == "OCSVM":
        clf = OneClassSVM(nu=0.1, kernel="rbf")
        clf.fit(feat_maj)
        scores_maj  = -clf.decision_function(feat_maj)
        scores_test = -clf.decision_function(feat_test)
    elif occ_type == "LOF":
        k = min(20, n_neighbors_cap)
        clf = LocalOutlierFactor(n_neighbors=k, novelty=True, contamination=0.1)
        clf.fit(feat_maj)
        scores_maj  = -clf.decision_function(feat_maj)
        scores_test = -clf.decision_function(feat_test)
    else:
        clf = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
        clf.fit(feat_maj)
        scores_maj  = -clf.decision_function(feat_maj)
        scores_test = -clf.decision_function(feat_test)

    threshold = np.percentile(scores_maj, 90)
    y_pred    = (scores_test >= threshold).astype(int)

    try:
        auc = roc_auc_score(y_test, scores_test) if len(np.unique(y_test)) >= 2 else float("nan")
    except Exception:
        auc = float("nan")

    f1  = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    gm  = gmean_score(y_test, y_pred)
    return {"AUC": auc, "F1": f1, "Recall": rec, "G-mean": gm}


# ─────────────────────────── KEEL .dat 解析（與 B/J 一致）────────────────────
def parse_keel_dat(filepath, minority_label=None):
    lines = Path(filepath).read_text(encoding="utf-8", errors="replace").splitlines()
    data_start = False
    rows = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        if s.lower() == "@data":
            data_start = True
            continue
        if data_start:
            rows.append(s)
    if not rows:
        raise ValueError(f"No data found in {filepath}")

    records = [[p.strip() for p in r.split(",")] for r in rows]
    df = pd.DataFrame(records)
    label_col = df.columns[-1]
    y_raw = df[label_col].astype(str).str.strip().values

    feat_df = df.iloc[:, :-1].copy()
    for col in feat_df.columns:
        conv = pd.to_numeric(feat_df[col], errors="coerce")
        if conv.isna().all():
            feat_df[col] = pd.Categorical(feat_df[col]).codes.astype(float)
        else:
            feat_df[col] = conv
    X = feat_df.values.astype(float)

    if minority_label is None:
        unique, counts = np.unique(y_raw, return_counts=True)
        minority_label = unique[np.argmin(counts)]

    y = (y_raw == minority_label).astype(int)
    return X, y, minority_label


# ─────────────────────────── 主流程 ──────────────────────────────────────────
def run_experiment():
    """
    執行順序：
      dataset → fold → [scale OF] → AE → config
              → [AE 訓練一次、抽 DF_maj/DF_min/DF_tst]
              → [MinMax fit 未清理 DF_maj] → FW → US(加權空間) → OCC

    為什麼 AE 在外、FW/US 在內：
      FW（elementwise 乘）與 US（刪列）都不影響 AE 的 input（未清理 majority），
      故每個 (AE, config) 的 AE 只訓練一次，FW × Sampler 共用同一份 DF → 省時（同 H/J）。
    """
    dataset_dirs = sorted([d for d in DATA_ROOT.iterdir() if d.is_dir()])
    if not dataset_dirs:
        raise FileNotFoundError(f"找不到任何資料夾於 {DATA_ROOT.resolve()}")

    # ── ★單一資料集 pilot 篩選★ ──
    if RUN_ONLY_DATASETS:
        dataset_dirs = [d for d in dataset_dirs if d.name in set(RUN_ONLY_DATASETS)]
        if not dataset_dirs:
            raise FileNotFoundError(
                f"RUN_ONLY_DATASETS={RUN_ONLY_DATASETS} 在 {DATA_ROOT.resolve()} 找不到對應資料夾")
    else:
        dataset_dirs = dataset_dirs[:1]   # 預設只取第一個
    print(f"▶ 本次將跑的資料集：{[d.name for d in dataset_dirs]}")

    param_configs = [(nl, rl) for nl in N_LAYERS_LIST for rl in BOTTLENECK_RATIOS]
    all_records   = []

    for ds_dir in dataset_dirs:
        ds_name = ds_dir.name
        print(f"\n{'='*68}\n▶ Dataset: {ds_name}")

        for fold in range(1, N_FOLDS + 1):
            file_prefix = re.sub(r'-fold.*$', '', ds_name)
            patterns_tra = [
                ds_dir / f"{file_prefix}-{fold}tra.dat",
                ds_dir / f"{ds_name}-5-fold-tra{fold}.dat",
                ds_dir / f"{ds_name}-5-tra{fold}.dat",
                ds_dir / f"{ds_name}_fold{fold}_train.dat",
            ]
            patterns_tst = [
                ds_dir / f"{file_prefix}-{fold}tst.dat",
                ds_dir / f"{ds_name}-5-fold-tst{fold}.dat",
                ds_dir / f"{ds_name}-5-tst{fold}.dat",
                ds_dir / f"{ds_name}_fold{fold}_test.dat",
            ]
            tra_file = next((p for p in patterns_tra if p.exists()), None)
            tst_file = next((p for p in patterns_tst if p.exists()), None)
            if tra_file is None or tst_file is None:
                print(f"  [SKIP] Fold {fold}: 找不到檔案")
                continue

            try:
                X_tra, y_tra, minority_label = parse_keel_dat(tra_file)
                X_tst, y_tst, _ = parse_keel_dat(tst_file, minority_label=minority_label)
                input_dim = X_tra.shape[1]
                X_maj = X_tra[y_tra == 0]
                X_min = X_tra[y_tra == 1]

                if len(X_maj) < 5:
                    print(f"  [SKIP] Fold {fold}: 訓練集正常樣本不足 ({len(X_maj)})")
                    continue
                if len(X_min) < 1:
                    print(f"  [SKIP] Fold {fold}: 訓練集無少數類（sampler 需參考點）")
                    continue
                if y_tst.sum() == 0:
                    print(f"  [SKIP] Fold {fold}: 測試集無少數類樣本")
                    continue

                scaler  = MinMaxScaler()
                X_maj_s = scaler.fit_transform(X_maj)
                X_min_s = scaler.transform(X_min)
                X_tst_s = scaler.transform(X_tst)
            except Exception as e:
                print(f"  [ERROR] Fold {fold} 資料載入失敗: {e}")
                continue

            for ae_type in AE_TYPES:
                for n_layers, ratio_label in param_configs:
                    ratio     = BOTTLENECK_RATIOS[ratio_label]
                    n_units   = max(2, round(input_dim * ratio))
                    cfg_label = f"h{n_layers}-{ratio_label}"

                    # AE 只訓練一次，抽 maj/test/min 三組 DF。
                    # ★ RNG 路徑對齊（讓 FW=none & Sampler=none 精準重現 baseline B）★
                    #   先 maj→test（對齊 B），再用 get/set_rng_state 包住 DF_min，
                    #   使「多抽 min」對 torch 亂數路徑零影響（與 J 完全一致）。
                    try:
                        extract = train_ae_and_get_extractor(
                            ae_type, X_maj_s, n_layers, n_units)
                        DF_maj = extract(X_maj_s)
                        DF_tst = extract(X_tst_s)
                        _rng_state = torch.get_rng_state()
                        DF_min = extract(X_min_s)
                        torch.set_rng_state(_rng_state)
                    except Exception as e:
                        print(f"  [ERROR] Fold{fold} {ae_type} {cfg_label}: AE 失敗 {e}")
                        continue

                    # 以【未清理】DF_maj 作為唯一 MinMax 基準（座標固定一次，對齊 B/J）。
                    scaler_df = MinMaxScaler().fit(DF_maj)
                    DF_maj_s = scaler_df.transform(DF_maj)
                    DF_min_s = scaler_df.transform(DF_min)
                    DF_tst_s = scaler_df.transform(DF_tst)

                    # ── FW 在外、Sampler 在內：同一組 weights 餵給該 FW 下所有 sampler ──
                    for fw in FW_METHODS:
                        try:
                            DF_maj_w, DF_min_w, DF_tst_w, _w = apply_weights(
                                DF_maj_s, DF_min_s, DF_tst_s, fw)
                        except Exception as e:
                            print(f"  [ERROR] Fold{fold} {ae_type} {cfg_label} FW={fw}: {e}")
                            continue

                        for sampler_name in SAMPLERS:
                            # ★ FW→US：在【加權後】DF_w 空間清理 majority ★
                            try:
                                DF_maj_clean, n_removed, sampler_status = undersample_df_weighted(
                                    DF_maj_w, DF_min_w, sampler_name)
                            except Exception as e:
                                print(f"  [ERROR] Fold{fold} {ae_type} {cfg_label} "
                                      f"FW={fw} Sampler={sampler_name}: 清理失敗 {e}")
                                continue

                            if len(DF_maj_clean) < 5:
                                print(f"  [SKIP] Fold{fold} {ae_type} {cfg_label} "
                                      f"FW={fw} Sampler={sampler_name}: 清理後 DF_maj 不足 "
                                      f"({len(DF_maj_clean)})")
                                continue

                            n_nb_cap = max(1, len(DF_maj_clean) - 1)  # 清理後重算 LOF k 上限

                            for occ_type in OCC_TYPES:
                                try:
                                    metrics = run_occ_eval(
                                        occ_type, DF_maj_clean, DF_tst_w, y_tst,
                                        n_nb_cap, do_scale=False)
                                except Exception as e:
                                    print(f"  [ERROR] Fold{fold} {ae_type} {cfg_label} "
                                          f"FW={fw} Sampler={sampler_name} {occ_type}: {e}")
                                    metrics = {m: float("nan") for m in METRIC_COLS}

                                all_records.append({
                                    "Dataset":    ds_name,
                                    "AE":         ae_type,
                                    "FW":         fw,
                                    "Sampler":    sampler_name,
                                    "OCC":        occ_type,
                                    "Config":     cfg_label,
                                    "Fold":       fold,
                                    "MajKept":       len(DF_maj_clean),
                                    "MajRemoved":    n_removed,
                                    "RemovedRate":   safe_removed_rate(n_removed, len(DF_maj_clean)),
                                    "SamplerStatus": sampler_status,
                                    "BaselineCheck": _baseline_tag(fw, sampler_name, occ_type),
                                    **metrics,
                                })

            print(f"  [fold {fold}] 完成 {len(param_configs)} cfg × "
                  f"{len(AE_TYPES)} AE × {len(FW_METHODS)} FW × "
                  f"{len(SAMPLERS)} US × {len(OCC_TYPES)} OCC")

    df_all = pd.DataFrame(all_records)

    # ── df_best：per-(Dataset, AE, FW, Sampler, OCC) 取 5-fold 平均 AUC 最高的 config ──
    if df_all.empty or "AUC" not in df_all.columns:
        print("\n⚠️  沒有可用的 AUC 結果，略過 best config 選取。")
        return df_all, pd.DataFrame()

    df_clean = df_all.dropna(subset=["AUC"])
    if df_clean.empty:
        print("\n⚠️  AUC 全為 NaN，略過 best config 選取。")
        return df_all, pd.DataFrame(columns=df_all.columns)

    best_cfg = (
        df_clean.groupby(["Dataset", "AE", "FW", "Sampler", "OCC", "Config"])["AUC"]
                .mean().reset_index()
                .sort_values("AUC", ascending=False)
                .drop_duplicates(["Dataset", "AE", "FW", "Sampler", "OCC"])
    )
    df_best = df_clean.merge(
        best_cfg[["Dataset", "AE", "FW", "Sampler", "OCC", "Config"]],
        on=["Dataset", "AE", "FW", "Sampler", "OCC", "Config"],
    )
    return df_all, df_best


def _baseline_tag(fw, sampler, occ):
    """標記哪些格子應重現舊 baseline，方便事後對拍。"""
    if fw == "none" and sampler == "none":
        return f"B_{occ}"
    if fw == "none" and sampler != "none":
        return f"J_{sampler}_{occ}"
    if fw != "none" and sampler == "none":
        return f"H_{fw}_{occ}"
    return ""   # 新貢獻格


# ─────────────────────────── Excel 樣式（與 J 共用）──────────────────────────
HEADER_FILL = PatternFill("solid", fgColor="2F5597")
SUBHDR_FILL = PatternFill("solid", fgColor="BDD7EE")
ALT_FILL    = PatternFill("solid", fgColor="F2F2F2")
BEST_FILL   = PatternFill("solid", fgColor="C6EFCE")   # 最佳 AUC / ΔAUC>0 綠底
WORSE_FILL  = PatternFill("solid", fgColor="FFC7CE")   # ΔAUC<0 紅底（Excel 標準 bad 樣式）

FW_FILL = {
    "none": PatternFill("solid", fgColor="FFFFFF"),
    "var":  PatternFill("solid", fgColor="DAEEF3"),
    "ivar": PatternFill("solid", fgColor="E2EFDA"),
    "mad":  PatternFill("solid", fgColor="FFF2CC"),
    "lap":  PatternFill("solid", fgColor="FCE4D6"),
}

HEADER_FONT  = Font(name="Arial", bold=True, color="FFFFFF", size=11)
SUBHDR_FONT  = Font(name="Arial", bold=True, color="1F3864", size=10)
BODY_FONT    = Font(name="Arial", size=10)
BOLD_FONT    = Font(name="Arial", bold=True, size=10)
GREEN_FONT   = Font(name="Arial", bold=True, size=10, color="006100")
RED_FONT     = Font(name="Arial", bold=True, size=10, color="9C0006")
CENTER_ALIGN = Alignment(horizontal="center", vertical="center", wrap_text=False)
LEFT_ALIGN   = Alignment(horizontal="left",   vertical="center")
THIN_BORDER  = Border(left=Side(style="thin"), right=Side(style="thin"),
                      top=Side(style="thin"),  bottom=Side(style="thin"))


def sc(cell, value, font=None, fill=None, align=None, fmt=None):
    cell.value  = value
    cell.border = THIN_BORDER
    if font:  cell.font          = font
    if fill:  cell.fill          = fill
    if align: cell.alignment     = align
    if fmt:   cell.number_format = fmt


def col_w(ws, col_letter, width):
    ws.column_dimensions[col_letter].width = width


# ─────────────────────────── all_per_fold 分頁 ───────────────────────────────
def write_per_fold(ws, df, title):
    ws.title = title
    cols = (["Dataset", "AE", "FW", "Sampler", "OCC", "Config", "Fold",
             "MajKept", "MajRemoved", "RemovedRate", "SamplerStatus", "BaselineCheck"]
            + METRIC_COLS)
    for c, h in enumerate(cols, 1):
        sc(ws.cell(1, c), h, font=HEADER_FONT, fill=HEADER_FILL, align=CENTER_ALIGN)
    if df.empty:
        sc(ws.cell(2, 1), "（無資料）", font=BODY_FONT, align=LEFT_ALIGN)
        return
    for r, (_, row) in enumerate(df.iterrows(), 2):
        fill = FW_FILL.get(row["FW"])
        for c, col in enumerate(cols, 1):
            sc(ws.cell(r, c), row[col], font=BODY_FONT, fill=fill,
               align=LEFT_ALIGN if col in ["Dataset", "SamplerStatus", "BaselineCheck"] else CENTER_ALIGN,
               fmt="0.0000" if col in METRIC_COLS + ["RemovedRate"] else None)
    widths = [22, 6, 8, 9, 8, 12, 6, 9, 11, 12, 16, 16] + [10] * len(METRIC_COLS)
    for i, w in enumerate(widths, 1):
        col_w(ws, get_column_letter(i), w)
    ws.freeze_panes = "A2"


# ─────────────────────────── best_grid 分頁（★核心比較表★）──────────────────
def write_best_grid(ws, df_best):
    """FW(列) × Sampler(欄) 的 best-config 平均，每個 OCC 一張表。

    每個 Sampler 群組欄位 = [AUC, F1, Recall, G-mean, ΔAUC]，其中
        ΔAUC = 該格 AUC − 同一 Sampler 欄『純 US(FW=none)』的 AUC。
    用途：直接讀出「在某個 sampler 上，先做 FW 比純 US 好多少」。
        ΔAUC > 0（綠）→ 先 FW 有幫助；< 0（紅）→ 先 FW 反而變差。
    全表最高 AUC 以綠字綠底標出。"""
    ws.title = "best_grid"

    if df_best.empty:
        sc(ws.cell(1, 1), "（無 best 結果）", font=BOLD_FONT, fill=SUBHDR_FILL, align=LEFT_ALIGN)
        col_w(ws, "A", 60)
        return

    g = (df_best.groupby(["OCC", "FW", "Sampler"])[METRIC_COLS]
                .agg(["mean", "std"]).reset_index())

    SUB_COLS = METRIC_COLS + ["ΔAUC"]      # 每個 Sampler 群組多一欄 ΔAUC
    grp_w    = len(SUB_COLS)

    r = 1
    for occ in OCC_TYPES:
        occ_block = g[g["OCC"] == occ]

        # 各 Sampler 欄的『純 US(FW=none)』AUC，作為 ΔAUC 基準
        none_auc = {}
        for sp in SAMPLERS:
            sub = occ_block[(occ_block["FW"] == "none") & (occ_block["Sampler"] == sp)]
            none_auc[sp] = sub[("AUC", "mean")].values[0] if not sub.empty else None

        # 大標題
        span = 1 + len(SAMPLERS) * grp_w
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=span)
        sc(ws.cell(r, 1),
           f"best_grid — OCC={occ}　(列=FW, 欄=Sampler；ΔAUC = 該格 − 同欄純US(FW=none))",
           font=Font(name="Arial", bold=True, size=12, color="1F3864"),
           fill=SUBHDR_FILL, align=CENTER_ALIGN)
        r += 1

        # 表頭列 1：Sampler 群組
        sc(ws.cell(r, 1), "FW＼Sampler", font=HEADER_FONT, fill=HEADER_FILL, align=CENTER_ALIGN)
        ws.merge_cells(start_row=r, start_column=1, end_row=r + 1, end_column=1)
        col = 2
        for sp in SAMPLERS:
            ws.merge_cells(start_row=r, start_column=col,
                           end_row=r, end_column=col + grp_w - 1)
            tag = sp if sp != "none" else "none(=純FW)"
            sc(ws.cell(r, col), tag, font=HEADER_FONT, fill=HEADER_FILL, align=CENTER_ALIGN)
            col += grp_w
        r += 1

        # 表頭列 2：metric（含 ΔAUC）
        col = 2
        for sp in SAMPLERS:
            for metric in SUB_COLS:
                sc(ws.cell(r, col), metric, font=SUBHDR_FONT, fill=SUBHDR_FILL, align=CENTER_ALIGN)
                col += 1
        r += 1

        # 全表最高 AUC（標綠）
        try:
            best_auc = occ_block[("AUC", "mean")].max()
        except Exception:
            best_auc = None

        # 內容：每個 FW 一列
        for fw in FW_METHODS:
            fill = FW_FILL.get(fw)
            tag = fw if fw != "none" else "none(=純US)"
            sc(ws.cell(r, 1), tag, font=BOLD_FONT, fill=fill, align=CENTER_ALIGN)
            col = 2
            for sp in SAMPLERS:
                sub = occ_block[(occ_block["FW"] == fw) & (occ_block["Sampler"] == sp)]
                auc_mean = None
                # 四個 metric
                for metric in METRIC_COLS:
                    try:
                        m = sub[(metric, "mean")].values[0]
                        s = sub[(metric, "std")].values[0]
                        display = f"{m:.4f} ± {s:.4f}"
                        if metric == "AUC":
                            auc_mean = m
                        is_best = (metric == "AUC" and best_auc is not None
                                   and abs(m - best_auc) < 1e-12)
                    except Exception:
                        display, is_best = "N/A", False
                    sc(ws.cell(r, col), display,
                       font=GREEN_FONT if is_best else BODY_FONT,
                       fill=BEST_FILL if is_best else fill, align=CENTER_ALIGN)
                    col += 1
                # ΔAUC 欄
                base = none_auc.get(sp)
                if fw == "none":
                    sc(ws.cell(r, col), "— (基準)", font=BODY_FONT, fill=fill, align=CENTER_ALIGN)
                elif auc_mean is not None and base is not None:
                    d = auc_mean - base
                    if d > 1e-6:
                        d_font, d_fill = GREEN_FONT, BEST_FILL
                    elif d < -1e-6:
                        d_font, d_fill = RED_FONT, WORSE_FILL
                    else:
                        d_font, d_fill = BODY_FONT, fill
                    sc(ws.cell(r, col), f"{d:+.4f}", font=d_font, fill=d_fill, align=CENTER_ALIGN)
                else:
                    sc(ws.cell(r, col), "N/A", font=BODY_FONT, fill=fill, align=CENTER_ALIGN)
                col += 1
            r += 1
        r += 1  # 表間空一列

    col_w(ws, "A", 16)
    ncol = 1 + len(SAMPLERS) * grp_w
    for i in range(2, ncol + 1):
        col_w(ws, get_column_letter(i), 16)


# ─────────────────────────── A~K 統整 flat 輸出 ──────────────────────────────
def make_ak_export_df(df, config_policy):
    if df.empty:
        return pd.DataFrame(columns=COMPARISON_EXPORT_COLS)
    out = pd.DataFrame({
        "Study": STUDY_ID, "Method": METHOD_ID, "Order": ORDER, "FeatureSet": FEATURE_SET,
        "Dataset": df["Dataset"], "AE": df["AE"], "FW": df["FW"],
        "Sampler": df["Sampler"], "OCC": df["OCC"], "Config": df["Config"],
        "Fold": df["Fold"], "ConfigPolicy": config_policy,
        "MajKept": df["MajKept"], "MajRemoved": df["MajRemoved"], "RemovedRate": df["RemovedRate"],
        "SamplerStatus": df["SamplerStatus"], "BaselineRef": df["BaselineCheck"],
        "OCCScope": OCC_SCOPE, "SamplerScaleMode": SAMPLER_SCALE_MODE,
    })
    for metric in METRIC_COLS:
        out[metric] = df[metric]
    return out[COMPARISON_EXPORT_COLS]


def write_ak_export(ws, df, title, config_policy):
    ws.title = title
    out = make_ak_export_df(df, config_policy=config_policy)
    for c, h in enumerate(COMPARISON_EXPORT_COLS, 1):
        sc(ws.cell(1, c), h, font=HEADER_FONT, fill=HEADER_FILL, align=CENTER_ALIGN)
    for r, (_, row) in enumerate(out.iterrows(), 2):
        fill = FW_FILL.get(row["FW"])
        for c, col in enumerate(COMPARISON_EXPORT_COLS, 1):
            sc(ws.cell(r, c), row[col], font=BODY_FONT, fill=fill,
               align=LEFT_ALIGN if col in ["Dataset", "Method", "Order", "BaselineRef", "OCCScope", "SamplerScaleMode"] else CENTER_ALIGN,
               fmt="0.0000" if col in METRIC_COLS + ["RemovedRate"] else None)
    widths = [7, 14, 12, 10, 22, 6, 8, 9, 8, 12, 6, 22, 9, 11, 12, 16, 16, 18, 48] + [10] * len(METRIC_COLS)
    for i, w in enumerate(widths, 1):
        col_w(ws, get_column_letter(i), w)
    ws.freeze_panes = "A2"


# ─────────────────────────── alignment_notes 分頁 ────────────────────────────
def write_alignment_notes(ws):
    """對齊備忘（與 J 同風格的 Item | Value 表），記錄 M 的設定與三個對拍點。"""
    ws.title = "alignment_notes"
    sc(ws.cell(1, 1), "Item",  font=HEADER_FONT, fill=HEADER_FILL, align=CENTER_ALIGN)
    sc(ws.cell(1, 2), "Value", font=HEADER_FONT, fill=HEADER_FILL, align=CENTER_ALIGN)

    rows = [
        ("Study", STUDY_ID),
        ("Method", METHOD_ID),
        ("Order", ORDER),
        ("FeatureSet", FEATURE_SET),
        ("BaselineRef", BASELINE_REF),
        ("AE", ", ".join(AE_TYPES)),
        ("AE epochs", AE_EPOCHS),
        ("AE batch size", AE_BATCH_SIZE),
        ("AE learning rate", AE_LR),
        ("VAE beta", VAE_BETA),
        ("Grid", f"{len(N_LAYERS_LIST)} layers × {len(BOTTLENECK_RATIOS)} ratios "
                 f"= {len(ALL_CONFIGS)} configs"),
        ("OCCScope", OCC_SCOPE),
        ("OCC", ", ".join(OCC_TYPES)),
        ("FW", ", ".join(FW_METHODS)),
        ("FW normalize", "max(w)=1（最重要 feature 保留 100%，weighted DF ∈ [0,1]）"),
        ("Sampler", ", ".join(SAMPLERS)),
        ("SamplerScaleMode", SAMPLER_SCALE_MODE),
        ("Pipeline",
         "OF→MinMax→AE→DF→MinMax(fit 未清理 DF_maj)→FW(套 maj/min/test)"
         "→US(在加權後 DF_w 清理 majority)→OCC(do_scale=False)"),
        ("對拍點 1（=B）",
         "FW=none & Sampler=none：應重現 baseline B 的對應 AE×OCC×Config。"),
        ("對拍點 2（=J）",
         "FW=none & Sampler≠none：應重現 J（DF 側 US）對應的 AE×Sampler×OCC×Config。"),
        ("對拍點 3（=H）",
         "FW≠none & Sampler=none：應重現 H（DF 側 FW）對應的 FW×OCC×Config。"),
        ("FW=none 恆等性",
         "normalize(ones)=ones → DF_w=DF_s，故 FW=none 路徑與 J 逐位元相同。"),
        ("ΔAUC 定義",
         "best_grid 中 ΔAUC = 該格 AUC − 同一 Sampler 欄的純 US(FW=none) AUC。"),
        ("Threshold", "訓練 majority 異常分數的第 90 百分位。"),
        ("LOF n_neighbors", "min(20, 清理後 majority 數 − 1)。"),
        ("Contamination", 0.1),
        ("Best config policy",
         "per-(Dataset, AE, FW, Sampler, OCC)，以 5-fold 平均 AUC 選 Config。"),
        ("Data leakage guard",
         "DF scaler 只 fit 未清理訓練 majority；FW weights 只在訓練 majority 算；"
         "sampler 只用 train fold；test 只做 transform。"),
    ]
    for i, (k, v) in enumerate(rows, 2):
        sc(ws.cell(i, 1), k, font=BOLD_FONT, fill=SUBHDR_FILL, align=LEFT_ALIGN)
        sc(ws.cell(i, 2), v, font=BODY_FONT, align=LEFT_ALIGN)

    col_w(ws, "A", 22)
    col_w(ws, "B", 92)
    ws.freeze_panes = "A2"


# ─────────────────────────── Excel 存檔 ──────────────────────────────────────
def save_excel(df_all, df_best):
    wb  = Workbook()
    ws1 = wb.active
    ws2 = wb.create_sheet()
    ws3 = wb.create_sheet()
    ws4 = wb.create_sheet()
    ws5 = wb.create_sheet()

    write_per_fold(ws1, df_all, "all_per_fold")
    write_best_grid(ws2, df_best)
    write_ak_export(ws3, df_all,  title="ak_all_export",  config_policy="all_configs")
    if df_best.empty:
        sc(ws4.cell(1, 1), "（無 best 結果）", font=BOLD_FONT, fill=SUBHDR_FILL, align=LEFT_ALIGN)
        ws4.title = "ak_best_export"
    else:
        write_ak_export(ws4, df_best, title="ak_best_export", config_policy="best_config_per_dataset")
    write_alignment_notes(ws5)

    wb.save(OUTPUT_FILE)
    print(f"\n✅ 結果已儲存至：{OUTPUT_FILE.resolve()}")


# ─────────────────────────── Entry Point ─────────────────────────────────────
if __name__ == "__main__":
    print("=" * 68)
    print("Study Two 整合：M — FW → US（DF 端；在加權空間清理）")
    print(f"AE        : {AE_TYPES}（聚焦；可擴 4 AE）")
    print(f"FW        : {FW_METHODS}（含 none = 純 US）")
    print(f"Sampler   : {SAMPLERS}（含 none = 純 FW）")
    print(f"OCC       : {OCC_TYPES}（聚焦；可擴 3 OCC）")
    print(f"OCCScope  : {OCC_SCOPE}")
    print(f"ScaleMode : {SAMPLER_SCALE_MODE}")
    print(f"資料集    : {'前 1 個' if not RUN_ONLY_DATASETS else RUN_ONLY_DATASETS}")
    print(f"總組合數  : {len(AE_TYPES)}AE × {len(FW_METHODS)}FW × {len(SAMPLERS)}US × "
          f"{len(OCC_TYPES)}OCC × {len(ALL_CONFIGS)}cfg = "
          f"{len(AE_TYPES)*len(FW_METHODS)*len(SAMPLERS)*len(OCC_TYPES)*len(ALL_CONFIGS)} / fold")
    print("Pipeline  : OF→MinMax→AE→DF→MinMax(fit 未清理 DF_maj)→FW→US(加權空間)→LOF")
    print("對拍       : FW=none&US=none→B；FW=none&US≠none→J；FW≠none&US=none→H")
    print("=" * 68)

    df_all, df_best = run_experiment()

    if df_all.empty:
        print("\n⚠️  沒有任何結果，請確認資料路徑與檔名格式。")
    else:
        save_excel(df_all, df_best)
        if not df_best.empty:
            print("\n── Best-config Overall（by FW × Sampler, OCC=LOF）──")
            piv = (df_best[df_best["OCC"] == OCC_TYPES[0]]
                   .groupby(["FW", "Sampler"])["AUC"].mean().unstack("Sampler").round(4))
            print(piv.to_string())
            print("\n提示：比較同一 Sampler 欄的不同 FW 列，即為『先 FW』帶來的 ΔAUC。")
