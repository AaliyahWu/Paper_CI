# Baseline C v3：OF_maj + DF_maj → OCC3 (自適應 DAE + OCSVM)
#
# ══════════════════════════════════════════════════════════════════════
# v3 相對於 v2 的改進項目（與 Baseline B v3 的改進保持一致）
# ══════════════════════════════════════════════════════════════════════
#
# 【核心改進：自適應壓縮維度強化，同步 B v3】
#   v2：只有 0.5 和 0.75 兩種壓縮比
#   v3：新增 ratio=1.0（不壓縮），根據 IR 更精細化調整
#       - 若 n_maj < 50：ratio=0.5（樣本少，強壓縮防過擬合）
#       - 若 n_maj 中等（50~200）：根據 IR 判斷
#           * IR > 10：ratio=0.5（高不平衡，強壓縮）
#           * 5 < IR ≤ 10：ratio=0.75（中等不平衡，輕壓縮）
#           * IR ≤ 5：ratio=1.0（低不平衡，不壓縮）[v3 新增]
#       - 若 n_maj 大（> 200）：根據 IR 判斷
#           * IR > 20：ratio=0.5（極高不平衡，強壓縮）
#           * 10 < IR ≤ 20：ratio=0.75（高不平衡，輕壓縮）
#           * 5 < IR ≤ 10：ratio=0.75（中等不平衡，輕壓縮）
#           * IR ≤ 5：ratio=1.0（低不平衡，不壓縮）[v3 新增]
#       目的：對低不平衡資料集（如 glass1, yeast1 低 IR）保留完整特徵空間，
#             DF 標準化（v2 已加入）確保 OF 與 DF 同尺度，避免維度爆炸問題
#
# 【改進2：nu 搜尋範圍回歸 {0.01, 0.05, 0.1}（同 B v3）】
#   v2：nu ∈ {0.01, 0.05, 0.1, 0.2}
#   v3：回歸 nu ∈ {0.01, 0.05, 0.1}（與 B v3 一致）
#
# 【繼承 v2 的所有改進】
#   - DF 獨立標準化 normalize_df()（只 fit majority DF，確保 OF/DF 同尺度）
#   - OCSVM gamma 搜尋 {scale, 0.5, 0.1, 0.05, 0.01, 0.001}（6 個值）
#   - 自適應 noise_std 和 patience
#   - epochs=100, batch=32, weight_decay=1e-4
#   - LayerNorm、兩層隱藏層
#
# 執行：python 03v3_run_baselineC.py --data_root .
# 輸出：./results/baseline_c_v3_results.xlsx
# ══════════════════════════════════════════════════════════════════════

from __future__ import annotations

import argparse
import copy
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


# ============================================================
# 隨機種子
# ============================================================
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# ============================================================
# KEEL 讀取器
# ============================================================
def load_keel_dat(dat_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    dat_path = Path(dat_path)
    with dat_path.open("r", encoding="utf-8", errors="ignore") as f:
        raw = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("%")]

    data_idx = next((i for i, ln in enumerate(raw) if ln.lower() == "@data"), None)
    header     = raw[:data_idx] if data_idx is not None else []
    data_lines = raw[data_idx + 1:] if data_idx is not None else raw

    attr_names, output_name = [], None
    for ln in header:
        low = ln.lower()
        if low.startswith("@attribute"):
            parts = ln.split(None, 2)
            if len(parts) >= 2:
                attr_names.append(parts[1].strip())
        elif low.startswith("@outputs"):
            output_name = ln.split(None, 1)[1].strip()

    rows = [[p.strip() for p in dl.split(",")] for dl in data_lines]
    if not rows:
        raise ValueError(f"無資料列: {dat_path}")
    if not attr_names:
        attr_names = [f"c{i}" for i in range(len(rows[0]))]
    rows = [r for r in rows if len(r) == len(attr_names)]
    df   = pd.DataFrame(rows, columns=attr_names)

    y_col = output_name if (output_name and output_name in df.columns) else df.columns[-1]
    y = df[y_col].astype(str)
    X = df.drop(columns=[y_col])
    for c in X.columns:
        conv = pd.to_numeric(X[c], errors="coerce")
        X[c] = conv if not conv.isna().any() else df[c].astype(str)
    return X, y


# ============================================================
# [v3 改進1] 自適應超參數計算（新增 ratio=1.0，同步 B v3）
# ============================================================
def adaptive_ae_params(n_maj: int, input_dim: int, ir: float) -> Tuple[float, float, int]:
    """
    根據資料集特性動態決定 (latent_ratio, noise_std, patience)

    v3 改進：新增 ratio=1.0 用於低不平衡資料集（IR ≤ 5），同步 B v3 邏輯
    DF 標準化（v2 已加入）確保 OF 與 DF 同尺度，控制串接後的維度影響

    參數：
      n_maj:      majority 訓練樣本數
      input_dim:  原始特徵維度
      ir:         Imbalance Ratio (n_maj / n_min)

    回傳：
      (latent_ratio, noise_std, patience)
    """
    if n_maj < 50:
        # 樣本極少（如 ecoli IR=43, n_maj≈43）：強壓縮 + 大噪聲 + 長耐心
        latent_ratio = 0.5
        noise_std    = 0.05
        patience     = 20
    elif n_maj < 200:
        # 樣本中等（如 cleveland n_maj≈102）
        if ir > 10:
            # 高不平衡：強壓縮
            latent_ratio = 0.5
        elif ir > 5:
            # 中等不平衡：輕壓縮
            latent_ratio = 0.75
        else:
            # [v3] 低不平衡：不壓縮，保留完整特徵（同 B v3）
            latent_ratio = 1.0
        noise_std = 0.02
        patience  = 15
    else:
        # 樣本充足（yeast1 n_maj≈675, abalone n_maj≈1459）
        if ir > 20:
            # 極高不平衡：強壓縮
            latent_ratio = 0.5
        elif ir > 10:
            # 高不平衡：輕壓縮
            latent_ratio = 0.75
        elif ir > 5:
            # 中等不平衡：輕壓縮
            latent_ratio = 0.75
        else:
            # [v3] 低不平衡：不壓縮，保留完整特徵（同 B v3）
            latent_ratio = 1.0
        noise_std = 0.01
        patience  = 15

    return latent_ratio, noise_std, patience


# ============================================================
# DAE 模型（架構與 B v3 / C v2 相同）
# ============================================================
def build_torch_autoencoder(input_dim: int, latent_dim: int):
    import torch.nn as nn

    hidden1 = max(16, min(128, input_dim * 4))
    hidden2 = max(8,  min(64,  input_dim * 2))

    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden1),
                nn.LayerNorm(hidden1),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden1, hidden2),
                nn.LayerNorm(hidden2),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden2, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden2),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden2, hidden1),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden1, input_dim),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

        def encode(self, x):
            return self.encoder(x)

    return AE()


@dataclass
class AEConfig:
    epochs:       int   = 100
    batch_size:   int   = 32
    lr:           float = 1e-3
    weight_decay: float = 1e-4
    noise_std:    float = 0.01
    latent_ratio: float = 0.5
    patience:     int   = 15


def train_autoencoder(X_maj: np.ndarray, cfg: AEConfig, seed: int = 42):
    import torch
    seed_everything(seed)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    input_dim = X_maj.shape[1]
    latent_dim = max(2, int(input_dim * cfg.latent_ratio))

    model   = build_torch_autoencoder(input_dim, latent_dim).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = torch.nn.MSELoss()
    Xt      = torch.tensor(X_maj, dtype=torch.float32)
    n       = Xt.size(0)

    best_loss, patience_cnt, best_state = float("inf"), 0, None

    for _ in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n)
        ep_loss, cnt = 0.0, 0
        for i in range(0, n, cfg.batch_size):
            xb    = Xt[perm[i: i + cfg.batch_size]].to(device)
            xb_in = xb + cfg.noise_std * torch.randn_like(xb) if cfg.noise_std > 0 else xb
            loss  = loss_fn(model(xb_in), xb)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); cnt += 1

        avg = ep_loss / max(cnt, 1)
        if avg < best_loss - 1e-6:
            best_loss, patience_cnt = avg, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.patience:
                if best_state: model.load_state_dict(best_state)
                break

    if best_state: model.load_state_dict(best_state)
    model.eval()
    return model, latent_dim


def encode_features(model, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outs   = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = torch.tensor(X[i: i + batch_size], dtype=torch.float32).to(device)
            outs.append(model.encode(xb).cpu().numpy())
    return np.vstack(outs)


# ============================================================
# 前處理工具
# ============================================================
FOLD_RE = re.compile(r"-(\d+)(tra|tst)\.dat$", re.IGNORECASE)


def discover_folds(dataset_dir: Path) -> Dict[int, Dict[str, Path]]:
    folds: Dict[int, Dict[str, Path]] = {}
    for p in dataset_dir.glob("*.dat"):
        m = FOLD_RE.search(p.name)
        if not m: continue
        fid, split = int(m.group(1)), m.group(2).lower()
        folds.setdefault(fid, {})[split] = p
    folds = {k: v for k, v in folds.items() if "tra" in v and "tst" in v}
    return dict(sorted(folds.items()))


def preprocess_train_test(
    X_train_df: pd.DataFrame, X_test_df: pd.DataFrame, y_train_bin: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """OF 標準化：StandardScaler 只 fit majority"""
    all_df  = pd.get_dummies(
        pd.concat([X_train_df, X_test_df], axis=0, ignore_index=True), drop_first=False
    )
    X_tr_raw = all_df.iloc[:len(X_train_df)].values.astype(np.float32)
    X_te_raw = all_df.iloc[len(X_train_df):].values.astype(np.float32)
    scaler   = StandardScaler()
    scaler.fit(X_tr_raw[y_train_bin == 0])
    return scaler.transform(X_tr_raw), scaler.transform(X_te_raw)


def normalize_df(
    DF_train: np.ndarray, DF_test: np.ndarray, ytr_bin: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    [v2 繼承] DF 標準化：確保 OF 與 DF 在同等尺度上
    只 fit majority 的 DF，再 transform 全體
    """
    scaler = StandardScaler()
    scaler.fit(DF_train[ytr_bin == 0])
    return scaler.transform(DF_train), scaler.transform(DF_test)


def minority_label_from_train(y: pd.Series) -> str:
    return str(y.value_counts().idxmin())


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """計算 G-mean、Recall(min)、F1"""
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    gmean = math.sqrt(tpr * tnr)
    return {"gmean": float(gmean), "recall_min": float(rec), "f1": float(f1)}


# ============================================================
# 核心流程
# ============================================================
def run_fold(train_path: Path, test_path: Path, seed: int, ae_cfg: AEConfig) -> Dict[str, float]:
    """
    Baseline C v3 單折流程：
      1. 讀資料
      2. OF 標準化（scaler 只 fit majority）
      3. [v3] 自適應計算 latent_ratio / noise_std / patience（新增 ratio=1.0）
      4. 訓練 DAE（只用 majority）→ 提取 DF
      5. [v2 繼承] DF 標準化（scaler 只 fit majority DF）
      6. 特徵串接：Z = [OF_norm | DF_norm]
      7. [v3] 調優 OCSVM（nu ∈ {0.01, 0.05, 0.1}，gamma 保持 6 個值）
      8. 重訓最終模型（全部訓練 majority Z）
      9. 測試集評估（OCSVM.predict()）
    """
    Xtr_df, ytr = load_keel_dat(train_path)
    Xte_df, yte = load_keel_dat(test_path)

    minor   = minority_label_from_train(ytr)
    ytr_bin = (ytr.astype(str) == minor).astype(int).values
    yte_bin = (yte.astype(str) == minor).astype(int).values

    # OF 標準化
    Xtr, Xte = preprocess_train_test(Xtr_df, Xte_df, ytr_bin)
    Xtr_maj  = Xtr[ytr_bin == 0]
    n_maj    = Xtr_maj.shape[0]
    n_min    = max((ytr_bin == 1).sum(), 1)
    ir       = n_maj / n_min
    input_dim = Xtr.shape[1]

    if n_maj < 5:
        raise ValueError(f"majority 樣本過少: {n_maj}")

    # [v3] 自適應超參數（新增 ratio=1.0 用於低不平衡資料集）
    lat_ratio, noise_std, patience = adaptive_ae_params(n_maj, input_dim, ir)
    ae_cfg_fold = AEConfig(
        epochs      = ae_cfg.epochs,
        batch_size  = max(8, min(ae_cfg.batch_size, n_maj // 2)),
        lr          = ae_cfg.lr,
        weight_decay = ae_cfg.weight_decay,
        noise_std   = noise_std,
        latent_ratio = lat_ratio,
        patience    = patience,
    )

    print(
        f"      [自適應] n_maj={n_maj}, IR={ir:.1f} → "
        f"latent_ratio={lat_ratio}, noise={noise_std}, patience={patience}"
    )

    # 訓練 DAE（只用 majority）
    print(f"      [DAE] 訓練中...", end=" ", flush=True)
    ae_model, latent_dim = train_autoencoder(Xtr_maj, ae_cfg_fold, seed=seed)
    print(f"完成 (latent_dim={latent_dim})")

    # 提取深度特徵
    print(f"      [DAE] 提取 DF...", end=" ", flush=True)
    DFtr_raw = encode_features(ae_model, Xtr)
    DFte_raw = encode_features(ae_model, Xte)
    print("完成")

    # [v2 繼承] DF 標準化（獨立的 scaler，只 fit majority DF）
    DFtr, DFte = normalize_df(DFtr_raw, DFte_raw, ytr_bin)

    # 特徵串接：OF_norm + DF_norm
    Ztr = np.concatenate([Xtr, DFtr], axis=1)   # (n_train, input_dim + latent_dim)
    Zte = np.concatenate([Xte, DFte], axis=1)
    concat_dim = Ztr.shape[1]

    print(f"      [串接] OF({input_dim}) + DF({latent_dim}) = {concat_dim} 維")

    # 調優 OCSVM
    tr_idx, va_idx = train_test_split(
        np.arange(len(ytr_bin)), test_size=0.25, random_state=seed, stratify=ytr_bin,
    )
    Z_fit_maj = Ztr[tr_idx][ytr_bin[tr_idx] == 0]
    Z_val     = Ztr[va_idx]
    y_val     = ytr_bin[va_idx]

    print(f"      [OCSVM] 調優...", end=" ", flush=True)
    best_auc    = -1.0
    best_params = {"nu": 0.05, "gamma": "scale"}

    # [v3 改進2] nu 回歸 {0.01, 0.05, 0.1}（同 B v3）；gamma 保持 v2 的 6 個值
    for nu in [0.01, 0.05, 0.1]:
        for gamma in ["scale", 0.5, 0.1, 0.05, 0.01, 0.001]:
            clf = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
            clf.fit(Z_fit_maj)
            val_anom = -clf.decision_function(Z_val)
            try:
                auc_val = roc_auc_score(y_val, val_anom)
            except Exception:
                auc_val = 0.0
            if auc_val > best_auc:
                best_auc    = auc_val
                best_params = {"nu": nu, "gamma": gamma}

    print(f"完成 (nu={best_params['nu']}, gamma={best_params['gamma']}, val_auc={best_auc:.4f})")

    # 最終模型（全部訓練 majority 的串接特徵）
    clf_final = OneClassSVM(kernel="rbf", nu=best_params["nu"], gamma=best_params["gamma"])
    clf_final.fit(Ztr[ytr_bin == 0])

    # 測試集評估
    test_anom = -clf_final.decision_function(Zte)
    auc_test  = roc_auc_score(yte_bin, test_anom)
    y_pred    = (clf_final.predict(Zte) == -1).astype(int)
    test_m    = compute_metrics(yte_bin, y_pred)

    return {
        "auc":          float(auc_test),
        "gmean":        float(test_m["gmean"]),
        "recall_min":   float(test_m["recall_min"]),
        "f1":           float(test_m["f1"]),
        "latent_dim":   float(latent_dim),
        "concat_dim":   float(concat_dim),
        "latent_ratio": float(lat_ratio),
        "noise_std":    float(noise_std),
        "ocsvm_nu":     float(best_params["nu"]),
        "ocsvm_gamma":  best_params["gamma"],
        "val_auc":      float(best_auc),
        "n_maj":        int(n_maj),
        "ir":           float(ir),
    }


# ============================================================
# 資料集定義
# ============================================================
DATASETS = {
    "ecoli-0137_vs_26":        "ecoli-0-1-3-7_vs_2-6-5-fold",
    "glass-01236_vs_456":      "glass-0-1-2-3_vs_4-5-6-5-fold",
    "yeast-05679_vs_45":       "yeast-0-5-6-7-9_vs_4-5-fold",
    "glass1":                  "glass1-5-fold",
    "yeast1":                  "yeast1-5-fold",
    "cleveland-0_vs_4":        "cleveland-0_vs_4-5-fold",
    "yeast-2_vs_8":            "yeast-2_vs_8-5-fold",
    "abalone-17_vs_7-8-9-10":  "abalone-17_vs_7-8-9-10-5-fold",
}


# ============================================================
# CLI & 主程式
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Baseline C v3: OF_maj + DF_maj → OCC3（自適應 DAE concat OCSVM）"
    )
    p.add_argument("--data_root", type=str,   default=".",     help="專案根目錄")
    p.add_argument("--out_dir",   type=str,   default="results")
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--ae_epochs", type=int,   default=100)
    p.add_argument("--ae_batch",  type=int,   default=32)
    p.add_argument("--ae_lr",     type=float, default=1e-3)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    root    = Path(args.data_root)
    out_dir = root / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ae_cfg = AEConfig(epochs=args.ae_epochs, batch_size=args.ae_batch, lr=args.ae_lr)

    print("=" * 70)
    print("Baseline C v3: OF_maj + DF_maj → OCC3（自適應 DAE concat OCSVM）")
    print("  [v3 改進1] 自適應 latent_ratio 新增 ratio=1.0（低不平衡資料集，同 B v3）")
    print("    - ratio=1.0：IR ≤ 5 時不壓縮，保留完整特徵空間")
    print("    - ratio=0.75：5 < IR ≤ 10/20，輕壓縮")
    print("    - ratio=0.5：IR > 10/20，強壓縮")
    print("  [v3 改進2] OCSVM nu 回歸 {0.01, 0.05, 0.1}（同 B v3）")
    print("  [繼承 v2] DF 獨立標準化、gamma {scale,0.5,0.1,0.05,0.01,0.001}、epochs=100、batch=32")
    print("=" * 70)

    per_fold_records, summary_records = [], []

    for ds_key, ds_folder in DATASETS.items():
        ddir = root / "data" / ds_folder
        if not ddir.exists():
            print(f"[警告] 資料夾不存在: {ddir}", file=sys.stderr)
            continue

        folds = discover_folds(ddir)
        if not folds:
            print(f"[警告] 無 fold: {ddir}", file=sys.stderr)
            continue

        print(f"\n【資料集】{ds_key} | {len(folds)} fold")
        print("-" * 70)
        fold_metrics = []

        for fid, paths in folds.items():
            try:
                print(f"  Fold {fid:02d}:")
                m = run_fold(paths["tra"], paths["tst"], seed=args.seed, ae_cfg=ae_cfg)
                fold_metrics.append(m)
                per_fold_records.append({
                    "dataset": ds_key, "dataset_dir": ddir.name, "fold": fid,
                    "auc": m["auc"], "gmean": m["gmean"],
                    "recall_min": m["recall_min"], "f1": m["f1"],
                    "latent_dim": int(m["latent_dim"]),
                    "concat_dim": int(m["concat_dim"]),
                    "latent_ratio": m["latent_ratio"],
                    "noise_std": m["noise_std"],
                    "ocsvm_nu": m["ocsvm_nu"], "ocsvm_gamma": m["ocsvm_gamma"],
                    "val_auc": m["val_auc"],
                    "n_maj": m["n_maj"], "ir": round(m["ir"], 2),
                })
                print(
                    f"    AUC={m['auc']:.4f} | G-mean={m['gmean']:.4f} | "
                    f"Recall={m['recall_min']:.4f} | F1={m['f1']:.4f}"
                )
            except Exception as e:
                print(f"    [錯誤] Fold {fid}: {e}", file=sys.stderr)

        if fold_metrics:
            keys = ["auc", "gmean", "recall_min", "f1"]
            avg  = {k: float(np.mean([fm[k] for fm in fold_metrics])) for k in keys}
            std  = {k: float(np.std( [fm[k] for fm in fold_metrics])) for k in keys}
            summary_records.append({
                "dataset": ds_key, "dataset_dir": ddir.name,
                "n_folds": len(fold_metrics),
                "auc_mean": avg["auc"],           "auc_std": std["auc"],
                "gmean_mean": avg["gmean"],        "gmean_std": std["gmean"],
                "recall_min_mean": avg["recall_min"], "recall_min_std": std["recall_min"],
                "f1_mean": avg["f1"],              "f1_std": std["f1"],
            })
            print(
                f"\n  【平均】AUC={avg['auc']:.4f}±{std['auc']:.4f} | "
                f"G-mean={avg['gmean']:.4f}±{std['gmean']:.4f} | "
                f"Recall={avg['recall_min']:.4f}±{std['recall_min']:.4f} | "
                f"F1={avg['f1']:.4f}±{std['f1']:.4f}"
            )

    # 輸出 Excel
    df_all     = pd.DataFrame(per_fold_records)
    df_summary = pd.DataFrame(summary_records)
    df_overall = pd.DataFrame([{
        "dataset": "ALL",
        "n_datasets": int(df_summary["dataset"].nunique()) if not df_summary.empty else 0,
        "n_folds": len(df_all),
        "auc_mean":  float(df_all["auc"].mean())  if not df_all.empty else np.nan,
        "auc_std":   float(df_all["auc"].std(ddof=0))  if not df_all.empty else np.nan,
        "gmean_mean": float(df_all["gmean"].mean()) if not df_all.empty else np.nan,
        "gmean_std":  float(df_all["gmean"].std(ddof=0)) if not df_all.empty else np.nan,
        "recall_min_mean": float(df_all["recall_min"].mean()) if not df_all.empty else np.nan,
        "recall_min_std":  float(df_all["recall_min"].std(ddof=0)) if not df_all.empty else np.nan,
        "f1_mean": float(df_all["f1"].mean()) if not df_all.empty else np.nan,
        "f1_std":  float(df_all["f1"].std(ddof=0)) if not df_all.empty else np.nan,
    }])

    xlsx_path = out_dir / "baseline_c_v3_results.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_all.to_excel(writer,     sheet_name="per_fold",     index=False)
        df_summary.to_excel(writer, sheet_name="summary",      index=False)
        df_overall.to_excel(writer, sheet_name="overall_mean", index=False)

    print("\n" + "=" * 70)
    print("【最終總結】")
    for row in summary_records:
        print(
            f"{row['dataset']:30s} | "
            f"AUC={row['auc_mean']:.4f}±{row['auc_std']:.4f} | "
            f"G-mean={row['gmean_mean']:.4f}±{row['gmean_std']:.4f} | "
            f"Recall={row['recall_min_mean']:.4f} | F1={row['f1_mean']:.4f}"
        )
    print(f"\n✓ 儲存至: {xlsx_path.resolve()}")


if __name__ == "__main__":
    main()
