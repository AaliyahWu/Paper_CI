# Baseline C v2：OF_maj + DF_maj → OCC3 (自適應 VAE + OCSVM)
#
# ══════════════════════════════════════════════════════════════════════
# v2 相對於 v1 的改進項目（與 Baseline B v2 / C v2 DAE 保持一致）
# ══════════════════════════════════════════════════════════════════════
#
# 【DAE → VAE 差異】
#   - noise_std（加噪輸入）→ kl_weight（KL 散度權重 β）
#   - Encoder 輸出 (mu, log_var)，推論時取 mu（確定性）
#
# 【v2 改進1】自適應壓縮維度（與 B v2 相同邏輯）
#   串接後維度 = input_dim + latent_dim，壓縮 DF 控制總維度
#
# 【v2 改進2】串接特徵的 DF 標準化
#   對 DF 部分再做一次 StandardScaler（只 fit majority DF）
#   確保 OF 與 DF 在相同尺度上
#
# 【v2 改進3】OCSVM gamma 搜尋針對高維空間加密
#   gamma ∈ {scale, 0.5, 0.1, 0.05, 0.01, 0.001}
#
# 【v2 改進4】nu 搜尋加入 0.2
#
# 【v2 改進5】自適應 kl_weight 和 patience（對應 DAE v2 的 noise_std/patience）
#
# 【v2 改進6】batch_size=32（適合小資料集）
#
# 執行：python 03v2_run_baselineC_VAE.py --data_root .
# 輸出：./results/baseline_c_vae_v2_results.xlsx
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
from typing import Dict, List, Tuple

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
# [v2 改進1] 自適應超參數（VAE 版：noise_std → kl_weight）
# ============================================================
def adaptive_vae_params(n_maj: int, input_dim: int, ir: float) -> Tuple[float, float, int]:
    """
    根據 n_maj 和 IR 動態決定 (latent_ratio, kl_weight, patience)

    latent_ratio 邏輯與 DAE C v2 相同；
    kl_weight 對應 noise_std：樣本少/IR 高 → 較大 kl_weight（加強正則化）
    特別考量：串接後維度 = input_dim + latent_dim，壓縮 DF 控制總維度
    """
    if n_maj < 50:
        latent_ratio = 0.5
        kl_weight    = 5e-3
        patience     = 20
    elif n_maj < 200:
        if ir > 10:
            latent_ratio = 0.5
        else:
            latent_ratio = 0.75
        kl_weight = 2e-3
        patience  = 15
    else:
        if ir > 20:
            latent_ratio = 0.5
        elif ir > 5:
            latent_ratio = 0.75
        else:
            latent_ratio = 0.75
        kl_weight = 1e-3
        patience  = 15

    return latent_ratio, kl_weight, patience


# ============================================================
# VAE 模型
# ============================================================
def build_vae(input_dim: int, latent_dim: int):
    """兩層 hidden MLP VAE，LayerNorm + LeakyReLU(0.1)"""
    import torch.nn as nn

    hidden1 = max(16, min(128, input_dim * 4))
    hidden2 = max(8,  min(64,  input_dim * 2))

    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder_net = nn.Sequential(
                nn.Linear(input_dim, hidden1),
                nn.LayerNorm(hidden1),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden1, hidden2),
                nn.LayerNorm(hidden2),
                nn.LeakyReLU(0.1),
            )
            self.fc_mu      = nn.Linear(hidden2, latent_dim)
            self.fc_log_var = nn.Linear(hidden2, latent_dim)

            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden2),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden2, hidden1),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden1, input_dim),
            )

        def encode(self, x):
            h = self.encoder_net(x)
            return self.fc_mu(h)

        def _encode_params(self, x):
            h = self.encoder_net(x)
            return self.fc_mu(h), self.fc_log_var(h)

        def reparameterize(self, mu, log_var):
            import torch
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(self, x):
            mu, log_var = self._encode_params(x)
            z = self.reparameterize(mu, log_var)
            return self.decoder(z), mu, log_var

    return VAE()


@dataclass
class VAEConfig:
    epochs:       int   = 100
    batch_size:   int   = 32
    lr:           float = 1e-3
    weight_decay: float = 1e-4
    kl_weight:    float = 1e-3
    latent_ratio: float = 0.5
    patience:     int   = 15


def vae_loss(recon, x, mu, log_var, kl_weight: float):
    import torch.nn.functional as F
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp()).mean()
    return recon_loss + kl_weight * kl


def train_vae(X_maj: np.ndarray, cfg: VAEConfig, seed: int = 42):
    import torch
    seed_everything(seed)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    input_dim  = X_maj.shape[1]
    latent_dim = max(2, int(input_dim * cfg.latent_ratio))

    model = build_vae(input_dim, latent_dim).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    Xt    = torch.tensor(X_maj, dtype=torch.float32)
    n     = Xt.size(0)

    best_loss, patience_cnt, best_state = float("inf"), 0, None

    for _ in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n)
        ep_loss, cnt = 0.0, 0
        for i in range(0, n, cfg.batch_size):
            xb = Xt[perm[i: i + cfg.batch_size]].to(device)
            recon, mu, log_var = model(xb)
            loss = vae_loss(recon, xb, mu, log_var, cfg.kl_weight)
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
    [v2 改進2] DF 標準化：確保 OF 與 DF 在同等尺度上
    只 fit majority 的 DF，再 transform 全體
    """
    scaler = StandardScaler()
    scaler.fit(DF_train[ytr_bin == 0])
    return scaler.transform(DF_train), scaler.transform(DF_test)


def minority_label_from_train(y: pd.Series) -> str:
    return str(y.value_counts().idxmin())


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {"gmean": float(math.sqrt(tpr * tnr)), "recall_min": float(rec), "f1": float(f1)}


# ============================================================
# 核心流程
# ============================================================
def run_fold(train_path: Path, test_path: Path, seed: int, vae_cfg: VAEConfig) -> Dict[str, float]:
    """
    Baseline C v2 (VAE) 單折流程：
      1. 讀資料
      2. OF 標準化（scaler 只 fit majority）
      3. [v2] 自適應計算 latent_ratio / kl_weight / patience
      4. 訓練 VAE（只用 majority）→ 提取 mu 作為 DF
      5. [v2 改進2] DF 標準化（scaler 只 fit majority DF）
      6. 特徵串接：Z = [OF_norm | DF_norm]
      7. 調優 OCSVM（nu 含 0.2，gamma 加入 0.5/0.05）
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

    # [v2] 自適應超參數
    lat_ratio, kl_weight, patience = adaptive_vae_params(n_maj, input_dim, ir)
    vae_cfg_fold = VAEConfig(
        epochs       = vae_cfg.epochs,
        batch_size   = max(8, min(vae_cfg.batch_size, n_maj // 2)),
        lr           = vae_cfg.lr,
        weight_decay = vae_cfg.weight_decay,
        kl_weight    = kl_weight,
        latent_ratio = lat_ratio,
        patience     = patience,
    )

    print(
        f"      [自適應] n_maj={n_maj}, IR={ir:.1f} → "
        f"latent_ratio={lat_ratio}, kl_weight={kl_weight}, patience={patience}"
    )

    # 訓練 VAE（只用 majority）
    print(f"      [VAE] 訓練中...", end=" ", flush=True)
    vae_model, latent_dim = train_vae(Xtr_maj, vae_cfg_fold, seed=seed)
    print(f"完成 (latent_dim={latent_dim})")

    # 提取深度特徵（mu）
    print(f"      [VAE] 提取 DF (mu)...", end=" ", flush=True)
    DFtr_raw = encode_features(vae_model, Xtr)
    DFte_raw = encode_features(vae_model, Xte)
    print("完成")

    # [v2 改進2] DF 標準化
    DFtr, DFte = normalize_df(DFtr_raw, DFte_raw, ytr_bin)

    # 特徵串接：OF_norm + DF_norm
    Ztr = np.concatenate([Xtr, DFtr], axis=1)
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

    # [v2 改進3&4] nu 加入 0.2，gamma 加入 0.5/0.05
    for nu in [0.01, 0.05, 0.1, 0.2]:
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

    # 最終模型
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
        "kl_weight":    float(kl_weight),
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
        description="Baseline C v2 (VAE): OF_maj + DF_maj → OCC3（自適應 VAE + OCSVM）"
    )
    p.add_argument("--data_root",  type=str,   default=".",      help="專案根目錄")
    p.add_argument("--out_dir",    type=str,   default="results")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--vae_epochs", type=int,   default=100)
    p.add_argument("--vae_batch",  type=int,   default=32)
    p.add_argument("--vae_lr",     type=float, default=1e-3)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    root    = Path(args.data_root)
    out_dir = root / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vae_cfg = VAEConfig(epochs=args.vae_epochs, batch_size=args.vae_batch, lr=args.vae_lr)

    print("=" * 70)
    print("Baseline C v2 (VAE): OF_maj + DF_maj → OCC3（自適應 VAE concat OCSVM）")
    print("  [v2 改進1] 自適應 latent_ratio（依 n_maj/IR 動態壓縮 DF）")
    print("  [v2 改進2] DF 獨立標準化（只 fit majority DF，確保 OF/DF 同尺度）")
    print("  [v2 改進3] OCSVM gamma 加入 0.5/0.05（高維串接特徵的細緻搜尋）")
    print("  [v2 改進4] OCSVM nu 加入 0.2")
    print("  [DAE→VAE] noise_std → kl_weight（自適應 β 正則化）")
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
                m = run_fold(paths["tra"], paths["tst"], seed=args.seed, vae_cfg=vae_cfg)
                fold_metrics.append(m)
                per_fold_records.append({
                    "dataset": ds_key, "dataset_dir": ddir.name, "fold": fid,
                    "auc": m["auc"], "gmean": m["gmean"],
                    "recall_min": m["recall_min"], "f1": m["f1"],
                    "latent_dim": int(m["latent_dim"]),
                    "concat_dim": int(m["concat_dim"]),
                    "latent_ratio": m["latent_ratio"],
                    "kl_weight": m["kl_weight"],
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
        "dataset":    "ALL",
        "n_datasets": int(df_summary["dataset"].nunique()) if not df_summary.empty else 0,
        "n_folds":    len(df_all),
        "auc_mean":          float(df_all["auc"].mean())          if not df_all.empty else np.nan,
        "auc_std":           float(df_all["auc"].std(ddof=0))     if not df_all.empty else np.nan,
        "gmean_mean":        float(df_all["gmean"].mean())        if not df_all.empty else np.nan,
        "gmean_std":         float(df_all["gmean"].std(ddof=0))   if not df_all.empty else np.nan,
        "recall_min_mean":   float(df_all["recall_min"].mean())   if not df_all.empty else np.nan,
        "recall_min_std":    float(df_all["recall_min"].std(ddof=0)) if not df_all.empty else np.nan,
        "f1_mean":           float(df_all["f1"].mean())           if not df_all.empty else np.nan,
        "f1_std":            float(df_all["f1"].std(ddof=0))      if not df_all.empty else np.nan,
    }])

    xlsx_path = out_dir / "baseline_c_vae_v2_results.xlsx"
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
