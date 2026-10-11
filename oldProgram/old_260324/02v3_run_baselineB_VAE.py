# Baseline B v3：DF_maj → OCC2 (VAE + OneClassSVM)
#
# ══════════════════════════════════════════════════════════════════════
# v3 VAE 版本說明（對應 DAE v3，架構相同，將 DAE 換成 VAE）
# ══════════════════════════════════════════════════════════════════════
#
# 【核心改進：自適應壓縮維度強化（同 DAE v3）】
#   根據 IR 與 n_maj 動態決定 latent_ratio：
#       - 若 n_maj < 50：ratio=0.5（樣本少，強壓縮防過擬合）
#       - 若 n_maj 中等（50~200）：根據 IR 判斷
#           * IR > 10：ratio=0.5  * 5 < IR ≤ 10：ratio=0.75  * IR ≤ 5：ratio=1.0
#       - 若 n_maj 大（> 200）：根據 IR 判斷
#           * IR > 20：ratio=0.5  * 5 < IR ≤ 20：ratio=0.75  * IR ≤ 5：ratio=1.0
#
# 【DAE → VAE 差異】
#   - noise_std（加噪輸入）→ kl_weight（KL 散度權重 β）
#   - Encoder 輸出 (mu, log_var)，訓練時透過重參數化採樣 z
#   - 推論時取 mu 作為確定性深度特徵（不採樣）
#   - 自適應 kl_weight：n_maj<50 → 5e-3；n_maj<200 → 2e-3；n_maj≥200 → 1e-3
#
# 【繼承 DAE v3 設定】
#   - 自適應 patience、latent_ratio（同 v3）
#   - OCSVM nu 範圍 {0.01, 0.05, 0.1}
#   - epochs=100, batch=32, weight_decay=1e-4
#   - LayerNorm、兩層隱藏層、StandardScaler 只 fit majority
#
# 執行：python 02v3_run_baselineB_VAE.py --data_root .
# 輸出：./results/baseline_b_vae_v3_results.xlsx
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
    header = raw[:data_idx] if data_idx is not None else []
    data_lines = raw[data_idx + 1:] if data_idx is not None else raw

    attr_names: List[str] = []
    output_name = None
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
        raise ValueError(f"未從檔案解析到任何資料列: {dat_path}")
    if not attr_names:
        attr_names = [f"c{i}" for i in range(len(rows[0]))]
    rows = [r for r in rows if len(r) == len(attr_names)]
    df = pd.DataFrame(rows, columns=attr_names)

    y_col = output_name if (output_name and output_name in df.columns) else df.columns[-1]
    y = df[y_col].astype(str)
    X = df.drop(columns=[y_col])
    for c in X.columns:
        converted = pd.to_numeric(X[c], errors="coerce")
        X[c] = converted if not converted.isna().any() else df[c].astype(str)
    return X, y


# ============================================================
# 自適應超參數計算（VAE 版：noise_std → kl_weight）
# ============================================================
def adaptive_vae_params(n_maj: int, input_dim: int, ir: float) -> Tuple[float, float, int]:
    """
    根據資料集特性動態決定 (latent_ratio, kl_weight, patience)

    latent_ratio 邏輯與 DAE v3 相同；
    kl_weight 對應 DAE v3 的 noise_std：樣本少 / IR 高 → 較大 kl_weight（加強正則化）

    參數：
      n_maj:     majority 訓練樣本數
      input_dim: 原始特徵維度
      ir:        Imbalance Ratio (n_maj / n_min)

    回傳：
      (latent_ratio, kl_weight, patience)
    """
    if n_maj < 50:
        # 樣本極少：強壓縮 + 大 KL 權重（強正則化）+ 長耐心
        latent_ratio = 0.5
        kl_weight    = 5e-3
        patience     = 20
    elif n_maj < 200:
        # 樣本中等
        if ir > 10:
            latent_ratio = 0.5
        elif ir > 5:
            latent_ratio = 0.75
        else:
            latent_ratio = 1.0
        kl_weight = 2e-3
        patience  = 15
    else:
        # 樣本充足
        if ir > 20:
            latent_ratio = 0.5
        elif ir > 5:
            latent_ratio = 0.75
        else:
            latent_ratio = 1.0
        kl_weight = 1e-3
        patience  = 15

    return latent_ratio, kl_weight, patience


# ============================================================
# VAE 模型建構
# ============================================================
def build_vae(input_dim: int, latent_dim: int):
    """
    兩層 hidden MLP VAE，使用 LayerNorm + LeakyReLU(0.1)。
    Encoder 輸出 (mu, log_var)；Decoder 以 z 重建輸入。
    """
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
            """回傳 mu（推論時作為確定性表示）"""
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
            recon = self.decoder(z)
            return recon, mu, log_var

    return VAE()


@dataclass
class VAEConfig:
    epochs: int         = 100        # 同 DAE v3
    batch_size: int     = 32         # 同 DAE v3（適合小資料集）
    lr: float           = 1e-3
    weight_decay: float = 1e-4       # 同 DAE v3
    # 以下三個由 adaptive_vae_params 在 run_fold 動態覆蓋
    kl_weight: float    = 1e-3
    latent_ratio: float = 0.75
    patience: int       = 15


def vae_loss(recon, x, mu, log_var, kl_weight: float):
    """重建損失（MSE）+ β * KL 散度"""
    import torch.nn.functional as F
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp()).mean()
    return recon_loss + kl_weight * kl


def train_vae(X_maj: np.ndarray, cfg: VAEConfig, seed: int = 42):
    """訓練 VAE（只用 majority class 樣本）"""
    import torch

    seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_dim  = X_maj.shape[1]
    latent_dim = max(2, int(input_dim * cfg.latent_ratio))

    model = build_vae(input_dim, latent_dim).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    Xt = torch.tensor(X_maj, dtype=torch.float32)
    n  = Xt.size(0)

    best_loss    = float("inf")
    patience_cnt = 0
    best_state   = None

    for _ in range(cfg.epochs):
        model.train()
        perm       = torch.randperm(n)
        epoch_loss = 0.0
        cnt        = 0
        for i in range(0, n, cfg.batch_size):
            xb = Xt[perm[i: i + cfg.batch_size]].to(device)
            recon, mu, log_var = model(xb)
            loss = vae_loss(recon, xb, mu, log_var, cfg.kl_weight)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item(); cnt += 1

        avg = epoch_loss / max(cnt, 1)
        if avg < best_loss - 1e-6:
            best_loss    = avg
            patience_cnt = 0
            best_state   = copy.deepcopy(model.state_dict())
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.patience:
                if best_state is not None:
                    model.load_state_dict(best_state)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, latent_dim


def encode_features(model, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """使用 VAE 提取深度特徵（取 mu，確定性推論）"""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = torch.tensor(X[i: i + batch_size], dtype=torch.float32).to(device)
            outs.append(model.encode(xb).cpu().numpy())
    return np.vstack(outs)


# ============================================================
# 前處理與工具函式
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
    X_train_df: pd.DataFrame,
    X_test_df:  pd.DataFrame,
    y_train_bin: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """StandardScaler 只在 majority fit"""
    all_df = pd.get_dummies(
        pd.concat([X_train_df, X_test_df], axis=0, ignore_index=True),
        drop_first=False
    )
    X_tr_raw = all_df.iloc[:len(X_train_df)].values.astype(np.float32)
    X_te_raw = all_df.iloc[len(X_train_df):].values.astype(np.float32)
    scaler   = StandardScaler()
    scaler.fit(X_tr_raw[y_train_bin == 0])
    return scaler.transform(X_tr_raw), scaler.transform(X_te_raw)


def minority_label_from_train(y: pd.Series) -> str:
    return str(y.value_counts().idxmin())


def metrics_from_pred(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {"gmean": float(math.sqrt(tpr * tnr)), "recall_min": float(rec), "f1": float(f1)}


# ============================================================
# 主流程：每個 fold
# ============================================================
def run_fold(train_path: Path, test_path: Path, seed: int, vae_cfg: VAEConfig) -> Dict[str, float]:
    """
    Baseline B v3 (VAE) 單折流程：
      1. 讀資料，確定 minority/majority
      2. 前處理（scaler 只 fit majority）
      3. 自適應計算 latent_ratio / kl_weight / patience
      4. 訓練 VAE（只用 majority）→ 提取 mu 作為深度特徵
      5. 訓練集切 75/25 → 搜尋最佳 (nu, gamma)
      6. 最終模型重訓（全部訓練 majority 特徵）
      7. 測試集評估（OCSVM.predict()）
    """
    Xtr_df, ytr = load_keel_dat(train_path)
    Xte_df, yte = load_keel_dat(test_path)

    minor   = minority_label_from_train(ytr)
    ytr_bin = (ytr.astype(str) == minor).astype(int).values
    yte_bin = (yte.astype(str) == minor).astype(int).values

    Xtr, Xte = preprocess_train_test(Xtr_df, Xte_df, ytr_bin)
    Xtr_maj  = Xtr[ytr_bin == 0]
    Xtr_min  = Xtr[ytr_bin == 1]

    if Xtr_maj.shape[0] < 5:
        raise ValueError(f"majority 樣本過少: {Xtr_maj.shape[0]}")

    n_maj     = Xtr_maj.shape[0]
    n_min     = Xtr_min.shape[0] if Xtr_min.shape[0] > 0 else 1
    ir        = n_maj / n_min
    input_dim = Xtr.shape[1]

    # 自適應超參數
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
        f"      [自適應超參數] n_maj={n_maj}, IR={ir:.1f} → "
        f"latent_ratio={lat_ratio}, kl_weight={kl_weight}, patience={patience}"
    )

    # 訓練 VAE
    print(f"      [VAE] 訓練中...", end=" ", flush=True)
    vae_model, latent_dim = train_vae(Xtr_maj, vae_cfg_fold, seed=seed)
    print(f"完成 (latent_dim={latent_dim})")

    # 提取深度特徵 DF（取 mu）
    Ztr = encode_features(vae_model, Xtr)
    Zte = encode_features(vae_model, Xte)

    # 訓練集切分調優
    tr_idx, va_idx = train_test_split(
        np.arange(len(ytr_bin)),
        test_size=0.25,
        random_state=seed,
        stratify=ytr_bin,
    )
    Z_fit_maj = Ztr[tr_idx][ytr_bin[tr_idx] == 0]
    Z_val     = Ztr[va_idx]
    y_val     = ytr_bin[va_idx]

    print(f"      [OCSVM] 調優 (nu, gamma)...", end=" ", flush=True)
    best_auc    = -1.0
    best_params = {"nu": 0.05, "gamma": "scale"}

    for nu in [0.01, 0.05, 0.1]:
        for gamma in ["scale", 1.0, 0.1, 0.01, 0.001]:
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

    # 重訓最終模型
    clf_final = OneClassSVM(kernel="rbf", nu=best_params["nu"], gamma=best_params["gamma"])
    clf_final.fit(Ztr[ytr_bin == 0])

    # 測試集評估
    test_anom  = -clf_final.decision_function(Zte)
    auc_test   = roc_auc_score(yte_bin, test_anom)
    y_pred     = (clf_final.predict(Zte) == -1).astype(int)
    test_m     = metrics_from_pred(yte_bin, y_pred)

    return {
        "auc":          float(auc_test),
        "gmean":        float(test_m["gmean"]),
        "recall_min":   float(test_m["recall_min"]),
        "f1":           float(test_m["f1"]),
        "latent_dim":   float(latent_dim),
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
    p = argparse.ArgumentParser(description="Baseline B v3 (VAE): DF_maj → OCC2（自適應 VAE + OCSVM）")
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
    print("Baseline B v3 (VAE): DF_maj → OCC2（自適應 VAE 壓縮 + OCSVM）")
    print("  [VAE 自適應] latent_ratio：IR ≤ 5 → 1.0 | 5 < IR ≤ 10/20 → 0.75 | IR > 10/20 → 0.5")
    print("  [VAE 自適應] kl_weight：n_maj<50 → 5e-3 | n_maj<200 → 2e-3 | n_maj≥200 → 1e-3")
    print("  [OCSVM nu] {0.01, 0.05, 0.1} | epochs=100, batch=32, weight_decay=1e-4")
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
                "auc_mean": avg["auc"], "auc_std": std["auc"],
                "gmean_mean": avg["gmean"], "gmean_std": std["gmean"],
                "recall_min_mean": avg["recall_min"], "recall_min_std": std["recall_min"],
                "f1_mean": avg["f1"], "f1_std": std["f1"],
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

    xlsx_path = out_dir / "baseline_b_vae_v3_results.xlsx"
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
