# Baseline C v1：OF_maj + DF_maj → OCC3 (DAE + OneClassSVM)
#
# ══════════════════════════════════════════════════════════════════════
# 設計說明
# ══════════════════════════════════════════════════════════════════════
#
# 【Baseline C 定義】
#   - 輸入特徵：OF_maj（原始特徵）+ DF_maj（DAE 深度特徵）= 串接後的增強特徵
#   - 訓練資料：只用 majority class
#   - OCC 方法：OCC3 = OCSVM（搭配增強特徵）
#
# 【與 Baseline A / B 的關係】
#   - Baseline A：OF_maj → OCC1（原始特徵直接接 OCSVM）
#   - Baseline B：DF_maj → OCC2（DAE 深度特徵接 OCSVM）
#   - Baseline C：OF_maj + DF_maj → OCC3（原始特徵 concat 深度特徵後接 OCSVM）
#
# 【技術細節】
#   - DAE（Denoising Autoencoder）：與 Baseline B 完全相同的去噪自編碼器
#     * 架構：input → hidden1 → hidden2 → latent → hidden2 → hidden1 → input
#     * 正規化：LayerNorm（比 BatchNorm 更適合小資料集的推論）
#     * 激活：LeakyReLU(0.1)（避免死神經元）
#     * 訓練：epochs=80，early stopping (patience=15)，noise_std=0.01，weight_decay=1e-4
#     * latent_ratio=1.0（不壓縮，latent_dim = max(2, int(input_dim * ratio))）
#   - 特徵串接：Z_concat = [OF_maj | DF_maj]，維度 = 2 × input_dim
#   - OCSVM 調優：在訓練集內部切 25% val，搜尋最佳 (nu, gamma)
#   - StandardScaler：只在 majority 上 fit（與 Baseline B v5 修正一致）
#   - 評估：直接用 OCSVM.predict()（與 A / B 保持一致，不做門檻搜尋）
#
# 執行方式：
#   python 03v1_run_baselineC.py --data_root .
# 輸出：
#   ./results/baseline_c_v1_results.xlsx
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
# 隨機種子設置
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
# KEEL 格式檔案讀取器（與 Baseline B 相同）
# ============================================================
def load_keel_dat(dat_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """
    讀取 KEEL 格式的 .dat 檔案
    回傳：(X_df, y_series) - 特徵矩陣、類別標籤
    """
    dat_path = Path(dat_path)
    with dat_path.open("r", encoding="utf-8", errors="ignore") as f:
        raw = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("%")]

    data_idx = next((i for i, ln in enumerate(raw) if ln.lower() == "@data"), None)
    header = raw[:data_idx] if data_idx is not None else []
    data_lines = raw[data_idx + 1 :] if data_idx is not None else raw

    attr_names: List[str] = []
    output_name: str | None = None

    for ln in header:
        low = ln.lower()
        if low.startswith("@attribute"):
            parts = ln.split(None, 2)
            if len(parts) >= 2:
                attr_names.append(parts[1].strip())
        elif low.startswith("@outputs"):
            output_name = ln.split(None, 1)[1].strip()

    rows: List[List[str]] = []
    for dl in data_lines:
        parts = [p.strip() for p in dl.split(",")]
        rows.append(parts)

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
        if converted.isna().any():
            X[c] = df[c].astype(str)
        else:
            X[c] = converted

    return X, y


# ============================================================
# DAE 自編碼器定義（與 Baseline B 完全相同）
# ============================================================
def build_torch_autoencoder(input_dim: int, latent_dim: int):
    """
    建構兩層隱藏層的 MLP 去噪自編碼器（DAE）
    架構：
      Encoder: input_dim → hidden1 → hidden2 → latent_dim
      Decoder: latent_dim → hidden2 → hidden1 → input_dim
    使用 LayerNorm + LeakyReLU(0.1)
    """
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
            z = self.encoder(x)
            return self.decoder(z)

        def encode(self, x):
            return self.encoder(x)

    return AE()


@dataclass
class AEConfig:
    """DAE 訓練超參數（與 Baseline B 相同）"""
    epochs: int = 80           # 訓練輪數（含 early stopping）
    batch_size: int = 256      # 批次大小
    lr: float = 1e-3           # 學習率
    weight_decay: float = 1e-4 # L2 正則化
    noise_std: float = 0.01    # 去噪雜訊標準差（DAE 核心）
    latent_ratio: float = 1.0  # latent_dim = max(2, int(input_dim * ratio))（不壓縮）


def train_autoencoder(X_maj: np.ndarray, cfg: AEConfig, seed: int = 42):
    """
    訓練 DAE（只用 majority class 樣本）
    回傳：(model, latent_dim)
    """
    import torch
    import torch.nn as nn

    seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_dim = X_maj.shape[1]
    latent_dim = max(2, int(input_dim * cfg.latent_ratio))

    model = build_torch_autoencoder(input_dim, latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    Xt = torch.tensor(X_maj, dtype=torch.float32)
    n = Xt.size(0)

    best_loss = float("inf")
    patience = 15
    patience_counter = 0
    best_state = None

    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        batch_count = 0

        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            xb = Xt[idx].to(device)
            # DAE：加入高斯雜訊後學習重構原始乾淨資料
            xb_in = xb + cfg.noise_std * torch.randn_like(xb) if cfg.noise_std > 0 else xb
            recon = model(xb_in)
            loss = loss_fn(recon, xb)  # 目標是重構乾淨的 xb，而非加噪的 xb_in
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / max(batch_count, 1)

        if avg_loss < best_loss - 1e-6:
            best_loss = avg_loss
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if best_state is not None:
                    model.load_state_dict(best_state)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model, latent_dim


def encode_features(model, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """
    使用已訓練的 DAE 提取深度特徵 DF
    回傳：深度特徵矩陣 (n_samples, latent_dim)
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    outs: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = torch.tensor(X[i : i + batch_size], dtype=torch.float32).to(device)
            outs.append(model.encode(xb).cpu().numpy())
    return np.vstack(outs)


# ============================================================
# 資料夾工具 & 前處理
# ============================================================
FOLD_RE = re.compile(r"-(\d+)(tra|tst)\.dat$", re.IGNORECASE)


def discover_folds(dataset_dir: Path) -> Dict[int, Dict[str, Path]]:
    """自動發現資料夾中的所有 fold（訓練/測試分割對）"""
    folds: Dict[int, Dict[str, Path]] = {}
    for p in dataset_dir.glob("*.dat"):
        m = FOLD_RE.search(p.name)
        if not m:
            continue
        fid = int(m.group(1))
        split = m.group(2).lower()
        folds.setdefault(fid, {})[split] = p
    folds = {k: v for k, v in folds.items() if "tra" in v and "tst" in v}
    return dict(sorted(folds.items(), key=lambda x: x[0]))


def preprocess_train_test(
    X_train_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    y_train_bin: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    前處理訓練和測試資料
    ★ StandardScaler 只在 majority（y_train_bin==0）上 fit，再 transform 全體
      - 確保特徵空間以 majority 分布為中心
      - 符合「只用 majority 訓練」的 OCC 設計原則
    """
    all_df = pd.concat([X_train_df, X_test_df], axis=0, ignore_index=True)
    all_df = pd.get_dummies(all_df, drop_first=False)

    X_train_raw = all_df.iloc[: len(X_train_df), :].values.astype(np.float32)
    X_test_raw  = all_df.iloc[len(X_train_df) :, :].values.astype(np.float32)

    # 只用 majority fit scaler
    X_train_maj_raw = X_train_raw[y_train_bin == 0]
    scaler = StandardScaler()
    scaler.fit(X_train_maj_raw)
    X_train = scaler.transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)
    return X_train, X_test


def minority_label_from_train(y_train: pd.Series) -> str:
    """從訓練集確定少數類標籤（樣本最少的類別）"""
    vc = y_train.value_counts()
    return str(vc.idxmin())


# ============================================================
# 評估指標計算
# ============================================================
def compute_metrics(y_true_bin: np.ndarray, y_pred_anom: np.ndarray) -> Dict[str, float]:
    """
    計算 G-mean、Recall(min)、F1
    y_true_bin: 1=minority(異常), 0=majority(正常)
    y_pred_anom: 1=預測為異常, 0=預測為正常
    """
    rec = recall_score(y_true_bin, y_pred_anom, zero_division=0)
    f1  = f1_score(y_true_bin, y_pred_anom, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_anom, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    gmean = math.sqrt(tpr * tnr)
    return {"gmean": float(gmean), "recall_min": float(rec), "f1": float(f1)}


# ============================================================
# 核心流程：每個 fold 的完整執行
# ============================================================
def run_fold(train_path: Path, test_path: Path, seed: int, ae_cfg: AEConfig) -> Dict[str, float]:
    """
    Baseline C v1 的單折執行流程：
      1. 讀取資料，確定 minority / majority
      2. 前處理：one-hot + StandardScaler（只在 majority fit）
      3. 訓練 DAE（只用 majority）→ 提取 DF
      4. 特徵串接：Z_concat = [OF | DF]（维度 = input_dim + latent_dim）
      5. 在訓練集內切 75% fit / 25% val，搜尋最佳 (nu, gamma)
      6. 用最佳參數在全體訓練 majority 的 Z_concat 上重訓 OCSVM
      7. 測試集評估：直接用 OCSVM.predict()（不做門檻搜尋，與 A/B 一致）
    """
    # ── 步驟 1：讀取資料 ─────────────────────────────────────────────
    Xtr_df, ytr = load_keel_dat(train_path)
    Xte_df, yte = load_keel_dat(test_path)

    minor = minority_label_from_train(ytr)
    ytr_bin = (ytr.astype(str) == minor).astype(int).values
    yte_bin = (yte.astype(str) == minor).astype(int).values

    # ── 步驟 2：前處理（scaler 只 fit majority） ────────────────────
    Xtr, Xte = preprocess_train_test(Xtr_df, Xte_df, ytr_bin)

    Xtr_maj = Xtr[ytr_bin == 0]   # OF_maj（訓練集 majority 的原始特徵）
    if Xtr_maj.shape[0] < 10:
        raise ValueError(f"訓練集中多數類樣本過少: maj={Xtr_maj.shape[0]} ({train_path})")

    # ── 步驟 3：訓練 DAE（只用 majority）→ 提取 DF ──────────────────
    print(f"      [DAE] 訓練中 (majority={Xtr_maj.shape[0]} 樣本)...", end=" ", flush=True)
    ae_model, latent_dim = train_autoencoder(Xtr_maj, ae_cfg, seed=seed)
    print(f"完成 (latent_dim={latent_dim})")

    print(f"      [DAE] 提取深度特徵 DF...", end=" ", flush=True)
    DFtr = encode_features(ae_model, Xtr)   # DF for 訓練集全體（含 minority，val 調參需要）
    DFte = encode_features(ae_model, Xte)   # DF for 測試集
    print("完成")

    # ── 步驟 4：特徵串接 OF + DF ─────────────────────────────────────
    # Baseline C 的核心：同時利用原始特徵與深度特徵
    # 訓練集：Z_concat shape = (n_train, input_dim + latent_dim)
    # 測試集：同上維度
    Ztr = np.concatenate([Xtr, DFtr], axis=1)   # OF_tr || DF_tr
    Zte = np.concatenate([Xte, DFte], axis=1)   # OF_te || DF_te

    concat_dim = Ztr.shape[1]
    print(f"      [串接] OF({Xtr.shape[1]}) + DF({latent_dim}) = {concat_dim} 維特徵")

    # ── 步驟 5：訓練集內部切 fit / val，搜尋最佳 (nu, gamma) ─────────
    tr_idx, va_idx = train_test_split(
        np.arange(len(ytr_bin)),
        test_size=0.25,
        random_state=seed,
        stratify=ytr_bin,
    )
    Z_fit_maj = Ztr[tr_idx][ytr_bin[tr_idx] == 0]   # fit：只用 majority 的串接特徵
    Z_val     = Ztr[va_idx]
    y_val     = ytr_bin[va_idx]

    print(f"      [OCSVM] 調優 (nu, gamma)...", end=" ", flush=True)
    best_auc   = -1.0
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
                best_auc   = auc_val
                best_params = {"nu": nu, "gamma": gamma}

    print(f"完成 (最佳: nu={best_params['nu']}, gamma={best_params['gamma']})")

    # ── 步驟 6：用最佳參數重訓最終模型（用全部訓練 majority 的串接特徵）──
    print(f"      [OCSVM] 訓練最終模型...", end=" ", flush=True)
    clf_final = OneClassSVM(kernel="rbf", nu=best_params["nu"], gamma=best_params["gamma"])
    clf_final.fit(Ztr[ytr_bin == 0])   # 全體訓練 majority 的 Z_concat
    print("完成")

    # ── 步驟 7：測試集評估 ────────────────────────────────────────────
    # 與 Baseline A / B 一致：直接用 predict()（不做門檻搜尋）
    print(f"      [評估] 測試集...", end=" ", flush=True)
    test_anom  = -clf_final.decision_function(Zte)
    auc_test   = roc_auc_score(yte_bin, test_anom)

    # predict(): inlier=+1, outlier=-1 → outlier 視為 minority(1)
    y_pred_raw = clf_final.predict(Zte)
    y_pred     = (y_pred_raw == -1).astype(int)

    test_m = compute_metrics(yte_bin, y_pred)
    print("完成")

    return {
        "auc":         float(auc_test),
        "gmean":       float(test_m["gmean"]),
        "recall_min":  float(test_m["recall_min"]),
        "f1":          float(test_m["f1"]),
        "latent_dim":  float(latent_dim),
        "concat_dim":  float(concat_dim),
        "ocsvm_nu":    float(best_params["nu"]),
        "ocsvm_gamma": best_params["gamma"],
        "val_auc":     float(best_auc),
    }


# ============================================================
# 資料集定義（與 Baseline A / B 相同）
# ============================================================
DATASETS = {
    # ── 原始三個資料集 ──────────────────────────────────────────────
    "ecoli-0137_vs_26":         "ecoli-0-1-3-7_vs_2-6-5-fold",
    "glass-01236_vs_456":       "glass-0-1-2-3_vs_4-5-6-5-fold",
    "yeast-05679_vs_45":        "yeast-0-5-6-7-9_vs_4-5-fold",
    # ── 低 IR 對照組 ─────────────────────────────────────────────────
    "glass1":                   "glass1-5-fold",
    "yeast1":                   "yeast1-5-fold",
    # ── 新增三個資料集 ────────────────────────────────────────────────
    "cleveland-0_vs_4":         "cleveland-0_vs_4-5-fold",
    "yeast-2_vs_8":             "yeast-2_vs_8-5-fold",
    "abalone-17_vs_7-8-9-10":  "abalone-17_vs_7-8-9-10-5-fold",
}


# ============================================================
# 命令列介面
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Baseline C v1: OF_maj + DF_maj → OCC3 (DAE + OCSVM)"
    )
    p.add_argument("--data_root",      type=str,   default=".",   help="專案根目錄（包含 ./data/...）")
    p.add_argument("--out_dir",        type=str,   default="results", help="輸出資料夾")
    p.add_argument("--seed",           type=int,   default=42,    help="隨機種子")
    p.add_argument("--ae_epochs",      type=int,   default=80,    help="DAE 訓練輪數（含 early stopping）")
    p.add_argument("--ae_batch",       type=int,   default=256,   help="DAE 批次大小")
    p.add_argument("--ae_lr",          type=float, default=1e-3,  help="DAE 學習率")
    p.add_argument("--ae_noise",       type=float, default=0.01,  help="DAE 加噪雜訊標準差")
    p.add_argument("--ae_latent_ratio",type=float, default=1.0,   help="latent_dim = max(2, int(input_dim * ratio))，預設 1.0")
    return p.parse_args()


# ============================================================
# 主程式
# ============================================================
def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    root = Path(args.data_root)
    if not root.exists():
        raise FileNotFoundError(f"資料根目錄不存在: {root}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ae_cfg = AEConfig(
        epochs=args.ae_epochs,
        batch_size=args.ae_batch,
        lr=args.ae_lr,
        noise_std=args.ae_noise,
        latent_ratio=args.ae_latent_ratio,
    )

    print("=" * 70)
    print("Baseline C v1: OF_maj + DF_maj → OCC3 (DAE + OCSVM)")
    print("  特徵策略: OF（原始特徵）concat DF（DAE 深度特徵）= 增強特徵")
    print("  訓練資料: 只用 majority class")
    print("  OCC 方法: OneClassSVM (RBF kernel)")
    print("  DAE 架構: 兩層隱藏層 + LayerNorm + LeakyReLU(0.1)")
    print(f"  DAE 設定: epochs={ae_cfg.epochs}, batch={ae_cfg.batch_size}, lr={ae_cfg.lr}, "
          f"noise_std={ae_cfg.noise_std}, weight_decay={ae_cfg.weight_decay}, latent_ratio={ae_cfg.latent_ratio}")
    print("  OCSVM 搜尋: nu ∈ {0.01, 0.05, 0.1} × gamma ∈ {scale, 1.0, 0.1, 0.01, 0.001}")
    print("  評估方式: OCSVM.predict()（與 Baseline A/B 一致，無門檻搜尋）")
    print("=" * 70)
    print(f"資料根目錄: {root.resolve()}")
    print(f"輸出資料夾: {out_dir.resolve()}")
    print(f"隨機種子:   {args.seed}")
    print()

    per_fold_records: List[Dict[str, object]] = []
    summary_records: List[Dict[str, object]] = []

    for ds_key, ds_folder in DATASETS.items():
        ddir = root / "data" / ds_folder
        if not ddir.exists():
            print(f"[警告] 資料集資料夾不存在: {ddir}", file=sys.stderr)
            continue

        folds = discover_folds(ddir)
        if not folds:
            print(f"[警告] 未在資料夾中發現 fold: {ddir}", file=sys.stderr)
            continue

        print(f"\n【資料集】{ds_key} ({ddir.name}) | {len(folds)} 個 fold")
        print("-" * 70)
        fold_metrics = []

        for fid, split_map in folds.items():
            train_path = split_map["tra"]
            test_path  = split_map["tst"]

            try:
                print(f"  Fold {fid:02d}:")
                m = run_fold(train_path, test_path, seed=args.seed, ae_cfg=ae_cfg)
                fold_metrics.append(m)

                per_fold_records.append({
                    "dataset":      ds_key,
                    "dataset_dir":  ddir.name,
                    "fold":         fid,
                    "auc":          m["auc"],
                    "gmean":        m["gmean"],
                    "recall_min":   m["recall_min"],
                    "f1":           m["f1"],
                    "latent_dim":   int(m["latent_dim"]),
                    "concat_dim":   int(m["concat_dim"]),
                    "ocsvm_nu":     m["ocsvm_nu"],
                    "ocsvm_gamma":  m["ocsvm_gamma"],
                    "val_auc":      m["val_auc"],
                })

                print(
                    f"    結果: AUC={m['auc']:.4f} | G-mean={m['gmean']:.4f} | "
                    f"Recall(min)={m['recall_min']:.4f} | F1={m['f1']:.4f}\n"
                    f"    參數: concat_dim={int(m['concat_dim'])} | "
                    f"nu={m['ocsvm_nu']} | gamma={m['ocsvm_gamma']} | val_auc={m['val_auc']:.4f}"
                )

            except Exception as e:
                print(f"    [錯誤] Fold {fid:02d} 執行失敗: {e}", file=sys.stderr)

        if fold_metrics:
            keys = ["auc", "gmean", "recall_min", "f1"]
            avg = {k: float(np.mean([fm[k] for fm in fold_metrics])) for k in keys}
            std = {k: float(np.std([fm[k]  for fm in fold_metrics])) for k in keys}

            summary_records.append({
                "dataset":          ds_key,
                "dataset_dir":      ddir.name,
                "n_folds":          len(fold_metrics),
                "auc_mean":         avg["auc"],
                "auc_std":          std["auc"],
                "gmean_mean":       avg["gmean"],
                "gmean_std":        std["gmean"],
                "recall_min_mean":  avg["recall_min"],
                "recall_min_std":   std["recall_min"],
                "f1_mean":          avg["f1"],
                "f1_std":           std["f1"],
            })

            print(
                f"\n  【{len(fold_metrics)} 折平均】\n"
                f"    AUC:        {avg['auc']:.4f} ± {std['auc']:.4f}\n"
                f"    G-mean:     {avg['gmean']:.4f} ± {std['gmean']:.4f}\n"
                f"    Recall(min):{avg['recall_min']:.4f} ± {std['recall_min']:.4f}\n"
                f"    F1:         {avg['f1']:.4f} ± {std['f1']:.4f}"
            )

    # ============================================================
    # 輸出 Excel 結果檔
    # ============================================================
    df_all     = pd.DataFrame(per_fold_records) if per_fold_records else pd.DataFrame()
    df_summary = pd.DataFrame(summary_records)  if summary_records  else pd.DataFrame()

    df_overall = pd.DataFrame([{
        "dataset":          "ALL",
        "n_datasets":       int(df_summary["dataset"].nunique()) if not df_summary.empty else 0,
        "n_folds":          int(len(df_all)),
        "auc_mean":         float(df_all["auc"].mean())         if not df_all.empty else np.nan,
        "auc_std":          float(df_all["auc"].std(ddof=0))    if not df_all.empty else np.nan,
        "gmean_mean":       float(df_all["gmean"].mean())       if not df_all.empty else np.nan,
        "gmean_std":        float(df_all["gmean"].std(ddof=0))  if not df_all.empty else np.nan,
        "recall_min_mean":  float(df_all["recall_min"].mean())  if not df_all.empty else np.nan,
        "recall_min_std":   float(df_all["recall_min"].std(ddof=0)) if not df_all.empty else np.nan,
        "f1_mean":          float(df_all["f1"].mean())          if not df_all.empty else np.nan,
        "f1_std":           float(df_all["f1"].std(ddof=0))     if not df_all.empty else np.nan,
    }])

    xlsx_path = out_dir / "baseline_c_v1_results.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_all.to_excel(writer,     sheet_name="per_fold",     index=False)
        df_summary.to_excel(writer, sheet_name="summary",      index=False)
        df_overall.to_excel(writer, sheet_name="overall_mean", index=False)

    if summary_records:
        print("\n" + "=" * 70)
        print("【最終總結】各資料集 5-fold 平均結果")
        print("=" * 70)
        for row in summary_records:
            print(
                f"{row['dataset']:30s} | "
                f"AUC={row['auc_mean']:.4f}±{row['auc_std']:.4f} | "
                f"G-mean={row['gmean_mean']:.4f}±{row['gmean_std']:.4f} | "
                f"Recall={row['recall_min_mean']:.4f}±{row['recall_min_std']:.4f} | "
                f"F1={row['f1_mean']:.4f}±{row['f1_std']:.4f}"
            )

    print("\n" + "=" * 70)
    print(f"✓ 結果已儲存至: {xlsx_path.resolve()}")
    print("=" * 70)
    print("  - baseline_c_v1_results.xlsx: Excel 彙整檔（per_fold / summary / overall_mean）")


if __name__ == "__main__":
    main()
