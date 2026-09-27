"""
01v1_run_baselineB.py — Baseline B
方法：DF_maj → OneClassSVM，5-fold 交叉驗證
評估指標：AUC / Recall(min) / F1 / G-mean

執行：python 01v1_run_baselineB.py --data_root .
輸出：./results/baseline_b_v1_results.xlsx
"""

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
# 工具函式
# ============================================================

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_keel_dat(dat_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """讀取 KEEL .dat 檔案，回傳 (X_df, y_series)"""
    with Path(dat_path).open("r", encoding="utf-8", errors="ignore") as f:
        raw = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("%")]

    data_idx = next((i for i, ln in enumerate(raw) if ln.lower() == "@data"), None)
    header = raw[:data_idx] if data_idx is not None else []
    data_lines = raw[data_idx + 1:] if data_idx is not None else raw

    attr_names: List[str] = []
    output_name: str | None = None

    for ln in header:
        low = ln.lower()
        if low.startswith("@attribute"):
            parts = ln.split(None, 2)
            if len(parts) >= 2:
                attr_names.append(parts[1].strip())
        elif low.startswith("@output"):   # 同時處理 @output 和 @outputs
            output_name = ln.split(None, 1)[1].strip().split(",")[0].strip()

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
        if converted.isna().any():
            X[c] = df[c].astype(str)
        else:
            X[c] = converted

    return X, y


def minority_label(y: pd.Series) -> str:
    """回傳樣本數最少的類別標籤"""
    return str(y.value_counts().idxmin())


# ============================================================
# 前處理
# ============================================================

def preprocess(
    X_train_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    y_train_bin: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    one-hot 編碼 + 標準化
    StandardScaler 只在 majority（y_train_bin==0）上 fit
    """
    all_df = pd.concat([X_train_df, X_test_df], axis=0, ignore_index=True)
    all_df = pd.get_dummies(all_df, drop_first=False)

    n_tr = len(X_train_df)
    X_train_raw = all_df.iloc[:n_tr].values.astype(np.float32)
    X_test_raw  = all_df.iloc[n_tr:].values.astype(np.float32)

    scaler = StandardScaler()
    scaler.fit(X_train_raw[y_train_bin == 0])   # 只用 majority fit
    return scaler.transform(X_train_raw), scaler.transform(X_test_raw)


# ============================================================
# 自編碼器
# ============================================================

@dataclass
class AEConfig:
    epochs: int       = 80
    batch_size: int   = 256
    lr: float         = 1e-3
    weight_decay: float = 1e-4
    noise_std: float  = 0.01
    latent_ratio: float = 1.0   # latent_dim = max(2, int(input_dim * ratio))


def build_autoencoder(input_dim: int, latent_dim: int):
    """兩層 hidden MLP 自編碼器，使用 LayerNorm + LeakyReLU(0.1)"""
    import torch.nn as nn  # type: ignore

    h1 = max(16, min(128, input_dim * 4))
    h2 = max(8,  min(64,  input_dim * 2))

    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, h1), nn.LayerNorm(h1), nn.LeakyReLU(0.1),
                nn.Linear(h1, h2),        nn.LayerNorm(h2), nn.LeakyReLU(0.1),
                nn.Linear(h2, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, h2), nn.LeakyReLU(0.1),
                nn.Linear(h2, h1),         nn.LeakyReLU(0.1),
                nn.Linear(h1, input_dim),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

        def encode(self, x):
            return self.encoder(x)

    return AE()


def train_autoencoder(X_maj: np.ndarray, cfg: AEConfig, seed: int = 42):
    """只用 majority 訓練自編碼器，含 early stopping（patience=15）"""
    import torch
    import torch.nn as nn  # type: ignore

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dim = X_maj.shape[1]
    latent_dim = max(2, int(input_dim * cfg.latent_ratio))

    model = build_autoencoder(input_dim, latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    Xt = torch.tensor(X_maj, dtype=torch.float32)
    n = Xt.size(0)

    best_loss, patience_counter, best_state = float("inf"), 0, None

    for _ in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss, batch_count = 0.0, 0

        for i in range(0, n, cfg.batch_size):
            xb = Xt[perm[i: i + cfg.batch_size]].to(device)
            xb_in = xb + cfg.noise_std * torch.randn_like(xb) if cfg.noise_std > 0 else xb
            loss = loss_fn(model(xb_in), xb)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item(); batch_count += 1

        avg_loss = epoch_loss / max(batch_count, 1)
        if avg_loss < best_loss - 1e-6:
            best_loss = avg_loss
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= 15:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, latent_dim


def encode_features(model, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """使用訓練好的 AE 提取深度特徵"""
    import torch  # type: ignore

    device = "cuda" if torch.cuda.is_available() else "cpu"
    outs: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = torch.tensor(X[i: i + batch_size], dtype=torch.float32).to(device)
            outs.append(model.encode(xb).cpu().numpy())
    return np.vstack(outs)


# ============================================================
# 評估指標
# ============================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """計算 G-mean、Recall(min)、F1"""
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr   = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    gmean = math.sqrt(tpr * tnr)
    return {"gmean": float(gmean), "recall_min": float(rec), "f1": float(f1)}


# ============================================================
# Fold 發現
# ============================================================

FOLD_RE = re.compile(r"-(\d+)(tra|tst)\.dat$", re.IGNORECASE)


def discover_folds(dataset_dir: Path) -> Dict[int, Dict[str, Path]]:
    """
    掃描資料夾，回傳 {fold_id: {'tra': Path, 'tst': Path}}
    檔名格式：{name}-{fold}tra.dat / {name}-{fold}tst.dat
    """
    folds: Dict[int, Dict[str, Path]] = {}
    for p in dataset_dir.glob("*.dat"):
        m = FOLD_RE.search(p.name)
        if not m:
            continue
        fid   = int(m.group(1))
        split = m.group(2).lower()
        folds.setdefault(fid, {})[split] = p
    folds = {k: v for k, v in folds.items() if "tra" in v and "tst" in v}
    return dict(sorted(folds.items()))


# ============================================================
# 單一 Fold 訓練與評估
# ============================================================

def run_fold(train_path: Path, test_path: Path, seed: int, ae_cfg: AEConfig) -> Dict[str, float]:
    """
    Baseline B 單一 fold 流程：
      1. 讀取資料，確定 minority label → binary label
      2. StandardScaler 只用 majority fit → transform 全體
      3. 訓練 AE（只用 majority）→ 提取深度特徵
      4. 訓練集切 75/25 做 OCSVM 超參數搜尋（以 val AUC 選最佳）
      5. 用最佳參數在全部 majority 重訓最終 OCSVM
      6. 測試集：直接用 predict()（與 Baseline A 一致，不做門檻調優）
      7. 回傳 AUC / G-mean / Recall(min) / F1
    """
    # 步驟 1：讀取資料
    Xtr_df, ytr = load_keel_dat(train_path)
    Xte_df, yte = load_keel_dat(test_path)

    minor   = minority_label(ytr)
    ytr_bin = (ytr.astype(str) == minor).astype(int).values
    yte_bin = (yte.astype(str) == minor).astype(int).values

    # 步驟 2：前處理（scaler 只看 majority）
    Xtr, Xte = preprocess(Xtr_df, Xte_df, ytr_bin)

    Xtr_maj = Xtr[ytr_bin == 0]
    if Xtr_maj.shape[0] < 10:
        raise ValueError(f"majority 樣本過少 ({Xtr_maj.shape[0]}) in {train_path}")

    # 步驟 3：訓練 AE & 提取深度特徵
    print(f"      [AE] 訓練 (majority={Xtr_maj.shape[0]})...", end=" ", flush=True)
    ae_model, latent_dim = train_autoencoder(Xtr_maj, ae_cfg, seed=seed)
    print(f"latent_dim={latent_dim}")

    print(f"      [AE] 提取深度特徵...", end=" ", flush=True)
    Ztr = encode_features(ae_model, Xtr)
    Zte = encode_features(ae_model, Xte)
    print("完成")

    # 步驟 4：OCSVM 超參數搜尋
    tr_idx, va_idx = train_test_split(
        np.arange(len(ytr_bin)), test_size=0.25,
        random_state=seed, stratify=ytr_bin,
    )
    Z_fit_maj = Ztr[tr_idx][ytr_bin[tr_idx] == 0]
    Z_val, y_val = Ztr[va_idx], ytr_bin[va_idx]

    print(f"      [OCSVM] 超參數搜尋...", end=" ", flush=True)
    best_auc, best_params = -1.0, {"nu": 0.05, "gamma": "scale"}

    for nu in [0.01, 0.05, 0.1]:
        for gamma in ["scale", 1.0, 0.1, 0.01, 0.001]:
            clf = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
            clf.fit(Z_fit_maj)
            val_score = -clf.decision_function(Z_val)
            try:
                auc_val = roc_auc_score(y_val, val_score)
            except Exception:
                auc_val = 0.0
            if auc_val > best_auc:
                best_auc = auc_val
                best_params = {"nu": nu, "gamma": gamma}

    print(f"nu={best_params['nu']}, gamma={best_params['gamma']}, val_auc={best_auc:.4f}")

    # 步驟 5：重訓最終 OCSVM（全部 majority）
    clf_final = OneClassSVM(kernel="rbf", nu=best_params["nu"], gamma=best_params["gamma"])
    clf_final.fit(Ztr[ytr_bin == 0])

    # 步驟 6：測試集評估
    test_score = -clf_final.decision_function(Zte)
    auc_test   = roc_auc_score(yte_bin, test_score)

    y_pred = (clf_final.predict(Zte) == -1).astype(int)   # -1=outlier → minority=1
    m = compute_metrics(yte_bin, y_pred)

    return {
        "auc":         float(auc_test),
        "gmean":       m["gmean"],
        "recall_min":  m["recall_min"],
        "f1":          m["f1"],
        "latent_dim":  float(latent_dim),
        "ocsvm_nu":    float(best_params["nu"]),
        "ocsvm_gamma": best_params["gamma"],
        "val_auc":     float(best_auc),
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
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline B: DF_maj → AE + OneClassSVM")
    p.add_argument("--data_root",        type=str,   default=".")
    p.add_argument("--out_dir",          type=str,   default="results")
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--ae_epochs",        type=int,   default=80)
    p.add_argument("--ae_batch",         type=int,   default=256)
    p.add_argument("--ae_lr",            type=float, default=1e-3)
    p.add_argument("--ae_noise",         type=float, default=0.01)
    p.add_argument("--ae_latent_ratio",  type=float, default=1.0,
                   help="latent_dim = max(2, int(input_dim * ratio))，預設 1.0 不壓縮")
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
    print("Baseline B: DF_maj → OneClassSVM (AE + OCSVM)")
    print(f"  AE: epochs={ae_cfg.epochs}, batch={ae_cfg.batch_size}, "
          f"lr={ae_cfg.lr}, noise={ae_cfg.noise_std}, latent_ratio={ae_cfg.latent_ratio}")
    print(f"  data_root={root.resolve()}")
    print(f"  out_dir={out_dir.resolve()}, seed={args.seed}")
    print("=" * 70)

    per_fold_records: List[Dict] = []
    summary_records:  List[Dict] = []

    for ds_key, ds_folder in DATASETS.items():
        ddir = root / "data" / ds_folder
        if not ddir.exists():
            print(f"[警告] 找不到資料夾: {ddir}", file=sys.stderr)
            continue

        folds = discover_folds(ddir)
        if not folds:
            print(f"[警告] 未找到 fold 檔案: {ddir}", file=sys.stderr)
            continue

        print(f"\n[資料集] {ds_key} | {len(folds)} folds")
        print("-" * 70)
        fold_metrics: List[Dict] = []

        for fid, split_map in folds.items():
            print(f"  Fold {fid:02d}:")
            try:
                m = run_fold(split_map["tra"], split_map["tst"], seed=args.seed, ae_cfg=ae_cfg)
                fold_metrics.append(m)
                per_fold_records.append({
                    "dataset": ds_key, "dataset_dir": ddir.name, "fold": fid,
                    **{k: (int(v) if k == "latent_dim" else v) for k, v in m.items()},
                })
                print(f"    AUC={m['auc']:.4f} | G-mean={m['gmean']:.4f} | "
                      f"Recall={m['recall_min']:.4f} | F1={m['f1']:.4f}")
            except Exception as e:
                print(f"    [錯誤] {e}", file=sys.stderr)

        if not fold_metrics:
            continue

        keys = ["auc", "gmean", "recall_min", "f1"]
        avg = {k: float(np.mean([fm[k] for fm in fold_metrics])) for k in keys}
        std = {k: float(np.std( [fm[k] for fm in fold_metrics])) for k in keys}

        summary_records.append({
            "dataset": ds_key, "dataset_dir": ddir.name,
            "n_folds": len(fold_metrics),
            **{f"{k}_mean": avg[k] for k in keys},
            **{f"{k}_std":  std[k] for k in keys},
        })

        print(f"\n  [平均] AUC={avg['auc']:.4f}±{std['auc']:.4f} | "
              f"G-mean={avg['gmean']:.4f}±{std['gmean']:.4f} | "
              f"Recall={avg['recall_min']:.4f}±{std['recall_min']:.4f} | "
              f"F1={avg['f1']:.4f}±{std['f1']:.4f}")

    # ── 輸出 Excel ──────────────────────────────────────────────
    df_all     = pd.DataFrame(per_fold_records)  if per_fold_records  else pd.DataFrame()
    df_summary = pd.DataFrame(summary_records)   if summary_records   else pd.DataFrame()
    df_overall = pd.DataFrame([{
        "dataset":         "ALL",
        "n_datasets":      int(df_summary["dataset"].nunique()) if not df_summary.empty else 0,
        "n_folds":         int(len(df_all)),
        **({f"{k}_mean": float(df_all[k].mean()) for k in ["auc","gmean","recall_min","f1"]}
           if not df_all.empty else {}),
        **({f"{k}_std":  float(df_all[k].std(ddof=0)) for k in ["auc","gmean","recall_min","f1"]}
           if not df_all.empty else {}),
    }])

    xlsx_path = out_dir / "baseline_b_v1_results.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_all.to_excel(writer,     sheet_name="per_fold",     index=False)
        df_summary.to_excel(writer, sheet_name="summary",      index=False)
        df_overall.to_excel(writer, sheet_name="overall_mean", index=False)

    if summary_records:
        print("\n" + "=" * 70)
        print("[最終總結]")
        for row in summary_records:
            print(f"  {row['dataset']:30s} | "
                  f"AUC={row['auc_mean']:.4f}±{row['auc_std']:.4f} | "
                  f"G-mean={row['gmean_mean']:.4f}±{row['gmean_std']:.4f} | "
                  f"Recall={row['recall_min_mean']:.4f}±{row['recall_min_std']:.4f} | "
                  f"F1={row['f1_mean']:.4f}±{row['f1_std']:.4f}")

    print("\n" + "=" * 70)
    print(f"結果已儲存: {xlsx_path.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
