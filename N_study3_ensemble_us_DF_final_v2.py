"""
N_study3_ensemble_us_DF_final_v2.py
===================================
實驗三（Study Three）路線一／集合運算式 Ensemble Under-sampling（DF_maj）

v2 相對 v1 的修正（依兩份審查意見逐條處理，並標明哪些「不採納」及理由）
────────────────────────────────────────────────────────────────────────
[P0-1] 亂數對齊：v1 宣稱「Strategy=none 精準重現 B、單一方法精準重現 J」——不成立。
       B/J 只在 import 時做一次 torch.manual_seed(42)，AE 權重取決於
       「for ae_type → for config」已經消耗多少亂數。N 若只跑 1 AE × 1 config，
       VAE 的初始化點就與 J 不同。
       修正：SEED_MODE="stable_per_cell"（預設）——每個
       (dataset, fold, AE, config) 用 stable hash 產生固定 seed，
       在建模前 torch.manual_seed()。結果不依賴執行順序，也不再假裝能對上舊 J。
       若真的要重現舊 J 的亂數路徑，設 SEED_MODE="legacy_global"
       ＋ AE_TYPES=4 個 ＋ CONFIG_MODE="grid" ＋ 相同資料集集合（見 alignment_notes）。

[P0-2] Study 3 的 primary baseline 改成「N 內部的 ENN」，不是舊 J 的數字。
       同一支程式、同一顆 VAE、同一份 DF、同一個 fold → none/ENN/CNN/TL/S1~S4
       全部 paired。新增 effect_vs_ENN 與 dataset-level paired 檢定。

[P0-3] fallback 不再參與投票。v1 中 sampler 失敗會退化成「全部保留」並繼續投票，
       等於讓失敗者投「全留」，S1 會直接塌成 none。
       修正：STRICT_FALLBACK=True → 任一投票者 fallback，該
       (dataset, fold, AE, config) 的所有 vote 策略標記 invalid 並跳過；
       重疊度統計也排除該筆。

[P0-4] rate-matched 的亂數種子用了 Python 內建 hash(ds_name)，跨 process 不穩定
       （PYTHONHASHSEED）。改用 zlib.crc32 的 stable hash，並把 ae_type 納入種子。

[P1-1] 策略重新編號：S1_Union → S2_Majority → S3_Intersection
       （保守 → 中間 → 激進），S4 標記為 ablation，不與 S1~S3 同位階。

[P1-2] rate-matched 逐次保留（不再只存平均）。可算出
       「策略贏過同刪除率隨機的次數 / 總次數」與 random 自身的變異。

[P1-3] 保留遮罩與 DF 快取落地成 .npz（含檔案指紋），Route 2 可直接載入，
       不必重跑 sampler 與 AE——這才是會議講的「子集留下來給第二階段用」。

[P1-4] 重疊度加上比例欄（DelInterRate / DelUnionRate / DelVoteRate_*），
       不同 dataset 的 majority 規模不同，絕對筆數不能直接平均。

[P1-5] ΔAUC 改成「同 (Dataset, Fold, AE, Config, OCC) 內先配對相減，再平均」，
       不是先各自平均再相減（有 skip 時兩者不等價）。

[P1-6] dataset-level 統計：先把 5 folds 平均成每個 dataset 一個值，
       再算 win/tie/loss 與 paired Wilcoxon。不把 85 folds 當 85 個獨立樣本。

[P2-1] OCSVM 明寫 gamma="scale"（sklearn>=0.22 的預設值，數值與 A~M 完全相同，
       只是讓第三章有明確可寫的參數）。套件版本寫進 alignment_notes。

[P2-2] 新增 SCAN_ONLY / OVERLAP_ONLY 模式：可先只掃資料集特性（n / dim / IR）
       來挑代表性 pilot，或只跑重疊度不跑 OCC。

[P2-3] KEEL 欄位若有非數值型，v1（沿用 A~M）是 train/test 各自
       pd.Categorical().codes，可能造成 mapping 錯位。
       →「不修改 parse_keel_dat」（改了就與 A~M 不對齊），改為新增診斷：
         偵測到非數值欄位就在 console 與 alignment_notes 明確警告。

── 審查意見中「不採納」的部分，理由寫在這裡供論文與口試引用 ──
  ✗「不能把三份子集直接 vstack 串接」
     → v1 起就從未 concat。三個方法回傳的是同一份 majority 上的 boolean
       keep mask，策略是對【同一筆樣本】投票，每個樣本最多出現一次。
       該意見來自未讀到程式碼的推測，不適用。
  ✗「parse_keel_dat 改成 train fit / test transform 的 category mapping」
     → 正確，但會讓 N 的資料前處理與 A~M 全部不一致，Study 2 baseline 就不能比。
       改為診斷警告；若確認有 categorical 資料集，應【整批】修 A~M 再重跑。
  ✗「把 PR-AUC 加進 METRIC_COLS」
     → PR-AUC 在極度不平衡下確實比 ROC-AUC 有資訊量，但 METRIC_COLS 是
       A~M 與 L_merge 共用的 schema。改為獨立的 EXTRA_METRIC_COLS，
       只出現在本檔的分析分頁，ak_*_export 的欄位與 I/J/K 位元級相同。

════════════════════════════════════════════════════════════════════════
與 A~M 的對齊清單（這些刻意「不動」，動了 Study 2 baseline 就不能比）
────────────────────────────────────────────────────────────────────────
  ✔ parse_keel_dat()          與 A/B/C/I/J/K 逐字相同（含檔名 pattern）
  ✔ MinMax 政策               fit 只用 training majority；DF 側再以未清理
                              DF_maj fit 一次；OCC 不做二次 MinMax
  ✔ AE 架構與超參數           AEModel / VAEModel / epochs=100 / bs=64 / lr=1e-3
                              / DAE noise=0.1 / SAE sparsity=1e-3 / VAE beta=1.0
  ✔ Grid 定義                 n_layers=[1,2,3] × 7 bottleneck ratios = 21 configs
  ✔ OCC 超參數                OCSVM(nu=0.1, rbf) / LOF(k=min(20, n-1),
                              novelty=True, contamination=0.1)
                              / iForest(n_estimators=100, contamination=0.1, rs=42)
  ✔ 判定門檻                  training majority anomaly score 的第 90 百分位
  ✔ 評估指標                  AUC / F1 / Recall / G-mean（gmean_score 同一份）
  ✔ Sampler 超參數            ENN(k=3, kind_sel=all) / CNN(k=1, rs=42) / TL 預設
                              sampling_strategy="auto"
  ✔ 退化保護                  清理後 majority < 5 筆即跳過；LOF k 重算
  ✔ 統整 schema               COMPARISON_EXPORT_COLS 與 I/J/K 完全相同，
                              Strategy 寫進 Sampler 欄 → L_merge 只要加一行 SOURCES
  ✘ 亂數種子                  【刻意不同】見 P0-1。這是 v2 唯一與 A~M 不同的
                              方法論設定，且它讓結果更可重現，不是更少。
════════════════════════════════════════════════════════════════════════

輸出：results/Study3_ensemble_us_DF.xlsx（11 分頁）
      results/mask_cache/*.npz（Route 2 用）
      results/Study3_warnings.log
      results/dataset_profile.csv（SCAN_ONLY 模式）
"""

import re
import sys
import json
import zlib
import platform
import warnings
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from hashlib import sha1

import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score, confusion_matrix, average_precision_score,
)

import imblearn
from imblearn.under_sampling import (
    EditedNearestNeighbours, CondensedNearestNeighbour, TomekLinks,
)

import scipy
from scipy.stats import wilcoxon

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ─────────────────────────── 路徑設定 ────────────────────────────────────────
DATA_ROOT   = Path("data")
RESULTS_DIR = Path("results")
OUTPUT_FILE = RESULTS_DIR / "Study3_ensemble_us_DF.xlsx"
CACHE_DIR   = RESULTS_DIR / "mask_cache"
WARN_LOG    = RESULTS_DIR / "Study3_warnings.log"
PROFILE_CSV = RESULTS_DIR / "dataset_profile.csv"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# warning 不再整批吞掉：console 安靜，但全部寫進 log（final run 必須留痕）
warnings.simplefilter("always")
_WARN_BUFFER = []


def _warn_to_log(message, category, filename, lineno, file=None, line=None):
    _WARN_BUFFER.append(f"{category.__name__}: {message} ({Path(filename).name}:{lineno})")


warnings.showwarning = _warn_to_log

N_FOLDS = 5
EPS     = 1e-12

# ── AE 訓練超參數（與 B / C / J 完全一致，不可動）──
AE_EPOCHS     = 100
AE_BATCH_SIZE = 64
AE_LR         = 1e-3
DAE_NOISE     = 0.1
SAE_SPARSITY  = 1e-3
VAE_BETA      = 1.0

# ══════════════════════════ 執行模式 ═════════════════════════════════════════
# SCAN_ONLY    : 只掃資料集特性（n / dim / IR）輸出 dataset_profile.csv，不跑實驗。
#                用來「事先」挑代表性 pilot 資料集（低/中/高 IR 各 2 個），
#                而不是取排序後前六個。挑完把名稱填進 DATASET_WHITELIST。
# OVERLAP_ONLY : 跑 AE + 三個 sampler，只算重疊度，不跑 OCC。
#                先確認三個方法的刪除集合是否互補，再決定要不要跑完整實驗。
SCAN_ONLY    = False
OVERLAP_ONLY = False

RUN_MODE = "pilot"   # "pilot" / "full"

if RUN_MODE == "pilot":
    # ⚠️ 跑 SCAN_ONLY 之後把挑好的 6 個資料集填進來，不要留空用 DATASET_LIMIT 取前六個
    DATASET_WHITELIST = []
    DATASET_LIMIT     = 6
    AE_TYPES          = ["VAE"]
    FIXED_CONFIGS     = ["h1-1/1"]
    N_RM_REPEATS      = 10
else:
    DATASET_WHITELIST = []
    DATASET_LIMIT     = 0
    AE_TYPES          = ["VAE"]     # Study 3 的自變數是策略，不是 AE；4 AE 放 appendix
    FIXED_CONFIGS     = ["h1-1/1"]
    N_RM_REPEATS      = 30

CONFIG_MODE = "fixed"    # "fixed"（不做 per-dataset oracle）/ "grid"（比照 J，僅供對照）

# SEED_MODE：
#   "stable_per_cell" → 每個 (dataset, fold, AE, config) 用 stable hash 決定 seed。
#                       不依賴執行順序，重跑必定相同，但【不會】等於舊 B/J 的數值。
#   "legacy_global"   → 沿用 A~M 的做法（import 時 manual_seed(42) 一次）。
#                       只有在 AE_TYPES = 4 個、CONFIG_MODE="grid"、資料集集合相同時，
#                       才會重現 J 的亂數路徑。
SEED_MODE   = "stable_per_cell"
GLOBAL_SEED = 42

OCC_TYPES   = ["OCSVM", "LOF", "iForest"]
# ⚠️ METRIC_COLS 與 A~M / L_merge 共用，不可增刪
METRIC_COLS = ["AUC", "F1", "Recall", "G-mean"]
# 本檔額外指標（不進 ak_export，不影響對齊）
ENABLE_PR_AUC     = True
EXTRA_METRIC_COLS = ["PR-AUC"] if ENABLE_PR_AUC else []
ALL_METRIC_COLS   = METRIC_COLS + EXTRA_METRIC_COLS

ENABLE_RATE_MATCHED = True
RM_SEED             = 2025

# 任一投票者 fallback → 該 cell 的 vote 策略全部作廢（不讓失敗者投「全留」）
STRICT_FALLBACK = True

# Route 2 用的快取
CACHE_MASKS = True
CACHE_DF    = True      # 連 DF_maj_s / DF_min_s / DF_tst_s / y_tst 一起存，Route 2 免重訓 AE

if SEED_MODE == "legacy_global":
    torch.manual_seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)

# ══════════════════════════ 投票者與策略定義 ════════════════════════════════
BASE_METHODS = ["ENN", "CNN", "TL"]
ENN_K, CNN_K, CNN_SEED = 3, 1, 42

AUTO_THRESHOLD_SWEEP = False   # True → 產生 T1..TK 全門檻（刪除率掃描曲線用）

# name -> (family, voters, threshold, role)
#   family : baseline / single / vote
#   role   : control（對照組）/ primary（主要策略）/ ablation（消融）
def build_strategy_specs():
    specs = {"none": ("baseline", (), 0, "control")}
    for m in BASE_METHODS:
        specs[m] = ("single", (m,), 1, "control")

    K = len(BASE_METHODS)
    all_voters = tuple(BASE_METHODS)
    if AUTO_THRESHOLD_SWEEP:
        for t in range(1, K + 1):
            specs[f"T{t}_keep{t}of{K}"] = ("vote", all_voters, t, "primary")
    else:
        # 保守 → 中間 → 激進，編號與激進程度同向（v1 的 S1/S3/S2 順序容易誤讀）
        specs["S1_Union"]        = ("vote", all_voters, 1,          "primary")
        specs["S2_Majority"]     = ("vote", all_voters, K // 2 + 1, "primary")
        specs["S3_Intersection"] = ("vote", all_voters, K,          "primary")

    # ablation：排除 Study 2 中已知有害的 CNN。這是 post-hoc 提出的，
    # 論文必須標明它不是事前定義的主要策略。
    no_cnn = tuple(m for m in BASE_METHODS if m != "CNN")
    if len(no_cnn) >= 2:
        specs["S4_" + "_".join(no_cnn) + "_abl"] = ("vote", no_cnn, len(no_cnn), "ablation")
    return specs


STRATEGY_SPECS = build_strategy_specs()
STRATEGIES     = list(STRATEGY_SPECS.keys())
PRIMARY_BASELINE = "ENN"   # Study 3 的主要對照：ensemble 有沒有比 single best 好

# ── 統整 metadata ──
STUDY_ID           = "N"
METHOD_ID          = "N_DF_ENSEMBLE_US"
FEATURE_SET        = "DF_maj"
BASELINE_REF       = "N_internal_none_and_single"
OCC_SCOPE          = "all_three_occ"
SAMPLER_SCALE_MODE = "df_majority_minmax_once_before_sampler_and_occ"

# ⚠️ 與 I/J/K 的 COMPARISON_EXPORT_COLS 逐欄相同，L_merge 可直接讀
COMPARISON_EXPORT_COLS = [
    "Study", "Method", "FeatureSet", "Dataset", "AE", "Sampler", "OCC",
    "Config", "Fold", "ConfigPolicy", "MajKept", "MajRemoved", "RemovedRate",
    "SamplerStatus", "BaselineRef", "OCCScope", "SamplerScaleMode",
] + METRIC_COLS

# ── Grid（與 B/C/G/H/J 對齊；順序也相同，legacy_global 才能重現 J 的亂數路徑）──
N_LAYERS_LIST     = [1, 2, 3]
BOTTLENECK_RATIOS = {
    "1/4": 0.25, "1/3": 1/3, "1/2": 0.5, "1/1": 1.0,
    "2/1": 2.0,  "3/1": 3.0, "4/1": 4.0,
}
ALL_CONFIGS    = [f"h{nl}-{rl}" for nl in N_LAYERS_LIST for rl in BOTTLENECK_RATIOS]
ACTIVE_CONFIGS = ALL_CONFIGS if CONFIG_MODE == "grid" else [
    c for c in FIXED_CONFIGS if c in ALL_CONFIGS
]


# ─────────────────────────── 工具 ────────────────────────────────────────────
def safe_removed_rate(n_removed, n_kept):
    denom = int(n_removed) + int(n_kept)
    return float(n_removed / denom) if denom > 0 else 0.0


def parse_config_label(cfg_label):
    m = re.match(r"^h(\d+)-(.+)$", cfg_label)
    if not m:
        raise ValueError(f"無法解析 config: {cfg_label}")
    return int(m.group(1)), m.group(2)


def stable_hash(*parts):
    """跨 process 穩定的整數 hash（Python 內建 hash() 會因 PYTHONHASHSEED 變動）。"""
    s = "|".join(str(p) for p in parts)
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF


def cell_seed(ds_name, fold, ae_type, cfg_label):
    """每個 (dataset, fold, AE, config) 的固定 seed —— 不依賴執行順序。"""
    return (GLOBAL_SEED + stable_hash(ds_name, fold, ae_type, cfg_label)) % (2**31 - 1)


def file_fingerprint(path):
    """訓練檔指紋，寫進 mask cache；Route 2 載入時可驗證是同一份資料。"""
    return sha1(Path(path).read_bytes()).hexdigest()[:16]


# ─────────────────────────── Under-sampling keep mask ───────────────────────
def make_sampler(name):
    """Sampler 超參數與 I/J/K 逐字相同。"""
    if name == "ENN":
        return EditedNearestNeighbours(
            n_neighbors=ENN_K, kind_sel="all", sampling_strategy="auto")
    if name == "CNN":
        return CondensedNearestNeighbour(
            n_neighbors=CNN_K, random_state=CNN_SEED, sampling_strategy="auto")
    if name == "TL":
        return TomekLinks(sampling_strategy="auto")
    return None


def compute_keep_masks(DF_maj_s, DF_min_s, methods):
    """三個方法在同一份 DF_maj 上各自算出保留哪些樣本（boolean mask）。

    ★ 這裡不做任何 concat / vstack：策略是對【同一筆 majority 樣本】投票，
      每筆樣本在最終訓練集中最多出現一次，不會變成樣本加權。

    回傳：(masks, statuses, any_fallback)
    """
    n_maj = len(DF_maj_s)
    masks, statuses = {}, {}
    any_fallback = False

    DF_all_s = np.vstack([DF_maj_s, DF_min_s])
    y_all    = np.array([0] * n_maj + [1] * len(DF_min_s))

    for m in methods:
        sampler = make_sampler(m)
        if sampler is None:
            masks[m], statuses[m] = np.ones(n_maj, dtype=bool), "fallback_unknown_sampler"
            any_fallback = True
            continue
        try:
            sampler.fit_resample(DF_all_s, y_all)
            idx = sampler.sample_indices_
        except Exception as e:
            masks[m] = np.ones(n_maj, dtype=bool)
            statuses[m] = f"fallback_sampler_error:{type(e).__name__}"
            any_fallback = True
            continue

        if int(np.sum(idx >= n_maj)) != len(DF_min_s):
            masks[m], statuses[m] = np.ones(n_maj, dtype=bool), "fallback_minority_changed"
            any_fallback = True
            continue

        keep_local = idx[idx < n_maj]
        # 索引完整性檢查（審查意見的 assertions）
        assert len(keep_local) == len(np.unique(keep_local)), f"{m}: 保留索引有重複"
        assert keep_local.max(initial=-1) < n_maj, f"{m}: 保留索引超出 majority 範圍"

        keep = np.zeros(n_maj, dtype=bool)
        keep[keep_local] = True
        masks[m] = keep
        statuses[m] = "ok_removed" if (~keep).sum() > 0 else "ok_no_removed"

    return masks, statuses, any_fallback


def apply_strategy(masks, statuses, strategy_name, n_maj, any_fallback):
    """把 keep masks 依策略合併。回傳 (keep_mask, status, valid)。

    STRICT_FALLBACK=True 時，只要有投票者 fallback，vote 策略一律 invalid：
    讓失敗的 sampler 投「全部保留」會靜默改變 ensemble 的定義
    （S1_Union 會直接塌成 none）。
    """
    family, voters, threshold, role = STRATEGY_SPECS[strategy_name]

    if family == "baseline":
        return np.ones(n_maj, dtype=bool), "none_baseline", True

    if family == "single":
        m = voters[0]
        return masks[m], statuses[m], not statuses[m].startswith("fallback")

    bad = [v for v in voters if statuses[v].startswith("fallback")]
    if bad and STRICT_FALLBACK:
        return None, "invalid_voter_fallback:" + "/".join(bad), False

    votes_keep = np.zeros(n_maj, dtype=int)
    for v in voters:
        votes_keep += masks[v].astype(int)
    keep = votes_keep >= threshold
    status = "ok_removed" if (~keep).sum() > 0 else "ok_no_removed"
    if bad:
        status = "vote_with_fallback:" + "/".join(bad)
    return keep, status, True


def rate_matched_keep(n_maj, n_removed, rng):
    keep = np.ones(n_maj, dtype=bool)
    if n_removed <= 0:
        return keep
    keep[rng.choice(n_maj, size=int(n_removed), replace=False)] = False
    return keep


# ─────────────────────────── 重疊度分析 ─────────────────────────────────────
def jaccard(a_mask, b_mask):
    inter = np.logical_and(a_mask, b_mask).sum()
    union = np.logical_or(a_mask, b_mask).sum()
    return float(inter / union) if union > 0 else float("nan")


def collect_overlap_record(ds_name, ae_type, cfg_label, fold, masks, n_maj):
    """三方法【刪除集合】的重疊結構。絕對筆數與比例同時輸出。"""
    del_masks = {m: ~masks[m] for m in BASE_METHODS}
    rec = {"Dataset": ds_name, "AE": ae_type, "Config": cfg_label,
           "Fold": fold, "MajTotal": n_maj}

    for m in BASE_METHODS:
        n_del = int(del_masks[m].sum())
        rec[f"Del_{m}"]     = n_del
        rec[f"DelRate_{m}"] = float(n_del / n_maj) if n_maj else 0.0

    for a, b in itertools.combinations(BASE_METHODS, 2):
        rec[f"Jaccard_{a}_{b}"] = jaccard(del_masks[a], del_masks[b])

    stacked = np.vstack([del_masks[m] for m in BASE_METHODS])
    n_inter, n_union = int(np.all(stacked, 0).sum()), int(np.any(stacked, 0).sum())
    rec["DelInterAll"], rec["DelUnionAll"] = n_inter, n_union
    rec["DelInterRate"] = float(n_inter / n_maj) if n_maj else 0.0
    rec["DelUnionRate"] = float(n_union / n_maj) if n_maj else 0.0

    del_votes = stacked.sum(axis=0)
    for v in range(len(BASE_METHODS) + 1):
        c = int((del_votes == v).sum())
        rec[f"DelVotes_{v}"]     = c
        rec[f"DelVoteRate_{v}"]  = float(c / n_maj) if n_maj else 0.0
    return rec


# ─────────────────────────── AE 模型（與 B/C/J 逐字相同）────────────────────
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


def train_ae_and_get_extractor(ae_type, X_maj_s, n_layers, n_units, seed=None):
    """訓練 AE（只用 training majority）。

    seed 不是 None 時，在【建模前】固定亂數 → 該 (dataset, fold, AE, config)
    的 AE 權重與執行順序無關（P0-1）。
    """
    if seed is not None:
        torch.manual_seed(seed)

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
            _, z = model(xt)
            return z.numpy()
    return extract


# ─────────────────────────── 評估（與 A~M 一致）─────────────────────────────
def gmean_score(y_true, y_pred_binary):
    cm = confusion_matrix(y_true, y_pred_binary, labels=[1, 0])
    if cm.shape == (2, 2):
        tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return float(np.sqrt(sens * spec))
    return 0.0


def run_occ_eval(occ_type, feat_maj_s, feat_test_s, y_test, n_neighbors_cap, do_scale=False):
    """OCC 參數與 A/B/I/J/K 逐字相同。

    OCSVM 只多明寫 gamma="scale"：那本來就是 sklearn>=0.22 的預設值，
    數值與 A~M 完全相同，只是讓第三章有明確可寫的參數（口試會問）。
    分數方向統一：三個 OCC 都取 -decision_function，分數越高越異常。
    """
    if do_scale:
        scaler      = MinMaxScaler()
        feat_maj_s  = scaler.fit_transform(feat_maj_s)
        feat_test_s = scaler.transform(feat_test_s)

    if occ_type == "OCSVM":
        clf = OneClassSVM(nu=0.1, kernel="rbf", gamma="scale")
    elif occ_type == "LOF":
        clf = LocalOutlierFactor(n_neighbors=min(20, n_neighbors_cap),
                                 novelty=True, contamination=0.1)
    else:
        clf = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)

    clf.fit(feat_maj_s)
    scores_maj  = -clf.decision_function(feat_maj_s)
    scores_test = -clf.decision_function(feat_test_s)

    threshold = np.percentile(scores_maj, 90)
    y_pred    = (scores_test >= threshold).astype(int)

    try:
        auc = roc_auc_score(y_test, scores_test) if len(np.unique(y_test)) >= 2 else float("nan")
    except Exception:
        auc = float("nan")

    out = {
        "AUC": auc,
        "F1": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        "Recall": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "G-mean": gmean_score(y_test, y_pred),
    }
    if ENABLE_PR_AUC:
        try:
            out["PR-AUC"] = average_precision_score(y_test, scores_test)
        except Exception:
            out["PR-AUC"] = float("nan")
    return out


# ─────────────────────────── KEEL .dat 解析（與 A~M 逐字相同）───────────────
def parse_keel_dat(filepath, minority_label=None):
    """⚠️ 刻意與 A/B/C/I/J/K 逐字相同，不做任何「改良」。

    已知限制：若某資料集含 categorical 欄位，train / test 各自
    pd.Categorical().codes 可能產生不同 mapping。
    本檔不修（改了就與 Study 2 baseline 不對齊），改為在主流程診斷並警告；
    若真的偵測到，應整批修 A~M 後重跑，不是只修 N。
    """
    lines = Path(filepath).read_text(encoding="utf-8", errors="replace").splitlines()
    data_start, rows = False, []
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
    n_categorical = 0
    for col in feat_df.columns:
        conv = pd.to_numeric(feat_df[col], errors="coerce")
        if conv.isna().all():
            n_categorical += 1
            feat_df[col] = pd.Categorical(feat_df[col]).codes.astype(float)
        else:
            feat_df[col] = conv
    X = feat_df.values.astype(float)

    if minority_label is None:
        unique, counts = np.unique(y_raw, return_counts=True)
        minority_label = unique[np.argmin(counts)]

    y = (y_raw == minority_label).astype(int)
    return X, y, minority_label, n_categorical


# ─────────────────────────── 資料集掃描（挑代表性 pilot 用）─────────────────
def scan_datasets(dataset_dirs):
    """輸出每個資料集的 n / dim / IR，供事先挑選代表性 pilot（非取前六個）。"""
    rows = []
    for ds_dir in dataset_dirs:
        ds_name = ds_dir.name
        tra = find_fold_files(ds_dir, ds_name, 1)[0]
        if tra is None:
            continue
        try:
            X, y, minlab, n_cat = parse_keel_dat(tra)
        except Exception as e:
            print(f"  [WARN] {ds_name}: {e}")
            continue
        n_min, n_maj = int(y.sum()), int((y == 0).sum())
        rows.append({
            "Dataset": ds_name, "n_train": len(y), "dim": X.shape[1],
            "n_majority": n_maj, "n_minority": n_min,
            "IR": round(n_maj / max(n_min, 1), 2),
            "minority_label": minlab, "n_categorical_cols": n_cat,
        })
    prof = pd.DataFrame(rows).sort_values("IR").reset_index(drop=True)
    prof.to_csv(PROFILE_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ 資料集特性已輸出：{PROFILE_CSV.resolve()}")
    print(prof.to_string(index=False))
    if (prof["n_categorical_cols"] > 0).any():
        print("\n⚠️  偵測到 categorical 欄位的資料集："
              f"{prof[prof['n_categorical_cols'] > 0]['Dataset'].tolist()}")
        print("   A~M 的 parse_keel_dat 對 train/test 各自編碼，可能 mapping 錯位。")
        print("   若要修，必須【整批】修 A~M 後重跑，不能只修 N。")
    print("\n建議：依 IR 低/中/高各挑 2 個填進 DATASET_WHITELIST，並在看到結果前固定。")
    return prof


def find_fold_files(ds_dir, ds_name, fold):
    """檔名 pattern 與 A~M 逐字相同。"""
    file_prefix = re.sub(r'-fold.*$', '', ds_name)
    pat_tra = [ds_dir / f"{file_prefix}-{fold}tra.dat",
               ds_dir / f"{ds_name}-5-fold-tra{fold}.dat",
               ds_dir / f"{ds_name}-5-tra{fold}.dat",
               ds_dir / f"{ds_name}_fold{fold}_train.dat"]
    pat_tst = [ds_dir / f"{file_prefix}-{fold}tst.dat",
               ds_dir / f"{ds_name}-5-fold-tst{fold}.dat",
               ds_dir / f"{ds_name}-5-tst{fold}.dat",
               ds_dir / f"{ds_name}_fold{fold}_test.dat"]
    return (next((p for p in pat_tra if p.exists()), None),
            next((p for p in pat_tst if p.exists()), None))


# ─────────────────────────── Route 2 用的快取 ───────────────────────────────
def save_mask_cache(ds_name, fold, ae_type, cfg_label, masks, maj_rows,
                    fingerprint, seed, DF_bundle=None):
    """把保留遮罩（與可選的 DF）存成 .npz，Route 2 直接載入，不必重跑 sampler / AE。

    meta 含檔案指紋與套件版本：Route 2 載入時可驗證是同一份資料、同一個表徵空間。
    maj_rows 是 majority 樣本在【該 fold 訓練檔】中的原始列位置，
    不是 fold 內重新編號，避免之後對不回原始資料。
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{ds_name}__f{fold}__{ae_type}__{cfg_label.replace('/', '-')}"
    meta = {
        "dataset": ds_name, "fold": fold, "ae": ae_type, "config": cfg_label,
        "train_file_sha1": fingerprint, "cell_seed": int(seed),
        "seed_mode": SEED_MODE, "base_methods": BASE_METHODS,
        "sampler_params": {"ENN_k": ENN_K, "CNN_k": CNN_K, "CNN_seed": CNN_SEED},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "sklearn": sklearn.__version__, "imblearn": imblearn.__version__,
                     "torch": torch.__version__},
        "scale_mode": SAMPLER_SCALE_MODE,
    }
    payload = {f"keep_{m}": masks[m] for m in BASE_METHODS}
    payload["maj_rows_in_train"] = maj_rows
    payload["meta_json"] = np.array(json.dumps(meta, ensure_ascii=False))
    if DF_bundle is not None:
        payload.update(DF_bundle)
    np.savez_compressed(CACHE_DIR / f"{tag}.npz", **payload)


# ─────────────────────────── 主流程 ──────────────────────────────────────────
def eval_all_occ(keep_mask, DF_maj_s, DF_tst_s, y_tst):
    DF_maj_clean = DF_maj_s[keep_mask]
    if len(DF_maj_clean) < 5:
        return None
    n_nb_cap = max(1, len(DF_maj_clean) - 1)
    out = {}
    for occ_type in OCC_TYPES:
        try:
            out[occ_type] = run_occ_eval(occ_type, DF_maj_clean, DF_tst_s,
                                         y_tst, n_nb_cap, do_scale=False)
        except Exception:
            out[occ_type] = {m: float("nan") for m in ALL_METRIC_COLS}
    return out


def run_experiment():
    dataset_dirs = sorted([d for d in DATA_ROOT.iterdir() if d.is_dir()])
    if DATASET_WHITELIST:
        dataset_dirs = [d for d in dataset_dirs if d.name in DATASET_WHITELIST]
    elif DATASET_LIMIT and DATASET_LIMIT > 0:
        print("⚠️  DATASET_WHITELIST 為空，將取排序後前 "
              f"{DATASET_LIMIT} 個資料集——這是方便取樣，不是代表性 pilot。")
        print("   建議先跑 SCAN_ONLY=True 依 IR 挑 6 個再填 whitelist。")
        dataset_dirs = dataset_dirs[:DATASET_LIMIT]
    if not dataset_dirs:
        raise FileNotFoundError(f"找不到任何資料夾於 {DATA_ROOT.resolve()}")

    if SCAN_ONLY:
        scan_datasets(dataset_dirs)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    all_records, overlap_records, invalid_records = [], [], []
    categorical_datasets = set()

    for ds_dir in dataset_dirs:
        ds_name = ds_dir.name
        print(f"\n{'='*68}\n▶ Dataset: {ds_name}")

        for fold in range(1, N_FOLDS + 1):
            tra_file, tst_file = find_fold_files(ds_dir, ds_name, fold)
            if tra_file is None or tst_file is None:
                print(f"  [SKIP] Fold {fold}: 找不到檔案")
                continue

            try:
                X_tra, y_tra, minority_label, n_cat = parse_keel_dat(tra_file)
                X_tst, y_tst, _, _ = parse_keel_dat(tst_file, minority_label=minority_label)
                if n_cat > 0:
                    categorical_datasets.add(ds_name)

                input_dim = X_tra.shape[1]
                maj_rows  = np.where(y_tra == 0)[0]          # 原始列位置，給 Route 2
                X_maj, X_min = X_tra[y_tra == 0], X_tra[y_tra == 1]

                if len(X_maj) < 5:
                    print(f"  [SKIP] Fold {fold}: 訓練集正常樣本不足 ({len(X_maj)})"); continue
                if len(X_min) < 1:
                    print(f"  [SKIP] Fold {fold}: 訓練集無少數類（sampler 需參考點）"); continue
                if y_tst.sum() == 0:
                    print(f"  [SKIP] Fold {fold}: 測試集無少數類樣本"); continue
                if len(X_maj) <= len(X_min):
                    print(f"  [WARN] Fold {fold}: 正常類並非多數類 "
                          f"(maj={len(X_maj)}, min={len(X_min)})，請確認標籤定義")

                scaler  = MinMaxScaler()
                X_maj_s = scaler.fit_transform(X_maj)   # fit 只用 training majority
                X_min_s = scaler.transform(X_min)       # minority 僅作 sampler 參考點
                X_tst_s = scaler.transform(X_tst)
                fp = file_fingerprint(tra_file)
            except Exception as e:
                print(f"  [ERROR] Fold {fold} 資料載入失敗: {e}")
                continue

            for ae_type in AE_TYPES:
                for cfg_idx, cfg_label in enumerate(ACTIVE_CONFIGS):
                    n_layers, ratio_label = parse_config_label(cfg_label)
                    n_units = max(2, round(input_dim * BOTTLENECK_RATIOS[ratio_label]))
                    seed = (cell_seed(ds_name, fold, ae_type, cfg_label)
                            if SEED_MODE == "stable_per_cell" else None)

                    try:
                        extract = train_ae_and_get_extractor(
                            ae_type, X_maj_s, n_layers, n_units, seed=seed)
                        DF_maj = extract(X_maj_s)
                        DF_tst = extract(X_tst_s)
                        _rng = torch.get_rng_state()
                        DF_min = extract(X_min_s)
                        torch.set_rng_state(_rng)
                    except Exception as e:
                        print(f"  [ERROR] Fold{fold} {ae_type} {cfg_label}: AE 失敗 {e}")
                        continue

                    scaler_df = MinMaxScaler().fit(DF_maj)   # 只 fit 未清理的 DF_maj
                    DF_maj_s  = scaler_df.transform(DF_maj)
                    DF_min_s  = scaler_df.transform(DF_min)
                    DF_tst_s  = scaler_df.transform(DF_tst)
                    n_maj     = len(DF_maj_s)

                    try:
                        masks, statuses, any_fb = compute_keep_masks(
                            DF_maj_s, DF_min_s, BASE_METHODS)
                    except Exception as e:
                        print(f"  [ERROR] Fold{fold} {ae_type} {cfg_label}: keep mask 失敗 {e}")
                        continue

                    if CACHE_MASKS:
                        bundle = None
                        if CACHE_DF:
                            bundle = {"DF_maj_s": DF_maj_s, "DF_min_s": DF_min_s,
                                      "DF_tst_s": DF_tst_s, "y_tst": y_tst}
                        save_mask_cache(ds_name, fold, ae_type, cfg_label, masks,
                                        maj_rows, fp, seed if seed is not None else -1,
                                        DF_bundle=bundle)

                    # fallback 的 cell 不納入重疊度統計（否則 Jaccard 被假 mask 汙染）
                    if not any_fb:
                        overlap_records.append(collect_overlap_record(
                            ds_name, ae_type, cfg_label, fold, masks, n_maj))
                    else:
                        invalid_records.append({
                            "Dataset": ds_name, "AE": ae_type, "Config": cfg_label,
                            "Fold": fold, "Reason": "voter_fallback",
                            "Detail": "; ".join(f"{k}={v}" for k, v in statuses.items()
                                                if v.startswith("fallback")),
                        })

                    if OVERLAP_ONLY:
                        continue

                    for strategy in STRATEGIES:
                        family, voters, threshold, role = STRATEGY_SPECS[strategy]
                        keep, status, valid = apply_strategy(
                            masks, statuses, strategy, n_maj, any_fb)
                        if not valid:
                            invalid_records.append({
                                "Dataset": ds_name, "AE": ae_type, "Config": cfg_label,
                                "Fold": fold, "Reason": f"strategy_invalid:{strategy}",
                                "Detail": status})
                            continue

                        n_kept    = int(keep.sum())
                        n_removed = n_maj - n_kept
                        mets = eval_all_occ(keep, DF_maj_s, DF_tst_s, y_tst)
                        if mets is None:
                            invalid_records.append({
                                "Dataset": ds_name, "AE": ae_type, "Config": cfg_label,
                                "Fold": fold, "Reason": f"degenerate:{strategy}",
                                "Detail": f"kept={n_kept}"})
                            continue

                        base_row = {
                            "Dataset": ds_name, "AE": ae_type, "Strategy": strategy,
                            "StrategyFamily": family, "Role": role,
                            "Voters": "+".join(voters) if voters else "-",
                            "Threshold": threshold, "Config": cfg_label, "Fold": fold,
                            "MajKept": n_kept, "MajRemoved": n_removed,
                            "RemovedRate": safe_removed_rate(n_removed, n_kept),
                            "SamplerStatus": status, "RM_Repeat": 0,
                        }
                        for occ_type, mm in mets.items():
                            all_records.append({**base_row, "OCC": occ_type, **mm})

                        # ── rate-matched 隨機對照：逐次保留，不只存平均 ──
                        if ENABLE_RATE_MATCHED and family == "vote" and n_removed > 0:
                            for rep in range(1, N_RM_REPEATS + 1):
                                rng = np.random.default_rng(
                                    stable_hash(RM_SEED, ds_name, fold, ae_type,
                                                cfg_label, strategy, rep))
                                rm_keep = rate_matched_keep(n_maj, n_removed, rng)
                                assert int((~rm_keep).sum()) == n_removed
                                rmm = eval_all_occ(rm_keep, DF_maj_s, DF_tst_s, y_tst)
                                if rmm is None:
                                    continue
                                for occ_type, mm in rmm.items():
                                    all_records.append({
                                        **base_row, "Strategy": f"RM_{strategy}",
                                        "StrategyFamily": "rate_matched",
                                        "Role": "control", "Voters": "random",
                                        "SamplerStatus": "rate_matched_random",
                                        "RM_Repeat": rep, "OCC": occ_type, **mm})

            print(f"  [fold {fold}] 完成 {len(ACTIVE_CONFIGS)} config × "
                  f"{len(AE_TYPES)} AE × {len(STRATEGIES)} 策略 × {len(OCC_TYPES)} OCC")

    if categorical_datasets:
        print(f"\n⚠️  含 categorical 欄位的資料集：{sorted(categorical_datasets)}")
        print("   A~M 的編碼方式對 train/test 各自處理，mapping 可能錯位；"
              "若要修必須整批修 A~M 後重跑。")

    df_all     = pd.DataFrame(all_records)
    df_overlap = pd.DataFrame(overlap_records)
    df_invalid = pd.DataFrame(invalid_records)

    if df_all.empty or "AUC" not in df_all.columns:
        return df_all, df_overlap, df_invalid

    if CONFIG_MODE == "grid":
        print("\n⚠️  CONFIG_MODE='grid'：ak_best_export 使用 per-dataset oracle 選法，"
              "只能當上界報告，不可寫成「固定 config 的成績」。")
    return df_all, df_overlap, df_invalid


# ─────────────────────────── 分析表 ─────────────────────────────────────────
PAIR_KEYS = ["Dataset", "AE", "OCC", "Config", "Fold"]


def _paired_delta(df, ref_strategy):
    """同 (Dataset, AE, OCC, Config, Fold) 內先配對相減，再往上聚合。

    先各自平均再相減在有 skip 時不等價，論文的 Δ 一定要用配對版。
    """
    if df.empty:
        return pd.DataFrame()
    real = df[df["StrategyFamily"] != "rate_matched"]
    ref = (real[real["Strategy"] == ref_strategy]
           .drop_duplicates(PAIR_KEYS)
           .set_index(PAIR_KEYS)[ALL_METRIC_COLS]
           .rename(columns={m: f"{m}_ref" for m in ALL_METRIC_COLS}))
    if ref.empty:
        return pd.DataFrame()
    merged = real.merge(ref, left_on=PAIR_KEYS, right_index=True, how="inner")
    for m in ALL_METRIC_COLS:
        merged[f"Δ{m}"] = merged[m] - merged[f"{m}_ref"]
    return merged


def build_effect_table(df, ref_strategy):
    """策略相對 ref 的配對 Δ，聚合到 (AE, OCC, Config, Strategy)。"""
    merged = _paired_delta(df, ref_strategy)
    if merged.empty:
        return pd.DataFrame()
    keys = ["AE", "OCC", "Config", "Strategy", "Role"]
    out = merged.groupby(keys)[["RemovedRate"] + ALL_METRIC_COLS
                               + [f"Δ{m}" for m in ALL_METRIC_COLS]].mean().reset_index()
    out.insert(0, "Reference", ref_strategy)
    return out.sort_values(keys).reset_index(drop=True)


def build_dataset_level_paired(df, ref_strategy):
    """dataset-level 配對統計：先把 5 folds 平均成每個 dataset 一個值，
    再算 win / tie / loss 與 paired Wilcoxon。

    不把 17×5=85 folds 當 85 個獨立觀測做顯著性檢定。
    """
    merged = _paired_delta(df, ref_strategy)
    if merged.empty:
        return pd.DataFrame()
    ds_lvl = (merged.groupby(["AE", "OCC", "Config", "Strategy", "Role", "Dataset"])
                    [["ΔAUC"] + [f"Δ{m}" for m in ALL_METRIC_COLS if m != "AUC"]]
                    .mean().reset_index())

    rows = []
    for (ae, occ, cfg, strat, role), g in ds_lvl.groupby(
            ["AE", "OCC", "Config", "Strategy", "Role"]):
        d = g["ΔAUC"].dropna().values
        if len(d) == 0 or strat == ref_strategy:
            continue
        win  = int((d > 1e-9).sum())
        loss = int((d < -1e-9).sum())
        tie  = int(len(d) - win - loss)
        try:
            p = float(wilcoxon(d).pvalue) if len(d) >= 5 and np.any(np.abs(d) > 1e-12) else float("nan")
        except Exception:
            p = float("nan")
        rows.append({
            "Reference": ref_strategy, "AE": ae, "OCC": occ, "Config": cfg,
            "Strategy": strat, "Role": role, "nDatasets": len(d),
            "MeanΔAUC": float(np.mean(d)), "MedianΔAUC": float(np.median(d)),
            "Win": win, "Tie": tie, "Loss": loss,
            "WinRate": float(win / len(d)), "Wilcoxon_p": p,
        })
    return pd.DataFrame(rows).sort_values(
        ["OCC", "Config", "Strategy"]).reset_index(drop=True)


def build_rate_matched_gap(df):
    """策略 vs 同刪除率隨機刪除。逐次保留後可算 random 的變異與勝出次數。

    Gap > 0 才代表「刪哪裡」有貢獻；Gap ≈ 0 表示效果全來自「刪多少」。
    """
    if df.empty:
        return pd.DataFrame()
    vote = df[df["StrategyFamily"] == "vote"]
    rm   = df[df["StrategyFamily"] == "rate_matched"].copy()
    if vote.empty or rm.empty:
        return pd.DataFrame()
    rm["Strategy"] = rm["Strategy"].str.replace("^RM_", "", regex=True)

    v = vote.groupby(PAIR_KEYS + ["Strategy"])[ALL_METRIC_COLS + ["RemovedRate"]].mean().reset_index()
    r = rm.groupby(PAIR_KEYS + ["Strategy"])["AUC"].agg(
        AUC_rand_mean="mean", AUC_rand_std="std", n_rep="count").reset_index()
    rw = rm.merge(v[PAIR_KEYS + ["Strategy", "AUC"]].rename(columns={"AUC": "AUC_strat"}),
                  on=PAIR_KEYS + ["Strategy"], how="inner")
    wins = (rw.assign(win=(rw["AUC_strat"] > rw["AUC"]).astype(int))
              .groupby(PAIR_KEYS + ["Strategy"])["win"].sum().reset_index()
              .rename(columns={"win": "StratBeatsRandom_n"}))

    merged = v.merge(r, on=PAIR_KEYS + ["Strategy"]).merge(wins, on=PAIR_KEYS + ["Strategy"])
    merged["GapAUC"] = merged["AUC"] - merged["AUC_rand_mean"]

    keys = ["AE", "OCC", "Config", "Strategy"]
    out = merged.groupby(keys).agg(
        RemovedRate=("RemovedRate", "mean"),
        AUC=("AUC", "mean"),
        AUC_rand_mean=("AUC_rand_mean", "mean"),
        AUC_rand_std=("AUC_rand_std", "mean"),
        GapAUC=("GapAUC", "mean"),
        BeatsRandom=("StratBeatsRandom_n", "sum"),
        TotalDraws=("n_rep", "sum"),
    ).reset_index()
    out["BeatRate"] = out["BeatsRandom"] / out["TotalDraws"].replace(0, np.nan)
    return out.sort_values(keys).reset_index(drop=True)


def build_strategy_overall(df):
    if df.empty:
        return pd.DataFrame()
    g = (df.groupby(["Strategy", "StrategyFamily", "Role", "Voters", "Threshold", "OCC"])
           [ALL_METRIC_COLS + ["RemovedRate", "MajKept", "MajRemoved"]]
           .mean().reset_index())
    order = {s: i for i, s in enumerate(STRATEGIES)}
    g["_ord"] = g["Strategy"].map(
        lambda s: order.get(s.replace("RM_", ""), 99) + (0.5 if s.startswith("RM_") else 0))
    return g.sort_values(["_ord", "OCC"]).drop(columns=["_ord"]).reset_index(drop=True)


def build_overlap_summary(df_overlap):
    if df_overlap.empty:
        return pd.DataFrame()
    num_cols = [c for c in df_overlap.columns
                if c not in ["Dataset", "AE", "Config", "Fold"]]
    return (df_overlap.groupby(["Dataset", "AE", "Config"])[num_cols]
                      .mean().reset_index())


# ─────────────────────────── Excel 樣式（與 J/L 共用設計）───────────────────
HEADER_FILL = PatternFill("solid", fgColor="2F5597")
ALT_FILL    = PatternFill("solid", fgColor="F2F2F2")
POS_FILL    = PatternFill("solid", fgColor="C6EFCE")
NEG_FILL    = PatternFill("solid", fgColor="F8CBAD")
FAMILY_FILL = {
    "baseline":     PatternFill("solid", fgColor="EEEEEE"),
    "single":       PatternFill("solid", fgColor="DAEEF3"),
    "vote":         PatternFill("solid", fgColor="E2EFDA"),
    "rate_matched": PatternFill("solid", fgColor="FFF2CC"),
}
HEADER_FONT  = Font(name="Arial", bold=True, color="FFFFFF", size=11)
BODY_FONT    = Font(name="Arial", size=10)
BOLD_FONT    = Font(name="Arial", bold=True, size=10)
CENTER_ALIGN = Alignment(horizontal="center", vertical="center")
LEFT_ALIGN   = Alignment(horizontal="left", vertical="center")
THIN_BORDER  = Border(left=Side(style="thin"), right=Side(style="thin"),
                      top=Side(style="thin"), bottom=Side(style="thin"))

INT_COLS = {"Fold", "MajKept", "MajRemoved", "MajTotal", "Threshold", "RM_Repeat",
            "DelInterAll", "DelUnionAll", "Win", "Tie", "Loss", "nDatasets",
            "BeatsRandom", "TotalDraws"}


def sc(cell, value, font=None, fill=None, align=None, fmt=None):
    cell.value  = value
    cell.border = THIN_BORDER
    if font:  cell.font          = font
    if fill:  cell.fill          = fill
    if align: cell.alignment     = align
    if fmt:   cell.number_format = fmt


def col_w(ws, letter, width):
    ws.column_dimensions[letter].width = width


def dump_table(ws, title, df, delta_cols=None, fmt="0.0000"):
    delta_cols = delta_cols or []
    if df is None or df.empty:
        sc(ws.cell(1, 1), f"{title}（無資料）", font=BOLD_FONT, align=LEFT_ALIGN)
        col_w(ws, "A", 90)
        return
    cols = list(df.columns)
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(cols))
    sc(ws.cell(1, 1), title,
       font=Font(name="Arial", bold=True, size=12, color="1F3864"), align=CENTER_ALIGN)
    for c, h in enumerate(cols, 1):
        sc(ws.cell(2, c), h, font=HEADER_FONT, fill=HEADER_FILL, align=CENTER_ALIGN)

    for r, (_, row) in enumerate(df.iterrows(), 3):
        base_fill = ALT_FILL if r % 2 == 0 else None
        if "StrategyFamily" in cols:
            base_fill = FAMILY_FILL.get(row.get("StrategyFamily"), base_fill)
        for c, col in enumerate(cols, 1):
            val  = row[col]
            fill = base_fill
            is_num = isinstance(val, (int, float, np.integer, np.floating)) and not pd.isna(val)
            if col in delta_cols and is_num:
                fill = POS_FILL if val > 1e-9 else (NEG_FILL if val < -1e-9 else fill)
            sc(ws.cell(r, c), val, font=BODY_FONT, fill=fill,
               align=LEFT_ALIGN if col in ["Dataset", "Voters", "SamplerStatus",
                                           "SamplerScaleMode", "Detail"] else CENTER_ALIGN,
               fmt=fmt if (is_num and col not in INT_COLS
                           and not col.startswith("Del_")
                           and not col.startswith("DelVotes_")) else None)

    for c, col in enumerate(cols, 1):
        width = 13
        if col == "Dataset":                                      width = 26
        if col in ["Strategy", "StrategyFamily", "Reference"]:    width = 20
        if col in ["SamplerStatus", "Voters", "Detail"]:          width = 26
        if col in ["SamplerScaleMode", "BaselineRef", "ConfigPolicy"]: width = 26
        col_w(ws, get_column_letter(c), width)
    ws.freeze_panes = "A3"


# ─────────────────────────── 統整輸出（schema 與 I/J/K 相同）────────────────
def make_ak_export_df(df, config_policy):
    """欄位與 I/J/K 的 COMPARISON_EXPORT_COLS 逐欄相同；Strategy 寫進 Sampler。

    PR-AUC 等本檔額外指標刻意不放進來，確保 L_merge 讀到的 schema 一致。
    """
    if df.empty:
        return pd.DataFrame(columns=COMPARISON_EXPORT_COLS)
    out = pd.DataFrame({
        "Study": STUDY_ID, "Method": METHOD_ID, "FeatureSet": FEATURE_SET,
        "Dataset": df["Dataset"], "AE": df["AE"], "Sampler": df["Strategy"],
        "OCC": df["OCC"], "Config": df["Config"], "Fold": df["Fold"],
        "ConfigPolicy": config_policy,
        "MajKept": df["MajKept"], "MajRemoved": df["MajRemoved"],
        "RemovedRate": df["RemovedRate"], "SamplerStatus": df["SamplerStatus"],
        "BaselineRef": BASELINE_REF, "OCCScope": OCC_SCOPE,
        "SamplerScaleMode": SAMPLER_SCALE_MODE,
    })
    for metric in METRIC_COLS:
        out[metric] = df[metric]
    return out[COMPARISON_EXPORT_COLS]


def write_ak_export(ws, df, title, config_policy):
    ws.title = title
    out = make_ak_export_df(df, config_policy)
    for c, h in enumerate(COMPARISON_EXPORT_COLS, 1):
        sc(ws.cell(1, c), h, font=HEADER_FONT, fill=HEADER_FILL, align=CENTER_ALIGN)
    for r, (_, row) in enumerate(out.iterrows(), 2):
        for c, col in enumerate(COMPARISON_EXPORT_COLS, 1):
            sc(ws.cell(r, c), row[col], font=BODY_FONT,
               align=LEFT_ALIGN if col in ["Dataset", "SamplerScaleMode"] else CENTER_ALIGN,
               fmt="0.0000" if col in METRIC_COLS + ["RemovedRate"] else None)
    widths = [8, 18, 12, 26, 8, 18, 10, 12, 6, 26, 10, 12, 12, 24, 26, 12, 48] \
             + [10] * len(METRIC_COLS)
    for i, w in enumerate(widths, 1):
        col_w(ws, get_column_letter(i), w)
    ws.freeze_panes = "A2"


def write_alignment_notes(ws):
    ws.title = "alignment_notes"
    strat_desc = "; ".join(
        f"{s}[{STRATEGY_SPECS[s][3]}"
        + (f", voters={'+'.join(STRATEGY_SPECS[s][1])}, keep>={STRATEGY_SPECS[s][2]}"
           if STRATEGY_SPECS[s][0] == "vote" else "")
        + "]" for s in STRATEGIES)

    notes = [
        ("Study / Method", f"{STUDY_ID} / {METHOD_ID}"),
        ("Route", "Study 3 Route 1 — set-operation ensemble under-sampling on the kept subsets."),
        ("Primary research question", "Does consensus-based ensemble under-sampling outperform the best SINGLE under-sampler (ENN) from Study 2, under a fixed representation?"),
        ("Primary baseline", f"{PRIMARY_BASELINE} computed INSIDE this run (not the older J numbers)."),
        ("Strategies", strat_desc),
        ("Vote semantics", "A majority sample is KEPT when #voters keeping it >= threshold. Voting is per ORIGINAL sample; no concatenation, so no sample is duplicated."),
        ("Seed mode", f"{SEED_MODE}. stable_per_cell derives a fixed seed per (dataset, fold, AE, config) via CRC32, so results do NOT depend on execution order."),
        ("Reproduce old B/J RNG", "Set SEED_MODE='legacy_global' AND AE_TYPES=['AE','DAE','SAE','VAE'] AND CONFIG_MODE='grid' AND the same dataset set. Only then is the torch RNG path identical to B/J. Under the default settings the AE weights differ from B/J by design; cross-script absolute AUC comparison was never valid."),
        ("Strict fallback", f"{STRICT_FALLBACK}. If any voter falls back, every vote strategy for that (dataset, fold, AE, config) is marked invalid and skipped. A failed sampler is never allowed to vote 'keep all'."),
        ("Rate-matched control", f"enabled={ENABLE_RATE_MATCHED}, repeats={N_RM_REPEATS}, stable CRC32 seed per (dataset, fold, AE, config, strategy, repeat). Each repeat is stored as its own row."),
        ("Config policy", "fixed_config_no_oracle" if CONFIG_MODE != "grid" else "best_config_per_dataset (ORACLE upper bound only)"),
        ("Active configs", ", ".join(ACTIVE_CONFIGS)),
        ("Fixed config rationale", "h1-1/1 uses a single hidden layer with latent dimension equal to the input dimension: a neutral representation that involves no compression-strength selection, so Study 3 isolates the subset-selection strategy. It is NOT chosen because it was best in the oracle table."),
        ("Oracle warning", "Study 1/2 best_overall values (e.g. 0.7066) are averages over PER-DATASET oracle configs, not the score of any single fixed config. 'Most Freq Config' is only the mode of those picks."),
        ("Methodological positioning", "Label-informed training-data cleaning followed by one-class classification: the training-fold minority is used ONLY to decide which majority samples to remove; it never enters OCC fitting, and the test fold is never touched."),
        ("Statistical unit", "Dataset, not fold. Deltas are paired within (Dataset, AE, OCC, Config, Fold), then averaged per dataset before win/tie/loss and Wilcoxon."),
        ("Run mode", f"{RUN_MODE}; whitelist={DATASET_WHITELIST or 'NONE (using DATASET_LIMIT=' + str(DATASET_LIMIT) + ')'}"),
        ("AE", f"{', '.join(AE_TYPES)} | epochs={AE_EPOCHS}, batch={AE_BATCH_SIZE}, lr={AE_LR}, DAE noise={DAE_NOISE}, SAE sparsity={SAE_SPARSITY}, VAE beta={VAE_BETA}"),
        ("OCC", "OCSVM(nu=0.1, kernel=rbf, gamma=scale) | LOF(n_neighbors=min(20, n_kept-1), novelty=True, contamination=0.1) | IsolationForest(n_estimators=100, contamination=0.1, random_state=42)"),
        ("gamma note", "gamma='scale' is sklearn's default since 0.22, so this is numerically identical to A~M; it is written explicitly so Chapter 3 can state it."),
        ("Sampler hyperparameters", f"ENN(n_neighbors={ENN_K}, kind_sel=all) | CNN(n_neighbors={CNN_K}, random_state={CNN_SEED}) | TomekLinks(default); sampling_strategy=auto (majority only)"),
        ("Scaling", SAMPLER_SCALE_MODE),
        ("Threshold", "90th percentile of training majority anomaly scores; all three OCC use -decision_function so higher = more anomalous."),
        ("Metrics", f"aligned with A~M: {', '.join(METRIC_COLS)}" + (f" | extra (this file only, not in ak_export): {', '.join(EXTRA_METRIC_COLS)}" if EXTRA_METRIC_COLS else "")),
        ("Degenerate guard", "Skip when the cleaned majority has fewer than 5 samples; LOF k recomputed after cleaning."),
        ("Data leakage guard", "MinMax fit on training majority only; DF scaler fit on uncleaned DF_maj only; voters use the train fold only; test is transform-only and never participates in sampling or selection."),
        ("Mask cache", f"enabled={CACHE_MASKS} (DF cached={CACHE_DF}) → {CACHE_DIR}. Stores keep masks, original majority row positions, train-file SHA1 and package versions so Route 2 can reuse the SAME representation without retraining the AE."),
        ("Categorical caveat", "parse_keel_dat is byte-identical to A~M and encodes train/test independently. Datasets containing categorical columns are flagged in console and dataset_profile.csv; fixing this requires re-running ALL of A~M, not just N."),
        ("Merge note", "ak_*_export matches I/J/K column-for-column (Strategy stored in the Sampler column); add this workbook to L_merge SOURCES to combine."),
        ("Versions", f"python={platform.python_version()}, numpy={np.__version__}, pandas={pd.__version__}, sklearn={sklearn.__version__}, imblearn={imblearn.__version__}, scipy={scipy.__version__}, torch={torch.__version__}"),
    ]
    sc(ws.cell(1, 1), "Item", font=HEADER_FONT, fill=HEADER_FILL, align=CENTER_ALIGN)
    sc(ws.cell(1, 2), "Value", font=HEADER_FONT, fill=HEADER_FILL, align=CENTER_ALIGN)
    for r, (k, v) in enumerate(notes, 2):
        fill = ALT_FILL if r % 2 == 0 else None
        sc(ws.cell(r, 1), k, font=BOLD_FONT, fill=fill, align=LEFT_ALIGN)
        sc(ws.cell(r, 2), v, font=BODY_FONT, fill=fill, align=LEFT_ALIGN)
    col_w(ws, "A", 28)
    col_w(ws, "B", 130)
    ws.freeze_panes = "A2"


def save_excel(df_all, df_overlap, df_invalid):
    delta_cols = [f"Δ{m}" for m in ALL_METRIC_COLS] + ["MeanΔAUC", "MedianΔAUC", "GapAUC"]
    policy = "fixed_config_no_oracle" if CONFIG_MODE != "grid" else "all_configs"

    per_fold_cols = ["Dataset", "AE", "Strategy", "StrategyFamily", "Role", "Voters",
                     "Threshold", "OCC", "Config", "Fold", "RM_Repeat",
                     "MajKept", "MajRemoved", "RemovedRate", "SamplerStatus"] + ALL_METRIC_COLS
    df_pf = df_all[[c for c in per_fold_cols if c in df_all.columns]] if not df_all.empty else df_all

    wb = Workbook()
    ws1 = wb.active
    dump_table(ws1, "all_per_fold（逐 fold 原始結果；RM_Repeat>0 為隨機對照的單次抽樣）", df_pf)
    ws1.title = "all_per_fold"

    dump_table(wb.create_sheet("strategy_overall"),
               "策略 × OCC 全域平均（含平均刪除率）", build_strategy_overall(df_all))

    dump_table(wb.create_sheet("effect_vs_ENN"),
               f"★ Study 3 核心表：相對 {PRIMARY_BASELINE} 的配對 Δ（ensemble 有沒有贏過 single best）",
               build_effect_table(df_all, PRIMARY_BASELINE), delta_cols=delta_cols)

    dump_table(wb.create_sheet("paired_stats_vs_ENN"),
               f"dataset-level 配對統計 vs {PRIMARY_BASELINE}（win/tie/loss + Wilcoxon；統計單位為資料集）",
               build_dataset_level_paired(df_all, PRIMARY_BASELINE), delta_cols=delta_cols)

    dump_table(wb.create_sheet("effect_vs_none"),
               "相對 none 的配對 Δ（under-sampling 整體有沒有幫助）",
               build_effect_table(df_all, "none"), delta_cols=delta_cols)

    dump_table(wb.create_sheet("paired_stats_vs_none"),
               "dataset-level 配對統計 vs none", 
               build_dataset_level_paired(df_all, "none"), delta_cols=delta_cols)

    dump_table(wb.create_sheet("rate_matched_gap"),
               "刪除位置效應：策略 − 同刪除率隨機（Gap>0 且 BeatRate 高才代表「刪哪裡」有貢獻）",
               build_rate_matched_gap(df_all), delta_cols=["GapAUC"])

    dump_table(wb.create_sheet("overlap_jaccard"),
               "三方法【刪除集合】重疊結構：Jaccard / 交集 / 刪除票數分布（含比例欄）",
               build_overlap_summary(df_overlap))

    dump_table(wb.create_sheet("invalid_log"),
               "被排除的 cell（voter fallback / 策略退化）——必須在論文中交代",
               df_invalid)

    write_ak_export(wb.create_sheet("ak_all_export"), df_all, "ak_all_export", policy)
    write_alignment_notes(wb.create_sheet("alignment_notes"))

    wb.save(OUTPUT_FILE)
    print(f"\n✅ 結果已儲存至：{OUTPUT_FILE.resolve()}")


def flush_warnings():
    if _WARN_BUFFER:
        WARN_LOG.write_text("\n".join(_WARN_BUFFER), encoding="utf-8")
        print(f"⚠️  {len(_WARN_BUFFER)} 筆 warning 已寫入：{WARN_LOG.resolve()}")


# ─────────────────────────── Entry Point ─────────────────────────────────────
if __name__ == "__main__":
    print("=" * 74)
    print("Study Three 路線一：集合運算式 Ensemble Under-sampling（DF_maj）v2")
    print(f"模式       : SCAN_ONLY={SCAN_ONLY} / OVERLAP_ONLY={OVERLAP_ONLY} / RUN_MODE={RUN_MODE}")
    print(f"Seed       : {SEED_MODE}（stable_per_cell 不依賴執行順序）")
    print(f"AE         : {AE_TYPES} | Config: {CONFIG_MODE} → {ACTIVE_CONFIGS}")
    print(f"投票者     : {BASE_METHODS}（STRICT_FALLBACK={STRICT_FALLBACK}）")
    print(f"策略       : {STRATEGIES}")
    print(f"主要對照   : {PRIMARY_BASELINE}（本次 run 內部計算）")
    print(f"Rate-match : {ENABLE_RATE_MATCHED}（repeats={N_RM_REPEATS}，逐次保留）")
    print("=" * 74)

    df_all, df_overlap, df_invalid = run_experiment()

    if SCAN_ONLY:
        flush_warnings()
        sys.exit(0)

    if df_all.empty and df_overlap.empty:
        print("\n⚠️  沒有任何結果，請確認資料路徑與檔名格式。")
        flush_warnings()
        sys.exit(0)

    if OVERLAP_ONLY:
        print("\n── 三方法刪除集合重疊度（全域平均）──")
        jac = [c for c in df_overlap.columns if c.startswith("Jaccard_")]
        rate = [c for c in df_overlap.columns if c.startswith("DelRate_")]
        vrate = [c for c in df_overlap.columns if c.startswith("DelVoteRate_")]
        print(df_overlap[rate + jac].mean().round(4).to_string())
        print("\n刪除票數比例（0 票 = 三方法都想保留）：")
        print(df_overlap[vrate].mean().round(4).to_string())
        build_overlap_summary(df_overlap).to_csv(
            RESULTS_DIR / "Study3_overlap_only.csv", index=False, encoding="utf-8-sig")
        print(f"\n✅ 已輸出：{(RESULTS_DIR / 'Study3_overlap_only.csv').resolve()}")
        flush_warnings()
        sys.exit(0)

    save_excel(df_all, df_overlap, df_invalid)

    real = df_all[df_all["StrategyFamily"] != "rate_matched"]
    print("\n── 策略 × OCC 平均 AUC ──")
    piv = real.groupby(["Strategy", "OCC"])["AUC"].mean().unstack("OCC").round(4)
    print(piv.reindex([s for s in STRATEGIES if s in piv.index]).to_string())

    print("\n── 平均刪除率 ──")
    print(real.groupby("Strategy")["RemovedRate"].mean().round(4).to_string())

    ps = build_dataset_level_paired(df_all, PRIMARY_BASELINE)
    if not ps.empty:
        print(f"\n── ★ dataset-level：各策略 vs {PRIMARY_BASELINE} ──")
        print(ps[["OCC", "Strategy", "nDatasets", "MeanΔAUC",
                  "Win", "Tie", "Loss", "Wilcoxon_p"]].round(4).to_string(index=False))

    if not df_invalid.empty:
        print(f"\n⚠️  有 {len(df_invalid)} 個 cell 被排除，詳見 invalid_log 分頁。")

    flush_warnings()
