"""
occ_tune_us2.py
===============
Study 3 路線二（S / T / U）：兩階段 ensemble under-sampling，第二階段用 ENN
（沿用 occ_tune_core.py 的讀資料、AE 特徵快取、OCC；occ_tune_ensemble.py 的階段執行器與選法；
  occ_tune_us.py 的 sampler、第一階段投票、遮罩指紋、Wilcoxon / Holm。三支舊程式一行都不改。
  筆電測試用 run_laptop_US2.py，Azure 全跑用 run_azure_US2.py）

────────────────────────────────────────────────────────────────────────────
方法（每個特徵空間一個代號：S＝OF、T＝DF、U＝OF+DF）
────────────────────────────────────────────────────────────────────────────
  第一階段（與 M / N / O 完全相同）：ENN / CNN / TL 從同一份未清理的多數類平行出發，對同一筆樣本投票
  第二階段（新）：把第一階段留下的多數類 ＋ 原本全部的少數類，再跑一次 ENN（k=3, kind_sel=all），
                  只刪多數類；留下的多數類拿去訓練 OCC。
                  ※ 取代舊圖的 Random / Center / Uncentered；最後筆數由 ENN 決定，不再固定為 N。

    策略名稱                 第一階段             角色
    2S_Majority_ENN          S2_Majority（≥2 票） 主要方法
    2S_ENN_ENN               ENN                  必要對照：效果是不是只來自「多清一次」
    2S_Union_ENN             S1_Union（≥1 票）    等價核對：S1 幾乎等於 none，所以這條幾乎等於單次 ENN
    2S_Intersection_ENN      S3_Intersection      次要／診斷：第一階段已刪 84~91%

  判讀順序（由寬到嚴）：贏 S2_Majority → 第二階段 ENN 有補救效果；
                        贏 單次 ENN   → 「先集成再 ENN」比直接 ENN 有額外價值；
                        贏 2S_ENN_ENN → 才能歸因於第一階段的 ensemble，而不是多清一次。

────────────────────────────────────────────────────────────────────────────
之前跑過的不用重跑：對照組直接讀舊的 I~O run（us_run_name，例如 US_azure_v2）
────────────────────────────────────────────────────────────────────────────
  ・none / ENN / S1 / S2 / S3 的結果列不重算，統整時直接從舊 run 的 cells 讀進來一起比較。
  ・每一格開跑時都會重算第一階段遮罩，並和舊 run 同一格存下來的遮罩指紋（MaskSig）逐一比對；
    任何一個不同，這一格就不存檔、原因寫進 errors.log（代表資料、AE 特徵或 sampler 與舊 run 不一致）。
  ・新策略留下的樣本若和某個對照完全相同（例如 2S_Union_ENN ＝ ENN），OCC 不重新訓練，
    直接沿用舊 run 那一格的 59 組結果（ReuseFrom 欄）。只有「真的不同」的保留集合才會花時間訓練。
  ・AE 特徵一樣讀 A~C 留下的 _df_cache（out_root 相同就共用），不重訓 AE。

  ・每一格另外用舊預設參數重算「單次 ENN」的三個 OCC，五項指標要與舊 run 相同（差 < 1e-9）才存檔：
    確認現在這台機器、這個環境算出來的 OCC 數值和舊 run 是同一套，新舊結果才能放在同一張表。

可以分批跑：runner 的 datasets 先填 5 個，跑完統整（檔名會標 _5of17ds）；之後把 datasets 換成 17 個、
用同一個 run_name 再執行同一個指令，已完成的資料集不會重跑，只補新的。

其餘規則與 occ_tune_us.py 相同：MinMax 只 fit 未清理的訓練多數類、sampler 在同一座標下清理、
OCC 在清理後的多數類上訓練（不再二次 MinMax）、門檻＝訓練分數第 90 百分位、OCC 59 組參數、
清理後 < 5 筆判 degenerate、以 5 折平均 AUC 挑選（oracle，論文需註明）。
"""
from __future__ import annotations

import os
import occ_tune_core as core          # 必須最先 import（先設定執行緒環境變數）

import json
import time
import socket
import argparse
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import EditedNearestNeighbours
import imblearn

from occ_tune_core import (N_FOLDS, THRESHOLD_PCT, OCC_TYPES, METRIC_COLS, EXTRA_COLS,
                           LEGACY_DEFAULT, param_combos, param_id, is_legacy_default, safe_name)
import occ_tune_ensemble as ens
from occ_tune_ensemble import (_prep, _metrics, occ_scores_all, _pick, run_phase, prepare_run,
                               code_fingerprint, lib_versions, safe_table, summary_guard)
import occ_tune_us as us
from occ_tune_us import SPACES, SPACE_LABEL, MIN_KEEP, TIE_TOL

US2_VERSION = "US2_tune_v2"
#   v2：每一格的數值算法與 v1 相同；補強驗收與續跑——AE 快取嚴格讀取（壞檔不會默默重訓）、
#       每格用舊預設參數抽核 OCC 數值是否與舊 run 相同、舊 run 程式指紋不同預設停止、
#       統整時重新確認舊對照齊全並清點遮罩診斷、資料集可以分批跑（先 5 個、之後補其餘）。
NUM_TOL = 1e-9                         # 與舊 run 核對數值時的容許差（與 A~O 的核對相同）
SPACE_STUDY2 = {"OF": "S", "DF": "T", "OFDF": "U"}
STUDY_SPACE2 = {v: k for k, v in SPACE_STUDY2.items()}
# 名稱 -> (第一階段＝occ_tune_us 的哪個策略, 角色)。順序固定：2S_ENN_ENN 先算，後面要拿它做包含關係檢查
TWO_STAGE = {
    "2S_ENN_ENN":          ("ENN", "control"),
    "2S_Union_ENN":        ("S1_Union", "check"),
    "2S_Majority_ENN":     ("S2_Majority", "primary"),
    "2S_Intersection_ENN": ("S3_Intersection", "secondary"),
}
ROLE_LABEL = {"primary": "主要方法", "control": "必要對照（多清一次）", "check": "等價核對",
              "secondary": "次要／診斷", "ref": "舊 run 對照"}
# 從舊 I~O run 讀進來當對照的策略（不重算結果，只重算遮罩來核對）
REF_STRATEGIES = ["none", "ENN", "S1_Union", "S2_Majority", "S3_Intersection"]
ALL_STRATEGIES = REF_STRATEGIES + list(TWO_STAGE)
US2_COLS = us.US_COLS + ["Stage1", "Stage1Kept"]
MASK_COLS = ["Space", "Strategy", "Role", "Dataset", "Fold", "AE", "Config", "NMaj", "NMin", "Stage1",
             "Stage1Kept", "MajKept", "RemovedRate", "MaskSig", "Status", "EqualTo", "JaccardENN",
             "SubsetOfENN", "ContainsENN_ENN", "NumCheckDiff"]
G_US = ["Study", "Space", "Strategy", "Dataset", "AE", "OCC"]


@dataclass
class Us2Config(us.UsConfig):
    run_name: str = "US2_tune"
    studies: list = field(default_factory=lambda: ["S", "T", "U"])
    # 舊的 I~O 調參 run（同一個 out_root 底下）。對照組的結果與遮罩核對都從這裡讀，一定要填。
    us_run_name: str = ""
    two_stage: list = field(default_factory=lambda: list(TWO_STAGE))
    # 套件版本與舊 run 不同時預設直接停止（新舊結果會放在同一張表比較）；確定要繼續才改 True
    allow_lib_mismatch: bool = False
    # 舊 run 記錄的 occ_tune_us.py 程式指紋與現在不同、又不是已確認數值相同的上一版時，預設停止；確定要繼續才改 True
    allow_ref_code_mismatch: bool = False
    # True：每格的數值抽核（見 numeric_spot_check）不通過就不存檔、寫進 errors.log。
    # False：不通過也存檔，但統整時整份標 PARTIAL。
    strict_numeric: bool = True


# ═════════════════════════ 要跑哪些空間 / 策略 ═══════════════════════════
def active_spaces(cfg):
    return [sp for sp in SPACES if SPACE_STUDY2[sp] in cfg.studies]


def active_two_stage(cfg):
    return [s for s in TWO_STAGE if s in cfg.two_stage]            # 固定順序


def n_params(cfg):
    return ens.n_params(cfg)


# ═════════════════════════ 與舊 I~O run 的銜接（usens 也共用）════════════════
def ref_cfg(cfg, studies=None):
    """指向舊 I~O run 的設定物件（只用來算路徑與簽章，不會拿去跑）。"""
    return us.UsConfig(data_root=cfg.data_root, out_root=cfg.out_root, run_name=cfg.us_run_name,
                       n_jobs=cfg.n_jobs, datasets=list(cfg.datasets),
                       expected_datasets=list(cfg.expected_datasets), exclude_datasets=list(cfg.exclude_datasets),
                       studies=list(studies) if studies else ["I", "J", "K", "M", "N", "O"],
                       ae_types=list(cfg.ae_types), configs=list(cfg.configs), ae_epochs=cfg.ae_epochs,
                       grid=cfg.grid, lof_train_score=cfg.lof_train_score, use_df_cache=cfg.use_df_cache,
                       exclude_categorical=cfg.exclude_categorical, enn_k=cfg.enn_k, cnn_k=cfg.cnn_k,
                       cnn_seed=cfg.cnn_seed)


def ref_cell_path(cfg, task):
    return us.cell_path(ref_cfg(cfg), task)


REF_SAME_KEYS = ["grid", "legacy_default", "ae", "lof_train_score", "threshold_pct", "sampler", "metrics",
                 "n_folds", "strategies"]


def ref_problems(cfg, need, datasets=None, profile=None, tasks=None):
    """檢查舊 I~O run 能不能當對照。回傳 (會擋下來的問題, 只需註記的資訊)。
    need = {空間: [需要的策略]}；datasets / profile / tasks 有給才檢查資料指紋與格子是否齊全。"""
    problems, info = [], []
    if not cfg.us_run_name:
        return ["沒有設定 us_run_name（舊的 I~O run 名稱，例如 US_azure_v2）"], info
    ref_dir = Path(cfg.out_root) / cfg.us_run_name
    sig_file = ref_dir / "signature.json"
    if not sig_file.exists():
        return [f"找不到 {sig_file}（us_run_name 或 out_root 是否與舊 I~O run 相同？）"], info
    saved = json.loads(sig_file.read_text(encoding="utf-8"))
    old = saved.get("hard", {})
    old_studies = old.get("studies", [])
    now = json.loads(json.dumps(us.us_signature(ref_cfg(cfg, old_studies)), default=str))
    diff = [k for k in REF_SAME_KEYS if old.get(k) != now.get(k)]
    if diff:
        problems.append(f"本次設定與舊 run（{cfg.us_run_name}）不同：{diff}（參數格、AE、sampler、門檻必須完全相同才能放在一起比）")
    for k, lab in (("ae_types", "AE 種類"), ("configs", "AE config")):
        extra = [x for x in getattr(cfg, k) if x not in old.get(k, [])]
        if extra:
            problems.append(f"本次的{lab} {extra} 不在舊 run 裡")
    if old.get("libs") != now.get("libs"):
        msg = f"套件版本與舊 run 不同：舊 {old.get('libs')}｜現在 {now.get('libs')}"
        (info if cfg.allow_lib_mismatch else problems).append(
            msg + ("（已設定 allow_lib_mismatch=True，繼續執行）" if cfg.allow_lib_mismatch
                   else "（新舊結果要放在同一張表比較；請用跑舊 run 的同一個環境，或確定要繼續時設 allow_lib_mismatch=True）"))
    if old.get("code") != now.get("code"):
        known = ens.COMPAT_PREVIOUS.get("us_version", {})          # 已逐位元組確認數值相同的上一版
        if known.get(old.get("us_version")) == old.get("code"):
            info.append(f"舊 run 由上一版程式（{old.get('us_version')}）產生；該版每一格的數值算法已確認與現行版相同")
        else:
            msg = (f"舊 run 記錄的 occ_tune_us.py 程式指紋（{old.get('us_version')} / {old.get('code')}）"
                   f"與現在的（{now.get('us_version')} / {now.get('code')}）不同，也不是已確認相容的上一版")
            (info if cfg.allow_ref_code_mismatch else problems).append(
                msg + ("（已設定 allow_ref_code_mismatch=True，繼續執行；每格仍有遮罩與數值抽核）"
                       if cfg.allow_ref_code_mismatch
                       else "：保留集合相同不代表 OCC 分數與門檻的算法也相同。請用跑舊 run 時的那份 occ_tune_us.py；"
                            "確定只是改了統整／說明文字時，才設 allow_ref_code_mismatch=True"))
    have_cfg = ref_cfg(cfg, old_studies)
    for sp, sts in need.items():
        have = us.active_strategies(have_cfg, sp) if sp in us.active_spaces(have_cfg) else []
        miss = [s for s in sts if s not in have]
        if miss:
            problems.append(f"舊 run 在 {SPACE_LABEL[sp]} 空間沒有跑這些策略：{miss}（舊 run 的 studies={old_studies}）")
    if profile is not None and datasets is not None:
        fps = saved.get("fingerprints", {})
        now_fp = dict(zip(profile["Dataset"], profile["fingerprint"]))
        bad = [d for d in datasets if fps.get(d) != now_fp.get(d)]
        if bad:
            problems.append(f"這些資料集不在舊 run 裡，或檔案內容與舊 run 不同：{bad}")
    if tasks is not None:
        miss = [t for t in tasks if not ref_cell_path(cfg, t).exists()]
        if miss:
            problems.append(f"舊 run 還缺 {len(miss)}/{len(tasks)} 格結果（例：{'|'.join(map(str, miss[0][:5]))}）；"
                            "請先把舊 run 跑完")
    if not list(ref_dir.glob(f"{cfg.us_run_name}_summary.xlsx")):
        info.append(f"舊 run 資料夾裡沒有正式的 {cfg.us_run_name}_summary.xlsx（可能尚未統整或為 PARTIAL）")
    return problems, info


def ref_dataset_count(cfg):
    """舊 I~O run 納入的資料集數（讀不到回傳 None）。用來判斷本次是不是只跑了其中一部分。"""
    f = Path(cfg.out_root) / cfg.us_run_name / "dataset_profile.csv"
    try:
        prof = pd.read_csv(f)
        return int((prof["Included"].astype(str).str.lower() == "true").sum())
    except Exception:
        return None


def check_ref_run(cfg, need, datasets, profile, tasks):
    problems, info = ref_problems(cfg, need, datasets, profile, tasks)
    for s in info:
        print("ℹ️  " + s)
    if problems:
        raise SystemExit("\n✋ 舊的 I~O run 不能直接當對照，程式停止：\n   - " + "\n   - ".join(problems))
    print(f"舊 run 檢查通過：{cfg.us_run_name}（設定、資料指紋相同，{len(tasks)} 格結果齊全）")


def load_ref_cell(cfg, task):
    """讀舊 run 同一格的結果。回傳 {(Space, Strategy): {"sig", "status", "mets": {(OCC, ParamID): 指標}}}。"""
    path = ref_cell_path(cfg, task)
    if not path.exists():
        raise RuntimeError(f"舊 run 缺這一格：{path}")
    df = pd.read_csv(path, dtype={"ParamID": str, "MaskSig": str, "Space": str, "Strategy": str,
                                  "OCC": str, "Status": str})
    out = {}
    cols = METRIC_COLS + EXTRA_COLS
    for (sp, st), g in df.groupby(["Space", "Strategy"], sort=False):
        mets = {(o, p): dict(zip(cols, vals))
                for o, p, *vals in g[["OCC", "ParamID"] + cols].itertuples(index=False, name=None)}
        out[(sp, st)] = {"sig": str(g["MaskSig"].iloc[0]), "status": str(g["Status"].iloc[0]), "mets": mets}
    return out


def assert_ref_mask(ref, space, strategy, keep, what):
    """本次重算的遮罩必須與舊 run 同一格存下的指紋相同。"""
    r = ref.get((space, strategy))
    if r is None:
        raise RuntimeError(f"{what}：舊 run 這一格沒有 {space}/{strategy}")
    sig = us.mask_sig(keep) if keep is not None else "-"
    if sig != r["sig"]:
        raise RuntimeError(f"{what}：{space}/{strategy} 的保留集合與舊 run 不同（本次 {sig}，舊 run {r['sig']}）。"
                           "代表資料、AE 特徵或 sampler 與舊 run 不一致，這一格不存檔")
    return r


def space_arrays(space, fd, DF_maj=None, DF_min=None, DF_tst=None):
    """回傳 (maj_s, min_s, tst_s)：座標與 occ_tune_us.run_us_task 完全相同。"""
    if space == "OF":
        return fd["X_maj_s"], fd["X_min_s"], fd["X_tst_s"]
    if space == "DF":                                    # 與 B / J / N 相同：以未清理的 DF_maj fit MinMax
        sc = MinMaxScaler().fit(DF_maj)
        return sc.transform(DF_maj), sc.transform(DF_min), sc.transform(DF_tst)
    Zm, Zn, Zt = (np.hstack([fd["X_maj_s"], DF_maj]), np.hstack([fd["X_min_s"], DF_min]),
                  np.hstack([fd["X_tst_s"], DF_tst]))     # 與 C / K / O 相同：串接後以未清理的 Z_maj fit MinMax
    sc = MinMaxScaler().fit(Zm)
    return sc.transform(Zm), sc.transform(Zn), sc.transform(Zt)


def load_df_strict(cfg, ds, fold, ae, c, fd):
    """嚴格讀 A~C 留下的 AE 特徵快取，回傳 (DF_maj, DF_tst, DF_min, "cache")；數值與 core.get_df / us.get_df3 相同。
    core.get_df 遇到壞檔或資料指紋不符時會「默默重新訓練 AE」並覆寫快取；這裡改成直接丟例外：
    這一格不存檔、原因寫進 errors.log，共享快取也不會被改寫。
    只有 allow_ae_retrain=True（或 use_df_cache=False）時才走舊的可重訓路徑。"""
    if getattr(cfg, "allow_ae_retrain", False) or not cfg.use_df_cache:
        return us.get_df3(cfg, ds, fold, ae, c, fd)
    path = us.df_cache_path(cfg, ds, fold, ae, c)
    if not path.exists():
        raise RuntimeError(f"AE 特徵快取不存在：{path}（請先確認 A~C 已跑完、out_root 相同）")
    try:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            arr = {k: z[k] for k in ("DF_maj", "DF_tst", "DF_min")}
    except Exception as e:
        raise RuntimeError(f"AE 特徵快取讀不進來或缺欄位（{path}）：{type(e).__name__}: {e}。"
                           "不會自動重訓；請從備份還原這個檔，或確定要重訓時設 allow_ae_retrain=True") from e
    bad = []
    if meta.get("train_sha1") != fd["train_sha1"] or meta.get("test_sha1") != fd["test_sha1"]:
        bad.append("資料檔指紋與現在的 .dat 不同")
    want = {"DF_maj": fd["n_maj"], "DF_min": fd["n_min"], "DF_tst": len(fd["y_tst"])}
    for k, n in want.items():
        if arr[k].ndim != 2 or len(arr[k]) != n:
            bad.append(f"{k} 形狀 {arr[k].shape}，應有 {n} 列")
        elif not np.isfinite(arr[k]).all():
            bad.append(f"{k} 含非有限值")
    if len({a.shape[1] for a in arr.values() if a.ndim == 2}) > 1:
        bad.append("三個陣列的欄數不同")
    if bad:
        raise RuntimeError(f"AE 特徵快取不符（{path}）：{'；'.join(bad)}。不會自動重訓")
    return (arr["DF_maj"].astype(np.float64), arr["DF_tst"].astype(np.float64),
            arr["DF_min"].astype(np.float64), "cache")


def metrics_diff(a, b):
    """兩組指標（AUC / F1 / Recall / G-mean / AP）的最大絕對差；任何一個不是有限值 → inf。"""
    d = 0.0
    for k in METRIC_COLS + EXTRA_COLS:
        x, y = a.get(k, np.nan), b.get(k, np.nan)
        if not (np.isfinite(x) and np.isfinite(y)):
            return float("inf")
        d = max(d, abs(float(x) - float(y)))
    return d


def numeric_spot_check(space, maj_s, tst_s, y, ctrl, ref, cfg):
    """數值抽核：用舊預設參數，在「單次 ENN」（ENN 無效時改用 none）留下的樣本上重新訓練三個 OCC，
    五項指標與舊 run 同一列比較。回傳 (抽核的策略, 最大差)。
    目的：遮罩相同只代表「留下的樣本」相同；這一步確認現在的程式與環境算出的 OCC 分數、門檻、指標
    也和舊 run 相同，新算的列才能和沿用／對照的舊列放在一起比。每個空間每格多 3 次訓練。"""
    st = next((s for s in ("ENN", "none") if ctrl[s][1] == "ok"), None)
    if st is None:
        return "-", np.nan
    Xtr, worst = maj_s[ctrl[st][0]], 0.0
    for occ in OCC_TYPES:
        p = next(q for q in param_combos(occ, cfg.grid) if is_legacy_default(occ, q))
        s_tr, s_te, _ = core._fit_score(occ, p, Xtr, tst_s, cfg.lof_train_score)
        m = _prep(np.asarray(s_tr, dtype=np.float64), np.asarray(s_te, dtype=np.float64))
        old = ref[(space, st)]["mets"].get((occ, param_id(p)), {})
        worst = max(worst, metrics_diff(_metrics(y, m["pred"], m["s_te"]), old))
    return st, worst


# ═════════════════════════ 第二階段 ENN ═══════════════════════════════════
def second_stage_enn(maj_s, min_s, keep1, cfg):
    """第一階段留下的多數類（keep1）＋全部少數類 → 再跑一次 ENN，只刪多數類。
    回傳對應「原始多數類列」的 boolean mask；防呆條件不成立時回傳 None。
    ・sampling_strategy=[0]：明確指定只清多數類（標籤 0）。第一階段若刪到多數類比少數類還少，
      'auto' 會改去清「另一類」，所以這裡不能用 'auto'。多數類比少數類多時兩者結果相同。
    ・座標沿用第一階段（不重新 fit MinMax、不重訓 AE）；只跑一次，不是反覆清到停止的 RENN。"""
    idx1 = np.flatnonzero(keep1)
    n1 = len(idx1)
    X = np.vstack([maj_s[idx1], min_s])
    y = np.array([0] * n1 + [1] * len(min_s))
    s = EditedNearestNeighbours(n_neighbors=cfg.enn_k, kind_sel="all", sampling_strategy=[0])
    try:
        s.fit_resample(X, y)
    except Exception as e:
        raise RuntimeError(f"第二階段 ENN 執行失敗：{type(e).__name__}: {e}") from e
    idx = np.asarray(s.sample_indices_)
    loc = idx[idx < n1]
    if int(np.sum(idx >= n1)) != len(min_s) or len(loc) != len(np.unique(loc)):
        return None                                      # 少數類被改動或索引重複：不應發生
    keep2 = np.zeros(len(maj_s), dtype=bool)
    keep2[idx1[loc]] = True                              # 映射回原始多數類的列
    return keep2


def _jaccard(a, b):
    u = int(np.sum(a | b))
    return float(np.sum(a & b) / u) if u else 1.0


# ═════════════════════════ 一個空間的完整評估 ═════════════════════════════
def eval_space2(space, maj_s, min_s, tst_s, y, cfg, base, ref):
    """回傳 (結果列, 遮罩診斷列)。結果列只有新的兩階段策略；對照組不重算（統整時讀舊 run）。"""
    n_maj, n_min = len(maj_s), len(min_s)
    masks, ok = us.compute_keep_masks(maj_s, min_s, cfg)
    combos = [(occ, p) for occ in OCC_TYPES for p in param_combos(occ, cfg.grid)]
    nan_m = {k: np.nan for k in METRIC_COLS + EXTRA_COLS}
    ctrl, mask_rows = {}, []
    enn_keep = None

    def mrow(st, role, stage1, n1, keep, status, equal):
        n_kept = int(keep.sum()) if keep is not None else np.nan
        e2 = ctrl2.get("2S_ENN_ENN")
        return {"Space": space, "Strategy": st, "Role": ROLE_LABEL[role], **base, "NMaj": n_maj, "NMin": n_min,
                "Stage1": stage1, "Stage1Kept": n1, "MajKept": n_kept,
                "RemovedRate": (n_maj - n_kept) / n_maj if keep is not None else np.nan,
                "MaskSig": us.mask_sig(keep) if keep is not None else "-", "Status": status, "EqualTo": equal,
                "JaccardENN": _jaccard(keep, enn_keep) if keep is not None and enn_keep is not None else np.nan,
                "SubsetOfENN": bool(np.all(enn_keep[keep])) if keep is not None and enn_keep is not None else np.nan,
                "ContainsENN_ENN": bool(np.all(keep[e2])) if keep is not None and e2 is not None else np.nan,
                "NumCheckDiff": num_diff if st == num_st else np.nan}

    ctrl2 = {}                                           # 新策略的最終遮罩
    # ── 對照組：只重算遮罩，並與舊 run 核對 ──
    for st in REF_STRATEGIES:
        keep, status = us.apply_strategy(masks, ok, st, n_maj)
        r = assert_ref_mask(ref, space, st, keep, f"{base['Dataset']} fold{base['Fold']}")
        if keep is not None and int(keep.sum()) < MIN_KEEP:
            status = "degenerate"
        if status != r["status"]:
            raise RuntimeError(f"{space}/{st} 的狀態與舊 run 不同（本次 {status}，舊 run {r['status']}）")
        ctrl[st] = (keep, status)
        if st == "ENN" and status == "ok":
            enn_keep = keep
    num_st, num_diff = numeric_spot_check(space, maj_s, tst_s, y, ctrl, ref, cfg)
    if cfg.strict_numeric and not (num_diff < NUM_TOL):
        raise RuntimeError(f"{base['Dataset']} fold{base['Fold']} {space}：數值抽核未通過——用舊預設參數重算 {num_st} 的 OCC，"
                           f"五項指標與舊 run 的最大差 {num_diff:.3g}（應 < {NUM_TOL:g}）。代表現在的程式或環境算出的 "
                           "OCC 數值與舊 run 不同，這一格不存檔")
    for st in REF_STRATEGIES:
        keep, status = ctrl[st]
        mask_rows.append(mrow(st, "ref", "-", np.nan, keep, status, "-"))

    # ── 兩階段策略 ──
    rows, seen = [], {}                                   # seen：遮罩指紋 → (策略名, 指標)
    for st in ctrl:                                       # 對照的遮罩先登記：相同就沿用舊 run 的結果
        keep, status = ctrl[st]
        if status == "ok":
            seen.setdefault(us.mask_sig(keep), (st, ref[(space, st)]["mets"]))
    for st in active_two_stage(cfg):
        src, role = TWO_STAGE[st]
        keep1, s1 = ctrl[src]
        n1 = int(keep1.sum()) if keep1 is not None else np.nan
        keep2, status = None, s1                          # 第一階段無效 / 退化 → 這個策略這格也是
        if s1 == "ok":
            keep2 = second_stage_enn(maj_s, min_s, keep1, cfg)
            status = "invalid_fallback" if keep2 is None else "ok"
            if keep2 is not None and int(keep2.sum()) < MIN_KEEP:
                status = "degenerate"
        reuse, mets = "-", None
        sig = us.mask_sig(keep2) if keep2 is not None else "-"
        n_kept = int(keep2.sum()) if keep2 is not None else np.nan
        if status == "ok":
            if sig in seen:
                reuse, mets = seen[sig]
            else:
                sc = occ_scores_all(maj_s[keep2], tst_s, cfg)
                mets = {}
                for occ, p in combos:
                    pid = param_id(p)
                    m = _prep(*sc[(occ, pid)])
                    mets[(occ, pid)] = _metrics(y, m["pred"], m["s_te"])
                seen[sig] = (st, mets)
            ctrl2[st] = keep2
        rr = (n_maj - n_kept) / n_maj if keep2 is not None else np.nan
        for occ, p in combos:
            pid = param_id(p)
            rows.append({"Study": SPACE_STUDY2[space], "Space": space, "Strategy": st, **base,
                         "OCC": occ, "ParamID": pid, "IsDefault": is_legacy_default(occ, p),
                         "MajKept": n_kept, "RemovedRate": rr, "MaskSig": sig, "ReuseFrom": reuse,
                         "Status": status, **(mets[(occ, pid)] if mets is not None else nan_m),
                         "Stage1": src, "Stage1Kept": n1})
        mask_rows.append(mrow(st, role, src, n1, keep2, status, reuse))
    return rows, mask_rows


def n_rows_space(cfg):
    return len(active_two_stage(cfg)) * n_params(cfg)


# ═════════════════════════ 工作格 ══════════════════════════════════════════
def cell_path(cfg, task):
    return us.cell_path(cfg, task)                       # cells/OF/... 或 cells/DF/...（與 occ_tune_us 相同規則）


def mask_path(cfg, task):
    p = cell_path(cfg, task)
    return p.parent.parent.parent / (p.parent.parent.name + "m") / p.parent.name / p.name


def _write(rows, cols, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    pd.DataFrame(rows, columns=cols).to_csv(tmp, index=False, encoding="utf-8", compression="gzip")
    os.replace(tmp, path)


def _check(rows, expected, what):
    if len(rows) != expected:
        raise RuntimeError(f"{what} 列數不符：{len(rows)} ≠ {expected}")
    bad = [r for r in rows if r["Status"] == "ok" and not np.isfinite(r["AUC"])]
    if bad:
        raise RuntimeError(f"{what} 有 {len(bad)} 列 AUC 無效，整格不存檔")


def run_us2_task(task, cfg):
    torch.set_num_threads(max(1, cfg.torch_threads_per_job))
    t0 = time.perf_counter()
    kind, ds, fold = task[0], task[1], task[2]
    fd = core.load_fold(cfg.data_root, ds, fold)
    y = fd["y_tst"]
    ref = load_ref_cell(cfg, task)
    rows, mrows, src = [], [], "-"
    if kind == "OF":
        base = {"Dataset": ds, "Fold": fold, "AE": "-", "Config": "-"}
        rows, mrows = eval_space2("OF", *space_arrays("OF", fd), y, cfg, base, ref)
        expected = n_rows_space(cfg)
    else:
        ae, c = task[3], task[4]
        DF_maj, DF_tst, DF_min, src = load_df_strict(cfg, ds, fold, ae, c, fd)
        base = {"Dataset": ds, "Fold": fold, "AE": ae, "Config": c}
        expected = 0
        for sp in [s for s in active_spaces(cfg) if s != "OF"]:
            r, m = eval_space2(sp, *space_arrays(sp, fd, DF_maj, DF_min, DF_tst), y, cfg, base, ref)
            rows += r
            mrows += m
            expected += n_rows_space(cfg)
    _check(rows, expected, kind)
    _write(mrows, MASK_COLS, mask_path(cfg, task))       # 先寫遮罩診斷
    _write(rows, US2_COLS, cell_path(cfg, task))         # 最後寫主檔 ＝ 這格完成
    return {"task": "|".join(map(str, task)), "kind": kind, "dataset": ds, "n_maj": int(fd["n_maj"]),
            "seconds": round(time.perf_counter() - t0, 3), "df_source": src, "host": socket.gethostname()}


def us2_signature(cfg):
    return {"us2_version": US2_VERSION, "studies": sorted(cfg.studies), "two_stage": active_two_stage(cfg),
            "ae_types": cfg.ae_types, "configs": cfg.configs, "grid": cfg.grid, "legacy_default": LEGACY_DEFAULT,
            "ae": core.ae_signature(cfg), "lof_train_score": cfg.lof_train_score,
            "threshold_pct": THRESHOLD_PCT, "cat_encoding": "train_fold_mapping_v1",
            "metrics": METRIC_COLS + EXTRA_COLS, "n_folds": N_FOLDS, "ref_run": cfg.us_run_name,
            "sampler": {"voters": us.VOTERS, "enn_k": cfg.enn_k, "cnn_k": cfg.cnn_k, "cnn_seed": cfg.cnn_seed,
                        "strategy": "auto", "min_keep": MIN_KEEP, "strict_fallback": True,
                        "stage2": {"method": "ENN", "n_neighbors": cfg.enn_k, "kind_sel": "all",
                                   "target": "majority_label_0", "passes": 1}},
            "two_stage_defs": {k: list(v) for k, v in TWO_STAGE.items()},
            "ref_strategies": REF_STRATEGIES,
            "numeric_check": {"on": "ENN_else_none", "theta": "legacy_default", "tol": NUM_TOL,
                              "strict": bool(cfg.strict_numeric)},
            "libs": {**lib_versions(), "imblearn": imblearn.__version__},
            "code": code_fingerprint(core._fit_score, core.get_df, occ_scores_all, _prep, _metrics,
                                     us.make_sampler, us.compute_keep_masks, us.apply_strategy, us.mask_sig,
                                     us.get_df3, second_stage_enn, eval_space2, space_arrays, load_ref_cell,
                                     assert_ref_mask, run_us2_task, active_spaces, active_two_stage,
                                     load_df_strict, metrics_diff, numeric_spot_check)}


US2_SUMMARY_KEYS = ["us2_version", "studies", "two_stage", "ae_types", "configs", "grid", "legacy_default",
                    "lof_train_score", "threshold_pct", "metrics", "n_folds", "sampler", "two_stage_defs",
                    "ref_strategies", "ref_run", "numeric_check"]


def build_tasks(cfg, datasets, profile):
    n_maj = dict(zip(profile["Dataset"], profile["n_majority_fold1"]))
    big_first = lambda t: -n_maj.get(t[1], 0)
    sp = active_spaces(cfg)
    phases = []
    if "OF" in sp:
        phases.append(("OF", sorted([("OF", ds, f) for ds in datasets for f in range(1, N_FOLDS + 1)],
                                    key=big_first)))
    if {"DF", "OFDF"} & set(sp):
        phases.append(("DF", sorted([("DF", ds, f, ae, c) for ds in datasets for f in range(1, N_FOLDS + 1)
                                     for ae in cfg.ae_types for c in cfg.configs], key=big_first)))
    return phases


def run(cfg: Us2Config):
    bad = set(cfg.studies) - set(STUDY_SPACE2)
    if bad or not cfg.studies:
        raise SystemExit(f"studies 只能是 S / T / U，收到 {cfg.studies}")
    bad = set(cfg.two_stage) - set(TWO_STAGE)
    if bad or not cfg.two_stage:
        raise SystemExit(f"two_stage 只能是 {list(TWO_STAGE)}，收到 {cfg.two_stage}")
    sp = active_spaces(cfg)
    datasets, profile, run_dir = prepare_run(cfg, f"兩階段 under-sampling（第二階段 ENN）studies={cfg.studies}",
                                             us2_signature, need_df=bool({"DF", "OFDF"} & set(sp)))
    phases = build_tasks(cfg, datasets, profile)
    check_ref_run(cfg, {s: REF_STRATEGIES for s in sp}, datasets, profile, [t for _, ts in phases for t in ts])
    print("空間：" + "、".join(f"{SPACE_STUDY2[s]}＝{SPACE_LABEL[s]}" for s in sp)
          + f"｜兩階段策略：{active_two_stage(cfg)}｜對照（讀舊 run，不重算）：{REF_STRATEGIES}")
    lock = core.acquire_lock(run_dir)
    try:
        core.cleanup_tmp(cfg)
        for name, tasks in phases:
            if run_phase(name, tasks, cfg, run_dir, run_us2_task, cell_path) == "interrupted":
                return "interrupted"
    finally:
        core.release_lock(lock)
    return "done"


# ═════════════════════════ 統整：讀檔 ═════════════════════════════════════
def _dtypes(cfg, datasets):
    pids = sorted({param_id(p) for o in OCC_TYPES for p in param_combos(o, cfg.grid)})
    return {"Study": pd.CategoricalDtype(list("ABCIJKMNOSTU")), "Space": pd.CategoricalDtype(SPACES),
            "Strategy": pd.CategoricalDtype(us.STRATEGIES + list(TWO_STAGE)),
            "Dataset": pd.CategoricalDtype(list(datasets)),
            "AE": pd.CategoricalDtype(list(cfg.ae_types) + ["-"]),
            "Config": pd.CategoricalDtype(list(cfg.configs) + ["-"]),
            "OCC": pd.CategoricalDtype(OCC_TYPES), "ParamID": pd.CategoricalDtype(pids),
            "ReuseFrom": pd.CategoricalDtype(us.STRATEGIES + list(TWO_STAGE) + ["-"]),
            "Status": pd.CategoricalDtype(us.STATUSES), "Stage1": pd.CategoricalDtype(us.STRATEGIES + ["-"]),
            "Fold": "int16"}


LOAD_COLS = [c for c in US2_COLS if c != "MaskSig"]
KEY8 = ["Space", "Strategy", "Dataset", "Fold", "AE", "Config", "OCC", "ParamID"]


def _cell_files(root, datasets, kinds):
    return [f for k in kinds for ds in datasets for f in sorted((root / k / safe_name(ds)).glob("*.csv.gz"))]


def load_new_rows(cfg, datasets):
    root = Path(cfg.out_root) / cfg.run_name / "cells"
    files = _cell_files(root, datasets, ("OF", "DF"))
    if not files:
        return pd.DataFrame(columns=LOAD_COLS), 0
    dt = _dtypes(cfg, datasets)
    df = pd.concat((pd.read_csv(f, usecols=LOAD_COLS, dtype=dt) for f in files), ignore_index=True)
    df["IsDefault"] = df["IsDefault"].astype(str).str.lower().eq("true")
    key = ["Study", "Space", "Strategy", "Dataset", "AE", "Config", "OCC", "ParamID", "Status"]
    n_unknown = int(df[key].isna().any(axis=1).sum())
    if n_unknown:
        print(f"⚠️  {n_unknown} 列的名稱不在本次設定內（可能是舊結果或設定被改過），已略過 → 結果標 PARTIAL")
        df = df.dropna(subset=key)
    df = df[df["Space"].isin(active_spaces(cfg)) & df["Strategy"].isin(active_two_stage(cfg))]
    return df.reset_index(drop=True), n_unknown


def load_ref_rows(cfg, datasets, strategies=REF_STRATEGIES):
    """舊 I~O run 的對照列（只留本次用到的空間、策略、AE、config）。"""
    root = Path(cfg.out_root) / cfg.us_run_name / "cells"
    sp = active_spaces(cfg)
    files = _cell_files(root, datasets, [k for k in ("OF", "DF")
                                         if (k == "OF" and "OF" in sp) or (k == "DF" and {"DF", "OFDF"} & set(sp))])
    if not files:
        return pd.DataFrame(columns=LOAD_COLS)
    dt = _dtypes(cfg, datasets)
    cols = [c for c in LOAD_COLS if c in us.US_COLS]
    parts = []
    for f in files:
        d = pd.read_csv(f, usecols=cols, dtype=dt)
        parts.append(d[d["Strategy"].isin(strategies) & d["Space"].isin(sp)])
    df = pd.concat(parts, ignore_index=True)
    df["IsDefault"] = df["IsDefault"].astype(str).str.lower().eq("true")
    df = df.dropna(subset=["Study", "Space", "Strategy", "Dataset", "AE", "Config", "OCC", "ParamID", "Status"])
    df["Stage1"] = pd.Categorical(["-"] * len(df), categories=dt["Stage1"].categories)
    df["Stage1Kept"] = np.nan
    return df[LOAD_COLS].reset_index(drop=True)


def load_mask_rows(cfg, datasets):
    root = Path(cfg.out_root) / cfg.run_name / "cells"
    files = [f for k in ("OF", "DF") for ds in datasets
             for f in sorted((root / f"{k}m" / safe_name(ds)).glob("*.csv.gz"))
             if (root / k / safe_name(ds) / f.name).exists()]                    # 只收主檔已完成的格子
    if not files:
        return pd.DataFrame(columns=MASK_COLS)
    df = pd.concat((pd.read_csv(f, dtype={"MaskSig": str, "Config": str, "AE": str, "Strategy": str,
                                          "Space": str, "EqualTo": str, "Status": str, "Stage1": str})
                    for f in files), ignore_index=True)
    return df[df["Space"].isin(active_spaces(cfg))].reset_index(drop=True)


# ═════════════════════════ 統整：比較表 ═══════════════════════════════════
_ORDER = {s: i for i, s in enumerate(ALL_STRATEGIES)}
_SP = {**{s: i for i, s in enumerate(SPACES)}, **{SPACE_LABEL[s]: i for i, s in enumerate(SPACES)}}
_OC = {o: i for i, o in enumerate(OCC_TYPES)}


def _sorted(df, cols=("Space", "OCC", "AE", "Strategy")):
    maps = {"Space": _SP, "OCC": _OC, "Strategy": _ORDER}
    cols = [c for c in cols if c in df.columns]
    return df.sort_values(cols, key=lambda s: s.astype(str).map(maps[s.name]) if s.name in maps
                          else s.astype(str), kind="stable").reset_index(drop=True)


def _stat_rows(m, label, ref_name, extra=None):
    """m：每列一個資料集，欄位 Space/Strategy/AE/OCC/Dataset/AUC/AUC_ref。統計單位＝資料集。"""
    if m.empty:
        return pd.DataFrame()
    m = m.assign(D=m["AUC"] - m["AUC_ref"])
    rows = []
    for k, g in m.groupby(["Space", "Strategy", "AE", "OCC"], sort=False, observed=True):
        d = g["D"].to_numpy(dtype=float)
        p, n_nz, how = us.wilcoxon_p(d)
        r = {"比較": label, "Space": SPACE_LABEL.get(str(k[0]), str(k[0])), "Strategy": str(k[1]),
             "角色": ROLE_LABEL[TWO_STAGE[str(k[1])][1]] if str(k[1]) in TWO_STAGE else ROLE_LABEL["ref"],
             "AE": str(k[2]), "OCC": str(k[3]), "對照": ref_name, "共同有效資料集數": len(d),
             "AUC_策略": float(g["AUC"].mean()), "AUC_對照": float(g["AUC_ref"].mean()),
             "平均ΔAUC": float(d.mean()), "中位數ΔAUC": float(np.median(d)),
             "勝": int((d > TIE_TOL).sum()), "平": int((np.abs(d) <= TIE_TOL).sum()), "負": int((d < -TIE_TOL).sum()),
             "檢定用資料集數（去掉平手）": n_nz, "Wilcoxon_p": p, "檢定方法": how}
        if extra:
            r.update(extra)
        rows.append(r)
    out = pd.DataFrame(rows)
    out["Holm_p"] = np.nan
    for _, idx in out.groupby(["Space", "AE", "OCC"]).groups.items():
        out.loc[idx, "Holm_p"] = us.holm(out.loc[idx, "Wilcoxon_p"].values)
    return out


def _ds_auc(best):
    keys = ["Space", "Strategy", "AE", "OCC", "Dataset"]
    return best.groupby(keys, observed=True)["AUC"].mean().reset_index()


COMPARE_REFS = [("none", "vs none（清理有沒有幫助）"),
                ("S2_Majority", "vs S2_Majority（第二階段 ENN 有沒有補救效果）"),
                ("ENN", "vs 單次 ENN（先集成再 ENN 有沒有額外價值）"),
                ("2S_ENN_ENN", "vs 2S_ENN_ENN（是不是只來自多清一次）")]


def comparisons_tuned(best, cfg):
    """各策略各自調參（config 與 θ 都逐資料集挑，oracle）後的逐資料集配對比較。"""
    ds = _ds_auc(best)
    on = ["Space", "AE", "OCC", "Dataset"]
    parts = []
    for ref, label in COMPARE_REFS:
        r = ds[ds["Strategy"] == ref][on + ["AUC"]].rename(columns={"AUC": "AUC_ref"})
        cand = ds[ds["Strategy"].isin([s for s in active_two_stage(cfg) if s != ref])]
        if r.empty or cand.empty:
            continue
        parts.append(_stat_rows(cand.merge(r, on=on), "[各自調參] " + label, ref))
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def comparisons_same_setting(ok, cfg):
    """同一個 AE config、同一組 OCC 參數下的配對比較：設定固定為「對照自己調出來的那一組」，
    新策略不另外挑 → 差異只來自保留的樣本不同（設定對對照有利，是偏保守的比較）。"""
    key = ["Space", "AE", "OCC", "Dataset", "Config", "ParamID"]
    on = ["Space", "AE", "OCC", "Dataset"]
    parts = []
    for ref, label in COMPARE_REFS:
        rb, pick = _pick(ok[ok["Strategy"] == ref], G_US, ["Config", "ParamID"])
        if rb.empty:
            continue
        r = rb.groupby(on, observed=True)["AUC"].mean().rename("AUC_ref").reset_index()
        cands = [s for s in active_two_stage(cfg) if s != ref]
        c = ok[ok["Strategy"].isin(cands)].merge(pick[key], on=key)
        if c.empty:
            continue
        c = (c.groupby(["Strategy"] + on, observed=True)
              .agg(AUC=("AUC", "mean"), nF=("Fold", "nunique")).reset_index())
        c = c[c["nF"] == N_FOLDS]                          # 新策略在這組設定下 5 折都有效才算
        t = _stat_rows(c.merge(r, on=on), "[同 config 同 θ] " + label, ref)
        if len(t):
            parts.append(t)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def mask_summary(mk):
    rows = []
    for (sp, st), g in mk.groupby(["Space", "Strategy"], sort=False):
        v = g[g["Status"] == "ok"]
        two = st in TWO_STAGE
        r = {"Space": SPACE_LABEL[sp], "Strategy": st, "角色": g["Role"].iloc[0], "第一階段": g["Stage1"].iloc[0],
             "格子數": len(g), "有效格子數": len(v), "退化(<5筆)": int((g["Status"] == "degenerate").sum()),
             "防呆 fallback（應為 0）": int((g["Status"] == "invalid_fallback").sum()),
             "第一階段平均保留筆數": float(v["Stage1Kept"].mean()) if two and len(v) else np.nan,
             "最終平均保留筆數": float(v["MajKept"].mean()) if len(v) else np.nan,
             "最終平均刪除率": float(v["RemovedRate"].mean()) if len(v) else np.nan,
             "第二階段多刪的比例（佔第一階段保留）":
                 float(((v["Stage1Kept"] - v["MajKept"]) / v["Stage1Kept"]).mean()) if two and len(v) else np.nan,
             "與單次 ENN 的 Jaccard（平均）": float(v["JaccardENN"].mean()) if len(v) else np.nan,
             "保留集合 ⊆ 單次 ENN 的格子比例": float(v["SubsetOfENN"].astype(float).mean()) if len(v) else np.nan,
             "保留集合 ⊇ 2S_ENN_ENN 的格子比例":
                 float(v["ContainsENN_ENN"].astype(float).mean()) if len(v) and v["ContainsENN_ENN"].notna().any()
                 else np.nan,
             "沿用其他策略結果的格子比例（保留集合完全相同）": float((v["EqualTo"] != "-").mean()) if two and len(v) else np.nan}
        rows.append(r)
    return _sorted(pd.DataFrame(rows), ("Space", "Strategy")) if rows else pd.DataFrame()


def equivalence_table(mk):
    """兩兩策略保留集合完全相同的比例：每個（資料集×fold×AE×config×空間）只算一次，不因 59 組參數重複計數。"""
    v = mk[mk["Status"] == "ok"]
    if v.empty:
        return pd.DataFrame()
    wide = v.pivot_table(index=["Space", "Dataset", "Fold", "AE", "Config"], columns="Strategy",
                         values="MaskSig", aggfunc="first")
    rows = []
    for sp in SPACES:
        if sp not in wide.index.get_level_values(0):
            continue
        w = wide.xs(sp, level="Space")
        strats = [s for s in ALL_STRATEGIES if s in w.columns]
        for a, b in combinations(strats, 2):
            if a not in TWO_STAGE and b not in TWO_STAGE:
                continue                                    # 對照之間的等價率舊 run 已有（03b 分頁）
            both = w[[a, b]].dropna()
            if len(both):
                rows.append({"Space": SPACE_LABEL[sp], "策略 1": a, "策略 2": b, "兩者皆有效的格子數": len(both),
                             "保留集合完全相同的格子數": int((both[a] == both[b]).sum()),
                             "保留集合完全相同的比例": float((both[a] == both[b]).mean())})
    return pd.DataFrame(rows)


def build_log(cfg, datasets):
    rows = []
    root = Path(cfg.out_root) / cfg.run_name / "cells"
    sp = active_spaces(cfg)
    for ds in datasets:
        r, ok = {"Dataset": ds}, True
        if "OF" in sp:
            n = len(list((root / "OF" / safe_name(ds)).glob("*.csv.gz")))
            r["OF 完成/應有"] = f"{n}/{N_FOLDS}"
            ok = ok and n == N_FOLDS
        if {"DF", "OFDF"} & set(sp):
            e = N_FOLDS * len(cfg.ae_types) * len(cfg.configs)
            n = len(list((root / "DF" / safe_name(ds)).glob("*.csv.gz")))
            r["DF 完成/應有"] = f"{n}/{e}"
            ok = ok and n == e
        r["完整"] = ok
        rows.append(r)
    return pd.DataFrame(rows)


def subset_tag(cfg, datasets):
    """本次資料集比舊 I~O run 少（分批跑）時，回傳檔名用的標記（例 "_5of17ds"）與舊 run 的資料集數。"""
    n_ref = ref_dataset_count(cfg)
    return (f"_{len(datasets)}of{n_ref}ds" if n_ref and len(datasets) < n_ref else ""), n_ref


def scope_notes(cfg, datasets):
    """只需註記、不影響正式與否的範圍說明：分批跑、資料集太少時檢定做不出顯著。"""
    out = []
    tag, n_ref = subset_tag(cfg, datasets)
    n = len(datasets)
    if tag:
        out.append(f"本次只納入 {n} 個資料集（舊 I~O run 有 {n_ref} 個），屬階段性結果，檔名加上 {tag}；"
                   "把 runner 的 datasets 換成全部資料集、用同一個 run_name 再跑，只會補沒跑過的資料集")
    if n < us.MIN_PAIRS:
        out.append(f"資料集只有 {n} 個（< {us.MIN_PAIRS}）：不做 Wilcoxon 檢定，p 值欄全部留空")
    elif 2 / 2 ** n > 0.05:
        out.append(f"資料集只有 {n} 個：Wilcoxon（雙尾、精確分布）最小可能的 p ＝ {2 / 2 ** n:.4f}，不可能小於 0.05；"
                   "這一批請只看 ΔAUC 與勝/平/負的方向，不要解讀顯著性")
    return out


def summary_path(cfg, datasets, complete):
    tag, _ = subset_tag(cfg, datasets)
    return Path(cfg.out_root) / cfg.run_name / (f"{cfg.run_name}_summary{tag}" + ("" if complete else "_PARTIAL")
                                                + ".xlsx")


def legal_pairs(cfg):
    return {(o, param_id(p)) for o in OCC_TYPES for p in param_combos(o, cfg.grid)}


def pairs_ok(d, legal):
    """(OCC, ParamID) 的配對必須剛好是參數格裡的那 59 組（例如 LOF 不能配到 OCSVM 的參數）。"""
    if not len(d):
        return False
    got = set(d[["OCC", "ParamID"]].drop_duplicates().astype(str).itertuples(index=False, name=None))
    return got == legal


def key_audit(df, mk, refrows, cfg, datasets):
    """逐鍵清點（每個資料集 × 空間一列）。三個部分都通過才算 OK：
    ① 新策略結果列：列數、重複、名稱與 (OCC, ParamID) 配對都在預期內、ok 列 AUC 有限、沒有防呆 fallback。
       鍵是完整的乘積（策略×fold×AE×config×OCC×參數），所以「列數相符＋無重複＋名稱都合法」＝逐鍵完全相符。
    ② 遮罩診斷：每格每個策略剛好一列、與結果主檔的保留筆數／狀態／沿用對象一致、每格的數值抽核都通過。
    ③ 舊對照（讀自舊 I~O run）：none / ENN / S1 / S2 / S3 的 59 組結果在每一格都齊全、無重複。"""
    sts, n_p, legal = active_two_stage(cfg), n_params(cfg), legal_pairs(cfg)
    ckey = ["Strategy", "Fold", "AE", "Config"]
    as_str = {c: str for c in ckey}
    rows = []
    for sp in active_spaces(cfg):
        aes, cfgs = (["-"], ["-"]) if sp == "OF" else (list(cfg.ae_types), list(cfg.configs))
        n_cell = N_FOLDS * len(aes) * len(cfgs)
        d_sp, m_sp = df[df["Space"] == sp], mk[mk["Space"] == sp]
        r_sp = refrows[refrows["Space"] == sp] if len(refrows) else refrows
        for ds in datasets:
            d, m = d_sp[d_sp["Dataset"] == ds], m_sp[m_sp["Dataset"] == ds]
            r = r_sp[r_sp["Dataset"] == ds] if len(r_sp) else r_sp
            exp, exp_m, exp_r = n_cell * len(sts) * n_p, n_cell * (len(REF_STRATEGIES) + len(sts)), \
                n_cell * len(REF_STRATEGIES) * n_p
            dup = int(d.duplicated(KEY8).sum())
            names_ok = (set(d["Strategy"].astype(str)) <= set(sts) and set(d["AE"].astype(str)) <= set(aes)
                        and set(d["Config"].astype(str)) <= set(cfgs)
                        and set(d["Fold"].astype(int)) <= set(range(1, N_FOLDS + 1)) and pairs_ok(d, legal))
            okm = d["Status"].astype(str) == "ok"
            bad = int((okm & ~np.isfinite(pd.to_numeric(d["AUC"], errors="coerce"))).sum())
            n_fb = int((d["Status"].astype(str) == "invalid_fallback").sum())
            # ② 遮罩診斷
            m_dup = int(m.duplicated(ckey).sum())
            m_names = len(m) > 0 and set(m["Strategy"].astype(str)) == set(REF_STRATEGIES + sts)
            a = d.drop_duplicates(ckey)[ckey + ["MajKept", "Status", "ReuseFrom"]].astype(
                {**as_str, "Status": str, "ReuseFrom": str})
            b = m[m["Strategy"].isin(sts)][ckey + ["MajKept", "Status", "EqualTo"]].astype(
                {**as_str, "Status": str, "EqualTo": str})
            jn = a.merge(b, on=ckey, suffixes=("", "_m"))
            same = (len(jn) == len(a) == len(b) and len(a) == n_cell * len(sts)
                    and bool((jn["Status"] == jn["Status_m"]).all()) and bool((jn["ReuseFrom"] == jn["EqualTo"]).all())
                    and bool((jn["MajKept"].fillna(-1) == jn["MajKept_m"].fillna(-1)).all()))
            nc = pd.to_numeric(m["NumCheckDiff"], errors="coerce").dropna() if len(m) else pd.Series(dtype=float)
            n_pass = int((nc < NUM_TOL).sum())
            # ③ 舊對照
            r_ok = (len(r) == exp_r and int(r.duplicated(KEY8).sum()) == 0 and pairs_ok(r, legal)
                    and set(r["Strategy"].astype(str)) == set(REF_STRATEGIES))
            row = {"Dataset": ds, "Space": SPACE_LABEL[sp], "應有列數": exp, "實際列數": len(d),
                   "重複": dup, "名稱與參數配對都在預期內": names_ok, "ok 列": int(okm.sum()),
                   "ok：實際訓練": int((okm & (d["ReuseFrom"].astype(str) == "-")).sum()),
                   "ok：沿用相同保留集合": int((okm & (d["ReuseFrom"].astype(str) != "-")).sum()),
                   "degenerate 列": int((d["Status"].astype(str) == "degenerate").sum()),
                   "invalid_fallback 列（應為 0）": n_fb, "ok 但 AUC 非有限": bad,
                   "遮罩診斷列 應有/實際": f"{exp_m}/{len(m)}", "遮罩診斷重複鍵": m_dup,
                   "遮罩診斷與結果主檔一致": bool(same and m_names),
                   "數值抽核 通過/應有（格）": f"{n_pass}/{n_cell}",
                   "數值抽核最大差": float(nc.max()) if len(nc) else np.nan,
                   "舊對照列 應有/實際": f"{exp_r}/{len(r)}", "舊對照齊全且無重複": bool(r_ok)}
            row["OK"] = bool(len(d) == exp and dup == 0 and names_ok and bad == 0 and n_fb == 0
                             and len(m) == exp_m and m_dup == 0 and same and m_names
                             and n_pass == n_cell and len(nc) == n_cell and r_ok)
            rows.append(row)
    t = pd.DataFrame(rows)
    return t, bool(len(t)) and bool(t["OK"].all())


def readme_table(cfg, datasets, complete, notes):
    g = cfg.grid
    items = [
        ("結果狀態", "✔ 正式：全部完成、無重複、逐鍵清點通過、對照遮罩與舊 run 逐格一致、無 sampler 防呆 fallback、"
                     "統整設定與開跑時相同" if complete
         else "⚠ 暫存（PARTIAL）：尚未全部完成或核對未通過（原因見「備註」），只能看進度，不可用於論文"),
        ("這份檔案是什麼", f"Study 3 路線二（S / T / U）：兩階段 ensemble under-sampling、第二階段 ENN 的 OCC 調參結果"
                           f"（run = {cfg.run_name}；對照讀自 {cfg.us_run_name}）"),
        ("方法代號", "S＝OF、T＝DF、U＝OF+DF（串接）。"),
        ("資料集", f"{len(datasets)} 個：{', '.join(datasets)}"),
        ("流程", "特徵 → MinMax（只 fit 未清理的訓練多數類）→ 第一階段：ENN / CNN / TL 平行、對同一筆樣本投票 → "
               "第二階段：第一階段留下的多數類＋全部少數類再跑一次 ENN（只刪多數類）→ OCC 在最後留下的多數類上訓練"
               "（不再二次 MinMax、不重訓 AE）→ 測試。"),
        ("兩階段策略", "2S_Majority_ENN（主要）：S2_Majority → ENN｜2S_ENN_ENN（必要對照）：ENN → ENN｜"
                       "2S_Union_ENN（等價核對）：S1_Union → ENN｜2S_Intersection_ENN（次要）：S3_Intersection → ENN。"
                       f"本次執行：{active_two_stage(cfg)}"),
        ("第二階段 ENN", f"EditedNearestNeighbours(n_neighbors={cfg.enn_k}, kind_sel=all)，與第一階段的 ENN 相同；"
                         "sampling_strategy 明確指定只清多數類（第一階段刪到多數類比少數類少時，'auto' 會清錯對象）。"
                         "只跑一次，不是反覆清到停止的 RENN。最後筆數由 ENN 決定，不再固定為 N（舊圖的 Random / "
                         "Center / Uncentered 與「大小固定為 N」不適用於本方法）。"),
        ("對照組從哪裡來", f"none / ENN / S1_Union / S2_Majority / S3_Intersection 的結果列直接讀舊 run（{cfg.us_run_name}），"
                           "沒有重算。每一格開跑時重算第一階段遮罩並與舊 run 的 MaskSig 逐一比對，不同就整格不存檔；"
                           "所以能存檔的格子，對照與新策略用的是同一份資料、同一顆 AE、同一組 sampler 結果。"),
        ("數值抽核", "每一格、每個空間都用舊預設參數（OCSVM nu=0.1/gamma=scale、LOF k=20、iForest 100 棵/256）在單次 ENN 留下的樣本上"
                     f"重新訓練三個 OCC，五項指標與舊 run 同一列比較，差 < {NUM_TOL:g} 才存檔。遮罩相同只代表留下的樣本相同；"
                     "這一步確認現在的程式與環境算出的 OCC 數值也和舊 run 相同，新算的列才能和舊列放在同一張表。"),
        ("AE 特徵", "嚴格讀取 A~C 留下的 _df_cache：檔案缺欄位、資料指紋不符、形狀不對或含非有限值就讓該格失敗並記錄，"
                    "不會自動重訓 AE，也不會改寫共享快取（除非設定 allow_ae_retrain=True）。"),
        ("沿用結果（ReuseFrom）", "新策略最後留下的樣本若與某個對照（或另一個新策略）完全相同，OCC 不重新訓練，"
                              "直接沿用那個策略同一格的 59 組結果。03 分頁列出各策略沿用的比例。"),
        ("怎麼判讀（由寬到嚴）", "① 贏 S2_Majority：第二階段 ENN 有補救效果；② 贏 單次 ENN：先集成再 ENN 比直接 ENN "
                              "有額外價值；③ 也贏 2S_ENN_ENN：才能歸因於第一階段的 ensemble，而不是多清一次。"
                              "只做到 ①，代表繞了一圈回到 I/J/K 的水準。"),
        ("02 分頁：各自調參", "每個策略各自在 21 個 config × 59 組 θ 中挑 5 折平均 AUC 最高者（OF 空間只挑 θ），"
                            "再逐資料集配對。這是「各方法能達到的最好表現」，差異同時包含清理與重新挑參數的影響。"),
        ("02b 分頁：同 config 同 θ", "把 AE config 與 OCC 參數固定為「對照自己調出來的那一組」，新策略不另外挑。"
                                  "差異只來自保留的樣本不同；因為設定是替對照挑的，這個比較對新策略偏保守。"
                                  "新策略在該設定下不足 5 折有效的資料集不列入（看「共同有效資料集數」）。"),
        ("檢定", f"逐資料集配對、統計單位是資料集。Wilcoxon signed-rank（雙尾）：|Δ| ≤ {TIE_TOL:g} 視為平手並剔除，"
                 f"剔除後不足 {us.MIN_PAIRS} 個資料集不檢定；沒有同值用精確分布，有同值用常態近似＋連續性校正"
                 "（與 occ_tune_us.py 同一個函式）。Holm_p：同一個對照、同一空間×AE×OCC 內的多重比較校正。"),
        ("03 / 03b 分頁", "03：每個策略的保留筆數、刪除率、與單次 ENN 的重疊程度（Jaccard）、包含關係。"
                          "預期的包含關係是 2S_ENN_ENN ⊆ 2S_Majority_ENN ⊆ 單次 ENN（座標固定、只刪多數類、無距離平手時成立）；"
                          "比例不是 100% 時請檢查重複點與距離平手。03b：兩兩策略保留集合完全相同的比例，"
                          "以格子（資料集×fold×AE×config×空間）為單位，不因 59 組參數重複計數。"),
        ("無效格子", f"清理後多數類 < {MIN_KEEP} 筆 → degenerate（資料造成，屬正常結果）：AUC 留空，該候選不足 5 折就不會被選出。"
                   "sampler 執行出錯 → 該格不存檔、原因寫進 errors.log、重跑會再試；防呆條件（少數類被改動、索引重複）"
                   "→ invalid_fallback，只要出現就整份標 PARTIAL。"),
        ("調參方式與定位", "sampler 固定、只調 OCC。⚠ 挑選與報告用的是同一批測試折 → 屬「最佳可達表現」（oracle），"
                        "有樂觀偏誤，與 A~O 同一性質；不能寫成「在未見資料上一定更好」。Holm 處理多重比較，不處理選參偏誤。"),
        ("方法定位", "以已標記的少數類輔助挑選訓練樣本（label-informed data cleaning），再以 one-class 方式訓練 OCC："
                   "少數類只用來決定刪哪些多數類，不進 OCC 訓練；測試折不參與 AE、MinMax、清理。"),
        ("尚未包含", "「同保留筆數隨機刪除」對照不在本檔；要主張改善來自「選對樣本」而不是「樣本數改變」時需另做。"),
        ("OCC 參數格", f"OCSVM nu ∈ {g['OCSVM']['nu']}、gamma ∈ {g['OCSVM']['gamma']}｜LOF n_neighbors ∈ "
                     f"{g['LOF']['n_neighbors']}｜iForest n_estimators ∈ {g['iForest']['n_estimators']}、"
                     f"max_samples ∈ {g['iForest']['max_samples']}（與 A~O 相同）"),
        ("AE", f"{cfg.ae_types}｜{len(cfg.configs)} configs｜epochs={cfg.ae_epochs}（讀 A~C 的 _df_cache）"),
        ("備註", "；".join(notes) if notes else "無"),
        ("版本", json.dumps({**core.soft_info(cfg), "imblearn": imblearn.__version__}, ensure_ascii=False)),
    ]
    return pd.DataFrame(items, columns=["項目", "內容"])


def summarize(cfg: Us2Config):
    datasets, profile = core.discover_datasets(cfg)
    guard_problems, guard_info = summary_guard(cfg, us2_signature, US2_SUMMARY_KEYS, datasets)
    # 統整時也重新確認舊對照：設定、資料指紋、每一格結果都還在（--summarize-only 也會做）
    all_tasks = [t for _, ts in build_tasks(cfg, datasets, profile) for t in ts]
    ref_p, ref_i = ref_problems(cfg, {s: REF_STRATEGIES for s in active_spaces(cfg)}, datasets, profile, all_tasks)
    notes = guard_problems + guard_info + ref_p + ref_i + scope_notes(cfg, datasets)
    new, n_unknown = load_new_rows(cfg, datasets)
    if new.empty:
        print("沒有結果可統整。")
        return None
    mk = load_mask_rows(cfg, datasets)
    log = build_log(cfg, datasets)
    refrows = load_ref_rows(cfg, datasets) if not ref_p else pd.DataFrame(columns=LOAD_COLS)
    df = pd.concat([refrows, new], ignore_index=True) if len(refrows) else new
    ok = df[df["Status"] == "ok"]
    best, _ = _pick(ok, G_US, ["Config", "ParamID"])
    dflt, _ = _pick(ok[ok["IsDefault"]], G_US, ["Config", "ParamID"])
    new_best = best[best["Strategy"].isin(active_two_stage(cfg))] if len(best) else best
    if new_best.empty or dflt.empty:
        out = summary_path(cfg, datasets, False)
        core.write_excel(out, [("00_說明", readme_table(cfg, datasets, False,
                                                         notes + ["目前還沒有任何候選跑滿 5 折，只輸出進度"]), None),
                               ("06_執行紀錄", log, None)])
        print(f"⚠️  目前還沒有任何設定跑滿 5 折，只輸出進度：{out.resolve()}")
        return out

    keys = ("Study", "Space", "Strategy", "AE", "OCC")
    k = list(keys)
    ov_b, ov_d = ens.overall(best, "調參後", keys), ens.overall(dflt, "原固定參數", keys)
    rr = best.groupby(k, observed=True).agg(刪除率_選中組合=("RemovedRate", "mean"),
                                            保留筆數_選中組合=("MajKept", "mean")).reset_index()
    tab = ov_b[k + ["資料集數", "AUC", "AUC_mean", "F1", "Recall", "G-mean", "AP", "最常選中參數",
                    "最常選中 Config"]].rename(columns={"AUC": "AUC_調參後", "AUC_mean": "m_b"})
    od = ov_d[k + ["AUC", "AUC_mean"]].rename(columns={"AUC": "AUC_原固定參數", "AUC_mean": "m_d"})
    for t in (tab, od, rr):
        for c in k:
            t[c] = t[c].astype(str)
    tab = tab.merge(od, on=k, how="left")
    tab["ΔAUC（調參後−原固定）"] = tab["m_b"] - tab["m_d"]
    tab = tab.merge(rr, on=k, how="left").drop(columns=["m_b", "m_d"])
    tab.insert(3, "角色", tab["Strategy"].map(lambda s: ROLE_LABEL[TWO_STAGE[s][1]] if s in TWO_STAGE
                                              else ROLE_LABEL["ref"]))
    tab["Space"] = tab["Space"].map(SPACE_LABEL)
    tab = _sorted(tab)

    errs = []
    sheets = {
        "02": safe_table("02 配對比較（各自調參）", lambda: _sorted(comparisons_tuned(best, cfg)), errs),
        "02b": safe_table("02b 配對比較（同 config 同 θ）", lambda: _sorted(comparisons_same_setting(ok, cfg)), errs),
        "03": safe_table("03 遮罩診斷", lambda: mask_summary(mk), errs),
        "03b": safe_table("03b 兩兩策略等價率", lambda: equivalence_table(mk), errs),
        "04": safe_table("04 每資料集最佳", lambda: ens.per_dataset_table(best, "調參後", G=G_US), errs),
        "04b": safe_table("04b 新策略最佳組合逐折明細",
                          lambda: new_best[G_US + ["Config", "ParamID", "Fold", "Stage1Kept", "MajKept", "RemovedRate",
                                                   "ReuseFrom"] + METRIC_COLS + EXTRA_COLS]
                          .astype({c: str for c in G_US + ["Config", "ParamID", "ReuseFrom"]})
                          .sort_values(G_US + ["Fold"]), errs),
    }
    notes += errs
    if not bool(log["完整"].all()):
        notes.append("還有格子沒跑完（見 06 分頁）")
    audit_tab, audit_ok = key_audit(new, mk, refrows, cfg, datasets)
    if not audit_ok:
        notes.append("逐鍵清點未通過（見 07 分頁）")
    dup = int(new.duplicated(KEY8).sum())
    if dup:
        notes.append(f"有 {dup} 列重複")
    n_fb = int((mk["Status"] == "invalid_fallback").sum()) if len(mk) else 0
    if n_fb:
        notes.append(f"有 {n_fb} 個（格子×策略）出現 sampler 防呆 fallback（見 03 分頁）")
    if n_unknown:
        notes.append(f"有 {n_unknown} 列的名稱不在本次設定內而被略過（設定可能被改過）")
    # 對照核對：能存檔的格子都已通過遮罩比對；這裡再確認每一格的對照遮罩列都在
    n_ref_cells = int(mk[mk["Strategy"].isin(REF_STRATEGIES)].shape[0]) if len(mk) else 0
    chk = pd.DataFrame([
        {"項目": "對照來源（舊 I~O run）", "結果": cfg.us_run_name},
        {"項目": "設定 / 資料 / 套件與舊 run 相同", "結果": "✔ 相同" if not ref_p else "✘ " + "；".join(ref_p)},
        {"項目": "逐格遮罩核對（none / ENN / S1 / S2 / S3 的 MaskSig 與舊 run 相同才存檔）",
         "結果": f"✔ {n_ref_cells} 個（格子×對照策略）全部相同（不同的格子不會存檔，見 errors.log）"},
        {"項目": "數值抽核（每格用舊預設參數重算單次 ENN 的三個 OCC，五項指標與舊 run 比較）",
         "結果": (lambda nc: f"{int((nc < NUM_TOL).sum()):,}/{len(nc):,} 個（格子×空間）通過，最大差 "
                             f"{(float(nc.max()) if len(nc) else float('nan')):.3g}")(
             pd.to_numeric(mk["NumCheckDiff"], errors="coerce").dropna() if len(mk) else pd.Series(dtype=float))},
        {"項目": "對照結果列（讀自舊 run，未重算；完整性見 07 分頁）", "結果": f"{len(refrows):,} 列"},
        {"項目": "新策略結果列", "結果": f"{len(new):,} 列（其中沿用相同保留集合 "
                                         f"{int(((new['Status'] == 'ok') & (new['ReuseFrom'].astype(str) != '-')).sum()):,} 列）"},
    ])
    complete = (bool(log["完整"].all()) and dup == 0 and not errs and audit_ok and n_fb == 0 and n_unknown == 0
                and not guard_problems and not ref_p)
    out = summary_path(cfg, datasets, complete)
    delta = ["平均ΔAUC", "中位數ΔAUC"]
    core.write_excel(out, [
        ("00_說明", readme_table(cfg, datasets, complete, notes), None),
        ("01_總表", tab, ["ΔAUC（調參後−原固定）"]),
        ("02_配對比較_各自調參", sheets["02"], delta),
        ("02b_配對比較_同config同θ", sheets["02b"], delta),
        ("03_遮罩診斷", sheets["03"], None),
        ("03b_兩兩策略等價率", sheets["03b"], None),
        ("04_每資料集最佳", sheets["04"], None),
        ("04b_新策略逐折明細", sheets["04b"], None),
        ("05_與舊run核對", chk, None),
        ("06_執行紀錄", log, None),
        ("07_逐鍵清點", audit_tab, None),
    ])
    print(f"✅ 統整完成：{out.resolve()}")
    if complete:
        print(f"✔ 驗收通過：{len(datasets)} 個資料集 × {cfg.studies} 全部完成，無重複，對照遮罩與 {cfg.us_run_name} 逐格一致。")
        for msg in scope_notes(cfg, datasets):
            print("ℹ️  " + msg)
    else:
        print("⚠️  尚未完成或核對未通過 → 檔名標 _PARTIAL，只能當進度參考。" + "；".join(notes))
    print("\n── 各策略（已調參）AUC ──")
    print(tab[["Study", "Space", "Strategy", "AE", "OCC", "AUC_調參後", "AUC_原固定參數", "刪除率_選中組合"]]
          .to_string(index=False))
    return out


def main(cfg: Us2Config, argv=None):
    ap = argparse.ArgumentParser(description="S/T/U 兩階段 under-sampling（第二階段 ENN）OCC 調參")
    ap.add_argument("--summarize-only", action="store_true", help="不跑實驗，只把已完成的結果統整成 Excel")
    args = ap.parse_args(argv)
    if not args.summarize_only:
        if run(cfg) == "interrupted":
            return
    summarize(cfg)
