"""
occ_tune_us.py
==============
Study 2（I / J / K）與 Study 3 路線一（M / N / O）的 OCC 超參數調整：共用核心程式
（沿用 occ_tune_core.py 的讀資料、AE、OCC、DF 快取、鎖；沿用 occ_tune_ensemble.py 的階段執行器與選法；
  筆電測試用 run_laptop_US.py，Azure 全跑用 run_azure_US.py）

────────────────────────────────────────────────────────────────────────────
方法對照（同一支程式、同一顆 AE、同一份特徵 → 全部可以逐資料集配對比較）
────────────────────────────────────────────────────────────────────────────
  特徵空間         不清理(none)   單一 under-sampling        ensemble under-sampling
  OF               ＝ A           I：ENN / CNN / TL          M：S1 / S2 / S3 / S4
  DF               ＝ B           J：ENN / CNN / TL          N：S1 / S2 / S3 / S4
  OF+DF（串接）     ＝ C           K：ENN / CNN / TL          O：S1 / S2 / S3 / S4

  ensemble 策略（與 N 相同：三個 sampler 從同一份未清理的多數類平行出發，各自決定保留哪些樣本，
  再對「同一筆樣本」投票；不依序清理、不串接、不會重複樣本）
    S1_Union         ≥1 個 sampler 保留就保留（聯集）
    S2_Majority      ≥2 個 sampler 保留才保留（多數決）
    S3_Intersection  3 個 sampler 都保留才保留（交集）
    S4_ENN_TL_abl    ENN 與 TL 都保留才保留（去掉 CNN 的消融，事後提出，論文需標明）

  流程（每個特徵空間都一樣，OCC 永遠在最後）：
    特徵 → MinMax（只 fit 未清理的訓練多數類）→ sampler 在這個座標下清理多數類
         → OCC（在清理後的多數類上訓練，不再二次 MinMax）→ 測試
    ※ 整條流程屬於「以已標記的少數類輔助挑選訓練樣本，再訓練 OCC」：
      少數類只用來決定刪哪些多數類，不進 OCC 訓練；測試集全程不參與清理與 AE。

────────────────────────────────────────────────────────────────────────────
調參方式（與 A~C 相同，屬「最佳可達表現」＝以測試折挑選的 oracle，論文需註明）
────────────────────────────────────────────────────────────────────────────
  sampler 參數固定（ENN k=3、CNN k=1/seed 42、TL 預設，與 I/J/K/N 相同），只調 OCC 參數（59 組）。
  每個 (方法, 策略, 資料集, AE, OCC) 在 21 個 AE config × 59 組中挑 5 折平均 AUC 最高者
  （OF 空間沒有 AE，只挑 θ）。同一次執行也保留原固定參數（θ＝舊預設）的結果。
  另外附兩個「不挑 config」的分析（統整時直接由同一批結果算，不需多跑）：
    08 固定表徵比較：事前指定 AE×config（預設 VAE × h1-1/1，與 N 相同）
    09 跨 config 穩定性：每個 config 各自比較，數策略贏過對照的 config 有幾個

────────────────────────────────────────────────────────────────────────────
工作格（寫檔成功 ＝ 完成；中斷後重跑只補沒完成的格子）
────────────────────────────────────────────────────────────────────────────
  OF 格（資料集×fold）            ：OF 空間（A / I / M）
  DF 格（資料集×fold×AE×config）  ：DF 與 OF+DF 空間（B / J / N 與 C / K / O），共用同一顆 AE
  AE 特徵（含 DF_min）直接讀 A~C 調參留下的 _df_cache（out_root 相同就共用，不重訓 AE）
  同一格內若兩個策略保留的樣本完全相同，OCC 只算一次並沿用（ReuseFrom 欄）；
  每一列另存保留集合的指紋（MaskSig 欄），03 分頁據此計算「兩兩策略」的等價率。
"""
from __future__ import annotations

import os
import occ_tune_core as core          # 必須最先 import（先設定執行緒環境變數）

import re
import json
import time
import socket
import argparse
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from hashlib import sha1

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import wilcoxon
from imblearn.under_sampling import EditedNearestNeighbours, CondensedNearestNeighbour, TomekLinks
import imblearn

from occ_tune_core import (N_FOLDS, THRESHOLD_PCT, OCC_TYPES, METRIC_COLS, EXTRA_COLS,
                           LEGACY_DEFAULT, BOTTLENECK_RATIOS, param_combos, param_id,
                           is_legacy_default, safe_name, dict_hash)
import occ_tune_ensemble as ens
from occ_tune_ensemble import (_prep, _metrics, occ_scores_all, _pick, run_phase, prepare_run,
                               code_fingerprint, lib_versions, safe_table, summary_guard)

US_VERSION = "US_tune_v3"
#   v3：數值算法與 v2 完全相同；sampler 執行出錯改為「該格失敗、記錄原因、重跑再試」（不再默默判無效），
#       Wilcoxon 的平手處理與檢定方法固定，統整時核對設定與開跑時相同，並加上 AE 快取開跑前檢查。
SPACES = ["OF", "DF", "OFDF"]
SPACE_LABEL = {"OF": "OF", "DF": "DF", "OFDF": "OF+DF"}
# 空間 → (none 對應的 baseline, 單一 sampler 的方法代號, ensemble 的方法代號)
SPACE_STUDY = {"OF": ("A", "I", "M"), "DF": ("B", "J", "N"), "OFDF": ("C", "K", "O")}
STUDY_SPACE = {s: sp for sp, t in SPACE_STUDY.items() for s in t[1:]}
VOTERS = ["ENN", "CNN", "TL"]
SINGLES = ["ENN", "CNN", "TL"]
# name -> (family, voters, keep threshold, role)；與 N 的 build_strategy_specs 相同
STRATEGY_SPECS = {
    "none":            ("baseline", (), 0, "control"),
    "ENN":             ("single", ("ENN",), 1, "control"),
    "CNN":             ("single", ("CNN",), 1, "control"),
    "TL":              ("single", ("TL",), 1, "control"),
    "S1_Union":        ("vote", tuple(VOTERS), 1, "primary"),
    "S2_Majority":     ("vote", tuple(VOTERS), 2, "primary"),
    "S3_Intersection": ("vote", tuple(VOTERS), 3, "primary"),
    "S4_ENN_TL_abl":   ("vote", ("ENN", "TL"), 2, "ablation"),
}
STRATEGIES = list(STRATEGY_SPECS)
STATUSES = ["ok", "degenerate", "invalid_fallback"]
US_COLS = (["Study", "Space", "Strategy", "Dataset", "Fold", "AE", "Config", "OCC", "ParamID", "IsDefault",
            "MajKept", "RemovedRate", "MaskSig", "ReuseFrom", "Status"] + METRIC_COLS + EXTRA_COLS)
MIN_KEEP = 5                                   # 清理後多數類 < 5 筆 → 該策略該格無效（與 I/J/K/N 相同）


@dataclass
class UsConfig(core.RunConfig):
    run_name: str = "US_tune"
    studies: list = field(default_factory=lambda: ["I", "J", "K", "M", "N", "O"])
    enn_k: int = 3
    cnn_k: int = 1
    cnn_seed: int = 42
    abc_run_name: str = ""
    # 開跑前確認 A~C 的 AE 特徵快取齊全（DF / OF+DF 空間才需要）；缺檔就停止。確定要重訓缺少的 AE 才改 True
    allow_ae_retrain: bool = False
    # 08 分頁「固定表徵比較」事前指定的 AE × config（與 N 的 PRIMARY_AE / PRIMARY_CONFIG 相同）；只影響統整
    primary_ae: str = "VAE"
    primary_config: str = "h1-1/1"


# ═════════════════════════ 要跑哪些空間 / 策略 ═══════════════════════════
def active_spaces(cfg):
    return [sp for sp in SPACES if any(s in cfg.studies for s in SPACE_STUDY[sp][1:])]


def active_strategies(cfg, space):
    _, single, vote = SPACE_STUDY[space]
    out = ["none"]
    if single in cfg.studies or vote in cfg.studies:     # 有 ensemble 時也保留 ENN（主要對照）
        out += SINGLES if single in cfg.studies else ["ENN"]
    if vote in cfg.studies:
        out += [s for s in STRATEGIES if STRATEGY_SPECS[s][0] == "vote"]
    return out


def study_of(space, strategy):
    fam = STRATEGY_SPECS[strategy][0]
    base, single, vote = SPACE_STUDY[space]
    return base if fam == "baseline" else single if fam == "single" else vote


# ═════════════════════════ under-sampling（與 N 相同）═════════════════════
def make_sampler(name, cfg):
    if name == "ENN":
        return EditedNearestNeighbours(n_neighbors=cfg.enn_k, kind_sel="all", sampling_strategy="auto")
    if name == "CNN":
        return CondensedNearestNeighbour(n_neighbors=cfg.cnn_k, random_state=cfg.cnn_seed,
                                         sampling_strategy="auto")
    if name == "TL":
        return TomekLinks(sampling_strategy="auto")
    raise ValueError(name)


def compute_keep_masks(maj_s, min_s, cfg):
    """三個 sampler 在同一份未清理的多數類上各自決定保留哪些樣本（boolean mask）。
    sampling_strategy='auto' 只刪多數類。
    ・sampler 本身執行出錯 → 直接丟出例外：這一格不存檔、原因寫進 errors.log，重跑同一指令會再試
      （舊版會默默把該 sampler 相關的策略判為無效，而且這一格算「已完成」，永遠不會重試）。
    ・防呆條件（少數類被改動、保留索引重複）→ fallback：用到該 sampler 的策略判無效（invalid_fallback），
      統整時只要出現就整份標 PARTIAL。正常情況下兩者都不會發生。"""
    n_maj = len(maj_s)
    X = np.vstack([maj_s, min_s])
    y = np.array([0] * n_maj + [1] * len(min_s))
    masks, ok = {}, {}
    for m in VOTERS:
        s = make_sampler(m, cfg)
        try:
            s.fit_resample(X, y)
        except Exception as e:
            raise RuntimeError(f"sampler {m} 執行失敗：{type(e).__name__}: {e}") from e
        idx = np.asarray(s.sample_indices_)
        keep_local = idx[idx < n_maj]
        if int(np.sum(idx >= n_maj)) != len(min_s) or len(keep_local) != len(np.unique(keep_local)):
            masks[m], ok[m] = np.ones(n_maj, dtype=bool), False
            continue
        keep = np.zeros(n_maj, dtype=bool)
        keep[keep_local] = True
        masks[m], ok[m] = keep, True
    return masks, ok


def apply_strategy(masks, ok, strategy, n_maj):
    """回傳 (keep mask 或 None, status)。任何需要的 sampler fallback → 無效（不讓失敗者投「全留」）。"""
    fam, voters, thr, _ = STRATEGY_SPECS[strategy]
    if fam == "baseline":
        return np.ones(n_maj, dtype=bool), "ok"
    if not all(ok[v] for v in voters):
        return None, "invalid_fallback"
    votes = np.sum([masks[v].astype(int) for v in voters], axis=0)
    return votes >= thr, "ok"


def mask_sig(keep):
    return sha1(np.packbits(keep).tobytes() + str(len(keep)).encode()).hexdigest()[:16]


# ═════════════════════════ 一個空間的完整評估 ═════════════════════════════
def eval_space(space, maj_s, min_s, tst_s, y, cfg, base):
    """一個特徵空間：算 keep masks → 每個策略（相同 mask 只算一次 OCC）→ 全部 59 組 θ 的結果列。"""
    strategies = active_strategies(cfg, space)
    n_maj = len(maj_s)
    need_masks = any(STRATEGY_SPECS[s][0] != "baseline" for s in strategies)
    masks, ok = compute_keep_masks(maj_s, min_s, cfg) if need_masks else ({}, {})
    combos = [(occ, p) for occ in OCC_TYPES for p in param_combos(occ, cfg.grid)]
    seen = {}                                    # mask 指紋 → (策略名, {(occ,pid): metrics})
    rows = []
    for st in strategies:
        keep, status = apply_strategy(masks, ok, st, n_maj)
        reuse, mets = "-", None
        n_kept = int(keep.sum()) if keep is not None else 0
        sig = mask_sig(keep) if keep is not None else "-"
        if keep is not None and n_kept < MIN_KEEP:
            status = "degenerate"
        if status == "ok":
            if sig in seen:
                reuse, mets = seen[sig]
            else:
                sc = occ_scores_all(maj_s[keep], tst_s, cfg)
                mets = {}
                for occ, p in combos:
                    pid = param_id(p)
                    m = _prep(*sc[(occ, pid)])
                    mets[(occ, pid)] = _metrics(y, m["pred"], m["s_te"])
                seen[sig] = (st, mets)
        rr = (n_maj - n_kept) / n_maj if keep is not None else np.nan
        for occ, p in combos:
            pid = param_id(p)
            m = mets[(occ, pid)] if mets is not None else {k: np.nan for k in METRIC_COLS + EXTRA_COLS}
            rows.append({"Study": study_of(space, st), "Space": space, "Strategy": st, **base,
                         "OCC": occ, "ParamID": pid, "IsDefault": is_legacy_default(occ, p),
                         "MajKept": n_kept if keep is not None else np.nan, "RemovedRate": rr,
                         "MaskSig": sig, "ReuseFrom": reuse, "Status": status, **m})
    return rows


def n_rows_space(cfg, space):
    return len(active_strategies(cfg, space)) * ens.n_params(cfg)


# ═════════════════════════ DF（含 DF_min）════════════════════════════════
def df_cache_path(cfg, ds, fold, ae, c):
    """與 core.get_df 的快取路徑相同。"""
    return (Path(cfg.out_root) / "_df_cache" / dict_hash(core.ae_signature(cfg)) / safe_name(ds)
            / f"f{fold}__{ae}__{safe_name(c)}.npz")


def get_df3(cfg, ds, fold, ae, c, fd):
    """回傳 (DF_maj, DF_tst, DF_min, 來源)。DF_maj/DF_tst 直接用 core.get_df（與 B/C 完全相同），
    DF_min（sampler 的參考點）從同一個快取檔讀；沒有快取就用同一顆 AE（同種子）重算。"""
    DF_maj, DF_tst, src = core.get_df(cfg, ds, fold, ae, c, fd)
    path = df_cache_path(cfg, ds, fold, ae, c)
    if cfg.use_df_cache and path.exists():
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        if meta.get("train_sha1") == fd["train_sha1"] and len(z["DF_min"]) == fd["n_min"]:
            return DF_maj, DF_tst, z["DF_min"].astype(np.float64), src
    n_layers = int(re.match(r"^h(\d+)-", c).group(1))
    n_units = max(2, round(fd["X_maj_s"].shape[1] * BOTTLENECK_RATIOS[c.split("-", 1)[1]]))
    a, b, m = core.train_ae_extract(ae, fd["X_maj_s"], fd["X_tst_s"], fd["X_min_s"], n_layers, n_units,
                                    cfg.ae_epochs, core.cell_seed(ds, fold, ae, c))
    return DF_maj, DF_tst, m.astype(np.float32).astype(np.float64), src


# ═════════════════════════ 工作格 ══════════════════════════════════════════
def cell_path(cfg, task):
    root = Path(cfg.out_root) / cfg.run_name / "cells"
    kind, ds, fold = task[0], task[1], task[2]
    if kind == "OF":
        return root / "OF" / safe_name(ds) / f"fold{fold}.csv.gz"
    return root / "DF" / safe_name(ds) / f"fold{fold}__{task[3]}__{safe_name(task[4])}.csv.gz"


def _write(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    pd.DataFrame(rows, columns=US_COLS).to_csv(tmp, index=False, encoding="utf-8", compression="gzip")
    os.replace(tmp, path)


def _check(rows, expected, what):
    if len(rows) != expected:
        raise RuntimeError(f"{what} 列數不符：{len(rows)} ≠ {expected}")
    bad = [r for r in rows if r["Status"] == "ok" and not np.isfinite(r["AUC"])]
    if bad:
        raise RuntimeError(f"{what} 有 {len(bad)} 列 AUC 無效，整格不存檔")


def run_us_task(task, cfg):
    torch.set_num_threads(max(1, cfg.torch_threads_per_job))
    t0 = time.perf_counter()
    kind, ds, fold = task[0], task[1], task[2]
    fd = core.load_fold(cfg.data_root, ds, fold)
    y = fd["y_tst"]
    rows, expected, src = [], 0, "-"
    if kind == "OF":
        base = {"Dataset": ds, "Fold": fold, "AE": "-", "Config": "-"}
        rows = eval_space("OF", fd["X_maj_s"], fd["X_min_s"], fd["X_tst_s"], y, cfg, base)
        expected = n_rows_space(cfg, "OF")
    else:
        ae, c = task[3], task[4]
        DF_maj, DF_tst, DF_min, src = get_df3(cfg, ds, fold, ae, c, fd)
        base = {"Dataset": ds, "Fold": fold, "AE": ae, "Config": c}
        spaces = [sp for sp in active_spaces(cfg) if sp != "OF"]
        if "DF" in spaces:                               # 與 B / J / N 相同：DF 以未清理的 DF_maj fit MinMax
            sc = MinMaxScaler().fit(DF_maj)
            rows += eval_space("DF", sc.transform(DF_maj), sc.transform(DF_min), sc.transform(DF_tst),
                               y, cfg, base)
            expected += n_rows_space(cfg, "DF")
        if "OFDF" in spaces:                             # 與 C / K 相同：[OF, DF] 串接後以未清理的 Z_maj fit MinMax
            Zm, Zn, Zt = (np.hstack([fd["X_maj_s"], DF_maj]), np.hstack([fd["X_min_s"], DF_min]),
                          np.hstack([fd["X_tst_s"], DF_tst]))
            sc = MinMaxScaler().fit(Zm)
            rows += eval_space("OFDF", sc.transform(Zm), sc.transform(Zn), sc.transform(Zt), y, cfg, base)
            expected += n_rows_space(cfg, "OFDF")
    _check(rows, expected, kind)
    _write(rows, cell_path(cfg, task))
    return {"task": "|".join(map(str, task)), "kind": kind, "dataset": ds, "n_maj": int(fd["n_maj"]),
            "seconds": round(time.perf_counter() - t0, 3), "df_source": src, "host": socket.gethostname()}


def us_signature(cfg):
    return {"us_version": US_VERSION, "studies": sorted(cfg.studies), "ae_types": cfg.ae_types,
            "configs": cfg.configs, "grid": cfg.grid, "legacy_default": LEGACY_DEFAULT,
            "ae": core.ae_signature(cfg), "lof_train_score": cfg.lof_train_score,
            "threshold_pct": THRESHOLD_PCT, "cat_encoding": "train_fold_mapping_v1",
            "metrics": METRIC_COLS + EXTRA_COLS, "n_folds": N_FOLDS,
            "sampler": {"voters": VOTERS, "enn_k": cfg.enn_k, "cnn_k": cfg.cnn_k, "cnn_seed": cfg.cnn_seed,
                        "strategy": "auto", "min_keep": MIN_KEEP, "strict_fallback": True},
            "strategies": {k: list(v[:3]) for k, v in STRATEGY_SPECS.items()},
            "libs": {**lib_versions(), "imblearn": imblearn.__version__},
            # 會產生數值或決定「算哪些策略」的函式全部納入：任何一個改了，舊格子就不能和新格子混在同一個 run
            "code": code_fingerprint(core._fit_score, core.get_df, occ_scores_all, _prep, _metrics, make_sampler,
                                     compute_keep_masks, apply_strategy, mask_sig, eval_space, get_df3,
                                     run_us_task, active_strategies, active_spaces, study_of)}


# 統整時要與開跑時（signature.json）相同的鍵
US_SUMMARY_KEYS = ["us_version", "studies", "ae_types", "configs", "grid", "legacy_default", "lof_train_score",
                   "threshold_pct", "metrics", "n_folds", "sampler", "strategies"]


def run(cfg: UsConfig):
    bad = set(cfg.studies) - set(STUDY_SPACE)
    if bad or not cfg.studies:
        raise SystemExit(f"studies 只能是 I / J / K / M / N / O，收到 {cfg.studies}")
    datasets, profile, run_dir = prepare_run(cfg, f"Under-sampling 調參 studies={cfg.studies}", us_signature,
                                             need_df=bool({"DF", "OFDF"} & set(active_spaces(cfg))))
    sp = active_spaces(cfg)
    print("空間與策略：" + "｜".join(f"{SPACE_LABEL[s]}: {active_strategies(cfg, s)}" for s in sp))
    print(f"每個策略 × 每個特徵空間：OCC 參數 {ens.n_params(cfg)} 組；AE 空間另外 {len(cfg.ae_types)} 種 AE × "
          f"{len(cfg.configs)} 個 config")
    lock = core.acquire_lock(run_dir)
    try:
        core.cleanup_tmp(cfg)
        n_maj = dict(zip(profile["Dataset"], profile["n_majority_fold1"]))
        big_first = lambda t: -n_maj.get(t[1], 0)
        phases = []
        if "OF" in sp:
            phases.append(("OF", sorted([("OF", ds, f) for ds in datasets for f in range(1, N_FOLDS + 1)],
                                        key=big_first)))
        if {"DF", "OFDF"} & set(sp):
            phases.append(("DF", sorted([("DF", ds, f, ae, c) for ds in datasets for f in range(1, N_FOLDS + 1)
                                         for ae in cfg.ae_types for c in cfg.configs], key=big_first)))
        for name, tasks in phases:
            if run_phase(name, tasks, cfg, run_dir, run_us_task, cell_path) == "interrupted":
                return "interrupted"
    finally:
        core.release_lock(lock)
    return "done"


# ═════════════════════════ 統整 ════════════════════════════════════════════
def _dtypes(cfg, datasets):
    pids = sorted({param_id(p) for o in OCC_TYPES for p in param_combos(o, cfg.grid)})
    return {"Study": pd.CategoricalDtype(list("ABCIJKMNO")), "Space": pd.CategoricalDtype(SPACES),
            "Strategy": pd.CategoricalDtype(STRATEGIES), "Dataset": pd.CategoricalDtype(list(datasets)),
            "AE": pd.CategoricalDtype(list(cfg.ae_types) + ["-"]),
            "Config": pd.CategoricalDtype(list(cfg.configs) + ["-"]),
            "OCC": pd.CategoricalDtype(OCC_TYPES), "ParamID": pd.CategoricalDtype(pids),
            "ReuseFrom": pd.CategoricalDtype(STRATEGIES + ["-"]), "Status": pd.CategoricalDtype(STATUSES),
            "MaskSig": str, "Fold": "int16"}


def load_rows(cfg, datasets):
    """回傳 (結果列, 因名稱不在本次設定內而略過的列數)。略過列數 > 0 時統整會標 PARTIAL。"""
    root = Path(cfg.out_root) / cfg.run_name / "cells"
    files = [f for k in ("OF", "DF") for ds in datasets for f in sorted((root / k / safe_name(ds)).glob("*.csv.gz"))]
    if not files:
        return pd.DataFrame(columns=US_COLS), 0
    df = pd.concat((pd.read_csv(f, dtype=_dtypes(cfg, datasets)) for f in files), ignore_index=True)
    df["IsDefault"] = df["IsDefault"].astype(str).str.lower().eq("true")
    key = ["Study", "Space", "Strategy", "Dataset", "AE", "Config", "OCC", "ParamID", "Status"]
    n_unknown = int(df[key].isna().any(axis=1).sum())
    if n_unknown:
        print(f"⚠️  {n_unknown} 列的名稱不在本次設定內（可能是舊結果或設定被改過），已略過 → 結果標 PARTIAL")
        df = df.dropna(subset=key)
    return df, n_unknown


G_US = ["Study", "Space", "Strategy", "Dataset", "AE", "OCC"]


def holm(p):
    """Holm 逐步校正（同一族群內）。NaN 不參與。"""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    idx = np.where(~np.isnan(p))[0]
    if not len(idx):
        return out
    order = idx[np.argsort(p[idx])]
    m, running = len(order), 0.0
    for r, i in enumerate(order):
        running = max(running, min(1.0, (m - r) * p[i]))
        out[i] = running
    return out


TIE_TOL = 1e-9          # |ΔAUC| ≤ 1e-9 視為平手：勝/平/負 與 Wilcoxon 用同一個標準
MIN_PAIRS = 5           # 有差異（非平手）的資料集 < 5 → 不做檢定（p 留空）


def _wx(x, how):
    """呼叫 scipy 的 Wilcoxon；相容新舊 scipy 的參數名稱（method / mode、asymptotic / approx）。"""
    tries = ([{"method": "exact"}, {"mode": "exact"}] if how == "exact" else
             [{"method": "asymptotic", "correction": True}, {"method": "approx", "correction": True},
              {"mode": "approx", "correction": True}])
    for kw in tries:
        try:
            return float(wilcoxon(x, zero_method="wilcox", **kw).pvalue)
        except (TypeError, ValueError):
            continue
    return np.nan


def wilcoxon_p(d):
    """配對 Wilcoxon signed-rank（雙尾）。回傳 (p, 有差異的資料集數, 檢定方法)。
    ・|Δ| ≤ TIE_TOL 視為平手並剔除（與「勝/平/負」同一標準；1e-16 等級的浮點誤差不會被當成真的差異）。
    ・剔除後不足 MIN_PAIRS 個 → 不檢定（NaN）。
    ・規則與 R 的 wilcox.test 預設相同：沒有同值且 n < 50 → 精確分布；否則常態近似＋同值校正＋連續性校正。
      明確指定方法，不依賴 scipy 各版本 method='auto' 的規則
      （舊寫法 wilcoxon(d)：d 含 0 時，新版 scipy 會改用「未做連續性校正的常態近似」，p 偏小）。"""
    d = np.asarray(d, dtype=float)
    nz = d[np.abs(d) > TIE_TOL]
    n = int(len(nz))
    if n < MIN_PAIRS:
        return np.nan, n, f"有差異的資料集 < {MIN_PAIRS}，未檢定"
    if len(np.unique(np.abs(nz))) == n and n < 50:
        return _wx(nz, "exact"), n, "精確分布"
    return _wx(nz, "normal"), n, "常態近似（同值＋連續性校正）"


def _ds_level(best, extra=()):
    keys = ["Space", "Strategy", "Study", "AE", "OCC"] + list(extra) + ["Dataset"]
    ds = best.groupby(keys, observed=True)["AUC"].mean().reset_index()
    for c in keys:
        ds[c] = ds[c].astype(str)
    return ds


def paired_table(best, ref, label, strategies=None):
    """逐資料集配對：同一 (空間, AE, OCC, 資料集) 下，策略（已調參）− 對照（已調參）。
    統計單位是資料集（先把 5 折平均）；只用兩邊都有效的「共同資料集」。
    ref = 策略名稱，或 "BEST_SINGLE"（逐資料集取 ENN/CNN/TL 中最高者，屬事後挑選、最嚴格的對照）。
    Wilcoxon p 另附 Holm 校正（族群＝同一對照、同一空間×AE×OCC 下的所有策略）。"""
    ds = _ds_level(best)
    on = ["Space", "AE", "OCC", "Dataset"]
    if ref == "BEST_SINGLE":
        r = ds[ds["Strategy"].isin(SINGLES)]
        n_single = r.groupby(on)["Strategy"].nunique().rename("n")
        r = r.groupby(on)["AUC"].max().rename("AUC_ref").reset_index().merge(n_single.reset_index(), on=on)
        r = r[r["n"] == len(SINGLES)].drop(columns="n")          # 三個 sampler 都有效的資料集才算
        cand = ds[~ds["Strategy"].isin(SINGLES + ["none"])]
    else:
        r = ds[ds["Strategy"] == ref][on + ["AUC"]].rename(columns={"AUC": "AUC_ref"})
        cand = ds[ds["Strategy"] != ref]
    if strategies is not None:
        cand = cand[cand["Strategy"].isin(strategies)]
    if cand.empty or r.empty:
        return pd.DataFrame()
    m = cand.merge(r, on=on)
    if m.empty:
        return pd.DataFrame()
    m["Δ"] = m["AUC"] - m["AUC_ref"]
    rows = []
    for k, g in m.groupby(["Space", "Strategy", "Study", "AE", "OCC"], sort=False):
        d = g["Δ"].values
        p, n_nz, how = wilcoxon_p(d)
        rows.append({"比較": label, "Space": SPACE_LABEL[k[0]], "Strategy": k[1], "Study": k[2], "AE": k[3],
                     "OCC": k[4], "對照": "單一 sampler 逐資料集最佳" if ref == "BEST_SINGLE" else ref,
                     "共同有效資料集數": len(d), "AUC_策略": g["AUC"].mean(),
                     "AUC_對照": g["AUC_ref"].mean(), "平均ΔAUC": float(d.mean()), "中位數ΔAUC": float(np.median(d)),
                     "勝": int((d > TIE_TOL).sum()), "平": int((np.abs(d) <= TIE_TOL).sum()),
                     "負": int((d < -TIE_TOL).sum()),
                     "檢定用資料集數（去掉平手）": n_nz, "Wilcoxon_p": p, "檢定方法": how})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["Holm_p"] = np.nan
    for _, idx in out.groupby(["Space", "AE", "OCC"]).groups.items():
        out.loc[idx, "Holm_p"] = holm(out.loc[idx, "Wilcoxon_p"].values)
    order = {s: i for i, s in enumerate(STRATEGIES)}
    return out.sort_values(["Space", "OCC", "AE", "Strategy"],
                           key=lambda s: s.map(order) if s.name == "Strategy" else s).reset_index(drop=True)


def comparisons(best, label_prefix=""):
    parts = [paired_table(best, "none", f"{label_prefix}vs none（清理有沒有幫助）")]
    if (best["Strategy"] == "ENN").any():
        parts.append(paired_table(best[best["Strategy"] != "none"], "ENN",
                                  f"{label_prefix}vs ENN（事前指定的主要對照，與 N 相同）"))
    if best["Strategy"].isin(SINGLES).any() and (best["Strategy"].astype(str).str.startswith("S")).any():
        parts.append(paired_table(best, "BEST_SINGLE",
                                  f"{label_prefix}vs 單一 sampler 逐資料集最佳（事後挑選，最嚴格）"))
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _effective_grid(n_kept, grid):
    """清理後筆數 n 太少時，LOF k 會被截成 n−1、iForest max_samples 被截成 n → 實際不同的組數會變少。"""
    n = int(n_kept)
    k_eff = len({min(k, n - 1) for k in grid["LOF"]["n_neighbors"]})
    if_eff = len({(e, min(ms, n)) for e in grid["iForest"]["n_estimators"] for ms in grid["iForest"]["max_samples"]})
    return k_eff, if_eff


def removal_table(df, cfg):
    """每格每策略一筆（與 θ 無關）：有效率、刪除率、實際有效的 OCC 參數組數。"""
    cells = df.drop_duplicates(["Space", "Strategy", "Dataset", "Fold", "AE", "Config"]).copy()
    rows = []
    n_lof, n_if = len(cfg.grid["LOF"]["n_neighbors"]), len(param_combos("iForest", cfg.grid))
    k_max = max(cfg.grid["LOF"]["n_neighbors"])
    ms_max = max(cfg.grid["iForest"]["max_samples"])
    for (sp, st), g in cells.groupby(["Space", "Strategy"], observed=True, sort=True):
        valid = g[g["Status"] == "ok"]
        kept = [int(n) for n in valid["MajKept"]]
        eff = [_effective_grid(n, cfg.grid) for n in kept]
        r = {"Space": SPACE_LABEL[sp], "Strategy": st, "Study": str(g["Study"].iloc[0]), "格子數": len(g),
             "有效格子數": len(valid), "退化(<5筆)": int((g["Status"] == "degenerate").sum()),
             "防呆 fallback（應為 0）": int((g["Status"] == "invalid_fallback").sum()),
             "平均刪除率（全部有效格子）": float(valid["RemovedRate"].mean()) if len(valid) else np.nan,
             "平均保留筆數": float(valid["MajKept"].mean()) if len(valid) else np.nan,
             "LOF 實際有效 k 組數（平均，名義 %d）" % n_lof: float(np.mean([e[0] for e in eff])) if eff else np.nan,
             # 截斷：至少一個 k 被 sklearn 改成 n−1；合併：截斷後有兩個以上的 k 變成同一個值（實際組數 < 名義）
             "LOF 有 k 被截斷的格子比例（n−1 < %d）" % k_max:
                 float(np.mean([n - 1 < k_max for n in kept])) if kept else np.nan,
             "LOF 參數合併的格子比例（實際組數 < 名義）": float(np.mean([e[0] < n_lof for e in eff])) if eff else np.nan,
             "iForest 實際有效組數（平均，名義 %d）" % n_if: float(np.mean([e[1] for e in eff])) if eff else np.nan,
             "iForest 有 max_samples 被截斷的格子比例（n < %d）" % ms_max:
                 float(np.mean([n < ms_max for n in kept])) if kept else np.nan,
             "iForest 參數合併的格子比例（實際組數 < 名義）":
                 float(np.mean([e[1] < n_if for e in eff])) if eff else np.nan}
        rows.append(r)
    return pd.DataFrame(rows)


def equivalence_table(df):
    """兩兩策略的等價率：同一格（資料集×fold×AE×config×空間）內，兩個策略保留的樣本是否完全相同。
    分母＝兩個策略在該格都有效的格子數。"""
    cells = df[df["Status"] == "ok"].drop_duplicates(["Space", "Strategy", "Dataset", "Fold", "AE", "Config"])
    wide = cells.pivot_table(index=["Space", "Dataset", "Fold", "AE", "Config"], columns="Strategy",
                             values="MaskSig", aggfunc="first", observed=True)
    rows = []
    for sp in SPACES:
        if sp not in wide.index.get_level_values(0):
            continue
        w = wide.xs(sp, level="Space")
        strats = [s for s in STRATEGIES if s in w.columns]
        for a, b in combinations(strats, 2):
            both = w[[a, b]].dropna()
            if not len(both):
                continue
            rows.append({"Space": SPACE_LABEL[sp], "策略 1": a, "策略 2": b, "兩者皆有效的格子數": len(both),
                         "保留集合完全相同的比例": float((both[a] == both[b]).mean())})
    return pd.DataFrame(rows)


def fixed_rep_table(ok, cfg):
    """08：事前指定的表徵（OF 空間本來就沒有 AE；DF / OF+DF 用 primary_ae × primary_config），
    只比較清理策略。兩種 θ：各策略自己挑 θ（仍是 oracle）／全部固定舊預設（完全不挑，等同 N 的主分析）。"""
    parts, notes = [], []
    if cfg.primary_ae not in cfg.ae_types or cfg.primary_config not in cfg.configs:
        notes.append(f"primary_ae/primary_config（{cfg.primary_ae} × {cfg.primary_config}）不在本次 AE/config 中，"
                     "DF 與 OF+DF 空間略過 08 分頁")
    mask = (ok["Space"] == "OF") | ((ok["AE"] == cfg.primary_ae) & (ok["Config"] == cfg.primary_config))
    sub = ok[mask]
    for label, d in (("θ 各自挑選", sub), ("θ 固定舊預設", sub[sub["IsDefault"]])):
        best, _ = _pick(d, G_US, ["ParamID"])
        if best.empty:
            continue
        t = comparisons(best, f"[{label}] ")
        if t.empty:
            continue
        t.insert(0, "表徵", t["Space"].map(lambda s: "OF（無 AE）" if s == "OF"
                                          else f"{cfg.primary_ae} × {cfg.primary_config}"))
        parts.append(t)
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()), notes


def rank_stability(ok, ref="ENN"):
    """09：每個 config 各自比較（config 內只挑 θ），數出策略贏過 ref 的 config 有幾個。
    不需要挑 config，結論不依賴任何一次 config 選擇（與 N 的 rank_stability 相同想法）。"""
    d = ok[ok["Space"] != "OF"]
    if d.empty or not (d["Strategy"] == ref).any():
        return pd.DataFrame()
    best, _ = _pick(d, G_US + ["Config"], ["ParamID"])
    if best.empty:
        return pd.DataFrame()
    ds = _ds_level(best, extra=("Config",))
    on = ["Space", "AE", "OCC", "Config", "Dataset"]
    r = ds[ds["Strategy"] == ref][on + ["AUC"]].rename(columns={"AUC": "AUC_ref"})
    m = ds[ds["Strategy"] != ref].merge(r, on=on)
    if m.empty:
        return pd.DataFrame()
    m["Δ"] = m["AUC"] - m["AUC_ref"]
    per_cfg = m.groupby(["Space", "Strategy", "Study", "AE", "OCC", "Config"])["Δ"].mean().reset_index()
    rows = []
    for k, g in per_cfg.groupby(["Space", "Strategy", "Study", "AE", "OCC"], sort=False):
        x = g["Δ"].values
        rows.append({"Space": SPACE_LABEL[k[0]], "Strategy": k[1], "Study": k[2], "AE": k[3], "OCC": k[4],
                     "對照": ref, "config 數": len(x), "贏的 config 數": int((x > TIE_TOL).sum()),
                     "輸的 config 數": int((x < -TIE_TOL).sum()), "平均ΔAUC（跨 config）": float(x.mean()),
                     "最小ΔAUC": float(x.min()), "最大ΔAUC": float(x.max())})
    out = pd.DataFrame(rows)
    order = {s: i for i, s in enumerate(STRATEGIES)}
    return out.sort_values(["Space", "OCC", "AE", "Strategy"],
                           key=lambda s: s.map(order) if s.name == "Strategy" else s).reset_index(drop=True)


def build_log(cfg, datasets, df):
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


def readme_table(cfg, datasets, complete, notes):
    g = cfg.grid
    items = [
        ("結果狀態", "✔ 正式：全部完成、無重複、逐鍵清點通過、none 與 A~C run 逐列一致、無 sampler 防呆 fallback、"
                     "統整設定與開跑時相同" if complete
         else "⚠ 暫存（PARTIAL）：尚未全部完成或核對未通過（原因見「備註」），只能看進度，不可用於論文"),
        ("這份檔案是什麼", f"Study 2（I/J/K）與 Study 3 路線一（M/N/O）的 OCC 調參結果（run = {cfg.run_name}）"),
        ("方法對照", "OF：none＝A、I＝單一 sampler、M＝ensemble｜DF：none＝B、J、N｜OF+DF：none＝C、K、O。"
                   "同一支程式、同一顆 AE、同一份特徵，所以所有策略可逐資料集配對比較。"),
        ("資料集", f"{len(datasets)} 個：{', '.join(datasets)}"),
        ("流程", "特徵 → MinMax（只 fit 未清理的訓練多數類）→ sampler 在這個座標下清理多數類 → OCC 在清理後的多數類上訓練"
               "（不再二次 MinMax）→ 測試。"),
        ("方法定位", "以已標記的少數類輔助挑選訓練樣本（label-informed data cleaning），再以 one-class 方式訓練 OCC："
                   "訓練折的少數類只用來決定刪哪些多數類，不進 OCC 訓練；測試折全程不參與 AE、清理與挑選以外的任何步驟。"
                   "論文不宜寫成「整條流程只用正常資料」。"),
        ("sampler（固定，不調）", f"ENN(n_neighbors={cfg.enn_k}, kind_sel=all)｜CNN(n_neighbors={cfg.cnn_k}, "
                                f"random_state={cfg.cnn_seed})｜TomekLinks；sampling_strategy=auto（只刪多數類）。與 I/J/K/N 相同。"),
        ("ensemble 策略", "S1_Union ≥1 個 sampler 保留就保留｜S2_Majority ≥2 個｜S3_Intersection 3 個都保留｜"
                        "S4_ENN_TL_abl：ENN 與 TL 都保留（去掉 CNN 的消融，屬事後提出）。三個 sampler 從同一份未清理資料"
                        "平行出發，投票對象是同一筆多數類樣本，不做串接、不重複樣本（與 N 相同）。"),
        ("結構性等價（附假設的命題）", "前提：同一份資料與座標、歐氏距離、沒有距離平手、ENN 用 kind_sel=all、CNN 用 1-NN、"
                            "sampler 正常完成。在此前提下：① TL 刪的點一定也被 ENN(k=3, all) 刪（其最近鄰就是少數類）"
                            "→ ENN 保留 ⊆ TL 保留 → S4 ≡ ENN；② TL 刪的點一定被 CNN 保留（CNN 的起始儲存集含全部少數類，"
                            "該點的 1-NN 是少數類而被誤分，因此會被加入）→ 三者同時刪除的點為空 → S1 ≡ none。"
                            "這是推論，不是 imblearn 文件的宣告；03b 分頁是實際資料上的驗證結果（屬實驗結果，不能用來拿掉前提）。"
                            "若比例不是 100%，應檢查重複點與距離平手。"),
        ("無效格子", f"清理後多數類 < {MIN_KEEP} 筆 → degenerate（資料本身造成，屬正常結果）：AUC 留空，"
                   "該 (策略, 資料集) 若因此不足 5 折就不會被選出（OF 空間＝該資料集整個不列入；AE 空間＝該 config 不列入候選）。"
                   "01 總表每個策略的「資料集數」可能不同，跨策略比較請看 02 分頁（只用兩邊都有效的共同資料集配對）。"
                   "sampler 執行出錯 → 該格不存檔、原因寫進 errors.log、重跑會再試；防呆條件（少數類被改動、索引重複）"
                   "→ invalid_fallback（不讓失敗的 sampler 投「全留」），只要出現就整份標 PARTIAL。"),
        ("OCC 參數「59 組」的實際有效數", "每個策略、每一格都會跑完 59 組並各存一列。但清理後筆數 n 很少時，"
                                      "LOF 的 k 會被截成 n−1、iForest 的 max_samples 被截成 n，多組會變成同一個模型；"
                                      "03 分頁列出「實際有效組數」、有參數被截斷的格子比例、以及截斷後參數合併"
                                      "（實際組數 < 名義）的格子比例（A~C 在 n<256 時 iForest 也有同樣情形）。"),
        ("調參方式與定位", "sampler 固定、只調 OCC。每個 (方法, 策略, 資料集, AE, OCC) 在 21 個 config × 59 組中挑 5 折平均 AUC "
                        "最高者（OF 空間只挑 59 組）。⚠ 挑選與報告用的是同一批測試折 → 屬「最佳可達表現」（oracle），"
                        "有樂觀偏誤；主表應描述為「候選格內事後選出的最高測試表現」，不能寫成「調參後在未見資料上一定更好」。"
                        "AE 空間的候選（21×59）比 OF 空間（59）多，oracle 偏誤也可能較大。Holm 校正處理的是多重比較，"
                        "不處理選參偏誤。要主張泛化效果需另做內層驗證（nested CV）。"),
        ("02 分頁怎麼看", "逐資料集配對、統計單位是資料集：策略（已調參）− 對照（已調參）。對照有三種："
                        "none（清理有沒有幫助）、ENN（事前指定的主要對照，贏 ENN 不代表贏 CNN/TL）、"
                        "單一 sampler 逐資料集最佳（事後挑選、最嚴格）。Wilcoxon signed-rank（雙尾）："
                        f"|Δ| ≤ {TIE_TOL:g} 視為平手並剔除（與勝/平/負同一標準），剔除後不足 {MIN_PAIRS} 個資料集不檢定；"
                        "沒有同值用精確分布，有同值用常態近似＋連續性校正（與 R 的 wilcox.test 預設相同，"
                        "不依 scipy 版本改變）。Wilcoxon_p 未校正；Holm_p 為同一對照、同一空間×AE×OCC 內的多重比較校正"
                        "（S1 vs none、S4 vs ENN 的差異全為 0 → p 留空、不佔族群名額；但 S4 vs none 與 ENN vs none "
                        "是同一個檢定，會重複佔一個名額，Holm 因此略偏保守）。"
                        "宣稱顯著時請用事前界定的主要比較與 Holm_p。"),
        ("05 / 08 / 09 各自還做了哪些挑選", "這三頁是用來拆開「挑選因素」的影響，不能合起來當成已排除 oracle 偏誤的證據：\n"
                                        "05 固定參數排名：θ 固定（同一組套到所有資料集），但 AE config 仍逐資料集挑。\n"
                                        f"08 固定表徵（{cfg.primary_ae} × {cfg.primary_config}，OF 空間無 AE）："
                                        "「θ 各自挑選」版仍逐資料集挑 θ；「θ 固定舊預設」版完全不挑 AE/config/θ，"
                                        "但其中「vs 單一 sampler 逐資料集最佳」的對照本身仍是事後挑選。\n"
                                        "09 跨 config 穩定性：不挑 config（逐一列出），但每個 config 內仍逐資料集挑 θ。"),
        ("尚未包含", "N 的「同刪除率隨機刪除」對照（rate-matched）不在本檔；需要時另以獨立 run 補做（不影響本 run）。"),
        ("門檻", f"訓練（清理後）多數類異常分數第 {THRESHOLD_PCT} 百分位；LOF 訓練分數用 fit 時算好的值（與 A~C 調參相同）。"),
        ("OCC 參數格", f"OCSVM nu ∈ {g['OCSVM']['nu']}、gamma ∈ {g['OCSVM']['gamma']}｜LOF n_neighbors ∈ "
                     f"{g['LOF']['n_neighbors']}｜iForest n_estimators ∈ {g['iForest']['n_estimators']}、"
                     f"max_samples ∈ {g['iForest']['max_samples']}（與 A~C 調參相同）"),
        ("與舊 I/J/K/N 的差異", "① OCC 參數改為調參；② AE 用 A~C 調參的每格固定種子與同一份 DF 快取；③ 名目欄位改為 train fold "
                              "編碼表（abalone19 可納入，舊 N 將其排除）；④ sampler 執行出錯不再默默判無效（該格失敗、"
                              "記錄原因、重跑再試）；⑤ LOF 門檻用 fit 時的訓練分數；⑥ Wilcoxon 的平手與檢定方法固定"
                              "（舊 N 用 wilcoxon(d) 預設值）。"
                              "舊預設參數下的 AUC 與舊 I 相同（I 沒有 AE，已測試）。"),
        ("AE", f"{cfg.ae_types}｜{len(cfg.configs)} configs｜epochs={cfg.ae_epochs}"),
        ("10 逐鍵清點", "從 signature.json 推出應有的每一個鍵（資料集×方法×fold×AE×config×OCC×參數），逐鍵比對缺少／多出／重複，"
                          "並檢查數值有效；任何一項不通過，檔名就標 _PARTIAL。與 audit_tuning_counts.py 同一套邏輯。"),
        ("備註", "；".join(notes) if notes else "無"),
        ("版本", json.dumps({**core.soft_info(cfg), "imblearn": imblearn.__version__}, ensure_ascii=False)),
    ]
    return pd.DataFrame(items, columns=["項目", "內容"])


def _order(df):
    so = {s: i for i, s in enumerate(STRATEGIES)}
    sp = {s: i for i, s in enumerate(SPACES)}
    sp.update({SPACE_LABEL[k]: v for k, v in sp.copy().items()})
    oc = {o: i for i, o in enumerate(OCC_TYPES)}
    maps = {"Space": sp, "OCC": oc, "Strategy": so}
    return df.sort_values(["Space", "OCC", "AE", "Strategy"],
                          key=lambda s: s.astype(str).map(maps[s.name]) if s.name in maps
                          else s.astype(str)).reset_index(drop=True)


def summarize(cfg: UsConfig):
    run_dir = Path(cfg.out_root) / cfg.run_name
    datasets, _ = core.discover_datasets(cfg)
    # 統整設定必須與開跑時相同（--summarize-only 前改過 runner 會在這裡被抓出來）
    guard_problems, guard_info = summary_guard(cfg, us_signature, US_SUMMARY_KEYS, datasets)
    df, n_unknown = load_rows(cfg, datasets)
    if df.empty:
        print("沒有結果可統整。")
        return None
    notes = guard_problems + guard_info
    ok = df[df["Status"] == "ok"]
    best, _ = _pick(ok, G_US, ["Config", "ParamID"])
    dflt, _ = _pick(ok[ok["IsDefault"]], G_US, ["Config", "ParamID"])
    log = build_log(cfg, datasets, df)
    if best.empty or dflt.empty:
        out = run_dir / f"{cfg.run_name}_summary_PARTIAL.xlsx"
        core.write_excel(out, [("00_說明", readme_table(cfg, datasets, False,
                                                         notes + ["目前還沒有任何候選跑滿 5 折，只輸出進度"]), None),
                               ("07_執行紀錄", log, None)])
        print(f"⚠️  目前還沒有任何設定跑滿 5 折，只輸出進度：{out.resolve()}")
        return out

    keys = ("Study", "Space", "Strategy", "AE", "OCC")
    ov_b, ov_d = ens.overall(best, "調參後", keys), ens.overall(dflt, "原固定參數", keys)
    k = list(keys)
    # 刪除率取「被選中組合」自己的 5 折（與同一列的 AUC 對應）
    rr = best.groupby(k, observed=True).agg(刪除率_選中組合=("RemovedRate", "mean"),
                                            保留筆數_選中組合=("MajKept", "mean")).reset_index()
    for c in k:
        rr[c] = rr[c].astype(str)
    tab = ov_b[k + ["資料集數", "AUC", "AUC_mean", "F1", "G-mean", "最常選中參數", "最常選中 Config"]].rename(
        columns={"AUC": "AUC_調參後", "AUC_mean": "m_b", "F1": "F1_調參後", "G-mean": "G-mean_調參後"})
    for c in k:
        tab[c] = tab[c].astype(str)
    od = ov_d[k + ["AUC", "AUC_mean"]].rename(columns={"AUC": "AUC_原固定參數", "AUC_mean": "m_d"})
    for c in k:
        od[c] = od[c].astype(str)
    tab = tab.merge(od, on=k, how="left")
    tab["ΔAUC（調參後−原固定）"] = tab["m_b"] - tab["m_d"]
    tab = tab.merge(rr, on=k, how="left").drop(columns=["m_b", "m_d"])
    tab["Space"] = tab["Space"].map(SPACE_LABEL)
    tab = _order(tab)
    # 資訊（不影響 PARTIAL）：有效資料集數比 none 少的策略 → 01 的平均不能直接跨策略比較，請看 02。
    # 期望的列由「空間 × 本次策略 × AE × OCC」展開，完全沒有有效資料集（01 沒有那一列）的策略也會列出（記為 0）。
    try:
        exp = pd.DataFrame([(SPACE_LABEL[sp], st, ae, occ) for sp in active_spaces(cfg)
                            for st in active_strategies(cfg, sp)
                            for ae in (["-"] if sp == "OF" else list(cfg.ae_types)) for occ in OCC_TYPES],
                           columns=["Space", "Strategy", "AE", "OCC"])
        nd = exp.merge(tab[["Space", "Strategy", "AE", "OCC", "資料集數"]], how="left",
                       on=["Space", "Strategy", "AE", "OCC"])
        nd["資料集數"] = nd["資料集數"].fillna(0).astype(int)
        base_n = nd[nd["Strategy"] == "none"].drop(columns="Strategy").rename(columns={"資料集數": "n_none"})
        mm = nd[nd["Strategy"] != "none"].merge(base_n, on=["Space", "AE", "OCC"])
        fewer = mm[mm["資料集數"] < mm["n_none"]]
        if len(fewer):
            items = [f"{r['Space']}/{r['Strategy']}/{r['AE']}/{r['OCC']}（{r['資料集數']}<{r['n_none']}）"
                     for _, r in fewer.head(12).iterrows()]
            notes.append(f"有 {len(fewer)} 列策略的有效資料集數比 none 少（01 的平均不宜跨策略直接比較，請看 02；"
                         f"0 表示該策略在所有資料集都不足 5 折有效）：" + "、".join(items)
                         + ("…" if len(fewer) > 12 else ""))
    except Exception as e:
        notes.append(f"（有效資料集數比對無法產生：{type(e).__name__}: {e}）")

    errs = []
    n1 = []

    def fixed_rep():
        t, nn = fixed_rep_table(ok, cfg)
        n1.extend(nn)
        return t

    def fixed_rank():
        fixed, _ = _pick(ok, G_US + ["ParamID"], ["Config"])
        if fixed.empty:
            return pd.DataFrame()
        fx = (fixed.groupby(["Study", "Space", "Strategy", "AE", "OCC", "ParamID"], observed=True)
                   .agg(AUC_mean=("AUC", "mean"), AUC_std=("AUC", "std"), 資料集數=("Dataset", "nunique"),
                        是否舊預設=("IsDefault", "first")).reset_index())
        fx["排名"] = fx.groupby(["Study", "Space", "Strategy", "AE", "OCC"], observed=True)["AUC_mean"].rank(
            ascending=False, method="min").astype(int)
        fx.insert(0, "說明", "同一組 θ 套到所有資料集；AE 空間的 config 仍逐資料集挑（屬事後挑選，非完全固定）")
        return fx.sort_values(["Study", "Space", "Strategy", "AE", "OCC", "排名"])

    sheets = {
        "02": safe_table("02 策略配對比較", lambda: comparisons(best), errs),
        "03": safe_table("03 刪除率與有效參數", lambda: removal_table(df, cfg), errs),
        "03b": safe_table("03b 兩兩策略等價率", lambda: equivalence_table(df), errs),
        "04": safe_table("04 每資料集最佳", lambda: ens.per_dataset_table(best, "調參後", G=G_US), errs),
        "04b": safe_table("04b 最佳組合逐折明細", lambda: best[G_US + ["Config", "ParamID", "Fold", "MajKept",
                                                                   "RemovedRate"] + METRIC_COLS + EXTRA_COLS]
                          .astype({c: str for c in G_US + ["Config", "ParamID"]}).sort_values(G_US + ["Fold"]), errs),
        "05": safe_table("05 固定參數排名", fixed_rank, errs),
        "08": safe_table("08 固定表徵比較", fixed_rep, errs),
        "09": safe_table("09 跨 config 穩定性", lambda: rank_stability(ok, "ENN"), errs),
    }
    notes += n1 + errs

    none_rows = df[df["Strategy"] == "none"].copy()
    chk = ens.check_against_abc(cfg, none_rows)
    audit_tab, audit_ok = ens.row_level_audit("us", cfg)
    if not audit_ok:
        notes.append("逐鍵清點未通過（見 10 分頁）")
    dup = int(df.duplicated(["Space", "Strategy", "Dataset", "Fold", "AE", "Config", "OCC", "ParamID"]).sum())
    # 沒填 abc_run_name ＝ 沒有核對 → 不能算正式結果
    abc_ok = bool(cfg.abc_run_name) and "結果" in chk and chk["結果"].astype(str).str.startswith("✔").all()
    if not cfg.abc_run_name:
        notes.append("未設定 abc_run_name：none 沒有與 A~C run 核對")
    elif not abc_ok:
        notes.append("none 與 A~C run 核對未通過（見 06 分頁）")
    if dup:
        notes.append(f"有 {dup} 列重複")
    n_fb = int(df.drop_duplicates(["Space", "Strategy", "Dataset", "Fold", "AE", "Config"])["Status"]
               .eq("invalid_fallback").sum())
    if n_fb:
        notes.append(f"有 {n_fb} 個（格子×策略）出現 sampler 防呆 fallback（見 03 分頁），請檢查 sampler 與資料")
    if n_unknown:
        notes.append(f"有 {n_unknown} 列的名稱不在本次設定內而被略過（設定可能被改過）")
    complete = (bool(log["完整"].all()) and dup == 0 and abc_ok and not errs and audit_ok
                and n_fb == 0 and n_unknown == 0 and not guard_problems)
    out = run_dir / (f"{cfg.run_name}_summary.xlsx" if complete else f"{cfg.run_name}_summary_PARTIAL.xlsx")
    delta = ["平均ΔAUC", "中位數ΔAUC"]
    core.write_excel(out, [
        ("00_說明", readme_table(cfg, datasets, complete, notes), None),
        ("01_總表", tab, ["ΔAUC（調參後−原固定）"]),
        ("02_策略配對比較", sheets["02"], delta),
        ("03_刪除率與有效參數", sheets["03"], None),
        ("03b_兩兩策略等價率", sheets["03b"], None),
        ("04_每資料集最佳", sheets["04"], None),
        ("04b_最佳組合逐折明細", sheets["04b"], None),
        ("05_固定參數排名", sheets["05"], None),
        ("06_none與AC結果核對", chk, None),
        ("07_執行紀錄", log, None),
        ("08_固定表徵比較", sheets["08"], delta),
        ("09_跨config穩定性", sheets["09"], ["平均ΔAUC（跨 config）"]),
        ("10_逐鍵清點", audit_tab, None),
    ])
    print(f"✅ 統整完成：{out.resolve()}")
    if complete:
        print(f"✔ 驗收通過：{len(datasets)} 個資料集 × {cfg.studies} 全部完成，無重複"
              + ("，none 與 A~C run 逐列一致。" if cfg.abc_run_name else "。"))
    else:
        print("⚠️  尚未完成或核對未通過 → 檔名標 _PARTIAL，只能當進度參考。" + "；".join(notes))
    print("\n── 各策略（已調參）AUC ──")
    print(tab[["Study", "Space", "Strategy", "AE", "OCC", "AUC_調參後", "AUC_原固定參數", "刪除率_選中組合"]]
          .to_string(index=False))
    return out


def main(cfg: UsConfig, argv=None):
    ap = argparse.ArgumentParser(description="I/J/K/M/N/O under-sampling OCC 調參")
    ap.add_argument("--summarize-only", action="store_true", help="不跑實驗，只把已完成的結果統整成 Excel")
    args = ap.parse_args(argv)
    if not args.summarize_only:
        if run(cfg) == "interrupted":
            return
    summarize(cfg)
