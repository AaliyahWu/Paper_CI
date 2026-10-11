"""
occ_tune_ensemble.py
====================
Study 1 的 D / E / F（決策層 ensemble）OCC 超參數調整：共用核心程式
（沿用 occ_tune_core.py 的讀資料、AE、OCC、快取、鎖、完成判定；
  筆電測試用 run_laptop_DEF.py，Azure 全跑用 run_azure_DEF.py）

  D / E：同一顆 AE、三種特徵空間的成員（跟 A / B / C 完全同一套流程）
    A 成員：OCC on OF            B 成員：OCC on DF            C 成員：OCC on [OF, DF]
    D：三個成員一起投票
      Vote3_hard ≥2 票｜Vote3_soft 等權平均｜Vote3_soft_istd 1/std 加權
    E：兩兩一組（AB / AC / BC）× {_eq, _istd, _hard_or ≥1 票, _hard_and 2 票}
  F：同一特徵空間（DF）、四種 AE 的成員（異質 AE ensemble）
    成員 = B_AE / B_DAE / B_SAE / B_VAE（同一個 config、同一種 OCC）
    Hetero_Vote_{eq, istd, hard_or ≥1 票, hard_maj ≥3 票}
    Hetero_Pair_{AE1}_{AE2}_{eq, istd, hard_or, hard_and}（C(4,2)=6 組）
  ※ D/E 的差異在「特徵空間」，F 的差異在「AE 種類」——F 不是 OF/DF/OF+DF 的投票。

────────────────────────────────────────────────────────────────────────────
ensemble 的「OCC 參數」怎麼調？兩種都做，外加原固定參數當對照
────────────────────────────────────────────────────────────────────────────
 (1) 共用參數（主要結果）：所有成員用「同一組」OCC 參數 θ，
     每個 (資料集, AE, OCC) 在 21 個 AE config × |θ| 組中挑 5 折平均 AUC 最高者。
     → 候選數與 B / C 完全相同，ensemble 沒有比 B/C 多試，比較才公平。
 (2) 成員各自調參（對照）：每個成員先用自己在同一 config 下 5 折平均 AUC 最高的 θ，
     組成 ensemble 後只挑 config。
 (3) 原固定參數：所有成員都用舊預設 → 用來算「調參前 vs 調參後」。

────────────────────────────────────────────────────────────────────────────
工作格（每一格寫檔成功 ＝ 完成；中斷後重跑只補沒完成的格子）
────────────────────────────────────────────────────────────────────────────
 D/E：OF 格（資料集×fold）→ DF 格（資料集×fold×AE×config）→ MT 格（成員各自調參）
 F  ：FH 格（資料集×fold×config，四種 AE 一起）→ FMT 格（成員各自調參）
 AE 特徵直接讀 A~C 調參留下的 _df_cache（out_root 相同就共用，不重訓 AE）

輸出：results_tuning/<run_name>/<run_name>_summary.xlsx（未完成時檔名加 _PARTIAL）
"""
from __future__ import annotations

import os
import occ_tune_core as core          # 必須最先 import：core 會先設定執行緒環境變數再載入 numpy / torch

import json
import time
import signal
import socket
import argparse
from itertools import combinations
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, average_precision_score

from occ_tune_core import (N_FOLDS, THRESHOLD_PCT, OCC_TYPES, METRIC_COLS, EXTRA_COLS,
                           LEGACY_DEFAULT, param_combos, param_id, is_legacy_default,
                           safe_name, dict_hash)

ENS_VERSION = "DEF_tune_v3"           # ensemble 計算方式的版本（寫進設定簽章）
#   v3：數值算法與 v2 完全相同；程式指紋改為涵蓋所有會產生數值的函式與方法定義，
#       統整時核對設定與開跑時相同，並加上 SIGTERM 中止與 AE 快取開跑前檢查。
EPS = 1e-12                           # 與舊 D / E / F 相同：正規化與 1/std 的除零保護
MEMBERS = ("A", "B", "C")
PAIRS = {"AB": ("A", "B"), "AC": ("A", "C"), "BC": ("B", "C")}
DE_METHODS = {
    "D": ["Vote3_hard", "Vote3_soft", "Vote3_soft_istd"],
    "E": [f"Pair_{p}_{s}" for p in PAIRS for s in ("eq", "istd", "hard_or", "hard_and")],
}
DE_MEMBERS = {"Vote3_hard": MEMBERS, "Vote3_soft": MEMBERS, "Vote3_soft_istd": MEMBERS,
              **{f"Pair_{p}_{s}": PAIRS[p] for p in PAIRS for s in ("eq", "istd", "hard_or", "hard_and")}}
F_VOTE = ["Hetero_Vote_eq", "Hetero_Vote_istd", "Hetero_Vote_hard_or", "Hetero_Vote_hard_maj"]
PAIR_SCHEMES = ("eq", "istd", "hard_or", "hard_and")
MODE_LABEL = {"shared": "共用參數", "member": "成員各自調參", "default": "原固定參數"}
ROW_COLS = (["Study", "Method", "Dataset", "Fold", "AE", "Config", "OCC", "ParamID", "IsDefault"]
            + METRIC_COLS + EXTRA_COLS)
ALL_AE = "ALL"                        # F 格子的 AE 欄位（四種 AE 一起算）


@dataclass
class EnsConfig(core.RunConfig):
    run_name: str = "DEF_tune"
    studies: list = field(default_factory=lambda: ["D", "E", "F"])
    # "shared" = 共用參數（必跑）；"member" = 成員各自調參
    modes: list = field(default_factory=lambda: ["shared", "member"])
    # A~C 調參的 run_name（同一個 out_root 底下）。統整時逐列核對 A/B/C 成員數值是否一致；
    # 沒填就無法核對，統整結果一律標 _PARTIAL。
    abc_run_name: str = ""
    # 開跑前會確認 A~C 留下的 AE 特徵快取齊全；缺檔就停止（避免默默重訓 AE、數值與 A~C 不同）。
    # 確定要重新訓練缺少的 AE 時才改成 True。
    allow_ae_retrain: bool = False


# ═════════════════════════ 方法清單 ════════════════════════════════════════
def f_pairs(cfg):
    return list(combinations(cfg.ae_types, 2))


def f_methods(cfg):
    return F_VOTE + [f"Hetero_Pair_{a}_{b}_{s}" for a, b in f_pairs(cfg) for s in PAIR_SCHEMES]


def methods_of(study, cfg):
    return f_methods(cfg) if study == "F" else DE_METHODS[study]


def all_methods(cfg):
    return list(MEMBERS) + DE_METHODS["D"] + DE_METHODS["E"] + f_methods(cfg)


def f_ae_groups(cfg):
    return list(dict.fromkeys(["+".join(cfg.ae_types)] + [f"{a}+{b}" for a, b in f_pairs(cfg)]))


def method_members(method, cfg):
    """回傳 ensemble 的成員清單：D/E 為 A/B/C；F 為 AE 名稱。"""
    if method in DE_MEMBERS:
        return DE_MEMBERS[method]
    if method.startswith("Hetero_Vote_"):
        return tuple(cfg.ae_types)
    parts = method.split("_")                      # Hetero_Pair_AE1_AE2_scheme...
    return (parts[2], parts[3])


def member_keys(study, method, ae, cfg):
    """ensemble 成員對應到「單一成員結果」的 (Study, AE) 鍵，用來和已調參的單一成員比較。"""
    if study in ("D", "E"):
        return [(t, "-" if t == "A" else ae) for t in DE_MEMBERS[method]]
    return [("B", a) for a in method_members(method, cfg)]


def n_methods(cfg, studies=None):
    return sum(len(methods_of(s, cfg)) for s in (studies or cfg.studies))


def n_params(cfg):
    return sum(len(param_combos(o, cfg.grid)) for o in OCC_TYPES)


def param_lookup(cfg):
    return {occ: {param_id(p): p for p in param_combos(occ, cfg.grid)} for occ in OCC_TYPES}


# ═════════════════════════ 指標與投票 ═══════════════════════════════════════
def _metrics(y, pred, score):
    """AUC / AP 用連續分數（sklearn，與 A~C 相同）；F1 / Recall / G-mean 用 0/1 預測。"""
    yb, pb = y.astype(bool), np.asarray(pred).astype(bool)
    tp = int(np.sum(pb & yb)); fp = int(np.sum(pb & ~yb))
    fn = int(np.sum(~pb & yb)); tn = int(np.sum(~pb & ~yb))
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {"AUC": float(roc_auc_score(y, score)), "F1": float(f1), "Recall": float(rec),
            "G-mean": float(np.sqrt(rec * spec)), "AP": float(average_precision_score(y, score))}


def _prep(s_tr, s_te):
    """一個成員：① 自己的門檻（訓練分數第 90 百分位）→ 0/1 預測，與單獨跑完全相同；
       ② 用訓練多數類分數的 min/max 正規化（不碰測試集統計）。"""
    thr = np.percentile(s_tr, THRESHOLD_PCT)
    lo, hi = float(s_tr.min()), float(s_tr.max())
    if hi - lo < EPS:
        n_tr, n_te = np.zeros_like(s_tr), np.zeros_like(s_te)
    else:
        n_tr, n_te = (s_tr - lo) / (hi - lo), (s_te - lo) / (hi - lo)
    return {"s_te": s_te, "pred": (s_te >= thr).astype(np.int8),
            "n_tr": n_tr, "n_te": n_te, "std": float(np.std(n_tr))}


def _soft(y, ms, scheme):
    if scheme == "eq":
        w = np.full(len(ms), 1.0 / len(ms))
    else:                                                   # istd：w ∝ 1/std，再歸一化成總和 1
        inv = np.array([1.0 / (m["std"] + EPS) for m in ms])
        w = inv / inv.sum()
    s_tr = sum(wi * m["n_tr"] for wi, m in zip(w, ms))
    s_te = sum(wi * m["n_te"] for wi, m in zip(w, ms))
    thr = np.percentile(s_tr, THRESHOLD_PCT)                # ensemble 分數自己的第 90 百分位
    return _metrics(y, s_te >= thr, s_te)


def _hard(y, ms, k):
    vote = np.sum([m["pred"].astype(int) for m in ms], axis=0)
    return _metrics(y, vote >= k, vote.astype(float))       # AUC 用票數排序（與舊 D/E/F 相同）


def ensemble_all(y, prepped, studies):
    """D / E。回傳 [(Study, Method, metrics), ...]；prepped = {"A": _prep(...), "B": ..., "C": ...}"""
    out = []
    if "D" in studies:
        ms = [prepped[t] for t in MEMBERS]
        out += [("D", "Vote3_hard", _hard(y, ms, 2)),
                ("D", "Vote3_soft", _soft(y, ms, "eq")),
                ("D", "Vote3_soft_istd", _soft(y, ms, "istd"))]
    if "E" in studies:
        for p, (i, j) in PAIRS.items():
            ms = [prepped[i], prepped[j]]
            out += [("E", f"Pair_{p}_eq", _soft(y, ms, "eq")),
                    ("E", f"Pair_{p}_istd", _soft(y, ms, "istd")),
                    ("E", f"Pair_{p}_hard_or", _hard(y, ms, 1)),
                    ("E", f"Pair_{p}_hard_and", _hard(y, ms, 2))]
    return out


def ensemble_F(y, prepped, aes):
    """F。回傳 [(Method, AE 群組, metrics), ...]；prepped = {"AE": _prep(...), "DAE": ..., ...}
    與舊 F 相同：Vote 用 or（≥1）與 strict majority（≥⌊n/2⌋+1，4 個 AE 時 ≥3）；Pair 用 or / and。"""
    out = []
    ms = [prepped[a] for a in aes]
    grp = "+".join(aes)
    out += [("Hetero_Vote_eq", grp, _soft(y, ms, "eq")),
            ("Hetero_Vote_istd", grp, _soft(y, ms, "istd")),
            ("Hetero_Vote_hard_or", grp, _hard(y, ms, 1)),
            ("Hetero_Vote_hard_maj", grp, _hard(y, ms, len(ms) // 2 + 1))]
    for a, b in combinations(aes, 2):
        pm = [prepped[a], prepped[b]]
        out += [(f"Hetero_Pair_{a}_{b}_eq", f"{a}+{b}", _soft(y, pm, "eq")),
                (f"Hetero_Pair_{a}_{b}_istd", f"{a}+{b}", _soft(y, pm, "istd")),
                (f"Hetero_Pair_{a}_{b}_hard_or", f"{a}+{b}", _hard(y, pm, 1)),
                (f"Hetero_Pair_{a}_{b}_hard_and", f"{a}+{b}", _hard(y, pm, 2))]
    return out


# ═════════════════════════ 成員分數 ════════════════════════════════════════
def occ_scores_all(Xtr, Xte, cfg):
    """同一份特徵跑完三個 OCC 的全部參數，回傳 {(occ, ParamID): (s_tr, s_te)}。
    與 core.eval_grid 一樣：LOF k / iForest max_samples 被截斷成同一值時直接沿用。"""
    out = {}
    for occ in OCC_TYPES:
        memo = {}
        for p in param_combos(occ, cfg.grid):
            key = None
            if occ == "LOF":
                key = int(min(p["n_neighbors"], len(Xtr) - 1))
            elif occ == "iForest":
                key = (p["n_estimators"], int(min(p["max_samples"], len(Xtr))))
            if key is not None and key in memo:
                sc = memo[key]
            else:
                s_tr, s_te, _ = core._fit_score(occ, p, Xtr, Xte, cfg.lof_train_score)
                sc = (np.asarray(s_tr, dtype=np.float64), np.asarray(s_te, dtype=np.float64))
                if key is not None:
                    memo[key] = sc
            out[(occ, param_id(p))] = sc
    return out


def of_cache_path(cfg, ds, fold):
    return Path(cfg.out_root) / cfg.run_name / "_of_scores" / safe_name(ds) / f"fold{fold}.npz"


def get_of_scores(cfg, ds, fold, fd):
    """A 成員（OF）的全部分數：有快取就讀，否則現算並存檔（先寫暫存檔再改名）。"""
    path = of_cache_path(cfg, ds, fold)
    keys = [(occ, param_id(p)) for occ in OCC_TYPES for p in param_combos(occ, cfg.grid)]
    meta = {"dataset": ds, "fold": fold, "train_sha1": fd["train_sha1"], "test_sha1": fd["test_sha1"],
            "grid": dict_hash(cfg.grid), "lof": cfg.lof_train_score, "keys": [f"{o}|{p}" for o, p in keys]}
    if path.exists():
        try:
            z = np.load(path, allow_pickle=False)
            if json.loads(str(z["meta"])) == meta:
                return {k: (z[f"tr{i}"], z[f"te{i}"]) for i, k in enumerate(keys)}
        except Exception:
            pass                                             # 壞檔或舊格式 → 重算
    scores = occ_scores_all(fd["X_maj_s"], fd["X_tst_s"], cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + f".tmp{os.getpid()}.npz")
    arrs = {}
    for i, k in enumerate(keys):
        arrs[f"tr{i}"], arrs[f"te{i}"] = scores[k]
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **arrs)
    os.replace(tmp, path)
    return scores


def _features_B(DF_maj, DF_tst):
    """與 core.run_task 的 B 完全相同：DF 再 MinMax（fit DF_maj）。"""
    sc = MinMaxScaler().fit(DF_maj)
    return sc.transform(DF_maj), sc.transform(DF_tst)


def _features_BC(fd, DF_maj, DF_tst):
    """與 core.run_task 完全相同：B = DF 再 MinMax；C = [OF, DF] 串接後 MinMax。"""
    B = _features_B(DF_maj, DF_tst)
    comb_maj, comb_tst = np.hstack([fd["X_maj_s"], DF_maj]), np.hstack([fd["X_tst_s"], DF_tst])
    sc = MinMaxScaler().fit(comb_maj)
    C = (sc.transform(comb_maj), sc.transform(comb_tst))
    return B, C


# ═════════════════════════ 工作格 ══════════════════════════════════════════
def cell_path(cfg, task, members=False):
    """OF: cells/OF/ds/fold1.csv.gz
       DF / MT / FH / FMT: cells/<kind>/ds/fold1__<AE>__<config>.csv.gz
       成員檔：DF → cells/DFm/...，FH → cells/FHm/...（主檔寫完才算完成）"""
    root = Path(cfg.out_root) / cfg.run_name / "cells"
    kind, ds, fold = task[0], task[1], task[2]
    if kind == "OF":
        return root / "OF" / safe_name(ds) / f"fold{fold}.csv.gz"
    ae, c = task[3], task[4]
    sub = f"{kind}m" if (members and kind in ("DF", "FH")) else kind
    return root / sub / safe_name(ds) / f"fold{fold}__{ae}__{safe_name(c)}.csv.gz"


def atomic_write_gz(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    pd.DataFrame(rows, columns=ROW_COLS).to_csv(tmp, index=False, encoding="utf-8", compression="gzip")
    os.replace(tmp, path)


def _check(rows, expected, what):
    if len(rows) != expected:
        raise RuntimeError(f"{what} 列數不符：{len(rows)} ≠ {expected}")
    bad = [r for r in rows if not np.isfinite(r["AUC"])]
    if bad:
        raise RuntimeError(f"{what} 有 {len(bad)} 列 AUC 無效，整格不存檔")


def _mrow(study, method, ds, fold, ae, c, occ, pid, isdef, m):
    return {"Study": study, "Method": method, "Dataset": ds, "Fold": fold, "AE": ae, "Config": c,
            "OCC": occ, "ParamID": pid, "IsDefault": isdef, **m}


def _task_of(task, cfg):
    _, ds, fold = task
    fd = core.load_fold(cfg.data_root, ds, fold)
    y = fd["y_tst"]
    scores = get_of_scores(cfg, ds, fold, fd)
    rows = []
    for occ in OCC_TYPES:
        for p in param_combos(occ, cfg.grid):
            pid = param_id(p)
            m = _prep(*scores[(occ, pid)])
            rows.append(_mrow("A", "A", ds, fold, "-", "-", occ, pid, is_legacy_default(occ, p),
                              _metrics(y, m["pred"], m["s_te"])))
    _check(rows, n_params(cfg), "OF")
    atomic_write_gz(rows, cell_path(cfg, task))
    return "-", fd["n_maj"]


def _de_studies(cfg):
    return [s for s in cfg.studies if s in ("D", "E")]


def _task_df(task, cfg):
    _, ds, fold, ae, c = task
    fd = core.load_fold(cfg.data_root, ds, fold)
    y = fd["y_tst"]
    of = get_of_scores(cfg, ds, fold, fd)
    DF_maj, DF_tst, src = core.get_df(cfg, ds, fold, ae, c, fd)
    (B_tr, B_te), (C_tr, C_te) = _features_BC(fd, DF_maj, DF_tst)
    sB, sC = occ_scores_all(B_tr, B_te, cfg), occ_scores_all(C_tr, C_te, cfg)
    studies = _de_studies(cfg)
    mem_rows, ens_rows = [], []
    for occ in OCC_TYPES:
        for p in param_combos(occ, cfg.grid):
            pid, isdef = param_id(p), is_legacy_default(occ, p)
            prepped = {"A": _prep(*of[(occ, pid)]), "B": _prep(*sB[(occ, pid)]), "C": _prep(*sC[(occ, pid)])}
            for t in ("B", "C"):
                mem_rows.append(_mrow(t, t, ds, fold, ae, c, occ, pid, isdef,
                                      _metrics(y, prepped[t]["pred"], prepped[t]["s_te"])))
            for study, method, m in ensemble_all(y, prepped, studies):
                ens_rows.append(_mrow(study, method, ds, fold, ae, c, occ, pid, isdef, m))
    _check(mem_rows, 2 * n_params(cfg), "DF 成員")
    _check(ens_rows, n_methods(cfg, studies) * n_params(cfg), "DF ensemble")
    atomic_write_gz(mem_rows, cell_path(cfg, task, members=True))   # 先寫成員檔
    atomic_write_gz(ens_rows, cell_path(cfg, task))                  # 最後寫主檔 ＝ 這格完成
    return src, fd["n_maj"]


def _mt_pid(members, pids):
    return " ".join(f"{t}[{pids[t]}]" for t in members)


def _task_mt(task, cfg):
    _, ds, fold, ae, c, plan = task
    fd = core.load_fold(cfg.data_root, ds, fold)
    y = fd["y_tst"]
    of = get_of_scores(cfg, ds, fold, fd)
    DF_maj, DF_tst, src = core.get_df(cfg, ds, fold, ae, c, fd)
    (B_tr, B_te), (C_tr, C_te) = _features_BC(fd, DF_maj, DF_tst)
    look = param_lookup(cfg)
    rows = []
    for occ, pa, pb, pc in plan:
        sb_tr, sb_te, _ = core._fit_score(occ, look[occ][pb], B_tr, B_te, cfg.lof_train_score)
        sc_tr, sc_te, _ = core._fit_score(occ, look[occ][pc], C_tr, C_te, cfg.lof_train_score)
        prepped = {"A": _prep(*of[(occ, pa)]),
                   "B": _prep(np.asarray(sb_tr, float), np.asarray(sb_te, float)),
                   "C": _prep(np.asarray(sc_tr, float), np.asarray(sc_te, float))}
        pids = {"A": pa, "B": pb, "C": pc}
        isdef = {t: is_legacy_default(occ, look[occ][pids[t]]) for t in MEMBERS}
        for study, method, m in ensemble_all(y, prepped, _de_studies(cfg)):
            mem = DE_MEMBERS[method]
            rows.append(_mrow(study, method, ds, fold, ae, c, occ, _mt_pid(mem, pids),
                              all(isdef[t] for t in mem), m))
    _check(rows, n_methods(cfg, _de_studies(cfg)) * len(OCC_TYPES), "MT")
    atomic_write_gz(rows, cell_path(cfg, task))
    return src, fd["n_maj"]


def _task_fh(task, cfg):
    """F：同一個 fold / config，四種 AE 的 DF 各自當一個成員（共用參數）。"""
    _, ds, fold, _, c = task
    fd = core.load_fold(cfg.data_root, ds, fold)
    y = fd["y_tst"]
    scores, srcs = {}, set()
    for ae in cfg.ae_types:
        DF_maj, DF_tst, src = core.get_df(cfg, ds, fold, ae, c, fd)
        srcs.add(src)
        scores[ae] = occ_scores_all(*_features_B(DF_maj, DF_tst), cfg)
    mem_rows, ens_rows = [], []
    for occ in OCC_TYPES:
        for p in param_combos(occ, cfg.grid):
            pid, isdef = param_id(p), is_legacy_default(occ, p)
            prepped = {ae: _prep(*scores[ae][(occ, pid)]) for ae in cfg.ae_types}
            for ae in cfg.ae_types:
                mem_rows.append(_mrow("B", "B", ds, fold, ae, c, occ, pid, isdef,
                                      _metrics(y, prepped[ae]["pred"], prepped[ae]["s_te"])))
            for method, grp, m in ensemble_F(y, prepped, cfg.ae_types):
                ens_rows.append(_mrow("F", method, ds, fold, grp, c, occ, pid, isdef, m))
    _check(mem_rows, len(cfg.ae_types) * n_params(cfg), "FH 成員")
    _check(ens_rows, len(f_methods(cfg)) * n_params(cfg), "FH ensemble")
    atomic_write_gz(mem_rows, cell_path(cfg, task, members=True))
    atomic_write_gz(ens_rows, cell_path(cfg, task))
    return "+".join(sorted(srcs)), fd["n_maj"]


def _task_fmt(task, cfg):
    """F 成員各自調參：plan = ((occ, {AE: ParamID}), ...)"""
    _, ds, fold, _, c, plan = task
    fd = core.load_fold(cfg.data_root, ds, fold)
    y = fd["y_tst"]
    look = param_lookup(cfg)
    feats, srcs = {}, set()
    for ae in cfg.ae_types:
        DF_maj, DF_tst, src = core.get_df(cfg, ds, fold, ae, c, fd)
        srcs.add(src)
        feats[ae] = _features_B(DF_maj, DF_tst)
    rows = []
    for occ, pid_items in plan:
        pids = dict(pid_items)
        prepped = {}
        for ae in cfg.ae_types:
            s_tr, s_te, _ = core._fit_score(occ, look[occ][pids[ae]], *feats[ae], cfg.lof_train_score)
            prepped[ae] = _prep(np.asarray(s_tr, float), np.asarray(s_te, float))
        isdef = {ae: is_legacy_default(occ, look[occ][pids[ae]]) for ae in cfg.ae_types}
        for method, grp, m in ensemble_F(y, prepped, cfg.ae_types):
            mem = method_members(method, cfg)
            rows.append(_mrow("F", method, ds, fold, grp, c, occ, _mt_pid(mem, pids),
                              all(isdef[a] for a in mem), m))
    _check(rows, len(f_methods(cfg)) * len(OCC_TYPES), "FMT")
    atomic_write_gz(rows, cell_path(cfg, task))
    return "+".join(sorted(srcs)), fd["n_maj"]


def run_ens_task(task, cfg):
    """worker 執行。成功 → 寫檔並回傳統計；失敗 → 丟例外（不寫檔，下次重跑會補）。"""
    torch.set_num_threads(max(1, cfg.torch_threads_per_job))
    t0 = time.perf_counter()
    fn = {"OF": _task_of, "DF": _task_df, "MT": _task_mt, "FH": _task_fh, "FMT": _task_fmt}[task[0]]
    src, n_maj = fn(task, cfg)
    return {"task": "|".join(map(str, task[:5])), "kind": task[0], "dataset": task[1],
            "n_maj": int(n_maj), "seconds": round(time.perf_counter() - t0, 3),
            "df_source": src, "host": socket.gethostname()}


# ═════════════════════════ 成員各自調參：決定每個成員的 θ ══════════════════
def _read_cells(files, dtypes=None):
    if not files:
        return pd.DataFrame(columns=ROW_COLS)
    return pd.concat((pd.read_csv(f, dtype=dtypes) for f in files), ignore_index=True)


def _pick(d, G, cand):
    """每組 G 挑 5 折平均 AUC 最高的 cand（只考慮 5 折完整者）；同分先選舊預設，再依名稱排序。
    規則與 core.select_rows 相同。回傳 (被選中的全部 fold 列, 選取表)。"""
    d = d[d["AUC"].notna()]
    agg = (d.groupby(G + cand, observed=True, sort=False)
             .agg(mAUC=("AUC", "mean"), nF=("Fold", "nunique"), nR=("Fold", "size"),
                  isdef=("IsDefault", "first")).reset_index())
    agg = agg[(agg["nF"] == N_FOLDS) & (agg["nR"] == N_FOLDS)].copy()
    if agg.empty:
        return d.iloc[0:0].assign(同分候選數=pd.Series(dtype=int)), agg
    agg["同分候選數"] = agg.groupby(G, observed=True)["mAUC"].transform(
        lambda x: int((x >= x.max() - 1e-12).sum()))
    sort_cols = []
    for c in G + cand:
        agg[f"_s_{c}"] = agg[c].astype(str)
        sort_cols.append(f"_s_{c}")
    sort_cols = sort_cols[:len(G)] + ["mAUC", "isdef"] + sort_cols[len(G):]
    agg = agg.sort_values(sort_cols, ascending=[True] * len(G) + [False, False] + [True] * len(cand))
    pick = agg.drop_duplicates([f"_s_{c}" for c in G])[G + cand + ["同分候選數"]]
    return d.merge(pick, on=G + cand), pick


def _member_files(cfg, datasets):
    """A 成員（OF 格）＋ B/C 成員（主檔已完成的 DF 格）＋ B 成員（主檔已完成的 FH 格）。"""
    root = Path(cfg.out_root) / cfg.run_name / "cells"
    files = [f for ds in datasets for f in sorted((root / "OF" / safe_name(ds)).glob("*.csv.gz"))]
    for ds in datasets:
        for kind in ("DF", "FH"):
            for f in sorted((root / f"{kind}m" / safe_name(ds)).glob("*.csv.gz")):
                if (root / kind / safe_name(ds) / f.name).exists():
                    files.append(f)
    return files


MEM_KEY = ["Study", "Dataset", "Fold", "AE", "Config", "OCC", "ParamID"]


def load_member_rows(cfg, datasets):
    df = _read_cells(_member_files(cfg, datasets),
                     {"ParamID": str, "Config": str, "AE": str, "Dataset": str, "Study": str, "OCC": str})
    if len(df):
        df["IsDefault"] = df["IsDefault"].astype(str).str.lower().eq("true")
        df = df.drop_duplicates(MEM_KEY)       # D/E 與 F 都會算到 B 成員（數值相同），只留一份
    return df


def build_member_plan(cfg, datasets, mem=None):
    """每個成員在每個 (資料集, AE, config, OCC) 的最佳 θ（以該成員 5 折平均 AUC 挑）。
    A 成員與 AE / config 無關，每個 (資料集, OCC) 一組。"""
    mem = load_member_rows(cfg, datasets) if mem is None else mem
    if mem.empty:
        return {}, {}
    planA, planBC = {}, {}
    if (mem["Study"] == "A").any():
        _, pa = _pick(mem[mem["Study"] == "A"], ["Dataset", "OCC"], ["ParamID"])
        planA = {(r.Dataset, r.OCC): r.ParamID for r in pa.itertuples()}
    if mem["Study"].isin(["B", "C"]).any():
        _, pbc = _pick(mem[mem["Study"].isin(["B", "C"])], ["Study", "Dataset", "AE", "Config", "OCC"],
                       ["ParamID"])
        planBC = {(r.Study, r.Dataset, r.AE, r.Config, r.OCC): r.ParamID for r in pbc.itertuples()}
    return planA, planBC


def mt_tasks(cfg, datasets, planA, planBC):
    tasks, missing = [], 0
    for ds in datasets:
        for f in range(1, N_FOLDS + 1):
            for ae in cfg.ae_types:
                for c in cfg.configs:
                    plan = []
                    for occ in OCC_TYPES:
                        k = (ds, ae, c, occ)
                        if (ds, occ) in planA and ("B",) + k in planBC and ("C",) + k in planBC:
                            plan.append((occ, planA[(ds, occ)], planBC[("B",) + k], planBC[("C",) + k]))
                    if len(plan) == len(OCC_TYPES):
                        tasks.append(("MT", ds, f, ae, c, tuple(plan)))
                    else:
                        missing += 1
    return tasks, missing


def fmt_tasks(cfg, datasets, planBC):
    tasks, missing = [], 0
    for ds in datasets:
        for f in range(1, N_FOLDS + 1):
            for c in cfg.configs:
                plan = []
                for occ in OCC_TYPES:
                    pids = tuple((ae, planBC.get(("B", ds, ae, c, occ))) for ae in cfg.ae_types)
                    if all(p is not None for _, p in pids):
                        plan.append((occ, pids))
                if len(plan) == len(OCC_TYPES):
                    tasks.append(("FMT", ds, f, ALL_AE, c, tuple(plan)))
                else:
                    missing += 1
    return tasks, missing


# ═════════════════════════ 設定簽章 ════════════════════════════════════════
def code_fingerprint(*funcs):
    """計算用函式的程式碼指紋：OCC / 投票 / 清理的程式一改，舊結果就不能接著續跑。"""
    import inspect
    import hashlib
    src = "".join(inspect.getsource(f) for f in funcs)
    return hashlib.sha256(src.encode()).hexdigest()[:12]


def lib_versions():
    """會影響數值的套件完整版本（含修補版號）。任何一個改變就不能接著續跑，避免新舊數值混在一起。"""
    import sklearn
    import scipy
    return {"sklearn": sklearn.__version__, "numpy": np.__version__, "scipy": scipy.__version__}


def ens_signature(cfg):
    return {"ens_version": ENS_VERSION, "studies": sorted(cfg.studies), "modes": sorted(cfg.modes),
            "ae_types": cfg.ae_types, "configs": cfg.configs, "grid": cfg.grid,
            "legacy_default": LEGACY_DEFAULT, "ae": core.ae_signature(cfg),
            "lof_train_score": cfg.lof_train_score, "threshold_pct": THRESHOLD_PCT,
            "cat_encoding": "train_fold_mapping_v1", "metrics": METRIC_COLS + EXTRA_COLS,
            "n_folds": N_FOLDS, "eps": EPS, "libs": lib_versions(),
            "method_defs": {"DE_MEMBERS": {k: list(v) for k, v in DE_MEMBERS.items()},
                            "F_VOTE": F_VOTE, "PAIR_SCHEMES": list(PAIR_SCHEMES)},
            # 會產生數值的函式全部納入：任何一個改了，舊格子就不能和新格子混在同一個 run
            "code": code_fingerprint(core._fit_score, core.get_df, occ_scores_all, _prep, _soft, _hard, _metrics,
                                     ensemble_all, ensemble_F, _features_B, _features_BC, get_of_scores,
                                     _task_of, _task_df, _mt_pid, _task_mt, _task_fh, _task_fmt, run_ens_task)}


# 統整時要與開跑時（signature.json）相同的鍵：不同就代表「統整用的候選集合 ≠ 實際跑的」
DEF_SUMMARY_KEYS = ["ens_version", "studies", "modes", "ae_types", "configs", "grid", "legacy_default",
                    "lof_train_score", "threshold_pct", "metrics", "n_folds", "eps", "method_defs"]

# 上一版（v2）開跑的 run 可以直接由本版接續與統整：v2 → v3 每一格的數值算法完全相同（已逐位元組比對），
# 差別只在統整檢查、中止處理與防呆。只認「版本字串＋當時的程式指紋」完全吻合的 v2（指紋由 v2 原始碼決定，
# 與機器無關），其餘設定鍵必須全部相同；接續時 signature.json 會升級為本版並在 sessions 記錄。
COMPAT_PREVIOUS = {"ens_version": {"DEF_tune_v2": "43fd6fb6c222"},
                   "us_version": {"US_tune_v2": "bb9a9b0abbdc"}}
NEW_ONLY_KEYS = {"method_defs"}               # v2 沒有記錄、但內容與 v3 相同的鍵


def upgradable_from_previous(old, new):
    """old / new：signature.json 的 "hard"（皆為 JSON 化後的 dict）。回傳相容的舊版字串或 None。"""
    for vk, allowed in COMPAT_PREVIOUS.items():
        ov = old.get(vk)
        if vk in new and ov in allowed and old.get("code") == allowed[ov]:
            ignore = {vk, "code"} | NEW_ONLY_KEYS
            if all(old.get(k) == new.get(k) for k in (set(old) | set(new)) - ignore):
                return ov
    return None


def summary_guard(cfg, sig_fn, keys, datasets):
    """統整前的一致性檢查（--summarize-only 也會做）。
    回傳 (會讓結果標 PARTIAL 的問題清單, 只需註記的資訊清單)。"""
    run_dir = Path(cfg.out_root) / cfg.run_name
    problems, info = [], []
    sig_file = run_dir / "signature.json"
    if not sig_file.exists():
        return ["找不到 signature.json（這個 run 還沒有開跑過）"], info
    old = json.loads(sig_file.read_text(encoding="utf-8")).get("hard", {})
    now = json.loads(json.dumps(sig_fn(cfg), default=str))
    prev = None
    for vk, allowed in COMPAT_PREVIOUS.items():   # v2 開跑、全程沒有用 v3 接續過的 run
        if vk in now and old.get(vk) in allowed and old.get("code") == allowed[old.get(vk)]:
            prev = old.get(vk)
            old = {**old, vk: now[vk], **{k: now.get(k) for k in NEW_ONLY_KEYS if k not in old}}
    if prev:
        info.append(f"此 run 由上一版程式（{prev}）產生；每一格的數值算法與本版相同")
    diff = [k for k in keys if old.get(k) != now.get(k)]
    if old.get("ae", {}).get("epochs") != now.get("ae", {}).get("epochs"):
        diff.append("ae_epochs")
    if diff:
        problems.append(f"統整時的設定與開跑時不同：{diff}（請用開跑時的 runner 設定統整，否則選法與清點的候選集合會不一致）")
    prof_file = run_dir / "dataset_profile.csv"
    if prof_file.exists():
        prof = pd.read_csv(prof_file)
        run_ds = prof.loc[prof["Included"].astype(str).str.lower() == "true", "Dataset"].tolist()
        extra = [d for d in datasets if d not in run_ds]
        left_out = [d for d in run_ds if d not in datasets]
        if extra:
            problems.append(f"統整的資料集不在開跑時的清單內：{extra}")
        if left_out:
            info.append(f"本次統整只納入 {len(datasets)} 個資料集（開跑時 {len(run_ds)} 個；未納入：{left_out}）")
    return problems, info


def check_signature(cfg, profile, sig_fn=ens_signature):
    run_dir = Path(cfg.out_root) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    sig_file = run_dir / "signature.json"
    sig = json.loads(json.dumps(sig_fn(cfg), default=str))
    fps = dict(zip(profile["Dataset"], profile["fingerprint"]))
    if sig_file.exists():
        old = json.loads(sig_file.read_text(encoding="utf-8"))
        if old.get("hard") != sig:
            prev = upgradable_from_previous(old.get("hard", {}), sig)
            if prev is None:
                diff = [k for k in sig if json.dumps(old.get("hard", {}).get(k), default=str)
                        != json.dumps(sig[k], default=str)]
                raise SystemExit(f"\n✋ 設定與這個 run（{cfg.run_name}）之前的設定不同：{diff}\n"
                                 f"   → 想保留舊結果：把 run_name 改成新名字。\n"
                                 f"   → 確定不要舊結果：刪掉 {run_dir} 再跑。")
            print(f"ℹ️  這個 run 是上一版（{prev}）開跑的；數值算法相同，改由本版接續（signature.json 已升級）")
            old["hard"] = sig
            old.setdefault("sessions", []).append({"upgraded_from": prev,
                                                   "at": time.strftime("%Y-%m-%d %H:%M:%S")})
        bad = [d for d, fp in fps.items() if d in old["fingerprints"] and old["fingerprints"][d] != fp]
        if bad:
            raise SystemExit(f"\n✋ 這些資料集的檔案內容跟上次不同：{bad}\n   請改 run_name 或刪除 {run_dir}。")
        old["fingerprints"].update(fps)
        old.setdefault("sessions", []).append(core.soft_info(cfg))
        sig_file.write_text(json.dumps(old, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    else:
        sig_file.write_text(json.dumps({"hard": sig, "fingerprints": fps, "sessions": [core.soft_info(cfg)]},
                                       ensure_ascii=False, indent=2, default=str), encoding="utf-8")


# ═════════════════════════ 主流程 ══════════════════════════════════════════
_MAIN_PID = None


def _sigterm_as_interrupt(signum, frame):
    """主程式收到 SIGTERM（例如 `kill <PID>`，nohup 背景執行時的一般停法）→ 比照 Ctrl+C 處理：
    結束所有 worker、釋放鎖，下次重跑從斷掉的地方接續。
    （舊版只有主程式被結束，worker 變成孤兒、一直握著 RUNNING.lock，重跑會被擋下。）"""
    if os.getpid() == _MAIN_PID:
        raise KeyboardInterrupt
    os._exit(128 + signum)                       # 萬一 worker 還沒換掉繼承來的 handler：直接結束


def _worker_init_ens(threads):
    """worker 啟動：SIGTERM 恢復預設（被主程式 terminate() 時立刻結束），其餘與 core 相同。"""
    try:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
    except (ValueError, OSError, AttributeError):
        pass
    core._worker_init(threads)


def run_phase(name, tasks, cfg, run_dir, task_fn, path_fn):
    """通用的階段執行器（D/E/F 與 under-sampling 調參共用）。"""
    todo = [t for t in tasks if not path_fn(cfg, t).exists()]
    print(f"── 階段 {name}：共 {len(tasks)} 格，已完成 {len(tasks) - len(todo)}，本次要跑 {len(todo)}", flush=True)
    if not todo:
        return "done"
    global _MAIN_PID
    _MAIN_PID = os.getpid()
    try:
        old_term = signal.signal(signal.SIGTERM, _sigterm_as_interrupt)
    except (ValueError, OSError, AttributeError):    # 非主執行緒等情形：維持原行為
        old_term = None
    try:
        return _run_phase_inner(name, todo, cfg, run_dir, task_fn)
    finally:
        if old_term is not None:
            signal.signal(signal.SIGTERM, old_term)


def _run_phase_inner(name, todo, cfg, run_dir, task_fn):
    timing_file, err_file = run_dir / "timing.csv", run_dir / "errors.log"
    done = failed = 0
    t_start = time.perf_counter()

    def on_result(res):
        pd.DataFrame([res]).to_csv(timing_file, mode="a", header=not timing_file.exists(),
                                   index=False, encoding="utf-8")

    def on_error(task, e):
        with open(err_file, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {task[:5]}: {e}\n")

    def progress():
        el = time.perf_counter() - t_start
        eta = el / max(done, 1) * (len(todo) - done - failed)
        print(f"  [{name}] 進度 {done + failed}/{len(todo)}（失敗 {failed}） 已用 {el/60:.1f} 分，"
              f"預估剩 {eta/60:.1f} 分", flush=True)

    try:
        if cfg.n_jobs <= 1:
            torch.set_num_threads(max(1, cfg.torch_threads_per_job))
            for t in todo:
                try:
                    on_result(task_fn(t, cfg))
                    done += 1
                except Exception as e:
                    failed += 1
                    on_error(t, f"{type(e).__name__}: {e}")
                if (done + failed) % cfg.progress_every == 0:
                    progress()
        else:
            ex = ProcessPoolExecutor(max_workers=cfg.n_jobs, initializer=_worker_init_ens,
                                     initargs=(cfg.torch_threads_per_job,))
            try:
                futs = {ex.submit(task_fn, t, cfg): t for t in todo}
                for fu in as_completed(futs):
                    try:
                        on_result(fu.result())
                        done += 1
                    except Exception as e:
                        failed += 1
                        on_error(futs[fu], f"{type(e).__name__}: {e}")
                    if (done + failed) % cfg.progress_every == 0:
                        progress()
                ex.shutdown(wait=True)
            except KeyboardInterrupt:
                # 中斷時直接結束 worker，不等它們把手上的格子算完（沒寫完的暫存檔下次開跑會清掉）
                for p in list(getattr(ex, "_processes", {}).values()):
                    try:
                        p.terminate()
                    except Exception:
                        pass
                ex.shutdown(wait=False, cancel_futures=True)
                raise
    except KeyboardInterrupt:
        print("\n⏸  已中斷。已完成的格子都存好了，重新執行同一個指令就會從斷掉的地方接續。")
        return "interrupted"
    if (done + failed) % cfg.progress_every:
        progress()
    if failed:
        print(f"⚠️  {failed} 格失敗，原因見 {err_file}。重跑同一指令只會補跑這些格子。")
    return "done" if not failed else "incomplete"


def check_df_cache(cfg, datasets):
    """開跑前確認 A~C 留下的 AE 特徵快取齊全（路徑規則與 core.get_df 相同）。
    缺檔時 core.get_df 會默默重訓 AE：很花時間，數值也可能與 A~C 不同（要到統整核對才會發現）→ 直接停下來說明。"""
    root = Path(cfg.out_root) / "_df_cache" / dict_hash(core.ae_signature(cfg))
    files = [root / safe_name(ds) / f"f{f}__{ae}__{safe_name(c)}.npz"
             for ds in datasets for f in range(1, N_FOLDS + 1) for ae in cfg.ae_types for c in cfg.configs]
    miss = [p for p in files if not p.exists()]
    print(f"AE 特徵快取：{root}（應有 {len(files)} 個，已有 {len(files) - len(miss)} 個）")
    if not cfg.use_df_cache:
        print("⚠️  use_df_cache=False：每一格都會重新訓練 AE（很慢，且數值可能與 A~C 不同）")
        return
    if not miss:
        return
    by_ds = {}
    for p in miss:
        by_ds[p.parent.name] = by_ds.get(p.parent.name, 0) + 1
    others = (sorted(q.name for q in root.parent.iterdir() if q.is_dir() and q.name != root.name)
              if root.parent.exists() else [])
    msg = (f"\n✋ A~C 留下的 AE 特徵快取不齊：缺 {len(miss)}/{len(files)} 個（依資料集：{by_ds}）\n"
           "   可能原因：① A~C 還沒跑完，或 out_root 與 A~C 不同；② 資料集清單與 A~C 不同；\n"
           "             ③ torch 主次版號、或 occ_tune_core.py 的讀檔／AE 程式與跑 A~C 時不同（快取指紋會不同）。\n"
           f"   _df_cache 底下其他指紋資料夾：{others or '無'}\n"
           "   → 確定要重新訓練缺少的 AE（很花時間，數值也可能與 A~C 不同，統整時的 A~C 核對會抓出來）：\n"
           "     在 runner 的設定加上 allow_ae_retrain=True 再跑。")
    if getattr(cfg, "allow_ae_retrain", False):
        print(msg.replace("✋", "⚠️ ") + "\n   （已設定 allow_ae_retrain=True → 繼續執行，缺少的 AE 會重新訓練）")
        return
    raise SystemExit(msg)


def prepare_run(cfg, title, sig_fn, need_df=True):
    """共用開跑檢查：資料集清單、5 折可讀、AE 快取齊全（need_df）、設定簽章。回傳 (datasets, profile, run_dir)。"""
    run_dir = Path(cfg.out_root) / cfg.run_name
    print("=" * 76)
    print(f"{title}：run={cfg.run_name} worker={cfg.n_jobs} host={socket.gethostname()}")
    datasets, profile = core.discover_datasets(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"資料集：納入 {len(datasets)} 個 → {datasets}")
    core.validate_folds(cfg, datasets)
    if need_df:
        check_df_cache(cfg, datasets)            # 先檢查快取，沒通過就不寫 signature（改好環境後可直接重跑）
    check_signature(cfg, profile, sig_fn)
    # 兩項檢查都通過才更新資料集清單（被擋下的嘗試不會覆蓋掉上一次正確的清單）
    profile.to_csv(run_dir / "dataset_profile.csv", index=False, encoding="utf-8-sig")
    return datasets, profile, run_dir


def run(cfg: EnsConfig):
    bad = set(cfg.studies) - {"D", "E", "F"}
    if bad or not cfg.studies:
        raise SystemExit(f"studies 只能是 D / E / F，收到 {cfg.studies}")
    if "shared" not in cfg.modes:
        raise SystemExit("modes 一定要包含 'shared'（成員分數與原固定參數對照都在這一階段產生）")
    if "F" in cfg.studies and len(cfg.ae_types) < 2:
        raise SystemExit("F 至少需要 2 種 AE")
    datasets, profile, run_dir = prepare_run(cfg, f"D/E/F ensemble 調參 studies={cfg.studies} "
                                                  f"modes={cfg.modes}", ens_signature)
    print(f"每格參數組合 {n_params(cfg)} 組；ensemble 方法 D/E {n_methods(cfg, _de_studies(cfg))} 種、"
          f"F {len(f_methods(cfg)) if 'F' in cfg.studies else 0} 種")
    lock = core.acquire_lock(run_dir)
    try:
        core.cleanup_tmp(cfg)
        n_maj = dict(zip(profile["Dataset"], profile["n_majority_fold1"]))
        big_first = lambda t: -n_maj.get(t[1], 0)
        phases = []
        if _de_studies(cfg):
            phases += [("OF", sorted([("OF", ds, f) for ds in datasets for f in range(1, N_FOLDS + 1)],
                                     key=big_first)),
                       ("DF", sorted([("DF", ds, f, ae, c) for ds in datasets for f in range(1, N_FOLDS + 1)
                                      for ae in cfg.ae_types for c in cfg.configs], key=big_first))]
        if "F" in cfg.studies:
            phases += [("FH", sorted([("FH", ds, f, ALL_AE, c) for ds in datasets
                                      for f in range(1, N_FOLDS + 1) for c in cfg.configs], key=big_first))]
        for name, tasks in phases:
            if run_phase(name, tasks, cfg, run_dir, run_ens_task, cell_path) == "interrupted":
                return "interrupted"
        if "member" in cfg.modes:
            planA, planBC = build_member_plan(cfg, datasets)
            later = []
            if _de_studies(cfg):
                later.append(("MT",) + mt_tasks(cfg, datasets, planA, planBC))
            if "F" in cfg.studies:
                later.append(("FMT",) + fmt_tasks(cfg, datasets, planBC))
            for name, tasks, missing in later:
                if missing:
                    print(f"  （{name}：{missing} 格的成員還沒有 5 折完整結果，這次先跳過）")
                if run_phase(name, sorted(tasks, key=big_first), cfg, run_dir, run_ens_task,
                             cell_path) == "interrupted":
                    return "interrupted"
    finally:
        core.release_lock(lock)
    return "done"


# ═════════════════════════ 統整與 Excel ═══════════════════════════════════
def _cat_dtypes(cfg, datasets, with_param=True):
    pids = sorted({param_id(p) for o in OCC_TYPES for p in param_combos(o, cfg.grid)})
    dt = {"Study": pd.CategoricalDtype(["A", "B", "C", "D", "E", "F"]),
          "Method": pd.CategoricalDtype(all_methods(cfg)),
          "Dataset": pd.CategoricalDtype(list(datasets)),
          # 只有 1 種 AE 時，F 的成員組合名稱會與 AE 名稱相同 → 去重，否則 pandas 會因類別重複而報錯
          "AE": pd.CategoricalDtype(list(dict.fromkeys(list(cfg.ae_types) + ["-"] + f_ae_groups(cfg)))),
          "Config": pd.CategoricalDtype(list(cfg.configs) + ["-"]),
          "OCC": pd.CategoricalDtype(OCC_TYPES), "Fold": "int16"}
    dt["ParamID"] = pd.CategoricalDtype(pids) if with_param else str
    return dt


def _load(cfg, datasets, sub, with_param=True):
    root = Path(cfg.out_root) / cfg.run_name / "cells" / sub
    files = [f for ds in datasets for f in sorted((root / safe_name(ds)).glob("*.csv.gz"))]
    if sub in ("DFm", "FHm"):      # 只收主檔已完成的格子
        files = [f for f in files if (f.parent.parent.parent / sub[:-1] / f.parent.name / f.name).exists()]
    df = _read_cells(files, _cat_dtypes(cfg, datasets, with_param))
    n_unknown = 0
    if len(df):
        df["IsDefault"] = df["IsDefault"].astype(str).str.lower().eq("true")
        key = ["Study", "Method", "Dataset", "AE", "Config", "OCC", "ParamID"]
        n_unknown = int(df[key].isna().any(axis=1).sum())
        if n_unknown:
            print(f"⚠️  {sub}：{n_unknown} 列的名稱不在本次設定內（可能是舊結果或設定被改過），已略過 → 結果標 PARTIAL")
            df = df.dropna(subset=key)
    return df, len(files), n_unknown


def row_level_audit(kind, cfg):
    """逐鍵清點（與 audit_tuning_counts.py 同一套邏輯）：從 signature.json 推出應有的每一個鍵，
    比對缺少／多出／重複，並檢查每列數值有效。回傳 (清點表, 是否全部通過)。"""
    try:
        import audit_tuning_counts as aud
    except ImportError:
        return pd.DataFrame([{"說明": "找不到 audit_tuning_counts.py（請放在同一個資料夾），未做逐鍵清點"}]), False
    fn = {"def": aud.audit_def, "us": aud.audit_us}[kind]
    out = fn(cfg.out_root, cfg.run_name, [])
    t = pd.DataFrame(out[0])
    if out[1]:
        t = pd.concat([t, pd.DataFrame([{"說明": f"cells/ 底下有不在納入清單的資料集資料夾：{out[1]}", "OK": False}])],
                      ignore_index=True)
    return t, bool(len(t)) and bool(t["OK"].fillna(False).all())


def safe_table(name, fn, errors):
    """產生一張分析表；資料不足時回傳說明列，程式錯誤時記進 errors（整份檔案會標 PARTIAL），
    不讓單一分頁中斷整份統整。"""
    try:
        out = fn()
        if isinstance(out, tuple):
            out = out[0]
        if out is None or not isinstance(out, pd.DataFrame) or out.empty or not len(out.columns):
            return pd.DataFrame([{"說明": f"{name}：目前沒有足夠的完整結果（5 折齊全的候選）可以產生這張表"}])
        return out
    except Exception as e:
        errors.append(f"{name} 無法產生（{type(e).__name__}: {e}）")
        return pd.DataFrame([{"說明": f"{name} 無法產生：{type(e).__name__}: {e}"}])


def _fmt(m, s):
    return "" if pd.isna(m) else f"{m:.4f} ± {s:.4f}" if not pd.isna(s) else f"{m:.4f}"


def overall(rows, label, keys=("Study", "Method", "AE", "OCC"), label_col="調參方式"):
    keys = list(keys)
    out = []
    for k, g in rows.groupby(keys, observed=True, sort=True):
        r = dict(zip(keys, k))
        r[label_col] = label
        r["資料集數"], r["fold 數"] = g["Dataset"].nunique(), len(g)
        for m in METRIC_COLS + EXTRA_COLS:
            r[f"{m}_mean"], r[f"{m}_std"] = g[m].mean(), g[m].std(ddof=1)
            r[m] = _fmt(r[f"{m}_mean"], r[f"{m}_std"])
        per_ds = g.drop_duplicates("Dataset")
        r["最常選中參數"] = per_ds["ParamID"].astype(str).mode().iloc[0]
        r["最常選中 Config"] = per_ds["Config"].astype(str).mode().iloc[0]
        out.append(r)
    return pd.DataFrame(out)


def _pretty(ov):
    return ov.drop(columns=[c for c in ov if c.endswith(("_mean", "_std"))])


def per_dataset_table(best, label, G=("Study", "Method", "Dataset", "AE", "OCC")):
    G = list(G)
    g = best.groupby(G, observed=True, sort=True)
    per = g.agg(Config=("Config", "first"), ParamID=("ParamID", "first"),
                AUC平均=("AUC", "mean"), AUC標準差=("AUC", "std"), F1=("F1", "mean"),
                Recall=("Recall", "mean"), Gmean=("G-mean", "mean"), AP=("AP", "mean"),
                有效folds=("Fold", "nunique"), 同分候選數=("同分候選數", "first")).reset_index()
    per.insert(2, "調參方式", label)
    return per.rename(columns={"Gmean": "G-mean"})


def vs_members(ens_best, mem_best, label, cfg):
    """ensemble（已調參）vs 同一資料集、同一 OCC 下「已調參」的各個成員（選法與 A~C 相同）。"""
    e = ens_best.groupby(["Study", "Method", "Dataset", "AE", "OCC"], observed=True)["AUC"].mean().reset_index()
    for c in ["Study", "Method", "Dataset", "AE", "OCC"]:
        e[c] = e[c].astype(str)
    m = mem_best.groupby(["Study", "Dataset", "AE", "OCC"], observed=True)["AUC"].mean()
    look = {tuple(map(str, k)): v for k, v in m.items()}
    best_vals, strs = [], []
    for r in e.itertuples():
        keys = member_keys(r.Study, r.Method, r.AE, cfg)
        vals = [look.get((s, r.Dataset, a, r.OCC), np.nan) for s, a in keys]
        best_vals.append(np.nanmax(vals) if not all(np.isnan(vals)) else np.nan)
        strs.append(vals)
    e["成員最佳"] = best_vals
    e["Δ"] = e["AUC"] - e["成員最佳"]
    e["_vals"] = strs
    out = []
    for k, g in e.groupby(["Study", "Method", "AE", "OCC"], sort=False):
        keys = member_keys(k[0], k[1], k[2], cfg)
        names = [s if s != "B" or k[0] != "F" else f"B_{a}" for s, a in keys]
        mean_vals = np.nanmean(np.array(g["_vals"].tolist(), dtype=float), axis=0)
        out.append({"Study": k[0], "Method": k[1], "調參方式": label, "AE": k[2], "OCC": k[3],
                    "資料集數": len(g), "成員": "+".join(names),
                    "AUC_ensemble": g["AUC"].mean(),
                    "各成員 AUC（已調參）": "  ".join(f"{n}={v:.4f}" for n, v in zip(names, mean_vals)),
                    "AUC_成員最佳（逐資料集取最高）": g["成員最佳"].mean(),
                    "ΔAUC（ensemble−成員最佳）": g["Δ"].mean(),
                    "勝/平/負（vs 成員最佳）": f"{int((g['Δ'] > 1e-9).sum())}/"
                                           f"{int((g['Δ'].abs() <= 1e-9).sum())}/{int((g['Δ'] < -1e-9).sum())}"})
    return pd.DataFrame(out), e.drop(columns=["_vals"])


def _sort_std(df, cfg):
    order = {m: i for i, m in enumerate(all_methods(cfg))}
    occ = {o: i for i, o in enumerate(OCC_TYPES)}
    return df.sort_values(["Method", "OCC", "AE"], key=lambda s: s.astype(str).map(order) if s.name == "Method"
                          else s.astype(str).map(occ) if s.name == "OCC" else s.astype(str)).reset_index(drop=True)


def check_against_abc(cfg, mem, studies=("A", "B", "C")):
    """把本 run 算出來的 A/B/C 成員數值，逐列跟 A~C 調參 run 的結果對照。
    通過條件：① 兩邊的鍵值（Study/Dataset/Fold/AE/Config/OCC/ParamID）都沒有重複；
              ② 本 run 的每一列都在 A~C run 找得到；③ 五個指標（AUC/F1/Recall/G-mean/AP）全部差 < 1e-9。"""
    if not cfg.abc_run_name:
        return pd.DataFrame([{"項目": "未設定 abc_run_name", "結果": "略過核對"}])
    root = Path(cfg.out_root) / cfg.abc_run_name / "cells"
    files = sorted(root.rglob("*.csv"))
    if not files:
        return pd.DataFrame([{"項目": f"找不到 {root}", "結果": "✘ 找不到 A~C 結果，無法核對"}])
    keys = ["Study", "Dataset", "Fold", "AE", "Config", "OCC", "ParamID"]
    mets = ["AUC", "F1", "Recall", "G-mean", "AP"]
    abc = pd.concat((pd.read_csv(f, usecols=keys + mets,
                                 dtype={"ParamID": str, "Config": str, "AE": str, "Dataset": str})
                     for f in files), ignore_index=True)
    mine = mem[keys + mets].copy()
    for c in ["Study", "Dataset", "AE", "Config", "OCC", "ParamID"]:
        mine[c], abc[c] = mine[c].astype(str), abc[c].astype(str)
    mine["Fold"], abc["Fold"] = mine["Fold"].astype(int), abc["Fold"].astype(int)
    dup_abc = int(abc.duplicated(keys).sum())
    abc = abc.drop_duplicates(keys)
    j = mine.merge(abc, on=keys, suffixes=("", "_ABC"))
    rows = []
    for st in studies:
        a, b = mine[mine["Study"] == st], j[j["Study"] == st]
        if not len(a):
            continue
        dup_mine = int(a.duplicated(keys).sum())
        # 兩邊任一個指標不是有限數值（NaN / inf）都算不一致；差值用 skipna=False，NaN 不會被略過
        nonfinite = int((~np.isfinite(b[mets + [f"{m}_ABC" for m in mets]].to_numpy(dtype=float))).any(axis=1).sum()) \
            if len(b) else 0
        r = {"成員": st, "本 run 列數": len(a), "對到 A~C run 的列數": len(b),
             "本 run 重複鍵": dup_mine, "A~C run 重複鍵（全部）": dup_abc, "含非有限值的列": nonfinite}
        for m in mets:
            r[f"max|Δ{m}|"] = float((b[m] - b[f"{m}_ABC"]).abs().max(skipna=False)) if len(b) else np.nan
        ok = (len(a) > 0 and len(b) == len(a) and dup_mine == 0 and dup_abc == 0 and nonfinite == 0
              and all(np.isfinite(r[f"max|Δ{m}|"]) and r[f"max|Δ{m}|"] < 1e-9 for m in mets))
        r["結果"] = "✔ 完全一致（五項指標皆有限且相同、鍵值唯一、無缺列）" if ok else "✘ 不一致或未對齊（請檢查）"
        rows.append(r)
    return pd.DataFrame(rows)


def build_log(cfg, datasets):
    rows = []
    root = Path(cfg.out_root) / cfg.run_name / "cells"
    exp_df = N_FOLDS * len(cfg.ae_types) * len(cfg.configs)
    exp_fh = N_FOLDS * len(cfg.configs)
    for ds in datasets:
        n = {k: len(list((root / k / safe_name(ds)).glob("*.csv.gz"))) for k in ("OF", "DF", "MT", "FH", "FMT")}
        r, ok = {"Dataset": ds}, True
        want = []
        if _de_studies(cfg):
            want += [("OF", N_FOLDS), ("DF", exp_df)] + ([("MT", exp_df)] if "member" in cfg.modes else [])
        if "F" in cfg.studies:
            want += [("FH", exp_fh)] + ([("FMT", exp_fh)] if "member" in cfg.modes else [])
        for k, e in want:
            r[f"{k} 完成/應有"] = f"{n[k]}/{e}"
            ok = ok and n[k] == e
        r["完整"] = ok
        rows.append(r)
    return pd.DataFrame(rows)


def readme_table(cfg, datasets, complete, notes):
    g = cfg.grid
    items = [
        ("結果狀態", "✔ 正式：全部完成、無重複、逐鍵清點通過、成員數值與 A~C run 逐列一致、統整設定與開跑時相同"
         if complete else "⚠ 暫存（PARTIAL）：尚未全部完成或核對未通過（原因見「備註」），只能看進度，不可用於論文"),
        ("這份檔案是什麼", f"Study 1 決策層 ensemble（{'/'.join(cfg.studies)}）的 OCC 調參結果（run = {cfg.run_name}）"),
        ("資料集", f"{len(datasets)} 個：{', '.join(datasets)}"),
        ("D / E 的成員", "A＝OCC on OF｜B＝OCC on DF（DF 再 MinMax）｜C＝OCC on [OF,DF] 串接後 MinMax。"
                       "三個成員用同一種 OCC，B、C 用同一顆 AE（同一 AE 類型、同一 config）。"),
        ("F 的成員", f"B_AE / B_DAE / B_SAE / B_VAE：{len(cfg.ae_types)} 種 AE 在同一個 config 下各自萃取的 DF，"
                    "同一種 OCC。F 比的是「不同 AE」的互補，不是 OF/DF/OF+DF。AE 欄寫成 AE+DAE+SAE+VAE 或 AE+DAE 等成員組合。"),
        ("D 投票方式", "Vote3_hard：各成員用自己的門檻切 0/1，≥2 票判異常，AUC 以票數 0~3 排序｜"
                     "Vote3_soft：各成員分數以訓練多數類 min/max 正規化後等權平均｜Vote3_soft_istd：權重 ∝ 1/std。"),
        ("E 投票方式", "AB / AC / BC × {eq, istd, hard_or ≥1 票, hard_and 2 票}；hard_or 與 hard_and 的 AUC 相同。"),
        ("F 投票方式", "Hetero_Vote：四個 AE 一起 × {eq, istd, hard_or ≥1 票, hard_maj ≥3 票}｜"
                     "Hetero_Pair：兩兩一組 × {eq, istd, hard_or, hard_and}。與舊 F 程式相同。"),
        ("門檻", f"成員：各自訓練多數類分數第 {THRESHOLD_PCT} 百分位；soft：合併後訓練分數的第 {THRESHOLD_PCT} 百分位。"
               "LOF 訓練分數用 fit 時算好的值（與 A~C 調參相同）。"),
        ("調參方式 ①「共用參數」（主要結果）",
         "所有成員用同一組 θ；每個 (資料集, AE, OCC) 在 config × |θ| 組中挑 5 折平均 AUC 最高者。"
         "候選數與 B/C 完全相同（OCSVM 882、LOF 105、iForest 252），與 A~C 比較沒有「多試幾次」的優勢。"),
        ("調參方式 ②「成員各自調參」", "每個成員先在同一個 config 下用自己的 5 折平均 AUC 挑 θ，組成 ensemble 後只挑 config。"
                                  "ParamID 欄寫出每個成員用的 θ。"),
        ("為什麼不做成員參數全組合", "OCSVM 3 個成員會變成 42³≈7.4 萬組（F 四個成員 ≈311 萬組）；用測試折挑選時候選越多越容易"
                                  "挑到運氣好的組合，ensemble 會被不公平地灌水。"),
        ("原固定參數", "所有成員都用 OCSVM nu=0.1/gamma=scale｜LOF k=20｜iForest 100 棵/256，仍挑 AE config（＝舊表做法）。"
                     "數值不會等於舊 D/E/F xlsx：舊程式只在開頭設一次種子，且 LOF 門檻算法不同。"),
        ("選法", "每個 (Study, Method, 資料集, AE, OCC) 挑 5 折平均 AUC 最高者；只考慮 5 折完整的候選；"
               "同分先選舊預設，再依名稱排序。總表＝挑中的 5 折 × 全部資料集的 fold 平均 ± 標準差。"
               "⚠ 挑選與報告用的是同一批測試折 → 屬「最佳可達表現」（oracle），有樂觀偏誤，與 A~C 同一性質；"
               "不能寫成「調參後在未見資料上一定更好」，要主張泛化效果需另做內層驗證（nested CV）。"
               "另外 A 只有 59 組候選、B/C/D/E 每種 AE 有 21×59 組，候選數不同，oracle 偏誤也可能不同。"),
        ("02 分頁怎麼看", "同一資料集、同一 OCC 下，ensemble（已調參）對比各成員（已調參，選法與 A~C 相同）。"
                        "「成員最佳」＝逐資料集取成員中 AUC 最高者（本身也是 oracle）；Δ>0 表示 ensemble 比"
                        "「事後挑最好的單一成員」還好。"),
        ("05 分頁怎麼看", "同一組 θ 套到所有資料集，看參數敏感度與舊預設排第幾。⚠ AE config 仍逐資料集用測試成績挑，"
                        "所以只是「θ 不挑」，不是完全沒有事後挑選。"),
        ("OCC 參數格", f"OCSVM nu ∈ {g['OCSVM']['nu']}、gamma ∈ {g['OCSVM']['gamma']}｜LOF n_neighbors ∈ "
                     f"{g['LOF']['n_neighbors']}｜iForest n_estimators ∈ {g['iForest']['n_estimators']}、"
                     f"max_samples ∈ {g['iForest']['max_samples']}（與 A~C 調參相同，依據見 A~C 的 00_說明）"),
        ("AE", f"{cfg.ae_types}｜{len(cfg.configs)} configs｜epochs={cfg.ae_epochs}"),
        ("09 逐鍵清點", "從 signature.json 推出應有的每一個鍵（資料集×方法×fold×AE×config×OCC×參數），逐鍵比對缺少／多出／重複，"
                          "並檢查數值有效；任何一項不通過，檔名就標 _PARTIAL。與 audit_tuning_counts.py 同一套邏輯。"),
        ("hard 投票的 AUC", "hard 投票以票數當排序分數，只有「成員數＋1」個等級，AUC 排序很粗；同一組成員的 or/and/maj "
                          "AUC 必然相同。比較 hard 投票時建議以 F1 / Recall / G-mean 為主。"),
        ("備註", "；".join(notes) if notes else "無"),
        ("版本", json.dumps(core.soft_info(cfg), ensure_ascii=False)),
    ]
    return pd.DataFrame(items, columns=["項目", "內容"])


def summarize(cfg: EnsConfig):
    run_dir = Path(cfg.out_root) / cfg.run_name
    datasets, _ = core.discover_datasets(cfg)
    notes = []
    # 統整設定必須與開跑時相同（--summarize-only 前改過 runner 會在這裡被抓出來）
    guard_problems, guard_info = summary_guard(cfg, ens_signature, DEF_SUMMARY_KEYS, datasets)
    notes += guard_problems + guard_info
    n_unknown = 0

    def load(sub, with_param=True):
        nonlocal n_unknown
        d, _, nu = _load(cfg, datasets, sub, with_param)
        n_unknown += nu
        return d

    parts = [load(s) for s in ("OF", "DFm", "FHm")]
    mem = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) else parts[0]
    ens_parts = [load(s) for s in ("DF", "FH")]
    ens = pd.concat([p for p in ens_parts if len(p)], ignore_index=True) if any(len(p) for p in ens_parts) \
        else ens_parts[0]
    if ens.empty or mem.empty:
        print("沒有結果可統整。")
        return None
    n_dup_mem = int(mem.duplicated(MEM_KEY).sum())
    b_mismatch = False
    # B 成員在 D/E（DFm）與 F（FHm）各算一次；兩份的五個指標都必須相同（且為有限值），否則代表有東西不一致
    if n_dup_mem:
        mets5 = ["AUC", "F1", "Recall", "G-mean", "AP"]
        dd = mem[mem.duplicated(MEM_KEY, keep=False)]
        g = dd.groupby(MEM_KEY, observed=True)[mets5]
        spread = (g.max() - g.min()).to_numpy(dtype=float)
        finite = np.isfinite(dd[mets5].to_numpy(dtype=float)).all()
        if not finite or (~np.isfinite(spread)).any() or float(np.abs(spread).max()) > 1e-12:
            b_mismatch = True
            notes.append("D/E 與 F 算出的 B 成員數值不一致（五個指標逐列比對）")
        mem = mem.drop_duplicates(MEM_KEY)
    ens = ens[ens["Study"].isin(cfg.studies)]

    G, cand = ["Study", "Method", "Dataset", "AE", "OCC"], ["Config", "ParamID"]
    mem_best, _ = _pick(mem, ["Study", "Dataset", "AE", "OCC"], ["Config", "ParamID"])
    sh_best, _ = _pick(ens, G, cand)
    df_best, _ = _pick(ens[ens["IsDefault"]], G, cand)
    if sh_best.empty or df_best.empty or mem_best.empty:
        # 已經有結果，但還沒有任何候選跑滿 5 折 → 只輸出進度表（與 core 相同的做法）
        log = build_log(cfg, datasets)
        out = run_dir / f"{cfg.run_name}_summary_PARTIAL.xlsx"
        core.write_excel(out, [("00_說明", readme_table(cfg, datasets, False,
                                                         notes + ["目前還沒有任何候選跑滿 5 折，只輸出進度"]), None),
                               ("08_執行紀錄", log, None)])
        print(f"⚠️  目前還沒有任何設定跑滿 5 折，只輸出進度：{out.resolve()}")
        return out
    ov_sh, ov_df = overall(sh_best, MODE_LABEL["shared"]), overall(df_best, MODE_LABEL["default"])
    errs = []
    try:
        vs_sh, vs_sh_ds = vs_members(sh_best, mem_best, MODE_LABEL["shared"], cfg)
        vs, vs_ds = [vs_sh], [vs_sh_ds.assign(調參方式=MODE_LABEL["shared"])]
    except Exception as e:
        errs.append(f"02 ensemble vs 成員（共用參數）無法產生（{type(e).__name__}: {e}）")
        vs, vs_ds = [], []
    per = [per_dataset_table(sh_best, MODE_LABEL["shared"])]
    detail = [sh_best.assign(調參方式=MODE_LABEL["shared"])]

    ov_mt, mt_ok = None, True
    if "member" in cfg.modes:
        mts = [load(s, with_param=False) for s in ("MT", "FMT")]
        mts = [m for m in mts if len(m)]
        mt = pd.concat(mts, ignore_index=True) if mts else pd.DataFrame()
        if len(mt):
            mt = mt[mt["Study"].isin(cfg.studies)]
            memS = mem.assign(**{c: mem[c].astype(str) for c in ["Study", "Dataset", "AE", "Config", "OCC", "ParamID"]})
            planA, planBC = build_member_plan(cfg, datasets, memS)
            exp = []
            for r in mt[["Study", "Method", "Dataset", "AE", "Config", "OCC"]].astype(str).drop_duplicates().itertuples():
                if r.Study == "F":
                    pids = {a: planBC.get(("B", r.Dataset, a, r.Config, r.OCC)) for a in cfg.ae_types}
                    memb = method_members(r.Method, cfg)
                else:
                    k = (r.Dataset, r.AE, r.Config, r.OCC)
                    pids = {"A": planA.get((r.Dataset, r.OCC)), "B": planBC.get(("B",) + k),
                            "C": planBC.get(("C",) + k)}
                    memb = DE_MEMBERS[r.Method]
                exp.append((r.Method, r.Dataset, r.AE, r.Config, r.OCC, _mt_pid(memb, pids)))
            exp = pd.DataFrame(exp, columns=["Method", "Dataset", "AE", "Config", "OCC", "ExpPID"])
            chk = mt[["Method", "Dataset", "AE", "Config", "OCC", "ParamID"]].astype(str).merge(
                exp, on=["Method", "Dataset", "AE", "Config", "OCC"], how="left")
            n_bad = int((chk["ParamID"] != chk["ExpPID"]).sum())
            if n_bad:
                mt_ok = False
                notes.append(f"成員各自調參：{n_bad} 列的成員 θ 與目前結果重算的不一致（請刪掉 cells/MT、cells/FMT 後重跑）")
            mt_best, _ = _pick(mt, G, ["Config"])
            if mt_best.empty:
                mt_ok = False
            else:
                ov_mt = overall(mt_best, MODE_LABEL["member"])
                try:
                    v, vd = vs_members(mt_best, mem_best, MODE_LABEL["member"], cfg)
                    vs.append(v)
                    vs_ds.append(vd.assign(調參方式=MODE_LABEL["member"]))
                except Exception as e:
                    errs.append(f"02 ensemble vs 成員（成員各自調參）無法產生（{type(e).__name__}: {e}）")
                per.append(per_dataset_table(mt_best, MODE_LABEL["member"]))
                detail.append(mt_best.assign(調參方式=MODE_LABEL["member"]))
        else:
            mt_ok = False

    key = ["Study", "Method", "AE", "OCC"]
    gain = ov_sh[key + ["資料集數", "AUC", "AUC_mean", "最常選中參數", "最常選中 Config"]].rename(
        columns={"AUC": "AUC_共用參數", "AUC_mean": "m_sh", "最常選中參數": "最常選中參數（共用）",
                 "最常選中 Config": "最常選中 Config（共用）"})
    if ov_mt is not None:
        gain = gain.merge(ov_mt[key + ["AUC", "AUC_mean"]].rename(
            columns={"AUC": "AUC_成員各自調參", "AUC_mean": "m_mt"}), on=key, how="left")
    gain = gain.merge(ov_df[key + ["AUC", "AUC_mean"]].rename(columns={"AUC": "AUC_原固定參數", "AUC_mean": "m_df"}),
                      on=key, how="left")
    gain["ΔAUC（共用−原固定）"] = gain["m_sh"] - gain["m_df"]
    if ov_mt is not None:
        gain["ΔAUC（成員各自−原固定）"] = gain["m_mt"] - gain["m_df"]
        gain["ΔAUC（成員各自−共用）"] = gain["m_mt"] - gain["m_sh"]
    gain = _sort_std(gain.drop(columns=[c for c in ("m_sh", "m_mt", "m_df") if c in gain]), cfg)
    delta_cols = [c for c in gain if c.startswith("ΔAUC")]

    def fixed_rank():
        fixed, _ = _pick(ens, G + ["ParamID"], ["Config"])
        if fixed.empty:
            return pd.DataFrame()
        fx = (fixed.groupby(["Study", "Method", "AE", "OCC", "ParamID"], observed=True)
                   .agg(AUC_mean=("AUC", "mean"), AUC_std=("AUC", "std"), 資料集數=("Dataset", "nunique"),
                        是否舊預設=("IsDefault", "first")).reset_index())
        fx["排名"] = fx.groupby(["Study", "Method", "AE", "OCC"], observed=True)["AUC_mean"].rank(
            ascending=False, method="min").astype(int)
        fx.insert(0, "說明", "同一組 θ 套到所有資料集；AE config 仍逐資料集挑（屬事後挑選，非完全固定）")
        return fx.sort_values(["Study", "Method", "AE", "OCC", "排名"])

    sheets = {
        "02": safe_table("02 ensemble vs 成員", lambda: _sort_std(pd.concat(vs, ignore_index=True), cfg), errs),
        "02b": safe_table("02b 逐資料集對照", lambda: pd.concat(vs_ds, ignore_index=True), errs),
        "05": safe_table("05 固定參數排名", fixed_rank, errs),
        "06": safe_table("06 成員調參結果",
                         lambda: _pretty(overall(mem_best.assign(Method=mem_best["Study"]), "成員（選法同 A~C）")), errs),
    }
    notes += errs

    chk_abc = check_against_abc(cfg, mem)
    log = build_log(cfg, datasets)
    audit_tab, audit_ok = row_level_audit("def", cfg)
    if not audit_ok:
        notes.append("逐鍵清點未通過（見 09 分頁）")
    dup = int(ens.duplicated(["Study", "Method", "Dataset", "Fold", "AE", "Config", "OCC", "ParamID"]).sum())
    # 沒填 abc_run_name ＝ 沒有核對 → 不能算正式結果
    abc_ok = bool(cfg.abc_run_name) and "結果" in chk_abc and chk_abc["結果"].astype(str).str.startswith("✔").all()
    if not cfg.abc_run_name:
        notes.append("未設定 abc_run_name：成員數值沒有與 A~C run 核對")
    elif not abc_ok:
        notes.append("成員數值與 A~C run 核對未通過（見 07 分頁）")
    if dup:
        notes.append(f"有 {dup} 列重複")
    if n_unknown:
        notes.append(f"有 {n_unknown} 列的名稱不在本次設定內而被略過（設定可能被改過）")
    complete = (bool(log["完整"].all()) and dup == 0 and mt_ok and abc_ok and not errs and audit_ok
                and not b_mismatch and not guard_problems and n_unknown == 0)
    out = run_dir / (f"{cfg.run_name}_summary.xlsx" if complete else f"{cfg.run_name}_summary_PARTIAL.xlsx")

    ov_all = pd.concat([o for o in (ov_sh, ov_mt, ov_df) if o is not None], ignore_index=True)
    det = pd.concat(detail, ignore_index=True)
    keep = ["Study", "Method", "調參方式", "Dataset", "AE", "OCC", "Config", "ParamID", "Fold"] + METRIC_COLS + EXTRA_COLS
    det = det[keep].astype({c: str for c in ["Study", "Method", "Dataset", "AE", "OCC", "Config", "ParamID"]}) \
                   .sort_values(["Study", "Method", "調參方式", "OCC", "AE", "Dataset", "Fold"])
    core.write_excel(out, [
        ("00_說明", readme_table(cfg, datasets, complete, notes), None),
        ("01_調參前後比較", gain, delta_cols),
        ("02_ensemble_vs_成員", sheets["02"], ["ΔAUC（ensemble−成員最佳）"]),
        ("02b_逐資料集對照", sheets["02b"], ["Δ"]),
        ("03_總表", _sort_std(_pretty(ov_all), cfg), None),
        ("04_每資料集最佳", pd.concat(per, ignore_index=True), None),
        ("04b_最佳組合逐折明細", det, None),
        ("05_固定參數排名_共用", sheets["05"], None),
        ("06_成員調參結果", sheets["06"], None),
        ("07_成員與AC結果核對", chk_abc, None),
        ("08_執行紀錄", log, None),
        ("09_逐鍵清點", audit_tab, None),
    ])
    print(f"✅ 統整完成：{out.resolve()}")
    if complete:
        print(f"✔ 驗收通過：{len(datasets)} 個資料集 × {cfg.studies} × {cfg.modes} 全部完成，無重複"
              + ("，成員數值與 A~C run 逐列一致。" if cfg.abc_run_name else "。"))
    else:
        print("⚠️  尚未完成或核對未通過 → 檔名標 _PARTIAL，只能當進度參考。" + "；".join(notes))
    print("\n── 調參前後（AUC，跨資料集平均）──")
    show = ["Study", "Method", "AE", "OCC"] + [c for c in gain if c.startswith("AUC_")]
    print(gain[show].to_string(index=False))
    return out


# ═════════════════════════ 入口 ════════════════════════════════════════════
def main(cfg: EnsConfig, argv=None):
    ap = argparse.ArgumentParser(description="D/E/F ensemble OCC 調參")
    ap.add_argument("--summarize-only", action="store_true", help="不跑實驗，只把已完成的結果統整成 Excel")
    args = ap.parse_args(argv)
    if not args.summarize_only:
        if run(cfg) == "interrupted":
            return
    summarize(cfg)
