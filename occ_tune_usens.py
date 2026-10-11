"""
occ_tune_usens.py
=================
P / Q / R：I / J / K 的 under-sampling ＋ OCC ensemble（原本 I~K 清理後只接「單一」OCC）
（沿用 occ_tune_core.py 的讀資料、AE 特徵快取、OCC；occ_tune_ensemble.py 的 _prep / _soft / _hard 投票與選法；
  occ_tune_us.py 的 sampler 與遮罩指紋；occ_tune_us2.py 的「與舊 I~O run 銜接」工具。舊的三支程式一行都不改。
  筆電測試用 run_laptop_USENS.py，Azure 全跑用 run_azure_USENS.py）

────────────────────────────────────────────────────────────────────────────
方法（每個特徵空間一個代號：P＝OF、Q＝DF、R＝OF+DF）
────────────────────────────────────────────────────────────────────────────
  特徵 → MinMax（只 fit 未清理的訓練多數類）→ 單一 sampler（ENN；可加 CNN / TL）清理多數類
       → 在「同一份」清理後的多數類上分別訓練 OCSVM、LOF、iForest → 三個 OCC 的輸出做決策層投票 → 測試

  誰在投票：同一個特徵空間、同一顆 AE、同一個 config、同一份清理後資料上的三種 OCC 演算法。
            （D / E 是「同一種 OCC、跨 OF/DF/OF+DF」；F 是「同一種 OCC、跨 AE」；這裡是「跨 OCC 演算法」，三者不同。）
  投票方式（計算規則直接呼叫 occ_tune_ensemble 的 _prep / _soft / _hard，與 D / E 完全相同）：
    Vote3_soft        三個成員各自用訓練分數 min/max 正規化後等權平均        ← 事前指定的主要結果
    Vote3_soft_istd   權重 ∝ 1/std
    Vote3_hard        各成員用自己的門檻切 0/1，≥2 票判異常
    Pair_<a>_<b>_{eq, istd, hard_or, hard_and}   兩兩一組（3 組 × 4 種）：消融對照，例如去掉 iForest
  三個成員用同一份訓練樣本、順序相同，訓練分數可以逐筆對齊，soft 投票的門檻直接取合併後訓練分數的第 90 百分位。

  同一次執行另外算 none（不清理）＋同樣的投票，當作「沒有 US 的三 OCC ensemble」對照，不佔方法代號：
                    單一 OCC          三種 OCC 投票
      無 US         A / B / C         none 列（本程式）
      有 US         I / J / K         P / Q / R（本程式）

────────────────────────────────────────────────────────────────────────────
OCC 參數怎麼給？三種演算法沒有「同一組 θ」可以共用，所以做兩種：
────────────────────────────────────────────────────────────────────────────
  tuned  ：成員各自調參。每個成員用它自己在「同一個空間、同一個 sampler、同一個 AE、同一個 config」下
           5 折平均 AUC 最高的 θ —— 這組 θ 直接從舊的 I~O run（us_run_name）的結果挑出來，不重新搜尋 59 組。
           組成 ensemble 後只再挑 AE config。（不做 42×5×12＝2,520 組的成員全組合：候選越多越容易挑到運氣好的組合。）
  default：三個成員都用舊預設（OCSVM nu=0.1/gamma=scale、LOF k=20、iForest 100 棵/256），仍挑 AE config。

之前跑過的不用重跑：
  ・成員的 59 組搜尋結果讀自舊 I~O run；本程式每格每個策略最多只訓練 6 個 OCC（3 成員 × tuned / default），
    因為做投票需要逐筆的分數，而舊 run 只存了指標。
  ・每一格重算的遮罩會與舊 run 的 MaskSig 比對（不同就不存檔）；每個成員重算出來的五項指標
    （AUC / F1 / Recall / G-mean / AP）在存檔前就與舊 run 同一列比對，差 ≥ 1e-9 這一格就不存檔；統整時再全體核對一次（05 分頁）。
  ・成員參數表（member_plan.csv）每次開跑都重新由舊 run 算出來，和已存的那份逐鍵比對；缺鍵會直接停止。
  ・AE 特徵嚴格讀 A~C 留下的 _df_cache：壞檔或不符就讓該格失敗，不會自動重訓 AE。

可以分批跑：runner 的 datasets 先填 5 個，跑完統整（檔名會標 _5of17ds）；之後把 datasets 換成 17 個、
用同一個 run_name 再執行同一個指令，已完成的資料集不會重跑，只補新的（成員參數表也只補新資料集的部分）。
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
import imblearn

from occ_tune_core import (N_FOLDS, THRESHOLD_PCT, OCC_TYPES, METRIC_COLS, EXTRA_COLS,
                           LEGACY_DEFAULT, param_combos, param_id, is_legacy_default, safe_name)
import occ_tune_ensemble as ens
from occ_tune_ensemble import (_prep, _soft, _hard, _metrics, _pick, run_phase, prepare_run,
                               code_fingerprint, lib_versions, safe_table, summary_guard)
import occ_tune_us as us
from occ_tune_us import SPACES, SPACE_LABEL, MIN_KEEP, TIE_TOL
import occ_tune_us2 as us2

USENS_VERSION = "USENS_tune_v2"
#   v2：每一格的數值算法與 v1 相同；補強驗收與續跑——成員參數表每次開跑都重新由舊 run 算出並核對
#       （缺鍵直接停止，不再默默變成 no_plan）、成員五項指標在存檔前就與舊 run 核對、AE 快取嚴格讀取、
#       統整時重新確認舊 run 齊全、資料集可以分批跑（先 5 個、之後補其餘）。
SPACE_STUDY3 = {"OF": "P", "DF": "Q", "OFDF": "R"}
STUDY_SPACE3 = {v: k for k, v in SPACE_STUDY3.items()}
SAMPLERS = ["ENN", "CNN", "TL"]
MODES = {"tuned": "成員各自調參", "default": "原固定參數"}
PAIRS3 = list(combinations(OCC_TYPES, 2))
PAIR_SCHEMES = ("eq", "istd", "hard_or", "hard_and")
VOTE3 = ["Vote3_soft", "Vote3_soft_istd", "Vote3_hard"]
METHODS = VOTE3 + [f"Pair_{a}_{b}_{s}" for a, b in PAIRS3 for s in PAIR_SCHEMES]
METHOD_MEMBERS = {**{m: tuple(OCC_TYPES) for m in VOTE3},
                  **{f"Pair_{a}_{b}_{s}": (a, b) for a, b in PAIRS3 for s in PAIR_SCHEMES}}
PRIMARY_METHOD = "Vote3_soft"
STATUSES = ["ok", "degenerate", "invalid_fallback", "no_plan"]
USENS_COLS = (["Study", "Space", "Strategy", "Mode", "Kind", "Method", "Dataset", "Fold", "AE", "Config",
               "ParamID", "IsDefault", "MajKept", "RemovedRate", "MaskSig", "Status"]
              + METRIC_COLS + EXTRA_COLS + ["RefAUC", "RefMaxDiff"])
PLAN_COLS = ["Space", "Strategy", "Dataset", "AE", "Config", "OCC", "ParamID", "Note"]
NOTE_NO_5FOLD = "舊 run 這個成員沒有 5 折都有效的結果（例如清理後退化）"
G_ENS = ["Study", "Space", "Strategy", "Mode", "Kind", "Method", "Dataset", "AE"]
PLAN_KEY = ["Space", "Strategy", "Dataset", "AE", "Config", "OCC"]


@dataclass
class UsEnsConfig(us.UsConfig):
    run_name: str = "USENS_tune"
    studies: list = field(default_factory=lambda: ["P", "Q", "R"])
    # 舊的 I~O 調參 run（同一個 out_root 底下）：成員的最佳 θ 與核對都從這裡讀，一定要填。
    us_run_name: str = ""
    # 清理用的單一 sampler（各自是獨立的實驗條件，不是 US ensemble）。none 對照一定會算。
    samplers: list = field(default_factory=lambda: ["ENN"])
    modes: list = field(default_factory=lambda: ["tuned", "default"])
    allow_lib_mismatch: bool = False
    # 舊 run 記錄的 occ_tune_us.py 程式指紋與現在不同、又不是已確認數值相同的上一版時，預設停止
    allow_ref_code_mismatch: bool = False
    # True：成員重算出來的五項指標與舊 run 同一列的差 ≥ 1e-9 就不存檔、寫進 errors.log。
    # False：照樣存檔，但統整時整份標 PARTIAL。
    strict_numeric: bool = True


# ═════════════════════════ 要跑哪些空間 / 策略 ═══════════════════════════
def active_spaces(cfg):
    return [sp for sp in SPACES if SPACE_STUDY3[sp] in cfg.studies]


def strategies(cfg):
    return ["none"] + [s for s in SAMPLERS if s in cfg.samplers]


def active_modes(cfg):
    return [m for m in MODES if m in cfg.modes]


def default_pids(cfg):
    out = {}
    for occ in OCC_TYPES:
        d = [param_id(p) for p in param_combos(occ, cfg.grid) if is_legacy_default(occ, p)]
        if len(d) != 1:
            raise SystemExit(f"參數格裡找不到（或有多個）{occ} 的舊預設參數，無法做 default 模式")
        out[occ] = d[0]
    return out


def n_rows_space(cfg):
    return len(strategies(cfg)) * len(active_modes(cfg)) * (len(OCC_TYPES) + len(METHODS))


# ═════════════════════════ 清理與投票 ═════════════════════════════════════
def single_keep_mask(name, maj_s, min_s, cfg):
    """一個 sampler 的保留遮罩；流程與防呆條件和 occ_tune_us.compute_keep_masks 相同（只是不必三個都跑）。
    回傳 None ＝ 防呆條件不成立（invalid_fallback）。算出來的遮罩之後還會與舊 run 的指紋比對。"""
    n_maj = len(maj_s)
    X = np.vstack([maj_s, min_s])
    y = np.array([0] * n_maj + [1] * len(min_s))
    s = us.make_sampler(name, cfg)
    try:
        s.fit_resample(X, y)
    except Exception as e:
        raise RuntimeError(f"sampler {name} 執行失敗：{type(e).__name__}: {e}") from e
    idx = np.asarray(s.sample_indices_)
    loc = idx[idx < n_maj]
    if int(np.sum(idx >= n_maj)) != len(min_s) or len(loc) != len(np.unique(loc)):
        return None
    keep = np.zeros(n_maj, dtype=bool)
    keep[loc] = True
    return keep


def ensemble_occ(y, prepped):
    """prepped = {"OCSVM": _prep(...), "LOF": ..., "iForest": ...}，三者來自同一份訓練樣本。
    回傳 [(Method, 指標), ...]；投票規則與 D / E 相同（_soft / _hard）。"""
    ms = [prepped[o] for o in OCC_TYPES]
    out = [("Vote3_soft", _soft(y, ms, "eq")), ("Vote3_soft_istd", _soft(y, ms, "istd")),
           ("Vote3_hard", _hard(y, ms, 2))]
    for a, b in PAIRS3:
        pm = [prepped[a], prepped[b]]
        out += [(f"Pair_{a}_{b}_eq", _soft(y, pm, "eq")), (f"Pair_{a}_{b}_istd", _soft(y, pm, "istd")),
                (f"Pair_{a}_{b}_hard_or", _hard(y, pm, 1)), (f"Pair_{a}_{b}_hard_and", _hard(y, pm, 2))]
    return out


def _ens_pid(members, pids):
    return " ".join(f"{o}[{pids[o]}]" for o in members)


def eval_space_ens(space, maj_s, min_s, tst_s, y, cfg, base, plan, ref):
    """一個特徵空間：每個策略（none＋各 sampler）× 每種參數給法 → 3 個成員列 ＋ 15 個 ensemble 列。"""
    n_maj = len(maj_s)
    look, dpid = ens.param_lookup(cfg), default_pids(cfg)
    nan_m = {k: np.nan for k in METRIC_COLS + EXTRA_COLS}
    what = f"{base['Dataset']} fold{base['Fold']}"
    rows = []
    for st in strategies(cfg):
        keep = np.ones(n_maj, dtype=bool) if st == "none" else single_keep_mask(st, maj_s, min_s, cfg)
        r = us2.assert_ref_mask(ref, space, st, keep, what)
        status = "invalid_fallback" if keep is None else "ok"
        n_kept = int(keep.sum()) if keep is not None else np.nan
        if keep is not None and n_kept < MIN_KEEP:
            status = "degenerate"
        if status != r["status"]:
            raise RuntimeError(f"{space}/{st} 的狀態與舊 run 不同（本次 {status}，舊 run {r['status']}）")
        sig = us.mask_sig(keep) if keep is not None else "-"
        rr = (n_maj - n_kept) / n_maj if keep is not None else np.nan
        Xtr = maj_s[keep] if status == "ok" else None
        fits = {}                                          # (OCC, ParamID) → _prep 結果；tuned 與 default 相同時只算一次
        for mode in active_modes(cfg):
            if mode == "tuned":
                miss = [o for o in OCC_TYPES if (space, st, o) not in plan]
                if miss:                                   # 參數表漏了這個鍵：不能當成「正常的沒有結果」存檔
                    raise RuntimeError(f"{what} {space}/{st}：成員參數表沒有 {miss} 的項目（member_plan.csv 不完整）")
                pids = {o: (plan[(space, st, o)] or None) for o in OCC_TYPES}
            else:
                pids = dict(dpid)
            s_mode = status
            if status == "ok" and any(p is None for p in pids.values()):
                s_mode = "no_plan"                         # 只會來自參數表註明的情形：舊 run 這個成員沒有 5 折都有效的結果
            common = {"Study": SPACE_STUDY3[space], "Space": space, "Strategy": st, "Mode": mode, **base,
                      "MajKept": n_kept, "RemovedRate": rr, "MaskSig": sig, "Status": s_mode}
            isdef = {o: (pids[o] is not None and is_legacy_default(o, look[o][pids[o]])) for o in OCC_TYPES}
            prepped = {}
            if s_mode == "ok":
                for o in OCC_TYPES:
                    k = (o, pids[o])
                    if k not in fits:
                        s_tr, s_te, _ = core._fit_score(o, look[o][pids[o]], Xtr, tst_s, cfg.lof_train_score)
                        fits[k] = _prep(np.asarray(s_tr, dtype=np.float64), np.asarray(s_te, dtype=np.float64))
                    prepped[o] = fits[k]
            for o in OCC_TYPES:
                m = _metrics(y, prepped[o]["pred"], prepped[o]["s_te"]) if s_mode == "ok" else nan_m
                old = r["mets"].get((o, pids[o]), {}) if pids[o] is not None else {}
                diff = us2.metrics_diff(m, old) if s_mode == "ok" else np.nan
                if s_mode == "ok" and cfg.strict_numeric and not (diff < us2.NUM_TOL):
                    raise RuntimeError(f"{what} {space}/{st}/{o}[{pids[o]}]：成員重算的五項指標與舊 run 的最大差 {diff:.3g}"
                                       f"（應 < {us2.NUM_TOL:g}）。代表現在的程式或環境算出的 OCC 數值與舊 run 不同，這一格不存檔")
                rows.append({**common, "Kind": "member", "Method": o, "ParamID": pids[o] or "-",
                             "IsDefault": isdef[o], **m, "RefAUC": old.get("AUC", np.nan), "RefMaxDiff": diff})
            ens_m = dict(ensemble_occ(y, prepped)) if s_mode == "ok" else {}
            for meth in METHODS:
                mem = METHOD_MEMBERS[meth]
                rows.append({**common, "Kind": "ensemble", "Method": meth,
                             "ParamID": _ens_pid(mem, {o: pids[o] or "-" for o in mem}),
                             "IsDefault": all(isdef[o] for o in mem), **ens_m.get(meth, nan_m), "RefAUC": np.nan,
                             "RefMaxDiff": np.nan})
    return rows


# ═════════════════════════ 成員的 θ：從舊 I~O run 挑 ═══════════════════════
def plan_path(cfg):
    return Path(cfg.out_root) / cfg.run_name / "member_plan.csv"


def expected_plan_keys(cfg, datasets):
    """本次應該要有的每一個參數表鍵：空間 × 策略 × 資料集 × AE × config × 成員。"""
    rows = []
    for sp in active_spaces(cfg):
        aes, cfgs = (["-"], ["-"]) if sp == "OF" else (list(cfg.ae_types), list(cfg.configs))
        rows += [(sp, st, ds, ae, c, o) for st in strategies(cfg) for ds in datasets for ae in aes for c in cfgs
                 for o in OCC_TYPES]
    return pd.DataFrame(rows, columns=PLAN_KEY)


def compute_member_plan(cfg, datasets):
    """由舊 I~O run 的 cells（只讀、不重跑）算出成員參數表：
    每個成員在每個 (空間, 策略, 資料集, AE, config) 下 5 折平均 AUC 最高的 θ（選法＝_pick，與 A~O 相同）。
    回傳 (參數表, 問題清單)。每一個預期的鍵都會有一列：
      ・ParamID 有值 → 正常；
      ・ParamID 空白且 Note 有說明 → 舊 run 有這個成員的結果，但不足 5 折有效（資料造成，屬正常）；
      ・舊 run 完全沒有這個鍵的任何結果 → 列為問題（開跑時會直接停止）。"""
    root = Path(cfg.out_root) / cfg.us_run_name / "cells"
    sp, sts = active_spaces(cfg), strategies(cfg)
    kinds = [k for k in ("OF", "DF") if (k == "OF" and "OF" in sp) or (k == "DF" and {"DF", "OFDF"} & set(sp))]
    files = us2._cell_files(root, datasets, kinds)
    exp = expected_plan_keys(cfg, datasets)
    if not files:
        return exp.assign(ParamID="", Note="MISSING"), [f"舊 run（{cfg.us_run_name}）沒有這些資料集的任何結果"]
    dt = us2._dtypes(cfg, datasets)
    cols = PLAN_KEY + ["Fold", "ParamID", "IsDefault", "Status", "AUC"]
    parts = []
    for f in files:
        d = pd.read_csv(f, usecols=cols, dtype=dt)
        parts.append(d[d["Strategy"].isin(sts) & d["Space"].isin(sp)])
    d = pd.concat(parts, ignore_index=True).dropna(subset=PLAN_KEY)       # AE / config 不在本次設定內的列會是 NaN
    d["IsDefault"] = d["IsDefault"].astype(str).str.lower().eq("true")
    present = d[PLAN_KEY].drop_duplicates().astype(str).assign(_present=True)
    _, pick = _pick(d[d["Status"] == "ok"], PLAN_KEY, ["ParamID"])
    pick = pick[PLAN_KEY + ["ParamID"]].astype(str) if len(pick) else pd.DataFrame(columns=PLAN_KEY + ["ParamID"])
    plan = exp.merge(pick, on=PLAN_KEY, how="left").merge(present, on=PLAN_KEY, how="left")
    has = plan["ParamID"].notna()
    plan["Note"] = np.where(has, "", np.where(plan["_present"].eq(True), NOTE_NO_5FOLD, "MISSING"))
    plan["ParamID"] = plan["ParamID"].fillna("")
    plan = plan[PLAN_COLS].sort_values(PLAN_KEY).reset_index(drop=True)
    problems = []
    legal = us2.legal_pairs(cfg)
    bad = plan[plan["ParamID"] != ""]
    bad = bad[[(o, pid) not in legal for o, pid in zip(bad["OCC"], bad["ParamID"])]]
    if len(bad):
        problems.append(f"參數表有 {len(bad)} 個參數不在本次的參數格內")
    miss = plan[plan["Note"] == "MISSING"]
    if len(miss):
        ex = miss.iloc[0]
        problems.append(f"舊 run 缺 {len(miss)}/{len(plan)} 個成員的結果（例：{ex.Space}/{ex.Strategy}/{ex.Dataset}/"
                        f"{ex.AE}/{ex.Config}/{ex.OCC}）；舊 run 是否跑完、是否含這些 sampler？")
    if plan.duplicated(PLAN_KEY).any():
        problems.append("參數表有重複鍵")
    return plan, problems


def read_plan_file(cfg):
    path = plan_path(cfg)
    if not path.exists():
        return None
    return pd.read_csv(path, dtype=str, keep_default_na=False)[PLAN_COLS]


def plan_diff(old, fresh):
    """已存的參數表 vs 這次由舊 run 重新算出來的：同一個鍵的 θ 必須相同。回傳不同的列數。"""
    m = old.merge(fresh, on=PLAN_KEY, suffixes=("_old", ""))
    return int((m["ParamID_old"] != m["ParamID"]).sum()), len(m)


def sync_member_plan(cfg, datasets):
    """開跑時呼叫：每次都重新由舊 run 算出本次資料集的參數表，
    ① 有問題（缺鍵、參數不合法）就停止；② 與已存的 member_plan.csv 逐鍵比對，不同就停止
    （代表舊 run 的結果在第一次開跑之後變了）；③ 把新資料集的部分補進檔案（分批跑時用）。"""
    fresh, problems = compute_member_plan(cfg, datasets)
    if problems:
        raise SystemExit("\n✋ 成員參數表無法建立，程式停止：\n   - " + "\n   - ".join(problems))
    old = read_plan_file(cfg)
    if old is not None:
        n_diff, n_common = plan_diff(old, fresh)
        if n_diff:
            raise SystemExit(f"\n✋ member_plan.csv 與這次由 {cfg.us_run_name} 重新算出來的不同（{n_diff}/{n_common} 個鍵）。\n"
                             f"   代表舊 run 的結果在第一次開跑之後變了。請確認 {cfg.us_run_name} 沒有被改動；\n"
                             f"   若確定要用現在的舊 run，請換一個新的 run_name 重新開始。")
        merged = pd.concat([old[~old["Dataset"].isin(datasets)], fresh], ignore_index=True)
        added = sorted(set(fresh["Dataset"]) - set(old["Dataset"]))
        print(f"成員參數表：與已存的 member_plan.csv 逐鍵相同（{n_common:,} 個鍵）"
              + (f"；新增資料集 {added}" if added else ""))
    else:
        merged = fresh
        print(f"成員參數表：由 {cfg.us_run_name} 挑出 {len(fresh):,} 個鍵，存成 member_plan.csv")
    n_none = int((fresh["ParamID"] == "").sum())
    if n_none:
        print(f"  （其中 {n_none} 個鍵在舊 run 不足 5 折有效 → 這些成員的 tuned 結果會標 no_plan，屬正常）")
    path = plan_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    merged.sort_values(PLAN_KEY).to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)
    return fresh


def plan_lookup(plan):
    """{(Dataset, AE, Config): ((Space, Strategy, OCC, ParamID 或 ""), ...)}"""
    out = {}
    for r in plan.itertuples(index=False):
        out.setdefault((r.Dataset, r.AE, r.Config), []).append((r.Space, r.Strategy, r.OCC, r.ParamID))
    return {k: tuple(v) for k, v in out.items()}


def plan_check_at_summary(cfg, datasets, df):
    """統整時：① member_plan.csv 涵蓋本次全部鍵、且與現在由舊 run 重新算出來的相同；
    ② 結果裡每個 tuned 成員實際用的 θ 就是參數表上的；③ no_plan 只出現在參數表註明的地方。回傳問題清單。"""
    if "tuned" not in cfg.modes:
        return []
    old = read_plan_file(cfg)
    if old is None:
        return ["找不到 member_plan.csv"]
    fresh, problems = compute_member_plan(cfg, datasets)
    problems = list(problems)
    n_diff, n_common = plan_diff(old, fresh)
    if n_common != len(fresh):
        problems.append(f"member_plan.csv 只涵蓋本次 {len(fresh)} 個鍵中的 {n_common} 個")
    if n_diff:
        problems.append(f"member_plan.csv 與現在由 {cfg.us_run_name} 重新算出來的不同（{n_diff} 個鍵）")
    mem = df[(df["Kind"] == "member") & (df["Mode"] == "tuned")][
        ["Space", "Strategy", "Dataset", "AE", "Config", "Method", "ParamID", "Status"]].astype(str)
    mem = mem.rename(columns={"Method": "OCC"}).drop_duplicates()
    j = mem.merge(old, on=PLAN_KEY, how="left", suffixes=("", "_plan"))
    n_nokey = int(j["ParamID_plan"].isna().sum())
    wrong = int(((j["Status"] == "ok") & (j["ParamID"] != j["ParamID_plan"])).sum())
    # 同一格只要有一個成員沒有參數，三個成員都會標 no_plan → 以「該 (空間, 策略, 資料集, AE, config) 是否有成員缺參數」判斷
    grp = ["Space", "Strategy", "Dataset", "AE", "Config"]
    lack = old[old["ParamID"] == ""][grp].drop_duplicates().assign(_lack=True)
    np_rows = j[j["Status"] == "no_plan"].merge(lack, on=grp, how="left")
    bad_np = int(np_rows["_lack"].isna().sum())
    if n_nokey:
        problems.append(f"{n_nokey} 個成員列在 member_plan.csv 找不到對應的鍵")
    if wrong:
        problems.append(f"{wrong} 個成員列實際使用的 θ 與 member_plan.csv 不同")
    if bad_np:
        problems.append(f"{bad_np} 個 no_plan 成員列不是參數表註明的情形（非預期）")
    return problems


# ═════════════════════════ 工作格 ══════════════════════════════════════════
def cell_path(cfg, task):
    return us.cell_path(cfg, task)


def _write(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    pd.DataFrame(rows, columns=USENS_COLS).to_csv(tmp, index=False, encoding="utf-8", compression="gzip")
    os.replace(tmp, path)


def _check(rows, expected, what):
    if len(rows) != expected:
        raise RuntimeError(f"{what} 列數不符：{len(rows)} ≠ {expected}")
    bad = [r for r in rows if r["Status"] == "ok" and not np.isfinite(r["AUC"])]
    if bad:
        raise RuntimeError(f"{what} 有 {len(bad)} 列 AUC 無效，整格不存檔")


def run_usens_task(task, cfg):
    """task = (kind, ds, fold, ae, config, plan)；plan = ((Space, Strategy, OCC, ParamID), ...)"""
    torch.set_num_threads(max(1, cfg.torch_threads_per_job))
    t0 = time.perf_counter()
    kind, ds, fold, ae, c, plan_items = task
    fd = core.load_fold(cfg.data_root, ds, fold)
    y = fd["y_tst"]
    ref = us2.load_ref_cell(cfg, task)
    plan = {(sp, st, o): pid for sp, st, o, pid in plan_items}
    base = {"Dataset": ds, "Fold": fold, "AE": ae, "Config": c}
    rows, expected, src = [], 0, "-"
    if kind == "OF":
        rows = eval_space_ens("OF", *us2.space_arrays("OF", fd), y, cfg, base, plan, ref)
        expected = n_rows_space(cfg)
    else:
        DF_maj, DF_tst, DF_min, src = us2.load_df_strict(cfg, ds, fold, ae, c, fd)
        for sp in [s for s in active_spaces(cfg) if s != "OF"]:
            rows += eval_space_ens(sp, *us2.space_arrays(sp, fd, DF_maj, DF_min, DF_tst), y, cfg, base, plan, ref)
            expected += n_rows_space(cfg)
    _check(rows, expected, kind)
    _write(rows, cell_path(cfg, task))
    return {"task": "|".join(map(str, task[:5])), "kind": kind, "dataset": ds, "n_maj": int(fd["n_maj"]),
            "seconds": round(time.perf_counter() - t0, 3), "df_source": src, "host": socket.gethostname()}


def usens_signature(cfg):
    return {"usens_version": USENS_VERSION, "studies": sorted(cfg.studies), "samplers": strategies(cfg),
            "modes": active_modes(cfg), "ae_types": cfg.ae_types, "configs": cfg.configs, "grid": cfg.grid,
            "legacy_default": LEGACY_DEFAULT, "ae": core.ae_signature(cfg),
            "lof_train_score": cfg.lof_train_score, "threshold_pct": THRESHOLD_PCT,
            "cat_encoding": "train_fold_mapping_v1", "metrics": METRIC_COLS + EXTRA_COLS, "n_folds": N_FOLDS,
            "eps": ens.EPS, "ref_run": cfg.us_run_name,
            "sampler": {"enn_k": cfg.enn_k, "cnn_k": cfg.cnn_k, "cnn_seed": cfg.cnn_seed, "strategy": "auto",
                        "min_keep": MIN_KEEP},
            "method_defs": {"members": list(OCC_TYPES), "methods": {k: list(v) for k, v in METHOD_MEMBERS.items()},
                            "primary": PRIMARY_METHOD},
            "plan_rule": "per(space,strategy,dataset,AE,config,OCC): argmax 5-fold mean AUC via _pick, from ref_run",
            "numeric_check": {"members": "5 metrics vs ref_run", "tol": us2.NUM_TOL,
                              "strict": bool(cfg.strict_numeric)},
            "libs": {**lib_versions(), "imblearn": imblearn.__version__},
            "code": code_fingerprint(core._fit_score, core.get_df, _prep, _soft, _hard, _metrics, us.make_sampler,
                                     us.mask_sig, us.get_df3, us2.space_arrays, us2.load_ref_cell,
                                     us2.assert_ref_mask, single_keep_mask, ensemble_occ, _ens_pid, eval_space_ens,
                                     run_usens_task, strategies, active_spaces, active_modes, default_pids,
                                     us2.load_df_strict, us2.metrics_diff, _pick, expected_plan_keys,
                                     compute_member_plan, plan_diff, plan_lookup)}


USENS_SUMMARY_KEYS = ["usens_version", "studies", "samplers", "modes", "ae_types", "configs", "grid",
                      "legacy_default", "lof_train_score", "threshold_pct", "metrics", "n_folds", "eps", "sampler",
                      "method_defs", "ref_run", "plan_rule", "numeric_check"]


def _need(cfg):
    return {s: strategies(cfg) for s in active_spaces(cfg)}


def task_shells(cfg, datasets):
    """每一格的 (kind, ds, fold, ae, config)；參數表在開跑時再接到後面。"""
    sp, out = active_spaces(cfg), []
    if "OF" in sp:
        out += [("OF", ds, f, "-", "-") for ds in datasets for f in range(1, N_FOLDS + 1)]
    if {"DF", "OFDF"} & set(sp):
        out += [("DF", ds, f, ae, c) for ds in datasets for f in range(1, N_FOLDS + 1)
                for ae in cfg.ae_types for c in cfg.configs]
    return out


def run(cfg: UsEnsConfig):
    bad = set(cfg.studies) - set(STUDY_SPACE3)
    if bad or not cfg.studies:
        raise SystemExit(f"studies 只能是 P / Q / R，收到 {cfg.studies}")
    if set(cfg.samplers) - set(SAMPLERS) or not cfg.samplers:
        raise SystemExit(f"samplers 只能是 {SAMPLERS}，收到 {cfg.samplers}")
    if set(cfg.modes) - set(MODES) or not cfg.modes:
        raise SystemExit(f"modes 只能是 {list(MODES)}，收到 {cfg.modes}")
    sp = active_spaces(cfg)
    datasets, profile, run_dir = prepare_run(cfg, f"US＋OCC ensemble studies={cfg.studies}", usens_signature,
                                             need_df=bool({"DF", "OFDF"} & set(sp)))
    n_maj = dict(zip(profile["Dataset"], profile["n_majority_fold1"]))
    big_first = lambda t: -n_maj.get(t[1], 0)
    shells = task_shells(cfg, datasets)
    us2.check_ref_run(cfg, _need(cfg), datasets, profile, shells)
    print("空間：" + "、".join(f"{SPACE_STUDY3[s]}＝{SPACE_LABEL[s]}" for s in sp)
          + f"｜策略：{strategies(cfg)}｜參數給法：{[MODES[m] for m in active_modes(cfg)]}"
          + f"｜成員 {OCC_TYPES} → {len(METHODS)} 種投票")
    lock = core.acquire_lock(run_dir)
    try:
        core.cleanup_tmp(cfg)
        look = plan_lookup(sync_member_plan(cfg, datasets)) if "tuned" in cfg.modes else {}
        tasks = [t + (look.get((t[1], t[3], t[4]), ()),) for t in shells]
        for name in ("OF", "DF"):
            part = sorted([t for t in tasks if t[0] == name], key=big_first)
            if part and run_phase(name, part, cfg, run_dir, run_usens_task, cell_path) == "interrupted":
                return "interrupted"
    finally:
        core.release_lock(lock)
    return "done"


# ═════════════════════════ 統整 ════════════════════════════════════════════
def _dtypes(cfg, datasets):
    return {"Study": pd.CategoricalDtype(["P", "Q", "R"]), "Space": pd.CategoricalDtype(SPACES),
            "Strategy": pd.CategoricalDtype(["none"] + SAMPLERS), "Mode": pd.CategoricalDtype(list(MODES)),
            "Kind": pd.CategoricalDtype(["member", "ensemble"]),
            "Method": pd.CategoricalDtype(list(OCC_TYPES) + METHODS),
            "Dataset": pd.CategoricalDtype(list(datasets)), "AE": pd.CategoricalDtype(list(cfg.ae_types) + ["-"]),
            "Config": pd.CategoricalDtype(list(cfg.configs) + ["-"]), "ParamID": str, "MaskSig": str,
            "Status": pd.CategoricalDtype(STATUSES), "Fold": "int16"}


KEY_ROW = ["Space", "Strategy", "Mode", "Method", "Dataset", "Fold", "AE", "Config"]


def load_rows(cfg, datasets):
    root = Path(cfg.out_root) / cfg.run_name / "cells"
    files = us2._cell_files(root, datasets, ("OF", "DF"))
    if not files:
        return pd.DataFrame(columns=USENS_COLS), 0
    dt = _dtypes(cfg, datasets)
    df = pd.concat((pd.read_csv(f, dtype=dt) for f in files), ignore_index=True)
    df["IsDefault"] = df["IsDefault"].astype(str).str.lower().eq("true")
    key = ["Study", "Space", "Strategy", "Mode", "Kind", "Method", "Dataset", "AE", "Config", "Status"]
    n_unknown = int(df[key].isna().any(axis=1).sum())
    if n_unknown:
        print(f"⚠️  {n_unknown} 列的名稱不在本次設定內（可能是舊結果或設定被改過），已略過 → 結果標 PARTIAL")
        df = df.dropna(subset=key)
    df = df[df["Space"].isin(active_spaces(cfg)) & df["Strategy"].isin(strategies(cfg))
            & df["Mode"].isin(active_modes(cfg))]
    return df.reset_index(drop=True), n_unknown


_SP = {**{s: i for i, s in enumerate(SPACES)}, **{SPACE_LABEL[s]: i for i, s in enumerate(SPACES)}}
_ST = {s: i for i, s in enumerate(["none"] + SAMPLERS)}
_MD = {**{m: i for i, m in enumerate(MODES)}, **{v: i for i, v in enumerate(MODES.values())}}
_ME = {m: i for i, m in enumerate(list(OCC_TYPES) + METHODS)}


def _sorted(df, cols=("Space", "Strategy", "Mode", "調參方式", "AE", "Method")):
    maps = {"Space": _SP, "Strategy": _ST, "Mode": _MD, "調參方式": _MD, "Method": _ME}
    cols = [c for c in cols if c in df.columns]
    return df.sort_values(cols, key=lambda s: s.astype(str).map(maps[s.name]) if s.name in maps
                          else s.astype(str), kind="stable").reset_index(drop=True)


def _ds_auc(best):
    keys = ["Space", "Strategy", "Mode", "AE", "Kind", "Method", "Dataset"]
    d = best.groupby(keys, observed=True)["AUC"].mean().reset_index()
    for c in keys:
        d[c] = d[c].astype(str)
    return d


def _wtl(d):
    return f"{int((d > TIE_TOL).sum())}/{int((np.abs(d) <= TIE_TOL).sum())}/{int((d < -TIE_TOL).sum())}"


def vs_members(best):
    """ensemble（只挑 config）vs 它的成員（同樣只挑 config；tuned 模式下＝I/J/K 或 A/B/C 的調參結果）。
    兩種基準：① 事後最佳固定成員＝成員中「跨資料集平均 AUC 最高」的那一種演算法：所有資料集用同一種，
                 但「哪一種最好」是看完這批結果才知道的（仍屬事後挑選，不是事前指定）；
              ② 逐資料集最佳成員＝每個資料集各挑最高的成員（事後挑選，最嚴格，實務上做不到）。"""
    ds = _ds_auc(best)
    G = ["Space", "Strategy", "Mode", "AE"]
    mem = ds[ds["Kind"] == "member"].pivot_table(index=G + ["Dataset"], columns="Method", values="AUC")
    rows, per_ds = [], []
    for k, g in ds[ds["Kind"] == "ensemble"].groupby(G + ["Method"], sort=False):
        members = list(METHOD_MEMBERS[k[4]])
        e = g.set_index("Dataset")["AUC"]
        try:
            m = mem.xs(k[:4], level=G)[members].reindex(e.index).dropna()
        except KeyError:
            continue
        e = e.loc[m.index]
        if not len(e):
            continue
        fixed = m.mean().idxmax()
        d_fix = (e - m[fixed]).to_numpy(dtype=float)
        d_ora = (e - m.max(axis=1)).to_numpy(dtype=float)
        p, n_nz, how = us.wilcoxon_p(d_fix)
        rows.append({"Space": SPACE_LABEL[k[0]], "Strategy": k[1], "調參方式": MODES[k[2]], "AE": k[3],
                     "Method": k[4], "主要結果": "★" if k[4] == PRIMARY_METHOD else "",
                     "成員": "+".join(members), "資料集數": len(e), "AUC_ensemble": float(e.mean()),
                     "各成員 AUC": "  ".join(f"{o}={m[o].mean():.4f}" for o in members),
                     "事後最佳固定成員": fixed, "AUC_事後最佳固定成員": float(m[fixed].mean()),
                     "ΔAUC（ensemble−事後最佳固定成員）": float(d_fix.mean()), "勝/平/負（vs 事後最佳固定成員）": _wtl(d_fix),
                     "檢定用資料集數（去掉平手）": n_nz, "Wilcoxon_p": p, "檢定方法": how,
                     "AUC_逐資料集最佳成員": float(m.max(axis=1).mean()),
                     "ΔAUC（ensemble−逐資料集最佳成員）": float(d_ora.mean()), "勝/平/負（vs 逐資料集最佳成員）": _wtl(d_ora)})
        per_ds.append(pd.DataFrame({"Space": SPACE_LABEL[k[0]], "Strategy": k[1], "調參方式": MODES[k[2]], "AE": k[3],
                                    "Method": k[4], "Dataset": e.index, "AUC_ensemble": e.values,
                                    **{f"AUC_{o}": m[o].values if o in members else np.nan for o in OCC_TYPES},
                                    "Δ（vs 事後最佳固定成員）": d_fix, "Δ（vs 逐資料集最佳成員）": d_ora}))
    out = pd.DataFrame(rows)
    if len(out):
        out["Holm_p"] = np.nan
        for _, idx in out.groupby(["Space", "Strategy", "調參方式", "AE"]).groups.items():
            out.loc[idx, "Holm_p"] = us.holm(out.loc[idx, "Wilcoxon_p"].values)
    return out, (pd.concat(per_ds, ignore_index=True) if per_ds else pd.DataFrame())


def us_effect(best):
    """同一種投票（或同一個成員）、同一種參數給法、同一顆 AE 下：有清理 − 不清理（none）。逐資料集配對。"""
    ds = _ds_auc(best)
    on = ["Space", "Mode", "AE", "Kind", "Method", "Dataset"]
    r = ds[ds["Strategy"] == "none"][on + ["AUC"]].rename(columns={"AUC": "AUC_ref"})
    m = ds[ds["Strategy"] != "none"].merge(r, on=on)
    rows = []
    for k, g in m.groupby(["Space", "Strategy", "Mode", "AE", "Kind", "Method"], sort=False):
        d = (g["AUC"] - g["AUC_ref"]).to_numpy(dtype=float)
        p, n_nz, how = us.wilcoxon_p(d)
        rows.append({"Space": SPACE_LABEL[k[0]], "Strategy": k[1], "調參方式": MODES[k[2]], "AE": k[3],
                     "類型": "單一 OCC（成員）" if k[4] == "member" else "ensemble", "Method": k[5],
                     "主要結果": "★" if k[5] == PRIMARY_METHOD else "", "共同有效資料集數": len(d),
                     "AUC_有清理": float(g["AUC"].mean()), "AUC_none": float(g["AUC_ref"].mean()),
                     "平均ΔAUC": float(d.mean()), "中位數ΔAUC": float(np.median(d)), "勝/平/負": _wtl(d),
                     "檢定用資料集數（去掉平手）": n_nz, "Wilcoxon_p": p, "檢定方法": how})
    out = pd.DataFrame(rows)
    if len(out):
        out["Holm_p"] = np.nan
        for _, idx in out.groupby(["Space", "Strategy", "調參方式", "AE"]).groups.items():
            out.loc[idx, "Holm_p"] = us.holm(out.loc[idx, "Wilcoxon_p"].values)
    return out


def member_check(df):
    """本程式重算的成員五項指標（AUC / F1 / Recall / G-mean / AP）vs 舊 I~O run 同一列。最大差必須 < 1e-9。
    （每一格存檔前已經核對過一次；這裡是全體再清點一次，strict_numeric=False 時靠這裡把關。）"""
    m = df[(df["Kind"] == "member") & (df["Status"] == "ok")]
    rows = []
    for k, g in m.groupby(["Space", "Strategy", "Mode", "Method"], observed=True, sort=True):
        dmax = pd.to_numeric(g["RefMaxDiff"], errors="coerce").to_numpy(dtype=float)
        nonfinite = int((~np.isfinite(dmax)).sum())
        mx = float(np.nanmax(dmax)) if len(dmax) and nonfinite < len(dmax) else np.nan
        d_auc = (g["AUC"] - g["RefAUC"]).abs().max(skipna=False)
        ok = len(dmax) > 0 and nonfinite == 0 and mx < us2.NUM_TOL
        rows.append({"Space": SPACE_LABEL[str(k[0])], "Strategy": str(k[1]), "調參方式": MODES[str(k[2])],
                     "成員": str(k[3]), "列數": len(dmax), "舊 run 找不到或非有限值的列": nonfinite,
                     "五項指標 max|Δ|": mx, "max|ΔAUC|": float(d_auc) if pd.notna(d_auc) else np.nan,
                     "結果": "✔ 五項指標與舊 run 完全一致" if ok else "✘ 不一致（請檢查套件版本、資料與 AE 快取）"})
    return pd.DataFrame(rows)


def _build_log(cfg, datasets):
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


def key_audit(df, cfg, datasets):
    """逐鍵清點：鍵是完整的乘積（策略×參數給法×方法×fold×AE×config），
    所以「列數相符＋無重複＋名稱都在預期集合內」＝逐鍵完全相符。另檢查 ok 列的 AUC 都是有限值。"""
    sts, mds, meths = strategies(cfg), active_modes(cfg), list(OCC_TYPES) + METHODS
    rows = []
    for sp in active_spaces(cfg):
        aes, cfgs = (["-"], ["-"]) if sp == "OF" else (list(cfg.ae_types), list(cfg.configs))
        exp = N_FOLDS * len(aes) * len(cfgs) * len(sts) * len(mds) * len(meths)
        d_sp = df[df["Space"] == sp]
        for ds in datasets:
            d = d_sp[d_sp["Dataset"] == ds]
            st = d["Status"].astype(str)
            dup = int(d.duplicated(KEY_ROW).sum())
            names_ok = (set(d["Strategy"].astype(str)) <= set(sts) and set(d["Mode"].astype(str)) <= set(mds)
                        and set(d["Method"].astype(str)) <= set(meths) and set(d["AE"].astype(str)) <= set(aes)
                        and set(d["Config"].astype(str)) <= set(cfgs)
                        and set(d["Fold"].astype(int)) <= set(range(1, N_FOLDS + 1)))
            bad = int(((st == "ok") & ~np.isfinite(pd.to_numeric(d["AUC"], errors="coerce"))).sum())
            n_fb = int((st == "invalid_fallback").sum())
            r = {"Dataset": ds, "Space": SPACE_LABEL[sp], "應有列數": exp, "實際列數": len(d), "重複": dup,
                 "名稱都在預期內": names_ok, "ok 列": int((st == "ok").sum()),
                 "degenerate 列": int((st == "degenerate").sum()), "no_plan 列": int((st == "no_plan").sum()),
                 "invalid_fallback 列（應為 0）": n_fb, "ok 但 AUC 非有限": bad}
            r["OK"] = bool(len(d) == exp and dup == 0 and names_ok and bad == 0 and n_fb == 0)
            rows.append(r)
    t = pd.DataFrame(rows)
    return t, bool(len(t)) and bool(t["OK"].all())


def readme_table(cfg, datasets, complete, notes):
    g = cfg.grid
    items = [
        ("結果狀態", "✔ 正式：全部完成、無重複、逐鍵清點通過、成員五項指標與舊 I~O run 逐列一致、成員參數表核對通過、"
                     "統整設定與開跑時相同"
         if complete else "⚠ 暫存（PARTIAL）：尚未全部完成或核對未通過（原因見「備註」），只能看進度，不可用於論文"),
        ("這份檔案是什麼", f"P / Q / R：under-sampling ＋ 三種 OCC 決策層投票的結果（run = {cfg.run_name}；"
                           f"成員參數與核對讀自 {cfg.us_run_name}）"),
        ("方法代號", "P＝OF、Q＝DF、R＝OF+DF（串接）。Strategy＝none 的列是「不清理＋同樣的投票」對照，不是 P/Q/R 本體。"),
        ("資料集", f"{len(datasets)} 個：{', '.join(datasets)}"),
        ("流程", "特徵 → MinMax（只 fit 未清理的訓練多數類）→ 單一 sampler 清理多數類（與 I/J/K 相同）→ "
               "在同一份清理後的多數類上分別訓練 OCSVM / LOF / iForest → 決策層投票 → 測試。"),
        ("誰在投票", "同一個特徵空間、同一顆 AE、同一個 config、同一份清理後資料上的三種 OCC 演算法。"
                   "與 D/E（同一種 OCC、跨 OF/DF/OF+DF）、F（同一種 OCC、跨 AE）是不同的 ensemble；"
                   "D/E/F 不能拿來當本表「無 US」那一格，所以另外算 none 對照。"),
        ("sampler（固定，不調）", f"本次：{[s for s in strategies(cfg) if s != 'none']}。ENN(n_neighbors={cfg.enn_k}, kind_sel=all)｜"
                                f"CNN(n_neighbors={cfg.cnn_k}, random_state={cfg.cnn_seed})｜TomekLinks；只刪多數類。"
                                "各 sampler 是分開比較的實驗條件，不是 US ensemble。只跑 ENN 時，結論限定為「ENN 清理後的 OCC ensemble」。"),
        ("投票方式", "Vote3_soft（★ 事前指定的主要結果）：各成員用訓練分數 min/max 正規化後等權平均｜Vote3_soft_istd：權重 ∝ 1/std｜"
                   "Vote3_hard：各成員用自己的門檻切 0/1，≥2 票判異常｜Pair_<a>_<b>_{eq, istd, hard_or, hard_and}：兩兩一組的消融對照。"
                   "計算規則與 D/E 相同（直接呼叫 occ_tune_ensemble 的 _prep / _soft / _hard）。"
                   "三種 OCC 的原始分數尺度不同，所以一定先正規化再平均。"),
        ("門檻", f"成員：各自訓練分數第 {THRESHOLD_PCT} 百分位；soft：合併後訓練分數的第 {THRESHOLD_PCT} 百分位"
               "（三個成員用同一份訓練樣本，分數逐筆對齊）。LOF 訓練分數用 fit 時算好的值。"),
        ("參數給法 ①「成員各自調參」", f"每個成員用它自己在同一個（空間, sampler, AE, config）下 5 折平均 AUC 最高的 θ，"
                                    f"這組 θ 直接從 {cfg.us_run_name} 的結果挑出（member_plan.csv），沒有重新搜尋；"
                                    "組成 ensemble 後只再挑 AE config（OF 空間沒有東西可挑）。"),
        ("參數給法 ②「原固定參數」", "三個成員都用舊預設：OCSVM nu=0.1/gamma=scale｜LOF k=20｜iForest 100 棵/256，仍挑 AE config。"),
        ("為什麼不做成員參數全組合", "42×5×12＝2,520 組／每個表徵設定；用測試折挑選時候選越多越容易挑到運氣好的組合，"
                                  "ensemble 會被不公平地灌水。"),
        ("選法與定位", "每個 (空間, 策略, 參數給法, 方法, 資料集, AE) 在 21 個 config 中挑 5 折平均 AUC 最高者。"
                     "⚠ 成員的 θ 與 ensemble 的 config 都是用測試折挑的 → 屬「最佳可達表現」（oracle），有樂觀偏誤，"
                     "與 A~O 同一性質；不能寫成「在未見資料上一定更好」。"),
        ("02 分頁怎麼看", "ensemble vs 它的成員。成員列在 tuned 模式下就是 I/J/K（none 時為 A/B/C）調參後的數值。"
                        "兩種基準：事後最佳固定成員（成員中跨資料集平均 AUC 最高的那一種演算法；所有資料集用同一種，"
                        "但哪一種最好是看完這批結果才選的，仍屬事後挑選）；"
                        "逐資料集最佳成員（每個資料集各挑最高者，事後挑選、最嚴格、實務上做不到）。"
                        "Wilcoxon 檢定對的是「事後最佳固定成員」；Holm_p 為同一（空間, 策略, 參數給法, AE）內 15 種投票的校正。"
                        "宣稱有效請以 ★ Vote3_soft 與 Holm_p 為準，其餘投票視為探索。"),
        ("03 分頁怎麼看", "同一種投票（或同一個成員）、同一種參數給法、同一顆 AE 下，有清理 − none 的逐資料集配對："
                        "清理對 ensemble 有沒有幫助、幫助是否與對單一 OCC 的幫助相當。"),
        ("05 分頁（核對）", f"本程式為了取得逐筆分數而重新訓練的成員，其五項指標（AUC / F1 / Recall / G-mean / AP）與 "
                          f"{cfg.us_run_name} 同一列逐列比較，差異必須 < {us2.NUM_TOL:g}；每一格在存檔前就先核對過（不一致不存檔），"
                          "統整時再全體清點一次。清理遮罩在執行時也已與舊 run 的 MaskSig 比對。"
                          "五項指標相同不能單獨證明逐筆分數完全相同，但加上程式、環境、資料與 AE 快取的指紋都相同，已足以支持對齊。"),
        ("成員參數表", "member_plan.csv：每次開跑都由舊 run 重新算出並與已存的逐鍵比對；缺鍵會直接停止。"
                     "統整時再確認結果實際使用的 θ 與參數表一致。no_plan 只會出現在參數表註明"
                     f"「{NOTE_NO_5FOLD}」的成員，其餘情形都會讓整份標 PARTIAL。"),
        ("AE 特徵", "嚴格讀取 A~C 留下的 _df_cache：檔案缺欄位、資料指紋不符、形狀不對或含非有限值就讓該格失敗並記錄，"
                    "不會自動重訓 AE，也不會改寫共享快取（除非設定 allow_ae_retrain=True）。"),
        ("hard 投票的 AUC", "hard 投票以票數當排序分數，只有「成員數＋1」個等級，AUC 很粗；比較 hard 投票時請以 F1 / Recall / G-mean 為主。"),
        ("方法定位", "以已標記的少數類輔助挑選訓練樣本（label-informed data cleaning），再以 one-class 方式訓練 OCC："
                   "少數類只用來決定刪哪些多數類，不進 OCC 訓練；測試折不參與 AE、MinMax、清理。"),
        ("OCC 參數格", f"OCSVM nu ∈ {g['OCSVM']['nu']}、gamma ∈ {g['OCSVM']['gamma']}｜LOF n_neighbors ∈ "
                     f"{g['LOF']['n_neighbors']}｜iForest n_estimators ∈ {g['iForest']['n_estimators']}、"
                     f"max_samples ∈ {g['iForest']['max_samples']}（與 A~O 相同）"),
        ("AE", f"{cfg.ae_types}｜{len(cfg.configs)} configs｜epochs={cfg.ae_epochs}（讀 A~C 的 _df_cache）"),
        ("備註", "；".join(notes) if notes else "無"),
        ("版本", json.dumps({**core.soft_info(cfg), "imblearn": imblearn.__version__}, ensure_ascii=False)),
    ]
    return pd.DataFrame(items, columns=["項目", "內容"])


def summarize(cfg: UsEnsConfig):
    datasets, profile = core.discover_datasets(cfg)
    guard_problems, guard_info = summary_guard(cfg, usens_signature, USENS_SUMMARY_KEYS, datasets)
    # 統整時也重新確認舊 run：設定、資料指紋、每一格結果都還在（--summarize-only 也會做）
    ref_p, ref_i = us2.ref_problems(cfg, _need(cfg), datasets, profile, task_shells(cfg, datasets))
    notes = guard_problems + guard_info + ref_p + ref_i + us2.scope_notes(cfg, datasets)
    df, n_unknown = load_rows(cfg, datasets)
    if df.empty:
        print("沒有結果可統整。")
        return None
    log = _build_log(cfg, datasets)
    ok = df[df["Status"] == "ok"]
    best, _ = _pick(ok, G_ENS, ["Config"])
    if best.empty:
        out = us2.summary_path(cfg, datasets, False)
        core.write_excel(out, [("00_說明", readme_table(cfg, datasets, False,
                                                         notes + ["目前還沒有任何候選跑滿 5 折，只輸出進度"]), None),
                               ("06_執行紀錄", log, None)])
        print(f"⚠️  目前還沒有任何設定跑滿 5 折，只輸出進度：{out.resolve()}")
        return out

    keys = ("Study", "Space", "Strategy", "Mode", "Kind", "Method", "AE")
    ov = ens.overall(best, "-", keys)
    tab = ov[list(keys) + ["資料集數", "fold 數", "AUC", "F1", "Recall", "G-mean", "AP", "最常選中 Config"]].copy()
    for c in keys:
        tab[c] = tab[c].astype(str)
    tab = _sorted(tab)
    tab.insert(5, "主要結果", tab["Method"].map(lambda m: "★" if m == PRIMARY_METHOD else ""))
    tab["Space"] = tab["Space"].map(SPACE_LABEL)
    tab["Mode"] = tab["Mode"].map(MODES)
    tab["Kind"] = tab["Kind"].map({"member": "單一 OCC（成員）", "ensemble": "ensemble"})
    tab = tab.rename(columns={"Mode": "調參方式", "Kind": "類型"})

    errs, vs_cache = [], {}

    def _vs(i):
        if not vs_cache:
            vs_cache["v"] = vs_members(best)
        return vs_cache["v"][i]

    sheets = {
        "02": safe_table("02 ensemble vs 成員", lambda: _sorted(_vs(0)), errs),
        "02b": safe_table("02b 逐資料集對照", lambda: _sorted(_vs(1)), errs),
        "03": safe_table("03 清理對 ensemble 的效果", lambda: _sorted(us_effect(best)), errs),
        "04": safe_table("04 每資料集最佳",
                         lambda: ens.per_dataset_table(best, "-", G=G_ENS).drop(columns=["調參方式"]), errs),
        "04b": safe_table("04b 主要結果逐折明細",
                          lambda: best[best["Method"].isin(list(OCC_TYPES) + [PRIMARY_METHOD])][
                              G_ENS + ["Config", "ParamID", "Fold", "MajKept", "RemovedRate"] + METRIC_COLS + EXTRA_COLS]
                          .astype({c: str for c in G_ENS + ["Config"]}).sort_values(G_ENS + ["Fold"]), errs),
        "05": safe_table("05 成員與舊 run 核對", lambda: member_check(df), errs),
    }
    notes += errs
    chk = sheets["05"]
    chk_ok = "結果" in chk and len(chk) > 0 and chk["結果"].astype(str).str.startswith("✔").all()
    if not chk_ok:
        notes.append("成員 AUC 與舊 I~O run 核對未通過（見 05 分頁）")
    if not bool(log["完整"].all()):
        notes.append("還有格子沒跑完（見 06 分頁）")
    audit_tab, audit_ok = key_audit(df, cfg, datasets)
    if not audit_ok:
        notes.append("逐鍵清點未通過（見 07 分頁）")
    dup = int(df.duplicated(KEY_ROW).sum())
    if dup:
        notes.append(f"有 {dup} 列重複")
    plan_p = [] if ref_p else plan_check_at_summary(cfg, datasets, df)
    notes += ["成員參數表：" + x for x in plan_p]
    n_np = int((df["Status"] == "no_plan").sum())
    if n_np and not plan_p:
        notes.append(f"有 {n_np} 列標為 no_plan：全部對應到參數表註明「{NOTE_NO_5FOLD}」的成員（已核對，屬正常，資訊）")
    if n_unknown:
        notes.append(f"有 {n_unknown} 列的名稱不在本次設定內而被略過（設定可能被改過）")
    complete = (bool(log["完整"].all()) and dup == 0 and not errs and audit_ok and chk_ok and n_unknown == 0
                and not guard_problems and not ref_p and not plan_p)
    out = us2.summary_path(cfg, datasets, complete)
    core.write_excel(out, [
        ("00_說明", readme_table(cfg, datasets, complete, notes), None),
        ("01_總表", tab, None),
        ("02_ensemble_vs_成員", sheets["02"], ["ΔAUC（ensemble−事後最佳固定成員）", "ΔAUC（ensemble−逐資料集最佳成員）"]),
        ("02b_逐資料集對照", sheets["02b"], ["Δ（vs 事後最佳固定成員）", "Δ（vs 逐資料集最佳成員）"]),
        ("03_清理對ensemble的效果", sheets["03"], ["平均ΔAUC", "中位數ΔAUC"]),
        ("04_每資料集最佳", sheets["04"], None),
        ("04b_主要結果逐折明細", sheets["04b"], None),
        ("05_成員與舊run核對", chk, None),
        ("06_執行紀錄", log, None),
        ("07_逐鍵清點", audit_tab, None),
    ])
    print(f"✅ 統整完成：{out.resolve()}")
    if complete:
        print(f"✔ 驗收通過：{len(datasets)} 個資料集 × {cfg.studies} 全部完成，無重複，成員五項指標與 {cfg.us_run_name} 逐列一致。")
        for msg in us2.scope_notes(cfg, datasets):
            print("ℹ️  " + msg)
    else:
        print("⚠️  尚未完成或核對未通過 → 檔名標 _PARTIAL，只能當進度參考。" + "；".join(notes))
    show = tab[tab["Method"].isin(list(OCC_TYPES) + [PRIMARY_METHOD])]
    print("\n── 成員 vs 主要投票（Vote3_soft）AUC ──")
    print(show[["Study", "Space", "Strategy", "調參方式", "Method", "AE", "資料集數", "AUC"]].to_string(index=False))
    return out


def main(cfg: UsEnsConfig, argv=None):
    ap = argparse.ArgumentParser(description="P/Q/R under-sampling ＋ OCC ensemble")
    ap.add_argument("--summarize-only", action="store_true", help="不跑實驗，只把已完成的結果統整成 Excel")
    args = ap.parse_args(argv)
    if not args.summarize_only:
        if run(cfg) == "interrupted":
            return
    summarize(cfg)
