"""
audit_tuning_counts.py — 跑完之後的「清點」工具（不重跑任何實驗，只讀 cells/ 底下的結果檔）

做法：先從每個 run 自己的 signature.json（AE 種類、config、OCC 參數格、studies、modes）與
dataset_profile.csv（納入的資料集）推出「應該要有的每一個鍵」，再和實際結果逐鍵比對：
    缺少的鍵（該跑沒跑）／多出的鍵（不該有的，例如 fold 編號錯、多出的 AE）／重複的鍵
所以整個 fold、整組 OCC、整個方法沒跑，都會被抓出來。

鍵的定義：
    A~C        ：Study × Fold × AE × Config × OCC × ParamID（A 的 AE / Config 為 "-"）
    D/E/F      ：成員列（A/B/C）＋ ensemble 列（Study × Method × Fold × AE × Config × OCC × ParamID）
                  成員各自調參：Study × Method × Fold × AE × Config × OCC 各 1 列
    I~O        ：Space × Strategy × Fold × AE × Config × OCC × ParamID（OF 空間 AE / Config 為 "-"）
另外分開報告：「結果列存在」與「計算有效」不同 ——
    A~C、D/E/F ：每一列 AUC 都必須是有限數值；
    I~O        ：Status = ok / degenerate（清理後 < 5 筆，資料造成的正常結果）/ invalid_fallback（sampler 防呆條件，
                  不應出現：出現就不通過），ok 再分成「實際訓練」與「保留集合相同、沿用結果」（ReuseFrom），
                  ok 列 AUC 必須有限。

用法（放在 occ_tune_*.py 同一個資料夾）：
    python audit_tuning_counts.py --abc full_azure_v2 --def DEF_azure_v2 --us US_azure_v2
    python audit_tuning_counts.py --abc test_laptop_abalone19            # 只查其中幾個也可以
輸出：results_tuning/audit_counts.csv（每個 階段×部分×資料集 一列）、audit_missing_examples.csv（缺漏範例）
"""
import argparse
import glob
import json
from itertools import product, combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from occ_tune_core import OCC_TYPES, N_FOLDS, param_combos, param_id, safe_name

FOLDS = list(range(1, N_FOLDS + 1))
DE_METHODS = {
    "D": ["Vote3_hard", "Vote3_soft", "Vote3_soft_istd"],
    "E": [f"Pair_{p}_{s}" for p in ("AB", "AC", "BC") for s in ("eq", "istd", "hard_or", "hard_and")],
}
F_VOTE = ["Hetero_Vote_eq", "Hetero_Vote_istd", "Hetero_Vote_hard_or", "Hetero_Vote_hard_maj"]
SCHEMES = ("eq", "istd", "hard_or", "hard_and")
US_SPACE_STUDY = {"OF": ("A", "I", "M"), "DF": ("B", "J", "N"), "OFDF": ("C", "K", "O")}
US_SINGLES = ["ENN", "CNN", "TL"]
US_VOTES = ["S1_Union", "S2_Majority", "S3_Intersection", "S4_ENN_TL_abl"]


# ═════════════════════════ 讀 run 的設定 ═══════════════════════════════════
def run_setup(root, run):
    d = Path(root) / run
    sig = json.loads((d / "signature.json").read_text(encoding="utf-8"))["hard"]
    prof = pd.read_csv(d / "dataset_profile.csv")
    datasets = prof.loc[prof["Included"].astype(str).str.lower() == "true", "Dataset"].tolist()
    combos = [(o, param_id(p)) for o in OCC_TYPES for p in param_combos(o, sig["grid"])]
    return d, sig, datasets, combos


def read_cells(files, keys, extra=()):
    cols = set(keys) | set(extra)
    if not files:
        return pd.DataFrame(columns=list(cols))
    df = pd.concat((pd.read_csv(f, usecols=lambda c: c in cols, dtype=str) for f in files), ignore_index=True)
    return df


def compare(stage, part, ds, expected, actual, keys, valid_fn=None, samples=None):
    """逐鍵比對。expected / actual 都是 DataFrame，keys 為比對欄位（全部轉成字串比）。"""
    E = expected[keys].astype(str)
    A = actual[keys].astype(str) if len(actual) else pd.DataFrame(columns=keys)
    dup = int(A.duplicated().sum())
    m = E.drop_duplicates().merge(A.drop_duplicates(), on=keys, how="outer", indicator=True)
    missing, extra = m[m["_merge"] == "left_only"], m[m["_merge"] == "right_only"]
    r = {"階段": stage, "部分": part, "Dataset": ds, "應有列數": len(E), "實際列數": len(A),
         "缺少": len(missing), "多出": len(extra), "重複": dup}
    ok = len(missing) == 0 and len(extra) == 0 and dup == 0
    if valid_fn is not None and len(actual):
        info, vok = valid_fn(actual)
        r.update(info)
        ok = ok and vok
    r["OK"] = bool(ok)
    if samples is not None:
        for kind, t in (("缺少", missing), ("多出", extra)):
            if len(t):
                samples.append(t.drop(columns="_merge").head(5).assign(階段=stage, 部分=part, Dataset=ds, 類型=kind))
    return r


def finite_auc(actual):
    a = pd.to_numeric(actual["AUC"], errors="coerce")
    bad = int((~np.isfinite(a)).sum())
    return {"AUC 非有限值": bad}, bad == 0


def grid(**kw):
    names = list(kw)
    return pd.DataFrame(list(product(*[kw[n] for n in names])), columns=names)


def with_combos(frame, combos):
    c = pd.DataFrame(combos, columns=["OCC", "ParamID"])
    return frame.merge(c, how="cross")


def files_of(d, sub, ds, pattern):
    return sorted(glob.glob(str(d / "cells" / sub / safe_name(ds) / pattern)))


def extra_dataset_dirs(d, subs, datasets):
    names = {safe_name(x) for x in datasets}
    found = set()
    for sub in subs:
        p = d / "cells" / sub
        if p.exists():
            found |= {q.name for q in p.iterdir() if q.is_dir()}
    return sorted(found - names)


# ═════════════════════════ A~C ════════════════════════════════════════════
def audit_abc(root, run, samples):
    d, sig, datasets, combos = run_setup(root, run)
    keys = ["Study", "Fold", "AE", "Config", "OCC", "ParamID"]
    rows = []
    for ds in datasets:
        if "A" in sig["studies"]:
            exp = with_combos(grid(Study=["A"], Fold=FOLDS, AE=["-"], Config=["-"]), combos)
            act = read_cells(files_of(d, "A", ds, "*.csv"), keys, ["AUC"])
            rows.append(compare("A~C", "A", ds, exp, act, keys, finite_auc, samples))
        bc = [s for s in ("B", "C") if s in sig["studies"]]
        if bc:
            exp = with_combos(grid(Study=bc, Fold=FOLDS, AE=sig["ae_types"], Config=sig["configs"]), combos)
            act = read_cells(files_of(d, "DF", ds, "*.csv"), keys, ["AUC"])
            rows.append(compare("A~C", "+".join(bc), ds, exp, act, keys, finite_auc, samples))
    extra = extra_dataset_dirs(d, ["A", "DF"], datasets)
    return rows, extra


# ═════════════════════════ D / E / F ══════════════════════════════════════
def audit_def(root, run, samples):
    d, sig, datasets, combos = run_setup(root, run)
    studies, modes, aes, cfgs = sig["studies"], sig["modes"], sig["ae_types"], sig["configs"]
    de = [s for s in studies if s in ("D", "E")]
    pairs = list(combinations(aes, 2))
    f_meth = [(m, "+".join(aes)) for m in F_VOTE] + \
             [(f"Hetero_Pair_{a}_{b}_{s}", f"{a}+{b}") for a, b in pairs for s in SCHEMES]
    k7 = ["Study", "Method", "Fold", "AE", "Config", "OCC", "ParamID"]
    k6 = ["Study", "Method", "Fold", "AE", "Config", "OCC"]
    rows = []
    for ds in datasets:
        if de:
            exp = with_combos(grid(Study=["A"], Method=["A"], Fold=FOLDS, AE=["-"], Config=["-"]), combos)
            rows.append(compare("D/E", "成員 A（OF 格）", ds, exp,
                                read_cells(files_of(d, "OF", ds, "*.csv.gz"), k7, ["AUC"]), k7, finite_auc, samples))
            exp = with_combos(grid(Study=["B", "C"], Fold=FOLDS, AE=aes, Config=cfgs), combos)
            exp["Method"] = exp["Study"]
            rows.append(compare("D/E", "成員 B、C（DFm）", ds, exp,
                                read_cells(files_of(d, "DFm", ds, "*.csv.gz"), k7, ["AUC"]), k7, finite_auc, samples))
            sm = pd.DataFrame([(s, m) for s in de for m in DE_METHODS[s]], columns=["Study", "Method"])
            base = sm.merge(grid(Fold=FOLDS, AE=aes, Config=cfgs), how="cross")
            rows.append(compare("D/E", "共用參數", ds, with_combos(base, combos),
                                read_cells(files_of(d, "DF", ds, "*.csv.gz"), k7, ["AUC"]), k7, finite_auc, samples))
            if "member" in modes:
                exp = base.merge(pd.DataFrame({"OCC": OCC_TYPES}), how="cross")
                rows.append(compare("D/E", "成員各自調參（MT）", ds, exp,
                                    read_cells(files_of(d, "MT", ds, "*.csv.gz"), k6, ["AUC"]), k6, finite_auc, samples))
        if "F" in studies:
            exp = with_combos(grid(Study=["B"], Method=["B"], Fold=FOLDS, AE=aes, Config=cfgs), combos)
            rows.append(compare("F", "成員 B（FHm）", ds, exp,
                                read_cells(files_of(d, "FHm", ds, "*.csv.gz"), k7, ["AUC"]), k7, finite_auc, samples))
            fm = pd.DataFrame([("F", m, g) for m, g in f_meth], columns=["Study", "Method", "AE"])
            base = fm.merge(grid(Fold=FOLDS, Config=cfgs), how="cross")
            rows.append(compare("F", "共用參數", ds, with_combos(base, combos),
                                read_cells(files_of(d, "FH", ds, "*.csv.gz"), k7, ["AUC"]), k7, finite_auc, samples))
            if "member" in modes:
                exp = base.merge(pd.DataFrame({"OCC": OCC_TYPES}), how="cross")
                rows.append(compare("F", "成員各自調參（FMT）", ds, exp,
                                    read_cells(files_of(d, "FMT", ds, "*.csv.gz"), k6, ["AUC"]), k6, finite_auc, samples))
    extra = extra_dataset_dirs(d, ["OF", "DF", "DFm", "MT", "FH", "FHm", "FMT"], datasets)
    return rows, extra


# ═════════════════════════ I ~ O ══════════════════════════════════════════
def us_strategies(studies, space):
    _, single, vote = US_SPACE_STUDY[space]
    out = ["none"]
    if single in studies or vote in studies:
        out += US_SINGLES if single in studies else ["ENN"]
    if vote in studies:
        out += US_VOTES
    return out


def us_valid(actual):
    st = actual["Status"].astype(str)
    okm = st == "ok"
    reuse = actual.get("ReuseFrom", pd.Series("-", index=actual.index)).astype(str)
    a = pd.to_numeric(actual["AUC"], errors="coerce")
    bad = int((okm & ~np.isfinite(a)).sum())
    unknown = int((~st.isin(["ok", "degenerate", "invalid_fallback"])).sum())
    n_fb = int((st == "invalid_fallback").sum())
    info = {"ok 列": int(okm.sum()), "ok：實際訓練": int((okm & (reuse == "-")).sum()),
            "ok：沿用相同保留集合": int((okm & (reuse != "-")).sum()),
            "degenerate 列": int((st == "degenerate").sum()), "invalid_fallback 列（應為 0）": n_fb,
            "ok 但 AUC 非有限": bad, "未知 Status": unknown}
    # degenerate（清理後 < 5 筆）是資料造成的正常結果；invalid_fallback 只會來自 sampler 防呆條件 → 不應出現
    return info, bad == 0 and unknown == 0 and n_fb == 0


def audit_us(root, run, samples):
    d, sig, datasets, combos = run_setup(root, run)
    studies, aes, cfgs = sig["studies"], sig["ae_types"], sig["configs"]
    spaces = [sp for sp in ("OF", "DF", "OFDF") if any(s in studies for s in US_SPACE_STUDY[sp][1:])]
    keys = ["Space", "Strategy", "Fold", "AE", "Config", "OCC", "ParamID"]
    extra_cols = ["Status", "ReuseFrom", "AUC", "MajKept"]
    rows, eff_parts = [], []
    for ds in datasets:
        if "OF" in spaces:
            exp = with_combos(grid(Space=["OF"], Strategy=us_strategies(studies, "OF"), Fold=FOLDS,
                                   AE=["-"], Config=["-"]), combos)
            act = read_cells(files_of(d, "OF", ds, "*.csv.gz"), keys, extra_cols)
            rows.append(compare("I~O", "OF 空間（A/I/M）", ds, exp, act, keys, us_valid, samples))
            eff_parts.append(act.assign(Dataset=ds))
        ae_sp = [sp for sp in spaces if sp != "OF"]
        if ae_sp:
            exp = pd.concat([grid(Space=[sp], Strategy=us_strategies(studies, sp), Fold=FOLDS, AE=aes, Config=cfgs)
                             for sp in ae_sp], ignore_index=True)
            act = read_cells(files_of(d, "DF", ds, "*.csv.gz"), keys, extra_cols)
            rows.append(compare("I~O", "+".join({"DF": "DF 空間（B/J/N）", "OFDF": "OF+DF 空間（C/K/O）"}[s]
                                               for s in ae_sp), ds, with_combos(exp, combos), act, keys, us_valid, samples))
            eff_parts.append(act.assign(Dataset=ds))
    extra = extra_dataset_dirs(d, ["OF", "DF"], datasets)
    return rows, extra, effective_grid(pd.concat(eff_parts, ignore_index=True) if eff_parts else pd.DataFrame(),
                                       sig["grid"])


def effective_grid(us, g):
    """清理後筆數太少時，LOF / iForest 名義組數中有幾組其實是同一個模型（k→n−1、max_samples→n）。"""
    if us.empty:
        return pd.DataFrame()
    c = us[us["Status"].astype(str) == "ok"].drop_duplicates(["Space", "Strategy", "Dataset", "Fold", "AE", "Config"])
    n = pd.to_numeric(c["MajKept"]).astype(int)
    ks, ms, ne = g["LOF"]["n_neighbors"], g["iForest"]["max_samples"], g["iForest"]["n_estimators"]
    c = c.assign(LOF有效=[len({min(k, v - 1) for k in ks}) for v in n],
                 iForest有效=[len({(e, min(m, v)) for e in ne for m in ms}) for v in n], 保留筆數=n)
    return (c.groupby(["Space", "Strategy"]).agg(有效格子數=("Fold", "size"), 平均保留筆數=("保留筆數", "mean"),
                                                 LOF有效組數_平均=("LOF有效", "mean"), LOF有效組數_最少=("LOF有效", "min"),
                                                 iForest有效組數_平均=("iForest有效", "mean"),
                                                 iForest有效組數_最少=("iForest有效", "min")).reset_index())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results_tuning")
    ap.add_argument("--abc", default="")
    ap.add_argument("--def", dest="def_", default="")
    ap.add_argument("--us", default="")
    a = ap.parse_args()
    rows, samples, extras, eff = [], [], [], pd.DataFrame()
    if a.abc:
        r, e = audit_abc(a.root, a.abc, samples); rows += r; extras += [("A~C", x) for x in e]
    if a.def_:
        r, e = audit_def(a.root, a.def_, samples); rows += r; extras += [("D/E/F", x) for x in e]
    if a.us:
        r, e, eff = audit_us(a.root, a.us, samples); rows += r; extras += [("I~O", x) for x in e]
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 500)
    root = Path(a.root)
    if len(res):
        show = [c for c in res.columns if c not in ("OK",)] + ["OK"]
        print(res[show].to_string(index=False))
        res.to_csv(root / "audit_counts.csv", index=False, encoding="utf-8-sig")
    if samples:
        pd.concat(samples, ignore_index=True).to_csv(root / "audit_missing_examples.csv", index=False,
                                                     encoding="utf-8-sig")
        print(f"\n缺漏／多出的範例：{(root / 'audit_missing_examples.csv').resolve()}")
    if extras:
        print("\n⚠️  cells/ 底下有不在 dataset_profile 納入清單的資料集資料夾：", extras)
    all_ok = bool(len(res)) and bool(res["OK"].all()) and not extras
    print(f"\n{'✔ 全部通過' if all_ok else '✘ 有問題'}：{int(res['OK'].sum()) if len(res) else 0}/{len(res)} 個 階段×部分×資料集 通過"
          + ("" if not extras else "（另有多出的資料集資料夾）"))
    if len(eff):
        print("\n── I~O：名義 LOF 5 組 / iForest 12 組，清理後實際有效的組數（只算 ok 格子）──")
        print(eff.round(2).to_string(index=False))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
