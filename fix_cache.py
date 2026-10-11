import sys, os, re, glob, shutil
import numpy as np, torch
import occ_tune_core as core
from occ_tune_core import safe_name, dict_hash, BOTTLENECK_RATIOS
from run_azure_DEF import CFG

torch.set_num_threads(1)
ROOT = os.path.join(CFG.out_root, "_df_cache")
OLD = os.path.join(ROOT, "33bc56a061f8")
NEW = os.path.join(ROOT, dict_hash(core.ae_signature(CFG)))
path = lambda d, ds, f, ae, c: os.path.join(d, safe_name(ds), f"f{f}__{ae}__{safe_name(c)}.npz")
print("搬到：", NEW)

def retrain(ds, fold, ae, c):
    fd = core.load_fold(CFG.data_root, ds, fold)
    nl = int(re.match(r"^h(\d+)-", c).group(1))
    nu = max(2, round(fd["X_maj_s"].shape[1] * BOTTLENECK_RATIOS[c.split("-", 1)[1]]))
    out = core.train_ae_extract(ae, fd["X_maj_s"], fd["X_tst_s"], fd["X_min_s"], nl, nu,
                                CFG.ae_epochs, core.cell_seed(ds, fold, ae, c))
    return [a.astype(np.float32) for a in out]

def same(d, ds, fold, ae, c):
    z = np.load(path(d, ds, fold, ae, c))
    return all(np.array_equal(z[k], v) for k, v in zip(["DF_maj", "DF_tst", "DF_min"], retrain(ds, fold, ae, c)))

# 只看 v2 名單裡、而且舊指紋資料夾裡有的資料集（yeast-2_vs_8 不在 v2，略過）
need = [ds for ds in CFG.datasets if os.path.isdir(os.path.join(OLD, safe_name(ds)))]
samples = [(OLD, ds, 1, "AE", "h1-1/4") for ds in need] + [(OLD, ds, 3, "VAE", "h3-4/1") for ds in need]
ab = next(((NEW, "abalone19-5-fold", f, ae, c) for f in range(1, 6) for ae in CFG.ae_types for c in CFG.configs
           if os.path.exists(path(NEW, "abalone19-5-fold", f, ae, c))), None)
if ab:
    samples.append(ab)
bad = []
for d, ds, fold, ae, c in samples:
    ok = same(d, ds, fold, ae, c)
    print(f"{'✔' if ok else '✘'} {ds} f{fold} {ae} {c}")
    if not ok:
        bad.append(ds)
print(f"\n驗證：{len(samples) - len(bad)}/{len(samples)} 格與現在的 core 逐位元組相同（v2 舊資料集 {len(need)} 個）")

if "--copy" in sys.argv:
    if bad:
        sys.exit("✘ 有不一致，不搬。請把輸出貼給 Claude。")
    n = 0
    for ds in need:
        os.makedirs(os.path.join(NEW, safe_name(ds)), exist_ok=True)
        for src in glob.glob(os.path.join(OLD, safe_name(ds), "*.npz")):
            dst = os.path.join(NEW, safe_name(ds), os.path.basename(src))
            if not os.path.exists(dst):
                try: os.link(src, dst)
                except OSError: shutil.copy2(src, dst)
                n += 1
    print(f"✔ 已放入 {n} 個快取（舊資料夾原封不動）")