import json, glob, os, collections
import numpy as np
import occ_tune_core as core
from run_azure_DEF import CFG

now = json.loads(json.dumps(core.ae_signature(CFG), default=str))
print("目前程式的快取指紋:", core.dict_hash(core.ae_signature(CFG)),
      "| torch", now["torch_major_minor"], "| code", now["code"])

for d in sorted(glob.glob("results_tuning/_df_cache/*/")):
    files = glob.glob(d + "*/*.npz")
    cnt = collections.Counter(os.path.basename(os.path.dirname(f)) for f in files)
    print(f"\n== {d}  共 {len(files)} 個")
    if files:
        meta = json.loads(str(np.load(files[0])["meta"]))["ae_signature"]
        print("   與目前不同的項目:", {k: (meta.get(k), now[k]) for k in now if meta.get(k) != now[k]})
    for k, v in sorted(cnt.items()):
        print(f"   {k}: {v}")

print("\n== 各 run 的統整檔")
for r in sorted(glob.glob("results_tuning/*/")):
    if "_df_cache" in r:
        continue
    print(r, [os.path.basename(x) for x in glob.glob(r + "*summary*.xlsx")])