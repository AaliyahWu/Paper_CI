# =============================================================================
# experiments/exp1_no_cluster_homo.py
# Exp 1: 不分群 × 3 OCC (3 種結果)
# 預測: 單分類器 (pool size = 1)
# Undersampling: 永遠 none
# 輸出: exp1_<OCC>.csv
# =============================================================================
import os
import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config as cfg
from common.runner import run_one_combination, get_device


def run_exp1():
    device = get_device()
    print(f"\n{'#'*70}\n# Exp 1: 不分群 × 同質 OCC\n{'#'*70}")

    for occ in cfg.ENABLED_OCC:
        run_one_combination(
            combo_label       = f"exp1 | OCC={occ}",
            output_filename   = f"exp1_{occ}.csv",
            clustering_method = None,
            k                 = None,
            occ_names         = [occ],
            ds_name           = None,
            undersampling     = "none",
            device            = device,
        )


if __name__ == "__main__":
    run_exp1()
