"""
run_laptop.py — 筆電測試版（ROG Zephyrus G14：R7-4800HS 8 核 16 緒、16 GB RAM）

目的：只用 1 個資料集（5 折全跑）確認程式沒問題：
  (1) A、B、C 都能跑完，Excel 產生正常、數字合理
  (2) 中途按 Ctrl+C，再執行一次會接著跑（建議實際試一次）
  這裡的結果只是測試，正式結果全部在 Azure 產生。

用法（雙擊或在命令列執行都可以；路徑以這個檔案所在的資料夾為準）：
    python run_laptop.py                     # 跑 + 統整成 Excel
    python run_laptop.py --summarize-only    # 不跑，只把已完成的結果做成 Excel

資料夾擺法：
    這個資料夾/
      ├─ occ_tune_core.py
      ├─ run_laptop.py
      ├─ run_azure.py
      └─ data/
           ├─ glass0-5-fold/  (glass0-5-1tra.dat ... glass0-5-5tst.dat)
           └─ ...
"""
from pathlib import Path
from occ_tune_core import RunConfig, main

BASE = Path(__file__).resolve().parent          # 不管從哪裡啟動，都用這個資料夾的 data/

CFG = RunConfig(
    data_root=str(BASE / "data"),
    out_root=str(BASE / "results_tuning"),
    run_name="test_laptop",
    n_jobs=4,                          # 同時跑 4 格：筆電 8 核，留一半讓你繼續用電腦；
                                       #   RAM 只剩約 5 GB，不建議再調高
    torch_threads_per_job=1,           # 平行時保持 1
    datasets=["glass0-5-fold"],        # 測試用 1 個小資料集（5 折全跑）
    expected_datasets=["glass0-5-fold"],
    studies=["A", "B", "C"],
)

if __name__ == "__main__":            # Windows 平行運算一定要有這行
    main(CFG)
