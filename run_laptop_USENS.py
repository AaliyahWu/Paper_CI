"""
run_laptop_USENS.py — P / Q / R under-sampling ＋ OCC ensemble：筆電測試版（ROG Zephyrus G14）
用 1 個資料集（5 折全跑）確認程式沒問題；正式結果全部在 Azure 產生。

請先跑過 run_laptop.py（A~C）與 run_laptop_US.py（I~O）：
  ・AE 特徵讀 A~C 留下的快取，不重訓 AE
  ・三個成員（OCSVM / LOF / iForest）各自的最佳參數，直接從 run_laptop_US.py 的結果挑，不重新搜尋 59 組
  ・重算出來的成員五項指標會和 run_laptop_US.py 的結果逐列核對（存檔前先核對一次，05 分頁再清點一次）
（沒跑過、或筆電上的舊結果是用不同版本的程式／套件跑的，開跑前的檢查會直接停下並說明；
  遇到這種情形不必為了測試把筆電的舊實驗重跑 —— 直接到 Azure 用 run_azure_USENS.py 的「第一批 5 個資料集」當測試即可。
  要停止執行請在視窗按 Ctrl+C）

同一個資料夾要有：occ_tune_core.py、occ_tune_ensemble.py、occ_tune_us.py、occ_tune_us2.py、occ_tune_usens.py

用法：
    python run_laptop_USENS.py
    python run_laptop_USENS.py --summarize-only
"""
from pathlib import Path
from occ_tune_usens import UsEnsConfig, main

BASE = Path(__file__).resolve().parent

CFG = UsEnsConfig(
    data_root=str(BASE / "data"),
    out_root=str(BASE / "results_tuning"),       # 與 run_laptop.py / run_laptop_US.py 相同
    run_name="test_laptop_USENS_abalone19",
    us_run_name="test_laptop_US_abalone19",      # run_laptop_US.py 的 run_name
    n_jobs=4,
    torch_threads_per_job=1,
    datasets=["abalone19-5-fold"],
    expected_datasets=["abalone19-5-fold"],
    studies=["P", "Q", "R"],
    samplers=["ENN", "CNN", "TL"],               # 與 I/J/K 對應；none（不清理）對照一定會算
    modes=["tuned", "default"],                  # 成員各自調參 ＋ 原固定參數
)

if __name__ == "__main__":                    # Windows 平行運算一定要有這行
    main(CFG)
