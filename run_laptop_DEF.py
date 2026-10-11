"""
run_laptop_DEF.py — D / E / F ensemble 調參：筆電測試版（ROG Zephyrus G14）
用 1 個資料集（5 折全跑）確認程式沒問題；正式結果全部在 Azure 產生。
請先跑過 run_laptop.py（A~C），這支會讀它留下的 AE 特徵快取，也會拿它的結果來核對（07 分頁）。
（沒跑過 A~C 時，開跑前的快取檢查會直接停下並說明；要停止執行請在視窗按 Ctrl+C）

用法：
    python run_laptop_DEF.py                     # 跑 + 統整成 Excel
    python run_laptop_DEF.py --summarize-only    # 不跑，只統整已完成的結果
"""
from pathlib import Path
from occ_tune_ensemble import EnsConfig, main

BASE = Path(__file__).resolve().parent

CFG = EnsConfig(
    data_root=str(BASE / "data"),
    out_root=str(BASE / "results_tuning"),     # 與 run_laptop.py 相同 → 共用 AE 特徵快取
    run_name="test_laptop_DEF_abalone19",
    abc_run_name="test_laptop_abalone19",      # run_laptop.py 的 run_name（核對用）
    n_jobs=4,
    torch_threads_per_job=1,
    datasets=["abalone19-5-fold"],
    expected_datasets=["abalone19-5-fold"],
    studies=["D", "E", "F"],
    modes=["shared", "member"],
)

if __name__ == "__main__":                    # Windows 平行運算一定要有這行
    main(CFG)
