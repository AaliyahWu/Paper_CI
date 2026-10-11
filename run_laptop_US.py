"""
run_laptop_US.py — Under-sampling 調參（I/J/K/M/N/O）：筆電測試版（ROG Zephyrus G14）
用 1 個資料集（5 折全跑）確認程式沒問題；正式結果全部在 Azure 產生。
請先跑過 run_laptop.py（A~C），這支會讀它留下的 AE 特徵快取，也會拿它的結果核對 none（06 分頁）。
（沒跑過 A~C 時，開跑前的快取檢查會直接停下並說明；要停止執行請在視窗按 Ctrl+C）

用法：
    python run_laptop_US.py
    python run_laptop_US.py --summarize-only
"""
from pathlib import Path
from occ_tune_us import UsConfig, main

BASE = Path(__file__).resolve().parent

CFG = UsConfig(
    data_root=str(BASE / "data"),
    out_root=str(BASE / "results_tuning"),
    run_name="test_laptop_US_abalone19",
    abc_run_name="test_laptop_abalone19",
    n_jobs=4,
    torch_threads_per_job=1,
    datasets=["abalone19-5-fold"],
    expected_datasets=["abalone19-5-fold"],
    studies=["I", "J", "K", "M", "N", "O"],
)

if __name__ == "__main__":                    # Windows 平行運算一定要有這行
    main(CFG)
