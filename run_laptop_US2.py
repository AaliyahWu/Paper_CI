"""
run_laptop_US2.py — S / T / U 兩階段 under-sampling（第二階段 ENN）：筆電測試版（ROG Zephyrus G14）
用 1 個資料集（5 折全跑）確認程式沒問題；正式結果全部在 Azure 產生。

請先跑過 run_laptop.py（A~C）與 run_laptop_US.py（I~O）：
  ・AE 特徵讀 A~C 留下的快取，不重訓 AE
  ・對照組（none / ENN / S1 / S2 / S3）直接讀 run_laptop_US.py 的結果，不重算；
    每一格還會把重算的第一階段遮罩和它存下來的遮罩指紋比對，不同就不存檔
（沒跑過、或筆電上的舊結果是用不同版本的程式／套件跑的，開跑前的檢查會直接停下並說明；
  遇到這種情形不必為了測試把筆電的舊實驗重跑 —— 直接到 Azure 用 run_azure_US2.py 的「第一批 5 個資料集」當測試即可。
  要停止執行請在視窗按 Ctrl+C）

同一個資料夾要有：occ_tune_core.py、occ_tune_ensemble.py、occ_tune_us.py、occ_tune_us2.py

用法：
    python run_laptop_US2.py
    python run_laptop_US2.py --summarize-only
看結果時先看 03_遮罩診斷、03b_兩兩策略等價率：2S_Majority_ENN 和單次 ENN、2S_ENN_ENN 有多少格真的不同。
"""
from pathlib import Path
from occ_tune_us2 import Us2Config, main

BASE = Path(__file__).resolve().parent

CFG = Us2Config(
    data_root=str(BASE / "data"),
    out_root=str(BASE / "results_tuning"),       # 與 run_laptop.py / run_laptop_US.py 相同
    run_name="test_laptop_US2_abalone19",
    us_run_name="test_laptop_US_abalone19",      # run_laptop_US.py 的 run_name（對照與核對用）
    n_jobs=4,
    torch_threads_per_job=1,
    datasets=["abalone19-5-fold"],
    expected_datasets=["abalone19-5-fold"],
    studies=["S", "T", "U"],
)

if __name__ == "__main__":                    # Windows 平行運算一定要有這行
    main(CFG)
