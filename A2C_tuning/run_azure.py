"""
run_azure.py — Azure 正式全跑版（Standard_D8as_v5：8 vCPU、32 GB RAM、無 GPU、Linux）
A / B / C × 15 個資料集 × 5 折

第一次使用：
    pip install numpy pandas scikit-learn openpyxl
    pip install torch --index-url https://download.pytorch.org/whl/cpu    # CPU 版，檔案小很多

啟動（背景執行，SSH 斷線也不會停）：
    nohup python run_azure.py > run_azure.log 2>&1 &
    tail -f run_azure.log            # 看進度；這裡按 Ctrl+C 只是停止「看」，程式照跑

中斷後（VM 重開機、被停止、手動 kill）：直接再執行同一個 nohup 指令，會從斷掉的地方接續。
    ※ 同一個 run 不能同時開兩個程式（作業系統檔案鎖；程式結束或當掉會自動解除，不用手動刪）。

跑完（或跑到一半想看進度）：
    python run_azure.py --summarize-only
    → 全部完成：results_tuning/full_azure_v1/full_azure_v1_summary.xlsx
    → 還沒完成：檔名會是 ..._summary_PARTIAL.xlsx（只能看進度，不能用在論文）
"""
import os
from pathlib import Path
from occ_tune_core import RunConfig, main

BASE = Path(__file__).resolve().parent

# 正式的 15 個資料集：多一個、少一個、任何一折缺檔或讀不進來，程式都會直接停止
DATASETS_15 = [
    "glass-0-1-6_vs_2-5-fold", "glass0-5-fold", "glass1-5-fold", "glass2-5-fold",
    "haberman-5-fold", "segment0-5-fold",
    "vehicle1-5-fold", "vehicle2-5-fold", "vehicle3-5-fold",
    "yeast-0-5-6-7-9_vs_4-5-fold", "yeast-1-2-8-9_vs_7-5-fold", "yeast-1-4-5-8_vs_7-5-fold",
    "yeast-1_vs_7-5-fold", "yeast-2_vs_8-5-fold", "yeast1-5-fold",
]

CFG = RunConfig(
    data_root=str(BASE / "data"),
    out_root=str(BASE / "results_tuning"),
    run_name="full_azure_v1",          # 不要跟筆電測試共用名字：正式結果只來自這台機器
    n_jobs=os.cpu_count() or 8,        # 用滿全部 vCPU（D8as_v5 = 8）
    torch_threads_per_job=1,
    datasets=DATASETS_15,
    expected_datasets=DATASETS_15,
    studies=["A", "B", "C"],
)

if __name__ == "__main__":
    main(CFG)
