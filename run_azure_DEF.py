"""
run_azure_DEF.py — D / E / F ensemble 調參：Azure 正式全跑版（Standard_D8as_v5：8 vCPU、32 GB RAM）
D（OF/DF/OF+DF 三成員投票）/ E（兩兩配對）/ F（四種 AE 的 DF 投票）× 17 個資料集 × 5 折
（取代上一版的 run_azure_DE.py；上一版如果還沒跑，直接改用這支）

⚠ 請在 A~C 調參（run_azure.py，run_name = full_azure_v2）跑完之後再跑：
   1. out_root 與 A~C 相同 → 直接讀 A~C 留下的 AE 特徵快取（_df_cache），不重訓 AE
   2. abc_run_name = full_azure_v2 → 統整時逐列核對 A/B/C 成員數值（07 分頁；不一致檔名會標 _PARTIAL）
   資料集名單直接 import 自 run_azure.py，保證與 A~C 完全相同。

同一個資料夾要有：occ_tune_core.py、occ_tune_ensemble.py、audit_tuning_counts.py、run_azure.py
   （少了 audit_tuning_counts.py 無法逐鍵清點，統整結果一律標 _PARTIAL）

啟動（背景執行）：
    nohup python run_azure_DEF.py > run_azure_DEF.log 2>&1 &
    tail -f run_azure_DEF.log
開跑前會先檢查 A~C 的 AE 特徵快取是否齊全（17 資料集 × 5 折 × 4 AE × 21 config = 7,140 個）；
   缺檔會直接停下並列出可能原因，不會默默重訓 AE。
要停止：kill <主程式 PID>（PID 寫在 results_tuning/DEF_azure_v2/RUNNING.lock 裡）→ worker 一起結束、鎖自動釋放。
   不要用 kill -9；若已經用了，再執行 pkill -f run_azure_DEF.py 把剩下的 worker 結束。
中斷後：再執行同一個 nohup 指令，會從斷掉的地方接續。
統整：python run_azure_DEF.py --summarize-only（請用與開跑時相同的設定；改過會被偵測並標 _PARTIAL）
    → 全部完成且核對通過：results_tuning/DEF_azure_v2/DEF_azure_v2_summary.xlsx
"""
import os
from pathlib import Path
from occ_tune_ensemble import EnsConfig, main
from run_azure import DATASETS_17          # 與 A~C 用同一份名單

BASE = Path(__file__).resolve().parent

CFG = EnsConfig(
    data_root=str(BASE / "data"),
    out_root=str(BASE / "results_tuning"),     # 與 A~C 相同 → 共用 AE 特徵快取
    run_name="DEF_azure_v2",
    abc_run_name="full_azure_v2",              # A~C 的 run_name（核對成員數值用）
    n_jobs=os.cpu_count() or 8,
    torch_threads_per_job=1,
    datasets=DATASETS_17,
    expected_datasets=DATASETS_17,
    studies=["D", "E", "F"],
    modes=["shared", "member"],                # 共用參數（主要）＋ 成員各自調參（對照）
    exclude_categorical=False,                 # 與 run_azure.py 相同
)

if __name__ == "__main__":
    main(CFG)
