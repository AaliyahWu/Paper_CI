"""
run_azure_US.py — Under-sampling 調參：Azure 正式全跑版（Standard_D8as_v5：8 vCPU、32 GB RAM）
Study 2：I（OF）/ J（DF）/ K（OF+DF）單一 sampler ENN / CNN / TL
Study 3 路線一：M（OF）/ N（DF）/ O（OF+DF）ensemble under-sampling S1 / S2 / S3 / S4
× 17 個資料集 × 5 折；none 策略 ＝ A / B / C（同一次執行內的配對基準）

⚠ 請在 A~C 調參（run_azure.py，run_name = full_azure_v2）跑完之後再跑：
   1. out_root 與 A~C 相同 → 直接讀 A~C 留下的 AE 特徵快取（含 DF_min），不重訓 AE
   2. abc_run_name = full_azure_v2 → 統整時逐列核對 none 與 A/B/C 是否一致（06 分頁）
   計算量約為 A~C 的 3~4 倍（每格要跑 8 個策略；相同的清理結果只算一次）。
   時間不夠時可先縮小 studies，例如先跑 ["J", "N"]（DF 空間），之後再用「新的 run_name」補其他空間。

同一個資料夾要有：occ_tune_core.py、occ_tune_ensemble.py、occ_tune_us.py、audit_tuning_counts.py、run_azure.py
   （少了 audit_tuning_counts.py 無法逐鍵清點，統整結果一律標 _PARTIAL）

啟動（背景執行）：
    nohup python run_azure_US.py > run_azure_US.log 2>&1 &
    tail -f run_azure_US.log
開跑前會先檢查 A~C 的 AE 特徵快取是否齊全（有 J/K/N/O 時才需要）；缺檔會直接停下並列出可能原因，不會默默重訓 AE。
要停止：kill <主程式 PID>（PID 寫在 results_tuning/US_azure_v2/RUNNING.lock 裡）→ worker 一起結束、鎖自動釋放。
   不要用 kill -9；若已經用了，再執行 pkill -f run_azure_US.py 把剩下的 worker 結束。
中斷後：再執行同一個 nohup 指令，會從斷掉的地方接續。
sampler 執行出錯的格子不會存檔（原因寫在 results_tuning/US_azure_v2/errors.log），重跑同一指令會再試。
統整：python run_azure_US.py --summarize-only（請用與開跑時相同的設定；改過會被偵測並標 _PARTIAL）
    → 全部完成且核對通過：results_tuning/US_azure_v2/US_azure_v2_summary.xlsx
"""
import os
from pathlib import Path
from occ_tune_us import UsConfig, main
from run_azure import DATASETS_17          # 與 A~C 用同一份名單

BASE = Path(__file__).resolve().parent

CFG = UsConfig(
    data_root=str(BASE / "data"),
    out_root=str(BASE / "results_tuning"),     # 與 A~C 相同 → 共用 AE 特徵快取
    run_name="US_azure_v2",
    abc_run_name="full_azure_v2",
    n_jobs=os.cpu_count() or 8,
    torch_threads_per_job=1,
    datasets=DATASETS_17,
    expected_datasets=DATASETS_17,
    studies=["I", "J", "K", "M", "N", "O"],
    enn_k=3, cnn_k=1, cnn_seed=42,             # 與 I/J/K/N 相同（sampler 不調參）
    exclude_categorical=False,
)

if __name__ == "__main__":
    main(CFG)
