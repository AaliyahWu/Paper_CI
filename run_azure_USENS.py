"""
run_azure_USENS.py — P / Q / R under-sampling ＋ OCC ensemble：Azure 正式版（Standard_D8as_v5）
P（OF）/ Q（DF）/ R（OF+DF）× 資料集 × 5 折
每個空間：none（對照）＋ ENN / CNN / TL 清理後，OCSVM、LOF、iForest 三個 OCC 做決策層投票
（Vote3_soft 為事前指定的主要結果；另有 Vote3_soft_istd、Vote3_hard 與兩兩配對 12 種當對照）

════════════════════════════════════════════════════════════════════════════
分批跑：先跑 5 個資料集，其餘之後再補（已跑完的不會重跑）
════════════════════════════════════════════════════════════════════════════
  要跑哪些資料集「不在這個檔案改」：這支直接 import run_azure_US2.py 的 DATASETS_NOW，
  所以只要改 run_azure_US2.py 裡那兩行註解，兩支程式的資料集就會一起換、保證一致。
  第一批（5 個）：直接執行 → results_tuning/USENS_azure_v2/USENS_azure_v2_summary_5of17ds.xlsx
  第二批（17 個）：在 run_azure_US2.py 把 DATASETS_NOW 換成 DATASETS_17 後，「run_name 不要改」，再執行同一個指令
                   → 只補剩下 12 個資料集；成員參數表（member_plan.csv）也只補新資料集的部分，
                     前 5 個資料集的參數會和第一批逐鍵核對，必須完全相同。
  ⚠ 只有 5 個資料集時 Wilcoxon 最小可能的 p 是 0.0625，第一批請只看 ΔAUC 與勝/平/負的方向。

════════════════════════════════════════════════════════════════════════════
需要 A~C（full_azure_v2）與 I~O（US_azure_v2）都已經跑完，而且用同一個 out_root、同一個 Python 環境
════════════════════════════════════════════════════════════════════════════
  1. AE 特徵讀 A~C 留下的 _df_cache（嚴格讀取，不會自動重訓 AE）
  2. 成員各自的最佳 OCC 參數直接從 US_azure_v2 的結果挑（member_plan.csv），不重新搜尋 59 組；
     每格每個策略最多只訓練 6 個 OCC（做投票需要逐筆分數，US_azure_v2 只存了指標）
  3. 每一格的清理遮罩會和 US_azure_v2 的遮罩指紋比對；重新訓練的成員五項指標（AUC / F1 / Recall / G-mean / AP）
     在存檔前就和 US_azure_v2 同一列比對。任何一項不同，這一格不存檔（原因寫在 errors.log）

同一個資料夾要有：occ_tune_core.py、occ_tune_ensemble.py、occ_tune_us.py、occ_tune_us2.py、occ_tune_usens.py、
                  run_azure.py、run_azure_US2.py

啟動（背景執行）：
    nohup python run_azure_USENS.py > run_azure_USENS.log 2>&1 &
    tail -f run_azure_USENS.log
   （開跑時要先讀 US_azure_v2 的結果來挑成員參數，會停在「舊 run 檢查通過」之後一下子，屬正常）
要停止：kill <主程式 PID>（PID 寫在 results_tuning/USENS_azure_v2/RUNNING.lock 裡）；不要用 kill -9。
中斷後：再執行同一個 nohup 指令，會從斷掉的地方接續。
可以和 run_azure_US2.py 同時跑嗎？可以（兩個 run 各自上鎖、都只讀 US_azure_v2），但會互搶 CPU；
   同時跑的話兩支的 n_jobs 各設 4，或一支跑完再跑另一支。
統整：python run_azure_USENS.py --summarize-only
"""
import os
from pathlib import Path
from occ_tune_usens import UsEnsConfig, main
from run_azure_US2 import DATASETS_NOW      # 這次要跑的資料集：在 run_azure_US2.py 改（兩支程式一起換）

BASE = Path(__file__).resolve().parent

CFG = UsEnsConfig(
    data_root=str(BASE / "data"),
    out_root=str(BASE / "results_tuning"),     # 與 A~C、I~O 相同
    run_name="USENS_azure_v2",                 # 兩批都用同一個名字，第二批才會接著補
    us_run_name="US_azure_v2",                 # 舊的 I~O run（成員參數與核對用）
    n_jobs=os.cpu_count() or 8,                # 和 run_azure_US2.py 同時跑時，兩支各改成 4
    torch_threads_per_job=1,
    datasets=DATASETS_NOW,
    expected_datasets=DATASETS_NOW,
    studies=["P", "Q", "R"],
    samplers=["ENN", "CNN", "TL"],             # 與 I/J/K 對應。⚠ 兩批之間不要改（改了要換 run_name 重跑）
    modes=["tuned", "default"],
    enn_k=3, cnn_k=1, cnn_seed=42,             # 與 US_azure_v2 相同（不同會被擋下）
    exclude_categorical=False,
)

if __name__ == "__main__":
    main(CFG)
