"""
run_azure_US2.py — S / T / U 兩階段 under-sampling（第二階段 ENN）：Azure 正式版（Standard_D8as_v5）
S（OF）/ T（DF）/ U（OF+DF）× 資料集 × 5 折
策略：2S_Majority_ENN（主要）、2S_ENN_ENN（必要對照）、2S_Union_ENN（等價核對）、2S_Intersection_ENN（次要）

════════════════════════════════════════════════════════════════════════════
分批跑：先跑 5 個資料集，其餘之後再補（已跑完的不會重跑）
════════════════════════════════════════════════════════════════════════════
  第一批：下面的 DATASETS_NOW = DATASETS_FIRST5（現在就是這樣），直接執行。
          跑完會產生 results_tuning/US2_azure_v2/US2_azure_v2_summary_5of17ds.xlsx（檔名標明只有 5/17）。
  第二批：把 DATASETS_NOW 那兩行的註解對調（改成 DATASETS_17），「run_name 不要改」，再執行同一個指令。
          程式看到前 5 個資料集的格子已經存在就跳過，只跑剩下 12 個；跑完產生正式的 US2_azure_v2_summary.xlsx。
          （5 個的那份檔案會留著，不會被蓋掉。）
  ⚠ 兩批之間不要改其他設定（studies、two_stage、參數格…）也不要換 occ_tune_*.py：改了程式會停下來並說明原因。
  ⚠ 只有 5 個資料集時，Wilcoxon 檢定最小可能的 p 是 0.0625，不可能 < 0.05。
     第一批請只看 ΔAUC 與勝/平/負的方向，以及 03 / 03b 分頁的遮罩診斷；顯著性等 17 個跑完再看。
  run_azure_USENS.py 會直接 import 這裡的 DATASETS_NOW，所以兩支程式只要改這一個地方。

第一批 5 個資料集怎麼挑的：每個資料來源各一個（glass / haberman / pima / vehicle / yeast）、訓練多數類都在
  約 500 筆以內（跑得快）、不平衡比例從 1.9 到 14.3、維度從 3 到 18。是依「資料大小與來源」挑的，沒有看結果；
  想換成別的 5 個可以直接改 DATASETS_FIRST5（必須是 17 個裡面的名字）。

════════════════════════════════════════════════════════════════════════════
需要 A~C（full_azure_v2）與 I~O（US_azure_v2）都已經跑完，而且用同一個 out_root、同一個 Python 環境
════════════════════════════════════════════════════════════════════════════
  1. AE 特徵讀 A~C 留下的 _df_cache（嚴格讀取：壞檔或不符就讓那一格失敗並記錄，不會自動重訓 AE）
  2. 對照組 none / ENN / S1 / S2 / S3 直接讀 US_azure_v2 的結果，不重算
  3. 每一格都會重算第一階段遮罩並和 US_azure_v2 存下來的遮罩指紋比對；另外用舊預設參數重算單次 ENN 的三個 OCC，
     五項指標要和 US_azure_v2 相同。任何一項不同，這一格不存檔（原因寫在 errors.log）
  4. 新策略留下的樣本若和某個對照完全相同，OCC 不重新訓練，直接沿用 US_azure_v2 那一格的結果
  → Azure 上的 results_tuning/US_azure_v2/（cells、signature.json、dataset_profile.csv）與 _df_cache 都要留著，
    只留 summary.xlsx 不夠。

同一個資料夾要有：occ_tune_core.py、occ_tune_ensemble.py、occ_tune_us.py、occ_tune_us2.py、run_azure.py
   （occ_tune_core / ensemble / us 請用當初跑 A~O 的那一份，不要換）

啟動（背景執行）：
    nohup python run_azure_US2.py > run_azure_US2.log 2>&1 &
    tail -f run_azure_US2.log            # 會印「已用幾分、預估剩幾分」
開跑前會先檢查：AE 特徵快取齊全、US_azure_v2 的設定／資料指紋／套件版本／程式指紋與現在相同、需要的格子都在。
   任何一項不符會直接停下並列出原因，不會默默用不一致的東西算下去。
要停止：kill <主程式 PID>（PID 寫在 results_tuning/US2_azure_v2/RUNNING.lock 裡）；不要用 kill -9。
中斷後：再執行同一個 nohup 指令，會從斷掉的地方接續。
統整：python run_azure_US2.py --summarize-only
"""
import os
from pathlib import Path
from occ_tune_us2 import Us2Config, main
from run_azure import DATASETS_17          # 與 A~O 用同一份名單

BASE = Path(__file__).resolve().parent

# ══════════ 這次要跑哪些資料集（分批跑只改這裡）══════════
DATASETS_FIRST5 = [
    "glass-0-1-6_vs_2-5-fold",     # glass｜多數類 140｜IR 10.8｜ 9 維
    "haberman-5-fold",             # haberman｜多數類 180｜IR 2.8｜ 3 維
    "pima-5-fold",                 # pima｜多數類 400｜IR 1.9｜ 8 維
    "vehicle1-5-fold",             # vehicle｜多數類 503｜IR 2.9｜18 維
    "yeast-1_vs_7-5-fold",         # yeast｜多數類 343｜IR 14.3｜ 7 維
]
DATASETS_NOW = DATASETS_FIRST5      # ← 第一批：先跑這 5 個
# DATASETS_NOW = DATASETS_17        # ← 第二批：5 個跑完後，把上一行加 #、這一行拿掉 #，再執行同一個指令（run_name 不要改）

assert set(DATASETS_NOW) <= set(DATASETS_17), "DATASETS_NOW 裡有不在 17 個資料集名單內的名字（請檢查拼字）"

CFG = Us2Config(
    data_root=str(BASE / "data"),
    out_root=str(BASE / "results_tuning"),     # 與 A~C、I~O 相同 → 共用 AE 特徵快取、讀得到 US_azure_v2
    run_name="US2_azure_v2",                   # 兩批都用同一個名字，第二批才會接著補
    us_run_name="US_azure_v2",                 # 舊的 I~O run（對照與核對用）
    n_jobs=os.cpu_count() or 8,                # 和 run_azure_USENS.py 同時跑時，兩支各改成 4
    torch_threads_per_job=1,
    datasets=DATASETS_NOW,
    expected_datasets=DATASETS_NOW,            # 必須剛好是這些資料集，多一個少一個都會停止
    studies=["S", "T", "U"],
    two_stage=["2S_ENN_ENN", "2S_Union_ENN", "2S_Majority_ENN", "2S_Intersection_ENN"],
    enn_k=3, cnn_k=1, cnn_seed=42,             # 與 US_azure_v2 相同（不同會被擋下）
    exclude_categorical=False,
)

if __name__ == "__main__":
    main(CFG)
