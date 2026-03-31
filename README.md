# OCC Screening & Autoencoder Pretraining on KEEL Datasets

研究目標：在 KEEL 不平衡二元資料集上，評估以 Autoencoder 特徵萃取搭配 One-Class Classification (OCC) 方法的異常偵測效果。

---

## 專案結構

```
.
├── keel_occ_screening.py   # 篩選 KEEL 資料集：OCC baseline AUC 評估
├── pretrain_v2.py          # 前測主程式：AE 架構搜尋 + OCC 評估
├── KEEL_Dataset/           # KEEL .dat 資料集（不納入版控）
├── data/                   # 選定的前測資料集（不納入版控）
├── results/                # 實驗輸出 Excel（不納入版控）
└── Pipfile                 # 套件依賴
```

---

## 流程說明

### Step 1 — 資料集篩選 (`keel_occ_screening.py`)

對 `KEEL_Dataset/` 下所有資料集做 5-fold OCC 評估（OCSVM / LOF / Isolation Forest），篩出至少 2 種 OCC 的 AUC < 0.7 的「困難」資料集，並依 Imbalance Ratio (IR) 中位數各取一個代表作為前測資料集。

輸出：`occ_auc_results.xlsx`（sheet: `all` / `hard`）

### Step 2 — 前測實驗 (`pretrain_v2.py`)

對選定資料集進行 Autoencoder 架構搜尋：
- 層數：2 層（h2）、3 層（h3）
- 瓶頸維度比例：`input_dim × {1/4, 1/3, 1/2, 1/1}`
- AE 種類 × OCC 種類 = 共 96 組/資料集

---

## 環境安裝

```bash
pip install pipenv
pipenv install
pipenv shell
```

或直接用 pip：

```bash
pip install torch pandas scikit-learn scipy numpy openpyxl
```

Python 版本：3.10

---

## 執行方式

```bash
# Step 1：篩選資料集
python keel_occ_screening.py

# Step 2：前測實驗（需先完成 Step 1 並將資料集放至 data/）
python pretrain_v2.py
```

---

## 資料集來源

[KEEL Dataset Repository](https://sci2s.ugr.es/keel/imbalanced.php) — 不平衡分類資料集
