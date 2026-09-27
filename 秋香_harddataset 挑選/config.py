# =============================================================================
# config.py
# 全域設定：路徑、實驗開關、K 值掃描、分群/OCC/DS/US 啟用清單
# =============================================================================
import os


# -----------------------------------------------------------------------------
# 路徑
# -----------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OCC_ROOT    = os.path.dirname(PROJECT_DIR)

BASE_DIR   = os.path.join(OCC_ROOT, "KEEL_Dataset")
RESULT_DIR = os.path.join(OCC_ROOT, "experiments_result")
FOLD_RESULT_SUBDIR = "fold_results"


# -----------------------------------------------------------------------------
# 共用實驗設定
# -----------------------------------------------------------------------------
RANDOM_STATE = 42
N_FOLDS      = 5

# GPU execution. Experiments fail fast instead of silently falling back to CPU.
DEVICE       = "cuda:0"
REQUIRE_CUDA = True

# OCC score calibration used by OCCWrapper. Also stored in CSV checkpoints.
SCORE_CALIBRATION = "zero_centered_robust_sigmoid_v1"
PIPELINE_VERSION = "ds_split_before_undersampling_torch_seed_v2"

# DS 用：train → train + dsel 切分比例
DSEL_RATIO   = 0.2

# AE / DEC（給 dec / dec-ap / dec-hdbscan 用）
AE_HIDDEN  = [64, 32]
LATENT_DIM = 10
AE_EPOCHS  = 100
DEC_EPOCHS = 150


# -----------------------------------------------------------------------------
# 哪些實驗要跑
# -----------------------------------------------------------------------------
ENABLED_EXPERIMENTS = {
    # Exp 11: train the existing local OCC pool on selected feature subspaces.
    "exp11": False,
    # Stage 2 KNORA-U tuning: fixed k=7, compare voting and KNNE.
    "exp6_tuning_knorau_options": False,
    # Stage 1 KNORA-U tuning: vary only its competence-region neighbor count.
    "exp6_tuning_knorau_k": False,
    # Final fixed configuration: repeat across seeds for stability assessment.
    "exp6_final": False,
    # Exp 10: route each query to nearby clusters before KNORA-U.
    "exp10": False,
    # Tune Isolation Forest after fixing balanced OCSVM and LOF settings.
    "exp6_tuning_if": False,
    # Exp 9: keep the winning pool and KNORA-U, change only its k-NN space.
    "exp9": False,
    # Exp 8: add global IF/OCSVM/LOF to the winning local Exp 6 pool.
    "exp8": False,
    # Tune LOF after selecting the balanced OCSVM candidate.
    "exp6_tuning_lof": False,
    # Tune OCSVM inside the winning heterogeneous pool after DSEL tuning.
    "exp6_tuning_ocsvm": False,
    # Tune the DSEL holdout ratio after KNORA-U wins the voting comparison.
    "exp6_tuning_dsel": False,
    # Compare fixed-pool aggregation/selection methods after exp6_tuning.
    "exp6_voting": False,
    # Local search around the best exp6 configurations. Disabled by default;
    # run experiments.exp6_tuning directly to avoid rerunning exp1-exp7.
    "exp6_tuning": False,
    # Exp 1 extension: no clustering, one OCC, no DS, 4 undersampling modes.
    # Disabled in main.py by default; run its module directly when needed.
    "exp1_2": False,
    "exp1": True,    # 不分群 × 同質 (3)
    "exp2": True,   # 分群 × 同質 (21)
    "exp3": True,   # 分群 × 同質 × DS (63)
    "exp4": True,   # 不分群 × 異質 (1)
    "exp5": True,   # 分群 × 異質 (7)
    "exp6": True,   # 分群 × 異質 × DS (21)
    "exp7": True,   # 不分群 × 異質 × DS (3)
}


# -----------------------------------------------------------------------------
# 元件啟用清單
# -----------------------------------------------------------------------------
ENABLED_OCC = ["IF", "OCSVM", "LOF"]

# 7 種分群
ENABLED_CLUSTERINGS = [
    "kmeans",       # 手動
    "gmm",          # 手動
    "dec",          # 手動
    "ap",           # 自動
    "hdbscan",      # 自動
    "dec-ap",       # 自動
    "dec-hdbscan",  # 自動
]

# 3 種 DS
ENABLED_DS = ["KNORA-U", "DES-KNN", "DES-P"]

# 4 種 undersampling（"none" 表示不做 US）
ENABLED_UNDERSAMPLING = ["none", "tomek", "enn", "allknn"]


# -----------------------------------------------------------------------------
# 手動分群的 K 值掃描範圍（kmeans / gmm / dec）
# 每個 K 值各輸出一份 CSV
# -----------------------------------------------------------------------------
MANUAL_K_LIST = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]

# Exp6 tuning stage 1: keep the experiment design fixed and only refine the
# promising clustering range.  Separate constants prevent changing exp2/3/5/6.
EXP6_TUNING_CLUSTERINGS = ["kmeans", "dec"]
EXP6_TUNING_K_LIST = [13, 15, 17, 19, 21, 25, 30]
EXP6_TUNING_DS = ["KNORA-U"]
EXP6_TUNING_UNDERSAMPLING = ["enn", "allknn"]

# Exp6 voting comparison: all other factors stay fixed so that only the final
# ensemble decision rule changes. Soft/Hard voting still reserve the same DSEL
# split as DES, ensuring that every method trains the OCC pool on identical data.
EXP6_VOTING_CLUSTERING = "kmeans"
EXP6_VOTING_K = 15
EXP6_VOTING_UNDERSAMPLING = "enn"
EXP6_VOTING_METHODS = [
    "Soft-Voting", "Hard-Voting", "KNORA-U", "DES-KNN", "DES-P",
]

# Exp6 DSEL tuning: only DSEL_RATIO changes; all other factors use the
# exp6_tuning All-dataset winner.
EXP6_DSEL_TUNING_RATIOS = [0.1, 0.2, 0.3, 0.4]
EXP6_DSEL_TUNING_CLUSTERING = "kmeans"
EXP6_DSEL_TUNING_K = 15
EXP6_DSEL_TUNING_DS = "KNORA-U"
EXP6_DSEL_TUNING_UNDERSAMPLING = "enn"

# OCSVM tuning includes sklearn's default nu=0.5 as a baseline. Gamma values
# use the already scaled feature space and a small logarithmic-style grid.
EXP6_OCSVM_TUNING_NU = [0.01, 0.03, 0.05, 0.1, 0.2, 0.5]
EXP6_OCSVM_TUNING_GAMMA = ["scale", 0.01, 0.03, 0.1, 0.3]
EXP6_OCSVM_TUNING_DSEL_RATIO = 0.4

# LOF tuning keeps the balanced OCSVM candidate fixed. The effective neighbor
# count is capped at cluster_size - 1 by pool_builder for small clusters.
EXP6_LOF_TUNING_NEIGHBORS = [5, 10, 15, 20, 30, 40]
EXP6_LOF_TUNING_METRICS = ["euclidean", "manhattan"]
EXP6_LOF_TUNING_DSEL_RATIO = 0.4
EXP6_LOF_TUNING_OCSVM_PARAMS = {"nu": 0.01, "gamma": 0.01}

# Isolation Forest tuning. The sklearn baseline (100, "auto", 1.0) is included.
EXP6_IF_TUNING_N_ESTIMATORS = [100, 300, 500]
EXP6_IF_TUNING_MAX_SAMPLES = ["auto", 0.5, 0.75, 1.0]
EXP6_IF_TUNING_MAX_FEATURES = [0.5, 0.75, 1.0]
EXP6_IF_TUNING_DSEL_RATIO = 0.4
EXP6_IF_TUNING_OCSVM_PARAMS = {"nu": 0.01, "gamma": 0.01}
EXP6_IF_TUNING_LOF_PARAMS = {"n_neighbors": 20, "metric": "euclidean"}

# Final balanced configuration selected after staged tuning. These seeds assess
# stability only; they must not be used to select another hyperparameter set.
EXP6_FINAL_SEEDS = [21, 42, 84, 126, 168]
EXP6_FINAL_DSEL_RATIO = 0.4
EXP6_FINAL_IF_PARAMS = {
    "n_estimators": 300,
    "max_samples": "auto",
    "max_features": 0.75,
}
EXP6_FINAL_OCSVM_PARAMS = {"nu": 0.01, "gamma": 0.01}
EXP6_FINAL_LOF_PARAMS = {"n_neighbors": 20, "metric": "euclidean"}

# KNORA-U stage 1 tuning. Voting and KNNE remain at their defaults so this
# experiment isolates the internal competence-neighborhood size.
EXP6_KNORAU_TUNING_K = [3, 5, 7, 9, 11, 15]
EXP6_KNORAU_TUNING_DSEL_RATIO = 0.4
EXP6_KNORAU_TUNING_IF_PARAMS = dict(EXP6_FINAL_IF_PARAMS)
EXP6_KNORAU_TUNING_OCSVM_PARAMS = dict(EXP6_FINAL_OCSVM_PARAMS)
EXP6_KNORAU_TUNING_LOF_PARAMS = dict(EXP6_FINAL_LOF_PARAMS)

# KNORA-U stage 2: k=7 won the AUC comparison, so only voting and KNNE vary.
EXP6_KNORAU_OPTIONS_K = 7
EXP6_KNORAU_OPTIONS_VOTING = ["hard", "soft"]
EXP6_KNORAU_OPTIONS_KNNE = [False, True]
EXP6_KNORAU_OPTIONS_DSEL_RATIO = 0.4

# Exp 8 global + local OCC pool.  Keep the winning Exp 6 structure fixed and
# change only which global OCCs are appended to the local heterogeneous pool.
EXP8_CLUSTERING = "kmeans"
EXP8_K = 15
EXP8_DS = "KNORA-U"
EXP8_UNDERSAMPLING = "enn"
EXP8_GLOBAL_OCC_OPTIONS = ["none", "IF", "OCSVM", "LOF", "all"]

# Exp 9 model-score-space DS.  "original" is the untouched Exp 6 control;
# "score" represents each sample by the local OCC anomaly-score vector.
EXP9_CLUSTERING = "kmeans"
EXP9_K = 15
EXP9_DS = "KNORA-U"
EXP9_UNDERSAMPLING = "enn"
EXP9_NEIGHBORHOOD_SPACES = ["original", "score"]
EXP9_SCORE_TRANSFORM = "dsel_zscore"

# Exp 10 cluster-aware pool routing. None is the all-cluster control.
EXP10_CLUSTERING = "kmeans"
EXP10_K = 15
EXP10_DS = "KNORA-U"
EXP10_UNDERSAMPLING = "enn"
EXP10_ROUTE_TOP_N = [None, 1, 2, 3, 5]

# Exp 11 feature-subspace OCC ensemble.
EXP11_CLUSTERING = "kmeans"
EXP11_K = 15
EXP11_DS = "KNORA-U"
EXP11_UNDERSAMPLING = "enn"
EXP11_FEATURE_METHODS = ["full", "correlation", "cluster-specific"]
EXP11_CORRELATION_THRESHOLD = 0.9
EXP11_KEEP_RATIO = 0.7


# -----------------------------------------------------------------------------
# OCC 最小樣本數要求（cluster 太小就跳過該 cluster）
# -----------------------------------------------------------------------------
MIN_SIZE = {
    "IF":    2,
    "OCSVM": 5,
    "LOF":   6,
}


# -----------------------------------------------------------------------------
# 資料集
# -----------------------------------------------------------------------------
DATASET_NAMES = [
    "abalone9-18", "abalone19", "ecoli-0_vs_1", "ecoli-0-1-3-7_vs_2-6",
    "ecoli1", "ecoli2", "ecoli3", "ecoli4",
    "glass-0-1-2-3_vs_4-5-6", "glass-0-1-6_vs_2", "glass-0-1-6_vs_5",
    "glass0", "glass1", "glass2", "glass4", "glass5", "glass6",
    "haberman", "iris0", "new-thyroid1", "new-thyroid2",
    "page-blocks0", "page-blocks-1-3_vs_4", "pima",
    "segment0", "shuttle-c0-vs-c4", "shuttle-c2-vs-c4",
    "vehicle0", "vehicle1", "vehicle2", "vehicle3",
    "vowel0", "wisconsin",
    "yeast-0-5-6-7-9_vs_4", "yeast-1_vs_7", "yeast-1-2-8-9_vs_7",
    "yeast-1-4-5-8_vs_7", "yeast1", "yeast-2_vs_4", "yeast-2_vs_8",
    "yeast3", "yeast4", "yeast5", "yeast6",
]

DATASETS = {n: os.path.join(BASE_DIR, f"{n}-5-fold") for n in DATASET_NAMES}


# -----------------------------------------------------------------------------
# 分類：哪些分群方法是手動（需要 K 值）/ 自動
# -----------------------------------------------------------------------------
MANUAL_CLUSTERINGS = {"kmeans", "gmm", "dec"}
AUTO_CLUSTERINGS   = {"ap", "hdbscan", "dec-ap", "dec-hdbscan"}
