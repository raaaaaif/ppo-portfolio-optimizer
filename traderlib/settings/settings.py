# directory
DATA_SAVE_DIR = "datasets"
TRAINED_MODEL_DIR = "trained_models"
TENSORBOARD_LOG_DIR = "tensorboard_log"
RESULTS_DIR = "results"

# date format: '%Y-%m-%d'

# stockstats technical indicator column names
# check https://pypi.org/project/stockstats/ for different names
# define your indicators below

TIME_ZONE_SELFDEFINED = "xxx"  # If neither of the above is your time zone
USE_TIME_ZONE_SELFDEFINED = 0  # 0 (default) or 1 (use the self defined)

# parameters for data sources
BINANCE_BASE_URL = "https://data.binance.vision/"  # binance url
trainStartDate = "2022-01-03"
trainEndDate = "2025-04-30"
testStartDate = "2025-05-01"
testEndDate = "2025-05-14"
tradeStartDate = "2025-05-15"
tradeEndDate = "2025-05-24"

INDICATORS = [
    "macd",
    "boll_ub",
    "boll_lb",
]
