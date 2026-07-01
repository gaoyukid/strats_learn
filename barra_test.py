import pandas as pd
import numpy as np
import logging
from scipy import stats

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ==============================================
# 1. Simulate raw market data (replace with real daily/monthly data in production)
# ==============================================
np.random.seed(42)

# Create stock ID list
stock_list = [f"STK{i:03d}" for i in range(500)]
n = len(stock_list)

# Raw dataset columns
df = pd.DataFrame({
    "stock_id": stock_list,
    "industry": np.random.choice(["Bank", "Tech", "Consumer", "Manufacture"], size=n),
    "market_cap": np.random.lognormal(22, 1.2, size=n),  # Total market capitalization
    "pb": np.random.lognormal(0.5, 0.8, size=n),         # Price-to-Book ratio for Value factor
    "ret_excess": np.random.normal(0, 0.08, size=n)     # Monthly stock excess return vs benchmark index
})

# ==============================================
# 2. General function to calculate standardized Factor Exposure (Barra standard 4-step pipeline)
# ==============================================
def calc_exposure(df, factor_col, industry_col="industry", winsor_pct=0.01):
    """
    Calculate standardized cross-sectional factor exposure
    :param df: DataFrame containing all stock observations
    :param factor_col: Column name of raw factor metric
    :param industry_col: Column name for industry classification
    :param winsor_pct: Percentage threshold for two-sided winsorization
    :return: Series of standardized factor exposure values
    """
    data = df[factor_col].copy()

    # Step1: Winsorization to clip extreme outliers
    lower = np.quantile(data, winsor_pct)
    upper = np.quantile(data, 1 - winsor_pct)
    data = np.clip(data, lower, upper)

    # Step2: Industry neutralization - subtract industry mean from raw factor values
    industry_mean = df.groupby(industry_col)[factor_col].transform("mean")
    data_neutral = data - industry_mean

    # Step3: Cross-sectional Z-score normalization, mean=0, std=1
    exposure = stats.zscore(data_neutral, nan_policy="omit")
    return pd.Series(exposure, index=df.index, name=f"{factor_col}_expo")

# ------------------------------
# Calculate exposure for two style factors
# ------------------------------
# Size factor: natural log of market cap
df["ln_cap"] = np.log(df["market_cap"])
df["size_expo"] = calc_exposure(df, factor_col="ln_cap")
logger.info("Completed calculation of Size factor exposure")

# Value factor: inverse price-to-book ratio (1/PB)
df["inv_pb"] = 1 / df["pb"]
df["value_expo"] = calc_exposure(df, factor_col="inv_pb")
logger.info("Completed calculation of Value factor exposure")

# ==============================================
# 3. Cross-sectional WLS regression to solve Factor Return (core Barra algorithm)
# ==============================================
def solve_factor_return(df, factor_expo_cols, ret_col="ret_excess", weight_col="market_cap"):
    """
    Weighted Least Squares regression to estimate period factor returns
    :param df: DataFrame with factor exposures, stock returns and weighting metrics
    :param factor_expo_cols: List of column names for style factor exposures
    :param ret_col: Column name for stock excess return (dependent variable y)
    :param weight_col: Column name for regression weight (market cap for A-share)
    :return: Dictionary mapping factor name to its estimated factor return
    """
    # Independent variable matrix X (factor exposure matrix)
    X = df[factor_expo_cols].values
    # Dependent variable vector y (stock excess returns)
    y = df[ret_col].values
    # Regression weight vector
    w = df[weight_col].values
    W = np.diag(w / np.sum(w))

    # WLS matrix formula: f = (X^T W X)^(-1) X^T W y
    X_T = X.T
    XTW = X_T @ W
    XTWX = XTW @ X
    XTWy = XTW @ y
    factor_ret = np.linalg.inv(XTWX) @ XTWy

    return dict(zip(factor_expo_cols, factor_ret))

# Estimate factor returns for current period
factor_cols = ["size_expo", "value_expo"]
factor_return_dict = solve_factor_return(df, factor_expo_cols=factor_cols)
logger.info("Finished WLS regression to solve factor returns")

# Log factor return results
logger.info("========== Period Factor Return Results ==========")
for name, ret in factor_return_dict.items():
    logger.info(f"{name}: {ret:.4%}")
logger.info("==================================================")

# ==============================================
# 4. Calculate single stock factor return contribution (Exposure × FactorReturn)
# ==============================================
for col in factor_cols:
    ret_name = col.replace("_expo", "_ret_contrib")
    df[ret_name] = df[col] * factor_return_dict[col]
logger.info("Calculated individual factor contribution for all stocks")

# Total return contribution from all style factors
df["style_total_contrib"] = sum([df[col.replace("_expo", "_ret_contrib")] for col in factor_cols])
# Specific return = stock excess return minus total style factor contribution
df["specific_ret"] = df["ret_excess"] - df["style_total_contrib"]
logger.info("Computed total style contribution and specific return for each stock")

# Log attribution results for top 5 stocks
show_cols = ["stock_id", "size_expo", "value_expo", "size_ret_contrib", "value_ret_contrib", "style_total_contrib", "specific_ret"]
sample_df = df[show_cols].head().round(4)
logger.info("\nStock performance attribution sample (first 5 entries):\n%s", sample_df)