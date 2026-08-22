import os
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# CONFIG
# ============================================================

DATA_FILE = "market_data_log.csv"
MODEL_FILE = "xgboost_direction_model.pkl"

MIN_MOVE = 0.0040          # 0.40%
FUTURE_HORIZON = 5         # next 5 observations
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Classes:
# 0 = NO TRADE
# 1 = LONG
# 2 = SHORT


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not os.path.exists(DATA_FILE):
        print(f"ERROR: {DATA_FILE} not found.")
        return None

    df = pd.read_csv(DATA_FILE)

    required = [
        "timestamp",
        "symbol",
        "current_price",
        "obi_top20",
        "spread"
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        print("Missing columns:", missing)
        return None

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    df["current_price"] = pd.to_numeric(
        df["current_price"],
        errors="coerce"
    )

    for col in [
        "obi_top20",
        "spread"
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "timestamp",
            "symbol",
            "current_price"
        ]
    )

    df = df.sort_values(
        ["symbol", "timestamp"]
    ).reset_index(drop=True)

    return df


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "top20_bid_sum",
    "top20_ask_sum",
    "obi_top20",
    "obi_top50",
    "spread",
    "bid_ask_ratio",
    "total_depth",
    "trend_10",
    "trend_20",
    "return_1",
    "return_3",
    "return_5",
    "volatility",
    "ofi_normalized",
    "obi_alignment",
    "spread_pct"
]


def create_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # DEPTH
    # --------------------------------------------------------

    if (
        "top20_bid_sum" not in df.columns
        and
        "top20_bid_volume" in df.columns
    ):
        df["top20_bid_sum"] = (
            df["top20_bid_volume"]
        )

    if (
        "top20_ask_sum" not in df.columns
        and
        "top20_ask_volume" in df.columns
    ):
        df["top20_ask_sum"] = (
            df["top20_ask_volume"]
        )

    if (
        "top20_bid_sum" not in df.columns
        or
        "top20_ask_sum" not in df.columns
    ):
        print("ERROR: Top20 bid/ask data missing.")
        return None

    # --------------------------------------------------------
    # TOP50
    # --------------------------------------------------------

    if "obi_top50" not in df.columns:
        print(
            "WARNING: obi_top50 missing. "
            "Using top20 as fallback."
        )

        df["obi_top50"] = df["obi_top20"]

    # --------------------------------------------------------
    # PRICE RETURNS
    # --------------------------------------------------------

    grouped = df.groupby("symbol")["current_price"]

    df["return_1"] = grouped.pct_change(1)
    df["return_3"] = grouped.pct_change(3)
    df["return_5"] = grouped.pct_change(5)

    # --------------------------------------------------------
    # DEPTH FEATURES
    # --------------------------------------------------------

    df["bid_ask_ratio"] = (
        df["top20_bid_sum"]
        /
        (
            df["top20_ask_sum"]
            + 1e-12
        )
    )

    df["total_depth"] = (
        df["top20_bid_sum"]
        +
        df["top20_ask_sum"]
    )

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    df["sma_10"] = (
        grouped.transform(
            lambda x:
            x.rolling(
                10,
                min_periods=10
            ).mean()
        )
    )

    df["sma_20"] = (
        grouped.transform(
            lambda x:
            x.rolling(
                20,
                min_periods=20
            ).mean()
        )
    )

    df["trend_10"] = (
        (
            df["current_price"]
            -
            df["sma_10"]
        )
        /
        (
            df["sma_10"]
            + 1e-12
        )
    )

    df["trend_20"] = (
        (
            df["current_price"]
            -
            df["sma_20"]
        )
        /
        (
            df["sma_20"]
            + 1e-12
        )
    )

    # --------------------------------------------------------
    # VOLATILITY
    # --------------------------------------------------------

    df["volatility"] = (
        df.groupby("symbol")["return_1"]
        .transform(
            lambda x:
            x.rolling(
                20,
                min_periods=10
            ).std()
        )
    )

    # --------------------------------------------------------
    # OFI
    # --------------------------------------------------------

    if "ofi" in df.columns:

        df["ofi"] = pd.to_numeric(
            df["ofi"],
            errors="coerce"
        )

        df["ofi_normalized"] = (
            df["ofi"]
            /
            (
                df["total_depth"]
                + 1e-12
            )
        )

    else:
        df["ofi_normalized"] = 0.0

    # --------------------------------------------------------
    # OBI AGREEMENT
    # --------------------------------------------------------

    df["obi_alignment"] = (
        df["obi_top20"]
        *
        df["obi_top50"]
    )

    # --------------------------------------------------------
    # SPREAD
    # --------------------------------------------------------

    df["spread_pct"] = (
        df["spread"]
        /
        (
            df["current_price"]
            + 1e-12
        )
    )

    return df


# ============================================================
# DIRECTION TARGET
# ============================================================

def create_target(df):

    df = df.copy()

    grouped = df.groupby("symbol")["current_price"]

    future_max = (
        grouped
        .transform(
            lambda x:
            x.shift(-1)
            .rolling(
                FUTURE_HORIZON,
                min_periods=FUTURE_HORIZON
            )
            .max()
        )
    )

    future_min = (
        grouped
        .transform(
            lambda x:
            x.shift(-1)
            .rolling(
                FUTURE_HORIZON,
                min_periods=FUTURE_HORIZON
            )
            .min()
        )
    )

    df["future_up_move"] = (
        (
            future_max
            -
            df["current_price"]
        )
        /
        (
            df["current_price"]
            + 1e-12
        )
    )

    df["future_down_move"] = (
        (
            df["current_price"]
            -
            future_min
        )
        /
        (
            df["current_price"]
            + 1e-12
        )
    )

    # --------------------------------------------------------
    # 0 = NO TRADE
    # 1 = LONG
    # 2 = SHORT
    # --------------------------------------------------------

    df["target"] = 0

    long_condition = (
        (df["future_up_move"] >= MIN_MOVE)
        &
        (
            df["future_up_move"]
            >=
            df["future_down_move"]
        )
    )

    short_condition = (
        (df["future_down_move"] >= MIN_MOVE)
        &
        (
            df["future_down_move"]
            >
            df["future_up_move"]
        )
    )

    df.loc[
        long_condition,
        "target"
    ] = 1

    df.loc[
        short_condition,
        "target"
    ] = 2

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_dataset(df):

    missing = [
        c for c in FEATURES
        if c not in df.columns
    ]

    if missing:
        print("Missing features:")
        for c in missing:
            print(" -", c)
        return None, None

    df = df.dropna(
        subset=FEATURES + ["target"]
    ).copy()

    if len(df) < 300:
        print(
            f"Only {len(df)} usable rows."
        )
        print(
            "At least 300+ rows recommended."
        )
        return None, None

    X = df[FEATURES]
    y = df["target"]

    return X, y


# ============================================================
# TRAIN
# ============================================================

def train_model():

    print("=" * 70)
    print("XGBOOST DIRECTION MODEL")
    print("=" * 70)

    df = load_data()

    if df is None:
        return

    df = create_features(df)

    if df is None:
        return

    df = create_target(df)

    X, y = prepare_dataset(df)

    if X is None:
        return

    print()
    print("Samples:", len(X))
    print()
    print("Target distribution:")

    print(
        y.value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # TIME SERIES SPLIT
    # --------------------------------------------------------

    split = int(
        len(X)
        *
        (1 - TEST_SIZE)
    )

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    print()
    print("Training:", len(X_train))
    print("Testing :", len(X_test))

    # --------------------------------------------------------
    # XGBOOST
    # --------------------------------------------------------

    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=5,
        subsample=0.80,
        colsample_bytree=0.80,
        gamma=0.10,
        reg_alpha=0.10,
        reg_lambda=1.0,

        objective="multi:softprob",
        num_class=3,

        eval_metric="mlogloss",

        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    print()
    print("Training XGBoost...")

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print()
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    print()
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1, 2],
            target_names=[
                "NO_TRADE",
                "LONG",
                "SHORT"
            ],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1, 2]
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    package = {
        "model": model,
        "features": FEATURES,
        "min_move": MIN_MOVE,
        "future_horizon": FUTURE_HORIZON,
        "classes": {
            0: "NO_TRADE",
            1: "LONG",
            2: "SHORT"
        }
    }

    joblib.dump(
        package,
        MODEL_FILE
    )

    print()
    print(
        f"MODEL SAVED: {MODEL_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    train_model()
    
