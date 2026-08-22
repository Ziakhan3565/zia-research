import os
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIG
# ============================================================

DATA_FILE = "market_data_log.csv"
MODEL_FILE = "xgboost_obi_model.pkl"

MIN_MOVE = 0.0040       # 0.40%
FUTURE_HORIZON = 5      # next 5 observations

TEST_SIZE = 0.20

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not os.path.exists(DATA_FILE):

        print(
            f"❌ '{DATA_FILE}' nahi mili!"
        )

        return None

    try:

        df = pd.read_csv(
            DATA_FILE
        )

    except Exception as e:

        print(
            f"❌ CSV error: {e}"
        )

        return None

    print(
        f"📥 Raw rows: {len(df):,}"
    )

    required = [
        "timestamp",
        "symbol",
        "current_price",
        "obi_top20",
        "spread"
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        print(
            "\n❌ Missing columns:"
        )

        for col in missing:
            print(
                f"   {col}"
            )

        return None

    # Timestamp
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    # Price
    df["current_price"] = pd.to_numeric(
        df["current_price"],
        errors="coerce"
    )

    # Numeric columns
    numeric_columns = [
        "obi_top20",
        "spread"
    ]

    for col in numeric_columns:

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
        [
            "symbol",
            "timestamp"
        ]
    ).reset_index(
        drop=True
    )

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # Bid / Ask depth
    # --------------------------------------------------------

    if (
        "top20_bid_sum" in df.columns and
        "top20_ask_sum" in df.columns
    ):

        df["bid_ask_ratio"] = (
            df["top20_bid_sum"] /
            (
                df["top20_ask_sum"] +
                1e-12
            )
        )

        df["total_depth"] = (
            df["top20_bid_sum"] +
            df["top20_ask_sum"]
        )

    elif (
        "top20_bid_volume" in df.columns and
        "top20_ask_volume" in df.columns
    ):

        df["bid_ask_ratio"] = (
            df["top20_bid_volume"] /
            (
                df["top20_ask_volume"] +
                1e-12
            )
        )

        df["total_depth"] = (
            df["top20_bid_volume"] +
            df["top20_ask_volume"]
        )

    else:

        print(
            "❌ Top20 bid/ask columns missing."
        )

        return None

    # --------------------------------------------------------
    # Top 50 OBI
    # --------------------------------------------------------

    if "obi_top50" not in df.columns:

        # If Top50 does not exist, use Top20
        print(
            "⚠️ obi_top50 nahi mila. "
            "Top20 OBI use hoga."
        )

        df["obi_top50"] = (
            df["obi_top20"]
        )

    # --------------------------------------------------------
    # Price returns
    # --------------------------------------------------------

    df["return_1"] = (
        df.groupby("symbol")[
            "current_price"
        ]
        .pct_change(1)
    )

    df["return_3"] = (
        df.groupby("symbol")[
            "current_price"
        ]
        .pct_change(3)
    )

    df["return_5"] = (
        df.groupby("symbol")[
            "current_price"
        ]
        .pct_change(5)
    )

    # --------------------------------------------------------
    # SMA trend
    # --------------------------------------------------------

    df["sma_10"] = (
        df.groupby("symbol")[
            "current_price"
        ]
        .transform(
            lambda x:
            x.rolling(
                10,
                min_periods=10
            ).mean()
        )
    )

    df["sma_20"] = (
        df.groupby("symbol")[
            "current_price"
        ]
        .transform(
            lambda x:
            x.rolling(
                20,
                min_periods=20
            ).mean()
        )
    )

    # Trend percentage
    df["trend_10"] = (
        (
            df["current_price"] -
            df["sma_10"]
        )
        /
        (
            df["sma_10"] +
            1e-12
        )
    )

    df["trend_20"] = (
        (
            df["current_price"] -
            df["sma_20"]
        )
        /
        (
            df["sma_20"] +
            1e-12
        )
    )

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    df["volatility"] = (
        df.groupby("symbol")[
            "return_1"
        ]
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

        if "total_depth" in df.columns:

            df["ofi_normalized"] = (
                df["ofi"] /
                (
                    df["total_depth"] +
                    1e-12
                )
            )

        else:

            df["ofi_normalized"] = (
                df["ofi"]
            )

    else:

        df["ofi_normalized"] = 0.0

    # --------------------------------------------------------
    # OBI agreement
    # --------------------------------------------------------

    df["obi_alignment"] = (
        df["obi_top20"] *
        df["obi_top50"]
    )

    # --------------------------------------------------------
    # Spread percentage
    # --------------------------------------------------------

    df["spread_pct"] = (
        df["spread"] /
        (
            df["current_price"] +
            1e-12
        )
    )

    return df


# ============================================================
# CREATE 0.40% TARGET
# ============================================================

def create_target(df):

    df = df.copy()

    print(
        "\n🎯 Creating 0.40% future-move target..."
    )

    # --------------------------------------------------------
    # Future maximum / minimum
    # --------------------------------------------------------

    future_max = []
    future_min = []

    for symbol, group in df.groupby(
        "symbol",
        sort=False
    ):

        prices = (
            group["current_price"]
            .values
        )

        max_values = np.full(
            len(prices),
            np.nan
        )

        min_values = np.full(
            len(prices),
            np.nan
        )

        for i in range(
            len(prices)
        ):

            start = i + 1

            end = min(
                i + 1 +
                FUTURE_HORIZON,
                len(prices)
            )

            if start >= end:
                continue

            future = prices[
                start:end
            ]

            max_values[i] = np.max(
                future
            )

            min_values[i] = np.min(
                future
            )

        future_max.extend(
            max_values
        )

        future_min.extend(
            min_values
        )

    # The above order is grouped order, therefore
    # recreate safely using groupby transform logic.
    df["future_max"] = (
        df.groupby("symbol")[
            "current_price"
        ]
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

    df["future_min"] = (
        df.groupby("symbol")[
            "current_price"
        ]
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

    # --------------------------------------------------------
    # Future move
    # --------------------------------------------------------

    df["future_up_move"] = (
        (
            df["future_max"] -
            df["current_price"]
        )
        /
        (
            df["current_price"] +
            1e-12
        )
    )

    df["future_down_move"] = (
        (
            df["current_price"] -
            df["future_min"]
        )
        /
        (
            df["current_price"] +
            1e-12
        )
    )

    # --------------------------------------------------------
    # Target
    #
    # 1 = meaningful bullish move >= 0.40%
    # 0 = no bullish move >= 0.40%
    #
    # Short opportunities are stored separately.
    # --------------------------------------------------------

    df["target"] = (
        df["future_up_move"] >=
        MIN_MOVE
    ).astype(int)

    # --------------------------------------------------------
    # Direction target
    #
    # 1  = Long
    # -1 = Short
    # 0  = No meaningful move
    # --------------------------------------------------------

    df["direction_target"] = 0

    long_condition = (
        (df["future_up_move"] >= MIN_MOVE) &
        (
            df["future_up_move"] >=
            df["future_down_move"]
        )
    )

    short_condition = (
        (df["future_down_move"] >= MIN_MOVE) &
        (
            df["future_down_move"] >
            df["future_up_move"]
        )
    )

    df.loc[
        long_condition,
        "direction_target"
    ] = 1

    df.loc[
        short_condition,
        "direction_target"
    ] = -1

    return df


# ============================================================
# PREPARE DATASET
# ============================================================

def prepare_dataset(df):

    features = [
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

    # Support alternate column names
    if (
        "top20_bid_sum" not in df.columns
    ):

        df["top20_bid_sum"] = (
            df["top20_bid_volume"]
        )

        df["top20_ask_sum"] = (
            df["top20_ask_volume"]
        )

    # Make numeric
    for col in features:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    missing = [
        col
        for col in features
        if col not in df.columns
    ]

    if missing:

        print(
            "\n❌ Missing training features:"
        )

        for col in missing:
            print(
                f"   - {col}"
            )

        return None, None

    # --------------------------------------------------------
    # Remove NaN
    # --------------------------------------------------------

    df = df.dropna(
        subset=
        features +
        [
            "target",
            "direction_target"
        ]
    ).copy()

    if len(df) < 100:

        print(
            f"\n❌ Only {len(df)} usable rows."
        )

        print(
            "Training ke liye zyada "
            "historical data collect karo."
        )

        return None, None

    X = df[
        features
    ]

    y = df[
        "target"
    ]

    return X, y


# ============================================================
# TIME SERIES SPLIT
# ============================================================

def time_split(
    X,
    y
):

    split_index = int(
        len(X) *
        (
            1 -
            TEST_SIZE
        )
    )

    X_train = X.iloc[
        :split_index
    ]

    X_test = X.iloc[
        split_index:
    ]

    y_train = y.iloc[
        :split_index
    ]

    y_test = y.iloc[
        split_index:
    ]

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model():

    print(
        "\n"
        + "=" * 60
    )

    print(
        "🚀 TRAINING 0.40% MICROSTRUCTURE MODEL"
    )

    print(
        "=" * 60
    )

    df = load_data()

    if df is None:
        return

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    df = create_features(
        df
    )

    if df is None:
        return

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    df = create_target(
        df
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    X, y = prepare_dataset(
        df
    )

    if X is None:
        return

    print(
        f"\n📊 Usable samples: "
        f"{len(X):,}"
    )

    print(
        "\n🎯 Target distribution:"
    )

    print(
        y.value_counts(
            normalize=True
        )
        .rename(
            {
                0: "NO_0.40%_UP_MOVE",
                1: "UP_MOVE_>=_0.40%"
            }
        )
        .mul(100)
        .round(2)
    )

    # --------------------------------------------------------
    # Time split
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = time_split(
        X,
        y
    )

    print(
        f"\nTraining samples: "
        f"{len(X_train):,}"
    )

    print(
        f"Testing samples : "
        f"{len(X_test):,}"
    )

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    model = XGBClassifier(

        n_estimators=300,

        learning_rate=0.03,

        max_depth=4,

        min_child_weight=5,

        subsample=0.80,

        colsample_bytree=0.80,

        gamma=0.10,

        reg_alpha=0.10,

        reg_lambda=1.0,

        objective="binary:logistic",

        eval_metric="logloss",

        random_state=RANDOM_STATE,

        n_jobs=-1
    )

    print(
        "\n🧠 Training XGBoost..."
    )

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "📊 MODEL VALIDATION"
    )

    print(
        "=" * 60
    )

    print(
        f"Accuracy  : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Precision : "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Recall    : "
        f"{recall * 100:.2f}%"
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    print(
        "Confusion Matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    # --------------------------------------------------------
    # Feature Importance
    # --------------------------------------------------------

    importance = pd.DataFrame({

        "feature":
            X.columns,

        "importance":
            model.feature_importances_

    })

    importance = (
        importance
        .sort_values(
            "importance",
            ascending=False
        )
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "🔬 FEATURE IMPORTANCE"
    )

    print(
        "=" * 60
    )

    print(
        importance.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save model package
    # --------------------------------------------------------

    model_package = {

        "model":
            model,

        "features":
            list(X.columns),

        "min_move":
            MIN_MOVE,

        "future_horizon":
            FUTURE_HORIZON,

        "training_rows":
            len(X_train),

        "test_rows":
            len(X_test),

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall
    }

    joblib.dump(
        model_package,
        MODEL_FILE
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        f"💾 Model saved:"
        f" {MODEL_FILE}"
    )

    print(
        "=" * 60
    )

    print(
        "\n✅ Training completed."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_model()
