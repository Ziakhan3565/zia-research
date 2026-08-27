# train_model.py
# ============================================================
# ZIA RESEARCH - ML TRAINING ENGINE
# Binance USD-M Futures compatible
#
# Pipeline:
#
# auto_collector.py
#        ↓
# market_data_log.csv
#        ↓
# train_model.py
#        ↓
# XGBoost
#        ↓
# xgboost_obi_model.pkl
#        ↓
# research_lab.py
#        ↓
# bot_engine.py
#        ↓
# FINAL SIGNAL
# ============================================================

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier


# ============================================================
# CONFIGURATION
# ============================================================

MARKET_DATA_FILE = "market_data_log.csv"

MODEL_FILE = "xgboost_obi_model.pkl"

METADATA_FILE = "ml_model_metadata.json"

MIN_ROWS = 150

FUTURE_STEPS = 5

TEST_SIZE = 0.20

RANDOM_STATE = 42


# ============================================================
# FEATURE LIST
# ============================================================
#
# IMPORTANT:
# Research Lab / prediction side should use THIS EXACT order.
#
# Do not manually reorder these features elsewhere.
# ============================================================

FEATURES: List[str] = [
    # Order book
    "top20_bid_sum",
    "top20_ask_sum",
    "obi_5",
    "obi_10",
    "obi_20",
    "obi_50",

    # Market microstructure
    "spread",
    "spread_pct",
    "bid_ask_ratio_20",
    "bid_ask_ratio_50",
    "top20_total_depth",
    "top50_total_depth",

    # Taker flow
    "taker_buy_volume",
    "taker_sell_volume",
    "taker_flow",
    "taker_flow_ratio",

    # Price / trend
    "price_return",
    "price_change",
    "sma_distance",
    "realized_volatility",

    # Research features
    "BOOK_IMB",
    "QUANT_IMPLY",
    "ADAPT_CONF",
    "BAYESIAN",
    "FOURIER_TREND",
]


# ============================================================
# MODEL CONFIG
# ============================================================

MODEL_PARAMS = {
    "n_estimators": 250,
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_child_weight": 2,
    "subsample": 0.90,
    "colsample_bytree": 0.90,
    "reg_alpha": 0.05,
    "reg_lambda": 1.50,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ============================================================
# SAFE NUMERIC
# ============================================================

def safe_numeric(
    series: pd.Series,
) -> pd.Series:
    """
    Convert a Series to numeric safely.
    """

    return pd.to_numeric(
        series,
        errors="coerce",
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_market_data(
    file_path: str = MARKET_DATA_FILE,
) -> Optional[pd.DataFrame]:
    """
    Load market_data_log.csv.
    """

    if not os.path.exists(file_path):

        print(
            f"❌ Market data file not found: {file_path}"
        )

        print(
            "   Run auto_collector.py first."
        )

        return None

    try:

        df = pd.read_csv(
            file_path
        )

    except Exception as e:

        print(
            f"❌ Could not read {file_path}: {e}"
        )

        return None

    if df.empty:

        print(
            "❌ Market data file is empty."
        )

        return None

    print(
        f"📂 Loaded {len(df):,} rows."
    )

    return df


# ============================================================
# REQUIRED BASE COLUMNS
# ============================================================

def ensure_base_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ensure important columns exist.

    Old collector compatibility is intentionally supported
    where possible.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Old OBI compatibility
    # --------------------------------------------------------

    if (
        "obi_20" not in df.columns
        and "obi_top20" in df.columns
    ):

        df["obi_20"] = df[
            "obi_top20"
        ]

    if (
        "obi_top20" not in df.columns
        and "obi_20" in df.columns
    ):

        df["obi_top20"] = df[
            "obi_20"
        ]

    # --------------------------------------------------------
    # Required numeric columns
    # --------------------------------------------------------

    defaults = {

        "top20_bid_sum": 0.0,

        "top20_ask_sum": 0.0,

        "obi_5": 0.0,

        "obi_10": 0.0,

        "obi_20": 0.0,

        "obi_50": 0.0,

        "spread": 0.0,

        "spread_pct": 0.0,

        "bid_ask_ratio_20": 1.0,

        "bid_ask_ratio_50": 1.0,

        "top20_total_depth": 0.0,

        "top50_total_depth": 0.0,

        "taker_buy_volume": 0.0,

        "taker_sell_volume": 0.0,

        "taker_flow": 0.0,

        "taker_flow_ratio": 0.0,
    }

    for column, default in defaults.items():

        if column not in df.columns:

            df[column] = default

    # --------------------------------------------------------
    # Required identifiers
    # --------------------------------------------------------

    if "symbol" not in df.columns:

        raise ValueError(
            "market_data_log.csv must contain 'symbol'."
        )

    if "current_price" not in df.columns:

        raise ValueError(
            "market_data_log.csv must contain "
            "'current_price'."
        )

    return df


# ============================================================
# SORT DATA
# ============================================================

def prepare_order(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sort data chronologically per symbol.
    """

    df = df.copy()

    if "timestamp" in df.columns:

        df["_timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
            utc=True,
        )

    else:

        df["_timestamp"] = pd.NaT

    df = df.sort_values(
        ["symbol", "_timestamp"],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    return df


# ============================================================
# PRICE FEATURES
# ============================================================

def add_price_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create price/trend/volatility features.
    """

    df = df.copy()

    df["current_price"] = safe_numeric(
        df["current_price"]
    )

    # --------------------------------------------------------
    # Price change
    # --------------------------------------------------------

    df["price_change"] = (
        df.groupby("symbol")[
            "current_price"
        ]
        .diff()
        .fillna(0.0)
    )

    # --------------------------------------------------------
    # Percentage return
    # --------------------------------------------------------

    df["price_return"] = (
        df.groupby("symbol")[
            "current_price"
        ]
        .pct_change()
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    # --------------------------------------------------------
    # SMA
    # --------------------------------------------------------

    df["sma_20"] = (
        df.groupby("symbol")[
            "current_price"
        ]
        .transform(
            lambda x:
            x.rolling(
                20,
                min_periods=1,
            ).mean()
        )
    )

    # --------------------------------------------------------
    # Price distance from SMA
    # --------------------------------------------------------

    df["sma_distance"] = (
        (
            df["current_price"]
            - df["sma_20"]
        )
        / (
            df["sma_20"]
            + 1e-12
        )
    )

    # --------------------------------------------------------
    # Realized volatility
    # --------------------------------------------------------

    df["realized_volatility"] = (
        df.groupby("symbol")[
            "price_return"
        ]
        .transform(
            lambda x:
            x.rolling(
                20,
                min_periods=2,
            ).std()
        )
        .fillna(0.0)
    )

    return df


# ============================================================
# MICROSTRUCTURE FEATURES
# ============================================================

def add_microstructure_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build Research Lab compatible microstructure features.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    numeric_columns = [
        "top20_bid_sum",
        "top20_ask_sum",
        "obi_5",
        "obi_10",
        "obi_20",
        "obi_50",
        "spread",
        "spread_pct",
        "bid_ask_ratio_20",
        "bid_ask_ratio_50",
        "top20_total_depth",
        "top50_total_depth",
        "taker_buy_volume",
        "taker_sell_volume",
        "taker_flow",
        "taker_flow_ratio",
    ]

    for column in numeric_columns:

        df[column] = safe_numeric(
            df[column]
        ).fillna(0.0)

    # --------------------------------------------------------
    # BOOK_IMB
    # --------------------------------------------------------

    df["BOOK_IMB"] = (
        (
            df["top20_bid_sum"]
            - df["top20_ask_sum"]
        )
        /
        (
            df["top20_bid_sum"]
            + df["top20_ask_sum"]
            + 1e-12
        )
    )

    # --------------------------------------------------------
    # QUANT_IMPLY
    # --------------------------------------------------------

    df["QUANT_IMPLY"] = np.clip(
        df["BOOK_IMB"] * 1.5,
        -1.0,
        1.0,
    )

    # --------------------------------------------------------
    # Adaptive confidence
    # --------------------------------------------------------

    fast_ma = (
        df.groupby("symbol")[
            "current_price"
        ]
        .transform(
            lambda x:
            x.rolling(
                3,
                min_periods=1,
            ).mean()
        )
    )

    slow_ma = (
        df.groupby("symbol")[
            "current_price"
        ]
        .transform(
            lambda x:
            x.rolling(
                10,
                min_periods=1,
            ).mean()
        )
    )

    volatility_price = (
        df["realized_volatility"]
        * df["current_price"]
    )

    df["ADAPT_CONF"] = np.clip(
        (
            fast_ma
            - slow_ma
        )
        /
        (
            volatility_price
            + 1e-12
        ),
        -1.0,
        1.0,
    )

    # --------------------------------------------------------
    # Bayesian
    # --------------------------------------------------------
    #
    # Prior is deliberately moderate.
    # We do NOT hard-code a huge directional bias.
    # --------------------------------------------------------

    prior = 0.50

    positive_likelihood = np.clip(
        0.50
        + (
            df["BOOK_IMB"]
            * 0.40
        ),
        0.05,
        0.95,
    )

    negative_likelihood = (
        1.0
        - positive_likelihood
    )

    posterior = (
        positive_likelihood
        * prior
    ) / (
        (
            positive_likelihood
            * prior
        )
        +
        (
            negative_likelihood
            * (1.0 - prior)
        )
        + 1e-12
    )

    df["BAYESIAN"] = np.clip(
        (
            posterior
            - 0.50
        )
        * 2.0,
        -1.0,
        1.0,
    )

    return df


# ============================================================
# FOURIER
# ============================================================

def calculate_fourier_series(
    prices: np.ndarray,
) -> np.ndarray:
    """
    Calculate a causal-ish rolling Fourier trend.

    Only the current rolling window is used.
    """

    length = len(prices)

    if length < 15:

        return np.zeros(
            length,
            dtype=float,
        )

    values = (
        np.asarray(
            prices,
            dtype=float,
        )
    )

    centered = (
        values
        - np.mean(values)
    )

    fft_values = np.fft.fft(
        centered
    )

    keep = max(
        1,
        int(
            len(fft_values)
            * 0.15
        ),
    )

    filtered = np.zeros_like(
        fft_values
    )

    filtered[:keep] = (
        fft_values[:keep]
    )

    filtered[-keep:] = (
        fft_values[-keep:]
    )

    curve = np.real(
        np.fft.ifft(
            filtered
        )
    )

    gradient = np.gradient(
        curve
    )

    return gradient


def add_fourier_feature(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add rolling Fourier trend per symbol.
    """

    df = df.copy()

    result = pd.Series(
        0.0,
        index=df.index,
        dtype=float,
    )

    for symbol, group in df.groupby(
        "symbol",
        sort=False,
    ):

        indexes = group.index

        prices = group[
            "current_price"
        ].values.astype(float)

        values = np.zeros(
            len(group),
            dtype=float,
        )

        # Rolling Fourier window.
        window = 30

        for i in range(
            len(prices)
        ):

            start = max(
                0,
                i - window + 1,
            )

            segment = prices[
                start:i + 1
            ]

            if len(segment) >= 15:

                fft_result = (
                    calculate_fourier_series(
                        segment
                    )
                )

                values[i] = (
                    fft_result[-1]
                )

            else:

                values[i] = 0.0

        result.loc[indexes] = values

    df["FOURIER_TREND"] = (
        result
    )

    return df


# ============================================================
# TARGET
# ============================================================

def create_target(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Future direction target.

    target = 1
        future price > current price

    target = 0
        future price <= current price

    IMPORTANT:
    Future price is used ONLY for target creation.
    It is NOT included in features.
    """

    df = df.copy()

    df["future_price"] = (
        df.groupby("symbol")[
            "current_price"
        ]
        .shift(
            -FUTURE_STEPS
        )
    )

    df = df.dropna(
        subset=[
            "future_price"
        ]
    ).copy()

    df["target"] = (
        df["future_price"]
        > df["current_price"]
    ).astype(int)

    return df


# ============================================================
# BUILD DATASET
# ============================================================

def build_training_dataset(
    df: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """
    Full feature engineering pipeline.
    """

    print(
        "\n🧠 Building ML research features..."
    )

    df = ensure_base_columns(
        df
    )

    df = prepare_order(
        df
    )

    df = add_price_features(
        df
    )

    df = add_microstructure_features(
        df
    )

    df = add_fourier_feature(
        df
    )

    df = create_target(
        df
    )

    # --------------------------------------------------------
    # Ensure all features exist
    # --------------------------------------------------------

    missing = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing:

        print(
            "❌ Missing ML features:"
        )

        for feature in missing:
            print(
                f"   - {feature}"
            )

        return None

    # --------------------------------------------------------
    # Numeric cleanup
    # --------------------------------------------------------

    for feature in FEATURES:

        df[feature] = safe_numeric(
            df[feature]
        )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = df.dropna(
        subset=FEATURES + [
            "target",
            "current_price",
            "symbol",
        ]
    ).copy()

    if len(df) < MIN_ROWS:

        print(
            f"❌ Not enough training rows: "
            f"{len(df)} / {MIN_ROWS}"
        )

        return None

    # --------------------------------------------------------
    # Feature clipping for stability
    # --------------------------------------------------------

    for feature in [
        "obi_5",
        "obi_10",
        "obi_20",
        "obi_50",
        "BOOK_IMB",
        "QUANT_IMPLY",
        "ADAPT_CONF",
        "BAYESIAN",
        "FOURIER_TREND",
        "taker_flow_ratio",
    ]:

        if feature in df.columns:

            if feature != "FOURIER_TREND":

                df[feature] = np.clip(
                    df[feature],
                    -10.0,
                    10.0,
                )

    return df


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def chronological_split(
    df: pd.DataFrame,
):
    """
    Time-series split.

    NO random train_test_split.

    Earlier data:
        TRAIN

    Later data:
        TEST
    """

    df = df.sort_values(
        ["_timestamp", "symbol"],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    split_index = int(
        len(df)
        * (1.0 - TEST_SIZE)
    )

    split_index = max(
        1,
        min(
            split_index,
            len(df) - 1,
        ),
    )

    train_df = df.iloc[
        :split_index
    ].copy()

    test_df = df.iloc[
        split_index:
    ].copy()

    return (
        train_df,
        test_df,
    )


# ============================================================
# MODEL TRAINING
# ============================================================

def train_model(
    train_df: pd.DataFrame,
) -> XGBClassifier:

    X_train = train_df[
        FEATURES
    ]

    y_train = train_df[
        "target"
    ]

    model = XGBClassifier(
        **MODEL_PARAMS
    )

    model.fit(
        X_train,
        y_train,
        verbose=False,
    )

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model: XGBClassifier,
    test_df: pd.DataFrame,
) -> float:

    X_test = test_df[
        FEATURES
    ]

    y_test = test_df[
        "target"
    ]

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print(
        "\n============================================================"
    )

    print(
        "📊 ML TEST RESULTS"
    )

    print(
        "============================================================"
    )

    print(
        f"Test rows : {len(test_df):,}"
    )

    print(
        f"Accuracy  : {accuracy * 100:.2f}%"
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    print(
        "Confusion Matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    print(
        "============================================================"
    )

    return float(
        accuracy
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def print_feature_importance(
    model: XGBClassifier,
):

    print(
        "\n🧠 FEATURE IMPORTANCE"
    )

    importance = (
        model.feature_importances_
    )

    rows = []

    for feature, value in zip(
        FEATURES,
        importance,
    ):

        rows.append(
            {
                "feature": feature,
                "importance": float(
                    value
                ),
            }
        )

    rows.sort(
        key=lambda x:
        x["importance"],
        reverse=True,
    )

    for row in rows:

        print(
            f"{row['feature']:25} "
            f"{row['importance']:.6f}"
        )


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model: XGBClassifier,
    accuracy: float,
    train_rows: int,
    test_rows: int,
):

    # --------------------------------------------------------
    # Save actual model
    # --------------------------------------------------------

    joblib.dump(
        model,
        MODEL_FILE,
    )

    print(
        f"\n💾 Model saved: {MODEL_FILE}"
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {

        "model_file":
            MODEL_FILE,

        "model_type":
            "XGBClassifier",

        "market":
            "BINANCE_USDM_FUTURES",

        "target":
            "future_price_5_steps_up",

        "future_steps":
            FUTURE_STEPS,

        "features":
            FEATURES,

        "feature_count":
            len(FEATURES),

        "accuracy":
            accuracy,

        "train_rows":
            train_rows,

        "test_rows":
            test_rows,

        "model_params":
            MODEL_PARAMS,

    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    print(
        f"🧾 Metadata saved: {METADATA_FILE}"
    )


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train_trend_aligned_model():

    print(
        "\n"
        "============================================================"
    )

    print(
        "🚀 ZIA RESEARCH ML TRAINING ENGINE"
    )

    print(
        "============================================================"
    )

    print(
        "Market: Binance USDⓈ-M Futures"
    )

    print(
        f"Target horizon: {FUTURE_STEPS} collected steps"
    )

    print(
        f"Features: {len(FEATURES)}"
    )

    print(
        "============================================================"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_market_data()

    if df is None:
        return False

    # --------------------------------------------------------
    # Build dataset
    # --------------------------------------------------------

    dataset = build_training_dataset(
        df
    )

    if dataset is None:
        return False

    print(
        f"\n✅ Final dataset: "
        f"{len(dataset):,} rows"
    )

    # --------------------------------------------------------
    # Target balance
    # --------------------------------------------------------

    target_counts = (
        dataset["target"]
        .value_counts()
        .sort_index()
    )

    print(
        "\n🎯 Target distribution:"
    )

    for target, count in target_counts.items():

        percentage = (
            count
            / len(dataset)
            * 100
        )

        label = (
            "DOWN / FLAT"
            if int(target) == 0
            else "UP"
        )

        print(
            f"   {label}: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )

    # --------------------------------------------------------
    # Need both classes
    # --------------------------------------------------------

    if dataset["target"].nunique() < 2:

        print(
            "❌ Training requires both target classes."
        )

        return False

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    train_df, test_df = (
        chronological_split(
            dataset
        )
    )

    print(
        f"\n📚 Train rows: {len(train_df):,}"
    )

    print(
        f"🧪 Test rows : {len(test_df):,}"
    )

    # --------------------------------------------------------
    # Verify both classes
    # --------------------------------------------------------

    if train_df["target"].nunique() < 2:

        print(
            "❌ Training set contains only one class."
        )

        return False

    if test_df["target"].nunique() < 2:

        print(
            "⚠️ Test set contains only one class."
        )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print(
        "\n🤖 Training XGBoost..."
    )

    model = train_model(
        train_df
    )

    print(
        "✅ XGBoost training complete."
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    accuracy = evaluate_model(
        model,
        test_df,
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    print_feature_importance(
        model
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_model(
        model=model,
        accuracy=accuracy,
        train_rows=len(train_df),
        test_rows=len(test_df),
    )

    print(
        "\n============================================================"
    )

    print(
        "✅ ML TRAINING FINISHED"
    )

    print(
        f"🎯 Holdout Accuracy: {accuracy * 100:.2f}%"
    )

    print(
        f"💾 Model: {MODEL_FILE}"
    )

    print(
        f"🧾 Metadata: {METADATA_FILE}"
    )

    print(
        "============================================================\n"
    )

    return True


# ============================================================
# MODEL PREDICTION HELPER
# ============================================================

def load_trained_model():

    if not os.path.exists(
        MODEL_FILE
    ):

        raise FileNotFoundError(
            f"{MODEL_FILE} not found. "
            "Train the model first."
        )

    return joblib.load(
        MODEL_FILE
    )


def predict_direction(
    feature_data: Dict[str, float],
) -> Dict[str, float]:
    """
    Prediction helper for research_lab.py.

    feature_data must contain the FEATURES listed above.

    Returns:

        direction:
            LONG / SHORT

        probability:
            0-1 probability of UP

        confidence:
            0-100
    """

    model = load_trained_model()

    row = {}

    for feature in FEATURES:

        value = feature_data.get(
            feature,
            0.0,
        )

        try:

            value = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            value = 0.0

        row[feature] = value

    X = pd.DataFrame(
        [row],
        columns=FEATURES,
    )

    probability_up = float(
        model.predict_proba(
            X
        )[0][1]
    )

    if probability_up >= 0.50:

        direction = "LONG"

        confidence = (
            probability_up
            * 100.0
        )

    else:

        direction = "SHORT"

        confidence = (
            (1.0 - probability_up)
            * 100.0
        )

    return {

        "direction":
            direction,

        "probability_up":
            probability_up,

        "confidence":
            confidence,
    }


# ============================================================
# TEST PREDICTION
# ============================================================

def test_saved_model():

    print(
        "\n🔎 Testing saved ML model..."
    )

    try:

        model = load_trained_model()

    except Exception as e:

        print(
            f"❌ {e}"
        )

        return

    print(
        f"✅ Loaded {MODEL_FILE}"
    )

    print(
        f"Expected features: {len(FEATURES)}"
    )

    print(
        "Feature order:"
    )

    for index, feature in enumerate(
        FEATURES,
        start=1,
    ):

        print(
            f"  {index:02d}. {feature}"
        )

    print(
        "\n✅ Model is ready for Research Lab prediction."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    train_trend_aligned_model()
