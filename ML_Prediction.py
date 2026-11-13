"""Box office revenue predicto.

Predicts a film's domestic revenue from opening theaters, distributor, rating,
and genres. The target is log-transformed because revenue is heavily skewed.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor

CSV_PATH = "boxoffice.csv"
RANDOM_STATE = 42
SHOW_PLOTS = True

# Set to True if release_days is only known AFTER a film is out. It would then
# leak information if you want to predict revenue before release.
DROP_RELEASE_DAYS = False


def to_number(series):
    """Turn strings like '$1,234,567' or '3,456' into numbers (bad values -> NaN)."""
    cleaned = series.astype(str).str.replace(r"[^0-9.]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def report(name, y_true_log, y_pred_log):
    """Print errors on the log scale and converted back to dollars."""
    mae_log = mean_absolute_error(y_true_log, y_pred_log)
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    r2 = r2_score(y_true_log, y_pred_log)
    mae_dollars = mean_absolute_error(np.expm1(y_true_log), np.expm1(y_pred_log))
    print(
        f"{name:<22} MAE(log)={mae_log:.3f}  RMSE(log)={rmse_log:.3f}  "
        f"R2={r2:.3f}  typical error factor ~{np.exp(mae_log):.2f}x  "
        f"MAE=${mae_dollars:,.0f}"
    )


def main():
    # --- 1. Load and inspect ---
    df = pd.read_csv(CSV_PATH, encoding="latin-1")
    print(df.head())
    print("Shape:", df.shape)
    df.info()

    # --- 2. Clean ---
    # world_revenue and opening_revenue are only known after release and are
    # tightly tied to the target, so using them would be data leakage.
    # budget is dropped because it has many missing values.
    to_drop = ["world_revenue", "opening_revenue", "budget"]
    if DROP_RELEASE_DAYS:
        to_drop.append("release_days")
    df = df.drop(columns=to_drop, errors="ignore")

    numeric_cols = [c for c in ["domestic_revenue", "opening_theaters", "release_days"]
                    if c in df.columns]
    for col in numeric_cols:
        df[col] = to_number(df[col])

    # Fill missing categoricals with the most common value.
    for col in ["MPAA", "genres"]:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Drop rows still missing values, AFTER the numeric conversion so that any
    # values turned into NaN by the conversion are also removed.
    df = df.dropna().reset_index(drop=True)
    print("Rows after cleaning:", len(df))

    # --- 3. Target and log transform ---
    # log1p handles zeros safely (log(1 + x)). Convert back with expm1.
    y = np.log1p(df["domestic_revenue"])
    X = df.drop(columns=["title", "domestic_revenue"])
    X["opening_theaters"] = np.log1p(X["opening_theaters"])
    if "release_days" in X.columns:
        X["release_days"] = np.log1p(X["release_days"])

    if SHOW_PLOTS:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.histplot(df["domestic_revenue"], kde=True, ax=axes[0])
        axes[0].set_title("Domestic revenue (raw): heavily skewed")
        sns.histplot(y, kde=True, ax=axes[1])
        axes[1].set_title("Domestic revenue (log1p): much more symmetric")
        plt.tight_layout()
        plt.show()

    # --- 4. Split first, so all preprocessing is learned from training data only ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # --- 5. Preprocessing + model in one pipeline ---
    numeric_features = [c for c in ["opening_theaters", "release_days"] if c in X.columns]
    preprocess = ColumnTransformer(
        [
            # Genres -> one 0/1 column per genre. min_df=0.05 drops genres that
            # appear in under 5% of films. The token pattern keeps 'Sci-Fi' whole.
            ("genres", CountVectorizer(binary=True, min_df=0.05,
                                       token_pattern=r"[A-Za-z\-]+"), "genres"),
            # Categories -> integer codes. Fine for tree models; unseen values -> -1.
            ("cats", OrdinalEncoder(handle_unknown="use_encoded_value",
                                    unknown_value=-1), ["distributor", "MPAA"]),
            ("nums", "passthrough", numeric_features),
        ]
    )  # No feature scaling: tree models don't need it.

    model = Pipeline(
        [("prep", preprocess), ("xgb", XGBRegressor(random_state=RANDOM_STATE))]
    )

    # --- 6. Compare against a baseline, with cross-validation ---
    baseline = DummyRegressor(strategy="median").fit(X_train, y_train)

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_mae = -cross_val_score(model, X_train, y_train, cv=cv,
                              scoring="neg_mean_absolute_error")
    print(f"\n5-fold CV MAE (log): {cv_mae.mean():.3f} +/- {cv_mae.std():.3f}")

    model.fit(X_train, y_train)

    print("\n--- Results ---")
    report("Baseline (median)", y_test, baseline.predict(X_test))
    report("XGBoost (train)", y_train, model.predict(X_train))
    report("XGBoost (test)", y_test, model.predict(X_test))

    # --- 7. What drives the predictions? ---
    names = model.named_steps["prep"].get_feature_names_out()
    importances = pd.Series(model.named_steps["xgb"].feature_importances_, index=names)
    print("\nTop 10 features:")
    print(importances.sort_values(ascending=False).head(10))


if __name__ == "__main__":
    main()