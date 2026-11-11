from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CAT = ["pickup", "delivery", "equipment"]
NUM = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
       "distance", "weight", "market_index", "quote_signal"]
FEATURES = CAT + NUM + ["year", "month", "day", "dayofweek", "dayofyear", "weekofyear"]

def features(frame):
    x = frame.copy()
    dt = pd.to_datetime(x["date"], errors="coerce")
    x["year"], x["month"], x["day"] = dt.dt.year, dt.dt.month, dt.dt.day
    x["dayofweek"], x["dayofyear"] = dt.dt.dayofweek, dt.dt.dayofyear
    x["weekofyear"] = dt.dt.isocalendar().week.astype(float)
    for col in NUM:
        if col not in x: x[col] = np.nan
    x["weight"] = pd.to_numeric(x["weight"], errors="coerce")
    x.loc[x["weight"] <= 0, "weight"] = np.nan
    for col in ["market_index", "weight"]:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    for col in CAT:
        if col not in x: x[col] = "Unknown"
        x[col] = x[col].fillna("Unknown").astype(str)
    return x[FEATURES]

def main():
    train = pd.read_csv(DATA / "train_test.csv")
    dates = np.sort(pd.to_datetime(train.date).unique())
    cutoff = dates[int(len(dates) * 0.80)]
    mask = pd.to_datetime(train.date) < cutoff
    X, y = features(train), train.posted_rate.astype(float)
    model = CatBoostRegressor(iterations=1400, depth=8, learning_rate=0.045,
        loss_function="RMSE", eval_metric="RMSE", l2_leaf_reg=5,
        random_seed=42, allow_writing_files=False, verbose=200)
    model.fit(X.loc[mask], y.loc[mask], cat_features=CAT,
        eval_set=(X.loc[~mask], y.loc[~mask]), early_stopping_rounds=100)
    p = np.maximum(model.predict(X.loc[~mask]), 1.0)
    print("Chronological holdout:", pd.Timestamp(cutoff).date())
    print("MAE:", mean_absolute_error(y.loc[~mask], p))
    print("RMSE:", mean_squared_error(y.loc[~mask], p) ** 0.5)
    print("R2:", r2_score(y.loc[~mask], p))
    final = CatBoostRegressor(iterations=max(100, model.best_iteration_ + 1),
        depth=8, learning_rate=0.045, loss_function="RMSE", l2_leaf_reg=5,
        random_seed=42, allow_writing_files=False, verbose=False)
    final.fit(X, y, cat_features=CAT)
    final.save_model(ROOT / "catboost_model.cbm")
    val = pd.read_csv(DATA / "validation.csv")
    template = pd.read_csv(DATA / "validation_predictions_template.csv")
    pred = np.maximum(final.predict(features(val)), 1.0)
    mapping = pd.Series(pred, index=val.load_id.astype(str))
    template["predicted_rate"] = template.load_id.astype(str).map(mapping)
    if template.predicted_rate.isna().any(): raise ValueError("Unmatched validation load_id")
    template[["load_id", "predicted_rate"]].to_csv(ROOT / "validation_predictions.csv", index=False)
    dec = pd.read_csv(DATA / "december_chart_inputs.csv")
    dec["predicted_rate"] = np.maximum(final.predict(features(dec)), 1.0)
    dec.to_csv(DATA / "december_chart_inputs.csv", index=False)

if __name__ == "__main__": main()
