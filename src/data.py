"""Dataset loading for the real manufacturing data."""
import pandas as pd

DATA_PATH = "small_custom_manufacturing_dataset_500.csv"
NUMERIC = ["temperature_C", "pressure_bar", "vibration_mm_s",
           "rotational_speed_rpm", "production_speed_units_hr", "tool_wear_min"]
CATEGORICAL = ["machine_id"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "defect"
UNITS = {"temperature_C": "°C", "pressure_bar": "bar", "vibration_mm_s": "mm/s",
         "rotational_speed_rpm": "rpm", "production_speed_units_hr": "units/hr",
         "tool_wear_min": "min"}


def load(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


# Label-free reference stats (so engineered features need no fitting)
_ref = load()[NUMERIC]
_MU, _SD = _ref.mean(), _ref.std()
ENGINEERED = [f"absdev_{c}" for c in NUMERIC] + ["heat_x_wear"]


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Two-sided deviation from typical operation + heat/wear interaction."""
    X = X.copy()
    for c in NUMERIC:
        X[f"absdev_{c}"] = ((X[c] - _MU[c]) / _SD[c]).abs()
    X["heat_x_wear"] = (((X["temperature_C"] - _MU["temperature_C"]) / _SD["temperature_C"]).clip(lower=0)
                        * ((X["tool_wear_min"] - _MU["tool_wear_min"]) / _SD["tool_wear_min"]).clip(lower=0))
    return X
