import joblib
import pandas as pd

from sklearn.preprocessing import RobustScaler, StandardScaler


def get_scaler(condition, scaler_path=None):
    if not scaler_path:
        try:
            scaler = joblib.load(scaler_path)
            print("- load scaler from", scaler_path)
        except:
            exit(f"error: {scaler_path} file not found")
    else:
        scaler = RobustScaler(quantile_range=(0.1, 0.9))
        scaler.fit(condition.copy(), len(condition.columns))
        joblib.dump(scaler, open(scaler_path), 'wb')

    return scaler


def scaler_transform(condition, scaler):
    """ scaler transformation """
    condition = condition.to_numpy()
    condition = scaler.transform(condition)
    return pd.DataFrame(condition, columns=condition.columns)