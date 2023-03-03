import os
import joblib
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler


# def get_scaler(condition, scaler_path=None):
#     if scaler_path is not None:
#         try:
#             scaler = joblib.load(scaler_path)
#             # print("- load scaler from", scaler_path)
#         except:
#             exit(f"error: {scaler_path} file not found")
#     else:
#         if os.path.exists(scaler_path):
#             exit(f"Scaler already existed: {scaler_path}")
#         scaler = RobustScaler(quantile_range=(0.1, 0.9))
#         scaler.fit(condition.copy(), len(condition.columns))
#         joblib.dump(scaler, open(scaler_path), 'wb')

#     return scaler


def scaler_transform(condition, scaler_path=None):
    scaler = get_scaler(condition, scaler_path)
    return pd.DataFrame(scaler.transform(condition),
                        index=condition.index,
                        columns=condition.columns)


def build_scaler(properties):
    scaler = RobustScaler(quantile_range=(0.1, 0.9))
    scaler.fit(properties, len(properties.columns))    
    return scaler


def get_scaler(property_list, save_folder,
               properties=None, rebuild=False):
    scaler_path = os.path.join(save_folder,
        f'scaler_{"-".join(property_list)}.pkl')
    
    if rebuild:
        scaler = build_scaler(properties)
        joblib.dump(scaler, open(scaler_path, 'wb'))
        return scaler
    
    try:
        scaler = joblib.load(scaler_path)
        return scaler
    except FileNotFoundError:
        exit(f'File not found: {scaler_path}')