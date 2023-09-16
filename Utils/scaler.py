import os
import joblib
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler


def scaler_transform(df_property, scaler):
    return pd.DataFrame(scaler.transform(df_property),
                        index=df_property.index,
                        columns=df_property.columns)


def build_scaler(prop_val, prop_name):
    scaler = RobustScaler(quantile_range=(25, 75))
    scaler.fit(prop_val, len(prop_name))
    return scaler
    
    
def save_scaler(scaler, scaler_path):
    joblib.dump(scaler, open(scaler_path, 'wb'))


def get_scaler(save_folder, df_property=None, rebuild=False):
    scaler_path = os.path.join(save_folder,
        f'scaler_{"-".join(df_property.columns)}.pkl')
    
    if rebuild:
        assert df_property is not None
        scaler = build_scaler(df_property, df_property.columns)
        joblib.dump(scaler, open(scaler_path, 'wb'))
        return scaler
    try:
        scaler = joblib.load(scaler_path)
        return scaler
    except FileNotFoundError:
        exit(f'File not found: {scaler_path}')
        
