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


def scaler_transform(df_property, scaler):
    return pd.DataFrame(scaler.transform(df_property),
                        index=df_property.index,
                        columns=df_property.columns)


# def get_scaler(scaler_folder, prop_name):
#     """get scaler to normalize properties
    
#     Args:
#         scaler_folder (str): the folder where the scaler file is stored
#         prop_name (List[str]): a list of property names
    
#     Returns:
#         scaler (RobustScaler): the scaler to normalize properties
#     """
#     scaler_path = os.path.join(scaler_folder,
#         f'scaler_{"-".join(prop_name)}.pkl')
#     try:
#         return joblib.load(scaler_path)
#     except FileNotFoundError:
#         exit(f'Scaler not exists: {scaler_path}')


def build_scaler(prop_val, prop_name):
    scaler = RobustScaler(quantile_range=(25, 75))
    scaler.fit(prop_val, len(prop_name))
    return scaler
    
    
def save_scaler(scaler, scaler_path):
    joblib.dump(scaler, open(scaler_path, 'wb'))


"""old version
"""
# def build_scaler(properties):
#     scaler = RobustScaler(quantile_range=(0.1, 0.9))
#     scaler.fit(properties, len(properties.columns))
#     return scaler


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
        
