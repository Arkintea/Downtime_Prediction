# data_preprocessing/preprocessor.py
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from typing import List, Tuple, Optional
import yaml
import warnings
warnings.filterwarnings("ignore")

try:
    with open("config/config.yaml", "r") as file:
        CONFIG = yaml.safe_load(file)
    ENCODER_DIR = CONFIG["paths"]["encoder_dir"]
except FileNotFoundError:
    ENCODER_DIR = "encoders/"

class Preprocessor:

    @staticmethod
    def parse_mixed_date(date_str: str) -> pd.Timestamp:
        if pd.isna(date_str) or date_str == '' or date_str is None:
            return pd.NaT
        
        date_str = str(date_str).strip()
        
        if date_str.lower() in ['nan', 'none', 'null']:
            return pd.NaT
            
        try:
            if ' ' in date_str and ':' in date_str:
                return pd.to_datetime(date_str.split(' ')[0], errors='coerce')
            if '/' in date_str:
                return pd.to_datetime(date_str, format='%m/%d/%Y', errors='coerce')
            if '-' in date_str:
                parts = date_str.split('-')
                if len(parts) >= 3 and len(parts[0]) == 4:
                    return pd.to_datetime(date_str, format='%Y-%m-%d', errors='coerce')
                return pd.to_datetime(date_str, format='%d-%m-%Y', errors='coerce')
        except Exception:
            pass
            
        return pd.to_datetime(date_str, errors='coerce')

    @staticmethod
    def reconstruct_datetime(df: pd.DataFrame) -> pd.DataFrame:
        if 'Date' not in df.columns or 'Timestamp' not in df.columns:
            return df
            
        df_copy = df.copy()
        
        try:
            df_copy['parsed_date'] = df_copy['Date'].astype(str).apply(Preprocessor.parse_mixed_date)
            df_copy['parsed_time'] = pd.to_datetime(
                df_copy['Timestamp'].astype(str), 
                format='%H:%M:%S', 
                errors='coerce'
            ).dt.time
            
            valid_dates = ~df_copy['parsed_date'].isna()
            valid_times = ~df_copy['parsed_time'].isna()
            
            if valid_dates.any() and valid_times.any():
                df_copy['datetime'] = pd.to_datetime(
                    df_copy['parsed_date'].astype(str) + ' ' + df_copy['parsed_time'].astype(str), 
                    errors='coerce'
                )
                
                valid_datetime = ~df_copy['datetime'].isna()
                if valid_datetime.any():
                    df_copy = df_copy[valid_datetime]
                    df_copy.set_index('datetime', inplace=True)
                    
        except Exception as e:
            print(f"Error in datetime reconstruction: {e}")
        
        columns_to_drop = ['Date', 'Timestamp', 'parsed_date', 'parsed_time']
        df_copy.drop(columns=[col for col in columns_to_drop if col in df_copy.columns], 
                    inplace=True, errors='ignore')
        
        return df_copy

    @staticmethod
    def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        
        if not isinstance(df_copy.index, pd.DatetimeIndex):
            df_copy['hour'] = 12
            df_copy['dayofweek'] = 1
            df_copy['month'] = 1
            df_copy['is_weekend'] = 0
        else:
            try:
                df_copy['hour'] = df_copy.index.hour
                df_copy['dayofweek'] = df_copy.index.dayofweek
                df_copy['month'] = df_copy.index.month
                df_copy['is_weekend'] = df_copy['dayofweek'].isin([5, 6]).astype(int)
            except Exception:
                df_copy['hour'] = 12
                df_copy['dayofweek'] = 1
                df_copy['month'] = 1
                df_copy['is_weekend'] = 0
        
        return df_copy

    @staticmethod
    def map_downtime_groups(df: pd.DataFrame) -> pd.DataFrame:
        if 'Downtime' not in df.columns:
            return df
            
        downtime_group_map = {
            'Sensor_Fault': 'Machine_Issue',
            'Overheating': 'Machine_Issue',
            'Tool_Wear': 'Machine_Issue',
            'Machine_Failure': 'Machine_Issue',
            'Unplanned_Stop': 'Machine_Issue',
            'Operator_Error': 'Operations',
            'Maintenance': 'Operations',
            'Scheduled_Inspection': 'Operations',
            'Planned_Stop': 'Operations',
            'Quality_Check': 'Operations',
            'Network_Issue': 'Infrastructure',
            'Power_Failure': 'Infrastructure',
            'Software_Glitch': 'Infrastructure',
            'Material_Shortage': 'Operations',
            'No_Machine_Failure': 'Normal'
        }
        
        df_copy = df.copy()
        df_copy['Downtime_Group'] = df_copy['Downtime'].map(downtime_group_map)
        
        unmapped_mask = df_copy['Downtime_Group'].isna()
        if unmapped_mask.any():
            df_copy['Downtime_Group'] = df_copy['Downtime_Group'].fillna('Unknown')
        
        return df_copy

    @staticmethod
    def label_encode_target(df: pd.DataFrame, target_col: str = "Downtime_Group", 
                           save_path: Optional[str] = None) -> Tuple[pd.DataFrame, LabelEncoder]:
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
            
        df_copy = df.copy()
        
        missing_count = df_copy[target_col].isna().sum()
        if missing_count > 0:
            df_copy[target_col] = df_copy[target_col].fillna('Unknown')
        
        le = LabelEncoder()
        df_copy[target_col] = le.fit_transform(df_copy[target_col].astype(str))
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(le, save_path)
        
        return df_copy, le

    @staticmethod
    def create_future_target(df: pd.DataFrame, target_col: str = "Downtime_Group", 
                           shift_periods: int = 10) -> pd.DataFrame:
        if target_col not in df.columns:
            return df
            
        df_copy = df.copy()
        df_copy['Future_Downtime_Label'] = df_copy[target_col].shift(-shift_periods)
        
        valid_future_labels = ~df_copy['Future_Downtime_Label'].isna()
        
        if not valid_future_labels.any():
            return df_copy
        
        df_copy = df_copy[valid_future_labels].copy()
        df_copy['Future_Downtime_Label'] = df_copy['Future_Downtime_Label'].astype(int)
        
        return df_copy

    @staticmethod
    def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        
        sensor_cols = [
            'Hydraulic_Pressure(bar)', 'Coolant_Pressure(bar)', 'Torque(Nm)',
            'Spindle_Speed(RPM)', 'Voltage(volts)', 'Coolant_Temperature',
            'Air_System_Pressure(bar)', 'Hydraulic_Oil_Temperature(?C)',
            'Spindle_Bearing_Temperature(?C)', 'Spindle_Vibration(?m)',
            'Tool_Vibration(?m)', 'Cutting(kN)'
        ]
        
        existing_sensor_cols = [col for col in sensor_cols if col in df_copy.columns]
        
        if not existing_sensor_cols:
            return df_copy
        
        for col in existing_sensor_cols:
            try:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                median_val = df_copy[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                df_copy[col] = df_copy[col].fillna(median_val)
                
                df_copy[f'{col}_lag1'] = df_copy[col].shift(1).fillna(median_val)
                df_copy[f'{col}_mean5'] = df_copy[col].rolling(window=5, min_periods=1).mean()
                df_copy[f'{col}_std5'] = df_copy[col].rolling(window=5, min_periods=1).std().fillna(0)
                
            except Exception as e:
                print(f"Error processing temporal features for {col}: {e}")

        df_copy = df_copy.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df_copy

    @staticmethod
    def one_hot_encode_features(df: pd.DataFrame, columns: List[str], 
                               save_path: Optional[str] = None) -> pd.DataFrame:
        df_copy = df.copy()
        
        existing_columns = [col for col in columns if col in df_copy.columns]
        
        if not existing_columns:
            return df_copy
        
        try:
            for col in existing_columns:
                df_copy[col] = df_copy[col].astype(str)
            
            df_copy = pd.get_dummies(df_copy, columns=existing_columns, drop_first=False)
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                joblib.dump(df_copy.columns.tolist(), save_path)
                
        except Exception as e:
            print(f"Error in one-hot encoding: {e}")
            raise e
        
        return df_copy

    @staticmethod
    def drop_unused_columns(df: pd.DataFrame, columns_to_drop: List[str]) -> pd.DataFrame:
        df_copy = df.copy()
        
        existing_columns = [col for col in columns_to_drop if col in df_copy.columns]
        
        if existing_columns:
            df_copy.drop(columns=existing_columns, inplace=True)
        
        return df_copy
    
    @staticmethod
    def align_one_hot_columns(df: pd.DataFrame, reference_columns: List[str]) -> pd.DataFrame:
        df_copy = df.copy()
        
        for col in reference_columns:
            if col not in df_copy.columns:
                df_copy[col] = 0
            
        extra_columns = [col for col in df_copy.columns if col not in reference_columns]
        if extra_columns:
            df_copy.drop(columns=extra_columns, inplace=True)
            
        df_copy = df_copy[reference_columns]
        
        return df_copy