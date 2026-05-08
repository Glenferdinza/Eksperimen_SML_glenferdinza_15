"""
automate_glenferdinza_15.py
Script otomasi preprocessing data latency jaringan.
Konversi dari proses eksperimen di notebook ke script Python.

Tahapan:
1. Data Loading — membaca raw CSV
2. Data Cleaning — handle missing values, duplicates, outliers
3. Feature Engineering — lagging features, rolling stats, encoding
4. Normalisasi — StandardScaler pada fitur numerik
5. Split Data — train/test split
6. Export — simpan dataset yang sudah siap dilatih
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import sys
import warnings

warnings.filterwarnings("ignore")


def load_data(filepath: str) -> pd.DataFrame:
    """
    Tahap 1: Memuat dataset mentah dari CSV.
    
    Args:
        filepath: path ke file CSV mentah
    
    Returns:
        DataFrame mentah
    """
    print("=" * 60)
    print("TAHAP 1: DATA LOADING")
    print("=" * 60)
    
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tahap 2: Data Cleaning
    - Handle missing values
    - Remove duplicates
    - Handle outliers (IQR method)
    - Filter invalid data
    
    Args:
        df: DataFrame mentah
    
    Returns:
        DataFrame yang sudah dibersihkan
    """
    print("\n" + "=" * 60)
    print("TAHAP 2: DATA CLEANING")
    print("=" * 60)
    
    initial_rows = len(df)
    
    # 2.1 Handle missing values
    missing = df.isnull().sum()
    print(f"\nMissing values:\n{missing[missing > 0]}")
    if missing.sum() > 0:
        # Untuk numerik: isi dengan median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"  Filled {col} nulls with median={median_val}")
    else:
        print("  No missing values found.")
    
    # 2.2 Remove duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df = df.drop_duplicates()
        print(f"\nRemoved {dup_count} duplicate rows")
    else:
        print(f"\nNo duplicates found.")
    
    # 2.3 Filter invalid data (latency harus positif)
    if "latency_ms" in df.columns:
        invalid = (df["latency_ms"] <= 0).sum()
        if invalid > 0:
            df = df[df["latency_ms"] > 0]
            print(f"Removed {invalid} rows with invalid latency (<= 0)")
    
    # 2.4 Outlier detection & handling menggunakan IQR
    if "latency_ms" in df.columns:
        Q1 = df["latency_ms"].quantile(0.25)
        Q3 = df["latency_ms"].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        outliers = ((df["latency_ms"] < lower) | (df["latency_ms"] > upper)).sum()
        # Clip outliers instead of removing
        df["latency_ms"] = df["latency_ms"].clip(lower=max(0, lower), upper=upper)
        print(f"\nOutlier handling (IQR): clipped {outliers} values "
              f"to range [{max(0, lower):.2f}, {upper:.2f}]")
    
    final_rows = len(df)
    print(f"\nCleaning complete: {initial_rows} -> {final_rows} rows "
          f"(removed {initial_rows - final_rows})")
    
    return df.reset_index(drop=True)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tahap 3: Feature Engineering
    - Parse timestamp
    - Create lag features
    - Rolling statistics
    - Cyclical encoding untuk jam dan hari
    
    Args:
        df: DataFrame yang sudah dibersihkan
    
    Returns:
        DataFrame dengan fitur tambahan
    """
    print("\n" + "=" * 60)
    print("TAHAP 3: FEATURE ENGINEERING")
    print("=" * 60)
    
    # 3.1 Parse timestamp jika ada
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        print("  Parsed and sorted by timestamp")
    
    # 3.2 Encode server (categorical -> numeric)
    if "server" in df.columns:
        le = LabelEncoder()
        df["server_encoded"] = le.fit_transform(df["server"])
        print(f"  Encoded 'server': {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # 3.3 Lag features (t-1, t-2, t-3)
    for lag in [1, 2, 3]:
        df[f"latency_lag_{lag}"] = df["latency_ms"].shift(lag)
        print(f"  Created latency_lag_{lag}")
    
    # 3.4 Rolling statistics (window=5)
    df["latency_rolling_mean_5"] = df["latency_ms"].rolling(window=5).mean()
    df["latency_rolling_std_5"] = df["latency_ms"].rolling(window=5).std()
    df["jitter_rolling_mean_5"] = df["jitter"].rolling(window=5).mean()
    print("  Created rolling mean & std (window=5)")
    
    # 3.5 Cyclical encoding untuk hour dan day_of_week
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    print("  Created cyclical encoding for hour and day_of_week")
    
    # 3.6 Interaksi fitur
    df["latency_jitter_ratio"] = df["latency_ms"] / (df["jitter"] + 1e-6)
    print("  Created latency_jitter_ratio")
    
    # Drop NaN dari lag/rolling
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"\n  Dropped {before - len(df)} rows with NaN from lag/rolling features")
    print(f"  Final shape: {df.shape}")
    
    return df


def normalize_and_split(df: pd.DataFrame, test_size: float = 0.2, 
                         random_state: int = 42) -> dict:
    """
    Tahap 4 & 5: Normalisasi dan Split Data
    
    Args:
        df: DataFrame dengan fitur lengkap
        test_size: proporsi data test
        random_state: seed untuk reprodusibilitas
    
    Returns:
        dict berisi X_train, X_test, y_train, y_test, scaler, feature_names
    """
    print("\n" + "=" * 60)
    print("TAHAP 4: NORMALISASI & SPLIT DATA")
    print("=" * 60)
    
    # Kolom yang tidak dipakai sebagai fitur
    drop_cols = ["timestamp", "server", "latency_ms"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    
    target = "latency_ms"
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols].values
    y = df[target].values
    
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Target: {target}")
    
    # Normalisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"\nApplied StandardScaler to features")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": feature_cols
    }


def export_preprocessed(df: pd.DataFrame, output_path: str) -> str:
    """
    Tahap 6: Export dataset yang sudah dipreprocess.
    
    Args:
        df: DataFrame hasil preprocessing
        output_path: path file output
    
    Returns:
        path file yang disimpan
    """
    print("\n" + "=" * 60)
    print("TAHAP 5: EXPORT DATA")
    print("=" * 60)
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", 
                exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to: {output_path}")
    print(f"Shape: {df.shape}")
    
    return output_path


def run_preprocessing(raw_path: str, output_path: str) -> dict:
    """
    Menjalankan seluruh pipeline preprocessing.
    
    Args:
        raw_path: path ke dataset mentah
        output_path: path untuk menyimpan hasil preprocessing
    
    Returns:
        dict berisi data splits dan metadata
    """
    print(">>> NETPREDICT - AUTOMATED PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # 1. Load
    df = load_data(raw_path)
    
    # 2. Clean
    df = clean_data(df)
    
    # 3. Feature Engineering
    df = feature_engineering(df)
    
    # 4-5. Normalize & Split
    splits = normalize_and_split(df)
    
    # 6. Export
    export_preprocessed(df, output_path)
    
    print("\n" + "=" * 60)
    print("[OK] PREPROCESSING COMPLETE!")
    print("=" * 60)
    
    return splits


if __name__ == "__main__":
    raw_path = sys.argv[1] if len(sys.argv) > 1 else "latency_raw.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "preprocessing/latency_preprocessing.csv"
    
    run_preprocessing(raw_path, output_path)
