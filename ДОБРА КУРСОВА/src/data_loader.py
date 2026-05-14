"""
Data loading and dataset merging (2a)
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a single dataset from CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with loaded data
    """
    df = pd.read_csv(file_path)
    return df


def merge_datasets(datasets: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge multiple datasets into one.
    
    Args:
        datasets: List of DataFrames to merge
        
    Returns:
        Merged DataFrame
    """
    return pd.concat(datasets, ignore_index=True)


def save_processed_data(df: pd.DataFrame, filename: str) -> None:
    """
    Save processed data to CSV.
    
    Args:
        df: DataFrame to save
        filename: Output filename
    """
    output_path = PROCESSED_DATA_PATH / filename
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")


def load_all_datasets() -> pd.DataFrame:
    """
    Load and merge all datasets from raw data folder.
    
    Returns:
        Merged DataFrame
    """
    csv_files = list(RAW_DATA_PATH.glob("*.csv"))
    datasets = [load_dataset(str(file)) for file in csv_files]
    
    if not datasets:
        raise FileNotFoundError(f"No CSV files found in {RAW_DATA_PATH}")
    
    return merge_datasets(datasets)
