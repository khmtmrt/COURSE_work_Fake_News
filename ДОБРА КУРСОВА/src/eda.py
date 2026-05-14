"""
Exploratory Data Analysis - statistics and histograms (2b)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.config import OUTPUTS_PATH

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def get_basic_stats(df: pd.DataFrame) -> dict:
    """
    Get basic statistics about the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_samples': len(df),
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.to_dict()
    }
    return stats


def plot_label_distribution(df: pd.DataFrame, label_column: str = 'label', save: bool = True) -> None:
    """
    Plot distribution of labels.
    
    Args:
        df: Input DataFrame
        label_column: Name of the label column
        save: Whether to save the plot
    """
    plt.figure(figsize=(10, 6))
    df[label_column].value_counts().plot(kind='bar')
    plt.title('Label Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save:
        output_path = OUTPUTS_PATH / 'label_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    
    plt.show()


def plot_text_length_distribution(df: pd.DataFrame, text_column: str = 'text', save: bool = True) -> None:
    """
    Plot distribution of text lengths.
    
    Args:
        df: Input DataFrame
        text_column: Name of the text column
        save: Whether to save the plot
    """
    df['text_length'] = df[text_column].str.len()
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(df['text_length'], bins=50, edgecolor='black')
    plt.title('Text Length Distribution')
    plt.xlabel('Text Length (characters)')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.boxplot(df['text_length'])
    plt.title('Text Length Box Plot')
    plt.ylabel('Text Length (characters)')
    
    plt.tight_layout()
    
    if save:
        output_path = OUTPUTS_PATH / 'text_length_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    
    plt.show()


def generate_summary_report(df: pd.DataFrame, label_column: str = 'label') -> str:
    """
    Generate a text summary report.
    
    Args:
        df: Input DataFrame
        label_column: Name of the label column
        
    Returns:
        Summary report as string
    """
    report = f"""
    === Dataset Summary Report ===
    
    Total samples: {len(df)}
    Total features: {len(df.columns)}
    
    Label distribution:
    {df[label_column].value_counts().to_string()}
    
    Missing values:
    {df.isnull().sum().to_string()}
    
    """
    return report
