"""
Model evaluation with metrics and confusion matrix (2e)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, Any
from src.config import OUTPUTS_PATH


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }
    
    return metrics


def print_evaluation_results(metrics: Dict[str, float]) -> None:
    """
    Print evaluation metrics.
    
    Args:
        metrics: Dictionary with metrics
    """
    print("\n=== Evaluation Results ===")
    for metric_name, value in metrics.items():
        print(f"{metric_name.capitalize()}: {value:.4f}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list = None,
    save: bool = True,
    filename: str = 'confusion_matrix.png'
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names
        save: Whether to save the plot
        filename: Output filename
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save:
        output_path = OUTPUTS_PATH / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {output_path}")
    
    plt.show()


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list = None,
    save: bool = True,
    filename: str = 'classification_report.txt'
) -> str:
    """
    Generate and save classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Label names
        save: Whether to save the report
        filename: Output filename
        
    Returns:
        Classification report as string
    """
    report = classification_report(y_true, y_pred, target_names=target_names)
    
    print("\n=== Classification Report ===")
    print(report)
    
    if save:
        output_path = OUTPUTS_PATH / filename
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Saved classification report to {output_path}")
    
    return report
