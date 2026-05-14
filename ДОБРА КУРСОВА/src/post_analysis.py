"""
Post-training analysis: feature importance, PCA experiments (2f)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from typing import Any, Optional
from src.config import OUTPUTS_PATH


def analyze_feature_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 10,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Calculate and analyze feature importance using permutation importance.
    
    Args:
        model: Trained model
        X: Features
        y: Labels
        n_repeats: Number of permutations
        random_state: Random state
        
    Returns:
        DataFrame with feature importance
    """
    print("Calculating permutation importance...")
    
    perm_importance = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )
    
    importance_df = pd.DataFrame({
        'feature_idx': range(X.shape[1]),
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    return importance_df


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    save: bool = True,
    filename: str = 'feature_importance.png'
) -> None:
    """
    Plot feature importance.
    
    Args:
        importance_df: DataFrame with feature importance
        top_n: Number of top features to plot
        save: Whether to save the plot
        filename: Output filename
    """
    plt.figure(figsize=(10, 8))
    
    top_features = importance_df.head(top_n)
    plt.barh(range(len(top_features)), top_features['importance_mean'])
    plt.yticks(range(len(top_features)), top_features['feature_idx'])
    plt.xlabel('Importance')
    plt.ylabel('Feature Index')
    plt.title(f'Top {top_n} Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save:
        output_path = OUTPUTS_PATH / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved feature importance plot to {output_path}")
    
    plt.show()


def perform_pca_analysis(
    embeddings: np.ndarray,
    n_components: int = 50,
    plot: bool = True
) -> tuple:
    """
    Perform PCA analysis on embeddings.
    
    Args:
        embeddings: Input embeddings
        n_components: Number of PCA components
        plot: Whether to plot explained variance
        
    Returns:
        Tuple of (PCA object, transformed embeddings)
    """
    print(f"Performing PCA with {n_components} components...")
    
    pca = PCA(n_components=n_components)
    embeddings_pca = pca.fit_transform(embeddings)
    
    print(f"Explained variance ratio (total): {pca.explained_variance_ratio_.sum():.4f}")
    
    if plot:
        plot_explained_variance(pca)
    
    return pca, embeddings_pca


def plot_explained_variance(
    pca: PCA,
    save: bool = True,
    filename: str = 'pca_explained_variance.png'
) -> None:
    """
    Plot cumulative explained variance.
    
    Args:
        pca: Fitted PCA object
        save: Whether to save the plot
        filename: Output filename
    """
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
             pca.explained_variance_ratio_, 'o-')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Component')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             np.cumsum(pca.explained_variance_ratio_), 'o-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance')
    plt.grid(True)
    
    plt.tight_layout()
    
    if save:
        output_path = OUTPUTS_PATH / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved explained variance plot to {output_path}")
    
    plt.show()
