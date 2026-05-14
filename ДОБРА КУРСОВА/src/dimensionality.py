"""
Dimensionality reduction using UMAP/t-SNE (2c)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from umap import UMAP
from typing import Literal, Optional
from src.config import OUTPUTS_PATH


def reduce_dimensions(
    embeddings: np.ndarray,
    method: Literal['umap', 'tsne'] = 'umap',
    n_components: int = 2,
    random_state: int = 42,
    **kwargs
) -> np.ndarray:
    """
    Reduce dimensionality of embeddings.
    
    Args:
        embeddings: Input embeddings
        method: Reduction method ('umap' or 'tsne')
        n_components: Number of output dimensions
        random_state: Random state for reproducibility
        **kwargs: Additional arguments for the reduction method
        
    Returns:
        Reduced embeddings
    """
    if method == 'umap':
        reducer = UMAP(
            n_components=n_components,
            random_state=random_state,
            **kwargs
        )
    elif method == 'tsne':
        reducer = TSNE(
            n_components=n_components,
            random_state=random_state,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"Applying {method.upper()} dimensionality reduction...")
    reduced_embeddings = reducer.fit_transform(embeddings)
    print(f"Reduced shape: {reduced_embeddings.shape}")
    
    return reduced_embeddings


def plot_embeddings_2d(
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
    title: str = "2D Embedding Visualization",
    save: bool = True,
    filename: Optional[str] = None
) -> None:
    """
    Plot 2D embeddings with labels.
    
    Args:
        embeddings_2d: 2D embeddings
        labels: Labels for coloring
        title: Plot title
        save: Whether to save the plot
        filename: Output filename
    """
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels,
        cmap='viridis',
        alpha=0.6,
        s=10
    )
    
    plt.colorbar(scatter, label='Label')
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.tight_layout()
    
    if save:
        if filename is None:
            filename = 'embeddings_2d_scatter.png'
        output_path = OUTPUTS_PATH / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    
    plt.show()


def create_embedding_dataframe(
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
    texts: Optional[list] = None
) -> pd.DataFrame:
    """
    Create a DataFrame from 2D embeddings and labels.
    
    Args:
        embeddings_2d: 2D embeddings
        labels: Labels
        texts: Optional list of original texts
        
    Returns:
        DataFrame with embeddings and labels
    """
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'label': labels
    })
    
    if texts is not None:
        df['text'] = texts
    
    return df
