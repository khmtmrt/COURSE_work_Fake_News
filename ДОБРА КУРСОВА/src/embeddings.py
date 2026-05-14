"""
Text embeddings using sentence-transformers (2b, 2c)
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Optional
from src.config import MODEL_NAME


class TextEmbedder:
    """
    Class for generating text embeddings using sentence-transformers.
    """
    
    def __init__(self, model_name: str = MODEL_NAME):
        """
        Initialize the embedder.
        
        Args:
            model_name: Name of the sentence-transformer model
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        print(f"Loaded model: {model_name}")
    
    def encode_texts(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            NumPy array of embeddings
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings
    
    def encode_dataframe(self, df: pd.DataFrame, text_column: str = 'processed_text') -> np.ndarray:
        """
        Encode texts from a DataFrame column.
        
        Args:
            df: Input DataFrame
            text_column: Name of the text column
            
        Returns:
            NumPy array of embeddings
        """
        texts = df[text_column].tolist()
        return self.encode_texts(texts)


def save_embeddings(embeddings: np.ndarray, filepath: str) -> None:
    """
    Save embeddings to disk.
    
    Args:
        embeddings: NumPy array of embeddings
        filepath: Path to save file
    """
    np.save(filepath, embeddings)
    print(f"Saved embeddings to {filepath}")


def load_embeddings(filepath: str) -> np.ndarray:
    """
    Load embeddings from disk.
    
    Args:
        filepath: Path to embeddings file
        
    Returns:
        NumPy array of embeddings
    """
    embeddings = np.load(filepath)
    print(f"Loaded embeddings from {filepath}")
    return embeddings
