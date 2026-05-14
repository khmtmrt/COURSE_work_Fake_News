"""
Text cleaning and preprocessing (2a, 2b)
"""

import re
import pandas as pd
from typing import List
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


def clean_text(text: str) -> str:
    """
    Clean text by removing special characters, URLs, etc.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^A-Za-z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Convert to lowercase
    text = text.lower()
    
    return text


def remove_stopwords(text: str, language: str = 'english') -> str:
    """
    Remove stopwords from text.
    
    Args:
        text: Input text
        language: Language for stopwords
        
    Returns:
        Text without stopwords
    """
    stop_words = set(stopwords.words(language))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)


def preprocess_dataframe(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """
    Preprocess entire dataframe.
    
    Args:
        df: Input DataFrame
        text_column: Name of the text column
        
    Returns:
        Preprocessed DataFrame
    """
    df = df.copy()
    
    # Remove duplicates
    df.drop_duplicates(subset=[text_column], inplace=True)
    
    # Remove missing values
    df.dropna(subset=[text_column], inplace=True)
    
    # Clean text
    df['cleaned_text'] = df[text_column].apply(clean_text)
    
    # Remove stopwords
    df['processed_text'] = df['cleaned_text'].apply(remove_stopwords)
    
    # Remove empty texts
    df = df[df['processed_text'].str.len() > 0]
    
    return df.reset_index(drop=True)
