"""
Gradio interface for fake news classification (2g)
"""

import gradio as gr
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.embeddings import TextEmbedder
from src.preprocessing import clean_text, remove_stopwords
from src.train import load_model
from src.config import MODELS_PATH, MODEL_NAME


class FakeNewsClassifier:
    """Fake news classifier with Gradio interface."""
    
    def __init__(self, model_path: str, embedder_model: str = MODEL_NAME):
        """
        Initialize classifier.
        
        Args:
            model_path: Path to trained model
            embedder_model: Sentence transformer model name
        """
        print("Loading model...")
        self.model = load_model(model_path)
        
        print("Loading embedder...")
        self.embedder = TextEmbedder(embedder_model)
        
        self.label_map = {0: "Real News", 1: "Fake News"}
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess input text."""
        cleaned = clean_text(text)
        processed = remove_stopwords(cleaned)
        return processed
    
    def predict(self, text: str) -> tuple:
        """
        Make prediction on input text.
        
        Args:
            text: Input news text
            
        Returns:
            Tuple of (prediction label, confidence)
        """
        if not text.strip():
            return "Please enter some text", {}
        
        # Preprocess
        processed_text = self.preprocess_text(text)
        
        if not processed_text:
            return "Text is too short after preprocessing", {}
        
        # Generate embedding
        embedding = self.embedder.encode_texts([processed_text], show_progress=False)
        
        # Predict
        prediction = self.model.predict(embedding)[0]
        
        # Get probabilities if available
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(embedding)[0]
            confidence = {self.label_map.get(i, f"Class {i}"): float(p) 
                         for i, p in enumerate(proba)}
        else:
            confidence = {self.label_map.get(prediction, f"Class {prediction}"): 1.0}
        
        result = self.label_map.get(prediction, f"Class {prediction}")
        
        return result, confidence


def create_interface(model_path: str):
    """
    Create Gradio interface.
    
    Args:
        model_path: Path to trained model
    """
    classifier = FakeNewsClassifier(model_path)
    
    def predict_wrapper(text):
        result, confidence = classifier.predict(text)
        return result, confidence
    
    # Create interface
    interface = gr.Interface(
        fn=predict_wrapper,
        inputs=gr.Textbox(
            lines=10,
            placeholder="Enter news article text here...",
            label="News Article"
        ),
        outputs=[
            gr.Textbox(label="Prediction"),
            gr.Label(label="Confidence", num_top_classes=2)
        ],
        title="🔍 Fake News Classifier",
        description="Enter a news article to classify it as real or fake news.",
        examples=[
            ["Breaking: Scientists discover new planet in our solar system."],
            ["Local community comes together to help flood victims."],
            ["You won't believe what this celebrity did next! Click here!"]
        ],
        theme=gr.themes.Soft()
    )
    
    return interface


if __name__ == "__main__":
    # Update this path to your trained model
    MODEL_FILE = MODELS_PATH / "logistic_regression.joblib"
    
    if not MODEL_FILE.exists():
        print(f"Error: Model file not found at {MODEL_FILE}")
        print("Please train a model first using the training notebook.")
        sys.exit(1)
    
    # Create and launch interface
    app = create_interface(str(MODEL_FILE))
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)
