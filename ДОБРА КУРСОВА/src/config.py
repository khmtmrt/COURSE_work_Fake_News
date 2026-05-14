"""
Configuration module - reads .env and stores constants
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = os.getenv("DATA_PATH", str(PROJECT_ROOT / "data"))
RAW_DATA_PATH = Path(DATA_PATH) / "raw"
PROCESSED_DATA_PATH = Path(DATA_PATH) / "processed"
MODELS_PATH = PROJECT_ROOT / "models"
OUTPUTS_PATH = PROJECT_ROOT / "outputs"

# Model configuration
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

# Training parameters
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))

# Create directories if they don't exist
for path in [RAW_DATA_PATH, PROCESSED_DATA_PATH, MODELS_PATH, OUTPUTS_PATH]:
    path.mkdir(parents=True, exist_ok=True)
