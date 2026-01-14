import numpy as np
from zenml import step
from data_preprocessing.load_and_clean import load_and_clean
from data_preprocessing.extract_window import extract_lstm_windows

@step
def preprocess_data(csv_path: str) -> np.ndarray:
    """
    Preprocesses raw battery CSV data for LSTM model.
    - Loads and cleans data
    - Extracts LSTM windows
    """
    df = load_and_clean(csv_path)
    windows = extract_lstm_windows(df)
    return windows