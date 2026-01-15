from models.Dense_AE import DenseAutoEncoder
from models.Lstm_2L_AE import LSTM2LayerAutoencoder
from models.Lstm_Deep_AE import LSTMDeepAutoEncoder

def get_model(model_name: str, window_size: int, n_features: int):

    if model_name == "DenseAutoEncoder":
        return DenseAutoEncoder(window_size=window_size, n_features=n_features)
    
    elif model_name == "LSTM2LayerAutoencoder":
        return LSTM2LayerAutoencoder(n_features=n_features)

    elif model_name == "LSTMDeepAutoEncoder":
        return LSTMDeepAutoEncoder(n_features=n_features)

    else:
        raise ValueError(f"Model {model_name} not recognized.")

    