from Dense_AE import DenseAutoEncoder
from Lstm_2L_AE import LSTM2LayerAutoencoder
from Lstm_Deep_AE import LSTMDeepAutoEncoder

def get_model(model_name: str, **kwargs):

    if model_name == "DenseAutoEncoder":
        return DenseAutoEncoder(**kwargs)
    
    elif model_name == "LSTM2LayerAutoencoder":
        return LSTM2LayerAutoencoder(**kwargs)
    
    elif model_name == "LSTMDeepAutoEncoder":
        return LSTMDeepAutoEncoder(**kwargs)
    
    else:
        raise ValueError(f"Model {model_name} not recognized.")

