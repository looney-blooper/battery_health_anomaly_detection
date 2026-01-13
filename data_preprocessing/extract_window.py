import numpy as np

FEATURES = [
    "voltage_load",
    "current_load",
    "temperature_battery",
    "temperature_mosfet"
]

def extract_lstm_windows(df, window_size=60, stride=10):
    X = []

    time_vals = df["time"].values
    data = df[FEATURES].values.astype(np.float32)

    i = 0
    while i < len(time_vals):
        end_time = time_vals[i] + window_size
        mask = (time_vals >= time_vals[i]) & (time_vals < end_time)

        if mask.sum() < window_size:
            i += stride
            continue

        window = data[mask][:window_size]
        X.append(window)

        i += stride

    return np.array(X)
