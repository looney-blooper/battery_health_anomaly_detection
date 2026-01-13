import pandas as pd

def load_and_clean(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)

    df.columns = [
        "start_time",
        "time",
        "mode",
        "voltage_charger",
        "temperature_battery",
        "voltage_load",
        "current_load",
        "temperature_mosfet",
        "temperature_resistor",
        "mission_type"
    ]

    df["start_time"] = pd.to_datetime(df["start_time"])
    df = df.sort_values("time")

    df = df[
        (df["mode"] == -1) &
        (df["mission_type"] == 0)
    ]

    NUMERIC_COLS = [
        "voltage_load",
        "current_load",
        "temperature_battery",
        "temperature_mosfet"
    ]

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=NUMERIC_COLS)

    return df
