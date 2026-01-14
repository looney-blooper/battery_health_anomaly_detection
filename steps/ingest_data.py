import pandas as pd
from zenml import step

@step
def ingest_data(csv_path: str) -> pd.DataFrame:
    """
    Loads raw battery CSV (DVC-tracked).
    """
    df = pd.read_csv(csv_path)
    return df