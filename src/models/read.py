from pathlib import Path
from typing import Optional, Union
import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# Attributi standard
COLUMNS = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach",
           "exang","oldpeak","slope","ca","thal","num"]

# Attributi di tipo Categorical 
CATEGORICAL = ["sex","cp","fbs","restecg","exang","slope","ca","thal"]

# Attributi di tipo Numerici
NUMERIC = ["age","trestbps","chol","thalach","oldpeak"]

TARGET_BIN = "target"

# Provo a caricare i dati dal file locale altrimenti li scarico da UCI_URL
def load_cleveland(path: Optional[Union[str, Path]] = "data/raw/processed.cleveland.data") -> pd.DataFrame:
    p = Path(path) if path is not None else None

    if p is not None and p.exists():
        df = pd.read_csv(p, header=None, names=COLUMNS, na_values="?")
    else:
        df = pd.read_csv(UCI_URL, header=None, names=COLUMNS, na_values="?")

    for c in CATEGORICAL: df[c] = df[c].astype("category")
    for c in NUMERIC: df[c] = pd.to_numeric(df[c], errors="coerce")

    # target binario: 0 = assente; 1 = presente (valori [1..4])
    df[TARGET_BIN] = (df["num"] > 0).astype(int)
    return df
