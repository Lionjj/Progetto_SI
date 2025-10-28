# checks.py
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    f1_score, roc_auc_score, confusion_matrix
)

from src.models.read import load_cleveland, TARGET_BIN
from src.utils.preprocessing import build_preprocessor


def quick_dummy_check():
    """Confronto con un classificatore banale (most_frequent)."""
    df = load_cleveland()
    X = df.drop(columns=[TARGET_BIN, "num"])
    y = df[TARGET_BIN].values
    prev = y.mean()
    print(f"[Dummy] Prevalenza classe positiva: {prev:.3f}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    dummy = Pipeline([
        ("pre", build_preprocessor()),
        ("clf", DummyClassifier(strategy="most_frequent"))
    ])
    dummy.fit(X_tr, y_tr)
    y_pred = dummy.predict(X_te)

    # proba costante ~ prevalenza: ROC-AUC ~ 0.5 (non informativo)
    print(f"[Dummy] F1: {f1_score(y_te, y_pred):.3f}")
    print("[Dummy] Nota: ROC-AUC con probabilità costante è ~0.5")


def check_transform_no_nan():
    """Verifica che dopo il preprocess non ci siano NaN/Inf."""
    df = load_cleveland()
    X = df.drop(columns=[TARGET_BIN, "num"])
    y = df[TARGET_BIN].values

    pre = build_preprocessor()
    Z = pre.fit_transform(X, y)
    Zdense = Z.toarray() if hasattr(Z, "toarray") else Z

    assert np.isfinite(Zdense).all(), "Trovati NaN/Inf dopo il preprocess!"
    print("[Preprocess] OK: nessun NaN/Inf dopo preprocess.")


def check_confusion_matrix_sum(model_path: str = "out/models/rf.joblib"):
    """Controlla che la CM sommi al numero di esempi nel test set."""
    from joblib import load

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Modello non trovato: {model_path}. Esegui prima il training."
        )

    df = load_cleveland()
    X = df.drop(columns=[TARGET_BIN, "num"])
    y = df[TARGET_BIN].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model = load(model_path)
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)

    cm = confusion_matrix(y_te, pred)
    assert cm.sum() == len(y_te), "La Confusion Matrix non somma a n_test!"
    print(f"[CM] OK: somma = {cm.sum()} (n_test={len(y_te)})")


def smoke_predict(model_path: str = "out/models/rf.joblib", n: int = 3):
    """Fumo: il modello carica e produce probabilità su N righe."""
    from joblib import load

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Modello non trovato: {model_path}. Esegui prima il training."
        )

    model = load(model_path)
    X_small = load_cleveland().drop(columns=[TARGET_BIN, "num"]).head(n)
    p = model.predict_proba(X_small)[:, 1]
    print(f"[Predict] predict_proba su {n} righe:", p)


if __name__ == "__main__":
    quick_dummy_check()
    check_transform_no_nan()
    check_confusion_matrix_sum("out/models/rf.joblib")  # cambia se il best non è RF
    smoke_predict("out/models/rf.joblib", n=3)
