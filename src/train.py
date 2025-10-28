from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional

# backend "headless" per server/venv senza display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    roc_curve, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay, classification_report, 
    average_precision_score
)

# Modelli scelti KNN, DecisionTree, RandomForest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Project modules
from src.models.read import load_cleveland, TARGET_BIN
from src.utils.preprocessing import build_preprocessor

try:
    from src.utils.preprocessing import build_preprocessor_no_fe
except Exception:
    build_preprocessor_no_fe = None

# ---------------------------------------------------------------------
# Cartelle output
OUT = Path("out")
FIG = OUT / "figures"
RES = OUT / "results"
MOD = OUT / "models"
for d in (FIG, RES, MOD):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
'''
Instanzia e ritorna dizzionari:
    - Modelli
    - Griglie iperparametri
'''
def _models_and_grids():
    models = {
        "knn":  KNeighborsClassifier(),
        "tree": DecisionTreeClassifier(random_state=42),
        "rf":   RandomForestClassifier(random_state=42),
    }
    grids = {
        "knn": {
            "clf__n_neighbors": [3, 5, 7, 11],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2],  # Manhattan / Euclidea
        },
        "tree": {
            "clf__criterion": ["gini", "entropy"],
            "clf__max_depth": [None, 3, 5, 8, 12],
            "clf__min_samples_split": [2, 5, 10],
        },
        "rf": {
            "clf__n_estimators": [200, 400, 800],
            "clf__max_depth": [None, 5, 8, 12],
            "clf__min_samples_split": [2, 5, 10],
        },
    }
    return models, grids

# ---------------------------------------------------------------------
"""Sceglie il preprocessore con/senza Feature Engineering."""
def _pick_preprocessor(use_fe: bool, add_age_bin: bool):
    if not use_fe and build_preprocessor_no_fe is not None:
        return build_preprocessor_no_fe()
    # default (con FE abilitabile)
    return build_preprocessor(add_age_bin=add_age_bin)

# ---------------------------------------------------------------------
"""
Genera e salva:
    - Confusion Matrix (png) + csv dei conteggi
    - ROC (png) + csv (fpr, tpr, thresholds)
    - PR (png)  + csv (recall, precision, thresholds)
Ritorna i numeri chiave: auc_roc, pr_auc e confusion matrix.
"""
def _plot_and_save_curves(name: str, y_true, proba, preds):
    # ===== Confusion Matrix =====
    cm = confusion_matrix(y_true, preds)
    ConfusionMatrixDisplay(cm).plot()
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(FIG / f"cm_{name}.png", dpi=200)
    plt.close()
    # salva anche i conteggi
    pd.DataFrame(cm, index=["TN","FN"], columns=["FP","TP"]).to_csv(RES / f"cm_{name}.csv")

    # ===== ROC =====
    fpr, tpr, thr_roc = roc_curve(y_true, proba)
    auc_roc = float(roc_auc_score(y_true, proba))
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc_roc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC - {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / f"roc_{name}.png", dpi=200)
    plt.close()
    # csv con i punti della curva
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": np.r_[thr_roc, np.nan][:len(fpr)]}) \
        .to_csv(RES / f"roc_points_{name}.csv", index=False)

    # ===== Precision–Recall =====
    prec, rec, thr_pr = precision_recall_curve(y_true, proba)
    pr_auc = float(average_precision_score(y_true, proba))  # area sotto PR
    plt.figure()
    plt.plot(rec, prec, label=f"AP={pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall - {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / f"pr_{name}.png", dpi=200)
    plt.close()
    # csv con i punti della curva
    pd.DataFrame({"recall": rec, "precision": prec,
                  "threshold": np.r_[thr_pr, np.nan][:len(rec)]}) \
        .to_csv(RES / f"pr_points_{name}.csv", index=False)

    # ritorna i numeri per il dizionario metriche
    return {"auc_roc": auc_roc, "pr_auc": pr_auc, "cm": cm.tolist()}

# ---------------------------------------------------------------------
"""Fit + CV + valutazione su test per un singolo modello."""
def _fit_one(
    X: pd.DataFrame,
    y: np.ndarray,
    name: str,
    use_fe: bool = True,
    add_age_bin: bool = False,
):
    pre = _pick_preprocessor(use_fe=use_fe, add_age_bin=add_age_bin)

    models, grids = _models_and_grids()
    if name not in models:
        raise ValueError(f"Modello '{name}' non supportato. Usa: knn, tree, rf")

    pipe = Pipeline([("pre", pre), ("clf", models[name])])

    # Split hold-out
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # CV stratificata 5-fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    gs = GridSearchCV(
        pipe,
        grids[name],
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
    gs.fit(Xtr, ytr)

    best = gs.best_estimator_
    proba = best.predict_proba(Xte)[:, 1]
    pred05 = (proba >= 0.5).astype(int)

    # Soglia che massimizza F1 (solo per analisi)
    prec_arr, rec_arr, thr = precision_recall_curve(yte, proba)
    f1s = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-12)
    best_idx = int(np.nanargmax(f1s))
    best_thr = float(thr[best_idx]) if best_idx < len(thr) else 0.5
    pred_best = (proba >= best_thr).astype(int)

    # === METRICHE ===
    curve_stats = _plot_and_save_curves(name, yte, proba, pred05)

    metrics = {
        "model": name,
        "best_params": gs.best_params_,
        "cv_best_f1": float(gs.best_score_),
        "thresholds": {"default": 0.5, "best_f1": best_thr},
        "test_default": {
            "accuracy": float(accuracy_score(yte, pred05)),
            "precision": float(precision_score(yte, pred05)),
            "recall": float(recall_score(yte, pred05)),
            "f1": float(f1_score(yte, pred05)),
            "roc_auc": curve_stats["auc_roc"],   # AUC-ROC
            "pr_auc": curve_stats["pr_auc"],     # PR-AUC (Average Precision)
        },
        "test_best_f1": {
            "accuracy": float(accuracy_score(yte, pred_best)),
            "precision": float(precision_score(yte, pred_best)),
            "recall": float(recall_score(yte, pred_best)),
            "f1": float(f1_score(yte, pred_best)),
        },
        "settings": {"use_fe": use_fe, "add_age_bin": add_age_bin},
    }

    # === SALVATAGGI ===
    dump(best, MOD / f"{name}.joblib")
    (RES / f"metrics_{name}.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(gs.cv_results_).to_csv(RES / f"cv_{name}.csv", index=False)
    (RES / f"classification_report_{name}_default.txt").write_text(
        classification_report(yte, pred05)
    )
    (RES / f"classification_report_{name}_bestf1.txt").write_text(
        classification_report(yte, pred_best)
    )

    try:
        pre_fitted = best.named_steps["pre"]
        feat_names = list(pre_fitted.get_feature_names_out())
        pd.Series(feat_names).to_csv(RES / f"feature_names_{name}.csv",
                                     index=False, header=False)
    except Exception:
        pass

    return metrics

# ---------------------------------------------------------------------
"""
Parametri:
    - models: lista di modelli. Se None -> ["rf"].
    - use_fe: abilita/disabilita il Feature Engineering nella pipeline.
    - add_age_bin: binning dell'età come feature categorica addizionale.

Salva:
    - out/models/*.joblib
    - out/results/metrics_*.json, cv_*.csv, summary.{csv,md,tex}
    - out/figures/{roc,pr,cm}_*.png
"""
def train_model(
    models: Optional[List[str]] = None,
    use_fe: bool = True,
    add_age_bin: bool = False,
):
    df = load_cleveland()                     # lettura del dataset
    X = df.drop(columns=[TARGET_BIN, "num"])  # rimuove target e colonna multiclass
    y = df[TARGET_BIN].values

    if not models:
        models = ["rf"]  # default

    all_res = []
    for m in models:
        print(f"\n>>> Training {m} | FE={use_fe} | age_bin={add_age_bin}")
        res = _fit_one(X, y, m, use_fe=use_fe, add_age_bin=add_age_bin)
        print(json.dumps(res, indent=2))
        all_res.append(res)

    rows = []
    for r in all_res:
        rows.append({
            "model": r["model"],
            "cv_best_f1": r["cv_best_f1"],
            "test_f1@0.5": r["test_default"]["f1"],
            "test_acc@0.5": r["test_default"]["accuracy"],
            "test_roc_auc": r["test_default"]["roc_auc"],
            "test_f1@best": r["test_best_f1"]["f1"],
        })
    summ = pd.DataFrame(rows).sort_values("test_f1@0.5", ascending=False)

    RES.mkdir(parents=True, exist_ok=True)
    summ.to_csv(RES / "summary.csv", index=False)
    try:
        summ.to_markdown(RES / "summary.md")
    except Exception:
        pass
    with open(RES / "summary.tex", "w") as f:
        f.write(summ.to_latex(index=False, float_format="%.3f"))

    print("\n=== Riepilogo ===")
    print(summ.to_string(index=False))

# ---------------------------------------------------------------------
# Esecuzione da CLI
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Training Heart Disease (Cleveland)")
    p.add_argument("models", nargs="*", default=["rf"], help="Modelli: knn tree rf")
    p.add_argument("--no-fe", action="store_true", help="Disabilita Feature Engineering")
    p.add_argument("--age-bin", action="store_true", help="Aggiunge il binning dell'età")
    args = p.parse_args()

    train_model(
        models=args.models,
        use_fe=not args.no_fe,
        add_age_bin=args.age_bin,
    )
