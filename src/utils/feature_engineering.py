from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# ---------------------------------------------------------------------
"""
    Feature derivate:
    - hr_reserve        = thalach / (220 - age)
    - chol_age          = chol / age
    - oldpeak_exang     = oldpeak * 1{exang==1}
    - oldpeak_slope     = oldpeak * slope_code
    - bp_age            = trestbps / age
    - age_bin           = binning di age in 5 fasce
"""
class FeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, add_age_bin: bool = False):
        self.add_age_bin = add_age_bin

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Cast numerico
        age      = pd.to_numeric(X["age"], errors="coerce")
        thalach  = pd.to_numeric(X["thalach"], errors="coerce")
        chol     = pd.to_numeric(X["chol"], errors="coerce")
        oldpeak  = pd.to_numeric(X["oldpeak"], errors="coerce")
        trestbps = pd.to_numeric(X["trestbps"], errors="coerce")

        # exang/slope possono essere category: prendo i codici (0,1,2…)
        if str(X["exang"].dtype) == "category":
            exang_num = X["exang"].cat.codes
        else:
            exang_num = pd.to_numeric(X["exang"], errors="coerce")

        if str(X["slope"].dtype) == "category":
            slope_num = X["slope"].cat.codes
        else:
            slope_num = pd.to_numeric(X["slope"], errors="coerce")

        # 1) HR reserve (evita div. per zero)
        max_hr = 220 - age
        max_hr = max_hr.replace(0, np.nan)
        X["hr_reserve"] = thalach / max_hr

        # 2) Colesterolo normalizzato per età
        X["chol_age"] = chol / age

        # 3) Interazione ischemia da sforzo
        X["oldpeak_exang"] = oldpeak * (exang_num == 1).astype(float)

        # 4) Interazione oldpeak * pendenza ST
        X["oldpeak_slope"] = oldpeak * slope_num.astype(float)

        # 5) Pressione a riposo rapportata all'età
        X["bp_age"] = trestbps / age

        # 6) Binning dell'età in fasce cliniche
        if self.add_age_bin:
            X["age_bin"] = pd.cut(age, bins=[0, 40, 50, 60, 70, 120], include_lowest=True)

        return X
# ---------------------------------------------------------------------