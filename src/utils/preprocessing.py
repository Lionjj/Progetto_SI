from __future__ import annotations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.read import CATEGORICAL, NUMERIC
from src.utils.feature_engineering import FeatureBuilder

# Colonne originali
BASE_NUM = NUMERIC
BASE_CAT = CATEGORICAL

# Nuove colonne dal FeatureBuilder (valori numerici)
NEW_NUM = ["hr_reserve", "chol_age", "oldpeak_exang", "oldpeak_slope", "bp_age"]

# ---------------------------------------------------------------------
"""
    Restituisce una Pipeline: FeatureBuilder -> ColumnTransformer(imputazione+scaling/one-hot).
"""
# Con FeatureBuilding
def build_preprocessor(add_age_bin: bool = False) -> Pipeline:
    # set di colonne ColumnTransformer DOPO il FeatureBuilder
    num_cols = BASE_NUM + NEW_NUM
    cat_cols = BASE_CAT + (["age_bin"] if add_age_bin else [])

    num = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ])

    cat = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer([
        ("num", num, num_cols),
        ("cat", cat, cat_cols),
    ])

    return Pipeline([
        ("feat", FeatureBuilder(add_age_bin=add_age_bin)),
        ("pre", pre),
    ])

# Senza FeatureBuilding
def build_preprocessor_no_fe() -> Pipeline:
    num = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, NUMERIC),
                             ("cat", cat, CATEGORICAL)])
# ---------------------------------------------------------------------