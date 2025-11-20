from mock_api import fetch_mofs_mock, fetch_plants_mock
from features import build_mof_kh_dataset, build_plant_A_dataset

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    r2_score,
    mean_absolute_error,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier, XGBRegressor
import lightgbm as lgb


# ==========================================
# 1) MOFs: KH_Classes classification
# ==========================================

print("=== MOF KH classification ===")

# Fetch MOF materials from mock API
mof_materials = fetch_mofs_mock()

# Build feature matrix X_mof, labels y_mof, full df, and label encoder
X_mof, y_mof, df_mof, le_kh = build_mof_kh_dataset(mof_materials)

print("Initial MOF dataset shape:", X_mof.shape)
print("KH class distribution (all):")
print(y_mof.value_counts())

# ---- Filter out ultra-rare classes (min_count = 2) ----
min_count = 2
class_counts = y_mof.value_counts()
valid_old_labels = class_counts[class_counts >= min_count].index

mask = y_mof.isin(valid_old_labels)
X_mof_filtered = X_mof.loc[mask]
y_mof_filtered = y_mof.loc[mask]

print("\nAfter filtering rare classes:")
print("Filtered shape:", X_mof_filtered.shape)
print("Filtered KH class distribution (old label IDs):")
print(y_mof_filtered.value_counts())

# ---- Reindex labels to 0..K-1 for XGBoost/LightGBM ----
unique_old_labels = np.sort(y_mof_filtered.unique())
old_to_new = {old: new for new, old in enumerate(unique_old_labels)}
new_to_old = {new: old for old, new in old_to_new.items()}

y_mof_reindexed = y_mof_filtered.map(old_to_new)

print("\nReindexed labels mapping (old -> new):")
print(old_to_new)

# ---- Train/test split with stratification on reindexed labels ----
Xtr, Xte, ytr, yte = train_test_split(
    X_mof_filtered,
    y_mof_reindexed,
    test_size=0.2,
    random_state=42,
    stratify=y_mof_reindexed,
)

# Labels used in this filtered, reindexed experiment
labels_used = np.arange(len(unique_old_labels))
# Human-readable KH class names for reports
target_names_used = le_kh.inverse_transform(unique_old_labels)

# ------------------------------------------
# RandomForest classifier (baseline)
# ------------------------------------------

clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
)
clf.fit(Xtr, ytr)

y_pred_rf = clf.predict(Xte)

print("\nClassification report (MOFs, KH classes) [RandomForest]:")
print(
    classification_report(
        yte,
        y_pred_rf,
        labels=labels_used,
        target_names=target_names_used,
    )
)

cm_rf = confusion_matrix(yte, y_pred_rf, labels=labels_used)
print("Confusion matrix [RandomForest] (rows=true, cols=pred):")
print(cm_rf)

# ------------------------------------------
# XGBoost classifier
# ------------------------------------------

xgb_clf = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42,
)

xgb_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", xgb_clf),
])

xgb_pipeline.fit(Xtr, ytr)
y_pred_xgb = xgb_pipeline.predict(Xte)

print("\nClassification report (MOFs, KH classes) [XGBoost]:")
print(
    classification_report(
        yte,
        y_pred_xgb,
        labels=labels_used,
        target_names=target_names_used,
    )
)

cm_xgb = confusion_matrix(yte, y_pred_xgb, labels=labels_used)
print("Confusion matrix [XGBoost] (rows=true, cols=pred):")
print(cm_xgb)

# ------------------------------------------
# LightGBM classifier
# ------------------------------------------

lgbm_clf = lgb.LGBMClassifier(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=64,
    class_weight="balanced",
    random_state=42,
    verbose=-1
)

lgbm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", lgbm_clf),
])

lgbm_pipeline.fit(Xtr, ytr)
y_pred_lgbm = lgbm_pipeline.predict(Xte)

print("\nClassification report (MOFs, KH classes) [LightGBM]:")
print(
    classification_report(
        yte,
        y_pred_lgbm,
        labels=labels_used,
        target_names=target_names_used,
    )
)

cm_lgbm = confusion_matrix(yte, y_pred_lgbm, labels=labels_used)
print("Confusion matrix [LightGBM] (rows=true, cols=pred):")
print(cm_lgbm)


# ==========================================
# 2) Plants: CO₂ assimilation regression
# ==========================================

print("\n=== Plant CO₂ assimilation regression ===")

plant_materials = fetch_plants_mock()

X_plant, y_plant, df_plant = build_plant_A_dataset(plant_materials)

print("Plant dataset shape:", X_plant.shape)
print("A (assimilation) stats:")
print(y_plant.describe())
print("Subtypes:", df_plant["Subtype"].unique())

# ---- Train/test split ----
Xtr_p, Xte_p, ytr_p, yte_p = train_test_split(
    X_plant,
    y_plant,
    test_size=0.2,
    random_state=42,
)

# ------------------------------------------
# RandomForest regressor (baseline)
# ------------------------------------------

rf_reg = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
)
rf_reg.fit(Xtr_p, ytr_p)

y_pred_rf_p = rf_reg.predict(Xte_p)

print("\nPlant CO₂ assimilation performance [RandomForest]:")
print("R²:", r2_score(yte_p, y_pred_rf_p))
print("MAE:", mean_absolute_error(yte_p, y_pred_rf_p))

# ------------------------------------------
# XGBoost regressor
# ------------------------------------------

xgb_reg = XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
)

xgb_reg_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", xgb_reg),
])

xgb_reg_pipeline.fit(Xtr_p, ytr_p)
y_pred_xgb_p = xgb_reg_pipeline.predict(Xte_p)

print("\nPlant CO₂ assimilation performance [XGBoost]:")
print("R²:", r2_score(yte_p, y_pred_xgb_p))
print("MAE:", mean_absolute_error(yte_p, y_pred_xgb_p))

# ------------------------------------------
# LightGBM regressor
# ------------------------------------------

lgbm_reg = lgb.LGBMRegressor(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=64,
    random_state=42,
)

lgbm_reg_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", lgbm_reg),
])

lgbm_reg_pipeline.fit(Xtr_p, ytr_p)
y_pred_lgbm_p = lgbm_reg_pipeline.predict(Xte_p)

print("\nPlant CO₂ assimilation performance [LightGBM]:")
print("R²:", r2_score(yte_p, y_pred_lgbm_p))
print("MAE:", mean_absolute_error(yte_p, y_pred_lgbm_p))
