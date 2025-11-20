from typing import List, Tuple
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from schema import Material
import numpy as np


# =========================
# MOF: KH_Classes dataset
# =========================

import numpy as np
import pandas as pd


def build_mof_kh_dataset(mof_df):
    """
    Build a tabular MOF dataset for KH-class prediction / ranking.

    This function now expects a **pandas DataFrame** (as returned by fetch_mofs_mock)
    instead of a list of Material objects.

    Returns:
        X         : np.ndarray of numeric features
        y         : np.ndarray of integer labels (if available, else zeros)
        df_mof    : DataFrame with at least an 'id' column
        le_kh     : kept for backward compatibility (set to None here)
    """
    # ---- Ensure DataFrame ----
    if not isinstance(mof_df, pd.DataFrame):
        # If some old code passes a list/dict, convert defensively
        mof_df = pd.DataFrame(mof_df)

    df = mof_df.copy()

    # ---- Ensure an ID column ----
    if "id" not in df.columns:
        for col in ["coreid", "refcode", "number", "mof_id"]:
            if col in df.columns:
                df["id"] = df[col].astype(str)
                break
        if "id" not in df.columns:
            df["id"] = df.index.astype(str)

    # ---- Pick numeric feature columns ----
    # Prefer "known" MOF descriptors first, then any other numeric columns.
    preferred = [
        "density_g_cm3",
        "asa_m2_g",
        "void_fraction",
        "pore_volume_cm3_g",
        "surface_area_m2_g",
        "kh_value",
    ]
    num_cols = [c for c in preferred if c in df.columns]

    # Add any other numeric columns (but avoid duplicates and the label column)
    numeric_all = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in numeric_all:
        if c not in num_cols and c not in ["kh_label_reindexed", "kh_label"]:
            num_cols.append(c)

    if not num_cols:
        raise ValueError(
            "build_mof_kh_dataset: no numeric feature columns found in MOF DataFrame."
        )

    X = df[num_cols].astype(float).fillna(0.0)

    # ---- Labels (if available) ----
    if "kh_label_reindexed" in df.columns:
        y = df["kh_label_reindexed"].astype(int).values
    elif "kh_label" in df.columns:
        # If these are ints already, just use them; if strings, you could encode.
        y = pd.to_numeric(df["kh_label"], errors="coerce").fillna(0).astype(int).values
    else:
        # Not strictly needed for the fused model pipeline, but keep API stable.
        y = np.zeros(len(df), dtype=int)

    le_kh = None  # not used in the fused pipeline

    return X, y, df, le_kh


# =========================
# Plant: A (assimilation) dataset
# =========================

import numpy as np
import pandas as pd


def build_plant_A_dataset(plant_materials: pd.DataFrame):
    """
    Build a supervised regression dataset for predicting CO₂ assimilation A.
    """
    if not isinstance(plant_materials, pd.DataFrame):
        plant_materials = pd.DataFrame(plant_materials)

    # IMPORTANT: make sure index is 0..N-1 so _row_id is positional
    df = plant_materials.reset_index(drop=True).copy()

    # Normalize key columns
    if "Ci " in df.columns and "Ci" not in df.columns:
        df["Ci"] = df["Ci "]
    if "Species " in df.columns and "Species" not in df.columns:
        df["Species"] = df["Species "].astype(str).str.strip()

    # Row id so we can map back by position
    df["_row_id"] = np.arange(len(df))

    if "A" not in df.columns:
        raise ValueError("build_plant_A_dataset: column 'A' not found in plant_materials")

    # Only use rows with a valid A for supervised training
    df_tr = df[df["A"].notna()].copy()

    # --- Numeric features ---
    num_cols = []
    for col in ["Ci", "Temp", "Qobs", "Pressure"]:
        if col in df_tr.columns:
            num_cols.append(col)

    for col in num_cols:
        df_tr[col] = pd.to_numeric(df_tr[col], errors="coerce").fillna(0.0)

    X = df_tr[num_cols].astype(float)
    y = pd.to_numeric(df_tr["A"], errors="coerce").values

    # --- Categorical features: Subtype, Pathway ---
    cat_frames = []
    if "Subtype" in df_tr.columns:
        cat_frames.append(
            pd.get_dummies(df_tr["Subtype"].astype(str).str.strip(), prefix="subtype")
        )
    if "Pathway" in df_tr.columns:
        cat_frames.append(
            pd.get_dummies(df_tr["Pathway"].astype(str).str.strip(), prefix="path")
        )

    if cat_frames:
        X = pd.concat([X] + cat_frames, axis=1)

    return X.values, y, df_tr
