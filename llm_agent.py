import os
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from mock_api import fetch_mofs_mock, fetch_plants_mock
from features import build_mof_kh_dataset
from gnn_data import MofGraphDataset

from sklearn.ensemble import RandomForestRegressor
from features import build_plant_A_dataset

import json
import re

from openai import OpenAI

# ============================
# 1. LLM CALL
# ============================

# Create a single, reusable client
_openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def call_llm(prompt: str) -> str:
    """
    Real LLM call using OpenAI's Chat Completions API.

    - Uses model="gpt-4.1-mini" by default (fast, good quality).
    - You can switch to "gpt-4.1" for higher quality if needed.

    Make sure OPENAI_API_KEY is set in your environment.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    response = _openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert assistant in materials science, MOFs, plant physiology, "
                    "and AI-driven material discovery. You answer clearly and concisely, and you "
                    "follow requested output formats exactly.\n\n"
                    "When MOF records include DOIs, you MUST use them as primary literature "
                    "anchors in your justifications (e.g., 'see DOI: 10.xxxx/xxxxx'). "
                    "When chemical formulas are provided, explicitly refer to them when "
                    "describing the framework."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content

def parse_goal_for_preferences(goal: str) -> dict:
    """
    Parse the structured goal text (built in the Streamlit UI) to extract:
      - MOF design weights: affinity, capacity, porosity, density, stability
      - Operating temperature range
      - Operating pressure range

    Returns a dict with sensible defaults if parsing fails.
    """
    prefs = {
        "affinity": 1.0,
        "capacity": 0.7,
        "porosity": 0.6,
        "density": 0.3,
        "stability": 0.5,
        "temp_min": None,
        "temp_max": None,
        "press_min": None,
        "press_max": None,
    }

    # --- Design priorities (0–1) ---
    # We look for lines like:
    # - Affinity / selectivity (KH class, model score): 0.80
    def extract_weight(label: str, default: float):
        pattern = rf"{label}.*:\s*([0-1](?:\.\d+)?)"
        m = re.search(pattern, goal)
        return float(m.group(1)) if m else default

    prefs["affinity"] = extract_weight("Affinity / selectivity", prefs["affinity"])
    prefs["capacity"] = extract_weight("Capacity \\(surface area / pore volume\\)", prefs["capacity"])
    prefs["porosity"] = extract_weight("Porosity / void fraction", prefs["porosity"])
    prefs["density"] = extract_weight("Low framework density", prefs["density"])
    prefs["stability"] = extract_weight("Stability / robustness", prefs["stability"])

    # --- Temperature range ---
    # Line: "Operating temperature range: 20–60 °C"
    m_temp = re.search(
        r"Operating temperature range:\s*([-+]?\d+)\s*[\u2013\-]\s*([-+]?\d+)",
        goal,
    )
    if m_temp:
        try:
            prefs["temp_min"] = float(m_temp.group(1))
            prefs["temp_max"] = float(m_temp.group(2))
        except ValueError:
            pass

    # --- Pressure range ---
    # Line: "Operating pressure range: 0.10–1.00 bar"
    m_press = re.search(
        r"Operating pressure range:\s*([0-9.]+)\s*[\u2013\-]\s*([0-9.]+)\s*bar",
        goal,
    )
    if m_press:
        try:
            prefs["press_min"] = float(m_press.group(1))
            prefs["press_max"] = float(m_press.group(2))
        except ValueError:
            pass

    return prefs



# ============================
# 2. LOAD / TRAIN FUSED MODEL
# ============================

def build_fusion_dataset():
    """
    Build fused (tabular + GNN embedding) dataset for MOFs.

    Returns:
        X_fused: np.ndarray (N, D_tab + D_emb)
        y: np.ndarray (N,)
        df_merged: pd.DataFrame with id, mof_id, kh_class, formula, doi, etc.
        class_names: list[str] in reindexed order
    """
    # --- Load embeddings ---
    emb_path = "embeddings/mof_gcn_embeddings.csv"
    df_emb = pd.read_csv(emb_path)

    # --- Rebuild tabular dataset ---
    mof_materials = fetch_mofs_mock()
    X_mof, y_mof, df_mof, le_kh = build_mof_kh_dataset(mof_materials)

    # Attach id to the tabular features for merging
    X_mof_with_id = X_mof.copy()
    X_mof_with_id["id"] = df_mof["id"].values

    # --- Merge embeddings with tabular features on mof_id ↔ id ---
    df_merged = df_emb.merge(
        X_mof_with_id,
        left_on="mof_id",
        right_on="id",
        how="inner",
    )

    # --- Bring over extra interpretability fields from df_mof (e.g., DOI, formula) ---
    extra_cols = [
        c
        for c in df_mof.columns
        if c in ["doi", "DOI", "chemical_formula", "formula", "ChemicalFormula", "chem_formula"]
    ]
    if extra_cols:
        df_extra = df_mof[["id"] + extra_cols].copy()
        df_merged = df_merged.merge(df_extra, on="id", how="left")

    # Labels already reindexed from the GNN pipeline
    y = df_merged["kh_label_reindexed"].astype(int).values

    emb_cols = [c for c in df_merged.columns if c.startswith("emb_")]
    tab_cols = [c for c in X_mof.columns]

    X_tab = df_merged[tab_cols].values
    X_emb = df_merged[emb_cols].values
    X_fused = np.hstack([X_tab, X_emb])

    # --- Get class names in reindexed order ---
    dataset = MofGraphDataset(root="./gnn_data", cif_dir="./cifs")
    class_names = dataset.classes  # KH class strings

    return X_fused, y, df_merged, class_names


def train_fused_xgb(X: np.ndarray, y: np.ndarray) -> XGBClassifier:
    """
    Train an XGBoost classifier on fused features.

    In a real setup, you would instead:
    - train once
    - save with joblib
    - load on demand

    Here we keep it simple and train on the fly.
    """
    xgb = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
    )
    xgb.fit(X, y)
    return xgb


# ============================
# 3. SCORING PIPELINE
# ============================

def score_mofs_with_fused_model(
    goal: str,
    top_k: int = 20,
) -> pd.DataFrame:
    """
    Score MOFs using the fused (tabular + GNN) XGBoost model.

    The *base* score is the model's probability for a "strong" adsorption class,
    but we also compute a scenario-dependent "utility" that depends on the
    user design priorities parsed from the goal:

      - affinity_weight   -> base model score
      - capacity_weight   -> asa_m2_g
      - porosity_weight   -> void_fraction
      - density_weight    -> 1 - normalized density_g_cm3  (lower density is better)
      - stability_weight  -> (currently not used numerically, but passed in goal)

    The returned DataFrame is sorted by `utility`, so changing the sliders in
    the UI changes which MOFs appear at the top.
    """
    # --- fused features ---
    X_fused, y, df_merged, class_names = build_fusion_dataset()

    # Train or load model
    model = train_fused_xgb(X_fused, y)

    # Predict class probabilities
    proba = model.predict_proba(X_fused)  # shape (N, num_classes)

    # Identify "strong" classes as target for base affinity score
    strong_indices = [i for i, cls in enumerate(class_names) if "strong" in cls.lower()]
    if not strong_indices:
        # fallback: use the last class
        target_col = proba.shape[1] - 1
        scores = proba[:, target_col]
    else:
        # If multiple strong-like classes, use max across them
        scores = proba[:, strong_indices].max(axis=1)

    df_scores = df_merged.copy()
    df_scores["score"] = scores
    df_scores["kh_class_name"] = [
        class_names[label] for label in df_scores["kh_label_reindexed"].astype(int)
    ]

    # --- parse user preferences from goal ---
    prefs = parse_goal_for_preferences(goal)
    w_aff = prefs["affinity"]
    w_cap = prefs["capacity"]
    w_por = prefs["porosity"]
    w_den = prefs["density"]

    # --- helper for safe min-max normalization ---
    def norm_col(df: pd.DataFrame, col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(0.0, index=df.index)
        vals = pd.to_numeric(df[col], errors="coerce")
        vmin = vals.min()
        vmax = vals.max()
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            return pd.Series(0.0, index=df.index)
        return (vals - vmin) / (vmax - vmin + 1e-12)

    # Normalized components
    norm_score = norm_col(df_scores, "score")
    norm_asa = norm_col(df_scores, "asa_m2_g")
    norm_vf = norm_col(df_scores, "void_fraction")
    norm_dens = norm_col(df_scores, "density_g_cm3")

    # Lower density is better, so use (1 - norm_dens)
    utility = (
        w_aff * norm_score
        + w_cap * norm_asa
        + w_por * norm_vf
        + w_den * (1.0 - norm_dens)
    )

    df_scores["utility"] = utility

    # Sort by scenario-dependent utility
    df_scores = df_scores.sort_values("utility", ascending=False).head(top_k).reset_index(drop=True)

    return df_scores


# ============================
# 4. PROMPT BUILDING
# ============================

def _format_formula_chemformula(formula: str) -> str:
    """
    Wrap a plain chemical formula string so it is ready for LaTeX chemformula,
    e.g. 'CO2' -> '\\ch{CO2}'.

    If it already looks like '\\ch{...}', return as-is.
    """
    if not isinstance(formula, str):
        return ""
    f = formula.strip()
    if not f:
        return ""
    if f.startswith("\\ch{") and f.endswith("}"):
        return f
    return f"\\ch{{{f}}}"


def _get_formula_from_row(row: pd.Series) -> Optional[str]:
    for key in ["chemical_formula", "formula", "ChemicalFormula", "chem_formula"]:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _get_doi_from_row(row: pd.Series) -> Optional[str]:
    for key in ["doi", "DOI"]:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def build_candidate_summary_row(row: pd.Series) -> str:
    """
    Render one MOF candidate into a compact text description for the LLM.
    """
    base_score = row.get("score", np.nan)
    util = row.get("utility", base_score)

    return (
        f"ID: {row['mof_id']} | KH class: {row.get('kh_class_name', 'NA')} | "
        f"model_score: {base_score:.3f} | utility: {util:.3f} | "
        f"density: {row.get('density_g_cm3', 'NA')}, "
        f"ASA: {row.get('asa_m2_g', 'NA')}, "
        f"pore volume: {row.get('pore_volume_cm3_g', 'NA')}, "
        f"void fraction: {row.get('void_fraction', 'NA')}"
    )




def build_agent_prompt(goal: str, df_top: pd.DataFrame) -> str:
    """
    Construct a prompt for the LLM that:
    - states the goal
    - lists candidate MOFs with model scores, descriptors, formula, and DOI
    - asks for structured JSON output + explanation referencing the literature
    """
    lines = []
    lines.append("You are an AI assistant helping select MOF candidates for gas adsorption tasks.")
    lines.append("You are given a ranked list of MOFs with:")
    lines.append("  - model-based scores (for adsorption/affinity),")
    lines.append("  - structural descriptors (density, surface area, pore volume, void fraction),")
    lines.append("  - chemical formulas when available,")
    lines.append("  - and DOIs to the primary literature when available.")
    lines.append("")
    lines.append(f"Goal: {goal}")
    lines.append("")
    lines.append("Here are the top candidates predicted by our ML/GNN fusion model (higher score is better):")
    lines.append("")

    for i, (_, row) in enumerate(df_top.iterrows(), start=1):
        lines.append(f"{i}. " + build_candidate_summary_row(row))

    lines.append("")
    lines.append(
        "Please do the following:\n"
        "1. Briefly explain which candidates you think are most promising for the stated goal.\n"
        "2. In your explanation, explicitly ground your reasoning in:\n"
        "   - their structural properties (KH class, surface area, void fraction, density, etc.), and\n"
        "   - the associated DOIs (when present), treating them as the primary literature. For example,\n"
        "     you can say 'experimental data in DOI 10.xxxx/xxxxx indicates strong CO₂ uptake...'.\n"
        "3. After your explanation, return a JSON object with the following keys:\n"
        '   - \"selected_ids\": a list of MOF IDs you recommend (e.g., [\"0000[Ag][nan]3[ASR]1\", ...])\n'
        '   - \"rationale\": a concise explanation (2–4 sentences) of why you chose them, referring to their properties and DOIs.\n'
        "Only output the JSON object after your explanation."
    )

    return "\n".join(lines)


# ============================
# 5. AGENT ENTRYPOINT (MOF-ONLY)
# ============================

def run_llm_agent_once(goal: str, top_k: int = 20) -> Dict[str, Any]:
    """
    Full pipeline:
      - score MOFs with fused model (scenario-aware utility)
      - build LLM prompt
      - call LLM
      - return both data and raw LLM response
    """
    # 1) Score candidates (now scenario-aware via `goal`)
    df_scores = score_mofs_with_fused_model(goal=goal, top_k=top_k)

    # 2) Build LLM prompt
    prompt = build_agent_prompt(goal, df_scores)

    # 3) Call LLM
    llm_resp = call_llm(prompt)

    return {
        "goal": goal,
        "candidates_df": df_scores,
        "prompt": prompt,
        "llm_response": llm_resp,
    }


# ============================
# 6. PLANT SCORING
# ============================

from typing import Optional

def score_plants_for_photosynthesis(
    goal: Optional[str] = None,
    top_k: int = 20,
) -> pd.DataFrame:
    """
    Score plants by predicted CO2 assimilation rate A using a RF regressor,
    then re-rank them based on how well their measurement temperature matches
    the desired operating temperature range from the goal (if provided).

    All plants from fetch_plants_mock() (including CAM) are returned with:
      - predicted_A (NaN for rows without training labels)
      - plant_utility (scenario-dependent score)
    """
    # Full pool: C3/C4 + CAM, etc.
    plant_materials = fetch_plants_mock()

    # Supervised training subset (only rows with A)
    X_tr, y_tr, df_tr = build_plant_A_dataset(plant_materials)

    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
    )
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_tr)

    # Make sure df_all index is 0..N-1 to match _row_id
    df_all = plant_materials.reset_index(drop=True).copy()

    # Initialize column
    df_all["predicted_A"] = np.nan

    # Row positions of training rows
    row_ids = df_tr["_row_id"].astype(int).values

    # Use positional indexing to assign predictions
    pred_col_idx = df_all.columns.get_loc("predicted_A")
    df_all.iloc[row_ids, pred_col_idx] = y_pred

    # --- parse prefs for temperature range (if goal provided) ---
    t_min = t_max = temp_mid = None
    if goal is not None:
        prefs = parse_goal_for_preferences(goal)
        t_min = prefs.get("temp_min")
        t_max = prefs.get("temp_max")
        if t_min is not None and t_max is not None:
            temp_mid = 0.5 * (t_min + t_max)

    # --- normalize predicted_A ---
    A_vals = pd.to_numeric(df_all["predicted_A"], errors="coerce")
    if A_vals.notna().sum() > 0:
        A_min = A_vals.min()
        A_max = A_vals.max()
        if np.isfinite(A_min) and np.isfinite(A_max) and A_max > A_min:
            norm_A = (A_vals - A_min) / (A_max - A_min + 1e-12)
        else:
            norm_A = pd.Series(0.5, index=df_all.index)
    else:
        # No predictions? set to neutral
        norm_A = pd.Series(0.5, index=df_all.index)

    # --- temperature match (1 = perfect match, 0 = bad) ---
    temp_match = pd.Series(0.5, index=df_all.index)  # neutral default
    if temp_mid is not None and "Temp" in df_all.columns:
        T_vals = pd.to_numeric(df_all["Temp"], errors="coerce")
        max_delta = 25.0
        delta = (T_vals - temp_mid).abs()
        temp_match = 1.0 - (delta.clip(upper=max_delta) / max_delta)

    # combine into a scenario-dependent utility
    df_all["plant_utility"] = 0.7 * norm_A + 0.3 * temp_match

    # Normalize id/name columns for UI & LLM
    if "plant_id" not in df_all.columns:
        if "id" in df_all.columns:
            df_all["plant_id"] = df_all["id"]
        else:
            df_all["plant_id"] = df_all.index.astype(str)

    if "name" not in df_all.columns:
        df_all["name"] = df_all.get("Species", df_all["plant_id"]).astype(str)

    # Sort by scenario-dependent utility and truncate
    df_all = df_all.sort_values("plant_utility", ascending=False).head(top_k).reset_index(drop=True)

    return df_all



def build_plant_summary_row(row: pd.Series) -> str:
    """
    Render one plant candidate into a compact text description for the LLM.
    """
    pathway = row.get("Pathway", "unknown")
    util = row.get("plant_utility", row.get("predicted_A", np.nan))

    return (
        f"ID: {row['plant_id']} | Species: {row['name']} | "
        f"Pathway: {pathway} | Subtype: {row.get('Subtype', 'NA')} | "
        f"predicted A: {row.get('predicted_A', np.nan):.3f} µmol m^-2 s^-1 | "
        f"scenario_utility: {util:.3f} | "
        f"Ci: {row.get('Ci', 'NA')}, Temp: {row.get('Temp', 'NA')} °C, "
        f"Qobs: {row.get('Qobs', 'NA')}"
    )



# ============================
# 7. BUILD HYBRID PROMPT
# ============================

def build_hybrid_agent_prompt(
    goal: str,
    df_mofs: pd.DataFrame,
    df_plants: pd.DataFrame,
) -> str:
    """
    Construct a prompt for the LLM that:
      - states the hybrid design goal
      - lists top MOFs (with model-based scores + descriptors)
      - lists top plants (with predicted photosynthetic efficiency)
      - asks for MOF–plant-inspired hybrid material ideas in JSON form,
        including cheminformatics fields (linker SMILES + metals) and
        a suggested CO₂-release method.
    """
    lines = []
    lines.append(
        "You are an AI assistant for designing hybrid materials inspired by both MOFs and plants."
    )
    lines.append("You are given:")
    lines.append("  - a ranked list of MOFs with model-based scores and structural descriptors;")
    lines.append("  - a ranked list of plants with predicted CO₂ assimilation performance.")
    lines.append("")
    lines.append(f"High-level goal: {goal}")
    lines.append("")

    # --- MOFs ---
    lines.append(
        "Top MOF candidates (higher score means more promising for CO₂ adsorption / affinity):"
    )
    for i, (_, row) in enumerate(df_mofs.iterrows(), start=1):
        lines.append(f"{i}. " + build_candidate_summary_row(row))
    lines.append("")

    # --- Plants ---
    lines.append(
        "Top plant candidates (higher predicted A means stronger CO₂ assimilation / photosynthetic performance):"
    )
    for i, (_, row) in enumerate(df_plants.iterrows(), start=1):
        lines.append(f"{i}. " + build_plant_summary_row(row))
    lines.append("")

    # --- Instructions + JSON schema ---
    lines.append(
        "Using both lists, do the following:\n"
        "1. Identify analogies between plant traits (e.g., photosynthetic pathway/subtype, high A, Ci,\n"
        "   response to light/temperature) and MOF properties (e.g., KH class/affinity, surface area,\n"
        "   density, pore volume, void fraction).\n"
        "2. Propose N hybrid material concepts, where N is specified by the user as `n_hybrids`.\n"
        "   Each hybrid combines one or more MOFs with one or more plant inspirations.\n"
        "3. For each hybrid, specify both:\n"
        "   - the structural concept (what the MOF / linker / functional groups look like);\n"
        "   - the plant-inspired functional analogy (how plant traits map to MOF behavior).\n"
        "4. For each hybrid, include a **cheminformatics** sub-object describing candidate organic linkers\n"
        "   and metal centers, suitable for sanity checks by RDKit and 3D visualization.\n"
        "5. Also suggest a plausible CO₂ release / regeneration method for that hybrid.\n"
        "\n"
        "Return a single JSON object with the following structure (no comments):\n"
        "{\n"
        '  "hybrid_ideas": [\n'
        "    {\n"
        '      "hybrid_id": "Hybrid1",\n'
        '      "based_on_mof_ids": ["<mof_id1>", "<mof_id2>"],\n'
        '      "based_on_plant_ids": ["<plant_id1>", "<plant_id2>"],\n'
        '      "concept": "Short description of the hybrid material idea",\n'
        '      "key_features": [\n'
        '        "bullet point about structural features",\n'
        '        "bullet point about adsorption behavior",\n'
        '        "bullet point about plant-inspired aspect"\n'
        "      ],\n"
        '      "cheminformatics": {\n'
        '        "linkers": [\n'
        "          {\n"
        '            "label": "CO₂-philic amine-functionalized linker",\n'
        '            "smiles": "CCN(CC)CCN",\n'
        '            "role": "captures CO₂ via reversible carbamate formation"\n'
        "          },\n"
        "          {\n"
        '            "label": "hydrophobic pore-modulating linker",\n'
        '            "smiles": "c1ccc(cc1)C(F)(F)F",\n'
        '            "role": "reduces water uptake while preserving CO₂ access"\n'
        "          }\n"
        "        ],\n"
        '        "metals": ["Zr", "Cu"]\n'
        "      },\n"
        '      "release_method": "temperature swing with mild heating under sweep gas",\n'
        '      "release_rationale": "explain why this method is appropriate for the proposed chemistry"\n'
        "    },\n"
        "    {\n"
        '      "hybrid_id": "Hybrid2",\n'
        "      ...\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "\n"
        "Requirements:\n"
        "- The array length of `hybrid_ideas` MUST equal the user-specified `n_hybrids`.\n"
        "- Every `cheminformatics.linkers[*].smiles` MUST be a valid SMILES string for a neutral\n"
        "  organic molecule (no counterions, no fragments like '.'), suitable for RDKit parsing.\n"
        "- `cheminformatics.metals` should list 1–3 metal symbols that are realistic for MOF synthesis\n"
        "  in this context (e.g., Zr, Zn, Cu, Mg, Al, etc.).\n"
        "- Avoid exotic or unstable species; prefer well-known MOF metals and linker families.\n"
        "\n"
        "First, give a 2–3 paragraph explanation of your reasoning. After that, output ONLY the JSON object."
    )

    return "\n".join(lines)


def run_llm_hybrid_agent(
    goal: str,
    top_k_mofs: int = 10,
    top_k_plants: int = 10,
    n_hybrids: int = 2,
) -> Dict[str, Any]:
    """
    Full pipeline for MOF–plant hybrid reasoning:
      - score MOFs with fused model
      - score plants with RF regressor
      - build a hybrid-focused LLM prompt
      - call LLM
      - return everything
    """
    # 1) Score MOFs
    df_mofs = score_mofs_with_fused_model(goal=goal, top_k=top_k_mofs)

    # 2) Score plants
    df_plants = score_plants_for_photosynthesis(top_k=top_k_plants)

    # 3) Build hybrid prompt (include n_hybrids as part of the goal text)
    goal_with_n = goal + f"\n\nUser-specified number of hybrid ideas (n_hybrids): {n_hybrids}."
    prompt = build_hybrid_agent_prompt(goal_with_n, df_mofs, df_plants)

    # 4) Call LLM
    llm_resp = call_llm(prompt)

    return {
        "goal": goal_with_n,
        "mofs_df": df_mofs,
        "plants_df": df_plants,
        "prompt": prompt,
        "llm_response": llm_resp,
    }


# ============================
# 8. JSON PARSING HELPERS
# ============================

def extract_json_block(text: str) -> Optional[str]:
    """
    Extract the *first* JSON object from an LLM response.

    Handles:
    - text before/after the JSON
    - accidental backticks
    - extra comments
    - whitespace issues

    Returns:
        raw_json_str or None
    """

    # 1) Remove code fences if present (```json ... ```)
    text = text.strip()
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)

    # 2) Find the first {...} block using a stack-based matcher
    stack = []
    start_index = None

    for i, ch in enumerate(text):
        if ch == "{":
            if start_index is None:
                start_index = i
            stack.append("{")
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack:
                    # Found matching closing brace
                    end_index = i + 1
                    return text[start_index:end_index]

    return None  # no valid JSON found


def parse_hybrid_json(text: str) -> Optional[dict]:
    """
    Parses the LLM hybrid-agent JSON output.

    Steps:
      - Extract JSON object from messy LLM output
      - Load via json.loads
      - Return Python dict or None
    """
    raw_json = extract_json_block(text)
    if raw_json is None:
        print("[WARN] No JSON object found in LLM response.")
        return None

    try:
        parsed = json.loads(raw_json)
        return parsed
    except json.JSONDecodeError as e:
        print("[WARN] JSON parsing failed:", e)
        print("Raw JSON was:", raw_json)
        return None


def link_hybrid_ideas_to_data(
    hybrid_json: dict,
    df_mofs: pd.DataFrame,
    df_plants: pd.DataFrame,
):
    """
    Given the parsed hybrid JSON, attach full MOF/plant data to each idea
    and propagate cheminformatics information (linker SMILES + metals).

    Returns a list of dicts, each like:
        {
           "hybrid_id": ...,
           "concept": ...,
           "key_features": [...],
           "release_method": str,
           "release_rationale": str,
           "mofs": <DataFrame of referenced MOFs>,
           "plants": <DataFrame of referenced plants>,
           "linker_smiles": [...],        # flattened list of SMILES for chem_tools
           "metals": [...],               # list of metal centers for chem_tools
           "cheminformatics": {...},      # original sub-object from LLM (optional)
        }
    """
    ideas = hybrid_json.get("hybrid_ideas", [])
    enriched = []

    for idea in ideas:
        mof_ids = idea.get("based_on_mof_ids", [])
        plant_ids = idea.get("based_on_plant_ids", [])

        # Find matching MOFs
        df_m = df_mofs[df_mofs["mof_id"].isin(mof_ids)].copy()

        # Find matching plants
        df_p = df_plants[df_plants["plant_id"].isin(plant_ids)].copy()

        # --- Cheminformatics block from LLM ---
        chem = idea.get("cheminformatics", {}) or {}
        linkers_def = chem.get("linkers", []) or []
        metals = chem.get("metals", []) or []

        # Flatten list of SMILES for chem_tools.validate_hybrid_idea_chemistry
        linker_smiles: List[str] = []
        for lk in linkers_def:
            if isinstance(lk, dict):
                smi = (lk.get("smiles") or "").strip()
                if smi:
                    linker_smiles.append(smi)

        enriched.append(
            {
                "hybrid_id": idea.get("hybrid_id", "UnknownHybrid"),
                "concept": idea.get("concept", ""),
                "key_features": idea.get("key_features", []),
                "release_method": idea.get("release_method", ""),
                "release_rationale": idea.get("release_rationale", ""),
                "mofs": df_m,
                "plants": df_p,
                # fields needed by chem_tools
                "linker_smiles": linker_smiles,
                "metals": metals,
                # keep full original chem sub-object in case you want it later
                "cheminformatics": chem,
            }
        )

    return enriched

