import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from llm_agent import (
    build_fusion_dataset,
    train_fused_xgb,
    call_llm,
)
from gnn_data import MofGraphDataset


# ============================
# 1. Utilities
# ============================

def normalize_series(s: pd.Series) -> pd.Series:
    """Min-max normalize a numeric series to [0, 1]. If constant, returns 0.5."""
    s = s.astype(float)
    s_min = s.min()
    s_max = s.max()
    if s_max <= s_min:
        return pd.Series(0.5, index=s.index)
    return (s - s_min) / (s_max - s_min)


def compute_multiobjective_scores(
    df: pd.DataFrame,
    weights: Dict[str, float],
) -> pd.Series:
    """
    Compute a scalar utility score per MOF based on:
      - affinity (model probability)
      - ASA
      - void fraction
      - density (as a penalty)

    df is expected to contain:
      - 'score' (model affinity, e.g. P(strong class))
      - 'asa_m2_g'
      - 'void_fraction'
      - 'density_g_cm3'

    weights has keys:
      - 'affinity', 'asa', 'void_fraction', 'density'
    """
    # Ensure all required columns exist, fill if missing
    cols = ["score", "asa_m2_g", "void_fraction", "density_g_cm3"]
    for c in cols:
        if c not in df.columns:
            df[c] = df[c].fillna(df[c].mean()) if c in df else 0.0

    s_aff = normalize_series(df["score"])
    s_asa = normalize_series(df["asa_m2_g"])
    s_vf = normalize_series(df["void_fraction"])
    s_dens = normalize_series(df["density_g_cm3"])

    w_aff = weights.get("affinity", 1.0)
    w_asa = weights.get("asa", 0.5)
    w_vf = weights.get("void_fraction", 0.5)
    w_dens = weights.get("density", 0.3)

    # Higher density is a penalty, so subtract
    utility = (
        w_aff * s_aff +
        w_asa * s_asa +
        w_vf * s_vf -
        w_dens * s_dens
    )
    return utility


def extract_json_block(text: str) -> Optional[str]:
    """
    Extract the first JSON object from an LLM response.
    This is a copy of the logic you used for hybrid parsing, adapted locally.
    """
    text = text.strip()
    # Remove ```json fences if present
    text = text.replace("```json", "").replace("```", "")

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
                    end_index = i + 1
                    return text[start_index:end_index]
    return None


def parse_weight_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse LLM output containing new weights.

    Expected JSON structure:
      {
        "new_weights": {
          "affinity": 1.0,
          "asa": 0.8,
          "void_fraction": 0.6,
          "density": 0.3
        },
        "rationale": "..."
      }
    """
    raw_json = extract_json_block(text)
    if raw_json is None:
        print("[WARN] No JSON found in LLM response.")
        return None

    try:
        parsed = json.loads(raw_json)
        return parsed
    except json.JSONDecodeError as e:
        print("[WARN] Failed to parse JSON:", e)
        print("Raw JSON was:", raw_json)
        return None


# ============================
# 2. Prompt building
# ============================

def build_closed_loop_prompt(
    goal: str,
    current_weights: Dict[str, float],
    df_top: pd.DataFrame,
    iteration: int,
) -> str:
    """
    Build a prompt that:
      - reminds the LLM of the global goal,
      - shows current multi-objective weights,
      - lists current top MOFs under those weights,
      - asks the LLM to propose updated weights in JSON.
    """
    lines = []
    lines.append("You are an expert assistant for multi-objective optimization of MOFs.")
    lines.append("We are using a scalar utility that combines:")
    lines.append("  - affinity (model-predicted KH / gas-adsorption score),")
    lines.append("  - accessible surface area (ASA),")
    lines.append("  - void fraction,")
    lines.append("  - density (as a penalty term).")
    lines.append("")
    lines.append(f"Global goal: {goal}")
    lines.append("")
    lines.append(f"Current iteration: {iteration}")
    lines.append("Current weights used in the scalar utility:")
    lines.append(f"  affinity:      {current_weights.get('affinity', 1.0):.3f}")
    lines.append(f"  asa:           {current_weights.get('asa', 0.5):.3f}")
    lines.append(f"  void_fraction: {current_weights.get('void_fraction', 0.5):.3f}")
    lines.append(f"  density:       {current_weights.get('density', 0.3):.3f} (penalty)")
    lines.append("")
    lines.append("Top MOF candidates under these weights (higher utility is better):")
    for i, (_, row) in enumerate(df_top.iterrows(), start=1):
        lines.append(
            f"{i}. ID={row['mof_id']} | KH_class={row['kh_class_name']} | "
            f"utility={row['utility']:.3f} | score={row['score']:.3f} | "
            f"density={row.get('density_g_cm3', 'NA')}, ASA={row.get('asa_m2_g', 'NA')}, "
            f"void_fraction={row.get('void_fraction', 'NA')}"
        )
    lines.append("")
    lines.append(
        "Please analyze how well these candidates match the global goal, and then propose updated weights.\n"
        "Return a JSON object with this structure:\n"
        "{\n"
        '  "new_weights": {\n'
        '    "affinity": <float>,\n'
        '    "asa": <float>,\n'
        '    "void_fraction": <float>,\n'
        '    "density": <float>\n'
        "  },\n"
        '  "rationale": "Short explanation of why these weights are better aligned with the goal."\n'
        "}\n"
        "You may slightly increase or decrease the weights, but keep them in a reasonable range (e.g., 0.0–2.0).\n"
        "Do not include any other keys at the top level except 'new_weights' and 'rationale'."
    )

    return "\n".join(lines)


# ============================
# 3. Closed-loop optimizer
# ============================

def run_closed_loop_optimization(
    goal: str,
    max_iters: int = 3,
    top_k: int = 15,
) -> Dict[str, Any]:
    """
    Full closed-loop pipeline:

      1. Build fused dataset and train fused XGBoost model.
      2. Compute baseline affinity scores for all MOFs.
      3. Initialize multi-objective weights.
      4. For each iteration:
           a. Compute multi-objective utility for each MOF.
           b. Take top_k MOFs under current utility.
           c. Send them + current weights to LLM.
           d. Parse new weights from LLM's JSON response.
      5. Return a log of all iterations and the final weights.

    Returns a dict:
      {
        "goal": ...,
        "class_names": [...],
        "iterations": [
           {
             "iteration": int,
             "weights": {...},
             "top_mofs": <DataFrame>,
             "llm_response": str,
             "parsed_weights": {...} or None,
           },
           ...
        ],
        "final_weights": {...}
      }
    """
    # ---- 1) Build dataset + model ----
    X_fused, y, df_merged, class_names = build_fusion_dataset()
    model = train_fused_xgb(X_fused, y)
    proba = model.predict_proba(X_fused)

    # We treat "strong" or last class as affinity dimension
    # (same logic as in score_mofs_with_fused_model)
    dataset = MofGraphDataset(root="./gnn_data", cif_dir="./cifs")
    cls_names = dataset.classes
    strong_indices = [i for i, cls in enumerate(cls_names) if "strong" in cls.lower()]
    if strong_indices:
        target_col = strong_indices[0]
    else:
        target_col = proba.shape[1] - 1

    df_all = df_merged.copy()
    df_all["score"] = proba[:, target_col]
    df_all["kh_class_name"] = [cls_names[label] for label in df_all["kh_label_reindexed"].astype(int)]

    # ---- 2) Initialize weights ----
    current_weights = {
        "affinity": 1.0,
        "asa": 0.7,
        "void_fraction": 0.7,
        "density": 0.5,  # penalty
    }

    iterations_log: List[Dict[str, Any]] = []

    for it in range(1, max_iters + 1):
        # 3a) Compute utility under current weights
        df_all["utility"] = compute_multiobjective_scores(df_all, current_weights)

        # 3b) Take top_k
        df_top = df_all.sort_values("utility", ascending=False).head(top_k).reset_index(drop=True)

        # 3c) Build prompt and call LLM
        prompt = build_closed_loop_prompt(goal, current_weights, df_top, iteration=it)
        llm_resp = call_llm(prompt)

        # 3d) Parse new weights
        parsed = parse_weight_json(llm_resp)
        if parsed and "new_weights" in parsed:
            new_w = parsed["new_weights"]
            # Safely update only known keys
            for key in ["affinity", "asa", "void_fraction", "density"]:
                if key in new_w:
                    try:
                        val = float(new_w[key])
                    except (TypeError, ValueError):
                        continue
                    # Optional: clamp to [0, 2]
                    val = max(0.0, min(2.0, val))
                    current_weights[key] = val
        else:
            print(f"[WARN] Iteration {it}: could not parse new weights, keeping previous ones.")

        iterations_log.append(
            {
                "iteration": it,
                "weights": current_weights.copy(),
                "top_mofs": df_top.copy(),
                "llm_response": llm_resp,
                "parsed_weights": parsed,
            }
        )

    result = {
        "goal": goal,
        "class_names": cls_names,
        "iterations": iterations_log,
        "final_weights": current_weights,
    }
    return result
