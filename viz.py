# viz.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Any


# ==============
# 1. MOF plots
# ==============

def plot_mof_score_hist(df_mofs: pd.DataFrame, bins: int = 20, show: bool = True, save_path: str = None):
    """
    Plot a histogram of MOF model scores.
    Expects df_mofs to have a 'score' column.
    """
    scores = df_mofs["score"].values

    plt.figure()
    plt.hist(scores, bins=bins)
    plt.xlabel("Model score (e.g., P(strong KH class))")
    plt.ylabel("Count")
    plt.title("Distribution of MOF scores")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    plt.close()


def plot_mof_property_vs_score(
    df_mofs: pd.DataFrame,
    property_col: str = "asa_m2_g",
    show: bool = True,
    save_path: str = None,
):
    """
    Scatter plot of a chosen MOF property vs model score.

    property_col options (given your features):
      - 'density_g_cm3'
      - 'asa_m2_g'
      - 'pore_volume_cm3_g'
      - 'void_fraction'
      (or any other continuous column present in df_mofs)
    """
    if property_col not in df_mofs.columns:
        raise ValueError(f"{property_col} not found in df_mofs.columns")

    x = df_mofs[property_col].values
    y = df_mofs["score"].values

    plt.figure()
    plt.scatter(x, y)
    plt.xlabel(property_col)
    plt.ylabel("Model score")
    plt.title(f"{property_col} vs model score (MOFs)")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    plt.close()


# ==============
# 2. Plant plots
# ==============

def plot_plant_predictedA_hist(df_plants: pd.DataFrame, bins: int = 20, show: bool = True, save_path: str = None):
    """
    Plot a histogram of predicted CO₂ assimilation A for plants.

    Expects df_plants to have 'predicted_A' column (from score_plants_for_photosynthesis).
    """
    if "predicted_A" not in df_plants.columns:
        raise ValueError("'predicted_A' not found in df_plants. Did you pass the scored DataFrame?")

    A = df_plants["predicted_A"].values

    plt.figure()
    plt.hist(A, bins=bins)
    plt.xlabel("Predicted CO₂ assimilation A (µmol m⁻² s⁻¹)")
    plt.ylabel("Count")
    plt.title("Distribution of predicted A (plants)")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    plt.close()


def plot_plant_subtype_boxplot(df_plants: pd.DataFrame, show: bool = True, save_path: str = None):
    """
    Boxplot of predicted A grouped by plant Subtype (e.g., C3, C4, NADP-ME).

    Expects 'Subtype' and 'predicted_A' columns.
    """
    if "Subtype" not in df_plants.columns or "predicted_A" not in df_plants.columns:
        raise ValueError("Expected columns 'Subtype' and 'predicted_A' in df_plants.")

    # Group by subtype
    grouped = df_plants.groupby("Subtype")["predicted_A"].apply(list)
    labels = list(grouped.index)
    data = [grouped[subtype] for subtype in labels]

    plt.figure()
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.xlabel("Plant Subtype")
    plt.ylabel("Predicted CO₂ assimilation A (µmol m⁻² s⁻¹)")
    plt.title("Predicted A by plant subtype")

    plt.xticks(rotation=30, ha="right")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    plt.close()


# ==============
# 3. Hybrid ideas plots
# ==============

def plot_hybrid_mof_scores(enriched_hybrids: List[Dict[str, Any]], show: bool = True, save_path: str = None):
    """
    For each hybrid idea, compute the mean MOF score of the referenced MOFs and plot as a bar chart.

    enriched_hybrids: output of link_hybrid_ideas_to_data(...)
      - each element has:
          'hybrid_id': str
          'mofs': DataFrame with 'score' column
    """
    hybrid_ids = []
    mean_scores = []

    for idea in enriched_hybrids:
        hid = idea.get("hybrid_id", "UnknownHybrid")
        df_mofs = idea.get("mofs", None)
        if df_mofs is None or df_mofs.empty:
            continue

        if "score" not in df_mofs.columns:
            continue

        hybrid_ids.append(hid)
        mean_scores.append(df_mofs["score"].mean())

    if not hybrid_ids:
        print("[WARN] No MOF scores found in enriched hybrids.")
        return

    x = np.arange(len(hybrid_ids))

    plt.figure()
    plt.bar(x, mean_scores)
    plt.xticks(x, hybrid_ids, rotation=20, ha="right")
    plt.ylabel("Mean MOF model score")
    plt.title("Hybrid ideas: mean MOF score per hybrid")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    plt.close()


def print_hybrid_summary(enriched_hybrids: List[Dict[str, Any]]):
    """
    Console-friendly summary of hybrid ideas:
      - hybrid_id
      - concept
      - how many MOFs and plants it references
    """
    print("\n=== Hybrid ideas summary ===")
    for idea in enriched_hybrids:
        hid = idea.get("hybrid_id", "UnknownHybrid")
        concept = idea.get("concept", "")
        mofs = idea.get("mofs", pd.DataFrame())
        plants = idea.get("plants", pd.DataFrame())
        print(f"\n--- {hid} ---")
        print("Concept:", concept)
        print(f"Referenced MOFs: {len(mofs)}")
        print(f"Referenced plants: {len(plants)}")
