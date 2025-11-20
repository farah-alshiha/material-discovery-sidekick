from llm_agent import run_llm_hybrid_agent, parse_hybrid_json, link_hybrid_ideas_to_data
from viz import (
    plot_mof_score_hist,
    plot_mof_property_vs_score,
    plot_plant_predictedA_hist,
    plot_plant_subtype_boxplot,
    plot_hybrid_mof_scores,
    print_hybrid_summary,
)


def main():
    goal = (
        "Design MOF–plant-inspired hybrid materials for ultra-efficient CO₂ capture and oxygen balance. "
        "Use high-affinity, high-surface-area MOFs together with plant species exhibiting strong CO₂ "
        "assimilation and robust environmental performance."
    )

    result = run_llm_hybrid_agent(
        goal=goal,
        top_k_mofs=10,
        top_k_plants=10,
    )

    print("\n=== HYBRID AGENT GOAL ===")
    print(result["goal"])

    df_mofs = result["mofs_df"]
    df_plants = result["plants_df"]

    # -----------------------
    # Basic visualizations
    # -----------------------
    plot_mof_score_hist(df_mofs, bins=10, show=True)
    plot_mof_property_vs_score(df_mofs, property_col="asa_m2_g", show=True)
    plot_plant_predictedA_hist(df_plants, bins=10, show=True)
    plot_plant_subtype_boxplot(df_plants, show=True)

    # -----------------------
    # Parse LLM JSON + hybrid-level plots
    # -----------------------
    hybrid_json = parse_hybrid_json(result["llm_response"])

    if hybrid_json:
        enriched = link_hybrid_ideas_to_data(
            hybrid_json,
            df_mofs=df_mofs,
            df_plants=df_plants,
        )

        print_hybrid_summary(enriched)
        plot_hybrid_mof_scores(enriched, show=True)
    else:
        print("[ERROR] Could not parse JSON from the LLM response.")


if __name__ == "__main__":
    main()
