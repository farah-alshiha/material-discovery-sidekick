from closed_loop_agent import run_closed_loop_optimization

def main():
    goal = (
        "Optimize MOFs for CO₂ capture from flue gas, balancing strong affinity with high accessible surface area "
        "and sufficient void fraction, while avoiding extremely dense materials that might limit diffusion."
    )

    result = run_closed_loop_optimization(
        goal=goal,
        max_iters=3,
        top_k=15,
    )

    print("\n=== CLOSED-LOOP OPTIMIZATION SUMMARY ===")
    print("Goal:", result["goal"])
    print("Final weights:", result["final_weights"])

    for it in result["iterations"]:
        print(f"\n--- Iteration {it['iteration']} ---")
        print("Weights:", it["weights"])
        print("Top 5 MOFs under these weights:")
        cols = [c for c in ["mof_id", "kh_class_name", "utility", "score", "density_g_cm3", "asa_m2_g", "void_fraction"] if c in it["top_mofs"].columns]
        print(it["top_mofs"][cols].head())

        print("\nLLM parsed weights JSON:")
        print(it["parsed_weights"])

if __name__ == "__main__":
    main()
