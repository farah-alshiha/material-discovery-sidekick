from llm_agent import run_llm_agent_once

def main():
    # You can customize this goal text freely
    goal = (
        "Identify MOFs that are promising candidates for CO₂ capture from flue gas, "
        "prioritizing strong affinity (high KH class), high surface area, and reasonable porosity. "
        "Also consider that very dense materials may suffer from diffusion limitations."
    )

    result = run_llm_agent_once(goal=goal, top_k=15)

    print("\n=== AGENT GOAL ===")
    print(result["goal"])

    print("\n=== TOP CANDIDATES (from model) ===")
    print(result["candidates_df"][["mof_id", "kh_class_name", "score"]].head())

    print("\n=== PROMPT SENT TO LLM (truncated) ===")
    print(result["prompt"][:1000], "...\n")

    print("=== LLM RAW RESPONSE ===")
    print(result["llm_response"])


if __name__ == "__main__":
    main()
