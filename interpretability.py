from typing import Dict, Any

def describe_mof_features(row: dict) -> str:
    """Build a short, deterministic description from MOF numerical features."""
    kh = row.get("kh_class_name", "unknown")
    asa = row.get("asa_m2_g", None)
    vf = row.get("void_fraction", None)
    dens = row.get("density_g_cm3", None)

    bits = [f"KH class: {kh}"]

    if asa is not None:
        if asa > 3000:
            bits.append(f"very high surface area (~{asa:.0f} m²/g)")
        elif asa > 1500:
            bits.append(f"high surface area (~{asa:.0f} m²/g)")
        else:
            bits.append(f"moderate surface area (~{asa:.0f} m²/g)")

    if vf is not None:
        if vf > 0.8:
            bits.append(f"very high void fraction ({vf:.2f})")
        elif vf > 0.5:
            bits.append(f"high void fraction ({vf:.2f})")
        else:
            bits.append(f"more compact pores ({vf:.2f})")

    if dens is not None:
        if dens > 1.5:
            bits.append(f"relatively dense framework ({dens:.2f} g/cm³)")
        else:
            bits.append(f"moderate density ({dens:.2f} g/cm³)")

    return "; ".join(bits)


def build_mof_evidence(row: dict) -> Dict[str, Any]:
    """
    Given a row from df_mofs, return a compact 'evidence' package for the LLM/UI.
    """
    doi = row.get("doi") or row.get("DOI") or None

    return {
        "mof_id": row.get("mof_id"),
        "kh_class_name": row.get("kh_class_name"),
        "feature_summary": describe_mof_features(row),
        "doi": doi,
        "doi_citation_hint": (
            f"This MOF was studied in DOI {doi}, which provides adsorption "
            "data and structural details relevant to gas storage and CO₂ capture."
            if doi else None
        ),
    }
    
