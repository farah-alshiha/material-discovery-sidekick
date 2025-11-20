import pandas as pd
from typing import List, Optional, Callable
from schema import Material, StructureFeatures, ChemistryFeatures, StabilityFeatures, PerformanceMetrics
from pathlib import Path

# Load local practice datasets
DATA_DIR = Path("./data")
mof_df = pd.read_csv('data/CR_data_CSD_modified_20250227.csv')
plant_df = pd.read_csv('data/Full_Aci_1.csv')


def infer_pathway_from_subtype(subtype: str) -> str:
    """
    Map the 'Subtype' field (e.g., NADP-ME, NAD-ME, PCK, CAM, C3) to a coarse
    photosynthesis pathway: C3, C4, or CAM.
    """
    if not isinstance(subtype, str):
        return "C3"
    s = subtype.strip().upper()
    if "CAM" in s:
        return "CAM"
    # Typical C4 biochemical subtypes
    if "NADP" in s or "NAD-ME" in s or "PCK" in s or "C4" in s:
        return "C4"
    # Fallback: assume C3
    return "C3"


def mof_row_to_material(row) -> Material:
    struct = StructureFeatures(
        lcd_angstrom=row.get("LCD (Å)"),
        pld_angstrom=row.get("PLD (Å)"),
        lfpd_angstrom=row.get("LFPD (Å)"),
        density_g_cm3=row.get("Density (g/cm3)"),
        asa_m2_g=row.get("ASA (m2/g)"),
        asa_m2_cm3=row.get("ASA (m2/cm3)"),
        nasa_m2_g=row.get("NASA (m2/g)"),
        nasa_m2_cm3=row.get("NASA (m2/cm3)"),
        pore_volume_cm3_g=row.get("PV (cm3/g)"),
        porosity=row.get("NPV (cm3/g)"),
        void_fraction=row.get("VF"),
    )
    chem = ChemistryFeatures(
        metal_nodes=row.get("Metal Types"),
        mofid_v1=row.get("mofid-v1"),
        mofid_v2=row.get("mofid-v2"),
        refcode=row.get("refcode"),
    )
    stab = StabilityFeatures(
        thermal_stability_C=row.get("Thermal_stability (℃)"),
        solvent_stability=row.get("Solvent_stability"),
        water_stability=row.get("Water_stability"),
    )
    perf = PerformanceMetrics(
        kh_class=row.get("KH_Classes"),
    )

    name = row.get("name")
    text_desc = (
        f"MOF {name or row.get('refcode', '')} with LCD={row.get('LCD (Å)')} Å, "
        f"PLD={row.get('PLD (Å)')} Å, density={row.get('Density (g/cm3)')} g/cm3, "
        f"ASA={row.get('ASA (m2/g)')} m2/g, KH class={row.get('KH_Classes')}."
    )

    return Material(
        id=str(row.get("coreid", row.get("refcode", row.get("number")))),
        name=name,
        material_type="MOF",
        source="local_mof_csv",
        structure=struct,
        chemistry=chem,
        stability=stab,
        performance=perf,
        text_description=text_desc,
        extra=row.to_dict(),
    )

def plant_row_to_material(row) -> Material:
    species_col = "Species " if "Species " in row.index else "Species"
    ci_col = "Ci " if "Ci " in row.index else "Ci"

    struct = StructureFeatures()
    chem = ChemistryFeatures(
        c3_c4_cam=row.get("Subtype"),
    )
    stab = StabilityFeatures(
        thermal_stability_C=row.get("Temp"),
    )
    perf = PerformanceMetrics(
        co2_assimilation_umol_m2_s=row.get("A"),
    )

    text_desc = (
        f"Plant {row[species_col]} ({row['Subtype']} subtype) with CO₂ assimilation rate A={row['A']}, "
        f"intercellular CO₂ Ci={row[ci_col]}, measured at {row['Temp']} °C and pressure={row['Pressure']}."
    )

    return Material(
        id=f"plant_{row.name}",
        name=row[species_col],
        material_type="Plant",
        source="local_mock_plant_dataset",
        structure=struct,
        chemistry=chem,
        stability=stab,
        performance=perf,
        text_description=text_desc,
        extra=row.to_dict(),
    )

MOF_MATERIALS: List[Material] = [mof_row_to_material(row) for _, row in mof_df.iterrows()]
PLANT_MATERIALS: List[Material] = [plant_row_to_material(row) for _, row in plant_df.iterrows()]

import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")  # adjust if you're using another directory


def _load_mof_csv(path: Path, origin_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["dataset_origin"] = origin_name
    return df


def fetch_mofs_mock(extra_mof_paths=None) -> pd.DataFrame:
    """
    Load the base MOF dataset and optionally merge additional MOF datasets.

    extra_mof_paths:
        - None  -> just load the base dataset
        - list of paths (str or Path) -> load and concatenate them.

    Assumes each CSV has at least a unique MOF identifier column
    (e.g. 'mof_id' or 'id') and optionally 'doi'.
    """
    # --- 1) base dataset ---
    base_path = DATA_DIR / "CR_data_CSD_modified_20250227.csv"  # your current main MOF CSV
    df_base = _load_mof_csv(base_path, origin_name="base_CR_CSD")

    dfs = [df_base]

    # --- 2) extra datasets (one at a time or many) ---
    if extra_mof_paths:
        for p in extra_mof_paths:
            p = Path(p)
            origin = p.stem  # e.g. "NewMOFs2025"
            dfs.append(_load_mof_csv(p, origin_name=origin))

    df_all = pd.concat(dfs, ignore_index=True)

    # --- 3) basic deduplication ---
    # Prefer to deduplicate by mof_id and doi if present
    keys = []
    if "mof_id" in df_all.columns:
        keys.append("mof_id")
    elif "id" in df_all.columns:
        keys.append("id")

    if "doi" in df_all.columns:
        keys.append("doi")

    if keys:
        df_all = df_all.drop_duplicates(subset=keys)

    return df_all



def fetch_plants_mock() -> pd.DataFrame:
    """
    Return a unified plant DataFrame from:
      - base A/Ci dataset (C3 / C4 with true subtypes if available)
      - CAM cross-sections
      - extra C3 A/Ci datasets

    Ensures:
      - Pathway = {C3, C4, CAM}
      - Subtype is more specific than Pathway (not just a copy)
      - dataset_origin marks where each row came from
    """

    dfs = []

    # 1) Base A/Ci dataset (the original one you were using, e.g. Full_Aci_1.csv)
    base_path = DATA_DIR / "Full_Aci_1.csv"
    if base_path.exists():
        df_base = pd.read_csv(base_path)

        # Make sure we have a Pathway column; you may already have one
        if "Pathway" not in df_base.columns:
            # Example: infer from an existing column or default to C3
            # Adjust this logic to your real schema.
            df_base["Pathway"] = df_base.get("Pathway_in", "C3")

        # Keep existing Subtype if present (NADP-ME, NAD-ME, etc.)
        if "Subtype" not in df_base.columns:
            df_base["Subtype"] = "Unknown_subtype"

        df_base["dataset_origin"] = "base_Aci"
        dfs.append(df_base)

    # 2) CAM cross-section dataset
    cam_path = DATA_DIR / "crosssections for Cam imputed.csv"
    if cam_path.exists():
        df_cam = pd.read_csv(cam_path)

        df_cam["Pathway"] = "CAM"
        # Subtype is now informative, not just "CAM"
        df_cam["Subtype"] = "CAM_crosssections"
        df_cam["dataset_origin"] = "cam_crosssections"

        dfs.append(df_cam)

    # 3) C3 ACi curves V1
    c3_v1_path = DATA_DIR / "C3-ACi-curves-V1.csv"
    if c3_v1_path.exists():
        df_c3_v1 = pd.read_csv(c3_v1_path)

        df_c3_v1["Pathway"] = "C3"
        df_c3_v1["Subtype"] = "C3_ACi_V1"
        df_c3_v1["dataset_origin"] = "c3_aci_v1"

        dfs.append(df_c3_v1)

    # 4) C3 ACi curves V1 (DPK variant)
    c3_v1_dpk_path = DATA_DIR / "C3-ACi-curves-V1-DPK.csv"
    if c3_v1_dpk_path.exists():
        df_c3_v1_dpk = pd.read_csv(c3_v1_dpk_path)

        df_c3_v1_dpk["Pathway"] = "C3"
        df_c3_v1_dpk["Subtype"] = "C3_ACi_V1_DPK"
        df_c3_v1_dpk["dataset_origin"] = "c3_aci_v1_dpk"

        dfs.append(df_c3_v1_dpk)

    if not dfs:
        raise RuntimeError("No plant datasets found in DATA_DIR.")

    # ---- Merge all datasets ----
    df_all = pd.concat(dfs, ignore_index=True)

    # Normalize plant id / name if needed
    if "id" not in df_all.columns:
        df_all["id"] = df_all.index.astype(str)
    if "name" not in df_all.columns:
        # Try to use Species / SpeciesName if available
        name_col = None
        for c in df_all.columns:
            if c.lower() in ("species", "speciesname", "plant_name"):
                name_col = c
                break
        if name_col:
            df_all["name"] = df_all[name_col].astype(str)
        else:
            df_all["name"] = "Unknown_plant"

    # This is what the rest of your pipeline expects:
    # - id
    # - name
    # - Pathway
    # - Subtype
    # - dataset_origin
    return df_all



