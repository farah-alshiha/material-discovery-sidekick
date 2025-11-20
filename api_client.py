from dataclasses import dataclass
from typing import List, Optional
import pandas as pd


# ============================
# Data classes
# ============================

@dataclass
class MofMaterial:
    mof_id: str
    density_g_cm3: Optional[float]
    asa_m2_g: Optional[float]
    void_fraction: Optional[float]
    kh_label_reindexed: int
    doi: Optional[str]
    raw: dict


@dataclass
class PlantRecord:
    plant_id: str
    name: str
    subtype: Optional[str]
    predicted_A: Optional[float]
    Ci: Optional[float]
    Temp: Optional[float]
    Qobs: Optional[float]
    raw: dict


# ============================
# MOF loader (from CSV)
# ============================

def fetch_mofs_from_csv(path: str) -> List[MofMaterial]:
    """
    Load MOF materials from a CSV file and return a list of MofMaterial.

    Assumes columns similar to your CR_data_CSD_modified_20250227.csv, e.g.:
      - 'REFCODE' or 'mof_id'
      - 'density_g_cm3'
      - 'ASA_m2_g' or 'asa_m2_g'
      - 'void_fraction'
      - 'kh_label_reindexed' (int)
    """
    df = pd.read_csv(path)

    # Try to be robust to slightly different column names
    id_col = "mof_id"
    if "mof_id" not in df.columns:
        if "REFCODE" in df.columns:
            id_col = "REFCODE"
        elif "refcode" in df.columns:
            id_col = "refcode"

    asa_col = "asa_m2_g"
    for cand in ["ASA_m2_g", "ASA (m2/g)", "asa_m2_g"]:
        if cand in df.columns:
            asa_col = cand
            break

    density_col = "density_g_cm3"
    for cand in ["density_g_cm3", "Density (g/cm3)", "Density"]:
        if cand in df.columns:
            density_col = cand
            break

    vf_col = "void_fraction"
    for cand in ["void_fraction", "Void fraction", "VF"]:
        if cand in df.columns:
            vf_col = cand
            break

    label_col = "kh_label_reindexed"
    for cand in ["kh_label_reindexed", "KH_label_reindexed", "kh_label"]:
        if cand in df.columns:
            label_col = cand
            break

    doi_col = None
    for cand in ["doi", "DOI", "source_doi", "lit_doi"]:
        if cand in df.columns:
            doi_col = cand
            break

    materials = []
    for _, row in df.iterrows():
        materials.append(
            MofMaterial(
                mof_id=str(row[id_col]),
                density_g_cm3=_safe_float(row.get(density_col)),
                asa_m2_g=_safe_float(row.get(asa_col)),
                void_fraction=_safe_float(row.get(vf_col)),
                kh_label_reindexed=int(row[label_col]),
                doi=str(row[doi_col]).strip() if doi_col and not pd.isna(row[doi_col]) else None,
                raw=row.to_dict(),
            )
        )
    return materials


# ============================
# Plant loader (from CSV)
# ============================

def fetch_plants_from_csv(path: str) -> List[PlantRecord]:
    """
    Load plant records from a CSV file and return a list of PlantRecord.

    Assumes columns similar to your Full_Aci_1.csv, e.g.:
      - 'PlantID' or 'id'
      - 'Name' or 'Species'
      - 'Subtype'
      - 'predicted_A' or 'A'
      - 'Ci'
      - 'Temp' (leaf temp)
      - 'Qobs' (light)
    """
    df = pd.read_csv(path)

    id_col = "plant_id"
    if "plant_id" not in df.columns:
        for cand in ["PlantID", "ID", "id"]:
            if cand in df.columns:
                id_col = cand
                break

    name_col = "Name"
    for cand in ["Name", "Species", "species"]:
        if cand in df.columns:
            name_col = cand
            break

    subtype_col = "Subtype"
    for cand in ["Subtype", "Type"]:
        if cand in df.columns:
            subtype_col = cand
            break

    A_col = "predicted_A"
    for cand in ["predicted_A", "A"]:
        if cand in df.columns:
            A_col = cand
            break

    ci_col = "Ci" if "Ci" in df.columns else None
    temp_col = "Temp" if "Temp" in df.columns else None
    q_col = "Qobs" if "Qobs" in df.columns else None

    plants: List[PlantRecord] = []
    for _, row in df.iterrows():
        plants.append(
            PlantRecord(
                plant_id=str(row[id_col]),
                name=str(row[name_col]),
                subtype=str(row[subtype_col]) if subtype_col in df.columns else None,
                predicted_A=_safe_float(row.get(A_col)) if A_col in df.columns else None,
                Ci=_safe_float(row.get(ci_col)) if ci_col else None,
                Temp=_safe_float(row.get(temp_col)) if temp_col else None,
                Qobs=_safe_float(row.get(q_col)) if q_col else None,
                raw=row.to_dict(),
            )
        )
    return plants


def _safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
