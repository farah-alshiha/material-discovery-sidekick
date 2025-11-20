# enzyme_map.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

# Base directory for local PDB files
PDB_DIR = Path(__file__).resolve().parent / "pdb"


@dataclass
class Enzyme:
    name: str
    role: str
    structure_type: str  # "protein" or "ligand"
    pdb_path: str | None = None
    smiles: str | None = None


# ----------------------------------------------------
# 1) CENTRAL REGISTRY OF ENZYMES (EDIT THIS FREELY)
# ----------------------------------------------------
# You just need to:
#   - put the PDB in pdb/<filename>.pdb
#   - add an entry here that points to it
#
# structure_type="protein"  -> uses pdb_path
# structure_type="ligand"   -> uses smiles (RDKit)
#

def _pdb(name: str) -> str | None:
    """Helper: return path to pdb/<name>.pdb if it exists, else None."""
    p = PDB_DIR / f"{name}.pdb"
    return str(p) if p.exists() else None


ENZYME_REGISTRY: Dict[str, Enzyme] = {
    # ---- Already existing example: RuBisCO ----
    "RUBISCO": Enzyme(
        name="RuBisCO",
        role="Key CO₂-fixing enzyme of the Calvin cycle in C₃, C₄ and CAM plants.",
        structure_type="protein",
        pdb_path=_pdb("rubisco"),
    ),

    # ---- EXAMPLE NEW ENZYME: PEPC (C₄ / CAM) ----
    "PEPC": Enzyme(
        name="Phosphoenolpyruvate carboxylase (PEPC)",
        role="Initial CO₂-fixing enzyme in C₄ and CAM pathways.",
        structure_type="protein",
        pdb_path=_pdb("pepc"),   # requires pdb/pepc.pdb
    ),

    # ---- EXAMPLE NEW ENZYME: NADP-ME ----
    "NADP_ME": Enzyme(
        name="NADP-malic enzyme",
        role="Decarboxylates malate in C₄ NADP-ME subtype chloroplasts.",
        structure_type="protein",
        pdb_path=_pdb("nadp_me"),  # requires pdb/nadp_me.pdb
    ),

    # ---- EXAMPLE LIGAND ENZYME (SMILES ONLY) ----
    # This is if you want to show just a small molecule (ligand) instead of a protein:
    "MALATE": Enzyme(
        name="Malate (ligand)",
        role="Organic acid shuttling CO₂ equivalents in C₄ and CAM cycles.",
        structure_type="ligand",
        smiles="C(C(=O)O)C(C(=O)O)O",  # example SMILES for malate
    ),
}

# You can ALWAYS add more, e.g.:
# "MDH": Enzyme(...), with pdb_path=_pdb("mdh")
# "CA":  carbonic anhydrase, etc.


# ----------------------------------------------------
# 2) MAPPING LOGIC FROM PLANT ROW -> WHICH ENZYMES
# ----------------------------------------------------
# This is where you say:
#   - C₃ plants → RuBisCO + maybe CA
#   - C₄ NADP-ME subtype → RuBisCO + PEPC + NADP-ME
#   - CAM → RuBisCO + PEPC + MALATE, etc.
#
# You can make this as detailed as you like.
# ----------------------------------------------------

def enzymes_for_plant_row(row: Dict[str, Any]) -> List[Enzyme]:
    """
    Given a plant row (from the merged plant DataFrame), return a list of Enzyme
    objects that should be visualized / used as inspiration for this plant.
    """
    pathway = str(row.get("Pathway", "")).upper()   # C3 / C4 / CAM
    subtype = str(row.get("Subtype", "")).upper()   # e.g. "NADP-ME", "NAD-ME", "PCK", ...
    species = str(row.get("name", row.get("Species", ""))).lower()

    enzymes: List[Enzyme] = []

    # ---- All higher plants: RuBisCO baseline ----
    rubisco = ENZYME_REGISTRY.get("RUBISCO")
    if rubisco and rubisco.pdb_path:
        enzymes.append(rubisco)

    # ---- C4 and CAM: add PEPC (if file exists) ----
    if "C4" in pathway or "CAM" in pathway:
        pepc = ENZYME_REGISTRY.get("PEPC")
        if pepc and pepc.pdb_path:
            enzymes.append(pepc)

    # ---- C4 NADP-ME subtype: add NADP-ME enzyme ----
    if "C4" in pathway and "NADP" in subtype:
        nadp_me = ENZYME_REGISTRY.get("NADP_ME")
        if nadp_me and nadp_me.pdb_path:
            enzymes.append(nadp_me)

    # ---- CAM: emphasize malate ligand as a 3D small molecule ----
    if "CAM" in pathway:
        mal = ENZYME_REGISTRY.get("MALATE")
        if mal and mal.smiles:
            enzymes.append(mal)

    # EXAMPLE: if you want species-specific logic:
    # if "zea mays" in species:
    #     # add a Zea mays specific PEPC if you want a different PDB
    #     pass

    return enzymes
