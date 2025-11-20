from typing import List, Dict, Any, Optional
import io

# --- Optional imports (RDKit, py3Dmol) ---
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem
    _HAS_RDKIT = True
except Exception:
    _HAS_RDKIT = False

try:
    import py3Dmol
    _HAS_PY3DMOL = True
except Exception:
    _HAS_PY3DMOL = False


def smiles_is_available() -> bool:
    """
    True if RDKit is available for SMILES parsing / drawing.
    """
    return _HAS_RDKIT


# ---------------------------------------------------
# 1) LINKER VALIDATION (lightweight, for this context)
# ---------------------------------------------------
def validate_hybrid_idea_chemistry(idea: Dict[str, Any]) -> Dict[str, Any]:
    """
    Very lightweight validation of linkers in a hybrid idea.

    Expected `idea` structure:
      idea["linker_smiles"] : list of SMILES strings  (or empty / missing)

    Returns:
      {
        "linkers": [
          {
            "smiles": <str>,
            "validation": {
               "is_valid": bool,
               "formula": Optional[str],
               "mol_weight": float,
               "n_atoms": int,
               "errors": [str],
               "warnings": [str],
            }
          },
          ...
        ],
        "metals": idea.get("metals", []),
      }
    """
    linkers_out = []
    metals = idea.get("metals", [])

    raw_linkers = idea.get("linker_smiles", []) or []
    if not isinstance(raw_linkers, list):
        raw_linkers = [raw_linkers]

    for smi in raw_linkers:
        smi = (smi or "").strip()
        if not smi:
            continue

        v = {
            "is_valid": False,
            "formula": None,
            "mol_weight": 0.0,
            "n_atoms": 0,
            "errors": [],
            "warnings": [],
        }

        if not _HAS_RDKIT:
            v["errors"].append("RDKit not available in this environment.")
        else:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    v["errors"].append("Could not parse SMILES.")
                else:
                    v["is_valid"] = True
                    v["n_atoms"] = mol.GetNumAtoms()

                    # approximate formula & weight
                    try:
                        from rdkit.Chem import Descriptors, rdMolDescriptors
                        v["mol_weight"] = Descriptors.MolWt(mol)
                        v["formula"] = rdMolDescriptors.CalcMolFormula(mol)
                    except Exception as ex:
                        v["warnings"].append(f"Could not compute formula/weight: {ex}")
            except Exception as ex:
                v["errors"].append(str(ex))

        linkers_out.append(
            {
                "smiles": smi,
                "validation": v,
            }
        )

    return {
        "linkers": linkers_out,
        "metals": metals,
    }


# ------------------------
# 2) 2D IMAGE FROM SMILES
# ------------------------
def make_linker_image(smiles: str, size: int = 200):
    """
    Produce a PIL.Image for a given SMILES string, or None on failure.
    Streamlit can display this directly via st.image(img).
    """
    if not _HAS_RDKIT:
        return None

    smiles = (smiles or "").strip()
    if not smiles:
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Generate 2D coordinates for nicer layout
        AllChem.Compute2DCoords(mol)
        img = Draw.MolToImage(mol, size=(size, size))
        return img
    except Exception:
        return None


# ------------------------
# 3) 3D HTML FROM SMILES
# ------------------------
def make_linker_3d_html(smiles: str, style: str = "stick") -> Optional[str]:
    """
    Build a simple 3D visualization of a SMILES molecule using RDKit + py3Dmol.

    Returns:
        HTML string suitable for st.components.v1.html(...), or None if unavailable.
    """
    if not (_HAS_RDKIT and _HAS_PY3DMOL):
        return None

    smiles = (smiles or "").strip()
    if not smiles:
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Add hydrogens for 3D geometry
        mol_h = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol_h, randomSeed=42) != 0:
            # embedding failure
            return None
        AllChem.MMFFOptimizeMolecule(mol_h)

        mb = Chem.MolToMolBlock(mol_h)

        view = py3Dmol.view(width=400, height=300)
        view.addModel(mb, "mol")
        if style == "sphere":
            view.setStyle({"sphere": {"scale": 0.3}})
        else:
            # default: stick
            view.setStyle({"stick": {}})

        view.zoomTo()
        return view._make_html()
    except Exception:
        return None


def enzyme_pdb_to_html(pdb_path: str, style: str = "cartoon"):
    """
    Load a local PDB file and convert it to an HTML 3Dmol.js viewer.
    """
    try:
        with open(pdb_path, "r") as f:
            pdb_block = f.read()
    except FileNotFoundError:
        return None

    viewer = py3Dmol.view(width=400, height=400)
    viewer.addModel(pdb_block, "pdb")
    viewer.setStyle({style: {"color": "spectrum"}})
    viewer.zoomTo()
    return viewer._make_html()