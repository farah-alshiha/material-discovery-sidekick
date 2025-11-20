import os
import py3Dmol
from pymatgen.core import Structure

import warnings
warnings.filterwarnings(
    "ignore",
    message="Issues encountered while parsing CIF",
    category=UserWarning,
)


def cif_to_3d_view(cif_path: str, style: str = "stick", surface: bool = False):
    """
    Create a py3Dmol viewer for a MOF CIF file with configurable style/surface.
    """
    if not os.path.exists(cif_path):
        raise FileNotFoundError(f"CIF not found: {cif_path}")

    with open(cif_path, "r") as f:
        cif_str = f.read()

    view = py3Dmol.view(width=600, height=400)
    view.addModel(cif_str, "cif")

    view.setStyle({style: {}})
    view.zoomTo()

    if surface:
        view.addSurface(py3Dmol.VDW, {"opacity": 0.4})

    return view


def detect_metal_elements(cif_path: str):
    """
    Very simple heuristic: find elements in the CIF that are likely metals.
    """
    if not os.path.exists(cif_path):
        return []

    try:
        struct = Structure.from_file(cif_path)
    except Exception:
        return []

    # Basic partition: treat typical organic elements as non-metals
    non_metal_like = {
        "H", "C", "N", "O", "F",
        "P", "S", "Cl", "Br", "I", "B", "Si"
    }

    elements = {str(sp) for sp in struct.composition.elements}
    metal_like = [e for e in elements if e not in non_metal_like]

    return sorted(metal_like)
