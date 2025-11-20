import os
from typing import List, Dict, Optional

import torch
from torch_geometric.data import Data, InMemoryDataset
import numpy as np
from pymatgen.core import Structure

from mock_api import fetch_mofs_mock
from schema import Material


def material_to_cif_path(material: Material, cif_dir: str) -> Optional[str]:
    """
    Map a Material to a CIF filepath.

    For your data, CIF filenames follow:
        <coreid>.cif  ==  <material.id>.cif

    Example:
        material.id = "0000[Ag][nan]3[ASR]1"
        -> cif path = "<cif_dir>/0000[Ag][nan]3[ASR]1.cif"
    """
    # We deliberately use material.id, which you set from 'coreid'
    mof_id = material.id
    if mof_id is None:
        return None

    # Main pattern: <id>.cif
    candidate = os.path.join(cif_dir, f"{mof_id}.cif")
    if os.path.exists(candidate):
        return candidate

    # Fallback: try refcode if present (in case some CIFs use that)
    refcode = material.chemistry.refcode
    if refcode:
        candidate2 = os.path.join(cif_dir, f"{refcode}.cif")
        if os.path.exists(candidate2):
            return candidate2

    # Nothing found
    return None



def structure_to_graph(structure: Structure,
                       cutoff: float = 5.0,
                       max_neighbors: int = 12) -> Data:
    """
    Convert a pymatgen Structure into a PyTorch Geometric Data graph.

    Node features:
      - atomic number (scaled) + fractional coordinates (3)
    Edge features:
      - distance
    """
    # Node features
    atomic_numbers = np.array([site.specie.Z for site in structure.sites], dtype=float)
    frac_coords = np.array([site.frac_coords for site in structure.sites], dtype=float)
    # simple scaling for Z so it's not gigantic compared to coords
    atomic_numbers_scaled = (atomic_numbers / 100.0).reshape(-1, 1)

    x = np.concatenate([atomic_numbers_scaled, frac_coords], axis=1)  # shape (N, 4)
    x = torch.tensor(x, dtype=torch.float)

    # Edge construction using radius cutoff
    cart_coords = np.array(structure.cart_coords)
    N = len(cart_coords)

    edge_index_list = []
    edge_attr_list = []

    for i in range(N):
        ri = cart_coords[i]
        dists = np.linalg.norm(cart_coords - ri, axis=1)

        neighbor_idx = np.where((dists > 1e-6) & (dists <= cutoff))[0]
        if len(neighbor_idx) > max_neighbors:
            neighbor_idx = neighbor_idx[np.argsort(dists[neighbor_idx])[:max_neighbors]]

        for j in neighbor_idx:
            edge_index_list.append([i, j])
            edge_attr_list.append([dists[j]])

    if len(edge_index_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


class MofGraphDataset(InMemoryDataset):
    def __init__(self, root: str, cif_dir: str,
                 transform=None, pre_transform=None):
        self.cif_dir = cif_dir
        self._root = root
        super().__init__(root, transform, pre_transform)

        data, slices, class_map, classes = torch.load(
        self.processed_paths[0],
        weights_only=False,
        )
        self.data, self.slices, self.class_map, self.classes = data, slices, class_map, classes


    @property
    def raw_file_names(self) -> List[str]:
        # We don't use raw folder in this simple setup
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return ["mof_graphs.pt"]

    def download(self):
        # Nothing to download; we rely on local CIFs + CSV via mock_api
        pass

    def process(self):
        # 1. Fetch MOF materials
        mof_materials: List[Material] = fetch_mofs_mock(limit=None)

        # First pass: collect KH class strings and counts
        kh_list = []
        for m in mof_materials:
            kh = m.performance.kh_class
            if kh is None or str(kh).lower() in ("", "unknown"):
                continue
            kh_list.append(kh)

        # Compute class counts
        unique_classes, counts = np.unique(kh_list, return_counts=True)
        # Keep only classes with at least 2 samples
        valid_classes = [c for c, cnt in zip(unique_classes, counts) if cnt >= 2]

        # Build mapping: KH string -> int label 0..K-1
        class_map: Dict[str, int] = {cls: i for i, cls in enumerate(sorted(valid_classes))}
        classes = sorted(valid_classes)

        data_list: List[Data] = []

        for m in mof_materials:
            kh = m.performance.kh_class
            if kh is None or kh not in class_map:
                continue

            cif_path = material_to_cif_path(m, self.cif_dir)
            if cif_path is None:
                # CIF not found; skip for now
                continue

            try:
                structure = Structure.from_file(cif_path)
            except Exception as e:
                print(f"[WARN] Failed to read CIF for {m.id} at {cif_path}: {e}")
                continue

            graph = structure_to_graph(structure)
            # assign label
            y = torch.tensor([class_map[kh]], dtype=torch.long)
            graph.y = y
            graph.mof_id = m.id
            data_list.append(graph)

        if len(data_list) == 0:
            raise RuntimeError("No MOF graphs were created. Check CIF paths and filenames.")

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        # Save class_map and classes along with data
        torch.save((data, slices, class_map, classes), self.processed_paths[0])
