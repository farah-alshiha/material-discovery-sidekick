import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from gnn_data import MofGraphDataset
from gnn_models import MofGCN


def main():
    root = "./gnn_data"
    cif_dir = "./cifs"
    batch_size = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # === Load dataset (no re-processing) ===
    dataset = MofGraphDataset(root=root, cif_dir=cif_dir)
    num_classes = len(dataset.classes)
    print(f"Total graphs: {len(dataset)}")
    print("Classes:", dataset.classes)

    # Inspect one sample for in_channels
    sample = dataset[0]
    in_channels = sample.x.size(1)
    print("Node feature dim:", in_channels)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # === Load trained model ===
    hidden_dim = 64  # must match what you used in main_gnn.py
    model = MofGCN(
        in_channels=in_channels,
        hidden_channels=hidden_dim,
        num_classes=num_classes,
        dropout=0.2,
    ).to(device)

    model_path = "trained_models/mof_gcn.pt"
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # === Extract embeddings ===
    all_embeddings = []
    all_labels = []
    all_ids = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            # get graph-level embeddings
            emb = model.encode(batch.x, batch.edge_index, batch.batch, batch.edge_attr)  # (B, hidden_dim)

            # y is (B, 1)
            y = batch.y.view(-1).cpu().numpy()

            # mof_id was stored as attribute in MofGraphDataset.process()
            # In PyG, non-tensor attributes become a list of length B
            mof_ids = batch.mof_id  # list of python strings

            all_embeddings.append(emb.cpu().numpy())
            all_labels.extend(y.tolist())
            all_ids.extend(list(mof_ids))

    E = np.vstack(all_embeddings)  # (N_graphs, hidden_dim)
    y_arr = np.array(all_labels)
    ids_arr = np.array(all_ids)

    print("Embeddings shape:", E.shape)
    print("Labels shape:", y_arr.shape)
    print("First 5 IDs:", ids_arr[:5])

    # === Save to disk for use in tabular ML ===
    os.makedirs("embeddings", exist_ok=True)

    # As NPZ (for Python)
    np.savez("embeddings/mof_gcn_embeddings.npz", embeddings=E, labels=y_arr, ids=ids_arr)

    # As CSV (easier inspection / merging)
    df_emb = pd.DataFrame(E, columns=[f"emb_{i}" for i in range(E.shape[1])])
    df_emb["kh_label_reindexed"] = y_arr
    df_emb["mof_id"] = ids_arr

    df_emb.to_csv("embeddings/mof_gcn_embeddings.csv", index=False)
    print("Saved embeddings to embeddings/mof_gcn_embeddings.csv")


if __name__ == "__main__":
    main()
