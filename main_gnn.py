import os
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F

from gnn_data import MofGraphDataset
from gnn_models import MofGCN


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
        loss = F.cross_entropy(out, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * batch.num_graphs
        total += batch.num_graphs
    return total_loss / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_correct = 0
    total = 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
        pred = out.argmax(dim=1)
        total_correct += int((pred == batch.y.view(-1)).sum())
        total += batch.num_graphs
    return total_correct / total if total > 0 else 0.0


def main():
    # === CONFIG ===
    root = "./gnn_data"
    cif_dir = "./cifs"
    batch_size = 32
    hidden_dim = 64
    lr = 1e-3
    num_epochs = 30

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # === DATASET ===
    dataset = MofGraphDataset(root=root, cif_dir=cif_dir)
    num_classes = len(dataset.classes)

    print(f"Total graphs: {len(dataset)}")
    print("Classes:", dataset.classes)

    # Inspect one sample to get in_channels
    sample = dataset[0]
    in_channels = sample.x.size(1)
    print("Node feature dimension:", in_channels)

    # Train/val/test split (80/10/10)
    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # === MODEL ===
    model = MofGCN(
        in_channels=in_channels,
        hidden_channels=hidden_dim,
        num_classes=num_classes,
        dropout=0.2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # === TRAIN LOOP ===
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    # === TEST EVAL ===
    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc = evaluate(model, test_loader, device)
    print(f"\nBest val_acc={best_val_acc:.4f}, test_acc={test_acc:.4f}")

    # Save model
    os.makedirs("trained_models", exist_ok=True)
    torch.save(model.state_dict(), "trained_models/mof_gcn.pt")
    print("Saved model to trained_models/mof_gcn.pt")


if __name__ == "__main__":
    main()