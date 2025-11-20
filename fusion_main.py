import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from xgboost import XGBClassifier

from mock_api import fetch_mofs_mock
from features import build_mof_kh_dataset
from gnn_data import MofGraphDataset


def main():
    # ============================
    # 1) Load GNN embeddings
    # ============================
    emb_path = "embeddings/mof_gcn_embeddings.csv"
    df_emb = pd.read_csv(emb_path)

    # Columns: emb_0, emb_1, ..., kh_label_reindexed, mof_id
    print("Embeddings shape:", df_emb.shape)
    print("Embedding columns:", [c for c in df_emb.columns if c.startswith("emb_")][:5], "...")

    # ============================
    # 2) Rebuild tabular MOF features
    # ============================
    mof_materials = fetch_mofs_mock()
    X_mof, y_mof, df_mof, le_kh = build_mof_kh_dataset(mof_materials)

    # X_mof: numeric features
    # df_mof: has 'id', 'name', 'kh_class', 'kh_label', etc.
    print("Tabular MOF shape:", X_mof.shape)

    # Attach 'id' to X_mof so we can merge on id ↔ mof_id
    X_mof_with_id = X_mof.copy()
    X_mof_with_id["id"] = df_mof["id"].values

    # ============================
    # 3) Merge embeddings + tabular on MOF id
    # ============================
    df_merged = df_emb.merge(
        X_mof_with_id,
        left_on="mof_id",
        right_on="id",
        how="inner",
    )

    print("Merged shape (embeddings + tabular):", df_merged.shape)

    # Sanity: labels come from GNN dataset (reindexed)
    y = df_merged["kh_label_reindexed"].astype(int).values

    # Identify feature groups
    emb_cols = [c for c in df_merged.columns if c.startswith("emb_")]
    tab_cols = [c for c in X_mof.columns]  # original tabular feature names

    X_tab = df_merged[tab_cols].values
    X_emb = df_merged[emb_cols].values
    X_fused = np.hstack([X_tab, X_emb])

    print("X_tab shape:", X_tab.shape)
    print("X_emb shape:", X_emb.shape)
    print("X_fused shape:", X_fused.shape)

    # ============================
    # 4) Get class names from GNN dataset
    # ============================
    # We use the same root/cif_dir as main_gnn.py
    dataset = MofGraphDataset(root="./gnn_data", cif_dir="./cifs")
    class_names = dataset.classes  # list of KH class strings in reindexed order
    num_classes = len(class_names)
    labels_used = np.arange(num_classes)

    print("Classes (reindexed order):", class_names)

    # ============================
    # 5) Helper to train & report
    # ============================

    def run_xgb_experiment(X, y, label):
        Xtr, Xte, ytr, yte = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        xgb = XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
        )

        xgb.fit(Xtr, ytr)
        y_pred = xgb.predict(Xte)

        print(f"\n=== XGBoost results [{label}] ===")
        print(
            classification_report(
                yte,
                y_pred,
                labels=labels_used,
                target_names=class_names,
            )
        )

    # ============================
    # 6) Run three experiments
    # ============================
    run_xgb_experiment(X_tab, y, label="Tabular only")
    run_xgb_experiment(X_emb, y, label="Embeddings only")
    run_xgb_experiment(X_fused, y, label="Fused (Tabular + GNN embeddings)")


if __name__ == "__main__":
    main()
