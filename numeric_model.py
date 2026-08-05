import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path

PATH = Path.cwd()
TRAIN = False
feature_cols = [
        "extNation", "extRarity", "extGrade", "extShield",
        "extCritical", "extPower"
    ]

# ── Dataset ────────────────────────────────────────────────────────────────────

class CardDataset(Dataset):
    def __init__(self, dataframe, feature_columns, target_column, scaler=None, encoders=None):
        df = dataframe.copy()

        # Encode string columns to integers
        self.encoders = encoders or {}
        string_cols = df[feature_columns].select_dtypes(include=["object", "str"]).columns
        for col in string_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                df[col] = self.encoders[col].fit_transform(df[col].astype(str))
            else:
                df[col] = self.encoders[col].transform(df[col].astype(str))

        # Scale numeric values
        self.scaler = scaler or StandardScaler()
        features = df[feature_columns].values.astype("float32")
        if scaler is None:
            features = self.scaler.fit_transform(features)
        else:
            features = self.scaler.transform(features)

        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets  = torch.tensor(df[target_column].values, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# ── Model ──────────────────────────────────────────────────────────────────────

class PricePredictor(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64),           nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)



# ── Training loop ──────────────────────────────────────────────────────────────

def train(csv_path, epochs=50, batch_size=32, lr=1e-3, val_split=0.2):
    global feature_cols

    df = pd.read_csv(csv_path)

    # Drop rows where any feature or target is missing
    df = df[feature_cols + ["midPrice"]].dropna()

    # Log-transform the target
    df["midPrice"] = df["midPrice"].clip(lower=0.01)
    df["logPrice"] = df["midPrice"].apply(lambda x: x ** 0.5)

    full_dataset = CardDataset(df, feature_cols, "logPrice")

    # Train/val split
    val_size   = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_df, val_df = train_test_split(df, test_size=val_split, stratify=df["extRarity"], random_state=42)
    train_set = CardDataset(train_df, feature_cols, "logPrice")
    val_set   = CardDataset(val_df,   feature_cols, "logPrice", scaler=train_set.scaler, encoders=train_set.encoders)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size)

    model     = PricePredictor(num_features=len(feature_cols))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.HuberLoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for features, targets in train_loader:
            optimizer.zero_grad()
            preds = model(features)
            loss  = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                preds    = model(features)
                val_loss += loss_fn(preds, targets).item()

        print(
            f"Epoch {epoch+1:>3}/{epochs} | "
            f"Train Loss: {train_loss/len(train_loader):.4f} | "
            f"Val Loss: {val_loss/len(val_loader):.4f}"
        )

    return model, full_dataset.scaler, full_dataset.encoders


# ── Inference ──────────────────────────────────────────────────────────────────

def predict(model, scaler, encoders, sample: dict) -> float:
    df = pd.DataFrame([sample])
    for col, enc in encoders.items():
        df[col] = enc.transform(df[col].astype(str))
    features = torch.tensor(
        scaler.transform(df.values.astype("float32")), dtype=torch.float32
    )
    model.eval()
    with torch.no_grad():
        log_price = model(features).item()
    return log_price ** 2  # undo sqrt transform


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if TRAIN:
        model, scaler, encoders = train("cards.csv", epochs=500, batch_size=64)

        # Save Model and Scaler
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'scaler_mean': torch.tensor(scaler.mean_),
            'scaler_scale': torch.tensor(scaler.scale_),
        }
        torch.save(checkpoint, PATH / "models" / "numeric" / "basic_model.pt")

        # Save Encoders
        torch.save(encoders, PATH / "models" / "numeric" / "basic_encoders.pt")
    else:
        # Load Model and Scaler
        model = PricePredictor(num_features=len(feature_cols))

        checkpoint = torch.load(PATH / "models" / "numeric" / 'basic_model.pt', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        scaler = StandardScaler()
        scaler.mean_ = checkpoint['scaler_mean'].numpy()
        scaler.scale_ = checkpoint['scaler_scale'].numpy()
        scaler.n_features_in_ = len(feature_cols)  # add this

        # Load Encoders
        encoders = torch.load(PATH / "models" / "numeric" / "basic_encoders.pt", weights_only=False)

    sample_card = {
        "extNation":   "Dark States",
        "extRarity":   "R",
        "extGrade":    1,
        "extShield":   5000,
        "extCritical": 1,
        "extPower":    8000,
    }

    price = predict(model, scaler, encoders, sample_card)
    print(f"Predicted mid price: ${price:.2f}")
