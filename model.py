import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import LabelEncoder, StandardScaler

class CardDataset(Dataset):
    def __init__(self, dataframe, feature_columns, target_column, text_column="extDescription",
                 scaler=None, encoders=None, tokenizer=None, max_length=192):
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

        # Text tokenization
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained("distilbert-base-uncased")
        tokenized = self.tokenizer(
            dataframe[text_column].tolist(),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        self.input_ids      = tokenized["input_ids"]
        self.attention_mask = tokenized["attention_mask"]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.input_ids[idx],
            self.attention_mask[idx],
            self.targets[idx]
        )

class CardPriceModel(nn.Module):
    def __init__(self, num_features, hidden_size=128, dropout=0.2):
        super().__init__()

        # Text branch
        self.text_encoder = AutoModel.from_pretrained("distilbert-base-uncased")

        # Numeric branch
        num_emb_size = hidden_size // 2
        self.num_encoder = nn.Sequential(
            nn.Linear(num_features, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, num_emb_size), nn.ReLU(),
        )

        # Fusion head
        self.fusion = nn.Sequential(
            nn.Linear(768 + num_emb_size, hidden_size), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, numeric_feats, input_ids, attention_mask):
        # Text branch
        text_emb = self.text_encoder(
            input_ids, attention_mask
        ).last_hidden_state[:, 0] # [batch, 768]

        # Numeric branch
        num_emb = self.num_encoder(numeric_feats) # [batch, 64]

        # Fuse
        combined = torch.cat([text_emb, num_emb], dim=-1) # [batch, 832]
        return self.fusion(combined).squeeze(1)
