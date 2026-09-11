import argparse
from pathlib import Path

import torch
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

from model import CardPriceModel
from train import predict

FEATURE_COLS = ["extNation", "extRarity", "extGrade", "extShield", "extCritical", "extPower"]
MODEL_DIR = Path("models") / "full"

"""
Sample Usage:
python infer.py --nation "Dark States" --rarity R --grade 1 --shield 5000 --critical 1 --power 8000 --text "During your turn, if you have a grade 3 or greater vanguard, this unit gets [Power]+5000."
"""

def load_artifacts():
    checkpoint = torch.load(MODEL_DIR / "full_model.pt", weights_only=False)

    model = CardPriceModel(num_features=len(FEATURE_COLS))
    model.load_state_dict(checkpoint["model_state_dict"])

    scaler = StandardScaler()
    scaler.mean_ = checkpoint["scaler_mean"].numpy()
    scaler.scale_ = checkpoint["scaler_scale"].numpy()
    scaler.n_features_in_ = len(FEATURE_COLS)

    encoders = torch.load(MODEL_DIR / "full_encoders.pt", weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    return model, scaler, encoders, tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Predict a card's mid price.")
    parser.add_argument("--nation", required=True)
    parser.add_argument("--rarity", required=True)
    parser.add_argument("--grade", type=int, required=True)
    parser.add_argument("--shield", type=int, required=True)
    parser.add_argument("--critical", type=int, required=True)
    parser.add_argument("--power", type=int, required=True)
    parser.add_argument("--text", required=True, help="Card effect text")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    sample_card = {
        "extNation":   args.nation,
        "extRarity":   args.rarity,
        "extGrade":    args.grade,
        "extShield":   args.shield,
        "extCritical": args.critical,
        "extPower":    args.power,
    }

    model, scaler, encoders, tokenizer = load_artifacts()
    price = predict(model, scaler, encoders, tokenizer, FEATURE_COLS, sample_card, args.text)
    print(f"Predicted mid price: ${price:.2f}")
