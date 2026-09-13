import os
import time
import pandas as pd
import psycopg2
import torch
import torch.nn as nn
from model import CardDataset, CardPriceModel
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


def connect_with_retry(database_url, retries=5, delay=2):
    for attempt in range(1, retries + 1):
        try:
            return psycopg2.connect(database_url)
        except psycopg2.OperationalError as e:
            if attempt == retries:
                raise
            print(f"WARNING: DB connection attempt {attempt}/{retries} failed: {e}")
            time.sleep(delay * attempt)

def train(csv_path, feature_cols, epochs_stage1=20, epochs_stage2=30, batch_size=32, val_split=0.2,
          learning_rate=1e-3, hidden_size=128, dropout=0.2, conn=None, run_id=None):
    def log_metric(step, metric_name, value):
        if not conn:
            return
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO metrics (run_id, step, metric_name, value) VALUES (%s, %s, %s, %s)",
                    (run_id, step, metric_name, value)
                )
            conn.commit()
        except Exception as e:
            print(f"WARNING: failed to log metric {metric_name}={value} at step {step}: {e}")
            conn.rollback()

    # Load data, dropping rows with missing values
    df = pd.read_csv(csv_path)
    string_cols = df.select_dtypes(include=["object", "str"]).columns
    df[string_cols] = df[string_cols].apply(lambda col: col.str.strip())
    df = df[feature_cols + ["cleanName", "extDescription", "midPrice"]].dropna()

    # Prepend the card title so it survives truncation of long effect text
    df["extDescription"] = df["cleanName"] + ": " + df["extDescription"]

    # Log-transform the target
    df["midPrice"] = df["midPrice"].clip(lower=0.01)
    df["logPrice"] = df["midPrice"].apply(lambda x: x ** 0.5)

    # Remove rarities with a single row for stratified split compatibility
    rarity_counts = df["extRarity"].value_counts()
    df = df[df["extRarity"].isin(rarity_counts[rarity_counts >= 2].index)]

    # Train-test split
    train_df, val_df = train_test_split(
        df, test_size=val_split, stratify=df["extRarity"], random_state=42
    )

    train_set = CardDataset(train_df, feature_cols, "logPrice")
    val_set   = CardDataset(val_df,   feature_cols, "logPrice",
                            scaler=train_set.scaler,
                            encoders=train_set.encoders,
                            tokenizer=train_set.tokenizer)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size)

    model   = CardPriceModel(num_features=len(feature_cols), hidden_size=hidden_size, dropout=dropout)
    loss_fn = nn.HuberLoss()

    # ── Stage 1: Frozen DistilBERT ─────────────────────────────────────────────
    print("Stage 1: Training numeric branch and fusion head...")
    for param in model.text_encoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam([
        {"params": model.num_encoder.parameters(), "lr": learning_rate},
        {"params": model.fusion.parameters(),      "lr": learning_rate},
    ])

    for epoch in range(epochs_stage1):
        model.train()
        train_loss = 0
        for numeric_feats, input_ids, attention_mask, targets in train_loader:
            optimizer.zero_grad()
            preds = model(numeric_feats, input_ids, attention_mask)
            loss  = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for numeric_feats, input_ids, attention_mask, targets in val_loader:
                preds    = model(numeric_feats, input_ids, attention_mask)
                val_loss += loss_fn(preds, targets).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss   = val_loss / len(val_loader)
        print(
            f"[S1] Epoch {epoch+1:>3}/{epochs_stage1} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )
        log_metric(epoch, "stage1_train_loss", avg_train_loss)
        log_metric(epoch, "stage1_val_loss", avg_val_loss)

    # ── Stage 2: Full fine-tune ────────────────────────────────────────────────
    print("\nStage 2: Fine-tuning all layers...")
    for param in model.text_encoder.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam([
        {"params": model.text_encoder.parameters(), "lr": learning_rate * 0.01},
        {"params": model.num_encoder.parameters(), "lr": learning_rate},
        {"params": model.fusion.parameters(),      "lr": learning_rate},
    ])

    for epoch in range(epochs_stage2):
        model.train()
        train_loss = 0
        for numeric_feats, input_ids, attention_mask, targets in train_loader:
            optimizer.zero_grad()
            preds = model(numeric_feats, input_ids, attention_mask)
            loss  = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for numeric_feats, input_ids, attention_mask, targets in val_loader:
                preds    = model(numeric_feats, input_ids, attention_mask)
                val_loss += loss_fn(preds, targets).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss   = val_loss / len(val_loader)
        print(
            f"[S2] Epoch {epoch+1:>3}/{epochs_stage2} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )
        log_metric(epoch, "stage2_train_loss", avg_train_loss)
        log_metric(epoch, "stage2_val_loss", avg_val_loss)

    return model, train_set.scaler, train_set.encoders, train_set.tokenizer


# ── Inference ──────────────────────────────────────────────────────────────────

def predict(model, scaler, encoders, tokenizer, feature_cols, sample: dict, effect_text: str, max_length=192) -> float:
    df = pd.DataFrame([sample])
    for col, enc in encoders.items():
        df[col] = enc.transform(df[col].astype(str))
    numeric_feats = torch.tensor(
        scaler.transform(df[feature_cols].values.astype("float32")), dtype=torch.float32
    )

    tokenized = tokenizer(
        [effect_text], padding="max_length", truncation=True,
        max_length=max_length, return_tensors="pt"
    )

    model.eval()
    with torch.no_grad():
        log_price = model(numeric_feats, tokenized["input_ids"], tokenized["attention_mask"]).item()
    return log_price ** 2  # undo sqrt transform


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    feature_cols = [
        "extNation", "extRarity", "extGrade", "extShield",
        "extCritical", "extPower",
    ]

    epochs_stage1 = int(os.environ.get("EPOCHS_STAGE1", 20))
    epochs_stage2 = int(os.environ.get("EPOCHS_STAGE2", 30))
    learning_rate = float(os.environ.get("LEARNING_RATE", 1e-3))
    batch_size    = int(os.environ.get("BATCH_SIZE", 32))
    hidden_size   = int(os.environ.get("HIDDEN_SIZE", 128))
    dropout       = float(os.environ.get("DROPOUT", 0.2))

    run_id = os.environ.get("RUN_ID")
    database_url = os.environ.get("DATABASE_URL")

    if bool(database_url) != bool(run_id):
        raise RuntimeError("DATABASE_URL and RUN_ID must both be set together, or both left unset")

    conn = connect_with_retry(database_url) if database_url else None
    status = ""

    if conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE runs SET status = %s WHERE id = %s", ("running", run_id))
        conn.commit()

    try:
        model, scaler, encoders, tokenizer = train(
            "card_data/compiled_data.csv", feature_cols,
            epochs_stage1=epochs_stage1, epochs_stage2=epochs_stage2,
            batch_size=batch_size, learning_rate=learning_rate,
            hidden_size=hidden_size, dropout=dropout,
            conn=conn, run_id=run_id,
        )

        out_dir = Path("models") / "full"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save model, scaler, and encoders
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "scaler_mean": torch.tensor(scaler.mean_),
            "scaler_scale": torch.tensor(scaler.scale_),
        }
        torch.save(checkpoint, out_dir / "full_model.pt")
        torch.save(encoders, out_dir / "full_encoders.pt")

        status = "completed"
    except Exception:
        status = "failed"
        raise
    finally:
        if conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE runs SET status = %s WHERE id = %s", (status, run_id))
            conn.commit()
            conn.close()
