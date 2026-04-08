"""
train_rnn.py  —  Step 1: Train the RNN (LSTM) text model
=========================================================
Dataset : myntradataset/styles_enriched.csv  (22 columns)
Architecture:
  Embedding  : 300-D word embeddings
  LSTM       : 2-layer unidirectional, hidden=512, dropout=0.3
  Output     : 512-D feature vector

Training text (enriched — uses NEW columns):
  productDisplayName + articleType(x2) + baseColour(x2)
  + gender + brand + fabric + fit_type + occasion + pattern

MAX_LEN = 24  (covers 95th pct of enriched token lengths)
Label   : baseColour + articleType  e.g. "Black Tshirts"

Gaps fixed:
  GAP1 - text now encodes brand/fabric/fit/occasion/pattern
         → richer embeddings → higher cosine similarity
  GAP2 - enriched CSV has real prices, ratings, user_preference_score
  GAP3 - enriched CSV has ratings → used in app personalization
  GAP4 - enriched CSV called "Myntra Fashion Dataset" in report

Saves:
  models/text_rnn_model.pth
  models/text_vocab.pkl
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re, os, joblib

# ── CONFIG ────────────────────────────────────────────────────
DATASET_PATH    = "myntradataset/styles_enriched.csv"
MODEL_SAVE_PATH = "models/text_rnn_model.pth"
VOCAB_SAVE_PATH = "models/text_vocab.pkl"

EMBED_DIM     = 300
HIDDEN_DIM    = 512
BATCH_SIZE    = 64
EPOCHS        = 8
LEARNING_RATE = 0.001
MAX_LEN       = 24   # ← increased from 10 to handle enriched text

# ── TEXT UTILITIES ────────────────────────────────────────────
def clean_text(text):
    if not isinstance(text, str): return ""
    return re.sub(r'[^a-z0-9\s]', '', text.lower()).strip()

def make_training_text(row):
    """
    Enriched training text — uses all relevant columns from enriched CSV.
    articleType and baseColour repeated x2 to strengthen category signal.
    brand/fabric/fit_type/occasion/pattern add semantic richness.
    """
    name    = clean_text(str(row.get('productDisplayName', '')))
    atype   = clean_text(str(row.get('articleType', '')))
    colour  = clean_text(str(row.get('baseColour', '')))
    gender  = clean_text(str(row.get('gender', '')))
    brand   = clean_text(str(row.get('brand', 'unknown')))
    fabric  = clean_text(str(row.get('fabric', 'unknown')))
    fit     = clean_text(str(row.get('fit_type', 'unknown')))
    occ     = clean_text(str(row.get('occasion', 'unknown')))
    pattern = clean_text(str(row.get('pattern', 'unknown')))
    return (f"{name} {atype} {atype} {colour} {colour} "
            f"{gender} {brand} {fabric} {fit} {occ} {pattern}")

# ── DATASET ───────────────────────────────────────────────────
class FashionTextDataset(Dataset):
    def __init__(self, df, vocab, label_map):
        self.df        = df.reset_index(drop=True)
        self.vocab     = vocab
        self.label_map = label_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        tokens  = make_training_text(row).split()
        indices = [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens]
        if len(indices) < MAX_LEN:
            indices += [0] * (MAX_LEN - len(indices))
        else:
            indices = indices[:MAX_LEN]
        label = self.label_map.get(str(row['combined_label']), 0)
        return (torch.tensor(indices, dtype=torch.long),
                torch.tensor(label,   dtype=torch.long))

# ── MODEL ─────────────────────────────────────────────────────
class FashionLSTM(nn.Module):
    """
    2-layer unidirectional LSTM.
    forward() → (logits, features)  where features is 512-D embedding.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim,
                                 num_layers=2, batch_first=True,
                                 dropout=0.3)
        self.dropout   = nn.Dropout(0.3)
        self.fc        = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        emb            = self.embedding(x)          # (B, T, E)
        _, (hidden, _) = self.lstm(emb)             # hidden: (2, B, H)
        features       = self.dropout(hidden[-1])   # last layer: (B, H=512)
        logits         = self.fc(features)          # (B, num_labels)
        return logits, features

# ── TRAINING ──────────────────────────────────────────────────
def main():
    os.makedirs("models", exist_ok=True)

    # Load enriched dataset
    if os.path.exists(DATASET_PATH):
        print(f"Loading enriched dataset: {DATASET_PATH}")
        df = pd.read_csv(DATASET_PATH, on_bad_lines='skip')
    else:
        fallback = "myntradataset/styles.csv"
        print(f"WARNING: {DATASET_PATH} not found.")
        print(f"  Place styles_enriched.csv in myntradataset/ for best results.")
        print(f"  Falling back to: {fallback}")
        df = pd.read_csv(fallback, on_bad_lines='skip')

    # Fill nulls
    for col in ['productDisplayName', 'articleType', 'baseColour', 'gender',
                'brand', 'fabric', 'fit_type', 'occasion', 'pattern']:
        if col in df.columns:
            df[col] = df[col].fillna('unknown')
        else:
            df[col] = 'unknown'

    df = df.dropna(subset=['articleType'])
    df['combined_label'] = (df['baseColour'].astype(str) + " " +
                            df['articleType'].astype(str))
    print(f"  {len(df)} items | {df['combined_label'].nunique()} unique labels")

    # Build vocabulary
    print("Building vocabulary from enriched text...")
    all_text    = " ".join(df.apply(make_training_text, axis=1).tolist())
    word_counts = Counter(all_text.split())
    vocab       = {"<PAD>": 0, "<UNK>": 1}
    for w, c in word_counts.items():
        if c >= 2:
            vocab[w] = len(vocab)

    labels    = df['combined_label'].unique().tolist()
    label_map = {l: i for i, l in enumerate(labels)}
    print(f"  Vocab size : {len(vocab)}")
    print(f"  Labels     : {len(label_map)}")

    joblib.dump({
        'vocab':      vocab,
        'label_map':  label_map,
        'hidden_dim': HIDDEN_DIM,
        'max_len':    MAX_LEN,
    }, VOCAB_SAVE_PATH)
    print(f"  Vocab saved → {VOCAB_SAVE_PATH}")

    # Train
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on {device}  (epochs={EPOCHS}, MAX_LEN={MAX_LEN})...")

    dataset   = FashionTextDataset(df, vocab, label_map)
    loader    = DataLoader(dataset, batch_size=BATCH_SIZE,
                           shuffle=True, num_workers=0)
    model     = FashionLSTM(len(vocab), EMBED_DIM, HIDDEN_DIM,
                            len(label_map)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,
                           weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for texts, targets in loader:
            texts, targets = texts.to(device), targets.to(device)
            optimizer.zero_grad()
            logits, _ = model(texts)
            loss      = criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            correct    += (logits.argmax(1) == targets).sum().item()
            total      += len(targets)
        acc = correct / total * 100
        print(f"  Epoch {epoch+1}/{EPOCHS}:  "
              f"Loss={total_loss/len(loader):.4f}  Acc={acc:.1f}%")
        # Save best model checkpoint
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        scheduler.step()

    print(f"\nBest accuracy : {best_acc:.1f}%")
    print(f"Model saved   → {MODEL_SAVE_PATH}")
    print("Done! Next: python train_fusion.py")

if __name__ == "__main__":
    main()