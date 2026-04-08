"""
train_fusion.py  —  Step 2: Build multimodal feature database
=============================================================
Dataset: myntradataset/styles_enriched.csv  (22 columns)

Architecture:
  ResNet50   → 2048-D  →  img_proj (Linear→BN→ReLU→Linear→L2) → 256-D  ┐
  LSTM       →  512-D  →  txt_proj (Linear→BN→ReLU→Linear→L2) → 256-D  ┘
                                         ↓
                              JOINT EMBEDDING SPACE (256-D)
                              trained with Triplet Loss

Gaps fixed in this file:
  GAP1 — Joint embedding space: both modalities projected to same 256-D space
          via Linear projection layers trained with Triplet Margin Loss
  GAP2 — Real prices from enriched CSV (no more simulation)
  GAP3 — Personalization: user_preference_score from enriched CSV stored
          in metadata → app uses it for re-ranking
  GAP4 — enriched CSV treated as "Myntra Fashion Dataset"

For fair compare_models evaluation we ALSO store:
  img_proj_features  (N, 256)  — image-only projected to 256-D
  txt_proj_features  (N, 256)  — text-only  projected to 256-D
  joint_features     (N, 256)  — multimodal joint embedding
All 3 in the SAME 256-D space → fair comparison in compare_models.py

Triplet Loss training:
  Anchor   = any item
  Positive = item with same articleType label
  Negative = item with DIFFERENT articleType label
  margin   = 0.4 (higher margin → more separation → better P@K)

Saves:
  models/fusion_recommender.pkl
  models/joint_projector.pth
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os, joblib, re, random
from tqdm import tqdm
from collections import defaultdict

try:
    import gdown
except ImportError:
    gdown = None

# --- LSTM (must match train_rnn.py EXACTLY) -------------------
class FashionLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim,
                                 num_layers=2, batch_first=True, dropout=0.3)
        self.dropout   = nn.Dropout(0.3)
        self.fc        = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        emb            = self.embedding(x)
        _, (hidden, _) = self.lstm(emb)
        features       = self.dropout(hidden[-1])   # (B, 512)
        return self.fc(features), features

# --- JOINT PROJECTOR ------------------------------------------
class JointProjector(nn.Module):
    
    #Projects CNN (2048-D) and RNN (512-D) into shared 256-D joint space.
    #Each modality can also be projected independently (for single-modal eval).

    def __init__(self, img_dim=2048, txt_dim=512, joint_dim=256):
        super().__init__()
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, 512),                            
            nn.BatchNorm1d(512),                                #2048 -> 512
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, joint_dim),                          #512 -> 256
        )
        self.txt_proj = nn.Sequential(
            nn.Linear(txt_dim, 512),
            nn.BatchNorm1d(512),                                #512 -> 512
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, joint_dim),                          #512 -> 256
        )

    def forward_img(self, x):
        return F.normalize(self.img_proj(x), dim=-1)

    def forward_txt(self, x):
        return F.normalize(self.txt_proj(x), dim=-1)

    def forward(self, img, txt):
        pi = self.forward_img(img)
        pt = self.forward_txt(txt)
        return F.normalize((pi + pt) / 2, dim=-1)

# --- TRIPLET DATASET ------------------------------------------
class TripletDataset(Dataset):
    """
    Hard-negative triplet sampling: negatives are chosen from
    the SAME masterCategory but DIFFERENT articleType.
    This forces the joint space to learn fine-grained distinctions.
    """
    def __init__(self, img_feats, txt_feats, article_labels, category_labels):
        self.img      = torch.tensor(img_feats, dtype=torch.float32)
        self.txt      = torch.tensor(txt_feats, dtype=torch.float32)
        self.art_lbl  = article_labels
        self.cat_lbl  = category_labels

        # Index by articleType for positive sampling
        self.by_art = defaultdict(list)
        for i, l in enumerate(article_labels):
            self.by_art[l].append(i)

        # Index by masterCategory for hard-negative sampling
        self.by_cat = defaultdict(list)
        for i, (a, c) in enumerate(zip(article_labels, category_labels)):
            self.by_cat[c].append(i)

        self.all_arts = list(self.by_art.keys())

    def __len__(self):
        return len(self.art_lbl)

    def __getitem__(self, idx):
        a_art = self.art_lbl[idx]
        a_cat = self.cat_lbl[idx]

        # Positive: same articleType
        pos_pool = [i for i in self.by_art[a_art] if i != idx]
        pos_idx  = random.choice(pos_pool) if pos_pool else idx

        # Hard negative: same masterCategory, different articleType
        hard_pool = [i for i in self.by_cat[a_cat]
                     if self.art_lbl[i] != a_art]
        if hard_pool:
            neg_idx = random.choice(hard_pool)
        else:
            # Fallback: random different articleType
            neg_art = random.choice([l for l in self.all_arts if l != a_art])
            neg_idx = random.choice(self.by_art[neg_art])

        return (self.img[idx],     self.txt[idx],
                self.img[pos_idx], self.txt[pos_idx],
                self.img[neg_idx], self.txt[neg_idx])

# --- TEXT UTILITIES (must match train_rnn.py) ------------------
def clean_text(text):
    if not isinstance(text, str): return ""
    return re.sub(r'[^a-z0-9\s]', '', text.lower()).strip()

def make_item_text(row):
    """MUST match make_training_text() in train_rnn.py exactly."""
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

def setup_dataset():
    folder = "myntradataset"
    if not os.path.exists(folder):
        if gdown is None:
            print("ERROR: pip install gdown"); return folder
        URL = "https://drive.google.com/uc?export=download&id=18BZUrFg6aY5sujhWlhVsUUtDr0Q24SEP"
        if not os.path.exists("myntradataset.zip"):
            print("Downloading dataset...")
            gdown.download(URL, "myntradataset.zip", quiet=False)
        import zipfile
        with zipfile.ZipFile("myntradataset.zip", 'r') as z:
            z.extractall()
    return folder

# --- MAIN -----------------------------------------------------
def main():
    os.makedirs("models", exist_ok=True)
    dataset_folder = setup_dataset()
    images_folder  = os.path.join(dataset_folder, "images")

    # Check prerequisites
    for f in ["models/text_vocab.pkl", "models/text_rnn_model.pth"]:        #load trained lstm model
        if not os.path.exists(f):
            print(f"ERROR: {f} not found. Run train_rnn.py first!"); return

    vocab_data = joblib.load("models/text_vocab.pkl")
    vocab      = vocab_data['vocab']
    label_map  = vocab_data['label_map']
    hidden_dim = vocab_data.get('hidden_dim', 512)
    max_len    = vocab_data.get('max_len', 24)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | hidden_dim={hidden_dim} | max_len={max_len}")

    # Load LSTM
    rnn = FashionLSTM(len(vocab), 300, hidden_dim, len(label_map))
    rnn.load_state_dict(torch.load("models/text_rnn_model.pth",
                                   map_location=device))
    rnn.to(device).eval()
    print("LSTM loaded")

    # Load ResNet50 
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet.to(device).eval()
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    print("ResNet50 loaded")

    enriched_path = os.path.join(dataset_folder, "styles_enriched.csv")
    plain_path    = os.path.join(dataset_folder, "styles.csv")
    if os.path.exists(enriched_path):
        print(f"Loading enriched dataset: {enriched_path}")
        df = pd.read_csv(enriched_path, on_bad_lines='skip')
    else:
        print(f"WARNING: enriched CSV not found, using {plain_path}")
        df = pd.read_csv(plain_path, on_bad_lines='skip')

    # Fill nulls for all columns
    for col in ['productDisplayName','articleType','baseColour','gender',
                'brand','fabric','fit_type','occasion','pattern',
                'masterCategory','usage','season']:
        if col in df.columns:
            df[col] = df[col].fillna('unknown')
        else:
            df[col] = 'unknown'

    df = df.dropna(subset=['articleType'])
    df['combined_label'] = (df['baseColour'].astype(str) + " " +
                            df['articleType'].astype(str))
    print(f"{len(df)} items loaded")

    # --- STEP 1: Extract raw CNN + RNN features ----------------
    image_feats, text_feats, valid_indices = [], [], []
    print("\nEncoding inventory (CNN + RNN)...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):                         #iterate through dataset
        img_path = os.path.join(images_folder, str(row['id']) + ".jpg")
        if not os.path.exists(img_path): continue
        try:
            # CNN (2048-D)
            img   = Image.open(img_path).convert('RGB')                     
            t_img = img_transform(img).unsqueeze(0).to(device)                  #image to vectors
            with torch.no_grad():
                f_img = resnet(t_img).flatten().cpu().numpy()                   #resnet50 output

            # RNN (512-D)
            txt    = make_item_text(row)                                        #text to vectors
            tokens = [vocab.get(t, vocab.get("<UNK>", 1))                       #tokenization
                      for t in txt.split()]
            tokens = tokens[:max_len] + [0] * max(0, max_len - len(tokens))     #padding
            t_txt  = torch.tensor([tokens], dtype=torch.long).to(device)
            with torch.no_grad():
                _, f_txt = rnn(t_txt)                                           #lstm output
                f_txt    = f_txt.flatten().cpu().numpy()

            image_feats.append(f_img)
            text_feats.append(f_txt)
            valid_indices.append(idx)
        except Exception:
            pass

    print(f"Encoded {len(valid_indices)} items successfully")

    valid_df = df.loc[valid_indices].reset_index(drop=True)
    valid_df['image_path'] = valid_df['id'].apply(
        lambda x: os.path.join(dataset_folder, "images", f"{x}.jpg"))

    img_arr = np.array(image_feats, dtype=np.float32)               # (N, 2048)       N = 44000 items
    txt_arr = np.array(text_feats,  dtype=np.float32)               # (N, 512)

    # Labels for triplet sampling
    art_labels = valid_df['articleType'].tolist()
    cat_labels = valid_df['masterCategory'].tolist()

    # --- STEP 2: Train Joint Projector with Triplet Loss -------
    print("\nTraining Joint Embedding Space...")
    print("  Method: Triplet Margin Loss with hard-negative mining")
    print("  Anchor + Positive = same articleType")
    print("  Hard Negative = same masterCategory, different articleType")

    JOINT_DIM    = 256
    JOINT_EPOCHS = 8
    MARGIN       = 0.4   # higher margin → more separation

    projector = JointProjector(2048, hidden_dim, JOINT_DIM).to(device)
    criterion = nn.TripletMarginWithDistanceLoss(
        distance_function=lambda a, b: 1 - F.cosine_similarity(a, b),
        margin=MARGIN,
        reduction='mean'
    )
    optimizer = optim.Adam(projector.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=JOINT_EPOCHS, eta_min=1e-5)

    triplet_ds = TripletDataset(img_arr, txt_arr, art_labels, cat_labels)
    triplet_dl = DataLoader(triplet_ds, batch_size=256, shuffle=True,
                            num_workers=0)

    for epoch in range(JOINT_EPOCHS):
        projector.train()
        total_loss = 0.0
        for batch in tqdm(triplet_dl,
                          desc=f"  Epoch {epoch+1}/{JOINT_EPOCHS}",
                          leave=False):
            (a_img, a_txt,
             p_img, p_txt,
             n_img, n_txt) = [b.to(device) for b in batch]

            anchor   = projector(a_img, a_txt)
            positive = projector(p_img, p_txt)
            negative = projector(n_img, n_txt)

            loss = criterion(anchor, positive, negative)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f"  Epoch {epoch+1}: TripletLoss = {total_loss/len(triplet_dl):.4f}")

    torch.save(projector.state_dict(), "models/joint_projector.pth")
    print("Joint projector saved → models/joint_projector.pth")

    # --- STEP 3: Compute all 3 embedding sets -----------------
    # We store 3 versions - all in same 256-D space for fair comparison
    print("\nComputing embeddings (all 3 modes in 256-D space)...")
    projector.eval()
    img_t = torch.tensor(img_arr).to(device)
    txt_t = torch.tensor(txt_arr).to(device)

    img_proj_list, txt_proj_list, joint_list = [], [], []
    CHUNK = 256
    with torch.no_grad():
        for i in range(0, len(img_t), CHUNK):
            ib = img_t[i:i+CHUNK]
            tb = txt_t[i:i+CHUNK]
            img_proj_list.append(projector.forward_img(ib).cpu().numpy())
            txt_proj_list.append(projector.forward_txt(tb).cpu().numpy())
            joint_list.append(projector(ib, tb).cpu().numpy())

    img_proj_arr  = np.vstack(img_proj_list)   # (N, 256) image projected
    txt_proj_arr  = np.vstack(txt_proj_list)   # (N, 256) text projected
    joint_arr     = np.vstack(joint_list)       # (N, 256) joint embedding

    print(f"  img_proj  : {img_proj_arr.shape}")
    print(f"  txt_proj  : {txt_proj_arr.shape}")
    print(f"  joint     : {joint_arr.shape}")

    # --- STEP 4: Save -----------------------------------------
    fusion_db = {
        # Raw features (for reference)
        'image_features':    img_arr,          # (N, 2048) raw CNN
        'text_features':     txt_arr,          # (N, 512)  raw RNN

        # Projected to 256-D joint space (for evaluation and app)

        'img_proj_features': img_proj_arr,     # (N, 256)  image-only projected
        'txt_proj_features': txt_proj_arr,     # (N, 256)  text-only projected
        'joint_features':    joint_arr,        # (N, 256)  multimodal joint
        'metadata':          valid_df,         # Metadata
        'vocab':             vocab,
        'label_map':         label_map,
        'hidden_dim':        hidden_dim,
        'max_len':           max_len,
        'joint_dim':         JOINT_DIM,
    }
    joblib.dump(fusion_db, "models/fusion_recommender.pkl")

    print(f"\nSaved → models/fusion_recommender.pkl")
    print(f"  Items          : {len(valid_df)}")
    print(f"  Raw CNN        : {img_arr.shape}")
    print(f"  Raw RNN        : {txt_arr.shape}")
    print(f"  Image projected: {img_proj_arr.shape}")
    print(f"  Text projected : {txt_proj_arr.shape}")
    print(f"  Joint          : {joint_arr.shape}")
    print("\nDone! Next: streamlit run app.py")
    print("       Also: python compare_models.py  ← generates comparison chart")

if __name__ == "__main__":
    main()