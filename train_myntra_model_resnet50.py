"""
train_myntra_model_resnet50.py
==============================
Extracts ResNet50 image features for every product in the Myntra dataset.
Improvements over original:
  - L2-normalizes all features before saving
    (so cosine sim == dot product → faster + compatible with fusion)
  - Saves normalized features under both 'features' and 'features_raw' keys
"""

import numpy as np
import pandas as pd
from PIL import Image
import joblib
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.neighbors import NearestNeighbors
import zipfile
import gdown
from tqdm import tqdm

# ─────────────────── GOOGLE DRIVE CONFIG ───────────────────
GOOGLE_DRIVE_FILE_ID = "18BZUrFg6aY5sujhWlhVsUUtDr0Q24SEP"
GOOGLE_DRIVE_ZIP_URL = (
    f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}"
)


def setup_dataset():
    """Download & extract Myntra dataset once."""
    zip_path       = "myntradataset.zip"
    dataset_folder = "myntradataset"

    if os.path.exists(dataset_folder) and os.path.exists(
        f"{dataset_folder}/images"
    ):
        return dataset_folder

    if not os.path.exists(zip_path):
        print("Downloading dataset from Google Drive...")
        gdown.download(GOOGLE_DRIVE_ZIP_URL, zip_path, quiet=False)

    if not os.path.exists(dataset_folder):
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall()

    return dataset_folder


# ─────────────────── FEATURE EXTRACTOR ───────────────────
class FeatureExtractor:
    """
    ResNet50 with final FC layer removed → 2048-D feature vector per image.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base        = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model  = nn.Sequential(*list(base.children())[:-1])  # drop classifier
        self.model  = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225],
            ),
        ])

    def extract(self, image: Image.Image) -> np.ndarray:
        if image.mode != "RGB":
            image = image.convert("RGB")
        t = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.model(t)
        return feats.squeeze().cpu().numpy()  # (2048,)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    return matrix / norms


# ─────────────────── MAIN ───────────────────
def main():
    os.makedirs("models", exist_ok=True)

    dataset_folder = setup_dataset()
    images_folder  = os.path.join(dataset_folder, "images")

    print("Loading metadata...")
    df = pd.read_csv("myntradataset/styles.csv", on_bad_lines="skip")

    extractor     = FeatureExtractor()
    features      = []
    metadata_rows = []

    print("Extracting image features using ResNet50...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(images_folder, f"{row['id']}.jpg")
        if not os.path.exists(img_path):
            continue
        try:
            img  = Image.open(img_path)
            feat = extractor.extract(img)
            features.append(feat)
            row_copy = row.copy()
            row_copy["image_path"] = img_path
            metadata_rows.append(row_copy)
        except Exception:
            continue

    features_raw  = np.array(features)                   # (N, 2048) raw
    features_norm = l2_normalize(features_raw)            # (N, 2048) unit-norm

    metadata = pd.DataFrame(metadata_rows)

    joblib.dump(
        {
            "features":     features_norm,   # normalized (default for KNN)
            "features_raw": features_raw,    # kept for reference
            "metadata":     metadata,
        },
        "models/fashion_recommender.pkl",
    )
    print("✅ Image features saved → models/fashion_recommender.pkl")
    print(f"   Total items: {len(features_norm)}")


if __name__ == "__main__":
    main()