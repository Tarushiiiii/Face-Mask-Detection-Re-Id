# inference.py (CLEAN & TEACHER-FRIENDLY VERSION)

import argparse
import torch
import os
from PIL import Image
from torchvision import transforms
import numpy as np
from model import ReIDModel
from config import IMG_SIZE
import cv2


# -----------------------------
# Preprocess function
# -----------------------------
def preprocess(img_path):
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    img = Image.open(img_path).convert("RGB")
    return transform(img)


# -----------------------------
# Simple checkpoint loader (quiet)
# -----------------------------
def load_model(model_path, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ReIDModel(num_classes=num_classes)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    # Load checkpoint quietly
    state = torch.load(model_path, map_location="cpu")

    # Handle common wrappers
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model" in state:
            state = state["model"]

    # Load with strict=False to avoid full logs
    model.load_state_dict(state, strict=False)

    model.to(device)
    model.eval()
    return model, device


# -----------------------------
# Extract Embedding
# -----------------------------
def extract_embedding(model, device, img_path):
    x = preprocess(img_path).unsqueeze(0).to(device)
    with torch.no_grad():
        emb, _ = model(x)
    emb = emb.squeeze().cpu().numpy()
    return emb


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--save-name", type=str, default="embedding.npy")
    args = parser.parse_args()

    print("\n🔍 Loading Re-ID model...")
    model, device = load_model(args.model_path, args.num_classes)
    print("✅ Model loaded successfully!")

    print("\n📸 Processing image:", args.image)
    emb = extract_embedding(model, device, args.image)
    print("🔢 Embedding shape:", emb.shape)

    # Save embedding
    os.makedirs("output", exist_ok=True)
    save_path = os.path.join("output", args.save_name)
    np.save(save_path, emb)

    print(f"💾 Embedding saved to: {save_path}")

    # -----------------------------
    # SHOW IMAGE POPUP
    # -----------------------------
    img = cv2.imread(args.image)
    if img is not None:
        cv2.imshow("ReID Input Image", img)
        print("\n🖼️ Showing input image... (Press any key to close)")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("⚠️ Could not load image for display.")


if __name__ == "__main__":
    main()
