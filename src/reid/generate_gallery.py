import os
import numpy as np
from inference import load_model_for_inference, extract_embedding

MODEL_PATH = "src/reid/output/finetuned_model.pth"
GALLERY_IMG_DIR = "src/reid/gallery"
OUTPUT_DIR = "src/reid/output"
NUM_CLASSES = 4

# Load model
model, device, _, _ = load_model_for_inference(
    MODEL_PATH,
    model_ctor_kwargs={"num_classes": NUM_CLASSES}
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in os.listdir(GALLERY_IMG_DIR):
    if fname.lower().endswith(".jpg"):
        img_path = os.path.join(GALLERY_IMG_DIR, fname)

        emb = extract_embedding(model, device, img_path)

        save_name = fname.replace(".jpg", ".npy")
        save_path = os.path.join(OUTPUT_DIR, save_name)

        np.save(save_path, emb)
        print(f"Saved: {save_path}")
