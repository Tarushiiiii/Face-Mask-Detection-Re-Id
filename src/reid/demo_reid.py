import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.linalg import norm
from inference import extract_embedding, load_model


# -------------------------
# Cosine similarity
# -------------------------
def similarity(e1, e2):
    e1 = e1 / norm(e1)
    e2 = e2 / norm(e2)
    return float(np.dot(e1, e2))


# -------------------------
# Load gallery images
# -------------------------
def load_gallery(gallery_dir):
    if not os.path.exists(gallery_dir):
        raise FileNotFoundError(f"Gallery folder not found: {gallery_dir}")

    gallery = []
    for file in os.listdir(gallery_dir):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(gallery_dir, file)
            img = cv2.imread(path)
            if img is not None:
                gallery.append((file, path, img))
    return gallery


# -------------------------
# Compute embeddings
# -------------------------
def compute_gallery_embeddings(model, device, gallery):
    gallery_embs = []
    for name, path, _ in gallery:
        emb = extract_embedding(model, device, path)
        gallery_embs.append((name, path, emb))
    return gallery_embs


# -------------------------
# Find best match
# -------------------------
def find_best_match(query_emb, gallery_embs):
    best_name = None
    best_path = None
    best_score = -1

    for name, path, emb in gallery_embs:
        score = similarity(query_emb, emb)
        if score > best_score:
            best_name = name
            best_path = path
            best_score = score

    return best_name, best_path, best_score


# -------------------------
# Visual Display
# -------------------------
def show_results(query_img, match_img, score):
    label = "SAME PERSON" if score > 0.8 else "DIFFERENT PERSON"
    color = "green" if score > 0.8 else "red"

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
    plt.title("Query Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Best Match\nScore: {score:.2f}\n{label}", color=color)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# -------------------------
# Full Demo Pipeline
# -------------------------
def run_demo(model_path, query_image):
    print("\n🔍 Running ReID Demo...")

    # Absolute paths for safety
    ROOT = os.path.dirname(os.path.abspath(__file__))
    gallery_dir = os.path.join(ROOT, "gallery")

    # Load model
    model, device = load_model(model_path, num_classes=4)

    # Query embedding
    query_emb = extract_embedding(model, device, query_image)
    query_img = cv2.imread(query_image)

    # Load gallery
    gallery = load_gallery(gallery_dir)
    gallery_embs = compute_gallery_embeddings(model, device, gallery)

    # Best match
    best_name, best_path, best_score = find_best_match(query_emb, gallery_embs)
    best_img = cv2.imread(best_path)

    print(f"\nBest Match: {best_name}")
    print(f"Similarity Score: {best_score:.3f}")
    print("Prediction:", "SAME PERSON" if best_score > 0.8 else "DIFFERENT PERSON")

    # Display results
    show_results(query_img, best_img, best_score)


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    run_demo(args.model_path, args.image)
