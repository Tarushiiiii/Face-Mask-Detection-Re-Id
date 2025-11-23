import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.linalg import norm
from inference import extract_embedding, load_model


def similarity(e1, e2):
    e1 = e1 / norm(e1)
    e2 = e2 / norm(e2)
    return float(np.dot(e1, e2))


def show_results(query_img, target_img, score):
    label = "SAME PERSON" if score > 0.8 else "DIFFERENT PERSON"
    color = "green" if score > 0.8 else "red"

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
    plt.title("Query Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Target Image\nScore: {score:.2f}\n{label}", color=color)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def run_manual(model_path, query_img_path, target_img_path):
    print("\n🔍 Manual ReID Comparison")

    # Load model
    model, device = load_model(model_path, num_classes=4)

    # Load both images
    query_img = cv2.imread(query_img_path)
    target_img = cv2.imread(target_img_path)

    if query_img is None:
        raise FileNotFoundError(f"Query image not found: {query_img_path}")
    if target_img is None:
        raise FileNotFoundError(f"Target image not found: {target_img_path}")

    # Extract embeddings
    q_emb = extract_embedding(model, device, query_img_path)
    t_emb = extract_embedding(model, device, target_img_path)

    # Compute similarity
    score = similarity(q_emb, t_emb)

    print("\nSimilarity Score:", score)
    print("Prediction:", "SAME PERSON" if score > 0.8 else "DIFFERENT PERSON")

    # Visualize
    show_results(query_img, target_img, score)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()

    run_manual(args.model_path, args.query, args.target)
