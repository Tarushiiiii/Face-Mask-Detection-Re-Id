import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms


# --------------------------------------------
# Image Preprocessing
# --------------------------------------------

def get_transform(img_size=(256, 128)):
    """Returns the standard transform for ReID inputs."""
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_image(path, transform=None):
    """Loads and transforms a single image."""
    img = Image.open(path).convert("RGB")
    if transform:
        img = transform(img)
    return img


# --------------------------------------------
# Embedding Extraction
# --------------------------------------------

def extract_embeddings(model, dataloader, device):
    """Extracts embeddings for an entire dataloader."""

    model.eval()
    embeddings = []
    labels = []
    cam_ids = []

    with torch.no_grad():
        for imgs, pids, cams in dataloader:
            imgs = imgs.to(device)
            embed, _ = model(imgs)

            embeddings.append(embed.cpu())
            labels.extend(pids)
            cam_ids.extend(cams)

    return torch.cat(embeddings, dim=0), np.array(labels), np.array(cam_ids)


# --------------------------------------------
# Distance Calculation
# --------------------------------------------

def compute_distance_matrix(query, gallery, metric="cosine"):
    """
    Computes distance matrix between query and gallery embeddings.
    metric = 'cosine' or 'euclidean'
    """
    if metric == "cosine":
        # Normalize embeddings
        query_norm = torch.nn.functional.normalize(query, dim=1)
        gallery_norm = torch.nn.functional.normalize(gallery, dim=1)
        dist = 1 - torch.mm(query_norm, gallery_norm.t())

    elif metric == "euclidean":
        m, n = query.size(0), gallery.size(0)
        dist = torch.pow(query, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gallery, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        dist.addmm_(query, gallery.t(), beta=1, alpha=-2)

    else:
        raise ValueError("Unsupported metric. Use 'cosine' or 'euclidean'.")

    return dist.cpu().numpy()


# --------------------------------------------
# Misc Helpers
# --------------------------------------------

def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, device):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model
