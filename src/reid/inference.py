"""
Robust inference loader for re-ID checkpoints.

Usage:
    python inference.py --model-path output/finetuned_model.pth --image examples/example_01.png

This script will:
 - try to load checkpoint intelligently (handle common wrappers and prefixes)
 - attempt strict load, then partial load (strict=False)
 - print missing/unexpected keys to help debugging
 - extract embedding from a single image and print/save it
"""

import argparse
import torch
import os
from PIL import Image
from torchvision import transforms
import numpy as np
from model import ReIDModel  # your local model file
from config import IMG_SIZE
import re

def preprocess(img_path):
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert("RGB")
    return transform(img)

def try_unwrap_checkpoint(ckpt):
    """
    Handle common checkpoint wrappers:
      - {'state_dict': {...}}
      - {'model': {...}}
      - saved state_dict directly
    """
    if isinstance(ckpt, dict):
        # if very likely a state dict (contains tensors), return as-is
        has_tensor_values = any(isinstance(v, torch.Tensor) for v in ckpt.values())
        if has_tensor_values:
            return ckpt
        # check common wrappers:
        for key in ("state_dict", "model", "net", "state"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    # fallback: return ckpt (may raise later)
    return ckpt

def strip_prefix_from_keys(state_dict, prefixes=("module.", "backbone.", "model.")):
    """
    Return new dict with any of the prefixes removed where present.
    This attempts several heuristics to map keys.
    """
    new_state = {}
    for k, v in state_dict.items():
        newk = k
        # remove common prefixes
        for p in prefixes:
            if newk.startswith(p):
                newk = newk[len(p):]
                break
        new_state[newk] = v
    return new_state

def match_and_load(model, state):
    """
    Try to load state into model in several ways:
      1) strict load
      2) strip 'module.' prefix then strict load
      3) partial load (strict=False) with filtered keys that match model keys
    Returns True on success (any successful load), and a dict of details.
    """
    model_state = model.state_dict()
    result = {"loaded": False, "method": None, "missing_keys": None, "unexpected_keys": None}

    # attempt 1: direct strict load
    try:
        model.load_state_dict(state, strict=True)
        result.update({"loaded": True, "method": "strict"})
        return result
    except Exception as e:
        # continue to next attempt
        pass

    # attempt 2: strip common prefixes and try strict
    state2 = strip_prefix_from_keys(state, prefixes=("module.",))
    try:
        model.load_state_dict(state2, strict=True)
        result.update({"loaded": True, "method": "strip_module_strict"})
        return result
    except Exception:
        pass

    # attempt 3: partial load - keep keys that match exactly model keys
    filtered = {k: v for k, v in state.items() if k in model_state and model_state[k].shape == v.shape}
    if filtered:
        missing = set(model_state.keys()) - set(filtered.keys())
        unexpected = set(state.keys()) - set(filtered.keys())
        model.load_state_dict(filtered, strict=False)
        result.update({
            "loaded": True,
            "method": "partial_filtered",
            "missing_keys_count": len(missing),
            "unexpected_keys_count": len(unexpected),
            "missing_keys_sample": list(missing)[:10],
            "unexpected_keys_sample": list(unexpected)[:10],
        })
        return result

    # attempt 4: try more aggressive remapping heuristics:
    # heuristic: some checkpoints use backbone.<idx>... while model uses backbone.layerX...
    # We provide no universal remapping here, but attempt to match suffix keys:
    model_keys = set(model_state.keys())
    mapped = {}
    for s_key, tensor in state.items():
        # try to find any model key that endswith the same suffix
        for m_key in model_keys:
            if m_key.endswith(s_key):
                if model_state[m_key].shape == tensor.shape:
                    mapped[m_key] = tensor
                    break
    if mapped:
        model.load_state_dict(mapped, strict=False)
        missing = set(model_state.keys()) - set(mapped.keys())
        unexpected = set(state.keys()) - set(mapped.keys())
        result.update({
            "loaded": True,
            "method": "heuristic_suffix_map",
            "mapped_count": len(mapped),
            "missing_keys_count": len(missing),
            "unexpected_keys_count": len(unexpected),
            "mapped_sample": list(mapped.keys())[:10]
        })
        return result

    # all attempts failed
    result.update({"loaded": False, "method": "failed_all"})
    return result

def load_checkpoint_generic(path, model):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location="cpu")
    state = try_unwrap_checkpoint(ckpt)
    # if still not a dict, try attribute access
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint format not recognized.")
    # Try progressively
    info = match_and_load(model, state)
    return info, state

def load_model_for_inference(model_path, model_ctor_kwargs=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_ctor_kwargs is None:
        model_ctor_kwargs = {}
    model = ReIDModel(**model_ctor_kwargs)  # instantiate your local model
    # attempt to load checkpoint
    print("Attempting to load checkpoint:", model_path)
    info, state = load_checkpoint_generic(model_path, model)
    print("Load attempt info:", info)
    if not info["loaded"]:
        print("WARNING: checkpoint could not be loaded into the current model cleanly.")
        print("You can still use the model (random init) but embeddings won't reflect checkpoint.")
    model.to(device)
    model.eval()
    return model, device, info, state

def extract_embedding(model, device, img_path):
    x = preprocess(img_path).unsqueeze(0).to(device)
    with torch.no_grad():
        emb, logits = model(x)
    # ensure numpy 1D vector
    emb_np = emb.cpu().numpy()
    if emb_np.ndim == 2 and emb_np.shape[0] == 1:
        emb_np = emb_np[0]
    return emb_np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--num-classes", type=int, default=751)
    parser.add_argument("--save-name", type=str, default="embedding.npy",
                        help="Filename to save the output embedding")
    args = parser.parse_args()

    # decide model constructor kwargs: if you know num_classes pass it
    ctor_kwargs = {}
    if args.num_classes:
        ctor_kwargs["num_classes"] = args.num_classes

    model, device, info, state = load_model_for_inference(args.model_path, model_ctor_kwargs=ctor_kwargs)

    # Show some helpful diagnostics if load was partial
    if "missing_keys_sample" in info:
        print("Sample missing keys (model expects these but checkpoint didn't have them):")
        print(info["missing_keys_sample"])

    if "unexpected_keys_sample" in info:
        print("Sample unexpected keys (checkpoint had these extra keys):")
        print(info["unexpected_keys_sample"])

    # Now extract embedding
    emb = extract_embedding(model, device, args.image)
    print("Embedding shape:", emb.shape)

    # Save embedding
    save_dir = "output"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, args.save_name)
    np.save(save_path, emb)      # emb is already a numpy array

    print(f"Saved embedding to: {save_path}")


if __name__ == "__main__":
    main()