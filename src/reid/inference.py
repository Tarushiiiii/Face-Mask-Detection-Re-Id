def extract_embedding(model, img_path):
    model.eval()
    img = preprocess(img_path)
    with torch.no_grad():
        embed, _ = model(img.unsqueeze(0))
    return embed
