import numpy as np
from numpy.linalg import norm

def similarity(emb1, emb2):
    emb1 = emb1 / norm(emb1)
    emb2 = emb2 / norm(emb2)
    return np.dot(emb1, emb2)

# Load embeddings
e1 = np.load("output/emb_01.npy")
e2 = np.load("output/emb_02.npy")

score = similarity(e1, e2)
print("Similarity Score:", score)

if score > 0.8:
    print("Prediction: SAME PERSON")
else:
    print("Prediction: DIFFERENT PERSON")
