import cv2
import os
import torch
import numpy as np
from inference import load_model_for_inference, extract_embedding
from scipy.spatial.distance import cosine

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
MODEL_PATH = "output/finetuned_model.pth"
GALLERY_DIR = "output"          # your .npy gallery embeddings
NUM_CLASSES = 4                 # adjust based on your training
THRESHOLD = 0.45                # lower = stricter matching
# ---------------------------------------------------------------


# Load ReID model
model, device, _, _ = load_model_for_inference(
    MODEL_PATH,
    model_ctor_kwargs={"num_classes": NUM_CLASSES}
)

# Load gallery embeddings
gallery = {}
for fname in os.listdir(GALLERY_DIR):
    if fname.endswith(".npy"):
        person_id = fname.replace(".npy", "")
        emb = np.load(os.path.join(GALLERY_DIR, fname))
        gallery[person_id] = emb

if len(gallery) == 0:
    print("⚠ No gallery embeddings found in output/*.npy")
    exit()

print("Loaded Gallery IDs:", list(gallery.keys()))

# Initialize HOG person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture(0)

print("\n[INFO] Starting Re-ID webcam... press Q to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect people (full-body)
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

    for (x, y, w, h) in boxes:
        # Crop person
        person = frame[y:y+h, x:x+w]
        if person.size == 0:
            continue

        # Save temporary crop
        tmp_path = "temp_person.jpg"
        cv2.imwrite(tmp_path, person)

        # Extract embedding
        emb = extract_embedding(model, device, tmp_path)

        # Match with gallery
        best_id = "Unknown"
        best_score = 10  # large distance = bad

        for pid, g_emb in gallery.items():
            dist = cosine(emb, g_emb)
            if dist < best_score:
                best_score = dist
                best_id = pid

        label = f"{best_id} ({best_score:.2f})"
        color = (0, 255, 0) if best_score < THRESHOLD else (0, 0, 255)
        if best_score >= THRESHOLD:
            label = "Unknown"

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame, label, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

    cv2.imshow("ReID Live", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
