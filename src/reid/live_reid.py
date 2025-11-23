import cv2
import os
import numpy as np
from scipy.spatial.distance import cosine
from inference import load_model_for_inference, extract_embedding

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
MODEL_PATH = "src/reid/output/finetuned_model.pth"
GALLERY_DIR = "src/reid/output"      
NUM_CLASSES = 4
THRESHOLD = 0.55
PERSON_SIZE = (128, 256) 
# ---------------------------------------------------------------

print("\n[INFO] Loading model...\n")
model, device, _, _ = load_model_for_inference(
    MODEL_PATH,
    model_ctor_kwargs={"num_classes": NUM_CLASSES}
)

# ---------------------------------------------------------------
# LOAD GALLERY EMBEDDINGS
# ---------------------------------------------------------------
gallery = {}
name_map = {}   # <-- store clean display names

for fname in os.listdir(GALLERY_DIR):
    if fname.endswith(".npy") and "emb" not in fname:
        pid = fname.replace(".npy", "")
        
        # convert person_01 → Person 01
        pretty = pid.replace("person_", "Person ").replace("_", " ")

        name_map[pid] = pretty
        
        emb = np.load(os.path.join(GALLERY_DIR, fname))
        gallery[pid] = emb

print("Gallery IDs Loaded:", name_map, "\n")

if len(gallery) == 0:
    raise Exception("❌ Gallery empty!")

# ---------------------------------------------------------------
# HOG PERSON DETECTOR
# ---------------------------------------------------------------
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture(0)
print("[INFO] Starting ReID Live Stream... Press Q to exit.\n")

# ---------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

    for (x, y, w, h) in boxes:
        crop = frame[y:y+h, x:x+w]
        if crop.size == 0:
            continue

        crop_resized = cv2.resize(crop, PERSON_SIZE)
        temp_path = "temp_crop.jpg"
        cv2.imwrite(temp_path, crop_resized)

        emb = extract_embedding(model, device, temp_path)

        best_id = "Unknown"
        best_score = 999

        for pid, g_emb in gallery.items():
            dist = cosine(emb, g_emb)

            if dist < best_score:
                best_score = dist
                best_id = pid

        # Decide label
        if best_score > THRESHOLD:
            label = f"Unknown ({best_score:.2f})"
            display_name = "Unknown"
            color = (0, 0, 255)
        else:
            display_name = name_map[best_id]   # <-- CLEAN NAME
            label = f"{display_name} ({best_score:.2f})"
            color = (0, 255, 0)

        # Draw on frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("ReID Live", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
