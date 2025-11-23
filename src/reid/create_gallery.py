import cv2
import numpy as np
from inference import load_model_for_inference, extract_embedding
import os
from scipy.spatial.distance import cosine

MODEL_PATH = "./output/finetuned_model.pth"
GALLERY_DIR = "./output"
NUM_CLASSES = 4
PERSON_SIZE = (128, 256)

os.makedirs(GALLERY_DIR, exist_ok=True)

print("[INFO] Loading model...")
model, device, _, _ = load_model_for_inference(
    MODEL_PATH,
    model_ctor_kwargs={"num_classes": NUM_CLASSES}
)

cap = cv2.VideoCapture(0)
print("[INFO] Press SPACE to capture gallery image. Press Q to quit.")

count = 1

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("Gallery Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):
        # Use SAME detector as live
        h, w = frame.shape[:2]
        crop = cv2.resize(frame, PERSON_SIZE)

        temp_path = f"temp_gallery.jpg"
        cv2.imwrite(temp_path, crop)

        emb = extract_embedding(model, device, temp_path)

        save_path = os.path.join(GALLERY_DIR, f"person_{count}.npy")
        np.save(save_path, emb)

        print(f"[SAVED] person_{count}.npy")
        count += 1

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
