import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["mask", "reid"], required=True)
args = parser.parse_args()

if args.task == "mask":
    subprocess.run(["python", "src/face_mask/detect_mask_video.py"])

elif args.task == "reid":
    subprocess.run(["python", "src/reid/inference.py"])
