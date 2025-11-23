import argparse
import subprocess
import sys
import os

python_exec = sys.executable
ROOT = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["mask", "reid"], required=True)
args = parser.parse_args()

if args.task == "mask":
    script = os.path.join(ROOT, "src", "face_mask", "detect_mask_video.py")
    subprocess.run([python_exec, script])

elif args.task == "reid":
    script = os.path.join(ROOT, "src", "reid", "inference.py")
    
    # <<< CHANGE THESE 2 LINES AS PER YOUR EXACT PATHS >>>
    model_path = os.path.join(ROOT, "src", "reid", "output", "finetuned_model.pth")
    image_path = os.path.join(ROOT, "examples", "example_01.png")

    subprocess.run([
        python_exec, 
        script,
        "--model-path", model_path,
        "--image", image_path,
        "--num-classes", "4"
    ])
