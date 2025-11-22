import os
from torch.utils.data import Dataset
from PIL import Image

class Market1501(Dataset):
    def __init__(self, root, transform=None, mode="train"):
        """
        mode: "train", "query", "gallery"
        Expects Market-1501 structure where images are in folders or filenames like: <pid>_c<cam>_...
        """
        self.root = root
        self.transform = transform
        self.mode = mode

        # assume images stored in root + /bounding_box_train, /query, /bounding_box_test
        if mode == "train":
            img_dir = os.path.join(root, "bounding_box_train")
        elif mode == "query":
            img_dir = os.path.join(root, "query")
        else:  # gallery / test
            img_dir = os.path.join(root, "bounding_box_test")

        self.img_paths = []
        self.pids = []
        self.cam_ids = []

        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith((".jpg", ".png")):
                continue
            pid = int(fname.split("_")[0])
            cam = int(fname.split("c")[1][0])
            self.img_paths.append(os.path.join(img_dir, fname))
            self.pids.append(pid)
            self.cam_ids.append(cam)

        # remap pids to contiguous labels (only for train typically)
        unique_pids = sorted(set(self.pids))
        self.pid2label = {pid: idx for idx, pid in enumerate(unique_pids)}
        self.labels = [self.pid2label[p] for p in self.pids]

        # expose num_classes for model init
        self.num_classes = len(unique_pids)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        pid = self.pids[idx]
        label = self.labels[idx]
        cam = self.cam_ids[idx]
        return img, label, cam
