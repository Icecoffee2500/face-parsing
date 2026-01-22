import os
import glob
import numpy as np
from PIL import Image
import torch
import json

from torch.utils.data import Dataset
from utils.transform import DefaultTransform


class CelebAMaskHQ(Dataset):
    PARTS = {
        "skin": 1,
        "l_brow": 2,
        "r_brow": 3,
        "l_eye": 4,
        "r_eye": 5,
        "eye_g": 6,
        "l_ear": 7,
        "r_ear": 8,
        "ear_r": 9,
        "nose": 10,
        "mouth": 11,
        "u_lip": 12,
        "l_lip": 13,
        "neck": 14,
        "neck_l": 15,
        "cloth": 16,
        "hair": 17,
        "hat": 18,
    }

    def __init__(self, images_dir, labels_dir, transform=None) -> None:
        super().__init__()
        self.images_dir = images_dir
        self.labels_dir = labels_dir

        if transform is None:
            transform = DefaultTransform()

        self.transform = transform

        self.image_files = []
        for filename in [
            x for x in os.listdir(self.images_dir)
            if os.path.splitext(x)[1] in ('.jpg', '.jpeg', '.png')
        ]:
            image_path = os.path.join(self.images_dir, filename)
            if os.path.isfile(image_path):
                self.image_files.append(image_path)

        self.image_files.sort()

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        image_path = self.image_files[idx]
        image_id = self._image_id_from_filename(image_path) # 0, 1, 2, ..., 2999

        image = Image.open(image_path)
        # mask image size is 512x512, so original image needs to be resized from 1024x1024 to 512x512
        image = image.resize((512, 512), Image.BILINEAR)

        label = self._build_label(image_id)

        image, label = self.transform(image, label)
        label = np.array(label).astype(np.int64)

        return image, label

    def _image_id_from_filename(self, image_path: str) -> int:
        filename = os.path.basename(image_path)
        stem, _ = os.path.splitext(filename)
        return int(stem)

    def _build_label(self, image_id: int) -> Image.Image:
        folder = image_id // 2000 # 0, 1, 2, ..., 14
        pattern = os.path.join(self.labels_dir, str(folder), f"{image_id:05d}_*.png")
        part_files = glob.glob(pattern)

        label = np.zeros((512, 512), dtype=np.uint8)
        for part_file in part_files:
            name = os.path.splitext(os.path.basename(part_file))[0].split("_", 1)[1]
            class_id = self.PARTS.get(name)
            if class_id is None:
                continue
            mask = np.array(Image.open(part_file).convert("L"))
            label[mask > 0] = class_id
        # 이 과정을 거치면 label은 (512x512) 크기의 이미지이고, 각 pixel은 1~18 사이의 정수값(class_id)을 가진다.

        return Image.fromarray(label, mode="L")

def load_or_create_split(split_file, total_len, val_ratio=0.2, seed=42):
    if os.path.isfile(split_file):
        with open(split_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['train_indices'], data['val_indices']

    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_len, generator=g).tolist()
    train_size = int((1.0 - val_ratio) * total_len)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    split_dir = os.path.dirname(split_file)
    if split_dir:
        os.makedirs(split_dir, exist_ok=True)
    with open(split_file, 'w', encoding='utf-8') as f:
        json.dump({'train_indices': train_indices, 'val_indices': val_indices}, f)

    return train_indices, val_indices