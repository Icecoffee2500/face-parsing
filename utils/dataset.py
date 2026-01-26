import os
import glob
import numpy as np
from PIL import Image
import torch
import json
import cv2

from torch.utils.data import Dataset
from utils.transform import DefaultTransform
from utils.transform import TrainTransform


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
    PART_ORDER = [
        "neck",
        "skin",
        "cloth",
        "l_ear",
        "r_ear",
        "l_brow",
        "r_brow",
        "l_eye",
        "r_eye",
        "nose",
        "mouth",
        "l_lip",
        "u_lip",
        "hair",
        "eye_g",
        "hat",
        "ear_r",
        "neck_l",
    ]

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
        base_dir = os.path.join(self.labels_dir, str(folder))

        label = np.zeros((512, 512), dtype=np.uint8)
        for part_name in self.PART_ORDER:
            class_id = self.PARTS.get(part_name)
            if class_id is None:
                continue
            part_file = os.path.join(base_dir, f"{image_id:05d}_{part_name}.png")
            if not os.path.isfile(part_file):
                continue
            mask = np.array(Image.open(part_file).convert("L"))
            label[mask > 0] = class_id
        # 이 과정을 거치면 label은 (512x512) 크기의 이미지이고, 각 pixel은 1~18 사이의 정수값(class_id)을 가진다.

        return Image.fromarray(label, mode="L")
    
    @staticmethod
    def visualize_mask(mask):
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        color_mapping = np.array([
            [0, 0, 0], 
            [204, 0, 0], 
            [76, 153, 0],
            [204, 204, 0], 
            [51, 51, 255], 
            [204, 0, 204], 
            [0, 255, 255],
            [51, 255, 255], 
            [102, 51, 0], 
            [255, 0, 0], 
            [102, 204, 0],
            [255, 255, 0], 
            [0, 0, 153], 
            [0, 0, 204], 
            [255, 51, 153], 
            [0, 204, 204], 
            [0, 51, 0], 
            [255, 153, 51], 
            [0, 204, 0]
        ], dtype=np.uint8)

        for index, color in enumerate(color_mapping):
            i, j = np.where(mask == index)
            color_mask[i, j] = color
        return color_mask

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

def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    tensor = tensor * std + mean  # Apply the unnormalize formula
    tensor = torch.clamp(tensor, 0, 1)  # Clamp the values to be between 0 and 1
    return tensor

if __name__ == '__main__':
    data_root = 'dataset/CelebAMask-HQ/'
    images_dir = os.path.join(data_root, 'CelebA-HQ-img')
    labels_dir = os.path.join(data_root, 'CelebAMask-HQ-mask-anno')

    base_dataset = CelebAMaskHQ(images_dir, labels_dir, transform=DefaultTransform())
    total_len = len(base_dataset)
    train_indices, val_indices = load_or_create_split(
        os.path.join(data_root, 'train_val_split.json'), total_len, val_ratio=0.2, seed=42
    )

    train_dataset = torch.utils.data.Subset(
        CelebAMaskHQ(images_dir, labels_dir, transform=TrainTransform(image_size=[512, 512])),
        train_indices,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    
    # celebamaskhq = CelebAMaskHQ("dataset/CelebAMask-HQ", 'train', 512)
    # loader = torch.utils.data.DataLoader(celebamaskhq, batch_size=8, shuffle=True, num_workers=4)
    # loader = torch.utils.data.DataLoader(celebamaskhq, batch_size=1, shuffle=True, num_workers=1)

    # print(celebamaskhq.label_names)
    # Check batch
    for i, batch in enumerate(train_loader):
        # Save face image
        face = unnormalize(batch[0][0]).permute(1, 2, 0).numpy()
        face = (face * 255).astype(np.uint8)
        # cv2.imwrite(f"samples/face_{i}.png", face[:, :, ::-1])
        cv2.imwrite(f"samples_legacy/face_{i}.png", face[:, :, ::-1])

        # Save visualized mask
        mask = CelebAMaskHQ.visualize_mask(batch[1][0].numpy())
        # cv2.imwrite(f"samples/mask_{i}.png", mask[:, :, ::-1])
        cv2.imwrite(f"samples_legacy/mask_{i}.png", mask[:, :, ::-1])

        if i >= 19:
            break