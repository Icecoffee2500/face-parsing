import os
import numpy as np
import cv2
import functools
import torch
import torchvision
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
# from skimage import io
import torchvision.transforms.functional as F
from PIL import Image
import random
from typing import List
import warnings
warnings.filterwarnings("ignore")
from utils.transform import TrainTransform
from pathlib import Path

@functools.lru_cache()
def _cached_imread(fname, flags=None):
    return cv2.imread(fname, flags=flags)

class CelebAMaskHQ(Dataset):
    def __init__(self, root, split, resolution: list[int], label_type='all'):
        assert os.path.isdir(root)
        self.resolution = resolution
        # self.root = root
        self.root = Path(root)
        self.split = split
        self.names = []

        if split != 'all':
            # CelebA-HQ to CelebA mapping
            hq_to_orig_mapping = dict()
            orig_to_hq_mapping = dict()
            # mapping_file = os.path.join(root, 'CelebA-HQ-to-CelebA-mapping.txt')
            mapping_file = Path(root) / 'CelebA-HQ-to-CelebA-mapping.txt'
            assert mapping_file.exists()
            for s in open(mapping_file, 'r'):
                if '.jpg' not in s:
                    continue
                idx, _, orig_file = s.split()
                hq_to_orig_mapping[int(idx)] = orig_file
                orig_to_hq_mapping[orig_file] = int(idx)

            # load partition
            # partition_file = os.path.join(root, 'list_eval_partition.txt')
            partition_file = Path(root) / 'list_eval_partition.txt'
            assert partition_file.exists()
            for s in open(partition_file, 'r'):
                if '.jpg' not in s:
                    continue
                orig_file, group = s.split()
                group = int(group)
                if orig_file not in orig_to_hq_mapping:
                    continue
                hq_id = orig_to_hq_mapping[orig_file]
                # split과 group에 해당하는 hq_id를 names에 추가 (train: 0, val: 1, test: 2)
                if split == 'train' and group == 0:
                    self.names.append(str(hq_id))
                elif split == 'val' and group == 1:
                    self.names.append(str(hq_id))
                elif split == 'test' and group == 2:
                    self.names.append(str(hq_id))
        else:
            self.names = [
                n[:-(len('.jpg'))]
                # for n in os.listdir(os.path.join(self.root, 'CelebA-HQ-img'))
                for n in (self.root / 'CelebA-HQ-img')
                if n.endswith('.jpg')
            ]

        self.label_setting = {
            'human': {
                'suffix': [
                    'neck', 'skin', 'cloth', 'l_ear', 'r_ear', 'l_brow', 'r_brow',
                    'l_eye', 'r_eye', 'nose', 'mouth', 'l_lip', 'u_lip', 'hair'
                ],
                'names': [
                    'bg', 'neck', 'face', 'cloth', 'rr', 'lr', 'rb', 'lb', 're',
                    'le', 'nose', 'imouth', 'llip', 'ulip', 'hair'
                ]
            },
            'aux': {
                'suffix': [
                    'eye_g', 'hat', 'ear_r', 'neck_l',
                ],
                'names': [
                    'normal', 'glass', 'hat', 'earr', 'neckl'
                ]
            },
            'all': {
                'suffix': [
                    'neck', 'skin', 'cloth', 'l_ear', 'r_ear', 'l_brow', 'r_brow',
                    'l_eye', 'r_eye', 'nose', 'mouth', 'l_lip', 'u_lip', 'hair',
                    'eye_g', 'hat', 'ear_r', 'neck_l',
                ],
                'names': [
                    'bg', 'neck', 'face', 'cloth', 'lr', 'rr', 'lb', 'rb', 'le',
                    're', 'nose', 'imouth', 'llip', 'ulip', 'hair',
                    'glass', 'hat', 'earr', 'neckl'
                ]
            }
        }[label_type]

        self.transforms_image = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # self.train_transform = TrainTransform(image_size=self.resolution)

        self.transforms_image_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def make_label(self, index, ordered_label_suffix):
        label = np.zeros((512, 512), np.uint8) # (512x512 (원본 label 크기)) 크기의 label 생성 (0으로 초기화))
        name_id = int(self.names[index])
        name5 = '%05d' % name_id
        label_path = self.root / 'CelebAMask-HQ-mask-anno' / str(name_id // 2000)
        for i, label_suffix in enumerate(ordered_label_suffix):
            label_value = i + 1 # label name은 background부터 시작하므로 +1.
            label_fname = label_path / (f"{name5}_{label_suffix}.png") # label (얼굴부위별) mask 파일 경로

            if label_fname.exists():
                # numpy array (512x512) - Resolution size / 각 pixel은 0 혹은 255 (Gray Scale)
                mask = _cached_imread(label_fname, cv2.IMREAD_GRAYSCALE)
                # 해당 label인 픽셀(mask > 0)에 해당 label value를 부여
                label = np.where(mask > 0, np.ones_like(label) * label_value, label)
        return label

    def __getitem__(self, index):
        name = self.names[index]
        # image = cv2.imread(os.path.join(self.root, 'CelebA-HQ-img',name + '.jpg'))
        image = cv2.imread((self.root / 'CelebA-HQ-img' / f"{name}.jpg").as_posix())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # (1024, 1024, 3) - 원본 이미지 크기 # numpy.ndarray
        label = self.make_label(index, self.label_setting['suffix']) # (512, 512) - 원본 label 크기 # numpy.ndarray
        
        # Convert to PIL for compatibility with torchvision functional transforms
        image = Image.fromarray(image)
        label = Image.fromarray(label)

        # Resize image and label to the desired size # 여기서 해상도 조절 가능
        image = F.resize(image, size=self.resolution, interpolation=Image.BICUBIC)
        label = F.resize(label, size=self.resolution, interpolation=Image.NEAREST)
        # Nearest는 가장 가까운 픽셀의 값을 복사해서 사용하기 때문에 class 값이 변경되지 않음 # Segmentation에서의 표준 방식

        # Convert to tensor
        if self.split=='train':
            image = self.transforms_image(image)
            # image, label = self.train_transform(image, label)
        else:
            image = self.transforms_image_test(image)
        
        # label = F.to_tensor(label) # torch.Size([1, 512, 512]) # label이 tensor로 변환되면서 (0~1) 사이의 값으로 변환됨 (FloatTensor)
        # label = torch.squeeze(label) * 255  # Assuming label images are in grayscale
        # # label = label.to(dtype=torch.float)
        # label = label.to(dtype=torch.long)
        label = torch.from_numpy(np.array(label, dtype=np.int64))

        data = {'image': image, 'label': {"segmentation":label, "lnm_seg": torch.zeros([5,2])}, "dataset": 0}
        return data

    def __len__(self):
        return len(self.names)

    def sample_name(self, index):
        return self.names[index]

    @property
    def label_names(self) -> List[str]:
        return self.label_setting['names']

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

def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    tensor = tensor * std + mean  # Apply the unnormalize formula
    tensor = torch.clamp(tensor, 0, 1)  # Clamp the values to be between 0 and 1
    return tensor


if __name__ == '__main__':
    celebamaskhq = CelebAMaskHQ("dataset/CelebAMask-HQ", 'train', 512)
    loader = torch.utils.data.DataLoader(celebamaskhq, batch_size=8, shuffle=True, num_workers=4)
    # loader = torch.utils.data.DataLoader(celebamaskhq, batch_size=1, shuffle=True, num_workers=1)

    print(celebamaskhq.label_names)
    # Check batch
    for i, batch in enumerate(loader):
        # Save face image
        face = unnormalize(batch['image'][0]).permute(1, 2, 0).numpy()
        face = (face * 255).astype(np.uint8)
        # cv2.imwrite(f"samples/face_{i}.png", face[:, :, ::-1])
        cv2.imwrite(f"samples2/face_{i}.png", face[:, :, ::-1])

        # Save visualized mask
        mask = celebamaskhq.visualize_mask(batch["label"]['segmentation'][0].numpy())
        # cv2.imwrite(f"samples/mask_{i}.png", mask[:, :, ::-1])
        cv2.imwrite(f"samples2/mask_{i}.png", mask[:, :, ::-1])

        if i >= 19:
            break

    # class_frequencies = np.zeros(len(celebamaskhq.label_names), dtype=int)

    # total_count = 0
    # for i, batch in enumerate(loader):
    #     labels = batch["label"]["segmentation"].numpy()
    #     for label in labels:
    #         unique = np.unique(label)
    #         for u in unique:
    #             class_frequencies[int(u)] += 1
    #         total_count += 1

    # # Print class frequencies
    # for class_name, frequency in zip(celebamaskhq.label_names, class_frequencies):
    #     print(f"{class_name}: {frequency}")

    # print("Total images: ", total_count)