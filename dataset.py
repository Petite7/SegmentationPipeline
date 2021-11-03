import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class MedicalDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, predict=False, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.predict = predict
        self.transform = transform
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.predict is True:
            # return test file names ans image set
            if self.transform is not None:
                argumentation = self.transform(image=image)
                image = argumentation["image"]
            return self.images[index], image
        else:
            # return train/val image&label set
            mask_path = os.path.join(self.mask_dir, self.images[index])
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
            mask[mask == 255.0] = 1.0
            # 0.0, 255.0
            # change pixel 255 -> 1 , easy for Relu to process
            if self.transform is not None:
                argumentation = self.transform(image=image, mask=mask)
                image = argumentation["image"]
                mask = argumentation["mask"]
            return image, mask


class ChicagoDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, predict=False, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.predict = predict
        self.transform = transform
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.predict is True:
            # return test file names ans image set
            if self.transform is not None:
                argumentation = self.transform(image=image)
                image = argumentation["image"]
            return self.images[index], image
        else:
            # return train/val image&label set
            mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_label.png"))
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
            mask[mask == 255.0] = 1.0
            # 0.0, 255.0
            # change pixel 255 -> 1 , easy for Relu to process
            if self.transform is not None:
                argumentation = self.transform(image=image, mask=mask)
                image = argumentation["image"]
                mask = argumentation["mask"]
            return image, mask

