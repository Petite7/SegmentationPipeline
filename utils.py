import os
import cv2
import csv
import torch
import torchvision
from PIL import Image
import numpy as np
import pandas as pd
import ttach as tta
from tqdm import tqdm
from dataset import MedicalDataset
from torch.utils.data import DataLoader


def save_checkpoint(model, optimizer, scheduler,  filename="TianChi_checkpoint.pth"):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict()
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint, model, optimizer=None, scheduler=None):
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])


def save_max_score(path, acc, dice):
    ma, md = load_max_score(path)
    ma = ma if (ma > acc) else acc
    md = md if (md > dice) else dice
    with open(path, 'w') as f:
        f.writelines(f"{ma},{md}")
    return float(ma), float(md)


def load_max_score(path):
    if os.path.isfile(path) is True:
        with open(path, 'r') as f:
            ma, md = f.readline().split(',')
            return float(ma), float(md)
    else:
        return 0, 0


def get_loaders(train_dir, train_mask_dir, train_transform, batch_size,
                predict=False,
                val_dir=None, val_mask_dir=None, val_transform=None, num_workers=4, pin_memory=True):
    train_ds = MedicalDataset(img_dir=train_dir, mask_dir=train_mask_dir, predict=predict, transform=train_transform)
    train_loaders = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                               shuffle=True)
    # Only return test dataset when predict
    if predict is True:
        return train_loaders
    val_ds = MedicalDataset(img_dir=val_dir, mask_dir=val_mask_dir, predict=predict, transform=val_transform)
    val_loaders = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                             shuffle=False)
    return train_loaders, val_loaders


def save_as_images(loader, model, folder="save_images/", device="cuda"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    print(f"> Saving model out as images to folder : {folder}")
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            pre = torch.sigmoid(model(x))
            pre = (pre > 0.5).float()
        torchvision.utils.save_image(pre, f"{folder}/pred_{idx}.png")
        # torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/pred_s_{idx}.png")


def predicted_as_images(loader, model, folder="predict/", ttach=False, device="cuda"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    print(f"> Saving predictions as images to folder : {folder}")
    loop = tqdm(loader)
    model.eval()
    if ttach is True:
        transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            tta.Rotate90(angles=[0, 90, 180, 270])
        ])
        model = tta.SegmentationTTAWrapper(model, transforms, merge_mode='mean')
    for i, (name, x) in enumerate(loop):
        x = x.to(device=device)
        with torch.no_grad():
            pre = torch.sigmoid(model(x))
            pre = (pre > 0.5).float()
        torchvision.utils.save_image(pre, f"{folder}/{name[0]}")


def rle_encode(im):
    im[im == 255.0] = 1.0
    pixels = im.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(512, 512)):
    s = mask_rle.split()
    if mask_rle == "nan":
        s = []
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 255
    return img.reshape(shape, order='F')


def encode_as_rle_csv(img_dir):
    data = pd.read_csv("test_a_samplesubmit.csv", header=None, sep="\t", names=['name', 'mask'])
    names = []
    masks = []
    for i, submit_title in tqdm(list(data.iterrows())):
        name = str(submit_title['name']).replace(".jpg", ".png")
        image = np.array(Image.open(img_dir + "/" + name).convert("L"))
        mask = rle_encode(image)
        name = name.replace(".png", ".jpg")
        names.append(name)
        masks.append(mask)
    df = pd.DataFrame({"C1": names, "C2": masks})
    df.to_csv(r"submit.csv", index=False, header=False, mode='w', sep="\t")


def mask2rle(img):
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def files_mask2rle(path):
    files = os.listdir(path)
    csv = open(r'submit_tta.csv', 'w')
    for file in files:
        fp = os.path.join(path, file)
        img = cv2.imread(fp)
        w, h = img.shape[1::-1]
        img = img[:, :, 0]
        img = img // 255
        result = mask2rle(img)
        csv.writelines("{},{} {},{}\n".format(file, w, h, result))

