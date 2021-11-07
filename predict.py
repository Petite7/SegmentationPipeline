import torch
import torch.nn as nn
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from models.Unet.ClassicUnet import Unet
from models.Segformer.SegFormer import SegFormerNet
from utils import *
from preprocess import pad_to_shape, crop_to_shape


# Hyper parameters etc.-------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_EPOCHS = 1
NUM_WORKS = 4
IMAGE_HEIGHT = 512  # 512 originally
IMAGE_WIDTH = 512  # 512 originally
PIN_MEMORY = True
LOAD_MODEL = True
TEST_IMG_DIR = "PTest"
PRED_IMG_DIR = "predict"
CHECKPOINT = r"Medical_checkpoint.pth"
# ----------------------------------------------------------------------------------


def main():
    test_transform = alb.Compose(
        [
            # alb.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            alb.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ]
    )

    # model = Unet(in_channels=3, out_channels=1).to(DEVICE)
    model = SegFormerNet(1, [IMAGE_HEIGHT, IMAGE_WIDTH]).to(DEVICE)
    model = nn.DataParallel(model)
    load_pretrain_predict(r'mit_b5.pth', model)

    test_loader = get_loaders(TEST_IMG_DIR, BATCH_SIZE, test_transform, BATCH_SIZE, predict=True, num_workers=NUM_WORKS, pin_memory=PIN_MEMORY)
    load_checkpoint(torch.load(CHECKPOINT), model)

    predicted_as_images(test_loader, model, folder="predict/", ttach=True, device=DEVICE)
    crop_to_shape(r'predict', r'RecoverPredict', file_path=r'original_test_wh.csv')
    files_mask2rle(r'RecoverPredict')


if __name__ == '__main__':
    main()
