import torch
import os
import shutil
import albumentations as Alb
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from models.Unet.ClassicUnet import Unet
from acc_loss import check_accuracy_binary
from models.Segformer.SegFormer import SegFormerNet
from utils import *
from acc_loss import DiceLossPlusBECLoss
from torch.utils.tensorboard import SummaryWriter

# Hyper parameters etc. =============================================================
RETRAIN = False
# Warning : GPU Memory x 2 , training time x 2!
FROM_TEACHER = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARN_RATE = 8.618e-5 if RETRAIN else 2.618e-4
LR_STEP = 10
DESCEND_RATE = 0.80
BATCH_SIZE = 3
NUM_EPOCHS = 100
NUM_WORKS = 8
IMAGE_HEIGHT = 512  # 512 originally
IMAGE_WIDTH = 512  # 512 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "./PadMedical/train"
TRAIN_MASK_DIR = "./PadMedical/train_mask"
VAL_IMG_DIR = "./PadMedical/val"
VAL_MASK_DIR = "./PadMedical/val_mask"
CHECKPOINT = "Medical_checkpoint.pth"
TEACHER = "Teacher.pth"
PROJECT_PATH = r'/mnt/share/ycchen/RunTest'
LOG_PATH = r'logs/Medical'


# ====================================================================================

# Add Tensorboard support.
if os.path.exists(LOG_PATH):
    os.system('rm -rf ' + LOG_PATH)
os.makedirs(LOG_PATH)
writer = SummaryWriter(LOG_PATH)
logs = os.path.join(PROJECT_PATH, LOG_PATH)
os.system('ln -s ' + logs + ' /mnt/tensorboard')


# tqdm : a progress bar
def train_fn(loader, model, optimizer, loss_fn, scaler, epoch, from_model=None):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward, amp.autocast: auto cast/align dtype
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            if from_model is not None:
                loss += loss_fn(predictions, from_model(data))

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        # update tensorboard
        writer.add_scalar('train_loss', loss.item(), epoch)


def main():
    # train/val dataset augmentations
    train_transform = Alb.Compose(
        [
            Alb.RandomRotate90(p=0.6),
            Alb.HorizontalFlip(p=0.6),
            Alb.VerticalFlip(p=0.6),
            Alb.ElasticTransform(p=0.5),
            Alb.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ]
    )

    val_transform = Alb.Compose(
        [
            Alb.RandomRotate90(p=0.5),
            Alb.HorizontalFlip(p=0.5),
            Alb.VerticalFlip(p=0.5),
            Alb.ElasticTransform(p=0.2),
            Alb.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ]
    )

    # models = Unet(in_channels=3, out_channels=1).to(DEVICE)

    model = SegFormerNet(1, [IMAGE_HEIGHT, IMAGE_WIDTH]).to(DEVICE)
    model = nn.DataParallel(model)
    load_pretrain_train(r'mit_b5.pth', model)

    teacher = None
    if FROM_TEACHER is True:
        teacher = SegFormerNet(1, [IMAGE_HEIGHT, IMAGE_WIDTH]).to(DEVICE)
        teacher = nn.DataParallel(teacher)
        load_pretrain_predict(r'mit_b5.pth', teacher)
        print(f"[!] Training supervised by model : {TEACHER}")
        load_checkpoint(torch.load(TEACHER), model)

    # BCE: Binary CrossEntropy, WithLogits: ADD sigmoid in the models
    # If the out_channels are 3 or more, use CrossEntropy loss
    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = DiceLossPlusBECLoss(k1=0.0, k2=1.0)

    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)

    # ====Multiple lr scheduler : cosine annealing, warm_up, step_lr, reduce_lr
    # scheduler_cos = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-9, last_epoch=5)

    warm_up_epochs = 15
    warm_up_with_step = lambda epo: (epo + 1) / warm_up_epochs if epo < warm_up_epochs else \
        DESCEND_RATE ** ((epo - warm_up_epochs) // LR_STEP)
    scheduler_warm_up = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_step)

    # scheduler_cond = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=DESCEND_RATE,
    #                                                       patience=LR_STEP//2, verbose=True,
    #                                                       min_lr=1e-10, cooldown=3)

    # scheduler_step = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP, gamma=DESCEND_RATE)

    scheduler = scheduler_warm_up

    # scaler : for auto scale/cast/align data type
    scaler = torch.cuda.amp.GradScaler()
    train_loader, val_loader = get_loaders(
        train_dir=TRAIN_IMG_DIR,
        train_mask_dir=TRAIN_MASK_DIR,
        train_transform=train_transform,
        batch_size=BATCH_SIZE,
        predict=False,
        val_dir=VAL_IMG_DIR,
        val_mask_dir=VAL_MASK_DIR,
        val_transform=val_transform,
        num_workers=NUM_WORKS,
        pin_memory=PIN_MEMORY
    )

    if LOAD_MODEL:
        print(f"> Loading checkpoint from : '{CHECKPOINT}'")
        load_checkpoint(torch.load(CHECKPOINT), model, optimizer, scheduler)

    ma, md = "0", "0"
    penalty = 0.01
    if RETRAIN:
        ma, md = load_max_score(r"max_score.csv")
        print(f"> Last train best score : acc = {ma}, dice = {md}")
    max_acc = float(ma) if RETRAIN else 0
    max_dice = float(md) - penalty if RETRAIN else 0

    # Train =========================================================================================================
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, epoch, teacher)

        # check accuracy
        acc, dice = check_accuracy_binary(val_loader, model, device=DEVICE)
        # scheduler.step(dice)
        scheduler.step()

        # add tensorboard scales
        writer.add_scalar('val_acc', acc, global_step=epoch)
        writer.add_scalar('val_dice', dice, global_step=epoch)

        # print epoch state
        for group in optimizer.param_groups:
            writer.add_scalar('learning rate', group['lr'], global_step=epoch)
            print(
                f"--- Epoch : {epoch}/{NUM_EPOCHS} | lr = {group['lr']:.7e}   |  acc = {acc:.5f}   |  dice = {dice:.5f}")

        # save models
        if dice >= max_dice:
            print(f"> Saving checkpoint to : '{CHECKPOINT}'")
            save_checkpoint(model, optimizer, scheduler, CHECKPOINT)
            max_acc, max_dice = save_max_score(r"max_score.csv", acc, dice)

            # print some examples to a folder
            save_as_images(val_loader, model, folder="save_images/", device=DEVICE)
    # Train =========================================================================================================


# num_works cannot be run in python main, so you have to do this
if __name__ == '__main__':
    main()
