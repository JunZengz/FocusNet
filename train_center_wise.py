
import os
import random
import time
import datetime
import numpy as np
import albumentations as A
import cv2
from PIL import Image
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import seeding, create_dir, print_and_save, shuffling, epoch_time, calculate_metrics
from model.FocusNet import *
from metrics import DiceLoss, DiceBCELoss
from sklearn.utils import shuffle
from lib import *

def load_names(path, file_path):
    f = open(file_path, "r")
    data = f.read().split("\n")[:-1]
    images = [os.path.join(path,"images", name) + ".jpg" for name in data]
    masks = [os.path.join(path,"masks", name) + ".jpg" for name in data]
    return images, masks

def load_kvasir_data(path):
    train_names_path = f"{path}/train.txt"
    valid_names_path = f"{path}/val.txt"

    train_x, train_y = load_names(path, train_names_path)
    valid_x, valid_y = load_names(path, valid_names_path)

    return (train_x, train_y), (valid_x, valid_y)


def load_test_data(path):
    images = sorted(glob(os.path.join(path, "images", "*.png")))
    image_names = [os.path.splitext(os.path.basename(file))[0] for file in images]
    samples = []
    for image_name in image_names:
        image = os.path.join(path, "images", f"{image_name}.png")
        mask = os.path.join(path, "masks", f"{image_name}.png")
        samples.append((image, mask))

    return samples


def load_polypdb_data_0(path):
    def get_data(path, name, modality):
        samples = []

        images = sorted(glob(os.path.join(path, name, modality, "images", "*.jpg")))
        image_names = [os.path.splitext(os.path.basename(file))[0] for file in images]

        mask_path = os.path.join(path, name, modality, "masks")
        for image_name in image_names:
            image = os.path.join(path, name, modality, "images", f"{image_name}.jpg")

            mask_jpg = os.path.join(mask_path, f"{image_name}.jpg")
            mask_png = os.path.join(mask_path, f"{image_name}.png")
            # 判断 .jpg 掩码文件是否存在，否则使用 .png
            if os.path.exists(mask_jpg):
                mask = mask_jpg
            elif os.path.exists(mask_png):
                mask = mask_png
            else:
                # 如果掩码文件不存在，跳过该样本
                continue

            samples.append((image, mask))

        return samples

    """ Training data """
    all_samples = []
    modality_list = {'BKAI': ['BLI', 'FICE', 'LCI', 'WLI'],
                     'Karolinska': ['WLI'],
                     'Simula': ['NBI', 'WLI']}
    for name in ['BKAI', 'Karolinska', 'Simula']:
        for modality in modality_list[name]:
            all_samples += get_data(path, name, modality)

    # 计算各部分的长度
    total_len = len(all_samples)
    train_len = int(0.8 * total_len)
    test_len = int(0.1 * total_len)
    val_len = total_len - train_len - test_len  # 剩下的就是验证集

    # 划分数据集
    train_samples = all_samples[:train_len]
    test_samples = all_samples[train_len:train_len + test_len]
    valid_samples = all_samples[train_len + test_len:]

    return [train_samples, test_samples, valid_samples]


def load_polypdb_data(path):
    def get_data(path, name, modality):
        samples = []

        images = sorted(glob(os.path.join(path, name, modality, "images", "*.jpg")))
        image_names = [os.path.splitext(os.path.basename(file))[0] for file in images]

        mask_path = os.path.join(path, name, modality, "masks")
        for image_name in image_names:
            image = os.path.join(path, name, modality, "images", f"{image_name}.jpg")

            mask_jpg = os.path.join(mask_path, f"{image_name}.jpg")
            mask_png = os.path.join(mask_path, f"{image_name}.png")
            # 判断 .jpg 掩码文件是否存在，否则使用 .png
            if os.path.exists(mask_jpg):
                mask = mask_jpg
            elif os.path.exists(mask_png):
                mask = mask_png
            else:
                # 如果掩码文件不存在，跳过该样本
                continue

            samples.append((image, mask))

        return samples

    """ Training data """
    train_samples = []
    valid_samples = []
    test_samples = []
    modality_list = {'BKAI': ['BLI', 'FICE', 'LCI', 'WLI'],
                     'Karolinska': ['WLI'],
                     'Simula': ['NBI', 'WLI']}
    for name in ['BKAI', 'Karolinska', 'Simula']:
        for modality in modality_list[name]:
            modality_data = get_data(path, name, modality)
            modality_len = len(modality_data)
            modality_train_len = int(0.8 * modality_len)
            modality_val_len = int(0.1 * modality_len)
            train_samples += modality_data[:modality_train_len]
            valid_samples += modality_data[modality_train_len:modality_train_len + modality_val_len]
            test_samples += modality_data[modality_train_len + modality_val_len:]

    return [train_samples, valid_samples, test_samples]


def load_polypdb_wli_data(path):
    def get_data(path):
        samples = []

        images = sorted(glob(os.path.join(path, "images", "*.jpg")))
        image_names = [os.path.splitext(os.path.basename(file))[0] for file in images]

        for image_name in image_names:
            image = os.path.join(path, "images", f"{image_name}.jpg")

            mask_jpg = os.path.join(path, "masks", f"{image_name}.jpg")
            mask_png = os.path.join(path, "masks", f"{image_name}.png")
            # 判断 .jpg 掩码文件是否存在，否则使用 .png
            if os.path.exists(mask_png):
                mask = mask_png
            elif os.path.exists(mask_jpg):
                mask = mask_jpg
            else:
                # 如果掩码文件不存在，跳过该样本
                continue

            samples.append((image, mask))

        return samples

    """ Training data """
    modality_data = get_data(path)
    modality_len = len(modality_data)
    modality_train_len = int(0.8 * modality_len)
    modality_val_len = int(0.1 * modality_len)
    train_samples = modality_data[:modality_train_len]
    valid_samples = modality_data[modality_train_len:modality_train_len + modality_val_len]
    test_samples = modality_data[modality_train_len + modality_val_len:]

    return [train_samples, valid_samples, test_samples]


class DATASET(Dataset):
    def __init__(self, images_path, masks_path, size, transform=None):
        super().__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        """ Image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        # image = Image.open(self.images_path[index]).convert("RGB")
        # mask = Image.open(self.masks_path[index]).convert("L")

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        image = cv2.resize(image, size)
        image = np.transpose(image, (2, 0, 1))
        image = image/255.0

        mask = cv2.resize(mask, size)
        mask = np.expand_dims(mask, axis=0)
        mask = mask/255.0

        return image, mask

    def __len__(self):
        return self.n_samples


class PolypDB_DATASET(Dataset):
    def __init__(self, samples_path, size, transform=None):
        super().__init__()

        self.samples_path = samples_path
        self.transform = transform
        self.n_samples = len(samples_path)

    def __getitem__(self, index):
        """ Image """
        image = cv2.imread(self.samples_path[index][0], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.samples_path[index][1], cv2.IMREAD_GRAYSCALE)
        # image = Image.open(self.images_path[index]).convert("RGB")
        # mask = Image.open(self.masks_path[index]).convert("L")

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        image = cv2.resize(image, size)
        image = np.transpose(image, (2, 0, 1))
        image = image/255.0

        mask = cv2.resize(mask, size)
        mask = np.expand_dims(mask, axis=0)
        mask = mask/255.0

        return image, mask

    def __len__(self):
        return self.n_samples



def train(model, loader, optimizer, loss_fn, device):
    model.train()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    for i, (x, y) in enumerate(loader):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        # y_pred = model(x)
        # loss = loss_fn(y_pred, y)

        sample = {'images': x, 'masks': y}
        out = model(sample)
        y_pred = out['prediction']
        loss = out['loss']

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        """ Calculate the metrics """
        batch_jac = []
        batch_f1 = []
        batch_recall = []
        batch_precision = []

        y_pred = torch.sigmoid(y_pred)
        for yt, yp in zip(y, y_pred):
            score = calculate_metrics(yt, yp)
            batch_jac.append(score[0])
            batch_f1.append(score[1])
            batch_recall.append(score[2])
            batch_precision.append(score[3])

        epoch_jac += np.mean(batch_jac)
        epoch_f1 += np.mean(batch_f1)
        epoch_recall += np.mean(batch_recall)
        epoch_precision += np.mean(batch_precision)

    epoch_loss = epoch_loss/len(loader)
    epoch_jac = epoch_jac/len(loader)
    epoch_f1 = epoch_f1/len(loader)
    epoch_recall = epoch_recall/len(loader)
    epoch_precision = epoch_precision/len(loader)

    return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]

def evaluate(model, loader, loss_fn, device):
    model.eval()

    epoch_loss = 0
    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            # y_pred = model(x)
            # loss = loss_fn(y_pred, y)

            sample = {'images': x, 'masks': y}
            out = model(sample)
            y_pred = out['prediction']
            loss = out['loss']

            epoch_loss += loss.item()

            """ Calculate the metrics """
            batch_jac = []
            batch_f1 = []
            batch_recall = []
            batch_precision = []

            y_pred = torch.sigmoid(y_pred)
            for yt, yp in zip(y, y_pred):
                score = calculate_metrics(yt, yp)
                batch_jac.append(score[0])
                batch_f1.append(score[1])
                batch_recall.append(score[2])
                batch_precision.append(score[3])

            epoch_jac += np.mean(batch_jac)
            epoch_f1 += np.mean(batch_f1)
            epoch_recall += np.mean(batch_recall)
            epoch_precision += np.mean(batch_precision)

        epoch_loss = epoch_loss/len(loader)
        epoch_jac = epoch_jac/len(loader)
        epoch_f1 = epoch_f1/len(loader)
        epoch_recall = epoch_recall/len(loader)
        epoch_precision = epoch_precision/len(loader)

        return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Directories """
    create_dir("files")

    """ configs """
    model_name = 'FocusNet'

    """ Training logfile """
    train_log_path = f"files/center_wise/{model_name}/train_log.txt"
    if os.path.exists(train_log_path):
        print("Log file exists")
    else:
        create_dir(f"files/center_wise/{model_name}")
        train_log = open(train_log_path, "w")
        train_log.write("\n")
        train_log.close()

    """ Record Date & Time """
    datetime_object = str(datetime.datetime.now())
    print_and_save(train_log_path, datetime_object)
    print("")

    """ Hyperparameters """
    image_size = 256
    size = (image_size, image_size)
    batch_size = 16
    num_epochs = 500
    lr = 1e-4
    early_stopping_patience = 50
    checkpoint_path = f"files/center_wise/{model_name}/checkpoint.pth"
    path = "data/PolypDB/PolypDB_center_wise/Simula/WLI"

    data_str = f"Image Size: {size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
    data_str += f"Early Stopping Patience: {early_stopping_patience}\n"
    print_and_save(train_log_path, data_str)

    """ Dataset """
    train_samples, valid_samples, test_samples = load_polypdb_wli_data(path)
    samples = shuffle(train_samples, random_state=42)
    data_str = f"Dataset Size:\nTrain: {len(train_samples)} - Valid: {len(valid_samples)} - Test: {len(test_samples)}\n"

    print_and_save(train_log_path, data_str)

    """ Data augmentation: Transforms """
    transform =  A.Compose([
        A.Rotate(limit=35, p=0.3),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)
    ])

    """ Dataset and loader """
    train_dataset = PolypDB_DATASET(train_samples, size, transform=transform)
    valid_dataset = PolypDB_DATASET(valid_samples, size, transform=None)

    # create_dir("data")
    # for i, (x, y) in enumerate(train_dataset):
    #     x = np.transpose(x, (1, 2, 0)) * 255
    #     y = np.transpose(y, (1, 2, 0)) * 255
    #     y = np.concatenate([y, y, y], axis=-1)
    #     cv2.imwrite(f"data/{i}.png", np.concatenate([x, y], axis=1))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    """ Model """
    device = torch.device('cuda')
    model = eval(model_name)()
    model = model.to(device)
    print(f"train model: {model_name}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = DiceBCELoss()
    loss_name = "BCE Dice Loss"
    data_str = f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    """ Training the model """
    best_valid_metrics = 0.0
    early_stopping_count = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_metrics = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss, valid_metrics = evaluate(model, valid_loader, loss_fn, device)
        scheduler.step(valid_loss)

        if valid_metrics[1] > best_valid_metrics:
            data_str = f"Valid F1 improved from {best_valid_metrics:2.4f} to {valid_metrics[1]:2.4f}. Saving checkpoint: {checkpoint_path}"
            print_and_save(train_log_path, data_str)

            best_valid_metrics = valid_metrics[1]
            torch.save(model.state_dict(), checkpoint_path)
            early_stopping_count = 0

        elif valid_metrics[1] < best_valid_metrics:
            early_stopping_count += 1

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        data_str += f"\tTrain Loss: {train_loss:.4f} - Jaccard: {train_metrics[0]:.4f} - F1: {train_metrics[1]:.4f} - Recall: {train_metrics[2]:.4f} - Precision: {train_metrics[3]:.4f}\n"
        data_str += f"\t Val. Loss: {valid_loss:.4f} - Jaccard: {valid_metrics[0]:.4f} - F1: {valid_metrics[1]:.4f} - Recall: {valid_metrics[2]:.4f} - Precision: {valid_metrics[3]:.4f}\n"
        print_and_save(train_log_path, data_str)

        if early_stopping_count == early_stopping_patience:
            data_str = f"Early stopping: validation loss stops improving from last {early_stopping_patience} continously.\n"
            print_and_save(train_log_path, data_str)
            break
