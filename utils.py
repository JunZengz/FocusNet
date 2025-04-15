import os
import random
import numpy as np
import cv2
from tqdm import tqdm
import torch
from sklearn.utils import shuffle
from metrics import precision, recall, F2, dice_score, jac_score
from sklearn.metrics import accuracy_score, confusion_matrix

""" Seeding the randomness. """
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

""" Create a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

""" Shuffle the dataset. """
def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def print_and_save(file_path, data_str):
    print(data_str)
    with open(file_path, "a") as file:
        file.write(data_str)
        file.write("\n")

def calculate_metrics(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    y_pred = y_pred > 0.5
    y_pred = y_pred.reshape(-1)
    y_pred = y_pred.astype(np.uint8)

    y_true = y_true > 0.5
    y_true = y_true.reshape(-1)
    y_true = y_true.astype(np.uint8)

    ## Score
    score_jaccard = jac_score(y_true, y_pred)
    score_f1 = dice_score(y_true, y_pred)
    score_recall = recall(y_true, y_pred)
    score_precision = precision(y_true, y_pred)
    score_fbeta = F2(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc, score_fbeta]


def generate_features_map_first(x, size=None):
    img = x[0, 0, :, :].cpu().numpy()
    pmin = np.min(img)
    pmax = np.max(img)
    img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
    img = img.astype(np.uint8)  # 转成unit8
    if size is not None:
        img = cv2.resize(img, size)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
    return img


def generate_features_map(x, size=None):
    avg_img = np.mean(x[0, :, :, :].cpu().numpy(), axis=0)
    pmin = np.min(avg_img)
    pmax = np.max(avg_img)
    avg_img = ((avg_img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
    avg_img = avg_img.astype(np.uint8)  # 转成unit8
    if size is not None:
        avg_img = cv2.resize(avg_img, size)
    avg_img = cv2.applyColorMap(avg_img, cv2.COLORMAP_JET)  # 生成heat map
    return avg_img


def generate_max_features_map(x, size=None):
    max_img = np.max(x[0, :, :, :].cpu().numpy(), axis=0)
    pmin = np.min(max_img)
    pmax = np.max(max_img)
    max_img = ((max_img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
    max_img = max_img.astype(np.uint8)  # 转成unit8
    if size is not None:
        max_img = cv2.resize(max_img, size)
    max_img = cv2.applyColorMap(max_img, cv2.COLORMAP_JET)  # 生成heat map
    return max_img


def save_feats_mean(x, size):
    x = x.detach().cpu().numpy()
    x = np.transpose(x[0], (1, 2, 0))
    x = np.mean(x, axis=-1)
    x = x/np.max(x)
    x = x * 255.0
    x = x.astype(np.uint8)
    if size is not None:
        x = cv2.resize(x, size)
    x = cv2.applyColorMap(x, cv2.COLORMAP_JET)
    # x = np.array(x, dtype=np.uint8)
    return x

def save_feats_max(x, size):
    x = x.detach().cpu().numpy()
    x = np.transpose(x[0], (1, 2, 0))
    x = np.max(x, axis=-1)
    x = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-6)
    x = x * 255.0
    x = x.astype(np.uint8)
    if size is not None:
        x = cv2.resize(x, size)
    x = cv2.applyColorMap(x, cv2.COLORMAP_JET)
    # x = np.array(x, dtype=np.uint8)
    return x