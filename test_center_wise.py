import os, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import torch
from model.FocusNet import *
from lib import *
from utils import create_dir, seeding
from utils import calculate_metrics
from train_center_wise import load_polypdb_wli_data

def load_test_data(path):
    images = sorted(glob(os.path.join(path, "images", "*.jpg")))
    image_names = [os.path.splitext(os.path.basename(file))[0] for file in images]
    samples = []

    for image_name in image_names:
        image = os.path.join(path, "images", f"{image_name}.jpg")
        mask_png = os.path.join(path, "masks", f"{image_name}.png")
        mask_jpg = os.path.join(path, "masks", f"{image_name}.jpg")
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


def evaluate(model, save_path, test_samples, size):
    """ Loading other comparitive model masks """
    # comparison_path = "/media/nikhil/LAB/ML/ME/COMPARISON/Kvasir-SEG/"

    # unet_mask = sorted(glob(os.path.join(comparison_path, "UNET", "results", "Kvasir-SEG", "mask", "*")))
    # deeplabv3plus_mask = sorted(glob(os.path.join(comparison_path, "DeepLabV3+_50", "results", "Kvasir-SEG", "mask", "*")))


    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []

    for i, (x, y) in tqdm(enumerate(test_samples), total=len(test_samples)):
        name = y.split("/")[-1].split(".")[0]

        """ Image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, size)
        save_img = image
        image = np.transpose(image, (2, 0, 1))
        image = image/255.0
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        image = image.to(device)

        """ Mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, size)
        save_mask = mask
        save_mask = np.expand_dims(save_mask, axis=-1)
        save_mask = np.concatenate([save_mask, save_mask, save_mask], axis=2)
        mask = np.expand_dims(mask, axis=0)
        mask = mask/255.0
        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        mask = mask.to(device)

        with torch.no_grad():
            """ FPS calculation """
            start_time = time.time()

            sample = {'images': image, 'masks': mask}
            out = model(sample)
            y_pred = out['prediction']

            end_time = time.time() - start_time
            time_taken.append(end_time)

            """ Evaluation metrics """
            y_pred = torch.sigmoid(y_pred)
            score = calculate_metrics(mask, y_pred)
            metrics_score = list(map(add, metrics_score, score))

            """ Predicted Mask """
            y_pred = y_pred[0].cpu().numpy()
            y_pred = np.squeeze(y_pred, axis=0)
            y_pred = y_pred > 0.5
            y_pred = y_pred.astype(np.int32)
            y_pred = y_pred * 255
            y_pred = np.array(y_pred, dtype=np.uint8)
            y_pred = np.expand_dims(y_pred, axis=-1)
            y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=2)

        """ Save the image - mask - pred """
        line = np.ones((size[0], 10, 3)) * 255
        cat_images = np.concatenate([
            save_img, line,
            save_mask, line,
            y_pred], axis=1)

        cv2.imwrite(f"{save_path}/joint/{name}.jpg", cat_images)
        cv2.imwrite(f"{save_path}/mask/{name}.jpg", y_pred)

    jaccard = metrics_score[0]/len(test_samples)
    f1 = metrics_score[1]/len(test_samples)
    recall = metrics_score[2]/len(test_samples)
    precision = metrics_score[3]/len(test_samples)
    acc = metrics_score[4]/len(test_samples)
    f2 = metrics_score[5]/len(test_samples)

    print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f} - F2: {f2:1.4f}")

    mean_time_taken = np.mean(time_taken)
    mean_fps = 1/mean_time_taken
    print("Mean FPS: ", mean_fps)


if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ configs """
    model_name = 'FocusNet'
    checkpoint_path = f"files/center_wise/{model_name}/checkpoint.pth"

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = eval(model_name)()
    model = eval(f'build_{model_name}')()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"test model: {model_name}")

    """ Test dataset """
    test_center_list = ['Simula', 'BKAI', 'Karolinska']
    for test_center in test_center_list:
        save_path = f"files/center_wise/{model_name}/results/{test_center}/WLI"

        test_path = f"data/PolypDB/PolypDB_center_wise/{test_center}/WLI"

        if test_center == 'Simula':
            _, _, test_samples = load_polypdb_wli_data(test_path)
        else:
            test_samples = load_test_data(test_path)

        print(f"test_center: {test_center}, test size: {len(test_samples)}")

        create_dir(save_path)
        for item in ["mask", "joint"]:
            create_dir(f"{save_path}/{item}")

        size = (256, 256)
        evaluate(model, save_path, test_samples, size)

