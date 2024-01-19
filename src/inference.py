import os
import os.path as osp
import argparse
import logging
import json

import numpy as np
import pandas as pd
from PIL import Image
import cv2

import torch
import torchvision.transforms as transforms
from torchvision.transforms import Compose
import rootutils
import pprint

from tqdm import tqdm
from icecream import ic

pp = pprint.PrettyPrinter(depth=4)
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.models.face_module import FaceLitModule
from src.data.components.face_dataset import FaceDataset
from src.data.components.transforms import get_img_trans
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def pred2labels():
    race = {
        "0": "Caucasian",
        "1": "Mongoloid",
        "2": "Negroid",
    }
    gender = {"1": "Male", "0": "Female"}
    skintone = {"3": "mid-light", "1": "light", "2": "mid-dark", "0": "dark"}
    emotion = {
        "3": "Neutral",
        "2": "Happiness",
        "0": "Anger",
        "5": "Surprise",
        "1": "Fear",
        "4": "Sadness",
    }
    masked = {"1": "unmasked", "0": "masked"}
    age = {
        "0": "Baby",
        "1": "Kid",
        "2": "Teenager",
        "3": "20-30s",
        "4": "40-50s",
        "5": "Senior",
    }
    return race, gender, age, skintone, emotion, masked


def torch2torchscript(input_path, output_path):
    if not osp.exists(output_path) and input_path is not None:
        model = FaceLitModule.load_from_checkpoint(input_path)
        script = model.to_torchscript()
        torch.jit.save(script, output_path)
        logger.info("Model saved in {}".format(output_path))
    else:
        pass  # do nothing
    logger.info("Load model from {}".format(output_path))
    model = torch.jit.load(output_path)
    return model


def load_image(img_path):
    # img = Image.open(image_path)
    # img = img.convert("RGB")
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
    return img


def posprocess_pred(vals):
    attr_vals = {}
    race_map, gender_map, age_map, skintone_map, emotion_map, masked_map = pred2labels()
    attrs_dict = {
        "race": race_map,
        "gender": gender_map,
        "age": age_map,
        "skintone": skintone_map,
        "emotion": emotion_map,
        "masked": masked_map,
    }

    for i, attr in enumerate(["race", "gender", "age", "skintone", "emotion", "masked"]):
        attr_val = vals[i]

        if attr == "age":
            attr_val = np.round(attr_val)
            attr_val = np.sum(attr_val).astype(int)
        else:
            attr_val = np.argmax(attr_val)

        attr_val = attrs_dict[attr][str(attr_val)]
        attr_vals[attr] = attr_val

    return attr_vals


def model_predict_images(
    model,
    image_paths,
    transforms=None,
):
    model.eval()
    preds = []

    image_names = [os.path.basename(image_path) for image_path in image_paths]
    imgs = [load_image(path) for path in image_paths]
    if transforms:
        imgs = [transforms(image=img)["image"] for img in imgs]
    imgs = torch.stack(imgs).to(device)
    with torch.no_grad():
        ys = model(imgs)

    ys = [y.cpu().numpy().tolist() for y in ys]
    ys = list(zip(*ys))

    for img_name, y in zip(image_names, ys):
        pred = posprocess_pred(y)
        pred["face_file_name"] = img_name
        preds.append(pred)

    return preds


def model_predict_img_dir(model, img_dir, transforms=None, batch_size=32):
    img_names = os.listdir(img_dir)
    model.eval()
    preds = {}

    for idx in tqdm(range(0, len(img_names), batch_size)):
        img_fns = img_names[idx:idx+batch_size]
        img_fps = [osp.join(img_dir, img_fn) for img_fn in img_fns]
        batch_preds = model_predict_images(model, img_fps, transforms)
        for i, pred in enumerate(batch_preds):
            preds[idx+i+1] = pred

    return preds


def load_image_batch_from_dir(root_dir, image_list, mean, std, image_size, crop_size):
    print("Loading images from {}".format(root_dir))
    test_dataset = FaceDataset(
        root_dir,
        image_list,
        mean,
        std,
        image_size=image_size,
        crop_size=crop_size,
        mode="predict",
        transform=True,
        predict_mode=True,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0
    )
    return test_dataloader


def warmup(model, tensor_shape=(224, 224)):
    input_tensor = torch.randn(1, 3, 224, 224).to(device)
    for _ in range(3):
        with torch.no_grad():
            model(input_tensor)
    print("Warmup done.")
    return model


def dict2csv(preds, csv_path, existing_csv_path, name2id_json_path):
    cols = [
        "race",
        "age",
        "emotion",
        "gender",
        "skintone",
        "masked"
    ]
    name2id = json.load(open(name2id_json_path, 'r'))
    df_existing = pd.read_csv(existing_csv_path)
    df_existing["face_file_name"] = [str(i+1) + ".jpg" for i in range(len(df_existing))]
    df_existing = df_existing.drop(columns=cols)
    df = pd.DataFrame.from_dict(preds, orient="index")
    df = df_existing.merge(df, left_on="face_file_name", right_on="face_file_name")
    df = df.drop(columns=["face_file_name"])
    df["image_id"] = df["file_name"].map(name2id)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str)
    parser.add_argument("--pth-path", type=str)        
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=224)    
    parser.add_argument("--image-dir", type=str)
    parser.add_argument("--csv", type=str, default="answer.csv")
    parser.add_argument("--existing-csv", type=str)
    parser.add_argument("--name2id-json", type=str)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-path", type=str)
    parser.add_argument("--root-dir", type=str)
    parser.add_argument("--image-list", type=str)
    args = parser.parse_args()

    model = torch2torchscript(args.ckpt_path, args.pth_path)

    # mean = [0.5, 0.5, 0.5]
    # std = [0.5, 0.5, 0.5]
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transforms = get_img_trans("test", args.image_size, args.crop_size, mean=mean, std=std)

    model = warmup(model)
    if args.image_path:
        pred = model_predict_images(model, [args.image_path], transforms)
        pp.pprint(pred)
    elif args.image_dir:
        preds = model_predict_img_dir(model, args.image_dir, transforms, args.batch_size)
        df = dict2csv(preds, args.csv, args.existing_csv, args.name2id_json)
        df.to_csv(args.csv, index=False)
        print("Predictions saved in {}".format(args.csv))        
    else:
        batchs = load_image_batch_from_dir(
            args.root_dir,
            args.image_list,
            args.mean,
            args.std,
            args.image_size
        )
        preds = model_predict_batchs(model, batchs)
        dict2csv(preds, args.csv_path)
        pp.pprint(preds)
