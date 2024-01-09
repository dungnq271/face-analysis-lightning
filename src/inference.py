import argparse
import numpy as np
import pandas as pd
import torch
import os
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import Compose
import rootutils
import pprint

pp = pprint.PrettyPrinter(depth=4)
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.models.face_module import FaceLitModule
from src.data.components.face_dataset import FaceDataset
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pred2labels():
    race = {
        "0": "Caucasian",
        "1": "Mongoloid",
        "2": "Negroid",
    }
    gender = {"1": "Male", "0": "Female"}
    skintone = {"3": "mid-light", "1": "light", "2": "mid-dark", "0": "dark"}
    emotion = {
        "4": "Neutral",
        "3": "Happiness",
        "0": "Anger",
        "6": "Surprise",
        "2": "Fear",
        "5": "Sadness",
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
    model = FaceLitModule()
    checkpoint = torch.load(input_path)
    model.load_state_dict(checkpoint["state_dict"])
    print("Model saved in {}".format(output_path))
    script = model.to_torchscript()
    torch.jit.save(script, output_path)
    return model


def load_model(ckpt_path):
    checkpoint = torch.load(ckpt_path)
    hyper_parameters = checkpoint["hyper_parameters"]
    print("Hyper parameters keys: {}".format(hyper_parameters.keys()))
    model = FaceLitModule.load_from_checkpoint(ckpt_path)
    model.to(device)
    model.eval()
    return model


def posprocess_pred(pred):
    race, gender, age, skintone, emotion, masked = pred2labels()
    attrs_dict = {
        "race": race,
        "gender": gender,
        "age": age,
        "skintone": skintone,
        "emotion": emotion,
        "masked": masked,
    }
    for key in ["race", "gender", "skintone", "emotion", "masked"]:
        pred[key] = pred[key].cpu().numpy()
        pred[key] = np.argmax(pred[key], axis=1)
        pred[key] = attrs_dict[key][str(pred[key][0])]
    pred["age"] = pred["age"].cpu().numpy()
    pred["age"] = np.round(pred["age"])
    pred["age"] = np.sum(pred["age"], axis=1).astype(int)
    pred["age"] = attrs_dict["age"][str(pred["age"][0])]
    return pred


def model_predict_batchs(model, batchs):
    model.freeze()
    preds = {}
    for i, (input_tensor, img_name) in enumerate(batchs):
        input_tensor = input_tensor.to(device)
        with torch.no_grad():
            pred = model(input_tensor)
        pred = posprocess_pred(pred)
        pred["file_name"] = img_name[0]
        preds[img_name[0]] = pred
        if i == 9:
            break
    return preds


def model_predict_image(model, image_path):
    model.freeze()
    img = Image.open(image_path)
    img = img.convert("RGB")
    transform = Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    img = transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    with torch.no_grad():
        pred = model(img)
    pred = posprocess_pred(pred)
    return pred


def load_image_batch_from_dir(root_dir, image_list, mean, std, image_size):
    print("Loading images from {}".format(root_dir))
    test_dataset = FaceDataset(
        root_dir,
        image_list,
        mean,
        std,
        image_size,
        crop_size=image_size,
        mode="predict",
        transform=True,
        predict_mode=True,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0
    )
    return test_dataloader


def warmup(model):
    input_tensor = torch.randn(1, 3, 224, 224).to(device)
    model.freeze()
    for _ in range(3):
        with torch.no_grad():
            model(input_tensor)
    print("Warmup done.")
    return model


def dict2csv(preds, csv_path):
    df = pd.DataFrame.from_dict(preds, orient="index")
    df = df[["file_name",
             "race",
             "age",
             "emotion",
             "gender",
             "skintone",
             "masked"]]
    df.to_csv(csv_path, index=False)
    print("Predictions saved in {}".format(csv_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--image_list", type=str)
    parser.add_argument("--image_path", type=str)
    args = parser.parse_args()
    ckpt_path = args.ckpt_path
    root_dir = args.root_dir
    image_list = args.image_list
    image_path = args.image_path
    model = load_model(ckpt_path)
    warmup(model)
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    image_size = 224
    if image_path:
        pred = model_predict_image(model, image_path)
        pp.pprint(pred)
    else:
        batchs = load_image_batch_from_dir(root_dir, image_list, mean, std, image_size)
        preds = model_predict_batchs(model, batchs)
        dict2csv(preds, "outputs/answer.csv")
        pp.pprint(preds)
