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

from tqdm import tqdm

pp = pprint.PrettyPrinter(depth=4)
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.models.face_custom_module import FaceCustomLitModule
from src.models.components.age_classifier import AgeClassifier
from src.models.components.gender_classifier import GenderClassifier
from src.models.components.backbone_maker import BackboneMaker
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
    model = FaceCustomLitModule()
    checkpoint = torch.load(input_path)
    model.load_state_dict(checkpoint["state_dict"])
    print("Model saved in {}".format(output_path))
    script = model.to_torchscript()
    torch.jit.save(script, output_path)
    return model


def load_model(ckpt_path):
    checkpoint = torch.load(ckpt_path)
    state_dict = checkpoint["state_dict"].keys()
    hyper_parameters = checkpoint["hyper_parameters"]
    # print("Hyper parameters keys: {}".format(hyper_parameters.keys()))
    # pp.pprint(state_dict)
    # pp.pprint(hyper_parameters)
    head = AgeClassifier()
    head.load_state_dict(checkpoint["state_dict"])
    model = FaceCustomLitModule.load_from_checkpoint(ckpt_path)
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
    for i, (input_tensor, img_name) in tqdm(enumerate(batchs)):
        input_tensor = input_tensor.to(device)
        with torch.no_grad():
            pred = model(input_tensor)
        pred = posprocess_pred(pred)
        pred["file_name"] = img_name[0]
        preds[img_name[0]] = pred
    return preds


def model_predict_image(model, image_path):
    model.freeze()
    image_name = os.path.basename(image_path)
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
    pred["file_name"] = image_name
    return pred


def model_predict_img_dir(model, img_dir):
    model.freeze()
    preds = {}
    for img_name in tqdm(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, img_name)
        pred = model_predict_image(model, img_path)
        preds[img_name] = pred
    return preds


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
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--csv_path", type=str)
    args = parser.parse_args()
    ckpt_path = args.ckpt_path
    root_dir = args.root_dir
    image_list = args.image_list
    image_path = args.image_path
    image_dir = args.image_dir
    csv_path = args.csv_path
    model = load_model(ckpt_path)
    # warmup(model)
    # mean = [0.5, 0.5, 0.5]
    # std = [0.5, 0.5, 0.5]
    # image_size = 224
    # if image_path:
    #     pred = model_predict_image(model, image_path)
    #     pp.pprint(pred)
    # elif image_dir:
    #     preds = model_predict_img_dir(model, image_dir)
    #     dict2csv(preds, csv_path)
    # else:
    #     batchs = load_image_batch_from_dir(root_dir,
    #                                        image_list,
    #                                        mean, std,
    #                                        image_size
    #                                        )
    #     preds = model_predict_batchs(model, batchs)
    #     dict2csv(preds, csv_path)
    #     pp.pprint(preds)
