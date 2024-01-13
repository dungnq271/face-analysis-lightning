import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor


def get_img_trans(
    phase,
    image_size=256,
    crop_size=224,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    normalize = A.Normalize(mean=mean, std=std)
    if phase == "train":
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.RandomCrop(crop_size, crop_size),
                A.HorizontalFlip(),
                normalize,
                ToTensor(),
            ]
        )
    elif phase in ["test", "val"]:
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.CenterCrop(crop_size, crop_size),
                normalize,
                ToTensor(),
            ]
        )
    else:
        raise KeyError
