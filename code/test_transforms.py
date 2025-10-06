import torch
import os
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from utils.load_data import load_images
from monai.transforms import Compose
from utils.images_utils import (
    data_pre_transforms,
    data_augmentations,
    data_post_transforms_train,
)

N_REPETITIONS = 2
OUT_DIR = "transform_figures"

os.makedirs(OUT_DIR, exist_ok=True)

dataset_path = (
    "/home/catarina_caldeira/Imagens/SynthRAD2023dataset/Task1/pelvis"
)

train_val_dataset, _, train_val_patient_folders, _ = load_images(
    dataset_path, 23
)

combined_data = torch.utils.data.ConcatDataset(list(train_val_dataset.values()))


augmentation_args = {
    # MRI, CT and Mask
    "randaffined_prob": 1.0,
    "randflipd_prob": 1.0,
    "rand2delasticd_prob": 1.0,
    "randrotate90d_prob": 1.0,
    # Only applied to MRI
    "randgibbsnoised_prob": 1.0,
    "randkspacespikenoised_prob": 1.0,
    "randriciannoised_prob": 1.0,
    "randbiasfieldd_prob": 1.0,
    "randgaussiannoised_prob": 1.0,
}

pre_transforms = Compose(data_pre_transforms())
post_transforms = Compose(data_post_transforms_train())
transforms_train_da_crop = data_augmentations(**augmentation_args)

monai_train_dataset = combined_data

for i in tqdm(range(N_REPETITIONS)):
    for j, data in enumerate(monai_train_dataset):
        data = pre_transforms(data)
        augmentation_idx = random.randint(4, len(transforms_train_da_crop) - 1)
        data = transforms_train_da_crop[augmentation_idx](data)
        data = post_transforms(data)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(data["mr"][0].numpy(), cmap="gray")
        axes[0].set_title(f"MRI {augmentation_idx} {data["mr"].max().item()}")
        axes[0].axis("off")

        axes[1].imshow(data["ct"][0].numpy(), cmap="gray")
        axes[1].set_title("CT")
        axes[1].axis("off")

        axes[2].imshow(data["mask"][0].numpy(), cmap="gray")
        axes[2].set_title("Mask")
        axes[2].axis("off")

        plt.savefig(os.path.join(OUT_DIR, f"{i}_{j}.png"))
        plt.close()
