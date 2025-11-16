import torch
import os
import matplotlib.pyplot as plt
from utils.load_data import load_images
from monai.transforms import Compose
from utils.images_utils import (
    data_pre_transforms_train,
    data_augmentations,
    data_post_transforms_train,
)
from monai.utils import set_determinism

OUT_DIR = "/home/catarina_caldeira/Desktop/code/utils/images_thesis/data_agumentations"
os.makedirs(OUT_DIR, exist_ok=True)

set_determinism(42)

# Dataset loading
dataset_path = "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1/"
train_val_dataset, _, train_val_patient_folders, _ = load_images(dataset_path, 25, False)

# Get AB patients only
a_keys = [k for k in train_val_dataset if k[1] == "AB"]
a_data = [train_val_dataset[k] for k in a_keys]
combined_data = torch.utils.data.ConcatDataset(a_data)

# Name mapping for titles
aug_name_mapping = {
    2: "2D Elastic",
    4: "Gibbs noise",
    5: "K-space spike noise", 
    7: "Bias field",
    8: "Gaussian noise"
}

# Initialize transforms
pre_transforms = Compose(data_pre_transforms_train())
post_transforms = Compose(data_post_transforms_train())
#transforms_train_da_crop = data_augmentations(**augmentation_args)

# Process one patient
patient_data = combined_data[9] 
patient_data = pre_transforms(patient_data)

# After pre transforms - original
after_pre_data = {
        "mr": patient_data["mr"].clone(),
        "ct": patient_data["ct"].clone(),  
        "mask": patient_data["mask"].clone()
    }
after_post_data = post_transforms(after_pre_data)
after_post_mr = after_post_data["mr"][0].numpy()
    
# Prepare augmented images
augmentations_to_show = [2, 4, 5, 7, 8]  
augmented_images = []

for key in augmentations_to_show:
    args = {
        "randaffined_prob": 0.0,
        "randflipd_prob": 0.0,
        "rand2delasticd_prob": 0.0,
        "randrotate90d_prob": 0.0,
        "randgibbsnoised_prob": 0.0,
        "randkspacespikenoised_prob": 0.0,
        "randriciannoised_prob": 0.0,
        "randbiasfieldd_prob": 0.0,
        "randgaussiannoised_prob": 0.0,
    }
    
    # Map the index to the corresponding dictionary key
    if key == 2:
        args["rand2delasticd_prob"] = 1.0
    elif key == 4:
        args["randgibbsnoised_prob"] = 1.0
    elif key == 5:
        args["randkspacespikenoised_prob"] = 1.0
    elif key == 7:
        args["randbiasfieldd_prob"] = 1.0
    elif key == 8:
        args["randgaussiannoised_prob"] = 1.0
    
    #augmented_images.append(args)
    
    transforms_train_da_crop = Compose(data_augmentations(**args))
    
    augmented_data = transforms_train_da_crop(patient_data)
    augmented_data = post_transforms({
        "mr": augmented_data["mr"],
        "ct": patient_data["ct"], 
        "mask": patient_data["mask"] 
    })
    
    augmented_images.append(augmented_data["mr"][0].numpy())
    
    

fig, axes = plt.subplots(5,2, figsize=(8,16))

for i, (aug_idx, img) in enumerate(zip(augmentations_to_show, augmented_images)):
    # Original (Top)
    axes[i,0].imshow(after_post_data["mr"][0].numpy(), cmap="gray")
    axes[i,0].set_title(f"MRI ground truth", fontsize=18)
    axes[i,0].axis("off")
    
    # Augmented (Bottom)
    axes[i,1].imshow(img, cmap="gray")
    axes[i,1].set_title(aug_name_mapping[aug_idx], fontsize=18)
    axes[i,1].axis("off")

plt.tight_layout()
plt.subplots_adjust(wspace=0.02)  # Reduce horizontal spacing between columns
plt.savefig(os.path.join(OUT_DIR, "augmentation_comparison_grid.png"), dpi=200)
plt.close()

