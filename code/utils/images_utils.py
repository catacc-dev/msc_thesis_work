from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from monai.transforms import (
    Compose,
    RandAffined,
    RandFlipd,
    Rand2DElasticd,
    RandGibbsNoised,
    RandKSpaceSpikeNoised,
    RandRicianNoised,
    RandBiasFieldd,
    RandGaussianNoised,
    RandSpatialCropd,
    SqueezeDimd,
    SpatialPadd,
    RandRotate90d,
    MaskIntensityd,
    ToTensord,
    MapTransform,
)

augmentation_args = {
    # MRI, CT and Mask
    "randaffined_prob": 0.5,
    "randflipd_prob": 0.5,
    "rand2delasticd_prob": 0.3,
    "randrotate90d_prob": 0.5,
    # Only applied to MRI
    "randgibbsnoised_prob": 0.5,
    "randkspacespikenoised_prob": 0.5,
    "randriciannoised_prob": 0.0,
    "randbiasfieldd_prob": 0.5,
    "randgaussiannoised_prob": 0.5,
}


class RemoveEmptySlices(MapTransform):
    def __init__(self, keys: list[str], reference_key: str):
        self.keys = keys
        self.reference_key = reference_key

    def __call__(self, data: dict) -> dict:
        empty_slices = (
            np.ptp(
                data[self.reference_key].reshape(
                    -1, data[self.reference_key].shape[-1]
                ),
                axis=0,
            )
            == 0
        )
        for key in self.keys:
            data[key] = data[key][..., ~empty_slices]
        return data


class KeepSlicesWithAtLeast(MapTransform):
    def __init__(self, keys: list[str], reference_key: str, pct: float):
        self.keys = keys
        self.reference_key = reference_key
        self.pct = pct

    def __call__(self, data: dict) -> dict:
        # data[self.reference_key].shape = [c, h, w, d]
        flat_array = data[self.reference_key].reshape(
            -1, data[self.reference_key].shape[-1]
        )
        # flat_array.shape = [chw, d]
        keep_slices = (flat_array.sum(axis=0) / flat_array.shape[0]) > self.pct
        # keep_slices.shape = [chw, d]
        for key in self.keys:
            data[key] = data[key][..., keep_slices]
        return data


class ScaleIntensityWithoutEmptySlicesd(MapTransform):
    def __init__(
        self,
        keys: list[str],
        reference_key: str
    ):
        
        self.keys = keys
        self.reference_key = reference_key
        
    def __call__(self, data: dict) -> dict:
        
        
        for key in self.keys:
            X_original = data[key]
            X_copy = X_original.clone()
            
            temp_data = {self.reference_key: X_copy, key: X_copy}
            
            temp_data = RemoveEmptySlices(
                keys=[key], 
                reference_key=self.reference_key
            )(temp_data)
            
            temp_data = KeepSlicesWithAtLeast(
                keys=[key],
                reference_key=self.reference_key,
                pct=0.1
            )(temp_data)
            
            X_filtered = temp_data[key]
            
            X_min = X_filtered.min()
            X_max = X_filtered.max()
            
            data[key] = 2 * ((X_original - X_min) / (X_max - X_min)) - 1
                
        return data
    
    
def from_normalize_to_HU(output: float):
    # From normalized generated sCT/CT to HU values
    b_min = -1
    b_max = 1
    a_min = -1024
    a_max = 3000
    output_hu = ((output - b_min) / (b_max - b_min)) * (a_max - a_min) + a_min
    return output_hu


def save_generated_images(
    input_mri,
    real_ct,
    output_sct,
    fold,
    dataset_name,
    result,
    patient_name,
    epoch,
    model_type,
    best_or_all,
    folder_name,
):

    # Saves a generated sample
    
    save_path = f"validations/{folder_name}/{dataset_name}/{model_type}/{result}/fold_{fold}"
    os.makedirs(save_path, exist_ok=True)

    real_ct = real_ct.cpu().numpy()
    output_sct = output_sct.cpu().numpy()

    output_hu_sct = from_normalize_to_HU(output_sct)
    output_hu_clip_sct = np.clip(
        output_hu_sct, -300, 300
    )  # clip between HU values to decrease the contrast

    output_hu_ct = from_normalize_to_HU(real_ct)
    output_hu_clip_ct = np.clip(
        output_hu_ct, -300, 300
    )  # clip between HU values to decrease the contrast

    num_slices = real_ct.shape[0]

    if input_mri is not None:
        input_mri = input_mri.cpu().numpy()
        num_cols = 3
    else:
        num_cols = 2

    fig, axes = plt.subplots(
        num_slices, num_cols, figsize=(8, num_cols * num_slices)
    )  
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    for slice_idx in range(num_slices):
        col = 0

        if input_mri is not None:
            axes[slice_idx, col].imshow(
                input_mri[slice_idx, 0, :, :], cmap="gray"
            )
            axes[slice_idx, col].axis("off")
            col += 1

        axes[slice_idx, col].imshow(
            output_hu_clip_ct[slice_idx, 0, :, :], cmap="gray"
        )
        axes[slice_idx, col].axis("off")
        col += 1

        axes[slice_idx, col].imshow(
            output_hu_clip_sct[slice_idx, 0, :, :], cmap="gray"
        )
        axes[slice_idx, col].axis("off")

    if best_or_all == "best":
        plt.savefig(
            f"{save_path}/fold{fold}_patient{patient_name}.png",
            dpi=100,
            bbox_inches="tight",
        )
    else:
        plt.savefig(
            f"{save_path}/fold{fold}_epoch{epoch}.png",
            dpi=100,
            bbox_inches="tight",
        )
    plt.close()


def save_generated_nifti_mha(
    evaluation_type,
    patient_mr_path,
    output_sct,
    fold,
    dataset_name,
    result,
    model_type,
    folder_name,
    patient_name,
    file_format,
):
    # Saves a generated image in Nifti or MHA from the validation/test set

    save_path = f"{evaluation_type}/{folder_name}/{dataset_name}/{model_type}/{result}/fold_{fold}"
    os.makedirs(save_path, exist_ok=True)

    mr_original_image = sitk.ReadImage(patient_mr_path)
    
    # 3D
    #output_sct_nifti = np.squeeze(output_sct, axis=0)  # 1,H,W,D
    #output_sct_nifti = np.squeeze(output_sct_nifti, axis=0)  # H,W,D
    
    # 2D
    output_sct_nifti = np.squeeze(output_sct, axis=1)  # D,H,W
    output_sct_nifti = np.transpose(
        output_sct_nifti, (0, 2, 1)
    )  # D,W,H -> z-y-x

    # 3D
    #output_sct_nifti = np.transpose(output_sct_nifti, (2, 1, 0))  # H, W, D → D, W, H
    
    sct_image = sitk.GetImageFromArray(output_sct_nifti)  # H,W,D -> x-y-z

    sct_image.CopyInformation(mr_original_image)

    output_filename = os.path.join(
        save_path, f"{patient_name}.{file_format}"
    )
    sitk.WriteImage(
        sct_image, output_filename
    )  # useCompression is False by default

def data_augmentations(
    randaffined_prob,
    randflipd_prob,
    rand2delasticd_prob,
    randgibbsnoised_prob,
    randkspacespikenoised_prob,
    randriciannoised_prob,
    randbiasfieldd_prob,
    randgaussiannoised_prob,
    randrotate90d_prob,
):
    return [  
        # Spatial - all
        RandAffined(
            keys=["mr", "ct", "mask"],
            rotate_range=np.pi / 36,  # randomly rotated within a range of -5º and +5º, along x
            translate_range=0,
            scale_range=0.05,  # x,y will be scaled by a random factor between -5% and +5% 
            prob=randaffined_prob,
            shear_range=0.3,  # shear transformation along x, y axes 
            padding_mode="zeros",  # any new voxels will be filled with zeros
            mode=["bilinear", "bilinear", "nearest"], # MRI, CT and Mask
        ),  
        # Spatial - all
        RandFlipd(
            keys=["mr", "ct", "mask"], prob=randflipd_prob, spatial_axis=1
        ),  # y axis
        # Spatial - all
        Rand2DElasticd(
            keys=["mr", "ct", "mask"],
            prob=rand2delasticd_prob,
            spacing=(20, 20),
            magnitude_range=(
                1,
                2,
            ),  # defines the strengh of the deformation field
            padding_mode="zeros",
            mode=["bilinear", "bilinear", "nearest"],
        ), 
        # Spatial - all
        RandRotate90d(
            keys=["mr", "ct", "mask"], prob=randrotate90d_prob
        ),  # randomly rotate the image in the (0,1) plane, i.e., around the width axis
        
        # Intensity - MRI
        # Mimic gibbs artifacts, caused by limited k-space sampling
        RandGibbsNoised(
            keys=["mr"], prob=randgibbsnoised_prob, alpha=(0.6, 0.8) # intensity of the gibbs noise filter - 1 value is randomly chosen from 0.6 to 0.8 - moderate to strong Gibbs artifact effect
        ), 
        # Intensity - MRI
        # Mimic spike artifacts: applies localized spikes in k-sapce
        RandKSpaceSpikeNoised(
            keys=["mr"],
            prob=randkspacespikenoised_prob,
            intensity_range=(13, 15), # range of intensity for the spike artifacts
        ),
        # Intensity - MRI
        # Adds rician noise to image - thermal noise in the scanner
        RandRicianNoised(
            keys=["mr"],
            prob=randriciannoised_prob,
            mean=0.0,
            std=0.02,
            sample_std=True,
        ),
        # Intensity - MRI
        # Mimics the perturbations in the magnetic field
        RandBiasFieldd(
            keys=["mr"],
            prob=randbiasfieldd_prob,
            coeff_range=(0.2, 0.3),  # range of random coefficients
        ),
        # Intensity - MRI
        # Applies gaussian noise to images
        RandGaussianNoised(
            keys=["mr"],
            prob=randgaussiannoised_prob,
            mean=0.0,
            std=0.1,
        ),
    ]


def data_pre_transforms_train():
    return [
        SpatialPadd(keys=["mr", "ct", "mask"], spatial_size=(256, 256, -1)),
        # maintain the original size: C,H,W,D=1
        RandSpatialCropd(
            keys=["mr", "ct", "mask"],
            roi_size=(-1, -1, 1),
            random_size=False,
            random_center=False 
        ),
        # took D - C,H,W - 2D image
        SqueezeDimd(keys=["mr", "ct", "mask"], dim=-1),
    ]


def data_pre_transforms_val():
    return [
        SpatialPadd(keys=["mr", "ct", "mask"], spatial_size=(256, 256, -1)),
    ]


def data_post_transforms_train():
    return [
        RandSpatialCropd(
            keys=["mr", "ct", "mask"],
            roi_size=(256, 256),  # train img
            random_size=False,
            random_center=False 
        ),
        MaskIntensityd(keys=["mr", "ct"], mask_key="mask"),
        ToTensord(keys=["mr", "ct"]),
    ]


def data_post_transforms_val():
    return [
        RandSpatialCropd(
            keys=["mr", "ct", "mask"],
            roi_size=(256, 256, -1),  # validation img
            random_size=False,
        ),
        MaskIntensityd(keys=["mr", "ct"], mask_key="mask"),
        ToTensord(keys=["mr", "ct"]),
    ]


def data_augmentation_transformations(
    randaffined_prob: float = 0,
    randflipd_prob: float = 0,
    rand2delasticd_prob: float = 0,
    randgibbsnoised_prob: float = 0,
    randkspacespikenoised_prob: float = 0,
    randriciannoised_prob: float = 0,
    randbiasfieldd_prob: float = 0,
    randgaussiannoised_prob: float = 0,
    randrotate90d_prob: float = 0,
    training: bool = False,
):
    # Compiles all data augmentation transformations to be applied on the training set
    
    if training:
        augmentations = data_augmentations(
            randaffined_prob,
            randflipd_prob,
            rand2delasticd_prob,
            randgibbsnoised_prob,
            randkspacespikenoised_prob,
            randriciannoised_prob,
            randbiasfieldd_prob,
            randgaussiannoised_prob,
            randrotate90d_prob,
        )
        transforms = Compose(
            [
                *data_pre_transforms_train(),
                *augmentations,
                *data_post_transforms_train(),
            ]
        )
    else:
        augmentations = []
        transforms = Compose(
            [
                *data_pre_transforms_val(),
                *augmentations,
                *data_post_transforms_val(),
            ]
        )
    return transforms
