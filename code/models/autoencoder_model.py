import click
import os
import numpy as np
import time

# Loading data and initial transformations
from utils.load_data import load_images

# Training and Validation per epoch
from trains.train import train_per_epoch_gen
from validations.validation import validation_per_epoch_gen

# Generator architecture model
from models.generator_monai import generator_unet

# Plotting results and Data Augmentation
from utils.plot_utils import plot_losses_generator, plot_mean_metrics
from utils.images_utils import data_augmentation_transformations

# K-fold cross-valdation
from sklearn.model_selection import StratifiedKFold, KFold

from utils.images_utils import (
    data_augmentation_transformations,
    augmentation_args,
)

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Subset

# Pytorch Lightning
from torchmetrics.regression import MeanAbsoluteError
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    MultiScaleStructuralSimilarityIndexMeasure,
)

# MONAI
from monai.utils import set_determinism
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    RandSpatialCropd,
    SpatialPadd,
    ScaleIntensityd,
    MaskIntensityd,
    ScaleIntensityRanged,
    ToTensord,
)
from monai.networks.layers.factories import Norm


def parse_list(ctx, param, value):
    return list(map(int, value.split(",")))


@click.command()
@click.option(
    "--dataset_path",
    type=str,
    default="/home/catarina_caldeira/Imagens/SynthRAD2023dataset/Task1/pelvis",
    help="Path that has the dataset",
)
@click.option(
    "--epoch_start", type=int, default=0, help="Epoch to start training from"
)  # change
@click.option(
    "--n_epochs", type=int, default=99, help="Number of epochs of training"
)  # change
@click.option(
    "--dataset_name",
    type=str,
    default="SynthRAD2023",
    help="Name of the dataset",
)
@click.option(
    "--batch_size_train",
    type=int,
    default=32,
    help="Size of the batches for training",
)
@click.option(
    "--batch_size_val",
    type=int,
    default=1,
    help="Size of the batches for validation",
)
@click.option(
    "--lr", type=float, default=0.0005, help="Learning rate for Adam optimizer"
)
@click.option(
    "--b1",
    type=float,
    default=0.9,
    help="Adam: decay of first order momentum of gradient",
)
@click.option(
    "--b2",
    type=float,
    default=0.999,
    help="Adam: decay of first order momentum of gradient",
)
@click.option(
    "--result",
    type=str,
    default="results_lr",
    help="Name of the folder to save results",
)  # change
@click.option(
    "--subsample_rate",
    type=int,
    default=2,
    help="Subsample rate for validation image slices",
)
@click.option(
    "--checkpoint_interval",
    default=100,
    help="Interval between model checkpoints",
)
@click.option(
    "--channels",
    type=str,
    default="64, 128, 256, 512, 512",
    callback=parse_list,
    help="channels of the generator",
)
@click.option(
    "--strides",
    type=str,
    default="2, 2, 2, 2",
    callback=parse_list,
    help="strides of the generator",
)
@click.option(
    "--num_res_units",
    type=int,
    default=0,
    help="num of residual units per layer",
)
@click.option(
    "--type_norm",
    type=str,
    default="INSTANCE",
    help='Choose between "INSTANCE" or "BATCH"',
)
@click.option(
    "--choosen_region",
    type=str,
    default="AB",
    help='Choose between "AB" or "HN" OR "TH"',
)


def main(
    epoch_start,
    n_epochs,
    dataset_name,
    batch_size_train,
    batch_size_val,
    lr,
    b1,
    b2,
    result,
    dataset_path,
    subsample_rate,
    checkpoint_interval,
    channels,
    strides,
    num_res_units,
    type_norm,
    choosen_region
):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    start_time = time.time()

    mae = MeanAbsoluteError().to(device)
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure().to(device)

    criterion = nn.L1Loss(reduction="none").to(
        device
    )  # L1 (MAE) loss function - returns loss per pixel instead of single scalar

    set_determinism(seed=42)

    # SynthRAD2023
    '''
    train_val_dataset, _, train_val_patient_folders, _ = load_images(
        dataset_path, 23, False
    )
    print("Train/val dataset contents:", train_val_dataset)
    print("Number of datasets:", len(train_val_dataset.values()))
    # print("train_val_dataset:", train_val_dataset)
    # print("Keys:", train_val_dataset.keys())

    combined_data = torch.utils.data.ConcatDataset(
        list(train_val_dataset.values())
    )  # pelvis data from center A and C
    
    all_patients_path = torch.utils.data.ConcatDataset(
        list(train_val_patient_folders.values())
    )

    a_keys = list(filter(lambda k: k[0] == "A", train_val_dataset))
    c_keys = list(filter(lambda k: k[0] == "C", train_val_dataset))

    y_centers = np.concatenate(
        [
            np.zeros(sum(len(train_val_dataset[k]) for k in a_keys)),
            np.ones(sum(len(train_val_dataset[k]) for k in c_keys)),
        ]
    )

    print("Total A samples:", sum(len(train_val_dataset[k]) for k in a_keys))
    print("Total C samples:", sum(len(train_val_dataset[k]) for k in c_keys))
    '''
    
    # SynthRAD2025
    
     # A+B+C 
    '''
    train_val_dataset, _, train_val_patient_folders, _ = load_images(
        dataset_path, 25, False
    )
    combined_data = torch.utils.data.ConcatDataset(list(train_val_dataset.values()))  # A+B+C 
    all_patients_path = torch.utils.data.ConcatDataset(
        list(train_val_patient_folders.values())
    )
    
    a_keys = list(filter(lambda k: k[0] == "A", train_val_dataset))
    b_keys = list(filter(lambda k: k[0] == "B", train_val_dataset))
    c_keys = list(filter(lambda k: k[0] == "C", train_val_dataset))
    
    y_centers = np.concatenate([
        np.zeros(sum(len(train_val_dataset[k]) for k in a_keys)), # A
        np.ones(sum(len(train_val_dataset[k]) for k in b_keys)), # B
        np.full(sum(len(train_val_dataset[k]) for k in c_keys), 2) # C
    ])
    print("Total A samples:", sum(len(train_val_dataset[k]) for k in a_keys))
    print("Total B samples:", sum(len(train_val_dataset[k]) for k in b_keys))
    print("Total C samples:", sum(len(train_val_dataset[k]) for k in c_keys))
    '''
    
    # AB+TH+HN
    '''
    train_val_dataset, _, train_val_patient_folders, _ = load_images(
        dataset_path, 25, False
    )
    combined_data = torch.utils.data.ConcatDataset([
    dataset for (center, region), dataset in train_val_dataset.items()
    if region in ["AB", "HN", "TH"]
    ])
    print(len(combined_data))
    
    a_keys = list(filter(lambda k: k[1] == "AB", train_val_dataset))
    b_keys = list(filter(lambda k: k[1] == "TH", train_val_dataset))
    c_keys = list(filter(lambda k: k[1] == "HN", train_val_dataset))
    
    y_centers = np.concatenate([
        np.zeros(sum(len(train_val_dataset[k]) for k in a_keys)), # AB
        np.ones(sum(len(train_val_dataset[k]) for k in b_keys)), # TH
        np.full(sum(len(train_val_dataset[k]) for k in c_keys), 2) # HN
    ])
    print("Total A samples:", sum(len(train_val_dataset[k]) for k in a_keys))
    print("Total B samples:", sum(len(train_val_dataset[k]) for k in b_keys))
    print("Total C samples:", sum(len(train_val_dataset[k]) for k in c_keys))
 
    all_patients_path = torch.utils.data.ConcatDataset([
        dataset for (center, region), dataset in train_val_patient_folders.items()
        if region in ["AB", "HN", "TH"]
        ])
    print(all_patients_path)
    '''
    
    # AB/TH/HN
    train_val_dataset, _, train_val_patient_folders, _ = load_images(
        dataset_path, 25, False
    )
    combined_data = torch.utils.data.ConcatDataset([
    dataset for (center, region), dataset in train_val_dataset.items()
    if region in [choosen_region]
    ])
    print(combined_data)
    
    a_keys = list(filter(lambda k: k[1] == choosen_region, train_val_dataset))
    
    print("Total samples:", sum(len(train_val_dataset[k]) for k in a_keys))
 
    all_patients_path = torch.utils.data.ConcatDataset([
        dataset for (center, region), dataset in train_val_patient_folders.items()
        if region in [choosen_region]
        ])
    #print(all_patients_path)
    
    # Dictionary that saves all folds losses per epoch
    fold_train_losses = {}
    fold_val_losses = {}

    # Dictionary for all folds metrics
    fold_psnr_values = {}
    fold_mae_values = {}
    fold_ssim_values = {}
    fold_ms_ssim_values = {}

    # List that saves all folds losses for all epochs
    all_fold_train_losses = []
    all_fold_val_losses = []
    all_fold_val_psnr = []
    all_fold_val_mae = []
    all_fold_val_ssim = []
    all_fold_val_ms_ssim = []

    # List that saves mean loss value
    fold_loss = []

    splits = 5

    # Set up k-fold cross-validation
    kfold = KFold(n_splits=splits, shuffle=True, random_state=42)

    for fold, (train_index, val_index) in enumerate(
        kfold.split(combined_data)
    ):
        print(
            f"Fold:{fold},train indexes:{train_index}, val indexes:{val_index}"
        )
        
        train_set = Subset(combined_data, train_index)
        valid_set = Subset(combined_data, val_index)
        
    # Set up Stratified k-fold cross-validation
    #skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)

    #for fold, (train_index, val_index) in enumerate(
    #    skf.split(combined_data, y_centers)
    #):
        if type_norm == "INSTANCE":
            model = generator_unet(channels, strides, type_norm, num_res_units)
        elif type_norm == "BATCH":
            type_norm = Norm.BATCH
            model = generator_unet(channels, strides, type_norm, num_res_units)

        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(b1, b2))
        '''
        print(f"Fold {fold}:")
        print(f"  Train set size: {len(train_index)}")
        print(
            f"  Class distribution in training set (A,C): {np.bincount(y_centers[train_index].astype(int))}"
        )
        print(f"  Validation set size: {len(val_index)}")
        print(
            f"  Class distribution in validation set (A,C): {np.bincount(y_centers[val_index].astype(int))}"
        )
        '''

        # For saving the best validation model
        os.makedirs(
            f"validations/saved_models/{dataset_name}/generator/{result}/fold_{fold}/",
            exist_ok=True,
        )

        # Initialize lists for fold losses
        fold_train_losses[fold] = []
        fold_val_losses[fold] = []
        fold_psnr_values[fold] = []
        fold_mae_values[fold] = []
        fold_ssim_values[fold] = []
        fold_ms_ssim_values[fold] = []

        # Split the dataset
        train_set = Subset(combined_data, train_index)
        valid_set = Subset(combined_data, val_index)

        train_patients_path = [
            all_patients_path[i] for i in train_index.tolist()
        ]
        print(f"Train patients: {train_patients_path}")

        val_patients_path = [all_patients_path[i] for i in val_index.tolist()]
        print(f"Validation patients: {val_patients_path}")

        # Data augmentation and MRI normalization of intensities

        transforms_train_da_crop = data_augmentation_transformations(
            **augmentation_args, training=True
        )

        monai_train_dataset = Dataset(
            data=train_set, transform=transforms_train_da_crop
        )
        
        # per slice
        transforms_val = data_augmentation_transformations(training=False)

        monai_valid_dataset = Dataset(data=valid_set, transform=transforms_val)

        train_loader = DataLoader(
            monai_train_dataset,
            batch_size=batch_size_train,
            shuffle=True,
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
        )

        valid_loader = DataLoader(
            monai_valid_dataset,
            batch_size=batch_size_val,
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
        )

        # Initialize best values
        best_epoch_val = -1
        best_global_models = (
            {}
        )  # saves the best model from epoch 0 until each checkpoint interval (0-99, 0-199, 0-299...)
        best_global_loss = float("inf")

        for epoch in range(epoch_start, n_epochs):

            # -> Training phase
            train_loss = train_per_epoch_gen(
                criterion,
                device,
                model,
                train_loader,
                optimizer,
                fold,
                epoch,
                n_epochs,
            )

            # Compute average training loss
            train_loss_avg = train_loss / len(train_loader)

            # Save training losses per fold
            fold_train_losses[fold].append(train_loss_avg)

            # -> Validation phase
            (
                val_loss,
                all_psnr_batches,
                all_mae_batches,
                all_ssim_batches,
                all_ms_ssim_batches,
            ) = validation_per_epoch_gen(
                device,
                model,
                subsample_rate,
                criterion,
                valid_loader,
                fold,
                epoch,
                n_epochs,
                dataset_name,
                result,
                psnr,
                mae,
                ssim,
                ms_ssim,
            )

            # Compute average validation loss
            val_loss_avg = val_loss / len(valid_loader)

            # Save validation losses per fold
            fold_val_losses[fold].append(val_loss_avg)
            fold_psnr_values[fold].append(
                np.mean(all_psnr_batches)
            )  # saving all mean values for 34 batches for each epoch in each fold
            fold_mae_values[fold].append(np.mean(all_mae_batches))
            fold_ssim_values[fold].append(np.mean(all_ssim_batches))
            fold_ms_ssim_values[fold].append(np.mean(all_ms_ssim_batches))

            # Track the best validation model based on mean loss per epoch
            if val_loss_avg < best_global_loss:
                best_global_loss = val_loss_avg
                best_epoch_val = epoch
                best_fold_val = fold
                best_global_models = {"model_state_dict": model.state_dict()}

            # Checkopint to save models in a interval of 100 epochs
            if (epoch + 1) % checkpoint_interval == 0 or epoch == n_epochs - 1:
                torch.save(
                    {
                        "model_state_dict": best_global_models[
                            "model_state_dict"
                        ]
                    },
                    f"validations/saved_models/{dataset_name}/generator/{result}/fold_{best_fold_val}/val_bestmodel_fold{best_fold_val}_epoch{best_epoch_val}_interval_0-{epoch}.pth",
                )

                torch.save(
                    {"model_state_dict": model.state_dict()},
                    f"validations/saved_models/{dataset_name}/generator/{result}/fold_{fold}/val_checkpoints_fold{fold}_epoch{epoch}.pth",
                )

        all_fold_train_losses.append(fold_train_losses[fold])
        all_fold_val_losses.append(fold_val_losses[fold])
        all_fold_val_psnr.append(fold_psnr_values[fold])
        all_fold_val_mae.append(fold_mae_values[fold])
        all_fold_val_ssim.append(fold_ssim_values[fold])
        all_fold_val_ms_ssim.append(fold_ms_ssim_values[fold])

        # Mean of validation losses and metrics for each fold
        fold_loss.append(np.mean([t for t in fold_val_losses[fold]]))
        print(f"Mean loss for this fold: {fold_loss[fold]}")

        # training_progress_gif(output_folder)

    # Calculate and print the mean and std for all folds
    mean_loss = np.mean(fold_loss)
    std_loss = np.std(fold_loss, ddof=1)

    # Print final results after cross-validation
    print("\nFinal Result after Cross-Validation:")
    print(f"Mean loss: {mean_loss:.4f}±{std_loss:.4f}")

    output_folder = f"../plots/{dataset_name}/generator/{result}"
    os.makedirs(output_folder, exist_ok=True)

    for i in range(splits):

        plot_losses_generator(
            all_fold_train_losses[i],
            all_fold_val_losses[i],
            os.path.join(
                output_folder, f"train_vs_validation_losses_fold{i}.png"
            ),
        )

        plot_mean_metrics(
            all_fold_val_ssim[i],
            os.path.join(output_folder, f"fold{i}_SSIM.png"),
            "SSIM",
            "Mean validation SSIM across Epochs",
        )

        plot_mean_metrics(
            all_fold_val_ms_ssim[i],
            os.path.join(output_folder, f"fold{i}_MS-SSIM.png"),
            "MS-SSIM",
            "Mean validation MS-SSIM across Epochs",
        )

    end_time = time.time()
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    print(f"Total time: {hours}h {minutes}m {seconds}s")

    pass


if __name__ == "__main__":
    main()

# python -m  models.autoencoder_model
