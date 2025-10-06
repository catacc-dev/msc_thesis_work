import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from mpl_toolkits.axes_grid1 import make_axes_locatable


# 3D
import json

path = "/home/catarina_caldeira/Desktop/code/metrics_results_25_docker_gaussianoverlap.json"
with open(path, 'r') as f:
    data=json.load(f)

patients = list(data.keys())

# Extract metrics in the same order as patients
psnr_values = [data[p]["psnr"] for p in patients]
mae_values = [data[p]["mae"] for p in patients]
ssim_values = [data[p]["ssim"] for p in patients]
msssim_values = [data[p]["ms_ssim"] for p in patients]

ab_psnr, ab_mae, ab_ssim, ab_msssim = [], [], [], []
th_psnr, th_mae, th_ssim, th_msssim = [], [], [], []
hn_psnr, hn_mae, hn_ssim, hn_msssim = [], [], [], []

for patient_id, metrics in data.items():
    region = patient_id[1:3]  # Adjust this if region code is different

    if region == "AB":
        ab_psnr.append(metrics["psnr"])
        ab_mae.append(metrics["mae"])
        ab_ssim.append(metrics["ssim"])
        ab_msssim.append(metrics["ms_ssim"])

    elif region == "TH":  # Assuming "THE" is shortened in patient_id
        th_psnr.append(metrics["psnr"])
        th_mae.append(metrics["mae"])
        th_ssim.append(metrics["ssim"])
        th_msssim.append(metrics["ms_ssim"])

    elif region == "HN":
        hn_psnr.append(metrics["psnr"])
        hn_mae.append(metrics["mae"])
        hn_ssim.append(metrics["ssim"])
        hn_msssim.append(metrics["ms_ssim"])

def print_stats(region, psnr, mae, ssim, msssim):
    print(f"\nRegion: {region}")
    print(f"  PSNR -> Mean: {np.mean(psnr):.4f},  Std: {np.std(psnr):.4f}")
    print(f"  MAE -> Mean: {np.mean(mae):.4f},   Std: {np.std(mae):.4f}")
    print(f"  SSIM -> Mean: {np.mean(ssim):.4f},  Std: {np.std(ssim):.4f}")
    print(f"  MS-SSIM -> Mean: {np.mean(msssim):.4f}, Std: {np.std(msssim):.4f}")

print_stats("AB", ab_psnr, ab_mae, ab_ssim, ab_msssim)
print_stats("THE", th_psnr, th_mae, th_ssim, th_msssim)
print_stats("HN", hn_psnr, hn_mae, hn_ssim, hn_msssim)


# --- Ranking Utility ---
def get_ranks(values, reverse=False):
    sorted_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=reverse)
    ranks = [0] * len(values)
    for rank, idx in enumerate(sorted_indices, start=1):
        ranks[idx] = rank
    return ranks

# Compute ranks
ranks_psnr = get_ranks(psnr_values, reverse=True)
ranks_mae = get_ranks(mae_values, reverse=False)
ranks_ssim = get_ranks(ssim_values, reverse=True)
ranks_msssim = get_ranks(msssim_values, reverse=True)

# Mean ranks per patient
mean_ranks = [np.mean([ranks_psnr[i], ranks_mae[i], ranks_ssim[i], ranks_msssim[i]]) for i in range(len(patients))]

# Organize by region
regions = {'AB': {'patients': [], 'mean_ranks': []}, 'HN': {'patients': [], 'mean_ranks': []}, 'TH': {'patients': [], 'mean_ranks': []}}

for patient, rank in zip(patients, mean_ranks):
    region = patient[1:3]
    if region in regions:
        regions[region]['patients'].append(patient)
        regions[region]['mean_ranks'].append(rank)

region_extremes = {}

for region, data in regions.items():
    if not data['mean_ranks']:
        continue
    worst_idx = np.argmax(data['mean_ranks'])
    best_idx = np.argmin(data['mean_ranks'])
    region_extremes[region] = {
        'worst': {'patient': data['patients'][worst_idx], 'rank': data['mean_ranks'][worst_idx]},
        'best': {'patient': data['patients'][best_idx], 'rank': data['mean_ranks'][best_idx]},
    }
    print(f"\n{region} Region:")
    print(f"  Worst: {data['patients'][worst_idx]} (rank: {data['mean_ranks'][worst_idx]:.2f})")
    print(f"  Best : {data['patients'][best_idx]} (rank: {data['mean_ranks'][best_idx]:.2f})")


def get_max_shape(image_list):
    max_h = max(img.shape[0] for img in image_list)
    max_w = max(img.shape[1] for img in image_list)
    return max_h, max_w

def pad_image_to_shape(img, target_shape, pad_value=-1024):
    h, w = img.shape
    target_h, target_w = target_shape
    pad_h = max(target_h - h, 0)
    pad_w = max(target_w - w, 0)
    
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    padded_img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)),
                        mode='constant', constant_values=pad_value)
    return padded_img
  
# --- Image Visualization ---
def difference_maps(patient_id_ab, patient_id_hn, patient_id_th, save_path, directory, directory_synthetic_ct):
    patients = [patient_id_ab, patient_id_hn, patient_id_th]
    titles = ['AB', 'HN', 'TH']
    slices = [22, 33]

    # Step 1: Collect all slices (un-padded) and their info
    slices_to_pad = []
    slice_info = []  # (patient_idx, slice_idx, type: 'mr'/'ct'/'sct'/'diff')
    masked_data = {}  # Store masked arrays for each patient

    for patient_idx, patient_id in enumerate(patients):
        found = False
        for region_folder in os.listdir(directory):
            region_folder_path = os.path.join(directory, region_folder)
            if not os.path.isdir(region_folder_path):
                continue
            if patient_id not in os.listdir(region_folder_path):
                continue

            patient_path = os.path.join(region_folder_path, patient_id)
            path_ct = os.path.join(patient_path, "ct.mha")
            path_mr = os.path.join(patient_path, "mr.mha")
            path_mask = os.path.join(patient_path, "mask.mha")
            path_sct = os.path.join(directory_synthetic_ct, f"{patient_id}.mha")

            sct = sitk.GetArrayFromImage(sitk.ReadImage(path_sct))
            ct = sitk.GetArrayFromImage(sitk.ReadImage(path_ct))
            mr = sitk.GetArrayFromImage(sitk.ReadImage(path_mr))
            mask = sitk.GetArrayFromImage(sitk.ReadImage(path_mask))
            masked_ct = np.where(mask == 1, ct, -1024)
            masked_sct = np.where(mask == 1, sct, -1024)
            masked_mr = mr * mask

            # Store for later use
            masked_data[patient_id] = {
                'masked_ct': masked_ct,
                'masked_sct': masked_sct,
                'masked_mr': masked_mr,
                'mask': mask,
                'ct': ct,
                'sct': sct
            }

            for slice_idx in slices:
                slices_to_pad.append(masked_mr[slice_idx])
                slice_info.append((patient_idx, slice_idx, 'mr'))
                slices_to_pad.append(masked_ct[slice_idx])
                slice_info.append((patient_idx, slice_idx, 'ct'))
                slices_to_pad.append(masked_sct[slice_idx])
                slice_info.append((patient_idx, slice_idx, 'sct'))
                # diff will be calculated after padding
                slices_to_pad.append(masked_sct[slice_idx])  # placeholder for diff, will be recalculated
                slice_info.append((patient_idx, slice_idx, 'diff'))

            found = True
            break
        if not found:
            print(f"Patient {patient_id} not found in dataset.")

    # Step 2: Compute max shape
    max_h, max_w = get_max_shape(slices_to_pad)

    # Step 3: Pad all slices
    padded_slices = []
    for idx, (patient_idx, slice_idx, img_type) in enumerate(slice_info):
        patient_id = patients[patient_idx]
        data = masked_data[patient_id]
        if img_type == 'mr':
            padded_slices.append(pad_image_to_shape(data['masked_mr'][slice_idx], (max_h, max_w),pad_value=-1))
        elif img_type == 'ct':
            padded_slices.append(pad_image_to_shape(data['masked_ct'][slice_idx], (max_h, max_w)))
        elif img_type == 'sct':
            padded_slices.append(pad_image_to_shape(data['masked_sct'][slice_idx], (max_h, max_w)))
        elif img_type == 'diff':
            # Pad sct and ct, then compute diff
            sct_slice = pad_image_to_shape(data['masked_sct'][slice_idx], (max_h, max_w))
            ct_slice = pad_image_to_shape(data['ct'][slice_idx], (max_h, max_w))
            mask_slice = pad_image_to_shape(data['mask'][slice_idx], (max_h, max_w), pad_value=0)
            diff_slice = np.where(mask_slice == 1, sct_slice - ct_slice, 0)
            padded_slices.append(diff_slice)

    # Step 4: Plotting
    fig, axes = plt.subplots(6, 4, figsize=(20, 25))
    plt.subplots_adjust(right=0.88)
    col_titles = ["MRI (masked)", "CT (masked)", "sCT", "sCT - CT (masked)"]
    for col in range(4):
        axes[0, col].set_title(col_titles[col], fontsize=30)

    row_idx = 0
    for patient_idx, patient_id in enumerate(patients):
        for slice_num, slice_idx in enumerate(slices):
            base = (patient_idx * len(slices) + slice_num) * 4
            mr_slice = padded_slices[base + 0]
            ct_slice = padded_slices[base + 1]
            sct_slice = padded_slices[base + 2]
            diff_slice = padded_slices[base + 3]

            im_mr = axes[row_idx, 0].imshow(mr_slice, cmap='gray', aspect='auto')
            im_ct = axes[row_idx, 1].imshow(ct_slice, cmap='gray', vmin=-1000, vmax=1000, aspect='auto')
            im_sct = axes[row_idx, 2].imshow(sct_slice, cmap='gray', vmin=-1000, vmax=1000, aspect='auto')
            im_diff = axes[row_idx, 3].imshow(diff_slice, cmap='bwr', vmin=-1000, vmax=1000, aspect='auto')

            axes[row_idx, 0].text(-0.03, 0.5, f'{patient_id}', transform=axes[row_idx, 0].transAxes,
                                  fontsize=29, ha='right', va='center', rotation='vertical')

            for col in range(4):
                axes[row_idx, col].axis('off')
            row_idx += 1

    
    # Add shared colorbars for HU (CT, sCT) and HU Difference
    ct_ax = axes[-1, 1]
    sct_ax = axes[-1, 2]
    diff_ax = axes[-1, 3]
    
    divider_mri = make_axes_locatable(axes[-1, 0])
    cax_mri = divider_mri.append_axes("bottom", size="7%", pad=0.10)
    cbar_mr = fig.colorbar(im_mr, cax=cax_mri, orientation='horizontal')
    cbar_mr.set_label("Range of values", fontsize=22)         
    cbar_mr.ax.tick_params(labelsize=14)        
    cbar_mr.ax.tick_params(labelrotation=45)

    divider_ct = make_axes_locatable(ct_ax)
    cax_ct = divider_ct.append_axes("bottom", size="7%", pad=0.10)
    cbar_ct = fig.colorbar(im_ct, cax=cax_ct, orientation='horizontal')
    cbar_ct.set_label("HU", fontsize=22)        
    cbar_ct.ax.tick_params(labelsize=14)         
    cbar_ct.ax.tick_params(labelrotation=45)

    divider_sct = make_axes_locatable(sct_ax)
    cax_sct = divider_sct.append_axes("bottom", size="7%", pad=0.10)
    fig.colorbar(im_sct, cax=cax_sct, orientation='horizontal')
    cbar_sct = fig.colorbar(im_sct, cax=cax_sct, orientation='horizontal')
    cbar_sct.set_label("HU", fontsize=22)
    cbar_sct.ax.tick_params(labelsize=14)
    cbar_sct.ax.tick_params(labelrotation=45)

    divider_diff = make_axes_locatable(diff_ax)
    cax_diff = divider_diff.append_axes("bottom", size="7%", pad=0.10)
    cbar_diff = fig.colorbar(im_diff, cax=cax_diff, orientation='horizontal')
    cbar_diff.set_label("HU Difference", fontsize=22)
    cbar_diff.ax.tick_params(labelsize=14)
    cbar_diff.ax.tick_params(labelrotation=45)
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.07, wspace=0.02, hspace=0.02)
    output_path = os.path.join(save_path, f"difference_slices_{patient_id_ab}_{patient_id_hn}_{patient_id_th}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()


# Multi-region

# --- Run for Worst Patients ---
difference_maps(
    patient_id_ab=region_extremes['AB']['worst']['patient'],
    patient_id_hn=region_extremes['HN']['worst']['patient'],
    patient_id_th=region_extremes['TH']['worst']['patient'],
    save_path="/home/catarina_caldeira/Desktop/code/utils/images_thesis/all_regions_model",
    directory="/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1",
    directory_synthetic_ct="/home/catarina_caldeira/Desktop/code/tests/mean_sCTs_generated_GaussianOverlap/SynthRAD2025/AE/8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionABHNTH/fold_all"
)


# --- Run for Best Patients ---
difference_maps(
    patient_id_ab=region_extremes['AB']['best']['patient'],
    patient_id_hn=region_extremes['HN']['best']['patient'],
    patient_id_th=region_extremes['TH']['best']['patient'],
    save_path="/home/catarina_caldeira/Desktop/code/utils/images_thesis/all_regions_model",
    directory="/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1",
    directory_synthetic_ct="/home/catarina_caldeira/Desktop/code/tests/mean_sCTs_generated_GaussianOverlap/SynthRAD2025/AE/8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionABHNTH/fold_all"
)


# Region-specific

# For single regions:
# AB
'''
difference_maps(
    patient_id_ab=region_extremes['AB']['worst']['patient'],
    patient_id_hn=None,
    patient_id_th=None,
    save_path="/home/catarina_caldeira/Desktop/code/utils/images_thesis/ab_regions_model",
    directory="/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1",
    directory_synthetic_ct="/home/catarina_caldeira/Desktop/code/tests/mean_sCTs_generated/SynthRAD2025/AE/8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionAB/fold_all"
)


# --- Run for Best Patients ---
difference_maps(
    patient_id_ab=region_extremes['AB']['best']['patient'],
    patient_id_hn=None,
    patient_id_th=None,
    save_path="/home/catarina_caldeira/Desktop/code/utils/images_thesis/ab_regions_model",
    directory="/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1",
    directory_synthetic_ct="/home/catarina_caldeira/Desktop/code/tests/mean_sCTs_generated/SynthRAD2025/AE/8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionAB/fold_all"
)

# HN
difference_maps(
    patient_id_ab=None,
    patient_id_hn=region_extremes['HN']['worst']['patient'],
    patient_id_th=None,
    save_path="/home/catarina_caldeira/Desktop/code/utils/images_thesis/hn_regions_model",
    directory="/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1",
    directory_synthetic_ct="/home/catarina_caldeira/Desktop/code/tests/mean_sCTs_generated/SynthRAD2025/AE/8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionHN/fold_all"
)


# --- Run for Best Patients ---
difference_maps(
    patient_id_ab=None,
    patient_id_hn=region_extremes['HN']['best']['patient'],
    patient_id_th=None,
    save_path="/home/catarina_caldeira/Desktop/code/utils/images_thesis/hn_regions_model",
    directory="/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1",
    directory_synthetic_ct="/home/catarina_caldeira/Desktop/code/tests/mean_sCTs_generated/SynthRAD2025/AE/8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionHN/fold_all"
)


# TH
difference_maps(
    patient_id_ab=None,
    patient_id_hn=None,
    patient_id_th=region_extremes['TH']['worst']['patient'],
    save_path="/home/catarina_caldeira/Desktop/code/utils/images_thesis/th_regions_model",
    directory="/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1",
    directory_synthetic_ct="/home/catarina_caldeira/Desktop/code/tests/mean_sCTs_generated/SynthRAD2025/AE/8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionTH/fold_all"
)


# --- Run for Best Patients ---
difference_maps(
    patient_id_ab=None,
    patient_id_hn=None,
    patient_id_th=region_extremes['TH']['best']['patient'],
    save_path="/home/catarina_caldeira/Desktop/code/utils/images_thesis/th_regions_model",
    directory="/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1",
    directory_synthetic_ct="/home/catarina_caldeira/Desktop/code/tests/mean_sCTs_generated/SynthRAD2025/AE/8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionTH/fold_all"
)
'''
    
    



# python -m utils.img_difference_maps
