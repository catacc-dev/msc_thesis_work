import numpy as np 
import h5py
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

def format_name(name):
    name = name.replace('esophagus', 'oesophagus')
    return name.replace('_', ' ').title()

def visualize_patient_dose_and_segments(segments_names, patient_id, slices):
    # --- Paths ---
    base_path = f"\\userdata\\patients\\{patient_id}"
    h5_path = os.path.join(base_path, f"{patient_id}_ct_and_sct_plan.mat")
    ct_path = os.path.join(base_path, "ct.mha")
    sct_path = os.path.join(base_path, f"{patient_id}.mha")
    mask_path = os.path.join(base_path, "mask.mha")
    segs_path = os.path.join(base_path, f"{patient_id}_seg_matrad")

    # --- Load HDF5 doses ---
    h5 = h5py.File(h5_path, "r")
    dose_sct = h5["resultGUI_sct"]["physicalDose"][::]
    dose_ct = h5["resultGUI_ct"]["physicalDose"][::]
    dose_ct_swapped = np.swapaxes(dose_ct, 1, 2)
    dose_sct_swapped = np.swapaxes(dose_sct, 1, 2)

    dose_sct_array = sitk.GetImageFromArray(dose_sct_swapped)
    dose_ct_array = sitk.GetImageFromArray(dose_ct_swapped)
    sitk.WriteImage(dose_sct_array, "dose_sct.mha")
    sitk.WriteImage(dose_ct_array, "dose_ct.mha")

    # --- Load Images ---
    ct = sitk.ReadImage(ct_path)
    sct = sitk.ReadImage(sct_path)
    mask = sitk.ReadImage(mask_path)

    ct_array = sitk.GetArrayFromImage(ct)
    sct_array = sitk.GetArrayFromImage(sct)
    mask_array = sitk.GetArrayFromImage(mask)

    # --- Load Segmentations ---
    all_segs_arrays = []
    all_segs_labels = []
    for name in segments_names:
        full_path_segment = os.path.join(segs_path, name + ".nii.gz")
        seg = sitk.ReadImage(full_path_segment)
        seg_array = sitk.GetArrayFromImage(seg)
        all_segs_arrays.append(seg_array)
        all_segs_labels.append(name)

    # --- Build ROI mask ---
    roi_mask = np.sum(all_segs_arrays, axis=0)
    roi_mask = (roi_mask > 0).astype(np.uint8)

    # --- Apply ROI mask ---
    masked_dose_ct = np.where(roi_mask == 1, dose_ct_swapped, np.nan)
    masked_dose_sct = np.where(roi_mask == 1, dose_sct_swapped, np.nan)
    masked_ct = np.where(mask_array == 1, ct_array, -1024)
    masked_sct = np.where(mask_array == 1, sct_array, -1024)

    # --- Colors ---
    unique_colors = {
        'brainstem': "#3A0202", 'common_carotid_artery_left': "#65cceb", 'common_carotid_artery_right': "#1974B1",
        'esophagus': "#7bff77", 'eye_lens_left': "#FF5E00", 'eye_lens_right': "#FF9050", 'hard_palate': "#7195FF",
        'inferior_pharyngeal_constrictor': "#ec9cf8", 'larynx_air': "#20eddc", 'masseter_left': "#4daf4a",
        'masseter_right': "#7bff77", 'middle_pharyngeal_constrictor': "#8D0057", 'optic_nerve_left': "#FF5E00",
        'optic_nerve_right': "#FF9050", 'parotid_gland_left': "#ff6eb9", 'parotid_gland_right': "#fd0fe9",
        'soft_palate': "#0000FF", 'spinal_cord': "#BFA3FF", 'submandibular_gland_left': "#53c5a1",
        'submandibular_gland_right': "#afffca", 'superior_pharyngeal_constrictor': "#c12fd7",
        'thyroid_gland': "#E4FF98", 'tongue': "#8A5EF3", 'kidney_left': "#006400", 'kidney_right': "#4CF04C",
        'liver': "#EBDB00", 'lung_upper_lobe_right': "#fd0fe9", 'lung_middle_lobe_right':"#FC5C00",
        'lung_lower_lobe_right': "#449700", 'lung_upper_lobe_left': "#FDD835", 'lung_lower_lobe_left': "#c12fd7",
        'stomach': "#FF0077", 'urinary_bladder': "#ac9986"
    }

    # --- Plotting ---
    fig, axes = plt.subplots(2, len(slices), figsize=(16, 8))
    plt.subplots_adjust(right=0.85)

    for col, slice_idx in enumerate(slices):
        axes[0, col].imshow(masked_ct[slice_idx], cmap="gray", aspect='auto')
        dose_im_ct = axes[0, col].imshow(masked_dose_ct[slice_idx], cmap="jet", aspect='auto')
        for seg_arr, name in zip(all_segs_arrays, all_segs_labels):
            color = unique_colors.get(name, '#000000')
            if np.sum(seg_arr[slice_idx]) > 0:
                axes[0, col].contour(seg_arr[slice_idx], colors=color, linewidths=1)
        axes[0, col].axis("off")

        axes[1, col].imshow(masked_sct[slice_idx], cmap="gray", aspect='auto')
        dose_im_sct = axes[1, col].imshow(masked_dose_sct[slice_idx], cmap="jet", aspect='auto')
        for seg_arr, name in zip(all_segs_arrays, all_segs_labels):
            color = unique_colors.get(name, '#000000')
            if np.sum(seg_arr[slice_idx]) > 0:
                axes[1, col].contour(seg_arr[slice_idx], colors=color, linewidths=1)
        axes[1, col].axis("off")

    axes[0, 0].text(-0.1, 0.5, "CT", fontsize=16, va='center', ha='center', rotation='vertical', transform=axes[0, 0].transAxes)
    axes[1, 0].text(-0.1, 0.5, "sCT", fontsize=16, va='center', ha='center', rotation='vertical', transform=axes[1, 0].transAxes)

    ax = axes[0, -1]
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(dose_im_ct, cax=cax).set_label("Dose [Gy]", fontsize=12)

    ax = axes[1, -1]
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(dose_im_sct, cax=cax).set_label("Dose [Gy]", fontsize=12)

    legend_names = []
    legend_colors = []
    for name in all_segs_labels:
        if name not in legend_names:
            legend_names.append(name)
            legend_colors.append(unique_colors.get(name, '#000000'))

    patches = [Line2D([0], [0], color=color, lw=2, label=format_name(name))
               for name, color in zip(legend_names, legend_colors)]

    fig.legend(handles=patches, loc='lower center', ncol=4,  bbox_to_anchor=(0.5, -0.01), fontsize=14,
               frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    # bottom=0.24 - HN
    # bottom=0.13 - TH
    plt.subplots_adjust(bottom=0.24, left=0.07, wspace=0.02, hspace=0.02)
    plt.savefig(f"{patient_id}_ct_sct_all_segments.png", dpi=300)
    #plt.show()


# --- Plot slices with all contours ---
slices_hn = [77,78,79,80]  # hn
#slices_th = [45, 50, 55, 60]  # th

'''
segments_names = ['spinal_cord', 'esophagus', 'lung_upper_lobe_right', 'lung_middle_lobe_right', 'lung_lower_lobe_right', 'lung_upper_lobe_left', 'lung_lower_lobe_left']
'''
segments_names = ['spinal_cord','esophagus','thyroid_gland', 
               'common_carotid_artery_right','common_carotid_artery_left', 
               'brainstem','optic_nerve_left','optic_nerve_right', 
               'submandibular_gland_right','submandibular_gland_left','larynx_air',
               'eye_lens_right','eye_lens_left', 
               'masseter_right','masseter_left','parotid_gland_left', 
               'parotid_gland_right','superior_pharyngeal_constrictor', 
               'middle_pharyngeal_constrictor','inferior_pharyngeal_constrictor',
               'tongue', 'hard_palate', 'soft_palate']


# 1HNA085 , 1THA028
visualize_patient_dose_and_segments(segments_names=segments_names, patient_id="1HNA085", slices=slices_hn)