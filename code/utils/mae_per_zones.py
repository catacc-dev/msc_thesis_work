import SimpleITK as sitk
import torch
import os

# Pytorch Lightning
from torchmetrics.regression import MeanAbsoluteError

def calc_masked(gt, pred, mask, metric_fn):
    # mask from Total Segmentator of a specific region 
    # From dose calculations table:
    # (soft tissues - TH:lung, AB: liver, HN: brain stem; air; bone - HN: skull, AB: vertebrae_L1, TH: sternum)
    
    gt = sitk.ReadImage(gt) # SimpleITK image
    pred = sitk.ReadImage(pred)
    mask = sitk.ReadImage(mask)
    
    gt = sitk.GetArrayFromImage(gt)
    pred = sitk.GetArrayFromImage(pred)
    mask = sitk.GetArrayFromImage(mask)
    
    mask_bool = mask == 1
    
    gt_tensor = torch.tensor(gt[mask_bool], dtype=torch.float32).to(metric_fn.device)
    pred_tensor = torch.tensor(pred[mask_bool], dtype=torch.float32).to(metric_fn.device)

    return metric_fn(pred_tensor, gt_tensor).item()
    #return metric_fn(gt[mask], pred[mask]) # metric_fn is the metric function

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
mae = MeanAbsoluteError().to(device)

directory_gt_ct = "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1"
directory_synthetic_ct = "/home/catarina_caldeira/Desktop/code/tests/mean_sCTs_generated/SynthRAD2025/AE/8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_ABC/fold_all"
soft_tissues = ['lung', 'brainstem', 'liver']
#bones = ['sternum', 'skull', 'vertebrae']

list_mae_th, list_mae_hn, list_mae_ab = [], [], []

for region_folder in os.listdir(directory_gt_ct):
    region_folder_path = os.path.join(directory_gt_ct, region_folder)
    for folder_patient in os.listdir(region_folder_path):
        region = folder_patient[1:3]
        patient_path = os.path.join(region_folder_path, folder_patient)
        path_to_gt = os.path.join(patient_path, "ct.mha")
        
        if region=='TH':
            path_to_seg = os.path.join(patient_path, f"{folder_patient}_seg_matrad")
            path_to_mask = os.path.join(path_to_seg, f"{soft_tissues[0]}.nii.gz")
            
            # mean image from ensemble of models
            path_to_pred = os.path.join(directory_synthetic_ct, f"{folder_patient}.mha")
            
            # Check if ground truth CT exists
            if not os.path.exists(path_to_pred):
                continue
            
            mae_th = calc_masked(path_to_gt, path_to_pred, path_to_mask, mae)
            print(f"{soft_tissues[0]} MAE for patient {folder_patient}: {mae_th:.2f}")
            list_mae_th.append(mae_th)
                
        elif region=='HN':
            path_to_seg = os.path.join(patient_path, f"{folder_patient}_seg_matrad")
            path_to_mask = os.path.join(path_to_seg, f"{soft_tissues[1]}.nii.gz")
            
            # mean image from ensemble of models
            path_to_pred = os.path.join(directory_synthetic_ct, f"{folder_patient}.mha")
            
            # Check if ground truth CT exists
            if not os.path.exists(path_to_pred):
                continue
            
            mae_hn = calc_masked(path_to_gt, path_to_pred, path_to_mask, mae)
            print(f"{soft_tissues[1]} MAE for patient {folder_patient}: {mae_hn:.2f}")
            list_mae_hn.append(mae_hn)
                
        elif region=='AB':
            path_to_seg = os.path.join(patient_path, f"{folder_patient}_seg_matrad")
            path_to_mask = os.path.join(path_to_seg, f"{soft_tissues[2]}.nii.gz")
            
            # mean image from ensemble of models
            path_to_pred = os.path.join(directory_synthetic_ct, f"{folder_patient}.mha")
            
            # Check if ground truth CT exists
            if not os.path.exists(path_to_pred):
                continue
            
            mae_ab = calc_masked(path_to_gt, path_to_pred, path_to_mask, mae)
            print(f"{soft_tissues[2]} MAE for patient {folder_patient}: {mae_ab:.2f}")
            list_mae_ab.append(mae_ab)
            


print(list_mae_ab)
print(list_mae_th)
print(list_mae_hn)
                
            
# python -m utils.mae_per_zones > log_mae_per_zones.txt