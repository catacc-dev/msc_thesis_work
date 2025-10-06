
import os
import SimpleITK as stik

# masks
def load_and_convert_images(directory: str, save_folder: str):

    for folder_patient in os.listdir(directory):
        if folder_patient.lower() in ["overview", "overviews"]: # overview - 2023; overviews - 2025
            continue
        
        patient_path = os.path.join(directory, folder_patient)
        mri_path = os.path.join(patient_path, "mr.mha")
        ct_path = os.path.join(patient_path, "ct.mha")
        mask_path = os.path.join(patient_path, "mask.mha")
        
        patient_output_dir = os.path.join(save_folder, folder_patient)
        os.makedirs(patient_output_dir, exist_ok=True)
        
        # Convert .mha -> .nii.gz
        output_mri_nii = os.path.join(patient_output_dir, "mr.nii.gz")
        image_mr = stik.ReadImage(mri_path)
        #mri_path_nii = f"{folder_patient}_mr.nii.gz"
        stik.WriteImage(image_mr, output_mri_nii)
        
        output_ct_nii = os.path.join(patient_output_dir, "ct.nii.gz" )
        image_ct = stik.ReadImage(ct_path)
        #ct_path_nii = f"{folder_patient}_ct.nii.gz"
        stik.WriteImage(image_ct, output_ct_nii)
        
        output_mask_nii = os.path.join(patient_output_dir, "mask.nii.gz" )
        image_mask = stik.ReadImage(mask_path)
        #mask_path_nii = f"{folder_patient}_mask.nii.gz"
        stik.WriteImage(image_mask, output_mask_nii)
                
                
if __name__ == "__main__":
    #directory = "/home/catarina_caldeira/Imagens/SynthRAD2025dataset/synthRAD2025_Task1_Train/Task1/HN"
    save_folder = "/home/catarina_caldeira/Desktop/code/utils/test_external"
    directory = "/home/catarina_caldeira/Imagens/SynthRAD2025dataset/synthRAD2025_Task1_Train_D/Task1/HN"
    load_and_convert_images(directory, save_folder)

