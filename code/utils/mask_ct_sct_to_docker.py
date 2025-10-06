import os
import SimpleITK as sitk

def load_and_save_images(test_folder_patients, directory: str, img_type: str = None):
    for folder_patient in os.listdir(directory):
        patient_id = folder_patient.replace('.mha', '')
        
        if patient_id not in test_folder_patients:
            continue

        if img_type:
            patient_path = os.path.join(directory, folder_patient)
            img_path = os.path.join(patient_path, f"{img_type}.mha")
        else:
            img_path = os.path.join(directory, f"{patient_id}.mha")

        if not os.path.exists(img_path):
            print(f"File not found for {patient_id}, skipping.")
            continue

        region = patient_id[1:3]

        save_dir = f"/home/catarina_caldeira/Desktop/input_{region}"
        os.makedirs(save_dir, exist_ok=True)

        image = sitk.ReadImage(img_path)
        output_path = os.path.join(save_dir, f"{patient_id}.mha")
        sitk.WriteImage(image, output_path)

        print(f"Saved: {output_path}")

        

test_folder_patients = ['1HNA038', '1HNA117', '1HNA124', '1HNA115', '1HNA116', '1HNA010', '1HNA085', 
                        '1HNA061', '1HNA096', '1HNC043', '1HNC036', '1HNC037', '1HNC017', '1HNC025', '1HNC099',
                        '1THB211', '1THB210', '1THB003', '1THB195', '1THB135', '1THB103', '1THB202', 
                        '1THB191', '1THB222', '1THA252', '1THA041', '1THA221', '1THA022', '1THA291', '1THA203', 
                        '1THA270', '1THA028', '1THA244',
                        '1ABB059', '1ABB132', '1ABB151', '1ABB084', '1ABB123', '1ABB069', '1ABB115', '1ABB070', 
                        '1ABB168', '1ABA047', '1ABA018', '1ABA014', '1ABA054', '1ABA084', '1ABA114', '1ABC127']

'''
load_and_save_images(test_folder_patients, 
                     directory="/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1/AB", 
                     save_folder="/home/catarina_caldeira/Desktop/ground_truth/ct", 
                     img_type="ct")

load_and_save_images(test_folder_patients, 
                     directory="/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1/AB", 
                     save_folder="/home/catarina_caldeira/Desktop/ground_truth/mask", 
                     img_type="mask")

load_and_save_images(test_folder_patients, 
                     directory="/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1/TH", 
                     save_folder="/home/catarina_caldeira/Desktop/ground_truth/ct", 
                     img_type="ct")

load_and_save_images(test_folder_patients, 
                     directory="/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1/TH", 
                     save_folder="/home/catarina_caldeira/Desktop/ground_truth/mask", 
                     img_type="mask")

load_and_save_images(test_folder_patients, 
                     directory="/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1/HN", 
                     save_folder="/home/catarina_caldeira/Desktop/ground_truth_hn/ct", 
                     img_type="ct")

load_and_save_images(test_folder_patients, 
                     directory="/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1/HN", 
                     save_folder="/home/catarina_caldeira/Desktop/ground_truth_hn/mask", 
                     img_type="mask")
'''
# sCT
load_and_save_images(
    test_folder_patients,
    directory="/home/catarina_caldeira/Desktop/code/tests/mean_sCTs_generated_GaussianOverlap/SynthRAD2025/AE/8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionABHNTH/fold_all",
)



# python -m utils.mask_ct_sct_to_docker



