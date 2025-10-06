import os
import SimpleITK as sitk


def decompress_files(directory: str, save_folder: str):

    for folder_patient in os.listdir(directory):
        
        
        test_folder_patients = ['1HNA038', '1HNA117', '1HNA124', '1HNA115', '1HNA116', '1HNA010', '1HNA085', 
                        '1HNA061', '1HNA096', '1HNC043', '1HNC036', '1HNC037', '1HNC017', '1HNC025', '1HNC099',
                        '1THB211', '1THB210', '1THB003', '1THB195', '1THB135', '1THB103', '1THB202', 
                        '1THB191', '1THB222', '1THA252', '1THA041', '1THA221', '1THA022', '1THA291', '1THA203', 
                        '1THA270', '1THA028', '1THA244',
                        '1ABB059', '1ABB132', '1ABB151', '1ABB084', '1ABB123', '1ABB069', '1ABB115', '1ABB070', 
                        '1ABB168', '1ABA047', '1ABA018', '1ABA014', '1ABA054', '1ABA084', '1ABA114', '1ABC127']

        if folder_patient not in test_folder_patients:
            continue
        
        # Original compressed .mha file
        patient_path = os.path.join(directory, folder_patient)
        ct_path = os.path.join(patient_path, "ct.mha")
        mask_path = os.path.join(patient_path, "mask.mha")
        patient_path_output = os.path.join(save_folder, folder_patient)
        
        # Decompressed .mha file
        image_ct = sitk.ReadImage(ct_path)
        output_ct_d = os.path.join(patient_path_output,"ct.mha")
        sitk.WriteImage(image_ct, output_ct_d, useCompression=False)
        
        # Decompressed .mha file
        image_mask = sitk.ReadImage(mask_path)
        output_mask_d = os.path.join(patient_path_output,"mask.mha")
        sitk.WriteImage(image_mask, output_mask_d, useCompression=False)
        
                
if __name__ == "__main__":
    
    save_folder = "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1/"
    directory = "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1"
    decompress_files(directory, save_folder)


