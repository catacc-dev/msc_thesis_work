import os
import shutil

test_folder_patients = ['1HNA038', '1HNA117', '1HNA124', '1HNA115', '1HNA116', '1HNA010', '1HNA085', 
                        '1HNA061', '1HNA096', '1HNC043', '1HNC036', '1HNC037', '1HNC017', '1HNC025', '1HNC099',
                        '1THB211', '1THB210', '1THB003', '1THB195', '1THB135', '1THB103', '1THB202', 
                        '1THB191', '1THB222', '1THA252', '1THA041', '1THA221', '1THA022', '1THA291', '1THA203', 
                        '1THA270', '1THA028', '1THA244',
                        '1ABB059', '1ABB132', '1ABB151', '1ABB084', '1ABB123', '1ABB069', '1ABB115', '1ABB070', 
                        '1ABB168', '1ABA047', '1ABA018', '1ABA014', '1ABA054', '1ABA084', '1ABA114', '1ABC127']

# sCT
'''
directory_dest = "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1/"
directory = "/home/catarina_caldeira/Desktop/code/tests/mean_sCTs_generated_GaussianOverlap/SynthRAD2025/AE/8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionABHNTH/fold_all"
for sct_files in os.listdir(directory):
        patient_id = sct_files.split(".mha")[0]
        
        if patient_id not in test_folder_patients:
                continue
        
        src_path = os.path.join(directory, sct_files)
        print(src_path)
        if sct_files.startswith("1AB"):
                dst_path = os.path.join(directory_dest, "AB", patient_id)
                print(src_path)
                print(dst_path)
                destination = shutil.copy2(src_path, dst_path)
        elif sct_files.startswith("1TH"):
                dst_path = os.path.join(directory_dest, "TH", patient_id)
                destination = shutil.copy2(src_path, dst_path)
        elif sct_files.startswith("1HN"):
                dst_path = os.path.join(directory_dest, "HN", patient_id)
                destination = shutil.copy2(src_path, dst_path)
'''


# All
directory = "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1/TH"
for folder_patient in os.listdir(directory):
    
    if folder_patient not in test_folder_patients:
            continue
    
    dest = "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1"
    os.makedirs(dest, exist_ok=True)

    src_path = os.path.join(directory, folder_patient)
    dst_path = os.path.join(dest, folder_patient)

    # Remove pasta de destino se já existir
    #shutil.rmtree(dst_path)

    # Copy
    shutil.copytree(src_path, dst_path)
    print("Copied to:", dst_path)
