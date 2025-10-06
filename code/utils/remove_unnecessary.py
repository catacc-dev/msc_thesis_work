import os
import shutil

directory = "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1/"

# Allowed files (excluding the special folder)
allowed_files = {"ct.mha", "mask.mha", "mr.mha"}

for folder_region in os.listdir(directory):
    region_path = os.path.join(directory, folder_region)
    
    if not os.path.isdir(region_path):
        continue  # Skip non-directories
    
    for folder_patient in os.listdir(region_path):
        patient_path = os.path.join(region_path, folder_patient)
        
        if not os.path.isdir(patient_path):
            continue  # Skip non-directories
        
        special_folder = f"{folder_patient}_seg_matrad"
        
        for item in os.listdir(patient_path):
            item_path = os.path.join(patient_path, item)
            
            if item == special_folder:
                print(f"Skipping (protected folder): {item_path}")
                continue
            
            if item not in allowed_files:
                if os.path.isfile(item_path):
                    os.remove(item_path)
                    print(f"Deleted file: {item_path}")