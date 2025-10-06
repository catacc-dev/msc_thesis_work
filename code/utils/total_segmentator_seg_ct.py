import os
import SimpleITK as sitk
from totalsegmentator.python_api import totalsegmentator

# FOR MATRAD SOFTWARE + docker

def convert_mha_to_niigz(mha_path, nii_path):
    image = sitk.ReadImage(mha_path)
    sitk.WriteImage(image, nii_path)

def convert_niigz_to_mha(nii_path, mha_path):
    image = sitk.ReadImage(nii_path)
    sitk.WriteImage(image, mha_path)

def main(directory_path: str, output_segmentation_dir: str, docker_matrad: str):
    # List of patients to process (AB, TH from ALL regions model + HN from HN model)
    test_folder_patients = ['1HNA038', '1HNA117', '1HNA124', '1HNA115', '1HNA116', '1HNA010', '1HNA085', 
                            '1HNA061', '1HNA096', '1HNC043', '1HNC036', '1HNC037', '1HNC017', '1HNC025', '1HNC099',
                            '1THB211', '1THB210', '1THB003', '1THB195', '1THB135', '1THB103', '1THB202', 
                             '1THB191', '1THB222', '1THA252', '1THA041', '1THA221', '1THA022', '1THA291', '1THA203', 
                             '1THA270', '1THA028', '1THA244',
                             '1ABB059', '1ABB132', '1ABB151', '1ABB084', '1ABB123', '1ABB069', '1ABB115', '1ABB070', 
                             '1ABB168', '1ABA047', '1ABA018', '1ABA014', '1ABA054', '1ABA084', '1ABA114', '1ABC127']

 

    for folder_patient in os.listdir(directory_path):
        region = folder_patient[1:3]
        if folder_patient not in test_folder_patients:
            continue

        patient_path = os.path.join(directory_path, folder_patient)
        ct_path = os.path.join(patient_path, "ct.mha")
        if not os.path.exists(ct_path):
            print(f"CT file not found for {folder_patient}, skipping.")
            continue

        print(f"\nProcessing patient: {folder_patient}")

        # Convert .mha to .nii.gz
        ct_nii_path = os.path.join(patient_path, f"{folder_patient}.nii.gz")
        convert_mha_to_niigz(ct_path, ct_nii_path)
        

        if docker_matrad == "docker":
            # Run TotalSegmentator (multi-label output)
            seg_nii_path = os.path.join(patient_path, f"{folder_patient}_seg.nii.gz") # segments with all labels in one nii.gz 
            totalsegmentator(ct_nii_path, seg_nii_path, task="total", ml=True)
            
             # Convert segmentation .nii.gz back to .mha
            seg_mha_path = os.path.join(output_segmentation_dir, f"{folder_patient}.mha")
            convert_niigz_to_mha(seg_nii_path, seg_mha_path)
            print(f"Saved segmentation for {folder_patient} to {seg_mha_path}")
        else:
            if region == 'AB':
                tasks = ["total"] 
            elif region == 'TH':
                tasks = ["total", "lung_nodules"]  
            elif region == 'HN':
                tasks = ["total", "brain_structures", "head_glands_cavities", 
                    "head_muscles", "headneck_bones_vessels", "headneck_muscles"]
            for task in tasks:
                print(f"Running task: {task}")
                ct_seg_folder = os.path.join(patient_path, f"{folder_patient}_seg_matrad")
                os.makedirs(ct_seg_folder, exist_ok=True)
                
                totalsegmentator(ct_nii_path, ct_seg_folder, task=task, ml=False)
                
                print(f"Saved segmentation for {folder_patient} (task: {task})")
        
            
            

if __name__ == "__main__":
    # Input: directory containing patient folders with ct.mha
    # Output: directory where you want to save the final .mha segmentations
    '''
    main(
        directory_path="/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1/AB", # AB / HN /TH 
        output_segmentation_dir="/home/catarina_caldeira/Desktop/ground_truth/segmentation",
        docker_matrad="matrad"
    )
    
    main(
        directory_path="/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1/TH", # AB / HN /TH 
        output_segmentation_dir="/home/catarina_caldeira/Desktop/ground_truth/segmentation",
        docker_matrad="matrad"
    )
    
    main(
        directory_path="/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1/HN", # AB / HN /TH 
        output_segmentation_dir="/home/catarina_caldeira/Desktop/ground_truth/segmentation",
        docker_matrad="matrad"
    )
    '''
    
    
    '''
    main(
        directory_path="/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1/AB", # AB / HN /TH 
        output_segmentation_dir="/home/catarina_caldeira/Desktop/ground_truth_ab/segmentation",
        docker_matrad="docker"
    )
    '''
    main(
        directory_path="/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1/HN", # AB / HN /TH 
        output_segmentation_dir="/home/catarina_caldeira/Desktop/ground_truth_hn/segmentation",
        docker_matrad="docker"
    )
    '''
    main(
        directory_path="/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1/TH", # AB / HN /TH 
        output_segmentation_dir="/home/catarina_caldeira/Desktop/ground_truth_th/segmentation",
        docker_matrad="docker"
    )
    '''
    


# python -m utils.total_segmentator_seg_ct