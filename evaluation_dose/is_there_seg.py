import os
import nibabel as nib
import numpy as np

dose_path = "/patients/"
all_results = {}

for patient_folder in os.listdir(dose_path):
    patient_path = os.path.join(dose_path, patient_folder)
    
    if not os.path.isdir(patient_path) or patient_folder in [".venv", "is_there_seg.py"]:
        continue  

    seg_path = os.path.join(patient_path, f"{patient_folder}_seg_matrad")
    if not os.path.exists(seg_path):
        continue  

    all_results[patient_folder] = {}

    for segs in os.listdir(seg_path):
        segment_type = None
        # Define segment filters based on patient type
        if patient_folder[1:3] == "HN" and segs in ["spinal_cord.nii.gz", "brain.nii.gz", "skull.nii.gz"]:
            segment_type = "HN_" + segs.split('.')[0] 
        elif patient_folder[1:3] in ["AB", "TH"] and segs == "sternum.nii.gz":
            segment_type = "TH/AB_sternum"
        else:
            continue  

        segment_file = os.path.join(seg_path, segs)
        try:
            img = nib.load(segment_file)
            data = img.get_fdata()
            
            is_valid = np.any(data > 0)
            status = "Valid" if is_valid else "Empty"
            print(f"{segment_file}: {status}")
            
            all_results[patient_folder][segment_type] = is_valid
        except Exception as e:
            print(f"Error reading {segment_file}: {str(e)}")
            all_results[patient_folder][segment_type] = None

# Print summary
print("\n=== Results ===")
for patient, segments in all_results.items():
    print(f"\nPatient: {patient}")
    for seg_name, status in segments.items():
        print(f"  {seg_name}: {'Valid' if status else 'Empty' if status is not None else 'Error'}")