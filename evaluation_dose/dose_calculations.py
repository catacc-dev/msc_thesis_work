#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
import numpy as np
from typing import Optional
import json
import os
import mat73
import math
import os
import scipy.io
   
   

class DoseMetrics():       
    def __init__(self, dose_path, prescribed_doses = {'AB': 2, 'HN': 2, 'TH': 2}):
        self.prescribed_dose = prescribed_doses
        self.dose_path = dose_path
    
    def load_mat_file(self, filepath):
        """Load MATLAB file, handling both v7.3 and older formats"""
        try:
            return scipy.io.loadmat(filepath, simplify_cells=True)
        except (NotImplementedError, TypeError):
            try:
                return mat73.loadmat(filepath)
            except Exception as e:
                warnings.warn(f"Failed to load {filepath}: {str(e)}")
                return None
             
    

    def score_patient(self, patient_id):
        region = patient_id[1:3]
        
        metrics = {}
        for rad_type in ['proton']:
            patient_folder = os.path.join(self.dose_path, patient_id)
            mat_file = os.path.join(patient_folder, f'{patient_id}_ct_and_sct_plan.mat')
            #print(mat_file)
            
            if os.path.isfile(mat_file):
                
                    # Load MAT files
                    mat = self.load_mat_file(mat_file)
                
                    # Extract dose cubes
                    gt_dose = mat['resultGUI_ct']['physicalDose']
                    pred_dose = mat['resultGUI_sct']['physicalDose']
                    
                    # Verify dose arrays contain non-zero values
                    if np.all(gt_dose == 0):
                        warnings.warn(f"Ground truth dose is all zeros for {patient_id}")
                        metrics[f'mae_target_{rad_type}'] = float('nan')
                    else:
                        
                        metrics[f'mae_target_{rad_type}'] = self.mae_dose(
                            gt_dose, pred_dose, region, threshold=0.9)
                        print(f"mae metric (1 fraction): {metrics[f'mae_target_{rad_type}']}")
                        #print(f"MAE for 30 fractions: {metrics[f'mae_target_{rad_type}']*30}")
                    
                    
                    # Calculate DVH metrics if available
                    if 'qi' in mat['resultGUI_ct'] and 'qi' in mat['resultGUI_sct']:
                        gt_dvh = mat['resultGUI_ct']['qi'] # for all structures
                        pred_dvh = mat['resultGUI_sct']['qi'] # for all structures
                        print(pred_dvh)
                        metrics[f'dvh_{rad_type}'], metrics[f'dvh_OAR_used_{rad_type}'] = self.dvh_metric(gt_dvh, pred_dvh, region)
                        print(f"dvh metric: {metrics[f'dvh_{rad_type}']}")
                    else:
                        warnings.warn(f"No DVH data found in MAT files for {patient_id}")
                        metrics[f'dvh_{rad_type}'] = float('nan')
                        
                    # Gamma passa rates
                    gamma_pass_rates = mat['gammaPassRateCell'][0][1] # whole CT gamma pass rate
                    #print(gamma_pass_rates)
                    metrics[f'gammas_pass_rate_{rad_type}']=gamma_pass_rates
                    print(f"gamma pass rate metric: {metrics[f'gammas_pass_rate_{rad_type}']}")

            else:
                warnings.warn(f"{patient_id}: {patient_id}_ct_plan.mat and/or {patient_id}_sct_plan.mat not present")
                metrics[f'mae_target_{rad_type}'] = float('nan')
                metrics[f'dvh_{rad_type}'] = float('nan')
        
        return metrics
       
           
    def mae_dose(self,
                 d_gt : np.ndarray, 
                 d_pred : np.ndarray,
                 region : str,
                 threshold: Optional[float] = 1) -> float:
        """
        Compute Mean Absolute Error (MAE) for the dose distributions given a certain
        threshold [0,1] relative to the prescribed dose.
    
        Parameters
        ----------
        d_gt : np.ndarray
            Dose distribution of the ground truth CT
        d_pred : np.ndarray
            Dose distribution of the predicted synthetic CT.
        region : str
            Which region is analyzed,.
        threshold : float, optional
            Theshold for determining the included voxels relative to the prescribed
            dose. It can be a value beteen 0 and 1. The default is 1.
    
        Returns
        -------
        mae_dose_value : float
            Mean absolute dose difference relative to the prescribed dose.
    
        """
        
        # Verify input arrays
        if not isinstance(d_gt, np.ndarray) or not isinstance(d_pred, np.ndarray):
            warnings.warn("Invalid dose array type")
            return float('nan')
            
        if d_gt.shape != d_pred.shape:
            warnings.warn("Dose array shapes don't match")
            return float('nan')
        
        # Threshold dose distributions
        abs_th = threshold * self.prescribed_dose[region]
        #print(abs_th)
        #print(d_gt)
        d_pred = d_pred[d_gt >= abs_th]
        #print(d_pred)
        d_gt = d_gt[d_gt >= abs_th]
        
        n = len(d_gt)
        if n == 0:
            warnings.warn(f"No voxels above threshold {abs_th} for region {region}. Returning NaN.")
            return float('nan')
            
        # Calculate MAE     
        mae_dose_value = np.sum(np.abs(d_gt - d_pred)/self.prescribed_dose[region])/n
        return float(mae_dose_value)
    
    def dvh_metric(self, 
                   gt_dvh : np.ndarray, 
                   pred_dvh : np.ndarray,
                   region : str,
                   eps : Optional[float] = 1e-12 ) -> float:
        """
        Calculate the dose volume histogram (DVH) metric from the given DVH
        parameters.

        Parameters
        ----------
        gt_dvh : np.ndarray
            DVH parameters for the dose calculation on ground truth CT.
        pred_dvh : np.ndarray
            DVH parameters for the dose calculation on predicted synthetic CT.
        region : str
            The region for this image. 
        eps : float, optional
            Small epsilon value to prevent division by zero
            
        Returns
        -------
        DVH_metric : float
            One combined metric based on several DVH parameters for the SynthRAD 
            challenge.

        """
        gt_organs = {}
        pred_organs = {}
        
        gt_organs = {gt_dvh[i]['name']: gt_dvh[i] for i in range(len(gt_dvh))}
        #print(gt_organs)
        pred_organs = {pred_dvh[i]['name']: pred_dvh[i] for i in range(len(pred_dvh))}
        
        # target metrics
        if region=='AB':
            gtv=['stomach']
        elif region=='TH':
            gtv=['lung_upper_lobe_right']
        elif region=='HN':
            gtv=['tongue', 'hard_palate', 'soft_palate']
        
        # Aggregate D98 and V95/CI errors over all targets
        D98_errors = []
        V95_errors = []
        
        for t in gtv:
                gtv_gt = gt_organs.get(t)
                print(t)
                gtv_pred = pred_organs.get(t)
                
                #gtv_gt = gt_organs[gtv]
                #gtv_pred = pred_organs[gtv]

                # Find D_98
                if gtv_gt['D_98'] is None or math.isnan(gtv_gt['D_98']):
                    warnings.warn('None or NaN value in ground truth GTV D98')
                    return float('nan')
                elif gtv_pred['D_98'] is None or math.isnan(gtv_pred['D_98']):
                    warnings.warn('None or NaN value in sCT GTV D98')
                    return float('nan')
                else:
                    D98_target = ( np.abs(gtv_gt['D_98'] - gtv_pred['D_98'] + eps ) /
                                (gtv_pred['D_98'] + eps ) ) # the denominator was changed!
                    D98_errors.append(D98_target)
            
                # Find CI
                ci_key = next((k for k in gtv_gt if k.startswith('CI')), None)
                #print(ci_key)
                    
                if gtv_gt[ci_key] is None or math.isnan(gtv_gt[ci_key]):
                    warnings.warn('None or NaN value in ground truth GTV CI')
                    return float('nan')
                elif gtv_pred[ci_key] is None or math.isnan(gtv_pred[ci_key]):
                    warnings.warn('None or NaN value in sCT GTV CI')
                    return float('nan')
                else:
                    V95_target = ( np.abs(gtv_gt[ci_key] - gtv_pred[ci_key] + eps ) /
                                ( gtv_pred[ci_key] + eps ) )  # the denominator was changed!
                    print(V95_target)
                    V95_errors.append(V95_target)         
                
                #target_term = D98_target + V95_target
                # Average over all targets
                target_term = np.mean(D98_errors) + np.mean(V95_errors)
                #print(target_term)
            

        # Define which 2 organs at risk are used for the evaluation (higher mean between D5 and Dmean)
        mean_D5_Dmean_per_OAR = {}
        for organ, organ_data in gt_organs.items():
                if organ not in gtv and 'D_5' in organ_data and 'mean' in organ_data:
                    val_D5 = organ_data['D_5']
                    val_mean = organ_data['mean']
                    if val_D5 is not None and val_mean is not None and not math.isnan(val_D5) and not math.isnan(val_mean):
                        mean_D5_Dmean_per_OAR[organ] = (val_D5 + val_mean) / 2
        OAR_used = sorted(mean_D5_Dmean_per_OAR, key=mean_D5_Dmean_per_OAR.get, reverse =True)

        OAR_used = OAR_used[ : min([3, len(OAR_used)])]
        #print(mean_D5_Dmean_per_OAR)
        #print(OAR_used)
            
            
        # Define OAR DVH term
        D2_OAR, Dmean_OAR = [], []
        for organ in OAR_used:
                oar_gt = gt_organs[organ]
                oar_pred = pred_organs[organ]
                
                # Determine D2 for selected organ at risk
                if oar_gt['D_2']  is None or math.isnan(oar_gt['D_2']):
                    warnings.warn(f'None or NaN value in ground truth {organ} D2')
                    return float('nan')
                elif oar_pred['D_2'] is None or math.isnan(oar_pred['D_2']):
                    warnings.warn(f'None or NaN value in sCT {organ} D2')
                    return float('nan')
                else:
                    D2_OAR.append( ( np.abs(oar_gt['D_2'] - oar_pred['D_2'] + eps ) /
                                    ( oar_pred['D_2'] + eps) ) ) # CHANGED
                
                # Determine Dmean for selected organ at risk
                if oar_gt['mean']  is None or math.isnan(oar_gt['mean']):
                    warnings.warn(f'None or NaN value in ground truth {organ} Dmean')
                    return float('nan')
                elif oar_pred['mean'] is None or math.isnan(oar_pred['mean']):
                    warnings.warn(f'None or NaN value in sCT {organ} Dmean')
                    return float('nan')
                else:            
                    Dmean_OAR.append( ( np.abs(oar_gt['mean'] - oar_pred['mean'] + eps ) /
                                        ( oar_pred['mean'] + eps) ) ) # CHANGED
                

        # Calcuate the OAR term and print a warning when less than 3 are used.
        if len(D2_OAR)>0:
                OAR_term = 1/(len(D2_OAR)) * np.sum(D2_OAR) + 1/(len(Dmean_OAR)) * np.sum(Dmean_OAR)
                print(OAR_term)
                #OAR_term = np.mean(D2_OAR) + np.mean(Dmean_OAR)
                #print(OAR_term)

        else:
                OAR_term = 0
        print(f'Used OARs: {OAR_used}')
        
        # Calculate sum
        return float(target_term + OAR_term), OAR_used
 
 
def convert_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray(elem) for elem in obj]
    else:
        return obj


if __name__=='__main__':
    
    dose_path = "C:/Users/catar/OneDrive - Universidade de Coimbra/Ambiente de Trabalho/Master Thesis/e0404-matRad-98ba2fb/userdata/patients/"
    all_results = {}
    
    for patient_folder in os.listdir(dose_path):
        
        patient_path = os.path.join(dose_path, patient_folder)
        
        if os.path.isdir(patient_path) and patient_folder!=".venv" and patient_folder!="__pycache__":
            
            #predicted_path = "C:\\Users\\catar\\OneDrive - Universidade de Coimbra\\Ambiente de Trabalho\\Master Thesis\\e0404-matRad-98ba2fb\\userdata\\patients\\1THA244\\1THA244.mha" # sct
            metrics = DoseMetrics(dose_path)
            print(f"patient: {patient_folder}")
            patient_metrics = metrics.score_patient(patient_folder)
                    
            result = {
                        "patient_id": patient_folder,
                        "MAE": patient_metrics.get('mae_target_proton'), 
                        "DVH": patient_metrics.get('dvh_proton'),
                        "OAR": patient_metrics.get('dvh_OAR_used_proton'),
                        "Gamma pass rate": patient_metrics.get('gammas_pass_rate_proton'),
                        
            }
            all_results[patient_folder] = result
                    
    # Define output file path
    output_file = os.path.join(dose_path, "all_metrics.json")
                
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(convert_ndarray(all_results), f, indent=4)

'''
#mat = scipy.io.loadmat('1ABA084/1ABA084_ct_and_sct_plan.mat')
mat=mat73.loadmat('1ABA084/1ABA084_ct_and_sct_plan.mat')
print(mat.keys())  # Show top-level keys
print(mat['resultGUI_ct'].keys())  # Show resultGUI contents
print(mat['resultGUI_ct']['qi'])
#print(np.unique(mat['resultGUI']['physicalDose']))  
#print(mat['resultGUI_sct']['qi'])

'''
