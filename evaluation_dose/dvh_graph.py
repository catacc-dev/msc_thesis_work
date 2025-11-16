import json
import os
import mat73
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.io
import warnings
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import numpy as np
from matplotlib.transforms import Bbox



data_path = "\\patients\\"
metrics_path = 'all_metrics.json'

with open(metrics_path, 'r') as f:
    all_metrics_data = json.load(f)

def best_patient(all_metrics_data):
    min_hn_dvh, min_th_dvh, min_ab_dvh = float('inf'), float('inf'), float('inf')
    best_hn_id, best_th_id, best_ab_id = None, None, None

    for patient_id, metrics in all_metrics_data.items():
        region = patient_id[1:3]
        patient_path = os.path.join(data_path, patient_id)

        if "DVH" in metrics and os.path.isdir(patient_path):
                current_dvh = metrics["DVH"]
                if region == "HN" and current_dvh < min_hn_dvh:
                    min_hn_dvh = current_dvh
                    best_hn_id = patient_id
                elif region == "TH" and current_dvh < min_th_dvh:
                    min_th_dvh = current_dvh
                    best_th_id = patient_id
                elif region == "AB" and current_dvh < min_ab_dvh:
                    min_ab_dvh = current_dvh
                    best_ab_id = patient_id
                    
    print(f"Best HN patient: {best_hn_id} with DVH metric: {min_hn_dvh}")
    print(f"Best TH patient: {best_th_id} with DVH metric: {min_th_dvh}")
    print(f"Best AB patient: {best_ab_id} with DVH metric: {min_ab_dvh}")
    
    return best_hn_id, best_th_id, best_ab_id

def load_mat_file(filepath):
        """Load MATLAB file, handling both v7.3 and older formats"""
        try:
            return scipy.io.loadmat(filepath, simplify_cells=True)
        except (NotImplementedError, TypeError):
            try:
                return mat73.loadmat(filepath)
            except Exception as e:
                warnings.warn(f"Failed to load {filepath}: {str(e)}")
                return None
            
def format_name(name):
    # Rename specific terms for display purposes only
    name = name.replace('esophagus', 'oesophagus')
    return name.replace('_', ' ').title()

            
             
def plot_dvh(best_patient_id, data_path):

    mat_file = os.path.join(data_path, best_patient_id, f'{best_patient_id}_ct_and_sct_plan.mat')

    mat = load_mat_file(mat_file)
    gt_dvh = mat['resultGUI_ct']['dvh']
    pred_dvh = mat['resultGUI_sct']['dvh']
    
    gt_dvh_values = mat['resultGUI_ct']['qi']
    pred_dvh_values = mat['resultGUI_sct']['qi']
    
    gt_organs_dict = {gt_dvh_values[i]['name']: gt_dvh_values[i] for i in range(len(gt_dvh_values))}
    #print(gt_organs_dict)
    pred_organs_dict = {pred_dvh_values[i]['name']: pred_dvh_values[i] for i in range(len(pred_dvh_values))}
    n_segments = len(gt_dvh)
    
    color_map = {
    'brainstem': "#380000",
    'common_carotid_artery_left': "#65cceb",
    'common_carotid_artery_right': "#1974B1",
    'esophagus': "#7400C2",
    'eye_lens_left': "#FF5E00",
    'eye_lens_right': "#FF9050",
    'hard_palate': "#7195FF",
    'inferior_pharyngeal_constrictor': "#ec9cf8",
    'larynx_air': "#20eddc",
    'masseter_left': "#4daf4a",
    'masseter_right': "#7bff77",
    'middle_pharyngeal_constrictor': "#8D0057",
    'optic_nerve_left': "#FF5E00",
    'optic_nerve_right': "#FF9050",
    'parotid_gland_left': "#ff6eb9",
    'parotid_gland_right': "#fd0fe9",
    'soft_palate': "#0000FF",
    'spinal_cord': "#BFA3FF",
    'submandibular_gland_left': "#53c5a1",
    'submandibular_gland_right': "#afffca",
    'superior_pharyngeal_constrictor': "#c12fd7",
    'thyroid_gland': "#E4FF98",
    'tongue': "#8A5EF3",
    'kidney_left': "#006400",
    'kidney_right': "#4CF04C",
    'liver': "#EBDB00",
    'lung_upper_lobe_right': "#CC3232", 
    'lung_middle_lobe_right':"#640202", 
    'lung_lower_lobe_right': "#640202D7", 
    'lung_upper_lobe_left': "#64020282", 
    'lung_lower_lobe_left': "#6402021B",
    'stomach': "#FF0077",
    'urinary_bladder': "#ac9986"
    }

    name_list = []
    fig, ax = plt.subplots(figsize=(12, 5))
    
    text_y_positions = {
        'D_98': -3,
        'D_2': -3,
        'mean': -3
        }
    
    hn_metrics_ct = {
        'D_98': [],
        'D_2': [],
        'mean': []
        }

    hn_metrics_sct = {
        'D_98': [],
        'D_2': [],
        'mean': []
    }
    
    metric_labels_hn = {
        'D_98': 'D98% (mean)',
        'D_2': 'D2% (mean)',
        'mean': 'Dmean (mean)'
    }  

    for seg in range(n_segments):
        
        # CT DVH
        gt_segment = gt_dvh[seg]
        x_gt = gt_segment['doseGrid']
        y_gt = gt_segment['volumePoints']
        name = gt_segment['name']
        values_gt = gt_organs_dict.get(name)

        name_list.append(name)
        name_clean = name.lower().strip()
        color = color_map.get(name_clean, "#000000")
        ax.plot(x_gt, y_gt, linestyle='-', linewidth=2, color=color, label=f'{format_name(name)}', zorder=3)

        # TH, AB
        if values_gt is not None and name in ["lung_upper_lobe_right","stomach"]: # target
            for metric, label in [('D_98', 'D98%')]:
                if metric in values_gt:
                    x_val = values_gt[metric]
                    ax.axvline(x=x_val, color='gray', linestyle='-', linewidth=0.8)
                    ax.text(x_val + 0.08, text_y_positions[metric], label, color='gray', va='center', ha='left')
                        
            # V_95
            ci_key = next((k for k in values_gt if k.startswith('CI')), None)
            y_val_ct = values_gt[ci_key]*100 # percentage
            ax.axhline(y=y_val_ct, color='gray', linestyle='-', linewidth=0.8)
        '''
        else: # OARs
            for metric, label in [('D_98', 'D98%')]:
                if metric in values_gt and metric in ['D_2','mean']:
                    hn_metrics_ct[metric].append(values_gt[metric])
        '''
            
        # HN
        if name in ["tongue", "hard_palate", "soft_palate"]: # target
            for metric in hn_metrics_ct.keys():
                if metric in values_gt and metric=='D_98':
                    hn_metrics_ct[metric].append(values_gt[metric])
                    
            # just for V95%
            print(values_gt)
            for key in values_gt:
                if key.startswith('CI'):
                    if key not in hn_metrics_ct:
                        hn_metrics_ct[key] = []
                    hn_metrics_ct[key].append(values_gt[key])
                    text_y_positions[key]=-3
                    metric_labels_hn[key]='V95% (mean)'
        '''
        else: # OAR
            for metric, label in [('D_2', 'D2%'), ('mean','Dmean')]:
                if metric in values_gt:
                    x_val_oars = values_gt[metric]
                    ax.axvline(x=x_val_oars, color='gray', linestyle='--', linewidth=0.8)
        '''
            
                


        # sCT DVH
        pred_segment = pred_dvh[seg]
        x_pred = pred_segment['doseGrid']
        y_pred = pred_segment['volumePoints']
        values_pred = pred_organs_dict.get(name)
        ax.plot(x_pred, y_pred, linestyle='--', linewidth=2, color=color, label=f'{format_name(name)}', zorder=3)

        # AB, TH
        if values_pred is not None and name in ["lung_upper_lobe_right", "stomach"]:
            for metric, label in [('D_98', 'D98%')]:
                if metric in values_pred:
                    x_val = values_pred[metric]
                    ax.axvline(x=x_val, color='gray', linestyle='--', linewidth=0.8)
                    
            # V_95
            ci_key = next((k for k in values_pred if k.startswith('CI')), None)
            y_val_sct = values_pred[ci_key]*100 # percentage
            ax.axhline(y=y_val_sct, color='gray', linestyle='--', linewidth=0.8)
            ax.text(-0.003, y_val_sct + 0.3, 'V95%', color='gray', va='bottom', ha='right')
        '''
        else: # OARs
            for metric, label in [('D_2', 'D2%'), ('mean','Dmean')]:
                if metric in values_pred:
                    x_val_oars = values_pred[metric]
                    ax.axvline(x=x_val_oars, color='gray', linestyle='--', linewidth=0.8)
        '''
                
        # HN   
        if name in ["tongue", "hard_palate", "soft_palate"]:  # target
            for metric in hn_metrics_sct.keys():
                if metric in values_pred and metric=='D_98':
                    hn_metrics_sct[metric].append(values_pred[metric])
                    
            # just for V95%
            for key in values_pred:
                if key.startswith('CI'):
                    if key not in hn_metrics_sct:
                        hn_metrics_sct[key] = []
                    hn_metrics_sct[key].append(values_pred[key])
                    text_y_positions[key]=-3
                    metric_labels_hn[key]='V95% (mean)'
        '''            
        else: # OARS
            for metric in hn_metrics_sct.keys():
                if metric in values_pred and metric in [('D_2','mean')]:
                    hn_metrics_sct[metric].append(values_pred[metric])
        '''
    
    for metric, values in hn_metrics_ct.items():
        if metric not in ['D_98','D_2','mean']: # CI
            y_val_ct_hn = np.mean(values)*100
            print(y_val_ct_hn)
            ax.axhline(y=y_val_ct_hn, color='gray', linestyle='-', linewidth=0.8)
        else:
            mean_val = np.mean(values)
            ax.axvline(x=mean_val, color='gray', linestyle='-', linewidth=0.8)
            if metric=='mean':
                ax.text(mean_val - 0.05, text_y_positions[metric], metric_labels_hn[metric], color='gray', va='center', ha='right')
            else:    
                ax.text(mean_val + 0.05, text_y_positions[metric], metric_labels_hn[metric], color='gray', va='center', ha='left')
    
        
    for metric, values in hn_metrics_sct.items():
        if metric not in ['D_98','D_2','mean']: # CI
            y_val_sct_hn = np.mean(values)*100
            print(y_val_sct_hn)
            ax.axhline(y=y_val_sct_hn, color='gray', linestyle='--', linewidth=0.8)
            ax.text(2.63, y_val_sct_hn + 0.3, metric_labels_hn[metric], color='gray', va='bottom', ha='right')
        else:
            mean_val = np.mean(values)
            ax.axvline(x=mean_val, color='gray', linestyle='--', linewidth=0.8)
            
    
      
            
     # Create organ legend (one entry per organ)
    unique_names = []
    unique_colors = []
    for name in name_list:
        if name not in unique_names:
            unique_names.append(name)
            unique_colors.append(color_map.get(name.lower().strip(), '#000000'))
   
    segment_legend = [
        Line2D([0], [0], color=color, lw=2, label=format_name(name))
    for name, color in zip(unique_names, unique_colors)
    ]

    style_legend = [
        Line2D([0], [0], color='k', lw=2, linestyle='-', label='CT'),
        Line2D([0], [0], color='k', lw=2, linestyle='--', label='sCT')
    ]
    
    
    first_legend = ax.legend(
        handles=segment_legend,
        loc='center left',
        bbox_to_anchor=(1, 0.5)
    )
    ax.add_artist(first_legend)

    
    ax.legend(
            handles=style_legend,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.13),
            fancybox=True, 
            shadow=True,
            ncol=2
        )
    
    ax.set_xlabel('Dose per fraction [Gy]', fontsize=12)
    ax.set_ylabel('Volume [%]', fontsize=12)
    plt.tight_layout()
    
    # Define os limites da figura com espaço extra à direita
    fig = plt.gcf()
    original_bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    padded_bbox = Bbox([
        [original_bbox.x0, original_bbox.y0-0.7],
        [original_bbox.x1 + 3, original_bbox.y1+1]  # +2 polegadas à direita
    ])
    
    plt.savefig(
                f"dvh_{best_patient_id}.png",
                dpi=200,
                bbox_inches=padded_bbox,
    )


best_hn_id, best_th_id, best_ab_id = best_patient(all_metrics_data)

plot_dvh(best_ab_id, data_path)
plot_dvh(best_hn_id, data_path)
plot_dvh(best_th_id, data_path)
plot_dvh("1THA028", data_path)


