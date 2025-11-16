import numpy as np
import json
import os
import matplotlib.pyplot as plt
import scipy
from dose_calculations import DoseMetrics
from dvh_graph import best_patient

data_path = "\\userdata\\patients\\"
metrics_path = 'all_metrics.json'

with open(metrics_path, 'r') as f:
    all_metrics_data = json.load(f)
    #print(all_metrics_data)


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

def stats(lst):
        return (np.mean(lst), np.std(lst)) if len(lst) > 0 else (np.nan, np.nan)
    
def mean_per_region(all_metrics_data, metric):
    ab_centers = {"A": [], "B": [], "C": []}
    th_centers = {"A": [], "B": []}
    hn_centers = {"A": [], "C": []}

    for patient_id, metrics in all_metrics_data.items():
        region = patient_id[1:3]   
        center = patient_id[3:4]  
        patient_path = os.path.join(data_path, patient_id)

        if os.path.isdir(patient_path) and patient_id != ".venv":
            if metric in metrics:
                value = metrics[metric]

                if region == "AB" and center in ab_centers:
                    ab_centers[center].append(value)

                elif region == "TH" and patient_id != "1THA028" and center in th_centers:
                    th_centers[center].append(value)

                elif region == "HN" and center in hn_centers:
                    hn_centers[center].append(value)

    print(f"===== {metric} =====")

    print("PER REGION:")
    mean_ab_region = np.mean([v for va in ab_centers.values() for v in va])
    print(f"AB region mean: {mean_ab_region}")
    
    mean_th_region = np.mean([v for va in th_centers.values() for v in va])
    print(f"TH region mean: {mean_th_region}")
    
    mean_hn_region = np.mean([v for va in hn_centers.values() for v in va])
    print(f"HN region mean: {mean_hn_region}")
    
    print(" AB por centro:")
    for c, vals in ab_centers.items():
        m, s = stats(vals)
        print(f"  Centro {c}: Média = {m:.3f}, DP = {s:.3f}")

    print("\n TH por centro:")
    for c, vals in th_centers.items():
        m, s = stats(vals)
        print(f"  Centro {c}: Média = {m:.3f}, DP = {s:.3f}")

    print("\n HN por centro:")
    for c, vals in hn_centers.items():
        m, s = stats(vals)
        print(f"  Centro {c}: Média = {m:.3f}, DP = {s:.3f}")

    return ab_centers, th_centers, hn_centers


def plot_gamma_index_hist(data_path, best_patient_id):
    
    dm = DoseMetrics(data_path)
    
    for patient_folder in os.listdir(data_path):
        #print(patient_folder)
        patient_path = os.path.join(data_path, patient_folder)
        #print(patient_path)
        if os.path.isdir(patient_path) and patient_folder==best_patient_id:
            mat_file = os.path.join(patient_path, f'{patient_folder}_ct_and_sct_plan.mat')
            mat = dm.load_mat_file(mat_file)
            gamma_index = mat['gammaCube']
            gamma_index_values = gamma_index[gamma_index>0]
            gamma_test_pass = gamma_index_values[gamma_index_values<=1]
            gamma_test_fail = gamma_index_values[gamma_index_values>1]

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.hist(gamma_test_pass, color='green', edgecolor='black', alpha=0.7, label='γ ≤ 1')
            ax.hist(gamma_test_fail, color='red', edgecolor='black', alpha=0.7, label='γ > 1')
            
            ax.axvline(x=1, color='gray', linestyle='--', linewidth=0.8)
            ax.set_xlim([0, 7])
            ax.set_xlabel("Gamma index")
            ax.set_yscale("log")
            ax.set_ylabel("Frequency (log scale)")
            ax.legend()
            
            plt.savefig(
                f"gamma_index_histogram_{best_patient_id}.png",
                dpi=200,
                bbox_inches="tight"
    )


   
def plot_mean_gamma_pass_rates(data, metric_y):
    
    regions = ["AB", "TH", "HN"]
    colors = ['#1a80bb','#b8b8b8', '#a00000']
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    positions = np.arange(1, len(data) + 1)
    parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)

    
    for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_edgecolor('black')
            pc.set_alpha(1)

    for i, vals in enumerate(data):
            quartile1, medians, quartile3 = np.percentile(vals, [25, 50, 75])
            whiskers = np.array([
                  adjacent_values(vals, quartile1, quartile3)])
            whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]
            
            ax.scatter(positions[i], medians, marker='o', color='white', s=30, zorder=3)
            ax.vlines(positions[i], quartile1, quartile3, color='k', linestyle='-', lw=5)
            ax.vlines(positions[i], whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)
            print("Q1:", quartile1)
            print("Q3:", quartile3)
            print("Upper whiskers:", whiskersMax)
            print("Lower whiskers:", whiskersMin)
            
            ax.hlines(medians, positions[i] - 0.25, positions[i] + 0.25, colors='black', linewidth=1, linestyles='--')

      
    ax.set_xlabel("Region",fontsize=12)
    ax.set_ylabel(metric_y, fontsize=12)
    ax.set_xticks(positions, regions)
    plt.subplots_adjust(bottom=0.15, wspace=0.05)
    plt.savefig(
                f"violin_plot_{metric_y}.png",
                dpi=200,
                bbox_inches="tight"
    )
     
     
#best_hn_id, best_th_id, best_ab_id = best_patient(all_metrics_data)
#plot_gamma_index_hist(data_path, best_hn_id)
#plot_gamma_index_hist(data_path,best_th_id)
#plot_gamma_index_hist(data_path,best_ab_id)


ab_centers_mae, th_centers_mae, hn_centers_mae = mean_per_region(all_metrics_data, "MAE")
ab_centers_dvh, th_centers_dvh, hn_centers_dvh = mean_per_region(all_metrics_data, "DVH")
ab_centers_gamma, th_centers_gamma, hn_centers_gamma = mean_per_region(all_metrics_data, "Gamma pass rate")

'''
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


regions = ["AB", "TH", "HN"]
for i, region_data in enumerate(data_gamma):
    media, inf, sup = mean_confidence_interval(region_data)
    print(f"Região {regions[i]} - Gamma - Média: {media:.2f}, Intervalo de confiança 95%: [{inf:.2f}, {sup:.2f}]")


plot_mean_gamma_pass_rates(data_mae, "MAE [Gy]")
plot_mean_gamma_pass_rates(data_dvh, "DVH")
plot_mean_gamma_pass_rates(data_gamma, "Gamma pass rate [%]")

'''