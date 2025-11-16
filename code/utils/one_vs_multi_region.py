import os
import json
import matplotlib.pyplot as plt
from scipy import stats

def print_average_metrics_with_ttest(rs_dict, mr_dict, metrics):
    print("### AVERAGES AND PAIRED T-TEST ###\n")
    for metric in metrics:
        print(f"--- METRIC: {metric.upper()} ---")
        for region in rs_dict.keys():
            rs_vals = rs_dict[region][metric]
            mr_vals = mr_dict[region][metric]

            # Check if lengths match
            if len(rs_vals) != len(mr_vals):
                print(f"Error: Different lengths for {region} in {metric}")
                continue

            # Compute means
            rs_mean = sum(rs_vals) / len(rs_vals)
            mr_mean = sum(mr_vals) / len(mr_vals)

            # Paired t-test
            result = stats.ttest_rel(rs_vals, mr_vals)

            # Print results
            print(f"Region: {region}")
            print(f"  Region-specific: {rs_mean:.4f}")
            print(f"  Multi-region:    {mr_mean:.4f}")
            print(f"  Paired t-test p-value: {result.pvalue}")

            if result.pvalue > 0.05:
                print("No evidence to reject H0 (difference not significant)")
            else:
                print("Evidence to reject H0 (difference significant)")

        print()


def plot_metrics_all_in_one(rs_dict, mr_dict, metrics):

    colors = {
        'AB': '#e41a1c',  # vermelho
        'TH': '#377eb8',  # azul
        'HN': '#4daf4a',  # verde
        'HN (D)': "#ffef3d",  
    }
    marker_mr = 'o'
    marker_rs = '^'
    marker_size = 200
    border_width = 1
    

    for idx, metric in enumerate(metrics):
        fig, ax = plt.subplots(1, 1, figsize=(10,8))

        vals = []
        for region in rs_dict:
            vals += rs_dict[region][metric] + mr_dict[region][metric]
            #print(vals)
        vmin, vmax = min(vals), max(vals)
        ax.plot([vmin, vmax], [vmin, vmax], 'k--', lw=1, zorder=3)

        for region, color in colors.items():
            rs_vals = rs_dict[region][metric]
            #print(rs_vals)
            mr_vals = mr_dict[region][metric]
            #print(f"mr: {mr_vals}")

            for rs_val, mr_val in zip(rs_vals, mr_vals):
                ax.scatter(
                    rs_val, mr_val,
                    color=color,
                    marker=marker_mr,
                    s=marker_size,
                    edgecolor='black',
                    linewidths=border_width,
                    alpha=0.8,
                    zorder=0
                )


        #if metric=='msssim':
        #    ax.set_title("MS-SSIM", fontsize=12)
        #else:
        #    ax.set_title(metric.upper(), fontsize=12)
        ax.set_xlabel('Region-specific', fontsize=12)
        ax.set_ylabel('Multi-region', fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.5)

        # Legenda global (sem duplicação)
        import matplotlib.lines as mlines
        import matplotlib.patches as mpatches

        circle = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=10, label='Trained Region-specific vs. Multi-region')

        color_patches = [
            mpatches.Patch(color='#e41a1c', label='Tested in AB'),
            mpatches.Patch(color='#377eb8', label='Tested in TH'),
            mpatches.Patch(color='#4daf4a', label='Tested in HN'),
            mpatches.Patch(color='#ffef3d', label='Tested in HN (D)'),
        ]

        # Combine handles and labels
        handles = [circle] + color_patches

        # Add combined legend below all subplots
        ax.legend(handles=handles, loc='lower right', fontsize=12)

        plt.tight_layout()
        save_folder = "/home/catarina_caldeira/Desktop/code/utils/images_thesis/comparison_one_vs_multi_regions"
        os.makedirs(save_folder, exist_ok=True)
        filename = os.path.join(save_folder, f"comparison_{metric}.png")
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

tested_one_ab="/home/catarina_caldeira/Desktop/code/results/ensemble_8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionAB_tested_AB.json"
tested_one_hn="/home/catarina_caldeira/Desktop/code/results/ensemble_8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionHN_tested_HN.json"
tested_one_th="/home/catarina_caldeira/Desktop/code/results/ensemble_8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionTH_tested_TH.json"
tested_one_hnd="/home/catarina_caldeira/Desktop/code/results/ensemble_8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionHN_tested_HND.json"

tested_multi_ab="/home/catarina_caldeira/Desktop/code/results/ensemble_8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionABHNTH_tested_AB.json"
tested_multi_hn="/home/catarina_caldeira/Desktop/code/results/ensemble_8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionABHNTH_tested_HN.json"
tested_multi_th="/home/catarina_caldeira/Desktop/code/results/ensemble_8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionABHNTH_tested_TH.json"
tested_multi_hnd="/home/catarina_caldeira/Desktop/code/results/ensemble_8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionABHNTH_tested_HND.json"


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

rs_dict = {
    'AB': load_json(tested_one_ab),
    'TH': load_json(tested_one_th),
    'HN': load_json(tested_one_hn),
    'HN (D)': load_json(tested_one_hnd),
}

mr_dict = {
    'AB': load_json(tested_multi_ab),
    'TH': load_json(tested_multi_th),
    'HN': load_json(tested_multi_hn),
    'HN (D)': load_json(tested_multi_hnd),
}
metrics = ["psnr", "mae", "ssim", "msssim"]

print_average_metrics_with_ttest(rs_dict, mr_dict, metrics)
plot_metrics_all_in_one(rs_dict, mr_dict, metrics)
 


# python -m utils.one_vs_multi_region