import json
import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib.lines import Line2D

# From docker function: segmentation_metrics.py 
classes_to_use = {
            "AB": [
                2, # kidney right
                3, # kidney left
                5, # liver
                6, # stomach
                *range(10, 14+1), #lungs
                *range(26, 50+1), #vertebrae
                51, #heart
                79, # spinal cord
                *range(92, 115+1), # ribs
                116 #sternum
            ],
            "HN": [
                15, # esophagus
                16, # trachea
                17, # thyroid
                *range(26, 50+1), #vertebrae
                79, #spinal cord
                90, # brain
                91, # skull
            ],
            "TH": [
                2, # kidney right
                3, # kidney left
                5, # liver
                6, # stomach
                *range(10, 14+1), #lungs
                *range(26, 50+1), #vertebrae
                51, #heart
                79, # spinal cord
                *range(92, 115+1), # ribs
                116 #sternum
            ]
        }

class_labels = {
    2: "kidney_right",
    3: "kidney_left",
    5: "liver",
    6: "stomach",
    10: "lung_upper_lobe_left",
    11: "lung_lower_lobe_left",
    12: "lung_upper_lobe_right",
    13: "lung_middle_lobe_right",
    14: "lung_lower_lobe_right",
    15: "esophagus",
    16: "trachea",
    17: "thyroid_gland",
    26: "vertebrae_S1",
    27: "vertebrae_L5",
    28: "vertebrae_L4",
    29: "vertebrae_L3",
    30: "vertebrae_L2",
    31: "vertebrae_L1",
    32: "vertebrae_T12",
    33: "vertebrae_T11",
    34: "vertebrae_T10",
    35: "vertebrae_T9",
    36: "vertebrae_T8",
    37: "vertebrae_T7",
    38: "vertebrae_T6",
    39: "vertebrae_T5",
    40: "vertebrae_T4",
    41: "vertebrae_T3",
    42: "vertebrae_T2",
    43: "vertebrae_T1",
    44: "vertebrae_C7",
    45: "vertebrae_C6",
    46: "vertebrae_C5",
    47: "vertebrae_C4",
    48: "vertebrae_C3",
    49: "vertebrae_C2",
    50: "vertebrae_C1",
    51: "heart",
    79: "spinal_cord",
    90: "brain",
    91: "skull",
    92: "rib_left_1",
    93: "rib_left_2",
    94: "rib_left_3",
    95: "rib_left_4",
    96: "rib_left_5",
    97: "rib_left_6",
    98: "rib_left_7",
    99: "rib_left_8",
    100: "rib_left_9",
    101: "rib_left_10",
    102: "rib_left_11",
    103: "rib_left_12",
    104: "rib_right_1",
    105: "rib_right_2",
    106: "rib_right_3",
    107: "rib_right_4",
    108: "rib_right_5",
    109: "rib_right_6",
    110: "rib_right_7",
    111: "rib_right_8",
    112: "rib_right_9",
    113: "rib_right_10",
    114: "rib_right_11",
    115: "rib_right_12",
    116: "sternum"
}

def load_json_metrics(json_path):
    with open(json_path) as f:
        data = json.load(f)
    
    average_metrics = data['aggregates']
    
    # Separate dictionaries for each metric type
    dice_metrics = {}
    hd95_metrics = {}
    
    # Extract DICE and HD95 metrics separately
    for k, v in average_metrics.items():
        if k.startswith("DICE"):
            class_id = k.replace("DICE_class_", "")
            dice_metrics[class_id] = {
                "mean": v["mean"], 
                "std": v["std"],
                "count": v["count"]
            }
        elif k.startswith("HD95"):
            class_id = k.replace("HD95_class_", "")
            hd95_metrics[class_id] = {
                "mean": v["mean"], 
                "std": v["std"],
                "count": v["count"]
            }
    #print(dice_metrics)
    
    # Combine metrics for classes that have both
    data = []
    for class_id in dice_metrics.keys():
        if class_id in hd95_metrics:
            dice_mean = dice_metrics[class_id]["mean"]
            dice_count = dice_metrics[class_id]["count"]
            hd95_mean = hd95_metrics[class_id]["mean"]
            hd95_count = hd95_metrics[class_id]["count"]
            
            if isinstance(dice_mean, (float, int)) and isinstance(hd95_mean, (float, int)):
                data.append({
                    "class": class_id,
                    "dice_mean": dice_mean,
                    "dice_count": dice_count,
                    "hd95_mean": hd95_mean,
                    "hd95_count": hd95_count,
                })
    data_df = pd.DataFrame(data)
    data_df.to_csv(f"out_metrics_per_class_{json_path.split("/")[-2]}", index=False)
    
    return data_df



def combine_metrics_with_labels(df, class_labels):

    # Map using integer keys
    df['class_name'] = df['class'].astype(int).map(class_labels)
    print(df)
    
    return df



def plot_dice_vs_hd95(df):
    
    plt.figure(figsize=(12,5))
    region = df["region"].iloc[0]
    
    color_map = {
        'Lung (upper lobe left)': '#640202',    
        'Lung (lower lobe left)': '#A00A0A',   
        'Lung (upper lobe right)': '#DF0000',  
        'Lung (middle lobe right)': '#F76464',    
        'Lung (lower lobe right)': '#F19C9C',    
        
        'Kidney (left)': '#006400',             
        'Kidney (right)': '#4CF04C',       
        
        'Bone (all vertebrae)': "#0000FF",     
        'Bone (all ribs)': "#43A5F5E8",           
        'Bone (sternum)': "#9BBEFF",            
        'Bone (skull)': "#C7D8FA",
        
        'Spinal cord': "#834FFD",
        'Esophagus': "#CA86F7",
        'Trachea': "#00CED1",
        'Brain': "#380000",
        'Thyroid': "#E4FF98",
        'Heart': "#FF5E00",
        'Stomach': "#FF0077",
        'Liver': "#FFEE00",
        }
    
    label_to_group = {
        'lung_upper_lobe_left': 'Lung (upper lobe left)',
        'lung_lower_lobe_left': 'Lung (lower lobe left)',
        'lung_upper_lobe_right': 'Lung (upper lobe right)',
        'lung_middle_lobe_right': 'Lung (middle lobe right)',
        'lung_lower_lobe_right': 'Lung (lower lobe right)',
        'kidney_left': 'Kidney (left)',
        'kidney_right': 'Kidney (right)',
        'liver': 'Liver',
        'stomach': 'Stomach',
        'heart': 'Heart',
        'vertebrae_S1': 'Bone (all vertebrae)',
        'vertebrae_L5': 'Bone (all vertebrae)',
        'vertebrae_L4': 'Bone (all vertebrae)',
        'vertebrae_L3': 'Bone (all vertebrae)',
        'vertebrae_L2': 'Bone (all vertebrae)',
        'vertebrae_L1': 'Bone (all vertebrae)',
        'vertebrae_T12': 'Bone (all vertebrae)',
        'vertebrae_T11': 'Bone (all vertebrae)',
        'vertebrae_T10': 'Bone (all vertebrae)',
        'vertebrae_T9': 'Bone (all vertebrae)',
        'vertebrae_T8': 'Bone (all vertebrae)',
        'vertebrae_T7': 'Bone (all vertebrae)',
        'vertebrae_T6': 'Bone (all vertebrae)',
        'vertebrae_T5': 'Bone (all vertebrae)',
        'vertebrae_T4': 'Bone (all vertebrae)',
        'vertebrae_T3': 'Bone (all vertebrae)',
        'vertebrae_T2': 'Bone (all vertebrae)',
        'vertebrae_T1': 'Bone (all vertebrae)',
        'vertebrae_C7': 'Bone (all vertebrae)',
        'vertebrae_C6': 'Bone (all vertebrae)',
        'vertebrae_C5': 'Bone (all vertebrae)',
        'vertebrae_C4': 'Bone (all vertebrae)',
        'vertebrae_C3': 'Bone (all vertebrae)',
        'vertebrae_C2': 'Bone (all vertebrae)',
        'vertebrae_C1': 'Bone (all vertebrae)',
        'rib_left_1': 'Bone (all ribs)',
        'rib_left_2': 'Bone (all ribs)',
        'rib_left_3': 'Bone (all ribs)',
        'rib_left_4': 'Bone (all ribs)',
        'rib_left_5': 'Bone (all ribs)',
        'rib_left_6': 'Bone (all ribs)',
        'rib_left_7': 'Bone (all ribs)',
        'rib_left_8': 'Bone (all ribs)',
        'rib_left_9': 'Bone (all ribs)',
        'rib_left_10': 'Bone (all ribs)',
        'rib_left_11': 'Bone (all ribs)',
        'rib_left_12': 'Bone (all ribs)',
        'rib_right_1': 'Bone (all ribs)',
        'rib_right_2': 'Bone (all ribs)',
        'rib_right_3': 'Bone (all ribs)',
        'rib_right_4': 'Bone (all ribs)',
        'rib_right_5': 'Bone (all ribs)',
        'rib_right_6': 'Bone (all ribs)',
        'rib_right_7': 'Bone (all ribs)',
        'rib_right_8': 'Bone (all ribs)',
        'rib_right_9': 'Bone (all ribs)',
        'rib_right_10': 'Bone (all ribs)',
        'rib_right_11': 'Bone (all ribs)',
        'rib_right_12': 'Bone (all ribs)',
        'sternum': 'Bone (sternum)',
        'skull': 'Bone (skull)',
        'spinal_cord': 'Spinal cord',
        'esophagus': 'Esophagus',
        'trachea': 'Trachea',
        'brain': 'Brain',
        'thyroid_gland': 'Thyroid',
    }
    
    df['group'] = df['class_name'].map(label_to_group)
    df['color'] = df['group'].map(color_map)
    

    plt.scatter(
        df['dice_mean'],
        df['hd95_mean'],
        c=df['color'],
        s=200,
        edgecolors='k',
        alpha=0.9,
        linewidths=1,
        zorder=3
    )

    # Show in legend just the segments present in df
    print(df)
    df_without_nan = df.dropna()
    used_groups = df_without_nan['group'].unique()
    #print(used_groups)
    
    rows_with_nan = df[df.isna().any(axis=1)]
    groups_with_nan = rows_with_nan['class_name'].unique()
    print(groups_with_nan)

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', label=k,
            markerfacecolor=color_map[k], markeredgecolor='k', markersize=10)
        for k in sorted(used_groups)
    ]

    plt.legend(
        handles=legend_handles,
        bbox_to_anchor=(1, 0.5),
        loc='center left',
        ncol=1,
        fontsize=13
    )

    plt.xlabel('Mean mDice', fontsize=14)
    plt.ylabel('Mean HD95 [mm]', fontsize=14)
    plt.grid(True)
        

    plt.tight_layout()
    plt.savefig(
        f"dice_vs_hd95_{region}.png",
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.5
    )


def plot_region_barplot_simple(df_ab, df_th, df_hn):
    data = []

    bone_structures = [
        'vertebrae_S1', 'vertebrae_L5', 'vertebrae_L4', 'vertebrae_L3',
        'vertebrae_L2', 'vertebrae_L1', 'vertebrae_T12', 'vertebrae_T11',
        'vertebrae_T10', 'vertebrae_T9', 'vertebrae_T8', 'vertebrae_T7',
        'vertebrae_T6', 'vertebrae_T5', 'vertebrae_T4', 'vertebrae_T3',
        'vertebrae_T2', 'vertebrae_T1', 'vertebrae_C7', 'vertebrae_C6',
        'vertebrae_C5', 'vertebrae_C4', 'vertebrae_C3', 'vertebrae_C2',
        'vertebrae_C1', 'rib_left_1', 'rib_left_2', 'rib_left_3', 'rib_left_4',
        'rib_left_5', 'rib_left_6', 'rib_left_7', 'rib_left_8', 'rib_left_9',
        'rib_left_10', 'rib_left_11', 'rib_left_12', 'rib_right_1',
        'rib_right_2', 'rib_right_3', 'rib_right_4', 'rib_right_5',
        'rib_right_6', 'rib_right_7', 'rib_right_8', 'rib_right_9',
        'rib_right_10', 'rib_right_11', 'rib_right_12', 'sternum', 'skull',
    ]

    def process_region(df_region, region_name):
        bone_df = df_region[df_region['class_name'].isin(bone_structures)]
        soft_df = df_region[~df_region['class_name'].isin(bone_structures)]

        if not bone_df.empty:
            data.append({
                "region": region_name,
                "structure_type": "Bones",
                "dice_mean": bone_df['dice_mean'].mean(),
                "hd95_mean": bone_df['hd95_mean'].mean(),
                "count": bone_df['dice_mean'].count()
            })

        if not soft_df.empty:
            data.append({
                "region": region_name,
                "structure_type": "Soft Tissues",
                "dice_mean": soft_df['dice_mean'].mean(),
                "hd95_mean": soft_df['hd95_mean'].mean()
            })

    # Process all regions
    process_region(df_ab, "AB")
    process_region(df_th, "TH")
    process_region(df_hn, "HN")

    df_plot = pd.DataFrame(data)
    print(f"{df_plot}")

    df_plot["region"] = pd.Categorical(df_plot["region"], categories=["HN", "TH", "AB"], ordered=True)
    df_plot = df_plot.sort_values(["region", "structure_type"])

    # Plot DICE
    plt.figure(figsize=(10, 6))
    for i, region in enumerate(["HN", "TH", "AB"]):
        sub_df = df_plot[df_plot['region'] == region]
        plt.barh(
            y=[i - 0.15, i + 0.15],  # offset for grouped bars
            width=sub_df["dice_mean"],
            height=0.25,
            color=["seagreen", "lightgray"],
            label=sub_df["structure_type"].values if i == 0 else ["", ""],
            zorder=2
        )
    plt.yticks([0, 1, 2], ["HN", "TH", "AB"], fontsize=13)
    plt.xlabel("Mean mDice", fontsize=14)
    plt.grid(True, axis="x", linestyle="--", alpha=0.5, zorder=1)
    plt.legend()
    plt.tight_layout()
    plt.savefig("grouped_mean_dice_metric.png", dpi=150)

    # Plot HD95
    plt.figure(figsize=(10, 6))
    for i, region in enumerate(["HN", "TH", "AB"]):
        sub_df = df_plot[df_plot['region'] == region]
        plt.barh(
            y=[i - 0.15, i + 0.15],
            width=sub_df["hd95_mean"],
            height=0.25,
            color=["darkorange", "lightgray"],
            label=sub_df["structure_type"].values if i == 0 else ["", ""],
            zorder=2
        )
    plt.yticks([0, 1, 2], ["HN", "TH", "AB"], fontsize=13)
    plt.xlabel("Mean HD95 [mm]", fontsize=14)
    plt.grid(True, axis="x", linestyle="--", alpha=0.5, zorder=1)
    plt.legend()
    plt.tight_layout()
    plt.savefig("grouped_mean_hd95_metric.png", dpi=150)


def main():
    
    df_ab = load_json_metrics('/home/catarina_caldeira/Desktop/evaluation/output_ab/metrics.json')
    df_ab['region'] = 'AB'
    print(df_ab)

    df_th = load_json_metrics('/home/catarina_caldeira/Desktop/evaluation/output_th/metrics.json')
    df_th['region'] = 'TH'
    #print(df_th)
    
    df_hn = load_json_metrics('/home/catarina_caldeira/Desktop/evaluation/output_hn/metrics.json')
    df_hn['region'] = 'HN'
    #print(df_hn)
    

    # Per segments
    df_ab_combined = combine_metrics_with_labels(df_ab, class_labels)
    #print(df_ab_combined)
    plot_dice_vs_hd95(df_ab_combined)
    
    
    df_th_combined = combine_metrics_with_labels(df_th, class_labels)
    #print(df_th_combined)
    plot_dice_vs_hd95(df_th_combined)

    

    df_hn_combined = combine_metrics_with_labels(df_hn, class_labels)
    #print(df_hn_combined)
    plot_dice_vs_hd95(df_hn_combined)
    
    
    # Per region
    plot_region_barplot_simple(df_ab_combined,df_th_combined,df_hn_combined)
    
    

if __name__ == "__main__":
    main()   
    
    
