import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

def heatmap_plot(data_best, text, model_type, dataset_type):
    metrics = ['MAE (HU)', 'PSNR (dB)', 'SSIM', 'MS-SSIM']
    models = ['Fold 1 vs Ensemble', 'Fold 2 vs Ensemble', 'Fold 3 vs Ensemble', 'Fold 4 vs Ensemble', 'Fold 5 vs Ensemble']
    
    norm = TwoSlopeNorm(vmin=-data_best.max(), vcenter=0, vmax=data_best.max())

    fig,ax = plt.subplots(figsize=(7,5))
    sns.heatmap(data_best, annot=True, 
                annot_kws={"size": 11}, 
                fmt=".4f", 
                cmap=sns.diverging_palette(215, 5, as_cmap=True, center='light'),
                xticklabels=metrics, yticklabels=models, linewidth=.5, 
                norm=norm, ax=ax)

    for i in range(data_best.shape[0]): #linhas
        for j in range(data_best.shape[1]): # colunas
            ax.text(j + 0.5, i + 0.8, text[i,j],
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=8,
                    color='black')

    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Models', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"utils/images_thesis/heatmaps/heatmap_{model_type}_{dataset_type}.png", dpi=200, bbox_inches="tight") 


# https://www.geeksforgeeks.org/seaborn-diverging_palette-method/

# AE
dataset_type = "23"
model_type = "ae"
data_diff = np.array([ 
[5.1095, -0.5163, -0.0048, -0.0071],  # Fold 0 differences (MAE, PSNR, SSIM, MS-SSIM)
[4.5871, -0.5322, -0.0049, -0.0060],  # Fold 1
[2.1853, -0.2880, -0.0045, -0.0049],  # Fold 2
[0.8591, -0.2094, -0.0033, -0.0031],  # Fold 3
[2.5141, -0.3709, -0.0053, -0.0061],  # Fold 4
])

# p-value is from two sided
text_pvalue = np.array([['(p<0.0001)*', '(p<0.0001)*', '(p<0.0001)*', '(p<0.0001)*'],  # Fold 0 (MAE, PSNR, SSIM, MS-SSIM)
                 ['(p<0.0001)*', '(p<0.0001)*', '(p=0.0001)*', '(p=0.0001)*'], 
                 ['(p=0.0012)*', '(p=0.0003)*', '(p<0.0001)*', '(p<0.0001)*'],
                 ['(p=0.0920)', '(p=0.0001)*', '(p<0.0001)*', '(p<0.0001)*'],
                 ['(p=0.0002)*', '(p=0.0001)*', '(p<0.0001)*', '(p<0.0001)*']]) 

heatmap_plot(data_diff, text_pvalue, model_type, dataset_type)


# cGAN
dataset_type = "23"
model_type = "cgan"
data_diff = np.array([ 
[3.8105, -0.8709, -0.0211, -0.0206],   # Fold 0 differences (MAE, PSNR, SSIM, MS-SSIM)
[-2.1630, -0.4491, -0.0042, -0.0074],   # Fold 1
[1.0080, -0.5160, -0.0041, -0.0076],  # Fold 2
[0.5166, -0.4940, -0.0071, -0.0044],   # Fold 3
[32.1495, -2.7302, -0.0186, -0.0352],  # Fold 4
])

text_pvalue = np.array([['(p=0.0009)*', '(p<0.0001)*', '(p<0.0001)*', '(p<0.0001)*'],  # Fold 0 (MAE, PSNR, SSIM, MS-SSIM)
                 ['(p=0.0421)*', '(p=0.0002)*', '(p<0.0001)*', '(p<0.0001)*'], 
                 ['(p=0.3523)', '(p=0.0001)*', '(p=0.0002)*', '(p<0.0001)*'],
                 ['(p=0.5723)', '(p<0.0001)*', '(p<0.0001)*', '(p=0.0030)*'],
                 ['(p<0.0001)*', '(p<0.0001)*', '(p<0.0001)*', '(p<0.0001)*']]) 


heatmap_plot(data_diff, text_pvalue, model_type, dataset_type)