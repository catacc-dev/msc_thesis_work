import matplotlib.pyplot as plt  
import numpy as np

'''
# Data AE (VAL - mean of each fold mean)
stages = ["Learning rate", "Number of layers", "Normalization type", "Training batch size", "Epochs"]  
psnr_values = [26.1186, 26.2066, 26.2429, 27.1197, 28.3591]
mae_values = [103.7187, 102.0999, 101.9525, 86.7862, 69.5388]
ssim_values = [0.8769, 0.8795, 0.8841, 0.9035, 0.9204]
ms_ssim_values = [0.8144, 0.8143, 0.8153, 0.8509, 0.8900] 
'''


# Data cGAN (VAL)

stages = ["Learning rate\n(G=D)", "Number\nof layers", "Normalization\ntype", "Training\nbatch size", 
          "Kernel size\n(D)", "Learning \nrate (D)", "Perceptual \nloss weight \n(λ)", 
          "One-sided \nlabel \nsmoothing (D)", "Epochs"]

psnr_values = [25.3274, 25.5256, 26.1676, 26.7804, 26.9944, 27.1197, 27.1075, 27.1091, 27.8002]
mae_values = [130.6231, 123.0122, 112.7216, 98.2191, 93.4577, 91.4154, 91.2948, 91.2199, 79.1158]
ssim_values = [0.7903, 0.8196, 0.8450, 0.8683, 0.8748, 0.8876, 0.8907, 0.8913, 0.9128]
ms_ssim_values = [0.7884, 0.7972, 0.8188, 0.8511, 0.8519, 0.8524, 0.8547, 0.8541, 0.8817]


# % change of baseline (primeiro valor)
def percent_change(values):
    baseline = values[0]
    values = np.array(values)
    change = (values - baseline) / baseline * 100
    return change

psnr_pct = percent_change(psnr_values)
mae_pct = percent_change(mae_values)
ssim_pct = percent_change(ssim_values)
ms_ssim_pct = percent_change(ms_ssim_values)


# Plot 
fig, axes = plt.subplots(figsize=(8, 6))

plt.plot(stages, psnr_pct, marker='o', label='PSNR (Higher=Better)', color='blue')
plt.plot(stages, mae_pct, marker='o', label='MAE (Lower=Better)', color='green')
plt.plot(stages, ssim_pct, marker='o', label='SSIM (Higher=Better)', color='red')
plt.plot(stages, ms_ssim_pct, marker='o', label='MS-SSIM (Higher=Better)', color='orange')

plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title("Percentage of change from baseline across HP optimization stages", fontsize=14)
plt.ylabel("% Change from baseline")
plt.xlabel("Optimization stage")
#plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.legend(loc='lower left')  



'''
axes[0,0].plot(stages, psnr_values, color="blue", marker="o", linestyle='-', linewidth=2)
axes[0,0].set_title("Cumulative Improvement for PSNR (Higher=Better)")
axes[0,0].set_ylabel("PSNR (dB)")
axes[0,0].set_xlabel("Optimization Stage")

axes[0,1].plot(stages, mae_values, color="green", marker="o", linestyle='-', linewidth=2)
axes[0,1].set_title("Cumulative Improvement for MAE (Lower=Better)")
axes[0,1].set_ylabel("MAE (HU)")
axes[0,1].set_xlabel("Optimization Stage")

axes[1,0].plot(stages, ssim_values, color="red", marker="o", linestyle='-', linewidth=2)
axes[1,0].set_title("Cumulative Improvement for SSIM (Higher=Better)")
axes[1,0].set_ylabel("SSIM")
axes[1,0].set_xlabel("Optimization Stage")

axes[1,1].plot(stages, ms_ssim_values, color="orange", marker="o", linestyle='-', linewidth=2)
axes[1,1].set_title("Cumulative Improvement for MS-SSIM (Higher=Better)")
axes[1,1].set_ylabel("MS-SSIM")
axes[1,1].set_xlabel("Optimization Stage")
'''


plt.tight_layout()
plt.savefig(f"/home/catarina_caldeira/Desktop/code/utils/pct_cumulative_hpsearch_cgan.png",bbox_inches = "tight", dpi=200) # muda
plt.close() 


# Print valores originais + % mudança
print(f"{'Stage':<25} {'PSNR':>8} {'PSNR %Δ':>10} {'MAE':>8} {'MAE %Δ':>10} {'SSIM':>8} {'SSIM %Δ':>10} {'MS-SSIM':>10} {'MS-SSIM %Δ':>12}")
for i in range(len(stages)):
    print(f"{stages[i]:<25} "
          f"{psnr_values[i]:8.4f} {psnr_pct[i]:9.2f}% "
          f"{mae_values[i]:8.4f} {mae_pct[i]:9.2f}% "
          f"{ssim_values[i]:8.4f} {ssim_pct[i]:9.2f}% "
          f"{ms_ssim_values[i]:8.4f} {ms_ssim_pct[i]:11.2f}%")
    
    
    
    
# python -m utils.cumulative_optimization_cascade_HPsearch