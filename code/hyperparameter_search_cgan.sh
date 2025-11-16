# --- SYNTHRAD2023

# Learning rate G and D
#for LR in 0.00001 0.00002 0.00005 0.0001 0.0002 0.0005 0.001 0.002 0.005 0.01
#python -m validations.best_model_cgan --ending_saved_model "0-199.pth" --channels "64, 128, 256, 512, 512" --strides "2,2,2,2" --num_res_units 0 --type_norm BATCH --best_models_path /home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2023/cGAN/5layers_unet_BN_200_lr_0.0005 > log_metrics_cgan_5layers_unet_BN_200_lr_0.0005.txt

# LAYERS
#python -m models.cgan_model --result 6layers_unet_BN_200_lr_0.0005 --lr 0.0005 --lr_d 0.0005 --channels "64, 128, 256, 512, 512, 512" --strides "2,2,2,2,2"  --batch_size_train 32 --n_epochs 200 --type_norm BATCH --num_res_units 0 
#python -m models.cgan_model --result 7layers_unet_BN_200_lr_0.0005 --lr 0.0005 --lr_d 0.0005 --channels "64, 128, 256, 512, 512, 512, 512" --strides "2,2,2,2,2,2"  --batch_size_train 32 --n_epochs 200 --type_norm BATCH --num_res_units 0 
#python -m models.cgan_model --result 8layers_unet_BN_200_lr_0.0005 --lr 0.0005 --lr_d 0.0005 --channels "64, 128, 256, 512, 512, 512, 512, 512" --strides "2,2,2,2,2,2,2"  --batch_size_train 32 --n_epochs 200 --type_norm BATCH --num_res_units 0 
#python -m validations.best_model_cgan --ending_saved_model "0-199.pth" --channels "64, 128, 256, 512, 512, 512" --strides "2,2,2,2,2" --num_res_units 0 --type_norm BATCH --best_models_path /home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2023/cGAN/6layers_unet_BN_200_lr_0.0005 > log_metrics_cgan_6layers_unet_BN_200_lr_0.0005.txt
#python -m validations.best_model_cgan --ending_saved_model "0-199.pth" --channels "64, 128, 256, 512, 512, 512, 512" --strides "2,2,2,2,2,2" --num_res_units 0 --type_norm BATCH --best_models_path /home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2023/cGAN/7layers_unet_BN_200_lr_0.0005 > log_metrics_cgan_7layers_unet_BN_200_lr_0.0005.txt
#python -m validations.best_model_cgan --ending_saved_model "0-199.pth" --channels "64, 128, 256, 512, 512, 512, 512, 512" --strides "2,2,2,2,2,2,2" --num_res_units 0 --type_norm BATCH --best_models_path /home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2023/cGAN/8layers_unet_BN_200_lr_0.0005 > log_metrics_cgan_8layers_unet_BN_200_lr_0.0005.txt

# RESUNET
#python -m models.cgan_model --result 7layers_resunet_BN_200_lr_0.0005 --lr 0.0005 --lr_d 0.0005 --channels "64, 128, 256, 512, 512, 512, 512" --strides "2,2,2,2,2,2"  --batch_size_train 32 --n_epochs 200 --type_norm BATCH --num_res_units 2
#python -m validations.best_model_cgan --ending_saved_model "0-199.pth" --channels "64, 128, 256, 512, 512, 512, 512" --strides "2,2,2,2,2,2" --num_res_units 2 --type_norm BATCH --best_models_path /home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2023/cGAN/7layers_resunet_BN_200_lr_0.0005 > log_metrics_cgan_7layers_resunet_BN_200_lr_0.0005.txt

# G IN + D IN 
# batch size 32
#python -m models.cgan_model --result 7layers_unet_ININ_200_lr_0.0005 --lr 0.0005 --lr_d 0.0005 --channels "64, 128, 256, 512, 512, 512, 512" --strides "2,2,2,2,2,2"  --batch_size_train 32 --n_epochs 200 --type_norm INSTANCE --num_res_units 0 
#python -m validations.best_model_cgan --ending_saved_model "0-199.pth" --channels "64, 128, 256, 512, 512, 512, 512" --strides "2,2,2,2,2,2" --num_res_units 0 --type_norm INSTANCE --best_models_path /home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2023/cGAN/7layers_unet_ININ_200_lr_0.0005 > log_metrics_cgan_7layers_unet_ININ_200_lr_0.0005.txt
# batch size 8
#python -m models.cgan_model --result 7layers_unet_ININ_200_lr_0.0005_batch8 --lr 0.0005 --lr_d 0.0005 --channels "64, 128, 256, 512, 512, 512, 512" --strides "2,2,2,2,2,2"  --batch_size_train 8 --n_epochs 200 --type_norm INSTANCE --num_res_units 0 
#ython -m validations.best_model_cgan --ending_saved_model "0-199.pth" --channels "64, 128, 256, 512, 512, 512, 512" --strides "2,2,2,2,2,2" --num_res_units 0 --type_norm INSTANCE --best_models_path /home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2023/cGAN/7layers_unet_ININ_200_lr_0.0005_batch8 > log_metrics_cgan_7layers_unet_ININ_200_lr_0.0005_batch8.txt

# batch size 1
#python -m models.cgan_model --result 7layers_unet_ININ_200_lr_0.0005_batch1 --lr 0.0005 --lr_d 0.0005 --channels "64, 128, 256, 512, 512, 512, 512" --strides "2,2,2,2,2,2"  --batch_size_train 1 --n_epochs 200 --type_norm INSTANCE --num_res_units 0 
#python -m validations.best_model_cgan --ending_saved_model "0-199.pth" --channels "64, 128, 256, 512, 512, 512, 512" --strides "2,2,2,2,2,2" --num_res_units 0 --type_norm INSTANCE --best_models_path /home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2023/cGAN/7layers_unet_ININ_200_lr_0.0005_batch1 > log_metrics_cgan_7layers_unet_ININ_200_lr_0.0005_batch1.txt


# Kernel size D 
#python -m models.cgan_model --result 7layers_unet_ININ_200_lr_0.0005_batch8_kernelD3 --lr 0.0005 --lr_d 0.0005 --channels "64, 128, 256, 512, 512, 512, 512" --strides "2,2,2,2,2,2"  --batch_size_train 8 --n_epochs 200 --type_norm INSTANCE --num_res_units 0 
#python -m validations.best_model_cgan --ending_saved_model "0-199.pth" --channels "64, 128, 256, 512, 512, 512, 512" --strides "2,2,2,2,2,2" --num_res_units 0 --type_norm INSTANCE --best_models_path /home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2023/cGAN/7layers_unet_ININ_200_lr_0.0005_batch8_kernelD3 > log_metrics_cgan_7layers_unet_ININ_200_lr_0.0005_batch8_kernelD3.txt

# LeakyReLU G
#python -m models.cgan_model --result 7layers_unet_ININ_200_lr_0.0005_batch8_kernelD3_LeakyRelUG --lr 0.0005 --lr_d 0.0005 --channels "64, 128, 256, 512, 512, 512, 512" --strides "2,2,2,2,2,2"  --batch_size_train 8 --n_epochs 200 --type_norm INSTANCE --num_res_units 0 
#python -m validations.best_model_cgan --ending_saved_model "0-199.pth" --channels "64, 128, 256, 512, 512, 512, 512" --strides "2,2,2,2,2,2" --num_res_units 0 --type_norm INSTANCE --best_models_path /home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2023/cGAN/7layers_unet_ININ_200_lr_0.0005_batch8_kernelD3_LeakyRelUG > log_metrics_cgan_7layers_unet_ININ_200_lr_0.0005_batch8_kernelD3_LeakyRelUG.txt

# Learning rate D
#for LR in 0.005 
#0.00001 0.00002 0.00005 0.0001 0.0002 0.0005 0.001 0.002 0.005 
#0.01
#do 
#    python -m models.cgan_model --result 7layers_unet_ININ_200_lr_0.0005_batch8_kernelD3_lrD_$LR --lr 0.0005 --lr_d $LR --channels "64, 128, 256, 512, 512, 512, 512" --strides "2,2,2,2,2,2"  --batch_size_train 8 --n_epochs 200 --type_norm INSTANCE --num_res_units 0 
#    python -m validations.best_model_cgan --ending_saved_model "0-199.pth" --channels "64, 128, 256, 512, 512, 512, 512" --strides "2,2,2,2,2,2" --num_res_units 0 --type_norm INSTANCE --best_models_path /home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2023/cGAN/7layers_unet_ININ_200_lr_0.0005_batch8_kernelD3_lrD_$LR > log_metrics_cgan_7layers_unet_ININ_200_lr_0.0005_batch8_kernelD3_lrD_$LR.txt
#done

# Lambda perceptual loss
#for PL in 1
#do
#    python -m models.cgan_model --result 7layers_unet_ININ_200_lrG_0.0005_lrD_0.005_batch8_kernelD3_PL$PL --lr 0.0005 --lr_d 0.005 --channels "64, 128, 256, 512, 512, 512, 512" --strides "2,2,2,2,2,2"  --batch_size_train 8 --n_epochs 200 --type_norm INSTANCE --num_res_units 0 --perceptual_loss True --lambda_pl $PL
#    python -m validations.best_model_cgan --ending_saved_model "0-199.pth" --channels "64, 128, 256, 512, 512, 512, 512" --strides "2,2,2,2,2,2" --num_res_units 0 --type_norm INSTANCE --best_models_path /home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2023/cGAN/7layers_unet_ININ_200_lrG_0.0005_lrD_0.005_batch8_kernelD3_PL$PL > log_metrics_cgan_7layers_unet_ININ_200_lrG_0.0005_lrD_0.005_batch8_kernelD3_PL$PL.txt
#done
# pl 10, 0.5 was done on terminal


# One sided label smoothing
#python -m models.cgan_model --result 7layers_unet_ININ_200_lrG_0.0005_lrD_0.005_batch8_kernelD3_PL1_OSLS --lr 0.0005 --lr_d 0.005 --channels "64, 128, 256, 512, 512, 512, 512" --strides "2,2,2,2,2,2"  --batch_size_train 8 --n_epochs 200 --type_norm INSTANCE --num_res_units 0 --perceptual_loss True --lambda_pl 1 --label_smoothing True
#python -m validations.best_model_cgan --ending_saved_model "0-199.pth" --channels "64, 128, 256, 512, 512, 512, 512" --strides "2,2,2,2,2,2" --num_res_units 0 --type_norm INSTANCE --best_models_path /home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2023/cGAN/7layers_unet_ININ_200_lrG_0.0005_lrD_0.005_batch8_kernelD3_PL1_OSLS > log_metrics_cgan_7layers_unet_ININ_200_lrG_0.0005_lrD_0.005_batch8_kernelD3_PL1_OSLS.txt

# dropout D 0.25
#python -m models.cgan_model --result 7layers_unet_ININ_200_lrG_0.0005_lrD_0.005_batch8_kernelD3_PL1ddd_OSLS_dropoutD --lr 0.0005 --lr_d 0.005 --channels "64, 128, 256, 512, 512, 512, 512" --strides "2,2,2,2,2,2"  --batch_size_train 8 --n_epochs 200 --type_norm INSTANCE --num_res_units 0 --perceptual_loss True --lambda_pl 1 --label_smoothing True
#python -m validations.best_model_cgan --ending_saved_model "0-199.pth" --channels "64, 128, 256, 512, 512, 512, 512" --strides "2,2,2,2,2,2" --num_res_units 0 --type_norm INSTANCE --best_models_path /home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2023/cGAN/7layers_unet_ININ_200_lrG_0.0005_lrD_0.005_batch8_kernelD3_PL1_OSLS_dropoutD > log_metrics_cgan_7layers_unet_ININ_200_lrG_0.0005_lrD_0.005_batch8_kernelD3_PL1_OSLS_dropoutD.txt

# Epochs
#python -m models.cgan_model --result 7layers_unet_ININ_800_lrG_0.0005_lrD_0.005_batch8_kernelD3_PL1_OSLS --lr 0.0005 --lr_d 0.005 --channels "64, 128, 256, 512, 512, 512, 512" --strides "2,2,2,2,2,2"  --batch_size_train 8 --n_epochs 800 --type_norm INSTANCE --num_res_units 0 --perceptual_loss True --lambda_pl 1 --label_smoothing True


# Val SynthRAD2023
#python -m validations.best_model_cgan --ending_saved_model "0-799.pth" --channels "64, 128, 256, 512, 512, 512, 512" --strides "2,2,2,2,2,2" --num_res_units 0 --type_norm INSTANCE --best_models_path /home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2023/cGAN/7layers_unet_ININ_800_lrG_0.0005_lrD_0.005_batch8_kernelD3_PL1_OSLS > log_metrics_cgan_7layers_unet_ININ_800_lrG_0.0005_lrD_0.005_batch8_kernelD3_PL1_OSLS_SynthRAD2023.txt


# Test SynthRAD2023
python -m tests.test_ensemble --model_state 'G_state_dict' --ending_saved_model "0-799.pth" --best_models_path /home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2023/cGAN/7layers_unet_ININ_800_lrG_0.0005_lrD_0.005_batch8_kernelD3_PL1_OSLS --channels "64, 128, 256, 512, 512, 512, 512" --strides "2,2,2,2,2,2" > log_metrics_TESTING_cgan_7layers_unet_ININ_800_lrG_0.0005_lrD_0.005_batch8_kernelD3_PL1_OSLS_synthrad2023.txt



# --- SYNTHRAD2025

# center approach (A+B+C)
#python -m models.cgan_model --result 7layers_unet_ININ_1000_lrG_0.0005_lrD_0.005_batch8_kernelD3_PL1_OSLS_SynthRAD2025 --lr 0.0005 --lr_d 0.005 --channels "64, 128, 256, 512, 512, 512, 512" --strides "2,2,2,2,2,2"  --batch_size_train 8 --n_epochs 1000 --type_norm INSTANCE --num_res_units 0 --perceptual_loss True --lambda_pl 1 --label_smoothing True --dataset_path "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1" --dataset_name "SynthRAD2025" --subsample_rate 1 
#python -m validations.best_model_cgan --ending_saved_model "0-999.pth" --channels "64, 128, 256, 512, 512, 512, 512" --strides "2,2,2,2,2,2" --num_res_units 0 --type_norm INSTANCE --dataset_path "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1"  --best_models_path /home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2025/cGAN/7layers_unet_ININ_1000_lrG_0.0005_lrD_0.005_batch8_kernelD3_PL1_OSLS_SynthRAD2025 > log_metrics_cgan_7layers_unet_ININ_800_lrG_0.0005_lrD_0.005_batch8_kernelD3_PL1_OSLS_SynthRAD2025_centerABC.txt
# - test in A+B+C
python -m tests.test_ensemble --model_state 'G_state_dict' --tested_in "ABC_centers" --file_format "mha" --channels "64, 128, 256, 512, 512, 512, 512" --strides "2,2,2,2,2,2"  --ending_saved_model "0-999.pth" --best_models_path "/home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2025/cGAN/7layers_unet_ININ_1000_lrG_0.0005_lrD_0.005_batch8_kernelD3_PL1_OSLS_SynthRAD2025" --dataset_path "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1" > log_metrics_TESTING_cgan_7layers_unet_ININ_800_lrG_0.0005_lrD_0.005_batch8_kernelD3_PL1_OSLS_synthrad2025_centerABC.txt
# - test in D
python -m tests.test_ensemble --model_state 'G_state_dict' --tested_in "D_center" --file_format "mha" --channels "64, 128, 256, 512, 512, 512, 512" --strides "2,2,2,2,2,2"  --ending_saved_model "0-999.pth" --best_models_path "/home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2025/cGAN/7layers_unet_ININ_1000_lrG_0.0005_lrD_0.005_batch8_kernelD3_PL1_OSLS_SynthRAD2025" --dataset_path "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train_D/Task1" > log_metrics_TESTING_cgan_7layers_unet_ININ_800_lrG_0.0005_lrD_0.005_batch8_kernelD3_PL1_OSLS_synthrad2025_centerD.txt

# region approach (AB+TH+HN)
#python -m models.cgan_model --result 7layers_unet_ININ_1000_lrG_0.0005_lrD_0.005_batch8_kernelD3_PL1_OSLS_SynthRAD2025_regionABHNTH --lr 0.0005 --lr_d 0.005 --channels "64, 128, 256, 512, 512, 512, 512" --strides "2,2,2,2,2,2"  --batch_size_train 8 --n_epochs 1000 --type_norm INSTANCE --num_res_units 0 --perceptual_loss True --lambda_pl 1 --label_smoothing True --dataset_path "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1" --dataset_name "SynthRAD2025" --subsample_rate 1 
#python -m validations.best_model_cgan --file_format "mha" --ending_saved_model "0-999.pth" --channels "64, 128, 256, 512, 512, 512, 512" --strides "2,2,2,2,2,2" --num_res_units 0 --type_norm INSTANCE --dataset_path "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1"  --best_models_path /home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2025/cGAN/7layers_unet_ININ_1000_lrG_0.0005_lrD_0.005_batch8_kernelD3_PL1_OSLS_SynthRAD2025_regionABHNTH > log_metrics_cgan_7layers_unet_ININ_800_lrG_0.0005_lrD_0.005_batch8_kernelD3_PL1_OSLS_SynthRAD2025_regionABTHHN.txt

# test in (AB+TH+HN)
#python -m tests.test_ensemble --file_format "mha" --channels "64, 128, 256, 512, 512, 512, 512" --strides "2,2,2,2,2,2"  --ending_saved_model "0-999.pth" --best_models_path "/home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2025/cGAN/7layers_unet_ININ_1000_lrG_0.0005_lrD_0.005_batch8_kernelD3_PL1_OSLS_SynthRAD2025" --dataset_path "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1" > log_metrics_TESTING_cgan_7layers_unet_ININ_1000_lrG_0.0005_lrD_0.005_batch8_kernelD3_PL1_OSLS_synthrad2025_regionABHNTH.txt




# bash hyperparameter_search_cgan.sh