
# 5 layers
#python -m validations.best_model_ae --ending_saved_model "0-199.pth" --channels "64, 128, 256, 512, 512" --strides "2,2,2,2" --num_res_units 0 --type_norm BATCH --best_models_path /home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2023/generator/5layers_unet_BN_200_lr_0.001  > log_metrics_5layers_unet_BN_200_lr_0.001.txt


# 6 layers
#python -m models.autoencoder_model --result 6layers_unet_BN_200_lr_0.001 --lr 0.001 --channels "64, 128, 256, 512, 512, 512" --strides "2,2,2,2,2" --batch_size_train 32 --n_epochs 200 --type_norm BATCH --num_res_units 0
# 7 layers
#python -m models.autoencoder_model --result 7layers_unet_BN_200_lr_0.001 --lr 0.001 --channels "64, 128, 256, 512, 512, 512, 512" --strides "2,2,2,2,2,2" --batch_size_train 32 --n_epochs 200 --type_norm BATCH --num_res_units 0
# 8 layers
#python -m models.autoencoder_model --result 8layers_unet_BN_200_lr_0.001 --lr 0.001 --channels "64, 128, 256, 512, 512, 512, 512, 512" --strides "2,2,2,2,2,2,2" --batch_size_train 32 --n_epochs 200 --type_norm BATCH --num_res_units 0

#python -m validations.best_model_ae --ending_saved_model "0-199.pth" --channels "64, 128, 256, 512, 512, 512" --strides "2,2,2,2,2" --num_res_units 0 --type_norm BATCH --best_models_path /home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2023/generator/6layers_unet_BN_200_lr_0.001  > log_metrics_6layers_unet_BN_200_lr_0.001.txt
#python -m validations.best_model_ae --ending_saved_model "0-199.pth" --channels "64, 128, 256, 512, 512, 512, 512" --strides "2,2,2,2,2,2" --num_res_units 0 --type_norm BATCH --best_models_path /home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2023/generator/7layers_unet_BN_200_lr_0.001  > log_metrics_7layers_unet_BN_200_lr_0.001.txt
#python -m validations.best_model_ae --ending_saved_model "0-199.pth" --channels "64, 128, 256, 512, 512, 512, 512, 512" --strides "2,2,2,2,2,2,2" --num_res_units 0 --type_norm BATCH --best_models_path /home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2023/generator/8layers_unet_BN_200_lr_0.001  > log_metrics_8layers_unet_BN_200_lr_0.001.txt


# Resunet
#python -m models.autoencoder_model --result 8layers_resunet_BN_200_lr_0.001 --lr 0.001 --channels "64, 128, 256, 512, 512, 512, 512, 512" --strides "2,2,2,2,2,2,2" --batch_size_train 32 --n_epochs 200 --type_norm BATCH --num_res_units 2
#python -m validations.best_model_ae --ending_saved_model "0-199.pth" --channels "64, 128, 256, 512, 512, 512, 512, 512" --strides "2,2,2,2,2,2,2" --num_res_units 2 --type_norm BATCH --best_models_path /home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2023/generator/8layers_resunet_BN_200_lr_0.001  > log_metrics_8layers_resunet_BN_200_lr_0.001.txt

# IN
# batch size 32
#python -m models.autoencoder_model --result 8layers_unet_IN_200_lr_0.001 --lr 0.001 --channels "64, 128, 256, 512, 512, 512, 512, 512" --strides "2,2,2,2,2,2,2" --batch_size_train 32 --n_epochs 200 --type_norm INSTANCE --num_res_units 0
#python -m validations.best_model_ae --ending_saved_model "0-199.pth" --channels "64, 128, 256, 512, 512, 512, 512, 512" --strides "2,2,2,2,2,2,2" --num_res_units 0 --type_norm INSTANCE --best_models_path /home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2023/generator/8layers_unet_IN_200_lr_0.001  > log_metrics_8layers_unet_IN_200_lr_0.001.txt
# batch size 8
#python -m models.autoencoder_model --result 8layers_unet_IN_200_lr_0.001_batch8 --lr 0.001 --channels "64, 128, 256, 512, 512, 512, 512, 512" --strides "2,2,2,2,2,2,2" --batch_size_train 8 --n_epochs 200 --type_norm INSTANCE --num_res_units 0
#python -m validations.best_model_ae --ending_saved_model "0-199.pth" --channels "64, 128, 256, 512, 512, 512, 512, 512" --strides "2,2,2,2,2,2,2" --num_res_units 0 --type_norm INSTANCE --best_models_path /home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2023/generator/8layers_unet_IN_200_lr_0.001_batch8  > log_metrics_8layers_unet_IN_200_lr_0.001_batch8.txt
# batch size 1
#python -m models.autoencoder_model --result 8layers_unet_IN_200_lr_0.001_batch1 --lr 0.001 --channels "64, 128, 256, 512, 512, 512, 512, 512" --strides "2,2,2,2,2,2,2" --batch_size_train 1 --n_epochs 200 --type_norm INSTANCE --num_res_units 0
#python -m validations.best_model_ae --ending_saved_model "0-199.pth" --channels "64, 128, 256, 512, 512, 512, 512, 512" --strides "2,2,2,2,2,2,2" --num_res_units 0 --type_norm INSTANCE --best_models_path /home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2023/generator/8layers_unet_IN_200_lr_0.001_batch1  > log_metrics_8layers_unet_IN_200_lr_0.001_batch1.txt

# ReLU or LeakyReLU G
#python -m models.autoencoder_model --result 8layers_unet_IN_200_lr_0.001_batch8_LeakyReLU --lr 0.001 --channels "64, 128, 256, 512, 512, 512, 512, 512" --strides "2,2,2,2,2,2,2" --batch_size_train 8 --n_epochs 200 --type_norm INSTANCE --num_res_units 0
#python -m validations.best_model_ae --ending_saved_model "0-199.pth" --channels "64, 128, 256, 512, 512, 512, 512, 512" --strides "2,2,2,2,2,2,2" --num_res_units 0 --type_norm INSTANCE --best_models_path /home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2023/generator/8layers_unet_IN_200_lr_0.001_batch8_LeakyReLU  > log_metrics_8layers_unet_IN_200_lr_0.001_batch8_LeakyReLU.txt

# Epochs
#python -m models.autoencoder_model --result 8layers_unet_IN_800_lr_0.001_batch8 --lr 0.001 --channels "64, 128, 256, 512, 512, 512, 512, 512" --strides "2,2,2,2,2,2,2" --batch_size_train 8 --n_epochs 800 --type_norm INSTANCE --num_res_units 0

#python -m validations.best_model_ae --ending_saved_model "0-799.pth" --channels "64, 128, 256, 512, 512, 512, 512, 512" --strides "2,2,2,2,2,2,2" --num_res_units 0 --type_norm INSTANCE --best_models_path /home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2023/generator/8layers_unet_IN_800_lr_0.001_batch8  > log_metrics_8layers_unet_IN_800_lr_0.001_batch8.txt

# Test SynthRAD2023
python -m tests.test_ensemble --model_state "model_state_dict" --tested_in "pelvis" --ending_saved_model "0-799.pth" --channels "64, 128, 256, 512, 512, 512, 512, 512" --strides "2,2,2,2,2,2,2" --num_res_units 0 --type_norm INSTANCE --best_models_path /home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2023/generator/8layers_unet_IN_800_lr_0.001_batch8 > log_metrics_TESTING_ae_8layers_unet_IN_1000_lr_0.001_batch8_synthrad2023.txt


# --- SynthRAD 2025

# center approach (A+B+C)
#python -m models.autoencoder_model --result 8layers_unet_IN_1000_lr_0.001_batch8 --lr 0.001 --dataset_path "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1" --channels "64, 128, 256, 512, 512, 512, 512, 512" --strides "2,2,2,2,2,2,2" --batch_size_train 8 --n_epochs 1000 --type_norm INSTANCE --num_res_units 0
#python -m validations.best_model_ae --ending_saved_model "0-999.pth" --dataset_path "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1" --channels "64, 128, 256, 512, 512, 512, 512, 512" --strides "2,2,2,2,2,2,2" --num_res_units 0 --type_norm INSTANCE --best_models_path /home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2025/generator/8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_ABC  > log_metrics_8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025.txt

# - test in A+B+C
python -m tests.test_ensemble --model_state "model_state_dict" --tested_in "ABC_centers" --file_format "mha" --ending_saved_model "0-999.pth" --best_models_path "/home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2025/generator/8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_ABC" --dataset_path "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1" > log_metrics_TESTING_ae_8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_centerABC.txt

#python -m tests.test_ensemble --tested_in "centers_2D_norm" --file_format "mha" --ending_saved_model "0-999.pth" --best_models_path "/home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2025/generator/8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_ABC" --dataset_path "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1" > log_metrics_TESTING_ae_8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_centerABC.txt
#python -m tests.test_ensemble --tested_in "centers_2D_norm_masked" --file_format "mha" --ending_saved_model "0-999.pth" --best_models_path "/home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2025/generator/8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_ABC" --dataset_path "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1" > log_metrics_TESTING_ae_8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_centerABC.txt
#python -m tests.test_ensemble --tested_in "centers_2D_norm_masked_datarange" --file_format "mha" --ending_saved_model "0-999.pth" --best_models_path "/home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2025/generator/8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_ABC" --dataset_path "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1" > log_metrics_TESTING_ae_8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_centerABC.txt


# - test in D
python -m tests.test_ensemble --model_state "model_state_dict" --tested_in "D_center" --file_format "mha" --ending_saved_model "0-999.pth" --best_models_path "/home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2025/generator/8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_ABC" --dataset_path "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train_D/Task1" > log_metrics_TESTING_ae_8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_centerD.txt



# region approach (AB+TH+HN)
#python -m models.autoencoder_model --result 8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionABHNTH --lr 0.001 --dataset_path "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1"  --channels "64, 128, 256, 512, 512, 512, 512, 512" --strides "2,2,2,2,2,2,2"  --batch_size_train 8 --n_epochs 1000 --type_norm INSTANCE --num_res_units 0  --dataset_path "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1" --dataset_name "SynthRAD2025" --subsample_rate 1 
#python -m validations.best_model_ae --ending_saved_model "0-999.pth"  --channels "64, 128, 256, 512, 512, 512, 512, 512" --strides "2,2,2,2,2,2,2" --num_res_units 0 --type_norm INSTANCE --dataset_path "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1"  --best_models_path /home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2025/generator/8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionABHNTH > log_metrics_8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionABHNTH.txt

# - test in AB+TH+HN
python -m tests.test_ensemble --tested_in "all_regions" --file_format "mha" --ending_saved_model "0-999.pth" --best_models_path "/home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2025/generator/8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionABHNTH" --dataset_path "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1" > log_metrics_TESTING_ae_8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionABHNTH_testABTHHN.txt
# - test in AB
python -m tests.test_ensemble --tested_in "AB" --file_format "mha" --ending_saved_model "0-999.pth" --best_models_path "/home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2025/generator/8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionABHNTH" --dataset_path "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1" > log_metrics_TESTING_ae_8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionABHNTH_testAB.txt
# - test in TH
python -m tests.test_ensemble --tested_in "TH" --file_format "mha" --ending_saved_model "0-999.pth" --best_models_path "/home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2025/generator/8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionABHNTH" --dataset_path "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1" > log_metrics_TESTING_ae_8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionABHNTH_testTH.txt
# - test in HN
python -m tests.test_ensemble --tested_in "HN" --file_format "mha" --ending_saved_model "0-999.pth" --best_models_path "/home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2025/generator/8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionABHNTH" --dataset_path "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1" > log_metrics_TESTING_ae_8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionABHNTH_testHN.txt
# - test in HN (centro D)
python -m tests.test_ensemble --tested_in "HND" --file_format "mha" --ending_saved_model "0-999.pth" --best_models_path "/home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2025/generator/8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionABHNTH" --dataset_path "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train_D/Task1" > log_metrics_TESTING_ae_8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionABHNTH_testHND.txt



# region approach (AB)
#python -m models.autoencoder_model --result 8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionAB --lr 0.001 --dataset_path "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1"  --channels "64, 128, 256, 512, 512, 512, 512, 512" --strides "2,2,2,2,2,2,2"  --batch_size_train 8 --n_epochs 1000 --type_norm INSTANCE --num_res_units 0  --dataset_path "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1" --dataset_name "SynthRAD2025" --subsample_rate 1 Results - SynthRAD2023
#python -m validations.best_model_ae --ending_saved_model "0-999.pth"  --channels "64, 128, 256, 512, 512, 512, 512, 512" --strides "2,2,2,2,2,2,2" --num_res_units 0 --type_norm INSTANCE --dataset_path "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1"  --best_models_path /home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2025/generator/8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionAB > log_metrics_8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionAB.txt

# -test in AB
python -m tests.test_ensemble  --tested_in "AB" --file_format "mha" --ending_saved_model "0-999.pth" --best_models_path "/home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2025/generator/8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionAB" --dataset_path "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1" > log_metrics_TESTING_ae_8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionAB_testAB.txt



# region approach (TH) 
#python -m models.autoencoder_model --choosen_region "TH" --result 8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionTH --lr 0.001 --dataset_path "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1"  --channels "64, 128, 256, 512, 512, 512, 512, 512" --strides "2,2,2,2,2,2,2"  --batch_size_train 8 --n_epochs 1000 --type_norm INSTANCE --num_res_units 0  --dataset_path "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1" --dataset_name "SynthRAD2025" --subsample_rate 1 
#python -m validations.best_model_ae --ending_saved_model "0-999.pth"  --channels "64, 128, 256, 512, 512, 512, 512, 512" --strides "2,2,2,2,2,2,2" --num_res_units 0 --type_norm INSTANCE --dataset_path "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1"  --best_models_path /home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2025/generator/8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionTH > log_metrics_8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionTH.txt

# - test in TH
python -m tests.test_ensemble  --tested_in "TH" --file_format "mha" --ending_saved_model "0-999.pth" --best_models_path "/home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2025/generator/8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionTH" --dataset_path "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1" > log_metrics_TESTING_ae_8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionTH_testTH.txt



# region approach (HN)
#python -m models.autoencoder_model --choosen_region "HN" --result 8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionHN --lr 0.001 --dataset_path "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1"  --channels "64, 128, 256, 512, 512, 512, 512, 512" --strides "2,2,2,2,2,2,2"  --batch_size_train 8 --n_epochs 1000 --type_norm INSTANCE --num_res_units 0  --dataset_path "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1" --dataset_name "SynthRAD2025" --subsample_rate 1 
#python -m validations.best_model_ae --ending_saved_model "0-999.pth"  --channels "64, 128, 256, 512, 512, 512, 512, 512" --strides "2,2,2,2,2,2,2" --num_res_units 0 --type_norm INSTANCE --dataset_path "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1"  --best_models_path /home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2025/generator/8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionHN > log_metrics_8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionHN.txt

# - test in HN
python -m tests.test_ensemble --tested_in "HN" --file_format "mha" --ending_saved_model "0-999.pth" --best_models_path "/home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2025/generator/8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionHN" --dataset_path "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1" > log_metrics_TESTING_ae_8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionHN_testHN.txt
# - test in HN (centro D)
python -m tests.test_ensemble --tested_in "HND" --file_format "mha" --ending_saved_model "0-999.pth" --best_models_path "/home/catarina_caldeira/Desktop/code/validations/saved_models/SynthRAD2025/generator/8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionHN" --dataset_path "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train_D/Task1" > log_metrics_TESTING_ae_8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionHN_testHND.txt



# bash hyperparameter_search_ae.sh