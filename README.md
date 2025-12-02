# Master Thesis: MRI-to-CT Synthesis

## Title: Development and Comparative Assessment of General and Anatomy-Specific Models for Synthetic Computed Tomography Generation

This repository contains training and evaluation scripts, models, and utilities for my Master's thesis research on **MRI-to-CT image synthesis** using deep generative models.

## Thesis Documents

- 📄 [**Thesis (PDF)**](docs/Master%20Thesis%20Final%20-%20Catarina%20Caldeira.pdf)
- 🎯 [**Presentation (PDF)**](docs/Master%20Thesis%20Presentation%20-%20Catarina%20Caldeira.pdf)


<details>
<summary>Table of Contents</summary>

1. [About the Project](#about-the-project)
    - [Repository Structure](#repository-structure)
2. [Getting Started](#getting-started)
   - [Installation](#installation)
   - [Usage](#usage)
3. [Contributing](#contributing)
4. [Acknowledgements](#acknowledgements)

</details>

## About the Project

This project explores conditional generative models for medical image synthesis with the following objectives:

- **Model exploration:** Compare general (multi-region) and anatomy-specific (region-specific) 2D deep learning models for MRI-to-CT synthesis using the SynthRAD2025 dataset after HP search in SynthRAD2023.
- **Clinical application:** Generate synthetic CT volumes suitable for treatment planning in three anatomical regions: head & neck (HN), thorax (TH), and abdomen (AB).
- **Multi-domain evaluation:** Assess synthetic CT quality across three complementary domains:
  - *Image quality metrics*: PSNR, MAE, SSIM, MS-SSIM (pixel-level fidelity).
  - *Geometric consistency*: Segmentation-based metrics (Dice, Hausdorff distance) on anatomical structures.
  - *Clinical feasibility*: Dose calculation metrics (MAE, DVH, Gamma pass rate) using proton therapy treatment plans.

### Repository Structure

```
code/
├── graph_geometric_metrics_all.py    # Anatomical structure segmentation metrics
├── test_ensemble.py                  # Ensemble inference (averaging of outputs of the models from 5-fold CV)
├── test_transforms.py                # Augmentation visualisation (experiment)
├── models/
│   ├── autoencoder_model.py          # AE 5-fold CV training
│   ├── cgan_model.py                 # cGAN 5-fold CV training
│   ├── discriminator.py              # PatchGAN discriminator
│   ├── generator.py                  # Original Pix2Pix generator (experiment)
│   └── generator_monai.py            # MONAI U-Net generator (used)
├── trains/
│   └── train.py                      # Per epoch training functions used in 5-fold CV
├── utils/
│   ├── load_data.py                  # Dataset loading & preprocessing
│   ├── data_augmentations_thesis.py  # Augmentation pipeline
│   ├── images_utils.py               # Transform utilities
│   ├── image_metrics_challenges.py   # Image quality metrics for 3D images
│   ├── plot_utils.py                 # Visualization helpers
│   ├── linear_regression.py          # Statistical analysis
│   ├── one_vs_multi_region.py        # Regional comparison plots
│   ├── statistic_ttest.py            # Paired t-tests
├── validations/
│   ├── best_model_cgan.py            # cGAN best model validation
│   ├── best_model_ae.py              # Autoencoder best model validation
│   └── validation.py                 # Per epoch validation functions used in 5-fold CV and testing ensemble
└── hyperparameter_search_*.sh        # Grid search scripts (**see these for complete training workflow examples**)

evaluation/
├── evaluate-local.py                 # Evaluation with TotalSegmentator of PSNR, MAE, MS-SSIM, DICE and HD95
├── segmentation_metrics.py           # Segmentation-based metrics
└── ts_utils.py                       # TotalSegmentator utilities


evaluation_dose/
├── dose_calculations.py              # Compute dose-based metrics (MAE, gamma index, DVH) from MATLAB/ITK dose files
├── dvh_graph.py                      # Identify best patients by DVH metric and plot dose-volume histograms
├── mean_metrics_per_region.py        # Compute and visualise mean metrics per anatomical region and centre
├── visualizing_plans.py              # Visualise dose distributions and anatomical segmentations for patients
├── is_there_seg.py                   # Check for presence and validity of segmentation files for each patient
└── all_metrics.json                  # Aggregated metrics for all patients

matlab_dose_calculation/
└── run_dose_calculations_for_all_regions.m     # MATLAB script for calculating dose calculations across all anatomical regions using TotalSegmentator outputs with matrad (added under matRad real github, folder userdata)
```

### Installation
1. Clone the repo
```bash
git clone https://github.com/catacc-dev/msc_thesis_work.git
cd msc_thesis_work
```

2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate      # Linux or macOS
venv\Scripts\activate         # Windows
```

3. Install packages
```bash
pip install -r requirements.txt
```

### Usage

1. Perform 5-fold cross-validation to save model gradients at the epoch with minimal MAE loss

-> Autoencoder

```bash
python -m models.autoencoder_model \
    --dataset_path /path/to/dataset \
    --epoch_start 0 \
    --n_epochs 99 \
    --dataset_name "SynthRAD2023" \
    --batch_size_train 32 \
    --batch_size_val 1 \
    --lr 0.0005 \
    --b1 0.9 \
    --b2 0.999 \
    --result "experiment_name" \
    --subsample_rate 2 \
    --checkpoint_interval 100 \
    --channels "64, 128, 256, 512, 512" \
    --strides "2, 2, 2, 2" \
    --num_res_units 0 \
    --type_norm "INSTANCE" \
    --choosen_region "AB"
```

-> cGAN

```bash
python -m models.cgan_model \
    --dataset_path /path/to/dataset \
    --best_model_path "model_checkpoint" \
    --epoch_start 0 \
    --n_epochs [number_of_epochs] \
    --dataset_name "SynthRAD2023" \
    --batch_size_train 32 \
    --batch_size_val 1 \
    --lr [learning_rate] \
    --lr_d [discriminator_learning_rate] \
    --b1 0.5 \
    --b2 0.999 \
    --checkpoint_interval 100 \
    --result "experiment_name" \
    --subsample_rate 7 \
    --perceptual_loss [True/False] \
    --apply_da [True/False] \
    --lambda_pixel 100 \
    --lambda_adversarial 1 \
    --lambda_pl [perceptual_loss_weight] \
    --label_smoothing [True/False] \
    --channels "64, 128, 256, 512, 512" \
    --strides "2, 2, 2, 2" \
    --num_res_units 0 \
    --type_norm "BATCH"
```

2.  Selecting optimal hyperparameters combination based on the average PSNR, MAE, SSIM, and MS-SSIM scores across the 5 cross-validation models

-> Autoencoder

```bash
python -m validations.best_model_ae \
    --dataset_path /path/to/dataset \
    --best_models_path /best/models/path
    --batch_size_val 1 \
    --channels "64, 128, 256, 512, 512" \
    --strides "2, 2, 2, 2" \
    --num_res_units 0 \
    --type_norm "INSTANCE" \
    --ending_saved_model "0-99.pth"
```

-> cGAN

```bash
python -m validations.best_model_ae \
    --dataset_path /path/to/dataset \
    --best_models_path /best/models/path
    --batch_size_val 1 \
    --channels "64, 128, 256, 512, 512" \
    --strides "2, 2, 2, 2" \
    --num_res_units 0 \
    --type_norm "INSTANCE" \
    --ending_saved_model "0-99.pth" \
    --file_format "mha"
```

3. Test the best ensemble of models (for both Autoencoder and cGAN)

```bash
python -m test.test_ensemble \
    --dataset_path /path/to/dataset \
    --best_models_path /best/models/path
    --batch_size_test 1 \
    --channels "64, 128, 256, 512, 512" \
    --strides "2, 2, 2, 2" \
    --num_res_units 0 \
    --type_norm "INSTANCE" \
    --ending_saved_model "0-99.pth" \
    --file_format "mha" \
    --tested_in "AB" \
    --model_state "G_state_dict"
```


## Contributing
Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also open an issue with the tag "enhancement". Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
   
## Acknowledgements

- **Datasets:** [SynthRAD2023 Grand Challenge](https://synthrad2023.grand-challenge.org/) and [SynthRAD2025 Grand Challenge](https://synthrad2025.grand-challenge.org/)
- **Segmentation tool:** [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)
- **Evaluation framework:** Adapted from [SynthRAD2025 evaluation](https://github.com/SynthRAD2025/evaluation)
- **Discriminator architecture:** Based on [Pix2Pix](https://arxiv.org/abs/1611.07004)
- **Dose calculation framework:** [matRad](https://github.com/e0404/matRad)





