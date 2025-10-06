#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import SimpleITK
import SimpleITK as sitk
import json
import numpy as np
import os
from typing import Optional
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.util.arraycrop import crop
from scipy.signal import fftconvolve
from scipy.ndimage import uniform_filter

# MAE, SSIM (slightly change to use the costum made implementation of 2025) where from SynthRAD2023
# MS-SSIM, PSNR was from SynthRAD2025


class ImageMetrics():
    def __init__(self):
        # Use fixed wide dynamic range
        self.dynamic_range = [-1024., 3000.]
    
    def score_patient(self, ground_truth_path, predicted_path, mask_path):
        
        gt = sitk.ReadImage(ground_truth_path)
        pred = sitk.ReadImage(predicted_path)
        mask = sitk.ReadImage(mask_path)
                
        caster = SimpleITK.CastImageFilter()
        caster.SetOutputPixelType(SimpleITK.sitkFloat32)
        caster.SetNumberOfThreads(1)

        gt = caster.Execute(gt)
        pred = caster.Execute(pred)
        mask = caster.Execute(mask)
        
        # Get numpy array from SITK Image
        gt_array = SimpleITK.GetArrayFromImage(gt)
        pred_array = SimpleITK.GetArrayFromImage(pred)
        mask_array = SimpleITK.GetArrayFromImage(mask)
        
        
        # Calculate image metrics
        mae_value = self.mae(gt_array,
                             pred_array,
                             mask_array)
        
        psnr_value = self.psnr(gt_array,
                               pred_array,
                               mask_array,
                               use_population_range=True)
        
        ssim_value = self.ssim(gt_array,
                               pred_array, 
                               mask_array)
        
        ms_ssim_value, ms_ssim_mask_value = self.ms_ssim(gt_array,
                                                        pred_array, 
                                                        mask_array)

        return {
            'mae': mae_value,
            'psnr': psnr_value,
            'ssim': ssim_value,
            'ms_ssim': ms_ssim_mask_value
        }
    
    def mae(self,
            gt: np.ndarray, 
            pred: np.ndarray,
            mask: Optional[np.ndarray] = None) -> float:
        """
        Compute Mean Absolute Error (MAE)
    
        Parameters
        ----------
        gt : np.ndarray
            Ground truth
        pred : np.ndarray
            Prediction
        mask : np.ndarray, optional
            Mask for voxels to include. The default is None (including all voxels).
    
        Returns
        -------
        mae : float
            mean absolute error.
    
        """
        if mask is None:
            mask = np.ones(gt.shape)
        else:
            #binarize mask
            mask = np.where(mask>0, 1., 0.)
            
        mae_value = np.sum(np.abs(gt*mask - pred*mask))/mask.sum() 
        return float(mae_value)
    
    
    def psnr(self,
             gt: np.ndarray, 
             pred: np.ndarray,
             mask: Optional[np.ndarray] = None,
             use_population_range: Optional[bool] = False) -> float:
        """
        Compute Peak Signal to Noise Ratio metric (PSNR)
    
        Parameters
        ----------
        gt : np.ndarray
            Ground truth
        pred : np.ndarray
            Prediction
        mask : np.ndarray, optional
            Mask for voxels to include. The default is None (including all voxels).
        use_population_range : bool, optional
            When a predefined population wide dynamic range should be used.
            gt and pred will also be clipped to these values.
    
        Returns
        -------
        psnr : float
            Peak signal to noise ratio..
    
        """
        if mask is None:
            mask = np.ones(gt.shape)
        else:
            #binarize mask
            mask = np.where(mask>0, 1., 0.)
            
        if use_population_range:            
            # Clip gt and pred to the dynamic range
            gt = np.clip(gt, a_min=self.dynamic_range[0], a_max=self.dynamic_range[1])
            pred = np.clip(pred, a_min=self.dynamic_range[0], a_max=self.dynamic_range[1])
            dynamic_range = self.dynamic_range[1]  - self.dynamic_range[0]
        else:
            dynamic_range = gt.max()-gt.min()
            pred = np.clip(pred, a_min=gt.min(), a_max=gt.max())
            
        # apply mask
        gt = gt[mask==1]
        pred = pred[mask==1]
        psnr_value = peak_signal_noise_ratio(gt, pred, data_range=dynamic_range)
        return float(psnr_value)
    
    
    def ssim(self,
              gt: np.ndarray, 
              pred: np.ndarray,
              mask: Optional[np.ndarray] = None) -> float:
        """
        Compute Structural Similarity Index Metric (SSIM)
    
        Parameters
        ----------
        gt : np.ndarray
            Ground truth
        pred : np.ndarray
            Prediction
        mask : np.ndarray, optional
            Mask for voxels to include. The default is None (including all voxels).
    
        Returns
        -------
        ssim : float
            structural similarity index metric.
    
        """
        # Clip gt and pred to the dynamic range
        gt = np.clip(gt, min(self.dynamic_range), max(self.dynamic_range))
        pred = np.clip(pred, min(self.dynamic_range), max(self.dynamic_range))

        if mask is not None:
            #binarize mask
            mask = np.where(mask>0, 1., 0.)
            
            # Mask gt and pred
            gt = np.where(mask==0, min(self.dynamic_range), gt)
            pred = np.where(mask==0, min(self.dynamic_range), pred)

        # Make values non-negative
        if min(self.dynamic_range) < 0:
            gt = gt - min(self.dynamic_range)
            pred = pred - min(self.dynamic_range)

        # Set dynamic range for ssim calculation and calculate ssim_map per pixel
        dynamic_range = self.dynamic_range[1] - self.dynamic_range[0]
        ssim_value_full, ssim_map = structural_similarity(gt, pred, data_range=dynamic_range, full=True)

        if mask is not None:
            # Follow skimage implementation of calculating the mean value:  
            # crop(ssim_map, pad).mean(dtype=np.float64), with pad=3 by default.
            pad = 3
            ssim_value_masked  = (crop(ssim_map, pad)[crop(mask, pad).astype(bool)]).mean(dtype=np.float64)
            return ssim_value_masked
        else:
            return ssim_value_full

        
        
    # Compute the luminance, contrast and structure components of the SSIM between two images    
    def structural_similarity_at_scale(self, im1,
        im2,
        *,
        luminance_weight=1,
        contrast_weight=1,
        structure_weight=1,
        win_size=None,
        gradient=False,
        data_range=None,
        channel_axis=None,
        gaussian_weights=False,
        full=False,
        **kwargs,):
            K1 = kwargs.pop('K1', 0.01)
            K2 = kwargs.pop('K2', 0.03)
            sigma = kwargs.pop('sigma', 1.5)
            if K1 < 0:
                raise ValueError("K1 must be positive")
            if K2 < 0:
                raise ValueError("K2 must be positive")
            if sigma < 0:
                raise ValueError("sigma must be positive")
            use_sample_covariance = kwargs.pop('use_sample_covariance', True)
            if gaussian_weights:
                # Set to give an 11-tap filter with the default sigma of 1.5 to match
                # Wang et. al. 2004.
                truncate = 3.5

            if win_size is None:
                if gaussian_weights:
                    # set win_size used by crop to match the filter size
                    r = int(truncate * sigma + 0.5)  # radius as in ndimage
                    win_size = 2 * r + 1
                else:
                    win_size = 7  # backwards compatibility
            if gaussian_weights:
                filter_func = gaussian
                filter_args = {'sigma': sigma, 'truncate': truncate, 'mode': 'reflect'}
            else:
                filter_func = uniform_filter
                filter_args = {'size': win_size}

            ndim = im1.ndim
            NP = win_size**ndim

            # filter has already normalized by NP
            if use_sample_covariance:
                cov_norm = NP / (NP - 1)  # sample covariance
            else:
                cov_norm = 1.0  # population covariance to match Wang et. al. 2004
            # compute (weighted) means
            ux = filter_func(im1, **filter_args)
            uy = filter_func(im2, **filter_args)


            # compute (weighted) variances and covariances
            uxx = filter_func(im1 * im1, **filter_args)
            uyy = filter_func(im2 * im2, **filter_args)
            uxy = filter_func(im1 * im2, **filter_args)
            vx = cov_norm * (uxx - ux * ux)
            vxsqrt = np.clip(vx, a_min=0, a_max=None) ** 0.5 # TODO: this is very ugly
            vy = cov_norm * (uyy - uy * uy)
            vysqrt = np.clip(vy, a_min=0, a_max=None) ** 0.5 # TODO: this is very ugly
            vxy = cov_norm * (uxy - ux * uy)

            R = data_range
            C1 = (K1 * R) ** 2
            C2 = (K2 * R) ** 2
            C3 = C2 / 2

            L = np.clip((2 * ux * uy + C1) / (ux * ux + uy * uy + C1), a_min=0, a_max=None) # TODO is this clipping necessary or do we increase K1 and K2?

            C = np.clip((2 * vxsqrt * vysqrt + C2) / (vx + vy + C2), a_min=0, a_max=None)
            S = np.clip((vxy + C3) / (vxsqrt * vysqrt + C3), a_min=0, a_max=None)

            result = (L ** luminance_weight) * (C ** contrast_weight) * (S ** structure_weight)
            # to avoid edge effects will ignore filter radius strip around edges
            pad = (win_size - 1) // 2

            # compute (weighted) mean of ssim. Use float64 for accuracy.
            mssim = crop(result, pad).mean(dtype=np.float64)

            if full:
                return mssim, result
            return mssim


    # Compute the masked MS-SSIM by masking the SSIM at every resolution level
    def ms_ssim(self, 
                gt: np.ndarray, 
                pred: np.ndarray, 
                mask: Optional[np.ndarray] = None, 
                scale_weights: Optional[np.ndarray] = None):

        # Clip gt and pred to the dynamic range
        gt = np.clip(gt, min(self.dynamic_range), max(self.dynamic_range))
        pred = np.clip(pred, min(self.dynamic_range), max(self.dynamic_range))

        # Handle mask
        if mask is not None:
            mask = np.where(mask > 0, 1., 0.)
            
            # Mask gt and pred
            gt = np.where(mask==0, min(self.dynamic_range), gt)
            pred = np.where(mask==0, min(self.dynamic_range), pred)


        # Make values non-negative
        if min(self.dynamic_range) < 0:
            gt = gt - min(self.dynamic_range)
            pred = pred - min(self.dynamic_range)

        # Set dynamic range for ssim calculation and calculate ssim_map per pixel
        dynamic_range = self.dynamic_range[1] - self.dynamic_range[0]


        # see Eq. 7 in https://live.ece.utexas.edu/publications/2003/zw_asil2003_msssim.pdf
        # Also, the final sentence of section 3.2 (Results)
        scale_weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]) if scale_weights is None else scale_weights
        luminance_weights = np.array([0, 0, 0, 0, 0, 0.1333]) if scale_weights is None else scale_weights 
        levels = len(scale_weights)

        downsample_filter = np.ones((2, 2, 2)) / 8

        gtx, gty, gtz = gt.shape

        # Due to the downsampling in the MS-SSIM, the minimum matrix size must be 97 in every dimension
        target_size = 97

        pad_values = [
            (np.clip((target_size - dim)//2, a_min=0, a_max=None), 
            np.clip(target_size - dim - (target_size - dim)//2, a_min=0, a_max=None)) 
            for dim in [gtx, gty, gtz]]

        gt = np.pad(gt, pad_values, mode='edge')
        pred = np.pad(pred, pad_values, mode='edge')
        mask = np.pad(mask, pad_values, mode='edge')
        
        min_size = (downsample_filter.shape[-1] - 1) * 2 ** (levels - 1) + 1

        ms_ssim_vals, ms_ssim_maps = [], []
        for level in range(levels):
            ssim_value_full, ssim_map = self.structural_similarity_at_scale(gt, pred, 
                luminance_weight=luminance_weights[level], 
                contrast_weight=scale_weights[level],
                structure_weight=scale_weights[level],
                data_range=dynamic_range, full=True)
            pad = 3
            # at every level, we get the ssim_value_full, which is just mean SSIM at a level L, and the 
            # SSIM map. The masked SSIM is the mean SSIM within this mask
            ssim_value_masked  = (crop(ssim_map, pad)[crop(mask, pad).astype(bool)]).mean(dtype=np.float64)

            ms_ssim_vals.append(ssim_value_full)
            ms_ssim_maps.append(ssim_value_masked)

            # The images are cleverly downsampled using an uniform filter
            # the mask is just downsampled by selecting every other line in every dimension
            filtered = [fftconvolve(im, downsample_filter, mode='same')
                for im in [gt, pred]]
            gt, pred, mask = [x[::2, ::2, ::2] for x in [*filtered, mask]]

        ms_ssim_val = np.prod([np.clip(x, a_min=0, a_max=1) for x in ms_ssim_vals])
        ms_ssim_mask_val = np.prod([np.clip(x, a_min=0, a_max=1) for x in ms_ssim_maps])

        return float(ms_ssim_val), float(ms_ssim_mask_val)



if __name__=='__main__':
    metrics = ImageMetrics()
    
    # synthrad2023
    #path_all = "/home/catarina_caldeira/Desktop/code/tests/mean_sCTs_generated_GaussianOverlap/SynthRAD2023/AE/8layers_unet_IN_800_lr_0.001_batch8/fold_all"
    #original_path = "/home/catarina_caldeira/Imagens/SynthRAD2023dataset/Task1/pelvis"
    
    # synthrad2025
    path_all = "/home/catarina_caldeira/Desktop/code/tests/mean_sCTs_generated_GaussianOverlap/SynthRAD2025/AE/8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_regionABHNTH/fold_all"
    
    #path_all="/home/catarina_caldeira/Desktop/code/tests/mean_sCTs_generated_GaussianOverlap_centers/SynthRAD2025/AE/8layers_unet_IN_1000_lr_0.001_batch8_SynthRAD2025_ABC/fold_all"
    original_path = "/home/catarina_caldeira/Imagens/SynthRAD2025/synthRAD2025_Task1_Train/Task1"
    
    list_c, list_a = [],[]
    results_dict = {}
    region_metrics = {}
    mae_list, psnr_list, ssim_list, ms_ssim_list = [],[],[],[]
    for patient_folder in os.listdir(original_path):
        
        patient_folder_path = os.path.join(original_path, patient_folder)
        
        # synthrad2023
        #ground_truth_path = os.path.join(patient_folder_path, f"ct.nii.gz")
        #mask_path = os.path.join(patient_folder_path, f"mask.nii.gz")
        
        # synthrad2025
        ground_truth_path = os.path.join(patient_folder_path, f"ct.mha")
        mask_path = os.path.join(patient_folder_path, f"mask.mha")
        
        if f"{patient_folder}.mha" not in os.listdir(path_all):
            continue
        
        predicted_path = os.path.join(path_all, f"{patient_folder}.mha")
        
        result = metrics.score_patient(ground_truth_path, predicted_path, mask_path)
        mae_list.append(result['mae'])
        psnr_list.append(result['psnr'])
        ssim_list.append(result['ssim'])
        ms_ssim_list.append(result['ms_ssim'])
        results_dict[patient_folder] = result
        
        region = patient_folder[1:3] 
        
        if region not in region_metrics:
            region_metrics[region] = {
                'mae': [],
                'psnr': [],
                'ssim': [],
                'ms_ssim': []
            }
            
        region_metrics[region]['mae'].append(result['mae'])
        region_metrics[region]['psnr'].append(result['psnr'])
        region_metrics[region]['ssim'].append(result['ssim'])
        region_metrics[region]['ms_ssim'].append(result['ms_ssim'])
        
        if patient_folder[2]=='C':
            list_c.append(result)
        else:
            list_a.append(result)
        
        print(f"{patient_folder} -> MAE: {result['mae']:.4f}, PSNR: {result['psnr']:.2f}, SSIM: {result['ssim']:.4f}, MS-SSIM: {result['ms_ssim']:.4f}")

    result_mean = {
            'mae': np.mean(mae_list),
            'std_mae': np.std(mae_list),
            'ssim': np.mean(ssim_list),
            'std_ssim': np.std(ssim_list),
            'ms_ssim': np.mean(ms_ssim_list),
            'std_ms_ssim': np.std(ms_ssim_list),
            'psnr': np.mean(psnr_list),
            'std_psnr': np.std(psnr_list)
        }
    results_dict["aggregates"] = result_mean
    
    for region, metrics_lists in region_metrics.items():
        region_mean = {
            'mae': np.mean(metrics_lists['mae']),
            'std_mae': np.std(metrics_lists['mae']),
            'ssim': np.mean(metrics_lists['ssim']),
            'std_ssim': np.std(metrics_lists['ssim']),
            'ms_ssim': np.mean(metrics_lists['ms_ssim']),
            'std_ms_ssim': np.std(metrics_lists['ms_ssim']),
            'psnr': np.mean(metrics_lists['psnr']),
            'std_psnr': np.std(metrics_lists['psnr'])
        }
        results_dict[f"aggregates_{region}"] = region_mean

    

    output_path = "metrics_results_25_docker_gaussianoverlap_centers_cgan.json"
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=4)
        
    # Statistical difference between centers in pelvis images (2023)
    '''
    from scipy import stats

    mae_a = [r['mae'] for r in list_a]
    mae_c = [r['mae'] for r in list_c]

    ssim_a = [r['ssim'] for r in list_a]
    ssim_c = [r['ssim'] for r in list_c]
    
    psnr_a = [r['psnr'] for r in list_a]
    psnr_c = [r['psnr'] for r in list_c]

    # Teste t não pareado (independent samples)
    def print_ttest(metric_a, metric_c, metric_name):
        result = stats.ttest_ind(metric_a, metric_c, equal_var=False)  
        
        print(f"\nResults for {metric_name}:")
        print("Mean A:", np.mean(metric_a), "| Mean C:", np.mean(metric_c))
        print("P-Value:", result.pvalue)
        
        if result.pvalue > 0.05:
            print("Não há evidência de diferença significativa entre os centros.")
        else:
            print("Há evidência de diferença significativa entre os centros!")

    # Realizar testes
    print_ttest(mae_a, mae_c, "MAE")
    print_ttest(ssim_a, ssim_c, "SSIM")
    print_ttest(psnr_a, psnr_c, "PSNR")
    '''
        
        
        
        
      
        
    