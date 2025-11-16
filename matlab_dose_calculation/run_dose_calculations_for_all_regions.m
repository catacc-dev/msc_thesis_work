% Main script to run dose calculations for multiple regions
function run_dose_calculations_for_all_regions()
    addpath(genpath('C:\Users\catar\OneDrive - Universidade de Coimbra\Ambiente de Trabalho\Master Thesis\e0404-matRad-98ba2fb\matRad'));
    % Define base directory where patient data is stored
    base_dir = 'C:\Users\catar\OneDrive - Universidade de Coimbra\Ambiente de Trabalho\Master Thesis\e0404-matRad-98ba2fb\userdata\patients';
    
    % Define regions and their corresponding segment names
    regions = {
        'TH', {'spinal_cord', 'esophagus', 'lung_upper_lobe_right','lung_middle_lobe_right','lung_lower_lobe_right',...
        'lung_upper_lobe_left','lung_lower_lobe_left'};
        'AB', {'liver', 'kidney_left', 'kidney_right', 'urinary_bladder', 'spinal_cord', 'stomach'}; 
        'HN', {'spinal_cord','esophagus','thyroid_gland', ...
               'common_carotid_artery_right','common_carotid_artery_left', ...
               'brainstem','optic_nerve_left','optic_nerve_right', ...
               'submandibular_gland_right','submandibular_gland_left','larynx_air', ...
               'eye_lens_right','eye_lens_left', ...
               'masseter_right','masseter_left','parotid_gland_left', ...
               'parotid_gland_right','superior_pharyngeal_constrictor', ...
               'middle_pharyngeal_constrictor','inferior_pharyngeal_constrictor',...
               'tongue', 'hard_palate', 'soft_palate'}
    };
    
    % Loop through each region 
    for i = 1:size(regions, 1)
        region = regions{i, 1};
        segment_names = regions{i, 2};
        
        % Find all patient directories for this region
        patient_dirs = dir(fullfile(base_dir, ['1' region '*']));
        disp(patient_dirs)
        
        % Process each patient in this region
        for j = 1:length(patient_dirs)
            patient_dir = patient_dirs(j).name;
            patient_id = patient_dir; 

            if strcmp(region,'TH')
                if strcmp(patient_id, '1THA028') 
                

                % Construct paths
                path_to_sct = fullfile(base_dir, patient_dir, [patient_id '.mha']); % sCT
                path_to_ct = fullfile(base_dir, patient_dir, 'ct.mha'); % CT
                path_to_segments = fullfile(base_dir, patient_dir, [patient_dir '_seg_matrad']);
                
                % Check if files exist
                if ~exist(path_to_sct, 'file') || ~exist(path_to_ct, 'file') || ~exist(path_to_segments, 'file')
                    warning('Files not found for patient %s. Skipping...', patient_id);
                    continue;
                end

                fprintf('Processing patient: %s\n', patient_id);

                % Run dose calculation for this patient
                [ct_ct, ~] = dose_calculations_regions(path_to_ct, path_to_segments, region, segment_names);
                [ct_sct, cst] = dose_calculations_regions(path_to_sct, path_to_segments, region, segment_names);

                % Assign target 
                if strcmp(region, 'TH')
                    target_name = 'lung_upper_lobe_right'; 
                    cst = assigned_target(target_name, cst);
                elseif strcmp(region, 'AB')
                    target_name = 'stomach'; 
                    cst = assigned_target(target_name, cst);
                elseif strcmp(region, 'HN')
                    target_names = {'tongue', 'hard_palate', 'soft_palate'};
                    % Assign each target sequentially
                    for i = 1:length(target_names)
                        cst = assigned_target(target_names{i}, cst);
                    end
                end
                
                process_patient(path_to_sct, ct_sct, ct_ct, cst, region, patient_id);
                end
            end 
        end
      
    end
end


function process_patient(path_to_image, ct_sct, ct_ct, cst, region_choosen, patient_id)
    %% Treatment planning setup

    %pln.numOfFractions = 30; % AB and HN
    pln.numOfFractions = 35; % TH
    pln.radiationMode = 'protons'; 
    pln.machine = 'Generic'; % machine model used for treatment delivery          
    pln.bioModel = 'constRBE'; % biological model to calculate relative biological effectiveness (RBE)
    pln.multScen = 'nomScen'; % scenario type for treatment planning ('nomScen' = nominal scenario)
    
    % Beam geometry settings
    pln.propStf.bixelWidth = 5; % [mm] corresponds to lateral spot spacing for particles (protons)
    pln.propStf.gantryAngles = [0 120 240]; % [°] gantry angles at which beams are delivered
    pln.propStf.couchAngles = [0 0 0]; % [°] patient position for each beam
    pln.propStf.numOfBeams = numel(pln.propStf.gantryAngles); 
    pln.propStf.isoCenter = ones(pln.propStf.numOfBeams,1) * matRad_getIsoCenter(cst,ct_sct,0);
    
    % Optimization and sequence flags (0=disabled)
    pln.propOpt.runDAO = 0; % dose and optimization algorithm flag
    pln.propSeq.runSequencing = 0; % sequencing algorithm flag for treatment delivery order

    % Dose calculation algorithm settings
    pln.propDoseCalc.calcLET = 0; % flag to calculate Linear Energy Transfer (LET), 0 = off
    pln.propDoseCalc.engine = 'HongPB'; % using Hong Pencil Beam dose calculation algorithm
    pln.propDoseCalc.doseGrid.resolution.x = 3; % [mm]
    pln.propDoseCalc.doseGrid.resolution.y = 3; % [mm]
    pln.propDoseCalc.doseGrid.resolution.z = 3; % [mm]
    
    % Optimization quantity
    pln.propOpt.quantityOpt = 'RBExDose';   % optimize relative biological effectiveness weighted dose
    
    %% Set objectives and constraints based on region
    for k = 1:size(cst,1)
        segment_name = cst{k,2};
        
        if strcmp(cst{k,3},'TARGET')
            % Target objectives
            if strcmp(region_choosen, 'TH') && strcmp(segment_name, 'lung_upper_lobe_right')
                cst{k,6} = {struct(DoseObjectives.matRad_SquaredDeviation(1000,70))};
            elseif strcmp(region_choosen, 'AB') && strcmp(segment_name, 'stomach')
                cst{k,6} = {struct(DoseObjectives.matRad_SquaredDeviation(1000,60))};
            elseif strcmp(region_choosen, 'HN') && (strcmp(segment_name, 'tongue') || ...
                   strcmp(segment_name, 'hard_palate') || ...
                   strcmp(segment_name, 'soft_palate'))
                cst{k,6} = {struct(DoseObjectives.matRad_SquaredDeviation(1000,60))};
            end
        else
            % OAR objectives
            if strcmp(region_choosen, 'TH')
                oars_lung = {'lung_middle_lobe_right','lung_lower_lobe_right','lung_upper_lobe_left','lung_lower_lobe_left'};
                if strcmp(segment_name, 'esophagus')
                    cst{k,6} = {
                        struct(DoseConstraints.matRad_MinMaxDose(0, 73.5)),  % D0.03cc < 73.5 Gy
                        struct(DoseConstraints.matRad_MinMaxDVH(60, 0, 15.3)),   % V60 < 15.3%
                        struct(DoseConstraints.matRad_MinMaxMeanDose(0, 30.6)) % Dmean < 30.6 Gy
                    };
                    cst{k,6}{1}.penalty = 300;
                    cst{k,6}{2}.penalty = 300;
                    cst{k,6}{3}.penalty = 300;
                elseif strcmp(segment_name, 'spinal_cord')
                    cst{k,6} = {struct(DoseConstraints.matRad_MinMaxDose(0, 50.5))};
                    cst{k,6}{1}.penalty = 300;
                elseif ismember(segment_name, oars_lung)
                    cst{k,6} = {
                        struct(DoseConstraints.matRad_MinMaxDVH(60, 0, 33)),  % V60 < 33%
                        struct(DoseConstraints.matRad_MinMaxDVH(5, 0, 33)),   % V5 < 33%
                        struct(DoseConstraints.matRad_MinMaxMeanDose(0, 18)) % Dmean < 18 Gy
                    };
                    cst{k,6}{1}.penalty = 300;
                    cst{k,6}{2}.penalty = 300;
                    cst{k,6}{3}.penalty = 300;
                end
            elseif strcmp(region_choosen, 'AB')
                % Add AB-specific constraints here
                if strcmp(segment_name, 'spinal_cord')
                    cst{k,6} = {struct(DoseConstraints.matRad_MinMaxDose(0, 50.5))};
                    cst{k,6}{1}.penalty = 300;
                elseif strcmp(segment_name, 'liver')
                    cst{k,6} = {struct(DoseConstraints.matRad_MinMaxMeanDose(0, 45))};
                    cst{k,6}{1}.penalty = 300;
                elseif strcmp(segment_name, 'kidney_right')
                    cst{k,6} = {struct(DoseConstraints.matRad_MinMaxMeanDose(0, 18))};
                    cst{k,6}{1}.penalty = 300;
                elseif strcmp(segment_name, 'kidney_left')
                    cst{k,6} = {struct(DoseConstraints.matRad_MinMaxMeanDose(0, 18))};
                    cst{k,6}{1}.penalty = 300;
                elseif strcmp(segment_name, 'urinary_bladder')
                    cst{k,6} = {struct(DoseConstraints.matRad_MinMaxDose(0, 65))};
                    cst{k,6}{1}.penalty = 300;
                end
            elseif strcmp(region_choosen, 'HN')
                % Add HN-specific constraints here
                if strcmp(segment_name, 'brainstem')
                    cst{k,6} = {struct(DoseConstraints.matRad_MinMaxDose(0, 54))};
                    cst{k,6}{1}.penalty = 300;
                elseif strcmp(segment_name, 'spinal_cord')
                    cst{k,6} = {struct(DoseConstraints.matRad_MinMaxDose(0, 48))};
                    cst{k,6}{1}.penalty = 300;
                elseif strcmp(segment_name, 'optic_nerve_left')
                    cst{k,6} = {struct(DoseConstraints.matRad_MinMaxDose(0, 54))};
                    cst{k,6}{1}.penalty = 300;
                elseif strcmp(segment_name, 'optic_nerve_right')
                    cst{k,6} = {struct(DoseConstraints.matRad_MinMaxDose(0, 54))};
                    cst{k,6}{1}.penalty = 300;
                elseif strcmp(segment_name, 'parotid_gland_left')
                    cst{k,6} = {struct(DoseConstraints.matRad_MinMaxMeanDose(0, 25))};
                    cst{k,6}{1}.penalty = 300;
                elseif strcmp(segment_name, 'parotid_gland_right')
                    cst{k,6} = {struct(DoseConstraints.matRad_MinMaxMeanDose(0, 25))};
                    cst{k,6}{1}.penalty = 300;
                elseif strcmp(segment_name, 'larynx_air')
                    cst{k,6} = {struct(DoseConstraints.matRad_MinMaxMeanDose(0, 40))};
                    cst{k,6}{1}.penalty = 300;
                elseif strcmp(segment_name, 'submandibular_gland_right')
                    cst{k,6} = {struct(DoseConstraints.matRad_MinMaxMeanDose(0, 39))};
                    cst{k,6}{1}.penalty = 300;
                elseif strcmp(segment_name, 'submandibular_gland_left')
                    cst{k,6} = {struct(DoseConstraints.matRad_MinMaxMeanDose(0, 39))};
                    cst{k,6}{1}.penalty = 300;
                elseif strcmp(segment_name, 'superior_pharyngeal_constrictor')
                    cst{k,6} = {struct(DoseConstraints.matRad_MinMaxMeanDose(0, 50))};
                    cst{k,6}{1}.penalty = 300;
                elseif strcmp(segment_name, 'inferior_pharyngeal_constrictor')
                    cst{k,6} = {struct(DoseConstraints.matRad_MinMaxMeanDose(0, 50))};
                    cst{k,6}{1}.penalty = 300;
                 elseif strcmp(segment_name, 'middle_pharyngeal_constrictor')
                    cst{k,6} = {struct(DoseConstraints.matRad_MinMaxMeanDose(0, 50))};
                    cst{k,6}{1}.penalty = 300;
                elseif strcmp(segment_name, 'eye_lens_right')
                    cst{k,6} = {struct(DoseConstraints.matRad_MinMaxDose(0, 5))};
                    cst{k,6}{1}.penalty = 300;
                elseif strcmp(segment_name, 'eye_lens_left')
                    cst{k,6} = {struct(DoseConstraints.matRad_MinMaxDose(0, 5))};
                    cst{k,6}{1}.penalty = 300;
                elseif strcmp(segment_name, 'esophagus')
                    cst{k,6} = {struct(DoseConstraints.matRad_MinMaxMeanDose(0, 30))};
                    cst{k,6}{1}.penalty = 300;
                elseif strcmp(segment_name, 'thyroid_gland')
                    cst{k,6} = {struct(DoseConstraints.matRad_MinMaxMeanDose(0, 40))};
                    cst{k,6}{1}.penalty = 300;
                elseif strcmp(segment_name, 'masseter_left')
                    cst{k,6} = {struct(DoseConstraints.matRad_MinMaxMeanDose(0, 35))};
                    cst{k,6}{1}.penalty = 300;
                elseif strcmp(segment_name, 'masseter_right')
                    cst{k,6} = {struct(DoseConstraints.matRad_MinMaxMeanDose(0, 35))};
                    cst{k,6}{1}.penalty = 300;
                elseif strcmp(segment_name, 'common_carotid_artery_left')
                    cst{k,6} = {struct(DoseConstraints.matRad_MinMaxDose(0, 40))};
                    cst{k,6}{1}.penalty = 300;
                elseif strcmp(segment_name, 'common_carotid_artery_right')
                    cst{k,6} = {struct(DoseConstraints.matRad_MinMaxDose(0, 40))};
                    cst{k,6}{1}.penalty = 300;
                end
            end
        end
    end
    
    %% Generate Beam Geometry STF
    stf = matRad_generateStf(ct_sct,cst,pln);
    
    %% Dose Calculation
    dij_sct = matRad_calcDoseInfluence(ct_sct,cst,stf,pln);
    
    %% Optimization and plan analysis - sCT 
    resultGUI_sct = matRad_fluenceOptimization(dij_sct, cst, pln);
    resultGUI_sct = matRad_planAnalysis(resultGUI_sct, ct_sct, cst, stf, pln);

    %% Recalculate Plan - CT
    % Let's use the existing optimized pencil beam weights and recalculate the RBE weighted dose
    resultGUI_ct = matRad_calcDoseForward(ct_ct,cst,stf,pln,resultGUI_sct.w);
    resultGUI_ct = matRad_planAnalysis(resultGUI_ct, ct_ct, cst, stf, pln);
    %disp(resultGUI.dvh);
    %disp(resultGUI.qi);

    %% Gamma index

    % Let's plot the transversal iso-center dose slice
    slice = matRad_world2cubeIndex(pln.propStf.isoCenter(1,:),ct_ct);
    slice = slice(3);

    doseDifference  = 2;
    distToAgreement = 2;
    n               = 1;
    
    [gammaCube,gammaPassRateCell] = matRad_gammaIndex(...
        resultGUI_ct.RBExDose,resultGUI_sct.RBExDose,...
        [ct_ct.resolution.x, ct_ct.resolution.y, ct_ct.resolution.z],...
        [doseDifference distToAgreement],slice,n,'global',cst);
    %% matrad

    
    %matRadGUI
        
    %% Save results
    save(fullfile(fileparts(path_to_image), [ patient_id '_ct_and_sct_plan.mat']), 'resultGUI_sct','resultGUI_ct', 'gammaPassRateCell','gammaCube','-v7.3');
    
    fprintf('Completed processing for patient: %s\n', patient_id);
end