clearvars
close all
clc

%% FUNCTIONAL MRI ANALYSIS

%% Load data
base_path = pwd;
addpath(genpath(fullfile(base_path, 'Matlab_tools')))
addpath(genpath(fullfile(base_path, 'Structural')))
addpath(genpath(fullfile(base_path, 'Matlab_tools', 'Nifti_tools')))

addpath(genpath(fullfile(base_path, 'Results')))
gm = load_untouch_nii('c1T1_2mm.nii');
gm   = double(gm.img);
wm = load_untouch_nii('c2T1_2mm.nii');
wm   = double(wm.img);
csf = load_untouch_nii('c3T1_2mm.nii');
csf  = double(csf.img);

addpath(genpath(fullfile(base_path, 'FMRI')))
epi = load_untouch_nii('FMRI_2_T1_2mm.nii.gz');
epi = double(epi.img);

addpath(genpath(fullfile(base_path, 'Atlas')))
atlas_mask = load_untouch_nii('Hammers_2_T1_2mm_int.nii.gz');
atlas_mask = double(atlas_mask.img);

%% 1. DATA PREPROCESSING

%% a.

% Obtain the tissue probability maps of gm, wm and csf using SPM

%% b.

% Threshold the three tissues probability maps and create for each tissue a binary mask

% figure(), histogram(gm(:),40), title('histogram gray matter')
threshold = 255*0.45;
gm = gm > threshold;
% implay(gm)

% figure(), histogram(wm(:),40), title('histogram white matter')
threshold = 255*0.95;
wm = wm > threshold;
% implay(wm)

% figure(), histogram(csf(:),40), title('histogram cerebrospinal fluid')
threshold = 255*0.85;
csf = csf > threshold;
% implay(csf)

% Provide a justification of the employed thresholds:
% by looking at the histograms we tought that a threshold of 0.45 for the
% gm, 0.95 for wm and 0.85 for csf could be reasonable and then
% from the visualization of the masks so obtained we didn't see isolated voxel
% and so we set the thresholds to these values

% Erode the masks (imerode matlab function)
se = strel('square', 2);
wm = imerode(wm,se);
% implay(wm)

csf = imerode(csf,se);
% implay(csf)

% Provide a justification of the erosion parameters:
% by visual inspection the optimal dimension of structuring element is 2 pixel 
% for the erosion of the masks because it erodes well the isolated pixels
% but doesn't delete too much
% we didn't erode the gray matter mask because it was not necessary

% Compute the mean fMRI signal of WM and CSF
idx_mask_WM = find(wm);
idx_mask_CSF = find(csf);

[nR, nC, nS, nVol] = size(epi);
data2D_WM = zeros([nVol, length(idx_mask_WM)]);
data2D_CSF = zeros([nVol, length(idx_mask_CSF)]);

for t = 1: nVol
    tmp = squeeze(epi(:,:,:,t)); % volume selection
    data2D_WM(t,:)  = tmp(idx_mask_WM); % wm tac extraction
    data2D_CSF(t,:) = tmp(idx_mask_CSF); % csf tac extraction
end

mean_WM_signal = nanmean(data2D_WM,2); % mean wm signal
mean_CSF_signal = nanmean(data2D_CSF,2); % mean csf signal

clear wm csf threshold se idx_mask_WM idx_mask_CSF t tmp data2D_WM data2D_CSF

%% c.

% Create the sumEPI image
sumepi = sum(epi,4);
% implay(sumepi/max(sumepi(:)))

% Create a binary mask for the sumEPI image
% figure(), histogram(sumepi(:),80), title('histogram sumepi')
threshold = 100000;
mask_sumepi = sumepi > threshold;
% implay(mask_sumepi)

% Provide a justification of the employed threshold:
% by looking at histogram we set the threshold at 100000 because it
% makes a clear separation between the two intensity groups

clear threshold

%% d.

% Mask the Hammers atlas with the GM binary mask and the sumEPI mask
mask_fin = atlas_mask .* gm;
% implay(mask_fin/max(mask_fin(:)))

mask_fin = mask_fin .* mask_sumepi;
% implay(mask_fin/max(mask_fin(:)))

clear gm mask_sumepi

%% e.

% ROI time activity curve extraction: for each masked ROI of the atlas, extract the mean
% fMRI signal

% Using the Hammers_labels.pdf file, discard for the following analyses the
% masked ROIs with less then 10 voxels and those belonging from: amygdala,
% cerebellum, brainstem, corpus callosum, substantia nigra, ventricles

idx_exl = [];
for i = 1:max(atlas_mask(:))
    if sum(mask_fin(:) == i) < 10
        idx_exl = [idx_exl i];
    end
end

% amyg (3 4); cer (17 18); bstem (19); cc (44); snigra (74 75); ventr (45 46 47 48 49)
idx_exl = [idx_exl 3 4 17 18 19 44 45 46 47 48 49 74 75];

idx_ROI = setdiff(1:max(atlas_mask(:)),idx_exl);
nROI = length(idx_ROI);
mean_ROI_TAC_original = zeros(nVol,nROI);

for i = 1:nROI
    
    name_ROI = idx_ROI(i);
    disp(['Working on ROI #' num2str(name_ROI)])
    
    % ROI time activity curve extraction (GM only)
    mask_ROI = (mask_fin == idx_ROI(i));
    data2D_GM  = zeros(nVol, sum(mask_ROI(:))); % TAC before denoising
    
    for t = 1: nVol
        tmp = squeeze(epi(:,:,:,t));
        data2D_GM(t,:) = tmp(logical(mask_ROI));
    end
    
    mean_ROI_TAC_original(:,i)  = nanmean(data2D_GM,2); % ROIs time courses before denoising
end

% we calculate TAC for gm only because we are interested in studing the
% functional connectivity, which focuses on gm

clear atlas_mask epi sumepi idx_exl i name_ROI mask_ROI t tmp data2D_GM

%% 2. DATA DENOISING

%% a. Noise regression

% For each ROI, remove the non-neural undesidered fluctuations from its temporal
% dynamic using a linear regression approach.
load('MOCOparams')

disp('Regress out motion and WM and CSF')
GC = [newMOCOparams mean_WM_signal mean_CSF_signal]; % Regression matrix (225 volumes x 8 regressors)
GC = zscore(GC); % Z-score of GC
beta = (GC'*GC)\GC'*mean_ROI_TAC_original; % LLS beta estimates
mean_ROI_TAC_regressed = mean_ROI_TAC_original - GC*beta; % residuals --> cleaned data

% Visualize the regression matrix
figure()
imagesc(GC)
title('regression matrix')
colormap gray
set(gcf, 'Position', get(0, 'Screensize'));
set(gca,'fontsize',16,'fontweight','bold')
colorbar

clear newMOCOparams mean_WM_signal mean_CSF_signal beta

%% b. Temporal filtering

% To take the slow components out, filter the signals obtained at point 2.a. with a high 
% pass filter choosing reasonable cut-off frequencies

TR = 2.6; % sec
freq_hp = 1/128; % [Hz]

% compute the spectrum of the regressed signal for the first ROI through fft
Fs = 1/TR;
f_FT = (0:Fs/nVol:Fs-Fs/nVol);
mean_ROI_TAC_regressed_firstROI = mean_ROI_TAC_regressed(:,1) - mean(mean_ROI_TAC_regressed(:,1));
FTx= fft(mean_ROI_TAC_regressed_firstROI,nVol);
S= (abs(FTx).^2)/nVol;
% FFT has even abs: plot 1:N/2
S = S(1:round(nVol/2));
f_FT = f_FT(1:round(nVol/2));
figure()
[b,a] = butter(3,[freq_hp]/(Fs/2),'high');
freqz(b,a,512,Fs)
hold on
subplot(2,1,1)
title('magnitude of the filter')
subplot(2,1,2)
plot(f_FT,S,'r')
xlim([f_FT(1) f_FT(end)])
xlabel('Frequency (Hz)')
title("spectrum of the first ROI's TAC")
grid on

flag=0; % set to 1 if we want to plot the frequency response of the filter
mean_ROI_TAC_filtered = hp_filter(mean_ROI_TAC_regressed,TR,freq_hp,flag);

% Provide a justification of the selected filtering frequency cut-off:
% we chose the cut-off frequency so that the filter doesn't remove frequencies related to the
% related to the neuronal activity (0.08-0.1 Hz). Furthermore 1/128 is the standard cut-off 
% frequency used by default in SPM12.

clear Fs f_FT freq_hp mean_ROI_TAC_regressed_firstROI FTx S b a

%% 3. VOLUME CENSORING

load('FDparams.mat');
FD = FD(:,1);
FD_thr = 0.35;
vol_cens = find(FD > FD_thr);

% For each identified volume to discard, remove also one before and two after volumes
vol_cens = unique([vol_cens-1, vol_cens, vol_cens+1, vol_cens+2]);
vol_final = setdiff(1:nVol,vol_cens);

% Update the number of volumes of the epi data matrix
nVol_reduced = length(vol_final);

% Exclusion of the censored volumes from 2D regressed data
mean_ROI_TAC = mean_ROI_TAC_filtered(vol_final,:);

clear FD FD_thr vol_cens

%% 4. CHECK OF PREPROCESSING STEP

% Plot the original time-course of the right hippocampus region and what obtained after each
% denoising step (noise regression and temporal filtering)

number_ROI_hippo = 1;  % idx ROI Hippocampus = 1
idx_ROI_selected=find(idx_ROI==number_ROI_hippo);

figure()
hold on
plot([1:TR:nVol*TR]/60 , mean_ROI_TAC_original(:,idx_ROI_selected) - mean(mean_ROI_TAC_original(:,idx_ROI_selected)),'LineWidth',2)
plot([1:TR:nVol_reduced*TR]/60 , mean_ROI_TAC(:,idx_ROI_selected)-mean(mean_ROI_TAC(:,idx_ROI_selected)),'LineWidth',2)
title('hippo time course after performing the denoising steps')
legend('original epi signal','preprocessed epi signal')
xlabel('time [min]')
set(gcf, 'Position', get(0, 'Screensize'));
set(gca,'fontsize',16,'fontweight','bold')

% Do you see a drift in the original signal? If so, is the denoising able
% to remove it?
% yes, we can see a drift in the original signal and the denoising has been
% able to remove it

clear number_ROI_hippo idx_ROI_selected

%% 5. STATIC FC MATRIX COMPUTATION
% Compute the pairwise Pearson’s correlation (and the relative p-value) between the timeseries
% of the ROIs. Visualize the FC matrix after applying the Fisher’s z-transform to the
% coefficients (atanh matlab function).
[FC, p] = corr(mean_ROI_TAC); % Pearson's correlation
zFC = atanh(FC); % Fisher's z-transform

figure()
subplot(1,2,1)
imagesc(zFC)
colorbar
axis square
colormap jet, caxis([-1 2]), colorbar
title('z-Fisher transformed FC matrix')
set(gca,'fontsize',16,'fontweight','bold')

subplot(1,2,2)
imagesc(p)
colorbar
axis square
colormap jet, caxis([0 1]), colorbar
title('p-values matrix')
set(gca,'fontsize',16,'fontweight','bold')
set(gcf, 'Position', get(0, 'Screensize'));

%% 6. Multiple Comparison Correction

% Perform a multiple comparison correction with Bonferroni or False Discovery Rate approach
% and a significance level alpha=0.05. If you decide to use the FDR method, use the provided
% fdr_bh.m function. Provide a justification of the chosen correction method.

% trying fdr
addpath(genpath(fullfile(base_path,'Matlab_tools','fdr_bh.mat')))
alpha=0.05; % significance level
h_fdr = fdr_bh(p,alpha,'dep','yes');
N_fdr = sum(h_fdr(:));
kept_fdr = N_fdr/(size(zFC,1)*size(zFC,2)); % = 0.3075

% trying bonferroni
number_tests = length(p(:)); 
h_bonferroni = p<(alpha / number_tests);
N_bonferroni = sum(h_bonferroni(:));
kept_bonferroni = N_bonferroni/(size(zFC,1)*size(zFC,2)); % = 0.1916 % bonferroni allows a spasification of almost 20% 

% sparsification
zFC_corr = zFC;
zFC_corr(~h_bonferroni) = nan;
% zFC_corr = zFC(h_bonferrini);

% binarization
zFC_binary = ~isnan(zFC_corr);

% Provide a justification of the chosen correction method:
% we chose Bonferroni correction because it allows to get a sparsification
% of the FC matrix of almost 20% which is a suggested value in literature

clear alpha h_fdr N_fdr kept_fdr h_bonferroni N_bonferroni kept_bonferroni number_tests

%% 7. GRAPH MEASURES

% To summarize the functional connectivity in terms of node centrality, for each ROI compute
% the node degree, the node strength and the normalized betweenness centrality (with
% betweenness_wei.m provided function). In the metrics computation, consider only the
% statistically significative functional connections obtained after the multiple comparison
% correction at point 6. Plot the node degree, the strength and the normalized betweenness
% centrality of the ROIs using the stem matlab function. Which are for each metric the 10 ROIs
% with the higher metrics values? Provide the indices of these regions.

addpath(genpath(fullfile(base_path,'Matlab_tools','betweenness_wei.mat')))

% node degree
node_degree=sum(zFC_binary, 'omitnan')';

% node strength
node_strength = sum(zFC_corr, 'omitnan')';

% normalized betweenness centrality 
% Node betweenness centrality is the fraction of all shortest paths in 
% the network that contain a given node. Nodes with high values of 
% betweenness centrality participate in a large number of shortest paths.

% since higher correlations are more naturally interpreted as shorter distances
% the input matrix should be some inverse of the connectivity matrix:
G = 1./zFC_corr;
% G(G == -inf) = +inf;
BC = betweenness_wei(G);

% Betweenness centrality may be normalised to the range [0,1] as
% BC/[(N-1)(N-2)], where N is the number of nodes in the network.
BC = BC/((nROI - 1)*(nROI - 2));

figure
subplot(3,1,1)
stem(idx_ROI,node_degree, 'LineWidth', 1.2)
title('node degree')
xlabel('ROIs')
subplot(3,1,2)
stem(idx_ROI,node_strength, 'LineWidth', 1.2)
title('node strength')
xlabel('ROIs')
subplot(3,1,3)
stem(idx_ROI,BC, 'LineWidth', 1.2)
title('normalized betweenness centrality')
xlabel('ROIs')

[ ~, degree_idx] = sort(node_degree, 'descend');
degree_idx =  degree_idx(1:10);
DEG = idx_ROI(degree_idx); % 31    30    58    82    50    57    59    27    62    67
[ ~, strength_idx] = sort(node_strength, 'descend');
strength_idx =  strength_idx(1:10);
STR = idx_ROI(strength_idx); % 31    58    59    50    61    20    25    68    30    82
[ ~, BC_idx] = sort(BC, 'descend');
BC_idx =  BC_idx(1:10);
BTW_NORM = idx_ROI(BC_idx); % 55    52    21    22    71    83    62    63    64     6

% the numbers above are the indexes for each metric  of the 10 ROIs with
% the higher values

% [r,p] = corr(node_degree, node_strength)
% [r,p] = corr(node_degree, BC)
% [r,p] = corr(node_strength, BC)

clear degree_idx strength_idx BC_idx G

%% DIFFUSION MRI ANALYSIS
%% 1. Diffusion signal visualization & understanding
%% a.

% Load the diffusion volumes, the bvals file and the bvecs file
base_path = pwd;
addpath(genpath(fullfile(base_path, 'Matlab_tools')))
addpath(genpath(fullfile(base_path, 'Matlab_tools', 'Nifti_tools')))

load('DMRI/bvals')
load('DMRI/bvecs')
dMRI_volumes=load_untouch_nii('DMRI/diffusion_volumes.nii');
dMRI_volumes=double(dMRI_volumes.img);
% size(dMRI_volumes) % 120   120    90   103

% How many different DWIs have been acquired?
nVols=size(dMRI_volumes,4); % 103
nSlices=size(dMRI_volumes,3);
nVox=size(dMRI_volumes,1);

% Excluding b=0, how many diffusion shells does this acquisition feature? (consider a
% small tolerance ±20 s/mm^2 in the shell definition)
shell_values=0:20:round(max(bvals))+100; % grid of shell values considering a small tolerance of ±20 s/mm^2 in the shell definition
shell_used=zeros(1,length(shell_values)); % = 0 if the corresponding value is unused
for i=1:nVols
    for j=1:length(shell_values)
        if bvals(i)>shell_values(j)-10 && bvals(i)<=shell_values(j)+10
            shell_used(j)=1; % =1 if the corresponding value is used
        end
    end
end

shell_values=shell_values(shell_used==1); % overwrite keeping only the values used
nShells=length(shell_values)-1; % excluding b = 0, 2

clear shell_used i j

%% b.

% select a voxel populated principally with cerebrospinal fluid
voxel_CSF_selected=[52,60,47];
% this voxel has been selected taking into account for several aspects:
% 1) DTI metrics, indeed FA value at theese coordinates is low (FA(52,60,47)=0.0262989077756403) as expected from a voxel populated principally with CSF.
% 2) a CSF mask created by applying a threshold of 85% and an erosion with a square structural element of size equal to 2
% to the probability matrix (the CSF one) obtained through the SPM software in the 'Resting state fMRI analysis'.
% 3) the anatomical atlas at theese coordinates has value 45 that corresponds to lateral ventricle,
% and ventricles are regions where the cerebrospinal fluid flows.


% Plot the diffusion signal of the selected voxel
signal_CSF_selected=squeeze(dMRI_volumes(voxel_CSF_selected(1),voxel_CSF_selected(2),voxel_CSF_selected(3),:));
figure()
plot(signal_CSF_selected,'o-')
title(['plot of unsorted signal for voxel  ',num2str(voxel_CSF_selected)])
xlim([0 length(bvals)])
ylabel('diffusion value')
% is the diffusion signal of this voxel ordered by its b-value?
% If not, sort it so that the signal points corresponding to the same shell
% are shown consequently (and shells are ordered in an ascending fashion).

% Sort the diffusion signal of the selected voxel as required
[~,idx_sort]=sort(bvals);
signal_CSF_selected_sorted=signal_CSF_selected(idx_sort);
figure()
plot(signal_CSF_selected_sorted,'o-')
title(['plot of sorted signal for voxel  ',num2str(voxel_CSF_selected)])
xlim([0 length(bvals)])
ylabel('diffusion value')
clear voxel_CSF_selected signal_CSF_selected idx_sort

%% c.
% By visually inspecting the sorted signal, provide a brief comment both on the inter
% b-value and on the intra b-value variabilities. Why do these signal variations occur?

% The intra b-value variability of the signal is probably due to random effects since
% there is the assumption of considering diffusion as a random walk, that in case of
% isotropic diffusion is described by a diffusion propagator with Gaussian probability distribution

% The inter b-value variability of the signal is due to the law behind diffusion MRI signal which is S=S0*exp(-b*D)
% and so the higher the b-value, the lower the signal acquired

%% 2. Diffusion tensor computation
%% a.
% From the entirety of the diffusion volumes data, create a new 4D matrix containing
% only the volumes corresponding to b=0 s/mm2 and to the shell closest to b=1000
% s/mm^2 identified at the point 1a.
diff_values=abs(shell_values(2:end)-1000);
i=find(diff_values==min(diff_values));
shell_closest=shell_values(i+1);
idx_valid=zeros(1,nVols); % =0 if the volume does not correspond neither to b=0 nor to b=shell_closest
for i=1:nVols
    if bvals(i)==0 || (bvals(i)>shell_closest-10 && bvals(i)<=shell_closest+10)
        idx_valid(i)=1; % =1 if the volume correspond to b=0 or to b=shell_closest
    end
end
dMRI_volumes_reduced=dMRI_volumes(:,:,:,idx_valid==1); % 4D matrix reduced
bvals_reduced=bvals(idx_valid==1); % there is the need of reducing also bvals and bvecs for next steps
bvecs_reduced=bvecs(:,idx_valid==1);
nVols_reduced=size(dMRI_volumes_reduced,4);

clear diff_values shell_closest idx_valid i dMRI_volumes bvals bvecs

%% b.
% Fit the voxel-wise diffusion tensor using the linear least square approach
% on the whole brain diffusion data created at point 2a.

% When performing the log(S/S0) transformation of the signal, use as S0 the voxel-wise mean value of all
% b=0 volumes of the available dataset.

% Use the eigenvalue/eigenvector decomposition to recover the FA / MD indices.

% extract S0 image
S0=mean(dMRI_volumes_reduced(:,:,:,bvals_reduced==0),4);
Slog=zeros(nVox,nVox,nSlices,nVols_reduced);
for i=1:nVols_reduced
    Slog(:,:,:,i)=log((dMRI_volumes_reduced(:,:,:,i)./S0)+eps); % log of the normalized signal for the fit
end

% loading of the brain mask
brain_mask=load_untouch_nii('DMRI/diffusion_brain_mask.nii');
brain_mask=brain_mask.img > 0.5; % the mask before were a binary matrix but its elements were not logic elements

implay(brain_mask) % sagittal visualization
figure % sagittal visualization
for i = 1:90
    imagesc(brain_mask(:,:,i))
    pause(0.1)
end
close

figure % assial visualization
for i = 1:120
    imagesc(squeeze(brain_mask(:,i,:)))
    pause(0.1)
end
close

figure % coronal visualization
for i = 1:120
    imagesc(squeeze(brain_mask(i,:,:)))
    pause(0.1)
end
close

% size(brain_mask) % 120   120    90 % y, z, x

% implay(S0)
% num = S0==0;
% sum(num(:))
brain_mask = brain_mask & S0>0; % removing voxels for which S0=0 due to an error in the acquisition step
% There are some voxel fow which the average signal for b = 0 (and so they are constant and equal to 0!), probably due to an error
% in the acquisition step, so we decided to remove them.

brain_mask(:,27,:)=false; % removing voxels for which there was an error in the acquisition step
% in the further steps we have to compute the MD maps and looking at implay(MD) we noticed that theese voxels (that are
% peripheral voxels in the brain mask) assume unusually high values, probably due to an error
% in the acquisition step, so we decided to remove them to avoid that they influence the results.

% build the B design matrix for the linear least squares approach
B=zeros(nVols_reduced,6);
B(:,1:3)=bvecs_reduced'.^2;
B(:,4)=bvecs_reduced(1,:).*bvecs_reduced(2,:);
B(:,5)=bvecs_reduced(1,:).*bvecs_reduced(3,:);
B(:,6)=bvecs_reduced(2,:).*bvecs_reduced(3,:);
B=B.*bvals_reduced';
% initialize the structures which will be used to contain DTI indexes
FA=zeros(nVox,nVox,nSlices);
MD=zeros(nVox,nVox,nSlices);
% start the cycle to fit the voxel-wise diffusion tensor
for k=1:nSlices
    % print fitting progress
    disp([' Fitting Slice ',num2str(k)])
    for i=1:1:nVox
        for j=1:1:nVox
            % check if current voxel belongs to the mask
            if (brain_mask(i,j,k))
                
                % extract the signal from each voxel
                VoxSignal=squeeze(Slog(i,j,k,:));
                % fit the DTI using LLS approach
                D=-(B'*B)\B'*VoxSignal;
                % reconstruct the diffusion tensor from the fitted parameters
                T=[D(1) D(4)/2 D(5)/2;
                   D(4)/2 D(2) D(6)/2;
                   D(5)/2 D(6)/2 D(3)];
               
                % compute eigenvalues and eigenvectors
                eigenvals=eig(T);
                % manage negative eigenvals as shown in laboratory:
                if eigenvals(1)<0 && eigenvals(2)<0 && eigenvals(3)<0
                    eigenvals=abs(eigenvals); % if all <0 -> take the absolute value
                end
                % otherwise -> put negatives to zero
                if eigenvals(1)<0, eigenvals(1)=0; end
                if eigenvals(2)<0, eigenvals(2)=0; end
                if eigenvals(3)<0, eigenvals(3)=0; end
                
                % compute Fractional Anisotropy index
                FA(i,j,k)=(1/sqrt(2))*sqrt((eigenvals(1)-eigenvals(2)).^2+(eigenvals(2)-eigenvals(3)).^2 + (eigenvals(1)-eigenvals(3)).^2) ...
                    /sqrt(eigenvals(1).^2+eigenvals(2).^2+eigenvals(3).^2);
                % compute Mean Diffusivity index
                MD(i,j,k)=mean(eigenvals);
            end
        end
    end
end

clear B i j k VoxSignal D T eigenvals eigenvects idx_sort principal_eigenvector

%% c.
% Provide the visualization of the FA and MD maps for a central slice.
nSlice_FA=60; % slice selected for Fractional Anisotropy map visualization
figure()
imagesc(squeeze(FA(:,nSlice_FA,:)))
colorbar
title(['FA index for slice ',num2str(nSlice_FA)])
nSlice_MD=60; % slice selected for Mean Diffusivity map visualization
figure()
imagesc(squeeze(MD(:,nSlice_MD,:)))
% imagesc(squeeze(MD(:,:,nSlice_MD)))
% in this case without doing brain_mask(:,27,:)=false through this visualization
% we can an ascquisitition error
colorbar
title(['MD index for slice ',num2str(nSlice_MD)])

clear nSlice_FA nSlice_MD

%% d.
% Mask the FA and MD maps (as done for the Hammers atlas in fMRI: point 1.d),
% extract their mean values in each ROI (fMRI: point 1.e.).

atlas_mask=load_untouch_nii('Atlas/Hammers_2_T1_2mm_int.nii.gz');
atlas_mask=double(atlas_mask.img);
gm = load_untouch_nii('Results/c1T1_2mm.nii');
gm   = double(gm.img);
threshold = 255*0.45;
gm = gm > threshold;
mask_diff=atlas_mask.*brain_mask.*gm;
%implay(mask_diff/max(mask_diff(:)))

idx_exl=[];
for i=1:max(atlas_mask(:))
    if sum(mask_diff(:)==i)<10
        idx_exl=[idx_exl i];
    end
end
% amyg (3 4); cer (17 18); bstem (19); cc (44); snigra (74 75); ventr (45 46 47 48 49)
idx_exl=[idx_exl 3 4 17 18 19 44 45 46 47 48 49 74 75];
idx_ROI_diff=setdiff(1:max(atlas_mask(:)),idx_exl);
nROI_diff=length(idx_ROI_diff);

% initialize the structures which will be used to contain ROI-mean DTI indexes
mean_ROI_FA=zeros(nROI_diff,1);
mean_ROI_MD=zeros(nROI_diff,1);
for i=1:nROI_diff
    name_ROI=idx_ROI_diff(i);
    disp(['Working on ROI #',num2str(name_ROI)])
    ROI_mask=mask_diff==name_ROI;
    FA_ROI=FA(ROI_mask);
    mean_ROI_FA(i)=mean(FA_ROI);
    MD_ROI=MD(ROI_mask);
    mean_ROI_MD(i)=mean(MD_ROI);
end
% visualization of the results
figure()
subplot(211), stem(idx_ROI_diff,mean_ROI_FA, 'LineWidth',1.2), xlabel('ROIs'), title('mean FA')
subplot(212), stem(idx_ROI_diff,mean_ROI_MD, 'LineWidth',1.2), xlabel('ROIs'), title('mean MD')

clear atlas_mask gm threshold brain_mask mask_diff idx_exl i name_ROI ROI_mask FA_ROI MD_ROI


%% DMRI/fMRI integration

idx=setdiff(idx_ROI_diff,idx_ROI); % name ROI

idx_delete = nan(1,length(idx));
for i = 1:length(idx)
    idx_delete(i) = find(idx_ROI_diff == idx(i));
end

mean_ROI_FA(idx_delete)=[];
mean_ROI_MD(idx_delete)=[];

clear idx idx_delete i

%% 1. Visual inspection

% Visualize the scatterplot of these variables:

% ROIs node degree versus ROIs FA
figure
subplot(2,3,1)
scatter(node_degree,mean_ROI_FA,'LineWidth',2)
xlabel("ROIs' node degree")
ylabel("ROIs' mean FA")
hold on
mod=fitlm(node_degree, mean_ROI_FA);
y = mod.Coefficients{1,1} + mod.Coefficients{2,1}*node_degree;
plot(node_degree, y, 'r', 'LineWidth', 2)
text(10, 0.3, ['R^2 = ',num2str(mod.Rsquared.Ordinary)])
text(10, 0.28, ['p-value = ', num2str(mod.Coefficients{2,4})])
grid on

% ROIs node strength versus ROIs FA
subplot(2,3,2)
scatter(node_strength,mean_ROI_FA, 'LineWidth', 2)
xlabel("ROIs' node strength")
ylabel("ROIs' mean FA")
hold on
mod=fitlm(node_strength, mean_ROI_FA);
y = mod.Coefficients{1,1} + mod.Coefficients{2,1}*node_strength;
plot(node_strength, y, 'r', 'LineWidth', 2)
text(4, 0.3, ['R^2 = ',num2str(mod.Rsquared.Ordinary)])
text(4, 0.28, ['p-value = ', num2str(mod.Coefficients{2,4})])
grid on

% ROIs node normalized betweenness centrality versus ROIs FA
subplot(2,3,3)
scatter(BC,mean_ROI_FA, 'LineWidth',2)
xlabel("ROIs' norm. betweenness centrality")
ylabel("ROIs' mean FA")
hold on
mod=fitlm(BC, mean_ROI_FA);
y = mod.Coefficients{1,1} + mod.Coefficients{2,1}*BC;
plot(BC, y, 'r', 'LineWidth', 2)
text(0.4, 0.3, ['R^2 = ',num2str(mod.Rsquared.Ordinary)])
text(0.4, 0.28, ['p-value = ', num2str(mod.Coefficients{2,4})])
grid on

% ROIs node degree versus ROIs MD
subplot(2,3,4)
scatter(node_degree,mean_ROI_MD, 'LineWidth',2)
xlabel("ROIs' node degree")
ylabel("ROIs' mean MD")
hold on
mod=fitlm(node_degree, mean_ROI_MD);
y = mod.Coefficients{1,1} + mod.Coefficients{2,1}*node_degree;
plot(node_degree, y, 'r', 'LineWidth', 2)
text(5, 0.0014, ['R^2 = ',num2str(mod.Rsquared.Ordinary)])
text(5, 0.00134, ['p-value = ', num2str(mod.Coefficients{2,4})])
grid on

% ROIs node strength versus ROIs MD
subplot(2,3,5)
scatter(node_strength,mean_ROI_MD, 'LineWidth', 2)
xlabel("ROIs' node strength")
ylabel("ROIs' mean MD")
hold on
mod=fitlm(node_strength, mean_ROI_MD);
y = mod.Coefficients{1,1} + mod.Coefficients{2,1}*node_strength;
plot(node_strength, y, 'r', 'LineWidth', 2)
text(2, 0.0014, ['R^2 = ',num2str(mod.Rsquared.Ordinary)])
text(2, 0.00134, ['p-value = ', num2str(mod.Coefficients{2,4})])
grid on

% ROIs node normalized betweenness centrality versus ROIs MD
subplot(2,3,6)
scatter(BC,mean_ROI_MD, 'LineWidth',2)
xlabel("ROIs' norm. betweenness centrality")
ylabel("ROIs' mean MD")
hold on
mod=fitlm(BC, mean_ROI_MD);
y = mod.Coefficients{1,1} + mod.Coefficients{2,1}*BC;
plot(BC, y, 'r', 'LineWidth', 2)
text(0.4, 0.0014, ['R^2 = ',num2str(mod.Rsquared.Ordinary)])
text(0.4, 0.00134, ['p-value = ', num2str(mod.Coefficients{2,4})])
grid on

sgtitle('fMRI metrics vs. dMRI metrics')

clear mod y

%% 2. Quantitative results

% Compute and provide the Pearson’s correlation between the six pairs of variables of point

[RHO,p1]= corrcoef(node_degree,mean_ROI_FA);
RHO_deg_FA=RHO(1,2);
p_deg_FA=p1(1,2);

[RHO,p1]= corrcoef(node_strength,mean_ROI_FA);
RHO_st_FA=RHO(1,2);
p_st_FA=p1(1,2);

[RHO,p1]= corrcoef(BC,mean_ROI_FA);
RHO_BC_FA=RHO(1,2);
p_BC_FA=p1(1,2);

[RHO,p1]= corrcoef(node_degree,mean_ROI_MD);
RHO_deg_MD=RHO(1,2);
p_deg_MD=p1(1,2);

[RHO,p1]= corrcoef(node_strength,mean_ROI_MD);
RHO_st_MD=RHO(1,2);
p_st_MD=p1(1,2);

[RHO,p1]= corrcoef(BC,mean_ROI_MD);
RHO_BC_MD=RHO(1,2);
p_BC_MD=p1(1,2);

vett_pearson_corr=[RHO_deg_FA RHO_st_FA RHO_BC_FA RHO_deg_MD RHO_st_MD RHO_BC_MD]'; 
vett_pearson_pvalues=[p_deg_FA p_st_FA p_BC_FA p_deg_MD p_st_MD p_BC_MD]';   
%vett_pearson_pvalues < 0.05

figure
bar(vett_pearson_corr);
ylim([-1 1])
title('correlations between fMRI and dMRI metrics')

% Discuss the results: is there a statistically significant relationship between any pair of
% variables?
% Pearson coefficient is a measure of linear correlation between the variables, 
% as we can see only the first two scatter presented statistical significant 
% relationship with absolute value > 0.5 but this is a negative correlation

clear p1

