%% demo.m  --  AITV-NB image restoration under negative binomial (NB) noise
%
% Demonstrates the proposed AITV-NB ADMM method on the cameraman image:
% the image is blurred and corrupted by overdispersed (negative binomial)
% photon-limited noise, then restored by AITV-NB.
%
% Reference:
%   Y. Lu, K. Bui, R. F. Marcia, "Negative Binomial Optimization for
%   Overdispersed Photon-Limited Imaging," Inverse Problems and Imaging.
%
% Requirements: MATLAB, Image Processing Toolbox, Statistics & ML Toolbox.

close all; clear; clc;
rng('default'); rng(42);                       % reproducibility

%% 1) Load image and scale to a low-photon peak
peak = 80;                                      % maximum photon count
I = double(im2gray(imread('cameraman.tif')));   % built-in 256x256 image
Q = max(I(:)) / peak;                           % map to original intensity range
I = I / Q;                                       % ground truth on [0, peak]
I(I == 0) = min(I(I > 0));                        % avoid exact zeros (NB log-likelihood)

%% 2) Degrade: blur, then add negative binomial noise
A = fspecial('gaussian', [10 10], 2);            % blur kernel (use A = 1 for pure denoising)
r = 10;                                          % NB dispersion (smaller r = more overdispersion)

I_blurry = myconv(I, A);
p  = r ./ (I_blurry + r);                         % NB success probability per pixel
u0 = double(nbinrnd(r, p));                       % noisy photon-count observation
u0 = u0 / max(u0(:));                             % normalize the input

noisy_psnr = psnr(u0*Q, I*Q, 255);
noisy_ssim = ssim(uint8(u0*Q), uint8(I*Q));

%% 3) Reconstruct with AITV-NB
alpha = 0.5;     % weight of the -alpha||.||_{2,1} term (edge preservation), in [0,1]
tau   = 0.3;     % regularization strength
beta0 = 0.25;    % initial ADMM penalty parameter (beta^0)

f = Deblur_NB_L1mL2(u0, A, alpha, tau, beta0, r);
f = mat2gray(f) * peak;                           % rescale for display/metrics

rec_psnr = psnr(f*Q, I*Q, 255);
rec_ssim = ssim(uint8(f*Q), uint8(I*Q));

fprintf('Noisy   : PSNR = %.2f dB, SSIM = %.3f\n', noisy_psnr, noisy_ssim);
fprintf('AITV-NB : PSNR = %.2f dB, SSIM = %.3f\n', rec_psnr,  rec_ssim);

%% 4) Display
figure('Position', [100 100 1150 400]);
subplot(1,3,1); imagesc(I);  axis image off; colormap gray; title('Ground truth');
subplot(1,3,2); imagesc(u0); axis image off; colormap gray;
    title(sprintf('Noisy (NB, r=%d)\nPSNR %.2f / SSIM %.2f', r, noisy_psnr, noisy_ssim));
subplot(1,3,3); imagesc(f);  axis image off; colormap gray;
    title(sprintf('AITV-NB\nPSNR %.2f / SSIM %.2f', rec_psnr, rec_ssim));
