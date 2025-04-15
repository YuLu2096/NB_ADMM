close all; clear all; clc
%% Set parameter and peak
peak = 80;

% read image
I = imread('cameraman256.png');
I = double(I);
I = im2gray(I);
rng('default');
rng(42);


Q = max(max(I)) /peak;
I = I / Q;
I(I == 0) = min(min(I(I > 0)));

% blury iamges
A=fspecial('gaussian', [10 10], 2);
%A=fspecial('disk', 3);
%A=fspecial('motion',10, 45);

%A = eye(size(I))
I_blurry = myconv(I, A);

% add noise 
r = 1;% r is the NB parameter
p = r./(I+r); % p is the NB probobility, mu = r(1-p)/p. Here we assume A is identity matrix
u0 = nbinrnd(r,p);% noise imagerndpr
%u0 = imnoise(uint8(I_blurry),'poisson');


u0 = double(u0);
u0 = u0/max(u0(:));

% compute psnr/ssim
noisy_psnr = psnr(u0*Q, I*Q, 255);
noisy_ssim = ssim(uint8(u0*Q), uint8(I*Q));

%% denoise by NB AITV and compute metrics
beta = 0.1;
alpha = 0.001;
tau = 0.1;
uAITV_NB = Deblur_NB_L1mL2(u0, A, 0, alpha, beta, tau, r);
uAITV_NB(uAITV_NB<0) = 0;
uAITV_NB =  mat2gray(uAITV_NB)*peak;
aitv_NB_psnr = psnr(uAITV_NB*Q, I*Q, 255);
aitv_NB_ssim = ssim(uint8(uAITV_NB*Q), uint8(I*Q));


%% plot figure
figure;
subplot(1,3,1); imagesc(I); axis off; axis image; colormap gray; title('Original');
subplot(1,3,2); imagesc(u0); axis off; axis image; colormap gray; title(sprintf('Noisy\n PSNR:%.2f/SSIM:%.2f',noisy_psnr, noisy_ssim))
subplot(1,3,3); imagesc(uAITV_NB); axis off; axis image; colormap gray; title(sprintf('AITV-NB\n PSNR:%.2f/SSIM:%.2f',aitv_NB_psnr, aitv_NB_ssim))
